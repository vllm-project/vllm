# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba 注意力后端基础模块。

本模块实现了 Mamba 类注意力后端的基础类，负责：
- 定义 Mamba 注意力元数据的基类
- 提供元数据构建器的默认实现
- 支持分块预填充和_prefix caching_

主要类：
- BaseMambaAttentionMetadata: Mamba 注意力元数据基类
- BaseMambaAttentionMetadataBuilder: 元数据构建器基类
"""

import abc
from dataclasses import dataclass, replace
from typing import Any, ClassVar, TypeVar

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

M = TypeVar("M", bound="BaseMambaAttentionMetadata")


@dataclass
class BaseMambaAttentionMetadata:
    """Mamba 注意力元数据基类。

    存储 Mamba 类注意力模型前向传播所需的元数据信息。
    支持预填充、解码和预测解码。

    Attributes:
        num_prefills: 预填充请求数
        num_prefill_tokens: 预填充 token 数
        num_decodes: 解码请求数
        num_decode_tokens: 解码 token 数
        num_reqs: 请求总数
        has_initial_states_p: 预填充请求是否有初始状态
        query_start_loc_p: 预填充 query 起始位置
        num_computed_tokens_p: 预填充已计算的 token 数
        state_indices_tensor_p: 预填充状态索引张量
        state_indices_tensor_d: 解码状态索引张量
        query_start_loc_d: 解码 query 起始位置
        num_accepted_tokens: 每个预测序列接受的 token 数
        block_idx_last_scheduled_token: 最后调度的 token 的块索引
        block_idx_first_scheduled_token_p: 预填充第一个调度的 token 的块索引
        block_idx_last_computed_token: 最后计算的 token 的块索引
        seq_lens: 序列长度
        cu_chunk_seqlen_p: 分块预填充的累积序列长度
        last_chunk_indices_p: 每个序列的最后一个块索引
        nums_dict: causal_conv1d 的 nums 字典
        batch_ptr: 批次指针
        token_chunk_offset_ptr: token 块偏移指针
    """
    num_prefills: int
    """预填充请求数。"""

    num_prefill_tokens: int
    """预填充 token 数。"""

    num_decodes: int
    """解码请求数。"""

    num_decode_tokens: int
    """解码 token 数。"""

    num_reqs: int
    """请求总数。"""

    # 以下张量仅用于预填充请求，如果批次中没有预填充请求则为 None
    has_initial_states_p: torch.Tensor | None
    """预填充请求是否有初始状态。"""

    query_start_loc_p: torch.Tensor | None
    """预填充 query 起始位置。"""

    num_computed_tokens_p: torch.Tensor | None
    """预填充已计算的 token 数。"""

    state_indices_tensor_p: torch.Tensor | None
    """预填充状态索引张量。"""

    # 以下张量用于解码请求和预测解码兼容性，
    # 如果批次中没有解码请求则为 None
    state_indices_tensor_d: torch.Tensor | None
    """解码状态索引张量。"""

    query_start_loc_d: torch.Tensor | None
    """解码 query 起始位置，形状：[num_decodes + 1,]"""

    # 每个预测序列接受的 token 数（用于加载正确的检查点）
    # 包括奖励 token（所以最小值为 1）
    num_accepted_tokens: torch.Tensor | None
    """每个预测序列接受的 token 数，形状：[batch,]"""

    # 以下张量仅用于 all 模式的前缀缓存，如果禁用则为 None
    block_idx_last_scheduled_token: torch.Tensor | None
    """最后调度的 token 的块索引。"""

    block_idx_first_scheduled_token_p: torch.Tensor | None
    """预填充第一个调度的 token 的块索引。"""

    block_idx_last_computed_token: torch.Tensor | None
    """最后计算的 token 的块索引。"""

    # 以下张量仅用于 align 模式的前缀缓存
    seq_lens: torch.Tensor
    """序列长度张量。"""

    # cu_chunk_seqlen_p 是形状为 (nchunks+1,) 的张量，包含每个块在其变长序列维度中的偏移量。
    # 第 i 个块包含从 cu_chunk_seqlen_p[i] 到 cu_chunk_seqlen_p[i+1] 的 token。
    cu_chunk_seqlen_p: torch.Tensor | None = None
    """分块预填充的累积序列长度，形状：(nchunks+1,)"""

    # last_chunk_indices_p 是形状为 (batch,) 的张量，包含（预填充）批次中每个序列的最后一个块的索引。
    last_chunk_indices_p: torch.Tensor | None = None
    """每个序列的最后一个块索引，形状：(batch,)"""

    # 以下属性用于 causal_conv1d 的 triton 实现
    nums_dict: dict | None = None
    """causal_conv1d 的 nums 字典。"""

    batch_ptr: torch.Tensor | None = None
    """批次指针。"""

    token_chunk_offset_ptr: torch.Tensor | None = None
    """token 块偏移指针。"""


class BaseMambaAttentionMetadataBuilder(AttentionMetadataBuilder[M], abc.ABC):
    """Mamba 注意力元数据构建器基类。

    负责构建 Mamba 类注意力模型运行所需的元数据对象。
    提供默认实现，子类可以覆盖以添加额外的元数据。

    Class Attributes:
        metadata_cls: 元数据类
        reorder_batch_threshold: 重排序批次阈值
        _cudagraph_support: CUDA 图支持级别
        supports_update_block_table: 是否支持更新块表
    """
    metadata_cls: type[M]
    reorder_batch_threshold: int = 1
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    # 如果使用预测解码则禁用
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """初始化 Mamba 元数据构建器。

        Args:
            kv_cache_spec: KV 缓存规格
            layer_names: 层名称列表
            vllm_config: vLLM 配置
            device: 设备类型
        """
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # 启用预测解码支持
        self.speculative_config = vllm_config.speculative_config
        self.compilation_config = vllm_config.compilation_config
        self.num_spec_tokens: int = vllm_config.num_speculative_tokens
        self.use_spec_decode = self.num_spec_tokens > 0

        assert isinstance(kv_cache_spec, MambaSpec)
        scheduler_config = vllm_config.scheduler_config
        self.decode_cudagraph_max_bs: int = scheduler_config.max_num_seqs
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        if self.vllm_config.cache_config.mamba_cache_mode == "all":
            # 前缀缓存模式：分配最大块数
            max_num_blocks = cdiv(
                self.vllm_config.model_config.max_model_len,
                self.kv_cache_spec.block_size,
            )
            # 预测解码不支持前缀缓存，所以保持与预填充缓冲区一致的形状
            # TODO: 根据需要减小此大小以进行仅解码的 cudagraph 捕获
            self.state_indices_tensor_d: torch.Tensor = torch.empty(
                (
                    self.decode_cudagraph_max_bs,
                    max_num_blocks,
                ),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_scheduled_token: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_computed_token: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
        else:
            # 非前缀缓存模式：仅分配所需块数
            self.state_indices_tensor_d = torch.empty(
                (self.decode_cudagraph_max_bs, 1 + self.num_spec_tokens),
                dtype=torch.int32,
                device=device,
            )

        # 对于预测解码，我们需要存储以下缓冲区以用于解码期间的 CUDA 图捕获
        if self.num_spec_tokens > 0:
            self.decode_num_accepted_tokens: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )

        self._init_reorder_batch_threshold(1, self.use_spec_decode)
        if self.use_spec_decode:
            self.supports_update_block_table = False

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """为完整 CUDA 图捕获构建元数据。

        目前，Mamba 仅支持解码的完整 CUDA 图。

        Args:
            common_attn_metadata: 通用注意力元数据

        Returns:
            构建的元数据对象
        """
        m = common_attn_metadata

        assert (
            m.max_query_len <= 1 + self.num_spec_tokens
            and m.num_reqs <= self.decode_cudagraph_max_bs
        ), (
            "Mamba only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )

        assert m.max_query_len == 1 + self.num_spec_tokens  # 仅解码

        num_accepted_tokens = None
        if self.num_spec_tokens > 0:
            num_accepted_tokens = torch.diff(m.query_start_loc)

        return self.build(0, m, num_accepted_tokens=num_accepted_tokens)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> M:
        """构建 Mamba 注意力元数据。

        这是 Mamba 类注意力后端的默认构建实现。
        子类（例如 Mamba2）可以覆盖以添加额外的元数据。

        Args:
            common_prefix_len: 公共前缀长度
            common_attn_metadata: 通用注意力元数据
            fast_build: 是否快速构建
            num_accepted_tokens: 接受的 token 数
            **kwargs: 其他参数

        Returns:
            构建的元数据对象
        """
        return self._compute_common_metadata(
            common_attn_metadata, num_accepted_tokens=num_accepted_tokens
        )

    def _compute_chunk_metadata(
        self,
        chunk_size: int,
        num_prefills: int,
        num_computed_tokens_p_cpu: torch.Tensor,
        query_start_loc_p_cpu: torch.Tensor,
    ) -> tuple[list[int], list[int], list[int]]:
        """计算 Mamba 模型的分块元数据。

        下面的代码仔细构建块，使得：
        1. 块仅包含来自*单个*序列的 token。
        2. 对于每个序列，我们保证可以每 chunk_size 个 token 检索一次 mamba 状态。
        约束（1）大大简化了 mamba kernel。
        约束（2）大大简化了 mamba 前缀缓存的实现（进行中）。
        我们需要处理与分块预填充的交互以满足约束（2）。

        Args:
            chunk_size: 块大小
            num_prefills: 预填充请求数
            num_computed_tokens_p_cpu: 预填充已计算的 token 数（CPU）
            query_start_loc_p_cpu: 预填充 query 起始位置（CPU）

        Returns:
            (cu_chunk_seqlen, seq_idx, last_chunk_indices) 元组
        """
        # TODO (tdoublep): 此代码可能可以优化。
        cu_chunk_seqlen = []
        seq_idx = []
        last_chunk_indices = []
        seqlen_pos = 0

        for req_idx in range(num_prefills):
            this_num_computed = num_computed_tokens_p_cpu[req_idx].item()
            this_new_tokens = (
                query_start_loc_p_cpu[req_idx + 1].item()
                - query_start_loc_p_cpu[req_idx].item()
            )

            # 如果已计算的 token 未与块对齐，使用第一个块来完成
            if this_num_computed % chunk_size != 0:
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(seqlen_pos)
                # 完成这个块需要多少 token？
                chunk_len = (
                    cdiv(this_num_computed, chunk_size) * chunk_size - this_num_computed
                )
                # 我们最多只能使用 this_new_tokens
                chunk_len = min(chunk_len, this_new_tokens)
                seqlen_pos += chunk_len
                this_new_tokens -= chunk_len

            n_chunks = cdiv(this_new_tokens, chunk_size)
            for chunk in range(n_chunks):
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(seqlen_pos)
                chunk_len = min(chunk_size, this_new_tokens)
                seqlen_pos += chunk_len
                this_new_tokens -= chunk_len

            assert this_new_tokens == 0
            last_chunk_indices.append(len(cu_chunk_seqlen) - 1)

        cu_chunk_seqlen.append(seqlen_pos)

        return cu_chunk_seqlen, seq_idx, last_chunk_indices

    def _build_chunk_metadata_tensors(
        self,
        chunk_size: int,
        common: M,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算分块元数据并返回为设备张量。

        Args:
            chunk_size: 块大小
            common: 通用元数据
            common_attn_metadata: 通用注意力元数据

        Returns:
            (cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p) 元组
        """
        num_reqs = common.num_reqs
        num_prefills = common.num_prefills
        num_decode_tokens = common.num_decode_tokens

        num_computed_tokens_cpu = (
            common_attn_metadata.compute_num_computed_tokens().cpu()
        )
        num_computed_tokens_p_cpu = num_computed_tokens_cpu[
            num_reqs - num_prefills : num_reqs
        ]
        query_start_loc_p_cpu = (
            common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
            - num_decode_tokens
        )

        cu_chunk_seqlen, seq_idx, last_chunk_indices = self._compute_chunk_metadata(
            chunk_size,
            num_prefills,
            num_computed_tokens_p_cpu,
            query_start_loc_p_cpu,
        )

        device = common_attn_metadata.query_start_loc.device
        cu_chunk_seqlen_p = torch.as_tensor(
            cu_chunk_seqlen,
            device=device,
            dtype=torch.int32,
        )
        seq_idx_p = torch.as_tensor(
            seq_idx,
            device=device,
            dtype=torch.int32,
        )
        last_chunk_indices_p = torch.as_tensor(
            last_chunk_indices,
            device=device,
            dtype=torch.int32,
        )
        return cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p

    def _compute_prefix_caching_block_indices(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        mamba_block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算前缀缓存的块索引。

        用于前缀缓存模式下，计算每个序列的块索引以便正确管理 Mamba 状态缓存。

        Args:
            common_attn_metadata: 通用注意力元数据
            mamba_block_size: Mamba 块大小

        Returns:
            (block_idx_last_computed_token, block_idx_first_scheduled_token,
             block_idx_last_scheduled_token) 元组
        """
        num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
        # 最后计算的 token 的块索引
        block_idx_last_computed_token = cdiv(num_computed_tokens, mamba_block_size) - 1
        # 第一个调度的 token 的块索引（小于等于）
        block_idx_first_scheduled_token = (
            cdiv(num_computed_tokens + 1, mamba_block_size) - 1
        )
        # 最后调度的 token 的块索引（小于等于）
        block_idx_last_scheduled_token = (
            cdiv(common_attn_metadata.seq_lens, mamba_block_size) - 1
        )
        # 如果是非计算的，设置为 0 以避免后续索引问题
        block_idx_last_computed_token = torch.clamp(
            block_idx_last_computed_token, min=0
        )
        # 如果是填充请求（0 序列长度），设置为 0
        block_idx_last_scheduled_token = torch.clamp(
            block_idx_last_scheduled_token, min=0
        )

        return (
            block_idx_last_computed_token,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
        )

    def _compute_common_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> M:
        """计算 Mamba1 和 Mamba2 共用的元数据。

        该方法负责：
        - 分割预填充和解码请求
        - 计算状态索引张量
        - 处理前缀缓存相关的元数据
        - 构建 causal_conv1d 元数据

        Args:
            common_attn_metadata: 通用注意力元数据
            num_accepted_tokens: 接受的 token 数（用于预测解码）

        Returns:
            构建的 Mamba 元数据对象
        """
        num_reqs = common_attn_metadata.num_reqs

        # 当启用预测解码时，将多 token 查询视为解码请求
        # 否则，使用默认的解码阈值以防止误分类预填充查询为解码请求
        decode_threshold = (
            self.reorder_batch_threshold if num_accepted_tokens is not None else 1
        )

        # 分割解码和预填充请求
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=decode_threshold,
                treat_short_extends_as_decodes=False,
            )
        )

        # 需要标志来指示是否有初始状态
        has_initial_states_p = None
        query_start_loc_p = None
        query_start_loc_d = None
        num_computed_tokens = None
        num_computed_tokens_p = None

        # 用于前缀缓存
        block_idx_first_scheduled_token = None
        block_idx_first_scheduled_token_p = None
        block_idx_last_computed_token = None
        block_idx_last_scheduled_token = None

        # 用于 causal_conv1d
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        if self.vllm_config.cache_config.mamba_cache_mode == "all":
            num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()

            # 返回形状为 (#requests, #max blocks) 的张量
            state_indices_tensor = common_attn_metadata.block_table_tensor
            # 额外的缓存相关变量：
            mamba_block_size = self.kv_cache_spec.block_size
            (
                block_idx_last_computed_token,
                block_idx_first_scheduled_token,
                block_idx_last_scheduled_token,
            ) = self._compute_prefix_caching_block_indices(
                common_attn_metadata, mamba_block_size
            )
        else:
            # 非前缀缓存模式：获取状态索引张量
            state_indices_tensor = mamba_get_block_table_tensor(
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.seq_lens,
                self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )

        # 确保状态索引张量至少是 2D 的
        if state_indices_tensor.dim() == 1:
            state_indices_tensor = state_indices_tensor.unsqueeze(-1)

        # 将状态索引张量分割为解码和预填充部分
        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor,
            [num_decodes, num_prefills],
            dim=0,
        )
        if self.vllm_config.cache_config.mamba_cache_mode != "all":
            # 限制解码部分的 token 数
            state_indices_tensor_d = state_indices_tensor_d[
                :, : 1 + self.num_spec_tokens
            ]
            # 预填充部分仅使用第一个块
            state_indices_tensor_p = state_indices_tensor_p[:, 0]

        # 有时即使启用了 specdec，也会得到应该视为解码的单 token 预填充块
        # 但没有设置 num_accepted_tokens。这些应该可以正常处理为非 spec 解码，
        # 因为只有一个 token，所以不会存在将接受的 token 放在错误槽位的风险
        if num_decodes > 0 and self.use_spec_decode and num_accepted_tokens is not None:
            query_start_loc_d = common_attn_metadata.query_start_loc[: num_decodes + 1]
            num_accepted_tokens = num_accepted_tokens[:num_decodes]

        if num_prefills > 0:
            if num_computed_tokens is None:
                num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()

            # 获取预填充的 query 起始位置（CPU 和 GPU）
            query_start_loc_p_cpu = (
                common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
                - num_decode_tokens
            )
            query_start_loc_p = (
                common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                - num_decode_tokens
            )
            # 检查哪些预填充请求有已计算的 token（即需要加载初始状态）
            has_initial_states_p = (
                num_computed_tokens[num_reqs - num_prefills : num_reqs] > 0
            )

            # 计算 causal_conv1d 的元数据
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    query_start_loc_p_cpu,
                    device=common_attn_metadata.query_start_loc.device,
                )
            )

            if self.vllm_config.cache_config.mamba_cache_mode == "all":
                assert num_computed_tokens is not None
                # 仅获取预填充部分的 num_computed_tokens
                num_computed_tokens_p = num_computed_tokens[
                    num_reqs - num_prefills : num_reqs
                ]
                assert block_idx_first_scheduled_token is not None
                # 仅获取预填充部分的 block_idx_first_scheduled_token
                block_idx_first_scheduled_token_p = block_idx_first_scheduled_token[
                    num_reqs - num_prefills : num_reqs
                ]

        # 构建元数据对象
        metadata = self.metadata_cls(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_p=query_start_loc_p,
            has_initial_states_p=has_initial_states_p,
            state_indices_tensor_p=state_indices_tensor_p,
            state_indices_tensor_d=state_indices_tensor_d,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc_d=query_start_loc_d,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token_p=block_idx_first_scheduled_token_p,
            block_idx_last_computed_token=block_idx_last_computed_token,
            num_computed_tokens_p=num_computed_tokens_p,
            num_reqs=num_reqs,
            seq_lens=common_attn_metadata.seq_lens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )

        return self._update_metadata_for_cudagraph_capture(metadata)

    def _update_metadata_for_cudagraph_capture(
        self,
        metadata: M,
    ) -> M:
        """为 CUDA 图捕获更新元数据。

        目前，Mamba 仅支持解码的完整 CUDA 图。

        该方法负责：
        - 为解码请求填充元数据张量以支持 CUDA 图捕获
        - 处理预测解码相关的元数据
        - 处理前缀缓存相关的元数据

        Args:
            metadata: 原始元数据对象

        Returns:
            更新后的元数据对象
        """
        state_indices_tensor_d = metadata.state_indices_tensor_d
        query_start_loc_d = metadata.query_start_loc_d
        num_accepted_tokens = metadata.num_accepted_tokens
        block_idx_last_scheduled_token = metadata.block_idx_last_scheduled_token
        block_idx_last_computed_token = metadata.block_idx_last_computed_token
        if (
            metadata.num_prefills == 0
            and metadata.num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            # 仅解码模式：为 CUDA 图捕获填充张量
            padded_bs = metadata.num_reqs
            # 复制状态索引张量到预分配缓冲区
            self.state_indices_tensor_d[: metadata.num_decodes].copy_(
                state_indices_tensor_d, non_blocking=True
            )
            state_indices_tensor_d = self.state_indices_tensor_d[:padded_bs]
            # 填充剩余部分
            state_indices_tensor_d[metadata.num_decodes :] = PAD_SLOT_ID

            # 处理预测解码的 num_accepted_tokens
            if self.use_spec_decode and num_accepted_tokens is not None:
                assert query_start_loc_d is not None
                query_start_loc_d = query_start_loc_d[: padded_bs + 1]
                # 复制到预分配缓冲区
                self.decode_num_accepted_tokens[: metadata.num_decodes].copy_(
                    num_accepted_tokens, non_blocking=True
                )
                num_accepted_tokens = self.decode_num_accepted_tokens[:padded_bs]
                # 填充为 1（第一个槽位索引）
                num_accepted_tokens[metadata.num_decodes :] = (
                    1  # pad with 1st slot index
                )

            # 处理前缀缓存模式下的块索引
            if self.vllm_config.cache_config.mamba_cache_mode == "all":
                assert block_idx_last_scheduled_token is not None
                assert block_idx_last_computed_token is not None
                # 复制到预分配缓冲区
                self.block_idx_last_scheduled_token[: metadata.num_decodes].copy_(
                    block_idx_last_scheduled_token[: metadata.num_decodes],
                    non_blocking=True,
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    : metadata.num_decode_tokens
                ]

                self.block_idx_last_computed_token[: metadata.num_decodes].copy_(
                    block_idx_last_computed_token[: metadata.num_decodes],
                    non_blocking=True,
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    : metadata.num_decode_tokens
                ]

        # 返回更新后的元数据
        return replace(
            metadata,
            state_indices_tensor_d=state_indices_tensor_d,
            query_start_loc_d=query_start_loc_d,
            num_accepted_tokens=num_accepted_tokens,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_last_computed_token=block_idx_last_computed_token,
        )

    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        """更新块表并返回新的元数据。

        当块表发生变化时（例如在调度新 token 后），需要更新状态索引张量。

        Args:
            metadata: 当前元数据对象
            blk_table: 新的块表
            slot_mapping: 槽位映射

        Returns:
            更新后的元数据对象
        """
        # 根据新的块表重新计算状态索引张量
        state_indices_tensor = mamba_get_block_table_tensor(
            blk_table,
            metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        # 确保状态索引张量至少是 2D 的
        if state_indices_tensor.dim() == 1:
            state_indices_tensor = state_indices_tensor.unsqueeze(-1)

        # 验证请求数是否匹配
        assert (
            metadata.num_prefills + metadata.num_decodes
            == state_indices_tensor.shape[0]
        ), (
            "更新块表时请求数不匹配。"
            f" 期望 {metadata.num_prefills + metadata.num_decodes}, "
            f"得到 {state_indices_tensor.shape[0]}."
        )

        # 分割为解码和预填充部分
        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor,
            [metadata.num_decodes, metadata.num_prefills],
            dim=0,
        )
        if self.vllm_config.cache_config.mamba_cache_mode != "all":
            # 限制解码部分的 token 数
            state_indices_tensor_d = state_indices_tensor_d[
                :, : 1 + self.num_spec_tokens
            ]
            # 预填充部分仅使用第一个块
            state_indices_tensor_p = state_indices_tensor_p[:, 0]

        # 创建新的元数据对象
        new_metadata = replace(
            metadata,
            state_indices_tensor_d=state_indices_tensor_d,
            state_indices_tensor_p=state_indices_tensor_p,
        )

        # 为 CUDA 图捕获更新元数据
        return self._update_metadata_for_cudagraph_capture(new_metadata)
