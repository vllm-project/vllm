# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EAGLE 推测解码 proposer 模块。

本模块实现了基于 EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）
架构的推测解码 proposer，负责：
- 使用 EAGLE 模型生成候选 token
- 支持树状注意力机制进行多 token 推测
- 支持 EAGLE3 架构和辅助隐藏状态
- 管理草稿模型的注意力层和 KV 缓存
- 支持并行推测（parallel drafting）

主要类：
- SpecDecodeBaseProposer: 推测解码基础 proposer 类
- EagleProposer: EAGLE 推测解码 proposer

EAGLE 架构特点：
1. 使用目标模型的隐藏状态作为输入
2. 通过轻量级解码头生成候选 token
3. 支持树状注意力进行高效多 token 生成
4. 与目标模型共享嵌入层和 LM 头以节省内存
"""

import ast
from dataclasses import replace
from importlib.util import find_spec
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.deepseek_eagle3 import Eagle3DeepseekV2ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.tree_attn import (
    TreeAttentionMetadata,
    TreeAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import _SAMPLING_EPS
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    compute_new_slot_mapping,
    copy_and_expand_eagle_inputs_kernel,
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
    eagle_step_update_slot_mapping_and_metadata,
    extend_all_queries_by_N,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class SpecDecodeBaseProposer:
    """推测解码基础 proposer 类。

    为 EAGLE 和其他推测解码方法提供通用功能。
    管理输入输出缓冲区、注意力层、CUDA 图调度等。

    Attributes:
        vllm_config: vLLM 配置
        speculative_config: 推测解码配置
        draft_model_config: 草稿模型配置
        method: 推测方法（eagle/eagle3/mtp/draft_model）
        pass_hidden_states_to_model: 是否传递隐藏状态到模型
        runner: 模型运行器
        device: 设备（CUDA）
        dtype: 数据类型
        max_model_len: 最大模型长度
        dp_rank: 数据并行 rank
        num_speculative_tokens: 每个请求的草稿 token 数量
        hidden_size: 隐藏层大小（从草稿模型配置获取）
        inputs_embeds_size: 输入嵌入大小
        parallel_drafting: 是否启用并行推测
        extra_slots_per_request: 每个请求的额外槽位数量
        net_num_new_slots_per_request: 每个请求的净新增槽位数量
        needs_extra_input_slots: 是否需要额外输入槽位
        parallel_drafting_token_id: 并行推测的 token ID
        parallel_drafting_hidden_state_tensor: 并行推测的隐藏状态张量
        use_local_argmax_reduction: 是否使用局部 argmax 归约
        max_num_tokens: 最大 token 数量
        token_arange_np: token 范围 NumPy 数组
        mm_registry: 多模态注册表
        supports_mm_inputs: 是否支持多模态输入
        draft_attn_groups: 草稿注意力组列表
        kv_cache_gid: KV 缓存组 ID
        eagle3_use_aux_hidden_state: EAGLE3 是否使用辅助隐藏状态
        compilation_config: 编译配置
        cudagraph_dispatcher: CUDA 图调度器
        input_ids: 输入 ID 缓冲区
        uses_mrope: 是否使用 M-RoPE
        uses_xdrope_dim: XRoPE 维度数
        draft_uses_xdrope_dim: 草稿模型 XRoPE 维度数
        mrope_positions: M-RoPE 位置缓冲区
        xdrope_positions: XRoPE 位置缓冲区
        positions: 位置缓冲区
        hidden_states: 隐藏状态缓冲区
        block_size: 注意力块大小
        arange: 范围张量
        is_rejected_token_mask: 被拒绝 token 的掩码
        is_masked_token_mask: 被屏蔽 token 的掩码
        inputs_embeds: 输入嵌入缓冲区
        backup_next_token_ids: 备份下一个 token ID 的 CPU-GPU 缓冲区
        _slot_mapping_buffer: 槽映射缓冲区
        allowed_attn_types: 允许的注意力后端类型
        tree_choices: 树状推测的 token 树选择
        num_drafts_per_level: 每层的草稿数
        cu_drafts_per_level: 每层的累积草稿数
        child_drafts_per_level: 每层的子草稿数
        tree_draft_pos_offsets: 树状草稿位置偏移
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        pass_hidden_states_to_model: bool,
        runner=None,
    ):
        """初始化基础 proposer。

        Args:
            vllm_config: vLLM 配置
            device: 设备（CUDA）
            pass_hidden_states_to_model: 是否传递隐藏状态到模型
            runner: 模型运行器（可选）
        """
        self.vllm_config = vllm_config
        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.pass_hidden_states_to_model = pass_hidden_states_to_model

        self.runner = runner
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        # 我们需要从草稿模型配置获取隐藏层大小，因为
        # 草稿模型的隐藏层大小可能与目标模型不同（如 Llama 3.3 70B）
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()

        # 统一 eagle、草稿模型和并行推测支持
        self.parallel_drafting: bool = self.speculative_config.parallel_drafting
        self.extra_slots_per_request = (
            1 if not self.parallel_drafting else self.num_speculative_tokens
        )
        self.net_num_new_slots_per_request = self.extra_slots_per_request - (
            1 if self.pass_hidden_states_to_model else 0
        )
        self.needs_extra_input_slots = self.net_num_new_slots_per_request > 0

        self.parallel_drafting_token_id: int = 0
        self.parallel_drafting_hidden_state_tensor: torch.Tensor | None = None
        if self.parallel_drafting:
            self._init_parallel_drafting_params()
        self.use_local_argmax_reduction: bool = (
            self.speculative_config.use_local_argmax_reduction
        )

        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)

        # 多模态数据支持
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )

        self.draft_attn_groups: list[AttentionGroup] = []
        self.kv_cache_gid: int = -1
        self.eagle3_use_aux_hidden_state: bool = (
            self._get_eagle3_use_aux_hidden_state_from_config()
        )

        self.compilation_config = self.vllm_config.compilation_config

        # CUDA 图调度器，仅用于 EAGLE 中的 PIECEWISE 调度
        # 键在 adjust_cudagraph_sizes_for_spec_decode 调用后通过
        # initialize_cudagraph_keys() 初始化
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # 持久性缓冲区（用于 CUDA 图）
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        # 使用草稿模型的 M-RoPE 设置，而非目标模型的
        # 即使目标是多模态的，草稿模型也可能是纯文本的
        self.uses_mrope = self.draft_model_config.uses_mrope
        self.uses_xdrope_dim = self.vllm_config.model_config.uses_xdrope_dim
        self.draft_uses_xdrope_dim = self.draft_model_config.uses_xdrope_dim
        if self.uses_mrope:
            # 注意：`mrope_positions` 故意多实现了一个额外的虚拟位置
            # 使其非连续，以便与 torch compile 一起工作
            # 详见：https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # 注意：启用 M-RoPE 时，无论输入模态如何，位置 ID 都是 3D 的
            # 对于纯文本输入，每个维度都有相同的位置 ID，
            # 使 M-RoPE 功能上等同于 1D-RoPE
            # 见 https://arxiv.org/abs/2409.12191 第 5 页
            self.mrope_positions = torch.zeros(
                (3, self.max_num_tokens + 1), dtype=torch.int64, device=device
            )
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions = torch.zeros(
                (self.uses_xdrope_dim, self.max_num_tokens + 1),
                dtype=torch.int64,
                device=device,
            )
        else:
            # RoPE 需要 (max_num_tokens,)
            self.positions = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=device
            )
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        # 将在初始化注意力后端时设置
        self.block_size: int = -1

        # 我们需要 +1 是因为 arange 用于设置 query_start_loc，
        # 它比 batch_size 多一个元素
        max_num_slots_for_arange = max(max_batch_size + 1, self.max_num_tokens)
        self.arange = torch.arange(
            max_num_slots_for_arange, device=device, dtype=torch.int32
        )

        if self.needs_extra_input_slots:
            self._raise_if_padded_drafter_batch_disabled()
            self._raise_if_multimodal()
            self._raise_if_mrope()

        self.is_rejected_token_mask: torch.Tensor | None = None
        self.is_masked_token_mask: torch.Tensor | None = None
        if self.needs_extra_input_slots:
            # 对于草稿模型和并行推测，我们需要跟踪哪些 token 被拒绝
            # 以便用填充槽位更新槽映射
            self.is_rejected_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )
            # 对于并行推测，我们还需要跟踪哪些 token 是用于在后续位置采样的
            # 并行填充 token。为简单起见，即使使用草稿模型也填充此张量
            self.is_masked_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.inputs_embeds_size),
            dtype=self.dtype,
            device=device,
        )

        self.backup_next_token_ids = CpuGpuBuffer(
            max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )

        self._slot_mapping_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # 在初始化期间确定允许的注意力后端
        self.allowed_attn_types: tuple | None = None
        if current_platform.is_rocm():
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
                ROCMAiterMLASparseMetadata,
            )
            from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

            rocm_types = [
                TritonAttentionMetadata,
                RocmAttentionMetadata,
                ROCMAiterMLASparseMetadata,
            ]
            # ROCM_AITER_FA 是可选后端
            # 这里检查 is_enabled() 以避免在 VLLM_ROCM_USE_AITER=0 时
            # 自动发现期间导入后端模块，这会触发 aiter 导入和 JIT 编译警告
            # 通过 attention_config 显式选择后端仍然有效，因为后端模块是
            # 直接加载的，而不是通过这个自动发现路径
            if find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)

            # TRITON_MLA 后端支持 MLA 模型（如 DeepSeek）
            from vllm.model_executor.layers.attention.mla_attention import (
                MLACommonMetadata,
            )

            rocm_types.append(MLACommonMetadata)

            # FlexAttention 后端支持
            from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

            rocm_types.append(FlexAttentionMetadata)

            self.allowed_attn_types = tuple(rocm_types)

        # 解析推测 token 树
        spec_token_tree = self.speculative_config.speculative_token_tree
        assert spec_token_tree is not None
        self.tree_choices: list[tuple[int, ...]] = ast.literal_eval(spec_token_tree)
        tree_depth = len(self.tree_choices[-1])
        # 预计算树的每层属性
        num_drafts_per_level = [0] * tree_depth
        for node in self.tree_choices:
            num_drafts_per_level[len(node) - 1] += 1
        self.cu_drafts_per_level = [num_drafts_per_level[0]]
        self.child_drafts_per_level = [num_drafts_per_level[0]]
        for level in range(1, tree_depth):
            self.cu_drafts_per_level.append(
                self.cu_drafts_per_level[-1] + num_drafts_per_level[level]
            )
            self.child_drafts_per_level.append(
                num_drafts_per_level[level] // num_drafts_per_level[level - 1]
            )
        # 预计算扁平化树中的草稿位置偏移
        self.tree_draft_pos_offsets = torch.arange(
            1, len(self.tree_choices) + 1, device=device, dtype=torch.int32
        ).repeat(max_batch_size, 1)

    def _raise_if_padded_drafter_batch_disabled(self):
        """如果禁用了填充草稿批次则抛出异常。

        草稿模型或并行推测仅支持填充草稿批次。
        """
        if self.speculative_config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting only "
                "supports padded drafter batch. Please unset "
                "disable_padded_drafter_batch in the speculative_config."
            )

    def _raise_if_multimodal(self):
        """如果启用多模态则抛出异常。

        草稿模型或并行推测尚不支持多模态模型。
        """
        if self.supports_mm_inputs:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting "
                "does not support multimodal models yet"
            )

    def _raise_if_mrope(self):
        """如果启用 M-RoPE 则抛出异常。

        草稿模型或并行推测尚不支持 M-RoPE。
        """
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting "
                "does not support M-RoPE yet"
            )

    def _init_parallel_drafting_params(self):
        """初始化并行推测参数。

        对于并行推测，我们需要为屏蔽槽位使用特定的 token ID，
        对于 EAGLE + 并行推测，我们需要为这些屏蔽槽位使用隐藏状态张量。
        """
        model_hf_config = self.draft_model_config.hf_config
        if hasattr(model_hf_config, "pard_token"):
            self.parallel_drafting_token_id = model_hf_config.pard_token
        elif hasattr(model_hf_config, "ptd_token_id"):
            self.parallel_drafting_token_id = model_hf_config.ptd_token_id
        else:
            raise ValueError(
                "For parallel drafting, the draft model config must have "
                "`pard_token` or `ptd_token_id` specified in its config.json."
            )

        if self.pass_hidden_states_to_model:
            self.parallel_drafting_hidden_state_tensor = torch.empty(
                self.hidden_size, dtype=self.dtype, device=self.device
            )

    def _get_positions(self, num_tokens: int):
        """获取位置张量。

        Args:
            num_tokens: token 数量

        Returns:
            位置张量
        """
        if self.uses_mrope:
            return self.mrope_positions[:, :num_tokens]
        if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            return self.xdrope_positions[:, :num_tokens]
        return self.positions[:num_tokens]

    def _set_positions(self, num_tokens: int, positions: torch.Tensor):
        """设置位置张量。

        Args:
            num_tokens: token 数量
            positions: 要设置的位置张量
        """
        if self.uses_mrope:
            self.mrope_positions[:, :num_tokens] = positions
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[:, :num_tokens] = positions
        else:
            # 如果目标模型使用 M-RoPE 但草稿模型不使用，则转换 M-RoPE 位置
            # 对于文本输入，所有 M-RoPE 维度都是相同的
            if self.vllm_config.model_config.uses_mrope:
                positions = positions[0]
            self.positions[:num_tokens] = positions

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """为 EAGLE 层返回槽映射字典。

        如果提供了 slot_mapping，首先复制到缓冲区中。

        Args:
            num_tokens: token 数量
            slot_mapping: 槽映射张量（可选）

        Returns:
            槽映射字典
        """
        if slot_mapping is not None:
            num_actual = slot_mapping.shape[0]
            self._slot_mapping_buffer[:num_actual].copy_(slot_mapping)
            if num_tokens > num_actual:
                self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)

        view = self._slot_mapping_buffer[:num_tokens]
        return {name: view for name in self._draft_attn_layer_names}

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """为 EAGLE 初始化 CUDA 图调度器键。

        EAGLE 仅支持 PIECEWISE CUDA 图（通过 mixed_mode）。
        应在 adjust_cudagraph_sizes_for_spec_decode 调用后调用此方法。

        Args:
            cudagraph_mode: CUDA 图模式
        """
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            eagle_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """从隐藏状态贪婪采样草稿 token。

        Args:
            hidden_states: 隐藏状态张量

        Returns:
            采样的草稿 token ID
        """
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)
        return self.model.compute_logits(hidden_states).argmax(dim=-1)

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        """生成草稿 token。

        使用 EAGLE 模型基于目标模型的隐藏状态生成候选 token。

        Args:
            target_token_ids: 目标 token ID [num_tokens]
            target_positions: 目标位置 [num_tokens] 或 [3, num_tokens]（M-RoPE）
            target_hidden_states: 目标隐藏状态 [num_tokens, hidden_size]
            next_token_ids: 下一个 token ID [batch_size]
            token_indices_to_sample: 要采样的 token 索引
            common_attn_metadata: 通用注意力元数据
            sampling_metadata: 采样元数据
            mm_embed_inputs: 多模态嵌入输入（可选）
            num_rejected_tokens_gpu: GPU 上的被拒绝 token 数量（可选）
            slot_mappings: 槽映射（可选）

        Returns:
            草稿 token ID 张量 [batch_size, num_speculative_tokens]
        """
        batch_size = common_attn_metadata.batch_size()

        if self.method == "eagle3":
            assert isinstance(
                self.model, (Eagle3LlamaForCausalLM, Eagle3DeepseekV2ForCausalLM)
            )
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size

        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        assert self.runner is not None

        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=0
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )

        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                num_input_tokens, common_attn_metadata.slot_mapping
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        # 如果只有一个草稿 token 要生成，提前退出
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids = self._greedy_sample(sample_hidden_states)
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        if self.uses_mrope:
            positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        if isinstance(attn_metadata, TreeAttentionMetadata):
            # 使用树状注意力进行草稿 - 需要完整 logits 进行 top-k 采样
            logits = self.model.compute_logits(sample_hidden_states)
            draft_token_ids_list = self.propose_tree(
                batch_size=batch_size,
                logits=logits,
                positions=positions,
                hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata,
                slot_mappings=slot_mappings,
            )
            # [batch_size, num_tree_tokens]
            return torch.cat(draft_token_ids_list, dim=1)

        draft_token_ids = self._greedy_sample(sample_hidden_states)

        if self.allowed_attn_types is not None and not isinstance(
            attn_metadata, self.allowed_attn_types
        ):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding with num_speculative_tokens > 1: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}"
            )

        # 生成剩余的草稿 token
        draft_token_ids_list = [draft_token_ids]

        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(batch_size)
        )

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        # 在填充草稿批次中，我们需要调整序列长度
        # 以移除"填充"（即被拒绝的 token）
        # 仅在我们有被拒绝 token 时应用此调整（即不是第一次提案）
        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            # 使 CPU 端影子无效以避免 H<>D 同步
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        block_size = self.block_size
        assert block_size > 0, "block_size has not been initialized."
        for token_index in range(self.num_speculative_tokens - 1):
            # 更新输入
            # 当 eagle 模型被编译时，转换为 int32 至关重要
            # tensor.argmax() 默认返回 int64
            input_ids = draft_token_ids_list[-1].int()
            # 对槽映射和元数据更新使用融合 kernel
            # 直接写入位置缓冲区以避免额外 D2D 复制（常见非 mrope 情况）
            positions_1d = positions[0] if self.uses_mrope else positions
            if self.uses_mrope:
                out_pos = self.mrope_positions[0, :batch_size]
            elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                out_pos = self.xdrope_positions[0, :batch_size]
            else:
                out_pos = self.positions[:batch_size]
            eagle_step_update_slot_mapping_and_metadata(
                positions_1d=positions_1d,
                block_table_tensor=common_attn_metadata.block_table_tensor,
                seq_lens=common_attn_metadata.seq_lens,
                block_size=block_size,
                max_model_len=self.max_model_len,
                out_clamped_positions=out_pos,
                out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
                input_batch_size=input_batch_size,
            )
            common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:batch_size]
            if self.uses_mrope:
                self.mrope_positions[1:, :batch_size] = self.mrope_positions[
                    0, :batch_size
                ]
                positions = self.mrope_positions[:, :batch_size]
            elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[
                    0, :batch_size
                ]
                positions = self.xdrope_positions[0, :batch_size]
            else:
                positions = self.positions[:batch_size]
            # 增加最大序列长度。我们无条件增加 max_seq_len，
            # 即使某些 seq_lens 可能已在上面被限制，
            # 因为 max_seq_len 作为序列长度的上限
            common_attn_metadata.max_seq_len = min(
                common_attn_metadata.max_seq_len + 1, self.max_model_len
            )

            # 同时更新 CPU 端影子；注意：这是临时的，应在
            # common_attn_metadata.seq_lens_cpu 弃用时移除
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu += 1
            if common_attn_metadata._num_computed_tokens_cpu is not None:
                common_attn_metadata._num_computed_tokens_cpu += 1

            # 重建注意力元数据
            for attn_group in self.draft_attn_groups:
                attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=token_index + 1,
                )
                for layer_name in attn_group.layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata

            # 复制输入到缓冲区用于 CUDA 图
            self.input_ids[:batch_size] = input_ids
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            # 运行模型
            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),
            ):
                ret_hidden_states = self.model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]
            draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        """设置第一次传递的输入。

        准备 EAGLE 模型的输入，包括 token ID、位置和隐藏状态。

        Args:
            target_token_ids: 目标 token ID
            next_token_ids: 下一个 token ID
            target_positions: 目标位置
            target_hidden_states: 目标隐藏状态
            token_indices_to_sample: 要采样的 token 索引
            cad: 通用注意力元数据
            num_rejected_tokens_gpu: GPU 上的被拒绝 token 数量

        Returns:
            (token 数量，要采样的 token 索引，更新后的注意力元数据)
        """
        if not self.needs_extra_input_slots:
            # 默认 EAGLE 路径：不需要重塑输入张量
            # 只需旋转 token ID，将下一个 token ID 插入每个请求的最后一个位置
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]
            # 将 token ID 移位一个 token
            # 例如：[a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            # 替换最后一个 token 为下一个 token
            # 例如：[b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
            self.input_ids[token_indices_to_sample] = next_token_ids

            # 复制输入到缓冲区用于 CUDA 图
            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]
            self._set_positions(num_tokens, target_positions)

            self.hidden_states[:num_tokens] = target_hidden_states

            return num_tokens, token_indices_to_sample, cad
        else:
            assert self.is_rejected_token_mask is not None
            assert self.is_masked_token_mask is not None
            # 1.
            # 调用自定义 triton kernel 将 input_ids 和 positions
            # 复制到预分配的缓冲区 self.input_ids, self.positions 的正确位置
            batch_size = cad.batch_size()
            # 由于我们可能必须为预填充复制大量数据，
            # 我们基于最大查询长度选择块大小并限制最大 256 个槽位/块
            max_num_tokens_per_request = (
                cad.max_query_len + self.net_num_new_slots_per_request
            )
            BLOCK_SIZE_TOKENS = min(
                256, triton.next_power_of_2(max_num_tokens_per_request)
            )
            num_blocks = (
                max_num_tokens_per_request + BLOCK_SIZE_TOKENS - 1
            ) // BLOCK_SIZE_TOKENS
            total_num_input_tokens = target_token_ids.shape[0]
            total_num_output_tokens = total_num_input_tokens + (
                self.net_num_new_slots_per_request * batch_size
            )

            token_indices_to_sample = torch.empty(
                batch_size * self.extra_slots_per_request,
                dtype=torch.int32,
                device=self.device,
            )

            # 目标隐藏状态写入草稿缓冲区的目标索引
            out_hidden_state_mapping = torch.empty(
                total_num_input_tokens, dtype=torch.int32, device=self.device
            )

            # Kernel 网格：每行（每个请求）一个程序
            grid = (batch_size, num_blocks)
            query_start_loc = cad.query_start_loc
            query_end_loc = cad.query_start_loc[1:] - 1
            if num_rejected_tokens_gpu is not None:
                query_end_loc = query_end_loc - num_rejected_tokens_gpu
            copy_and_expand_eagle_inputs_kernel[grid](
                # 来自目标模型的（填充）输入
                target_token_ids_ptr=target_token_ids,
                target_positions_ptr=target_positions,
                next_token_ids_ptr=next_token_ids,  # 每个请求一个采样 token
                # 写入草稿缓冲区
                out_input_ids_ptr=self.input_ids,
                out_positions_ptr=self.positions,  # 暂不支持 mrope
                out_is_rejected_token_mask_ptr=self.is_rejected_token_mask,
                out_is_masked_token_mask_ptr=self.is_masked_token_mask,
                out_new_token_indices_ptr=token_indices_to_sample,
                out_hidden_state_mapping_ptr=out_hidden_state_mapping,
                # 输入元数据
                query_start_loc_ptr=query_start_loc,
                query_end_loc_ptr=query_end_loc,
                padding_token_id=0,
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                # 尺寸信息
                # 注意我们可以从网格大小免费推导 batch_size
                total_input_tokens=total_num_input_tokens,
                num_padding_slots_per_request=self.extra_slots_per_request,
                shift_input_ids=self.pass_hidden_states_to_model,
                BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
            )
            if self.pass_hidden_states_to_model:
                assert self.parallel_drafting_hidden_state_tensor is not None
                self.hidden_states[out_hidden_state_mapping] = target_hidden_states
                # 使用 torch.where 避免 DtoH 同步从布尔索引
                mask = self.is_masked_token_mask[:total_num_output_tokens]
                torch.where(
                    mask.unsqueeze(1),
                    self.parallel_drafting_hidden_state_tensor,
                    self.hidden_states[:total_num_output_tokens],
                    out=self.hidden_states[:total_num_output_tokens],
                )

            # 2.
            # 基于新位置和拒绝掩码重新计算槽映射
            assert self.block_size > 0, "block_size has not been initialized."
            new_slot_mapping = compute_new_slot_mapping(
                cad=cad,
                new_positions=self.positions[:total_num_output_tokens],
                is_rejected_token_mask=self.is_rejected_token_mask[
                    :total_num_output_tokens
                ],
                block_size=self.block_size,
                num_new_tokens=self.net_num_new_slots_per_request,
                max_model_len=self.max_model_len,
            )

            # 3. 用新的（元）数据更新通用注意力元数据
            new_cad = extend_all_queries_by_N(
                cad,
                N=self.net_num_new_slots_per_request,
                arange=self.arange,
                new_slot_mapping=new_slot_mapping,
            )

            return total_num_output_tokens, token_indices_to_sample, new_cad

    def model_returns_tuple(self) -> bool:
        """检查模型是否返回元组。

        Returns:
            如果模型返回 (last_hidden_states, hidden_states) 元组则返回 True
        """
        return self.method not in ("mtp", "draft_model")

    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor:
        """从 CPU 准备下一个 token ID。

        基于 CPU 上的采样 token ID 计算每个请求的下一个 token ID。
        如果请求没有采样 token ID（如在初始解码步骤），
        则回退使用请求状态获取下一个 token ID。

        Args:
            sampled_token_ids: 采样的 token ID 列表
            requests: 请求状态字典
            gpu_input_batch: GPU 输入批次
            num_scheduled_tokens: 每个请求的调度 token 数量

        Returns:
            下一个 token ID 张量
        """
        req_ids = gpu_input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids:
                # 常见情况
                next_token_id = token_ids[-1]
            else:
                # 部分预填充（罕见情况）
                # 从请求状态获取下一个 token ID
                req_id = req_ids[i]
                req_state = requests[req_id]
                seq_len = req_state.num_computed_tokens + num_scheduled_tokens[req_id]
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(
            next_token_ids, dtype=torch.int32, device=self.input_ids.device
        )
        return next_token_ids

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """准备填充模式的下一个 token ID。

        计算每个请求的下一个 token ID 和有效采样 token 数量，
        考虑"丢弃"请求（其下一个 token 未采样，来自 request.get_token_id()）
        的"备份"token ID。还通过 sampled_token_ids 计算被拒绝 token 数量。

        Args:
            common_attn_metadata: 通用注意力元数据
            sampled_token_ids: 采样的 token ID [batch_size, num_tokens]
            requests: 请求状态字典
            gpu_input_batch: GPU 输入批次
            discard_request_mask: 丢弃请求掩码

        Returns:
            (下一个 token ID, 有效采样 token 数量)
        """
        # 当没有有效下一个 token 时预计算 get_token_id
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    common_attn_metadata.seq_lens_cpu[i].item()
                )
                for i in range(num_reqs)
            ],
            dtype=np.int32,
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        next_token_ids = torch.empty(batch_size, dtype=torch.int32, device=device)
        valid_sampled_tokens_count = next_token_ids.new_empty(batch_size)

        # Kernel 网格：每行（每个请求）一个程序
        grid = (batch_size,)

        # 找到块大小的下一个 2 的幂
        BLOCK_SIZE_TOKENS = triton.next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[grid](
            sampled_token_ids,
            discard_request_mask,
            backup_tokens_gpu,
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """准备填充模式的输入。

        更新 common_attn_metadata 用于推测解码，
        但不考虑被拒绝 token。所有 token 都作为输入包含给推测器，
        被拒绝 token 用作填充并在稍后通过 token_indices_to_sample 过滤。
        此函数中不应引入阻塞 CPU 操作。

        Args:
            common_attn_metadata: 通用注意力元数据
            spec_decode_metadata: 推测解码元数据
            valid_sampled_tokens_count: 有效采样 token 数量

        Returns:
            (更新后的注意力元数据，要采样的 token 索引，被拒绝 token 数量)
        """
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )
        num_rejected_tokens_gpu = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )

        # Kernel 网格：每行（每个请求）一个程序
        grid = (num_reqs,)
        eagle_prepare_inputs_padded_kernel[grid](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

    def propose_tree(
        self,
        batch_size: int,
        # [num_tokens, vocab_size]
        logits: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        # [num_tokens, hidden_size]
        hidden_states: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> list[torch.Tensor]:
        """使用树状注意力生成草稿 token。

        基于预定义的 token 树结构，使用树状注意力机制
        并行生成多个草稿 token。

        Args:
            batch_size: 批次大小
            logits: logits 张量 [num_tokens, vocab_size]
            positions: 位置张量 [num_tokens]
            hidden_states: 隐藏状态 [num_tokens, hidden_size]
            common_attn_metadata: 通用注意力元数据
            slot_mappings: 槽映射（可选）

        Returns:
            每层的草稿 token ID 列表
        """
        tree_attn_metadata_builder = self.draft_attn_groups[0].get_metadata_builder()
        assert isinstance(tree_attn_metadata_builder, TreeAttentionMetadataBuilder)

        total_num_drafts = self.cu_drafts_per_level[0]
        level_num_drafts = total_num_drafts
        # 在树根层为每个子节点采样一个草稿 token
        num_children = self.child_drafts_per_level[0]
        if num_children == 1:
            draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
        else:
            draft_token_ids = torch.topk(logits, num_children, dim=-1).indices.view(
                batch_size, -1
            )
        draft_token_ids_list = [draft_token_ids]
        draft_hidden_states = hidden_states.view(batch_size, 1, -1)

        # 初始化空张量用于与层输出拼接
        tree_input_ids = torch.empty(
            0, device=self.input_ids.device, dtype=self.input_ids.dtype
        )
        tree_positions = torch.empty(
            0, device=self.positions.device, dtype=self.positions.dtype
        )
        tree_hidden_states = torch.empty(
            0, device=self.hidden_states.device, dtype=self.hidden_states.dtype
        )
        # 预计算草稿 token 位置
        flattened_draft_positions = (
            positions.view(batch_size, -1) + self.tree_draft_pos_offsets[:batch_size, :]
        )
        tree_depth = len(self.cu_drafts_per_level)
        for level in range(tree_depth - 1):
            # 获取 RoPE 的草稿位置
            draft_positions = positions + (level + 1)
            exceeds_max_model_len = (positions + total_num_drafts) >= self.max_model_len
            # 屏蔽超出最大模型长度的位置 ID
            # 否则我们可能在 RoPE 中得到越界错误
            draft_positions = torch.where(
                exceeds_max_model_len,
                0,
                draft_positions,
            ).view(batch_size, -1)

            if level_num_drafts > 1:
                # 为每个草稿重复位置
                draft_positions = draft_positions.repeat_interleave(
                    level_num_drafts, dim=1
                )

            if num_children > 1:
                # 为每个子节点重复草稿隐藏状态
                draft_hidden_states = draft_hidden_states.repeat_interleave(
                    num_children, dim=1
                )

            # 拼接草稿 token、位置和隐藏状态
            tree_input_ids = torch.cat([tree_input_ids, draft_token_ids], dim=1)
            tree_positions = torch.cat([tree_positions, draft_positions], dim=1)
            tree_hidden_states = torch.cat(
                [tree_hidden_states, draft_hidden_states], dim=1
            )

            # 为下一层草稿构建新的注意力元数据
            # 这对支持树状注意力是必要的
            query_len = total_num_drafts
            common_attn_metadata = replace(
                common_attn_metadata,
                query_start_loc=query_len * self.arange[: batch_size + 1],
                seq_lens=common_attn_metadata.seq_lens + level_num_drafts,
                num_actual_tokens=batch_size * query_len,
                max_query_len=query_len,
            )
            attn_metadata = tree_attn_metadata_builder.build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=level + 1
            )

            # 对所有草稿层应用新的注意力元数据
            per_layer_attn_metadata = {}
            for attn_group in self.draft_attn_groups:
                for layer_name in attn_group.layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata

            # 考虑最大模型长度
            attn_metadata.max_seq_len = min(
                attn_metadata.max_seq_len, self.max_model_len
            )
            # 对于超过最大模型长度的请求，我们将序列长度设置为 1
            # 以最小化注意力开销
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # 计算槽映射
            block_size = tree_attn_metadata_builder.kv_cache_spec.block_size
            query_positions = flattened_draft_positions[:, level : level + query_len]
            block_numbers = query_positions // block_size
            block_ids = attn_metadata.block_table.gather(dim=1, index=block_numbers)
            slot_mapping = block_ids * block_size + query_positions % block_size
            # 屏蔽超出最大模型长度的槽映射
            # 否则 KV 缓存将意外地用填充 token 更新
            slot_mapping[exceeds_max_model_len] = PADDING_SLOT_ID
            attn_metadata.slot_mapping = slot_mapping.view(-1)

            # 复制输入到缓冲区用于 CUDA 图
            num_tokens = attn_metadata.num_actual_tokens
            input_ids = tree_input_ids.view(-1)
            self.input_ids[:num_tokens] = input_ids
            self.positions[:num_tokens] = tree_positions.view(-1)
            self.hidden_states[:num_tokens] = tree_hidden_states.view(num_tokens, -1)

            cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                num_tokens
            )
            num_input_tokens = batch_desc.num_tokens
            # 运行模型
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(
                    num_input_tokens, attn_metadata.slot_mapping
                ),
            ):
                last_hidden_states, hidden_states = self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self.positions[:num_input_tokens],
                    hidden_states=self.hidden_states[:num_input_tokens],
                    inputs_embeds=None,
                )

            # 获取草稿 token 的输出隐藏状态
            draft_hidden_states = hidden_states[:num_tokens].view(
                batch_size, query_len, -1
            )[:, -level_num_drafts:]
            draft_last_hidden_states = last_hidden_states[:num_tokens].view(
                batch_size, query_len, -1
            )[:, -level_num_drafts:]

            # 获取草稿 token 的输出 logits
            logits = self.model.compute_logits(
                draft_last_hidden_states.reshape(batch_size * level_num_drafts, -1)
            )

            # 在下一树层为每个子节点采样一个草稿 token
            num_children = self.child_drafts_per_level[level + 1]
            if num_children == 1:
                draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
            else:
                draft_token_ids = torch.topk(logits, num_children, dim=-1).indices.view(
                    batch_size, -1
                )
            draft_token_ids_list.append(draft_token_ids)

            # 更新下一树层的草稿数计数器
            level_num_drafts = self.cu_drafts_per_level[level + 1] - total_num_drafts
            total_num_drafts = self.cu_drafts_per_level[level + 1]
        return draft_token_ids_list

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """准备输入。

        更新 common_attn_metadata 以考虑被拒绝 token 和新采样 token。
        还返回应输入到推测器的 token 索引。

        Args:
            common_attn_metadata: 通用注意力元数据
            sampled_token_ids: 采样的 token ID 列表
            num_draft_tokens: 每个请求的草稿 token 数量

        Returns:
            (更新后的注意力元数据，token 索引)
        """
        # 例如：
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # 此函数计算中间值：
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # 并返回：
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0
            for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_seq_lens_cpu = common_attn_metadata.seq_lens_cpu - num_rejected_tokens

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # 例如假设 num_tokens_per_req_np = [2, 4, 3]
        # 这意味着 `new_query_start_locs` 是：
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(
            new_query_start_loc_np[:-1], new_num_tokens_per_req_np
        )
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offsets = (
            self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded
        )

        # 扩展起始位置以匹配 token 模式
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np
        )
        # 最终 token 索引是：
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offsets + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=new_seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices

    def get_model_name(self, model: nn.Module) -> str:
        """获取模型名称。

        Args:
            model: 模型

        Returns:
            模型类名
        """
        if hasattr(model, "module"):  # 多 GPU
            model = model.module
        return model.__class__.__name__

    def _get_model(self) -> nn.Module:
        """获取模型。

        获取模型的默认方法。子类可以重写此方法以自定义模型加载。

        Returns:
            加载的模型
        """
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("eagle_head"):
            model = get_model(
                vllm_config=self.vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
            )
        return model

    def load_model(self, target_model: nn.Module) -> None:
        """加载模型。

        加载草稿模型并识别草稿注意力层，共享嵌入层和 LM 头。

        Args:
            target_model: 目标模型
        """
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        self.model = self._get_model()

        # 找到草稿层（草稿模型添加的注意力层）
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self._draft_attn_layer_names = (
            set(all_attn_layers.keys()) - target_attn_layer_names
        )

        if self.supports_mm_inputs:
            # 即使目标模型是多模态的，我们也可以使用纯文本草稿模型
            try:
                dummy_input_ids = torch.tensor([[1]], device=self.input_ids.device)
                self.model.embed_input_ids(dummy_input_ids, multimodal_embeddings=None)
            except (NotImplementedError, AttributeError, TypeError):
                logger.warning(
                    "Draft model does not support multimodal inputs, "
                    "falling back to text-only mode"
                )
                self.supports_mm_inputs = False

        if supports_multimodal(target_model):
            # 处理多模态
            assert hasattr(target_model, "config")
            if self.get_model_name(target_model) in [
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
                "HunYuanVLForConditionalGeneration",
                "GlmOcrForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.vision_config.image_token_id
                )
            elif self.get_model_name(target_model) == "KimiK25ForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.media_placeholder_token_id
                )
            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
            target_language_model = cast(
                SupportsMultiModal, target_model
            ).get_language_model()
        else:
            target_language_model = target_model

        self._maybe_share_embeddings(target_language_model)
        self._maybe_share_lm_head(target_language_model)

        if self.parallel_drafting and self.pass_hidden_states_to_model:
            assert self.parallel_drafting_hidden_state_tensor is not None
            self.parallel_drafting_hidden_state_tensor.copy_(
                self.model.combine_hidden_states(
                    self.model.mask_hidden.view(3 * self.hidden_size)
                )
                if self.eagle3_use_aux_hidden_state
                else self.model.mask_hidden.view(self.hidden_size)
            )

    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        """可能共享嵌入层。

        一些草稿模型可能没有自己的嵌入层，或者可能有目标模型
        嵌入层的重复副本。在这些情况下，我们共享目标模型的
        嵌入层以节省内存。

        Args:
            target_language_model: 目标语言模型
        """
        if get_pp_group().world_size == 1:
            inner_model = getattr(target_language_model, "model", None)
            if inner_model is None:
                raise AttributeError("Target model does not have 'model' attribute")
            if hasattr(inner_model, "embed_tokens"):
                target_embed_tokens = inner_model.embed_tokens
            elif hasattr(inner_model, "embedding"):
                target_embed_tokens = inner_model.embedding
            else:
                raise AttributeError(
                    "Target model does not have 'embed_tokens' or 'embedding' attribute"
                )

            share_embeddings = False
            if hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE 模型
                if not self.model.has_own_embed_tokens:
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model without its own embed_tokens in the"
                        " checkpoint. Sharing target model embedding weights with the"
                        " draft model."
                    )
                elif (
                    isinstance(target_embed_tokens.weight, torch.Tensor)
                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                    # TODO: 卸载到 CPU 进行比较以避免在 CI 测试环境中
                    # 额外使用 GPU 内存（GPU 内存有限）
                    and torch.equal(
                        target_embed_tokens.weight.cpu(),
                        self.model.model.embed_tokens.weight.cpu(),
                    )
                ):
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model with embed_tokens identical to the target"
                        " model. Sharing target model embedding weights with the draft"
                        " model."
                    )
                else:
                    logger.info(
                        "Detected EAGLE model with distinct embed_tokens weights. "
                        "Keeping separate embedding weights from the target model."
                    )
            else:
                # MTP 模型
                share_embeddings = True
                logger.info(
                    "Detected MTP model. "
                    "Sharing target model embedding weights with the draft model."
                )

            if share_embeddings:
                if hasattr(self.model.model, "embed_tokens"):
                    del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_embed_tokens
        else:
            logger.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        """可能共享 LM 头。

        一些草稿模型可能没有自己的 LM 头，或者可能有目标模型
        LM 头的重复副本。在这些情况下，我们共享目标模型的
        LM 头以节省内存。

        Args:
            target_language_model: 目标语言模型
        """
        share_lm_head = False
        if hasattr(self.model, "has_own_lm_head"):
            # EAGLE 模型
            if not self.model.has_own_lm_head:
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model without its own lm_head in the checkpoint. "
                    "Sharing target model lm_head weights with the draft model."
                )
            elif (
                hasattr(target_language_model, "lm_head")
                and isinstance(target_language_model.lm_head.weight, torch.Tensor)
                and isinstance(self.model.lm_head.weight, torch.Tensor)
                # TODO: 卸载到 CPU 进行比较以避免在 CI 测试环境中
                # 额外使用 GPU 内存（GPU 内存有限）
                and torch.equal(
                    target_language_model.lm_head.weight.cpu(),
                    self.model.lm_head.weight.cpu(),
                )
            ):
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model with lm_head identical to the target model. "
                    "Sharing target model lm_head weights with the draft model."
                )
            else:
                logger.info(
                    "Detected EAGLE model with distinct lm_head weights. "
                    "Keeping separate lm_head weights from the target model."
                )
        else:
            # MTP 模型
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )

        if share_lm_head and hasattr(target_language_model, "lm_head"):
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_language_model.lm_head

            # MTP 模型通过 shared_head.head（每个 MTP 层内的 ParallelLMHead）
            # 调用 compute_logits，而不是 self.model.lm_head
            # 如果检查点在 MTP 层路径省略了 lm_head 权重副本，
            # shared_head.head 会保持未初始化并产生 NaN logits
            # 总是显式共享它
            inner = getattr(self.model, "model", None)
            layers = getattr(inner, "layers", None) if inner else None
            if layers is not None:
                items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
                for layer in items:
                    sh = getattr(layer, "shared_head", None)
                    if sh is not None and hasattr(sh, "head"):
                        del sh.head
                        sh.head = target_language_model.lm_head
                        logger.info(
                            "Shared target model lm_head with MTP shared_head.head."
                        )

        if self.use_local_argmax_reduction:
            if not hasattr(self.model, "get_top_tokens"):
                raise ValueError(
                    "use_local_argmax_reduction is enabled but draft model "
                    f"{self.model.__class__.__name__} does not implement "
                    "get_top_tokens()."
                )
            # 如果草稿模型有词表重映射则警告，这会强制回退到
            # 完整 logits 路径（抵消优化）
            if (
                hasattr(self.model, "draft_id_to_target_id")
                and self.model.draft_id_to_target_id is not None
            ):
                logger.warning(
                    "use_local_argmax_reduction is enabled but draft model "
                    "uses draft_id_to_target_id vocab remapping. The "
                    "optimization will be bypassed (falling back to full "
                    "logits gather + argmax)."
                )
            else:
                logger.info(
                    "Using local argmax reduction for draft token generation "
                    "(communication: O(2*tp_size) vs O(vocab_size))."
                )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """运行虚拟推理。

        用于初始化模型和 CUDA 图。

        Args:
            num_tokens: token 数量
            use_cudagraphs: 是否使用 CUDA 图
            is_graph_capturing: 是否正在捕获图
            slot_mappings: 槽映射（可选）
        """
        # 注意：当使用基于树的推测解码时，根据树的深度
        # 调整前向传递次数
        for fwd_idx in range(
            self.num_speculative_tokens if not is_graph_capturing else 1
        ):
            if fwd_idx <= 1:
                cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                    self._determine_batch_execution_and_padding(
                        num_tokens, use_cudagraphs=use_cudagraphs
                    )
                )

            # 确保在 CUDA 图捕获期间使用 EAGLE 自己的缓冲区
            if (
                self._draft_attn_layer_names
                and slot_mappings is not None
                and next(iter(self._draft_attn_layer_names)) in slot_mappings
            ):
                slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
            else:
                slot_mapping_dict = slot_mappings or {}

            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None

                kwargs = dict(
                    input_ids=input_ids,
                    positions=self._get_positions(num_input_tokens),
                    inputs_embeds=inputs_embeds,
                )
                if self.pass_hidden_states_to_model:
                    kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]
                self.model(**kwargs)

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        """从配置获取 EAGLE3 是否使用辅助隐藏状态。

        一些 EAGLE3 头（如 nvidia/gpt-oss-120b-Eagle3-v2）不使用辅助隐藏状态，
        直接使用最后一层输出，就像 EAGLE1 一样。
        它们可能通过在 hf_config 的"eagle_config"字典中设置
        "use_aux_hidden_state"为 False 来表明这一点。

        Returns:
            如果使用辅助隐藏状态则返回 True
        """
        if self.method != "eagle3":
            return False
        # 默认假设 EAGLE3 头使用辅助隐藏状态
        use_aux_hidden_state = True
        eagle_config = getattr(self.draft_model_config.hf_config, "eagle_config", None)
        if eagle_config is not None:
            use_aux_hidden_state = eagle_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """验证所有草稿层属于同一个 KV 缓存组。

        需要这个假设以确保所有草稿层可以使用相同的注意力元数据。
        未来可能扩展到多个注意力元数据。

        Args:
            kv_cache_config: KV 缓存配置
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (
            len(
                set(
                    [
                        kv_cache_groups[layer_name]
                        for layer_name in self._draft_attn_layer_names
                    ]
                )
            )
            == 1
        ), "All drafting layers should belong to the same kv cache group"

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """初始化草稿层的注意力后端。

        使用 kv_cache_config 为草稿层创建 AttentionGroups。
        从模型运行器的 initialize_metadata_builders 调用。

        Args:
            kv_cache_config: KV 缓存配置
            kernel_block_sizes: kernel 块大小列表（可选）
        """
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        # 找到草稿层属于哪个 kv_cache_group
        self.validate_same_kv_cache_group(kv_cache_config)
        kv_cache_spec = None
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if self._draft_attn_layer_names & set(group.layer_names):
                self.kv_cache_gid = gid
                kv_cache_spec = group.kv_cache_spec
                break

        attention_groups: dict[tuple[str, str], AttentionGroup] = {}
        if kv_cache_spec is not None:
            for layer_name in self._draft_attn_layer_names:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                backend_key = attn_backend.full_cls_name()
                if backend_key not in attention_groups:
                    layer_kv_cache_spec = kv_cache_spec
                    if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                        layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                            layer_name
                        ]

                    kernel_block_size = (
                        kernel_block_sizes[self.kv_cache_gid]
                        if kernel_block_sizes is not None
                        and self.kv_cache_gid < len(kernel_block_sizes)
                        else None
                    )
                    attn_group = AttentionGroup(
                        backend=attn_backend,
                        layer_names=[layer_name],
                        kv_cache_spec=layer_kv_cache_spec,
                        kv_cache_group_id=self.kv_cache_gid,
                    )
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    attention_groups[backend_key] = attn_group
                else:
                    attention_groups[backend_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        self.block_size = (
            self.draft_attn_groups[0].get_metadata_builder().kv_cache_spec.block_size
        )
        logger.debug("Using block size %d for drafting layers", self.block_size)

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        """确定批次执行和填充策略。

        根据 CUDA 图模式和数据并行配置确定如何执行批次。

        Args:
            num_tokens: token 数量
            use_cudagraphs: 是否使用 CUDA 图

        Returns:
            (CUDA 图模式，填充后的 token 数量，跨数据并行的 token 数量)
        """
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
        )
        num_tokens_padded = batch_desc.num_tokens

        # 运行数据并行时需要额外协调，因为我们需要跨 ranks 协调
        # TODO(Flechman): 支持 DBO 微批次
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.vllm_config.parallel_config,
                    allow_microbatching=False,
                    num_tokens_padded=num_tokens_padded,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )
            assert not should_ubatch, "DBO ubatching not implemented for EAGLE"

            # 提取 DP 同步值
            if num_tokens_across_dp is not None:
                dp_rank = self.dp_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # 重新调度以获得正确的 batch_descriptor
                cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # 断言确保商定的 token 计数正确
                # 否则 num_tokens_across_dp 将不再有效
                assert batch_desc.num_tokens == num_tokens_padded
                num_tokens_across_dp[dp_rank] = num_tokens_padded

        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp


class EagleProposer(SpecDecodeBaseProposer):
    """EAGLE 推测解码 proposer。

    继承自 SpecDecodeBaseProposer，专门用于 EAGLE 架构。
    与基础类的主要区别是 pass_hidden_states_to_model=True，
    这意味着 EAGLE 模型接收目标模型的隐藏状态作为输入。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        """初始化 EAGLE proposer。

        Args:
            vllm_config: vLLM 配置
            device: 设备（CUDA）
            runner: 模型运行器（可选）
        """
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )


# 注意（woosuk）：以下代码目前未使用，我们总是使用 argmax
# 来采样草稿 token。我们将在找到管理草稿概率张量的方法后使用此代码。
# 参考 https://github.com/vllm-project/vllm/pull/16899 了解详情。
# 注意（woosuk）：这里的逻辑与主采样代码重复。
# 我们应该重构此代码以重用相同的采样实现。
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算概率并采样下一个 token。

    根据采样参数从 logits 计算概率并采样下一个 token。
    目前未使用，保留供未来使用。

    Args:
        logits: logits 张量
        sampling_metadata: 采样元数据

    Returns:
        (下一个 token ID, 概率)
    """
    if sampling_metadata.all_greedy:
        # 对于贪婪请求，草稿概率在拒绝采样中未使用
        # 因此我们可以只返回 logits
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    assert sampling_metadata.temperature is not None

    # 使用 epsilon 比较检测贪婪采样（温度约 0.0）
    # 与 sampler.py 的_SAMPLING_EPS 阈值一致
    temperature = sampling_metadata.temperature
    # 避免除以零（如果有贪婪请求）
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        temperature = torch.where(is_greedy, 1.0, temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # 注意（woosuk）：我们忽略了大多数采样参数
    # 在生成草稿 token 时。我们只使用温度。虽然这
    # 可能会降低接受率，但它不会影响拒绝采样后
    # 生成 token 的分布。

    # TODO(woosuk): 考虑随机种子
    q = torch.empty_like(probs)
    q.exponential_()
    # 注意（woosuk）：我们不应该使用`probs.div_(q)` 因为草稿概率
    # 稍后将用于拒绝采样
    next_token_ids = probs.div(q).argmax(dim=-1).view(-1)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(is_greedy, greedy_token_ids, next_token_ids)
    return next_token_ids, probs
