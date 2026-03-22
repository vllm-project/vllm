# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ExtractHiddenStatesProposer 模块。

本模块实现了基于提取隐藏状态的推测解码 proposer，负责：
- 缓存目标模型的隐藏状态到 KV 缓存
- 无需实际注意力计算，仅提取和存储隐藏状态
- 用于 KV 转移（KV transfer）等场景

主要类：
- ExtractHiddenStatesProposer: 提取隐藏状态的 proposer

注意：此 proposer 不进行实际的推测，而是返回采样的 token 作为草稿 token，
确保它们总是能验证通过。主要目的是缓存隐藏状态而非推测。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backend import AttentionMetadataBuilder, CommonAttentionMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

PADDING_SLOT_ID = -1


class ExtractHiddenStatesProposer:
    """提取隐藏状态的推测解码 proposer。

    该 proposer 使用 ExtractHiddenStatesModel 模型在 KV 缓存中
    缓存隐藏状态，而不进行实际的注意力计算。
    这允许我们提取和存储隐藏状态以供后续使用（如 KV 转移）。

    此 proposer 不进行实际的推测，而是返回采样的 token 作为"草稿"token，
    确保它们总是能验证通过（匹配）。主要目的是缓存隐藏状态，而非推测。

    Attributes:
        vllm_config: vLLM 配置
        device: 设备（CUDA）
        dtype: 数据类型
        dp_rank: 数据并行 rank
        model: 加载的模型（在 load_model 中初始化）
        attn_layer_names: 注意力层名称列表
        attn_metadata_builder: 注意力元数据构建器
        max_num_tokens: 最大 token 数量用于缓冲区
        hf_config: HuggingFace 模型配置
        num_hidden_states: 隐藏状态数量（由 eagle_aux_hidden_state_layer_ids 决定）
        hidden_size: 隐藏层大小
        hidden_states: 隐藏状态缓冲区 [max_num_tokens, num_hidden_states, hidden_size]
        cudagraph_dispatcher: CUDA Graph 调度器
        _slot_mapping_buffer: 槽映射缓冲区
    """

    def __init__(self, vllm_config: VllmConfig, device):
        """初始化 ExtractHiddenStatesProposer。

        Args:
            vllm_config: vLLM 配置
            device: 设备（CUDA）

        Raises:
            AssertionError: 如果 speculative_config 未设置
            AssertionError: 如果 num_speculative_tokens 不等于 1
            ValueError: 如果 disable_padded_drafter_batch 为 True
            ValueError: 如果 eagle_aux_hidden_state_layer_ids 未设置
        """
        assert vllm_config.speculative_config is not None

        # 提取隐藏状态方法要求 num_speculative_tokens == 1
        assert vllm_config.speculative_config.num_speculative_tokens == 1
        if vllm_config.speculative_config.disable_padded_drafter_batch:
            raise ValueError(
                "disable_padded_drafter_batch is not supported with "
                "extract_hidden_states method"
            )
        self.vllm_config = vllm_config
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        # 模型和注意力层跟踪（在 load_model 中初始化）
        self.model: nn.Module | None = None
        self.attn_layer_names: list[str] = []
        self.attn_metadata_builder: AttentionMetadataBuilder | None = None

        # 缓冲区的最大 token 数量
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens + max_batch_size
        )

        self.hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        layer_ids = getattr(self.hf_config, "eagle_aux_hidden_state_layer_ids", None)
        if not layer_ids:
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids must be set in the draft "
                "model config for extract_hidden_states method"
            )
        self.num_hidden_states = len(layer_ids)
        self.hidden_size = vllm_config.model_config.get_hidden_size()
        # 分配隐藏状态缓冲区
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.num_hidden_states, self.hidden_size),
            dtype=self.dtype,
            device=device,
        )
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # 分配槽映射缓冲区
        self._slot_mapping_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

    def propose(
        self,
        sampled_token_ids: torch.Tensor,
        target_hidden_states: list[torch.Tensor],
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        """通过调用 ExtractHiddenStatesModel 生成草稿 token。

        ExtractHiddenStatesModel 在 KV 缓存中缓存隐藏状态，
        而不进行实际的注意力计算。这允许我们提取和存储隐藏状态
        以供后续使用（如 KV 转移）。

        此 proposer 不进行实际的推测，而是返回采样的 token 作为"草稿"token，
        确保它们总是能验证通过（匹配）。主要目的是缓存隐藏状态，而非推测。

        Args:
            sampled_token_ids: 来自目标模型的采样 token ID [batch_size, 1]
            target_hidden_states: 来自目标模型的隐藏状态列表
                （每个辅助隐藏状态层一个）
            common_attn_metadata: 注意力元数据
            slot_mappings: KV 缓存的槽映射（未使用，为接口兼容性提供）

        Returns:
            草稿 token，形状与 sampled_token_ids 相同
        """
        assert self.model is not None and isinstance(target_hidden_states, list)

        # target_hidden_states 是列表，每个元素是一个层的张量
        # 每个张量形状：[num_tokens, hidden_size]
        # 堆叠为形状：[num_tokens, num_hidden_states, hidden_size]
        stacked_hidden_states = torch.stack(target_hidden_states, dim=1)
        num_tokens = stacked_hidden_states.shape[0]

        # 复制隐藏状态到缓冲区
        self.hidden_states[:num_tokens] = stacked_hidden_states

        assert self.attn_metadata_builder is not None
        attn_metadata = self.attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )

        # 我们假设所有 cache-only 层属于同一个 KV 缓存组，
        # 因此使用相同的注意力元数据。
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        # 确定批次执行模式和填充
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        # 设置 forward 上下文并执行模型
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
            self.model(
                hidden_states=self.hidden_states[:num_input_tokens],
            )

        # 返回采样的 token 作为"草稿"token
        # 形状：[batch_size, 1] 与 num_speculative_tokens=1 匹配
        return sampled_token_ids

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """返回 cache-only 注意力层的 slot_mapping 字典。

        如果提供了 slot_mapping，首先将其复制到缓冲区中。

        Args:
            num_tokens: token 数量
            slot_mapping: 输入的槽映射（可选）

        Returns:
            槽映射字典，每个注意力层名称对应同一个视图
        """
        if slot_mapping is not None:
            num_actual = slot_mapping.shape[0]
            self._slot_mapping_buffer[:num_actual].copy_(slot_mapping)
            if num_tokens > num_actual:
                self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)

        view = self._slot_mapping_buffer[:num_tokens]
        return {name: view for name in self.attn_layer_names}

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        """确定批次执行模式和填充 token 数量。

        Args:
            num_tokens: 实际 token 数量
            use_cudagraphs: 是否使用 CUDA Graphs

        Returns:
            三元组：
                - cudagraph_mode: CUDA Graph 模式
                - num_tokens_padded: 填充后的 token 数量
                - num_tokens_across_dp: 跨数据并行的 token 数量（如果启用 DP）
        """
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
        )
        num_tokens_padded = batch_desc.num_tokens

        # 当启用数据并行时需要额外协调，因为我们需要跨 rank 同步
        # TODO(Flechman): 支持 DBO ubatching
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
            assert not should_ubatch, (
                "DBO ubatching not implemented for extract_hidden_states"
            )

            # 提取 DP 同步后的值
            if num_tokens_across_dp is not None:
                dp_rank = self.dp_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # 使用 DP 填充重新 dispatch 以获得正确的 batch_descriptor
                cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # 断言确保商定的 token 数量正确
                # 否则 num_tokens_across_dp 将不再有效
                assert batch_desc.num_tokens == num_tokens_padded
                num_tokens_across_dp[dp_rank] = num_tokens_padded

        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """初始化 CUDA Graph 调度器的键。

        仅支持 PIECEWISE CUDA Graphs（通过 mixed_mode）。
        应在 adjust_cudagraph_sizes_for_spec_decode 之后调用。

        Args:
            cudagraph_mode: CUDA Graph 模式
        """
        assert self.vllm_config.speculative_config is not None
        if (
            not self.vllm_config.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            proposer_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            proposer_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(proposer_cudagraph_mode)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """运行虚拟推理以初始化模型。

        Args:
            num_tokens: token 数量
            use_cudagraphs: 是否使用 CUDA Graphs
            is_graph_capturing: 是否正在捕获图
            slot_mappings: 槽映射（可选）
        """
        assert self.model is not None, "Model must be initialized before dummy_run"
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        # 在 CUDA Graph 捕获期间使用我们自己的槽映射缓冲区
        if (
            self.attn_layer_names
            and slot_mappings is not None
            and self.attn_layer_names[0] in slot_mappings
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
            self.model(
                hidden_states=self.hidden_states[:num_input_tokens],
            )

    def _build_attn_metadata_builder(
        self, draft_attn_layers: dict[str, AttentionLayerBase]
    ) -> AttentionMetadataBuilder:
        """从草稿注意力层构建注意力元数据构建器。

        Args:
            draft_attn_layers: 草稿注意力层字典

        Returns:
            注意力元数据构建器

        Raises:
            ValueError: 如果没有找到注意力层
        """
        if not draft_attn_layers:
            raise ValueError("No attention layers found for ExtractHiddenStatesModel")
        layer = next(iter(draft_attn_layers.values()))
        attn_backend = layer.get_attn_backend()
        return attn_backend.get_builder_cls()(
            layer.get_kv_cache_spec(self.vllm_config),
            self.attn_layer_names,
            self.vllm_config,
            self.device,
        )

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """为推测解码准备下一个 token ID。

        由于 num_speculative_tokens == 1，sampled_token_ids 的形状为
        (batch_size, 1)。对于每个请求，我们使用采样的 token
        （如果有效且未被丢弃）或使用来自请求状态的备份 token。

        Args:
            common_attn_metadata: 通用注意力元数据
            sampled_token_ids: 采样的 token ID
            requests: 请求状态字典
            gpu_input_batch: GPU 输入批次
            discard_request_mask: 丢弃请求掩码

        Returns:
            二元组：
                - next_token_ids: 下一个 token ID
                - valid_sampled_tokens_count: 有效采样 token 数量
        """
        num_reqs = gpu_input_batch.num_reqs
        device = sampled_token_ids.device

        # 为被丢弃/无效的请求计算备份 token
        backup_tokens_gpu = torch.tensor(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    common_attn_metadata.seq_lens_cpu[i].item()
                )
                for i in range(num_reqs)
            ],
            dtype=torch.int32,
            device=device,
        )

        assert discard_request_mask.dtype == torch.bool

        # 当 num_speculative_tokens == 1 时，恰好有一个 token
        sampled = sampled_token_ids[:, 0]
        is_valid = (sampled >= 0) & (sampled < gpu_input_batch.vocab_size)
        valid_sampled_tokens_count = is_valid.to(torch.int32)

        # 使用采样 token 当且仅当有效且未被丢弃
        use_sampled = is_valid & ~discard_request_mask[:num_reqs]
        next_token_ids = torch.where(
            use_sampled, sampled.to(torch.int32), backup_tokens_gpu
        )

        return next_token_ids, valid_sampled_tokens_count

    def load_model(self, target_model: nn.Module) -> None:
        """加载 ExtractHiddenStatesModel 模型。

        该方法实例化 ExtractHiddenStatesModel 模型，该模型用于
        在推测解码期间缓存隐藏状态。模型使用 cache-only 注意力
        （无计算，仅缓存 KV 状态）。

        Args:
            target_model: 目标模型（为与 EagleProposer 接口兼容而传入，
                但此处不使用）
        """
        # 在加载草稿模型之前获取目标模型的注意力层
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()  # type: ignore[type-abstract]
        )

        assert self.vllm_config.speculative_config is not None
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("extract_hidden_states"):
            self.model = get_model(
                vllm_config=self.vllm_config, model_config=draft_model_config
            )

        # 识别草稿模型的注意力层（与目标模型的差异）
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        draft_attn_layers = {
            name: layer
            for name, layer in all_attn_layers.items()
            if name not in target_attn_layer_names
        }
        self.attn_layer_names = list(draft_attn_layers.keys())
        # ExtractHiddenStatesModel 应该恰好有一个注意力层
        assert len(draft_attn_layers) == 1, (
            "ExtractHiddenStatesModel should have exactly one "
            f"attention layer, found {len(draft_attn_layers)}"
        )
        self.attn_metadata_builder = self._build_attn_metadata_builder(
            draft_attn_layers
        )

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """验证所有 drafting 层属于同一个 KV 缓存组。

        由于在 load_model 中断言只有一个注意力层，这自然满足。

        Args:
            kv_cache_config: KV 缓存配置
        """
        assert len(self.attn_layer_names) == 1
