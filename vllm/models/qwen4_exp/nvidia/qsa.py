# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA attention owner."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, cast
from weakref import WeakValueDictionary

import torch
from torch import nn

from vllm import _custom_ops as custom_ops
from vllm import envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import (
    Q_TOKEN_KV_BLOCK_SPARSE_TS_GROUP_SIZES,
    QSAForwardMetadata,
)
from . import model
from .indexer_qsa import QSAIndexer

_Q_TOKEN_KV_BLOCK_SPARSE_PREFILL_GROUP_SIZE = 4
_Q_TOKEN_KV_BLOCK_SPARSE_TS_HEAD_SIZE = 256
_Q_TOKEN_KV_BLOCK_SPARSE_TS_SPARSE_BLOCK_SIZE = 4
_Q_TOKEN_KV_BLOCK_SPARSE_TS_TILE_Q = 64
_Q_TOKEN_KV_BLOCK_SPARSE_TS_DEVICE_CAPABILITIES = (100, 103)
_Q_TOKEN_KV_BLOCK_SPARSE_TS_EAGER_PLAN_CACHE_CAPACITY = 4


@dataclass(frozen=True)
class _QTokenKvBlockSparseTSPreparedState:
    key: tuple[object, ...]
    workspace: torch.Tensor
    plan: object


class _QTokenKvBlockSparseTSWorkspaces:
    """Share scratch and immutable query routes across ordered layers.

    One instance belongs to a model's static forward context, including MTP.
    The PrimTS owner rejects DBO and microbatching. Graph geometries stay
    disjoint so their split-KV counter layouts cannot overwrite each other.
    Weak entries do not outlive the plans or forward metadata owning storage.
    """

    def __init__(self) -> None:
        self._buffers: WeakValueDictionary[tuple[object, ...], torch.Tensor] = (
            WeakValueDictionary()
        )

    def get(
        self, key: tuple[object, ...], num_bytes: int, device: torch.device
    ) -> torch.Tensor:
        key = (device, num_bytes, *key)
        workspace = self._buffers.get(key)
        if workspace is None:
            workspace = torch.empty(num_bytes, dtype=torch.uint8, device=device)
            self._buffers[key] = workspace
        return workspace

    def get_qo_indptr(
        self,
        key: tuple[object, ...],
        offsets_cpu: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        # Keep the address stable while FlashInfer plans still bind these routes.
        key = ("qo_indptr", device, *key)
        offsets = self._buffers.get(key)
        if offsets is None:
            offsets = offsets_cpu.to(device, non_blocking=True)
            self._buffers[key] = offsets
        return offsets


def _supports_q_token_kv_block_sparse_ts_geometry(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    sparse_block_size: int,
    max_group_size: int,
) -> bool:
    """Return whether every configured QSA route fits the PrimTS kernel."""

    return (
        head_size == _Q_TOKEN_KV_BLOCK_SPARSE_TS_HEAD_SIZE
        and sparse_block_size == _Q_TOKEN_KV_BLOCK_SPARSE_TS_SPARSE_BLOCK_SIZE
        and num_kv_heads > 0
        and num_heads % num_kv_heads == 0
        and max_group_size in Q_TOKEN_KV_BLOCK_SPARSE_TS_GROUP_SIZES
        and max_group_size * (num_heads // num_kv_heads)
        <= _Q_TOKEN_KV_BLOCK_SPARSE_TS_TILE_Q
    )


def _supports_q_token_kv_block_sparse_ts_device() -> bool:
    """Match the exact architectures implemented by the FlashInfer runtime."""

    return any(
        current_platform.is_device_capability(capability)
        for capability in _Q_TOKEN_KV_BLOCK_SPARSE_TS_DEVICE_CAPABILITIES
    )


def _has_q_token_kv_block_sparse_ts_attention() -> bool:
    """Probe the optional FlashInfer API only when backend resolution needs it."""

    from .ops.qsa import has_q_token_kv_block_sparse_ts_attention

    return has_q_token_kv_block_sparse_ts_attention()


def _resolve_q_token_kv_block_sparse_ts_backend(
    *, capable: bool, availability_probe: Callable[[], bool]
) -> bool:
    """Select PrimTS when supported, or preserve the Triton fallback."""

    backend = envs.VLLM_QSA_ATTENTION_BACKEND
    if backend == "triton":
        return False
    available = availability_probe() if capable else False
    supported = capable and available
    if backend == "prims_ts" and not supported:
        raise RuntimeError(
            "VLLM_QSA_ATTENTION_BACKEND=prims_ts requires an SM100 or SM103 GPU, "
            "FlashInfer QToken-KvBlock-Sparse-Attention, and a supported QSA "
            "geometry (head size 256, sparse block size 4, and grouped heads "
            "within TileQ64)"
        )
    return supported


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with Triton or FlashInfer PrimTS."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        *,
        qsa_sparse_block_size: int = _Q_TOKEN_KV_BLOCK_SPARSE_TS_SPARSE_BLOCK_SIZE,
        qsa_max_group_size: int = _Q_TOKEN_KV_BLOCK_SPARSE_PREFILL_GROUP_SIZE,
    ) -> None:
        # Reuse FlashAttention's metadata/DCP initialization, but do not apply
        # its dense-kernel FP8 capability gate. QSA never calls the inherited
        # dense attention kernel: it owns FP8 query quantization, cache update,
        # descales, and sparse attention end to end below.
        base_kv_cache_dtype = (
            "auto" if is_quantized_kv_cache(kv_cache_dtype) else kv_cache_dtype
        )
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=base_kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
        )
        self.kv_cache_dtype = kv_cache_dtype
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "Qwen4Exp QSA supports BF16 and FP8-E4M3 KV caches"
            )
        self.supports_quant_query_input = False
        geometry_supported = _supports_q_token_kv_block_sparse_ts_geometry(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            sparse_block_size=qsa_sparse_block_size,
            max_group_size=qsa_max_group_size,
        )
        self.use_q_token_kv_block_sparse_ts = (
            _resolve_q_token_kv_block_sparse_ts_backend(
                capable=_supports_q_token_kv_block_sparse_ts_device()
                and geometry_supported,
                availability_probe=_has_q_token_kv_block_sparse_ts_attention,
            )
        )

    def _get_q_token_kv_block_sparse_ts_prepared_state(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
        logical_positions: torch.Tensor,
        out: torch.Tensor,
        qo_indptr_cpu: torch.Tensor | None,
        qo_topology: tuple[int, ...] | None,
        group_size: int,
        sparse_block_size: int,
        max_seq_len_kv: int,
        *,
        persistent: bool = False,
    ) -> _QTokenKvBlockSparseTSPreparedState:
        """Return workspace and plan for one graph-stable QSA geometry."""

        from .ops.qsa import (
            q_token_kv_block_sparse_ts_combined_workspace_size,
            q_token_kv_block_sparse_ts_prepare_attention,
        )

        state_key = (
            query.device,
            tuple(query.shape),
            query.stride(),
            query.dtype,
            tuple(out.shape),
            out.stride(),
            out.dtype,
            tuple(key_cache.shape),
            key_cache.stride(),
            key_cache.dtype,
            key_cache.data_ptr(),
            tuple(value_cache.shape),
            value_cache.stride(),
            value_cache.dtype,
            value_cache.data_ptr(),
            tuple(block_indices.shape),
            block_indices.stride(),
            block_indices.dtype,
            tuple(block_table.shape),
            block_table.stride(),
            block_table.dtype,
            tuple(token_to_req.shape),
            token_to_req.stride(),
            token_to_req.dtype,
            tuple(logical_positions.shape),
            logical_positions.stride(),
            logical_positions.dtype,
            group_size,
            sparse_block_size,
            max_seq_len_kv,
            qo_topology,
        )
        if persistent:
            state = layer._q_token_kv_block_sparse_ts_graph_states.get(state_key)
            if state is not None:
                return state
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "QToken-KvBlock-Sparse-Attention PrimTS plans must be "
                    "prepared during CUDA-graph warmup"
                )
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "eager QToken-KvBlock-Sparse-Attention PrimTS state cannot "
                    "be used during CUDA-graph capture"
                )
            state = layer._q_token_kv_block_sparse_ts_eager_states.get(state_key)
            if state is not None:
                layer._q_token_kv_block_sparse_ts_eager_states.move_to_end(state_key)
                return state

        required_bytes = q_token_kv_block_sparse_ts_combined_workspace_size(
            query,
            key_cache,
            block_table,
            block_indices.shape[1],
            max_seq_len_kv=max_seq_len_kv,
            o_data_type=out.dtype,
            qo_indptr=qo_indptr_cpu,
            seq_len_q=group_size if qo_indptr_cpu is not None else None,
            kv_block_size=sparse_block_size,
        )
        if persistent:
            # Share only equal workspace layouts, not layer-specific K/V maps.
            # Query strides and cache pointers do not affect scratch sizing.
            workspace_key = (
                "graph",
                tuple(query.shape),
                query.dtype,
                out.dtype,
                tuple(key_cache.shape[1:]),
                key_cache.dtype,
                block_indices.shape[1],
                group_size,
                sparse_block_size,
                max_seq_len_kv,
                None if qo_indptr_cpu is None else qo_indptr_cpu.numel() - 1,
            )
            workspace = layer._q_token_kv_block_sparse_ts_workspaces.get(
                workspace_key, required_bytes, query.device
            )
        else:
            eager_workspace = layer._q_token_kv_block_sparse_ts_eager_workspace
            if (
                eager_workspace is None
                or eager_workspace.device != query.device
                or eager_workspace.numel() < required_bytes
            ):
                # Every eager plan binds typed views into this arena. Drop those
                # plans and the old arena before growing so none retain stale
                # workspace storage. PrimTS rejects DBO/microbatching, and
                # ordinary eager launches are sequential on the current stream.
                layer._q_token_kv_block_sparse_ts_eager_states.clear()
                layer._q_token_kv_block_sparse_ts_eager_workspace = None
                del eager_workspace
                eager_workspace = layer._q_token_kv_block_sparse_ts_workspaces.get(
                    ("eager",), required_bytes, query.device
                )
                layer._q_token_kv_block_sparse_ts_eager_workspace = eager_workspace
            workspace = eager_workspace
        plan = q_token_kv_block_sparse_ts_prepare_attention(
            query,
            key_cache,
            block_indices,
            workspace,
            out,
            max_seq_len_kv=max_seq_len_kv,
            qo_indptr=qo_indptr_cpu,
            seq_len_q=group_size if qo_indptr_cpu is not None else None,
            kv_block_size=sparse_block_size,
        )
        state = _QTokenKvBlockSparseTSPreparedState(state_key, workspace, plan)
        if persistent:
            layer._q_token_kv_block_sparse_ts_graph_states[state_key] = state
        else:
            eager_states = layer._q_token_kv_block_sparse_ts_eager_states
            eager_states[state_key] = state
            eager_states.move_to_end(state_key)
            while (
                len(eager_states)
                > _Q_TOKEN_KV_BLOCK_SPARSE_TS_EAGER_PLAN_CACHE_CAPACITY
            ):
                eager_states.popitem(last=False)
        return state

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        logical_positions: torch.Tensor,
        use_prefill_config: bool = False,
        query_start_offsets: tuple[int, ...] | None = None,
        has_prefill: bool = True,
        uniform_decode_query_len: int | None = None,
        persistent_plan: bool = False,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        qo_indptr_cache: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        indices_are_blocks = bool(getattr(layer, "qsa_indices_are_blocks", False))
        token_to_req = token_to_req[:num_tokens]
        logical_positions = logical_positions[:num_tokens]
        if query.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q/output")

        query_for_attention = query[:num_tokens]
        bmm1_scale = self.scale
        bmm2_scale = 1.0
        fp8_query_buffer: torch.Tensor | None = None
        if is_quantized_kv_cache(self.kv_cache_dtype):
            if kv_cache.dtype != torch.uint8:
                raise ValueError("FP8 QSA cache storage must use encoded uint8 bytes")
            kv_cache = kv_cache.view(current_platform.fp8_dtype())
            fp8_query_buffer = getattr(layer, "_qsa_fp8_query_buffer", None)
            if fp8_query_buffer is None or fp8_query_buffer.shape[0] < num_tokens:
                raise RuntimeError("QSA owner did not provide its FP8 query buffer")
            query_for_attention = fp8_query_buffer[:num_tokens]
            custom_ops.scaled_fp8_quant(
                query[:num_tokens].view(num_tokens, -1),
                scale=layer._q_scale,
                output=query_for_attention.view(num_tokens, -1),
            )
            bmm1_scale *= layer._q_scale_float * layer._k_scale_float
            bmm2_scale = layer._v_scale_float
        elif kv_cache.dtype != torch.bfloat16:
            raise ValueError("BF16 QSA cache storage must use BF16")

        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)

        if self.use_q_token_kv_block_sparse_ts:
            from .ops.qsa import (
                q_token_kv_block_sparse_ts_qo_indptr,
                q_token_kv_block_sparse_ts_run_prepared,
            )

            if not indices_are_blocks:
                raise RuntimeError(
                    "PrimTS QSA requires compact sparse-block indexer output"
                )

            # The owner stores the combined cache as [P,Hkv,N,2D]. The common
            # transpose/split above gives Triton's [P,N,Hkv,D] views; PrimTS
            # consumes HND pages, so recover [P,Hkv,N,D] without a copy.
            prims_key_cache = canonicalize_singleton_dim_strides(
                key_cache.transpose(1, 2)
            )
            prims_value_cache = canonicalize_singleton_dim_strides(
                value_cache.transpose(1, 2)
            )
            prims_output = output[:num_tokens]
            route_query_start_offsets = query_start_offsets
            if has_prefill:
                # Prefill always uses packed Q. Fast drafting metadata may omit
                # CPU request boundaries, in which case Q1 is the only
                # request-independent grouping.
                if route_query_start_offsets is None:
                    group_size = 1
                    route_query_start_offsets = (0, num_tokens)
                else:
                    group_size = _Q_TOKEN_KV_BLOCK_SPARSE_PREFILL_GROUP_SIZE
                use_fixed_layout = False
            elif query_start_offsets is None:
                # Missing CPU boundaries cannot prove multi-token request
                # ownership. Keep those launches request-independent.
                group_size = 1
                use_fixed_layout = True
            elif layer.q_token_kv_block_sparse_ts_decode_group_size == 1:
                group_size = 1
                use_fixed_layout = True
            elif uniform_decode_query_len == 1:
                # Standalone MTP recurrence invokes the layer once per draft
                # position, even when the configured target-verification
                # width is MTP + 1. Q1 is request-independent and can keep the
                # fixed layout without pretending those separate invocations
                # are one wider group.
                group_size = 1
                use_fixed_layout = True
            elif (
                uniform_decode_query_len
                == layer.q_token_kv_block_sparse_ts_decode_group_size
            ):
                # Uniform target verification contributes MTP + 1 adjacent
                # rows per live request. Fixed routing is legal only after the
                # shared metadata builder proves those exact CPU boundaries.
                assert uniform_decode_query_len is not None
                group_size = uniform_decode_query_len
                use_fixed_layout = True
            else:
                # Decode always uses the fixed layout. A runtime query width
                # that does not prove the configured MTP+1 group falls back
                # to request-independent Q1 instead of introducing a packed
                # decode route and a second graph-facing layout.
                group_size = 1
                use_fixed_layout = True

            route_qo_indptr_cpu: torch.Tensor | None = None
            route_qo_indptr: torch.Tensor | None = None
            if use_fixed_layout:
                if num_tokens % group_size:
                    raise RuntimeError(
                        "fixed QSA decode rows must be divisible by the query "
                        f"group size ({num_tokens=} {group_size=})"
                    )
                num_query_groups = num_tokens // group_size
                route_query = query_for_attention.view(
                    num_query_groups,
                    1,
                    group_size,
                    query_for_attention.shape[1],
                    query_for_attention.shape[2],
                )
                route_output = prims_output.view_as(route_query)
            else:
                assert route_query_start_offsets is not None
                route_key = (
                    route_query_start_offsets,
                    num_tokens,
                    group_size,
                    query.device,
                )
                routes = (
                    None if qo_indptr_cache is None else qo_indptr_cache.get(route_key)
                )
                if routes is None:
                    route_qo_indptr_cpu = q_token_kv_block_sparse_ts_qo_indptr(
                        route_query_start_offsets,
                        num_tokens,
                        group_size,
                    )
                    routes = (
                        route_qo_indptr_cpu,
                        layer._q_token_kv_block_sparse_ts_workspaces.get_qo_indptr(
                            route_key, route_qo_indptr_cpu, query.device
                        ),
                    )
                    if qo_indptr_cache is not None:
                        qo_indptr_cache[route_key] = routes
                route_qo_indptr_cpu, route_qo_indptr = routes
                route_query = query_for_attention
                route_output = prims_output

            # max_seq_len_kv is the model's per-request logical context bound,
            # not the number of physical pages allocated across all requests.
            # The dense table only proves that each request row can address
            # that model-length bound.
            metadata_block_table = attn_metadata.block_table
            sparse_block_size = int(layer.indexer.compress_ratio)
            dense_row_capacity = (
                metadata_block_table.shape[1] * prims_key_cache.shape[2]
            )
            max_seq_len_kv = int(layer.q_token_kv_block_sparse_ts_max_seq_len_kv)
            if max_seq_len_kv > dense_row_capacity:
                raise RuntimeError(
                    "QSA model length exceeds the per-request dense block-table "
                    f"capacity ({max_seq_len_kv=} {dense_row_capacity=})"
                )
            state = self._get_q_token_kv_block_sparse_ts_prepared_state(
                layer,
                route_query,
                prims_key_cache,
                prims_value_cache,
                logical_indices,
                metadata_block_table,
                token_to_req,
                logical_positions,
                route_output,
                route_qo_indptr_cpu,
                route_query_start_offsets if route_qo_indptr_cpu is not None else None,
                group_size,
                sparse_block_size,
                max_seq_len_kv,
                persistent=persistent_plan,
            )

            q_token_kv_block_sparse_ts_run_prepared(
                state.plan,
                route_query,
                prims_key_cache,
                prims_value_cache,
                metadata_block_table,
                logical_indices,
                token_to_req,
                logical_positions,
                route_output,
                qo_indptr=route_qo_indptr,
                sm_scale=bmm1_scale,
                v_scale=bmm2_scale,
            )
            return output

        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query_for_attention,
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            use_prefill_config,
            output[:num_tokens],
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False
    _q_token_kv_block_sparse_ts_graph_states: dict[
        tuple[object, ...], _QTokenKvBlockSparseTSPreparedState
    ]
    _q_token_kv_block_sparse_ts_eager_states: OrderedDict[
        tuple[object, ...], _QTokenKvBlockSparseTSPreparedState
    ]
    _q_token_kv_block_sparse_ts_eager_workspace: torch.Tensor | None
    _q_token_kv_block_sparse_ts_workspaces: _QTokenKvBlockSparseTSWorkspaces
    q_token_kv_block_sparse_ts_max_seq_len_kv: int

    def _clear_q_token_kv_block_sparse_ts_prepared_storage(self) -> None:
        self._q_token_kv_block_sparse_ts_graph_states.clear()
        self._q_token_kv_block_sparse_ts_eager_states.clear()
        self._q_token_kv_block_sparse_ts_eager_workspace = None

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind one cache generation and discard plans bound to its predecessor.

        vLLM first binds a minimal cache while profiling CUDA-graph memory,
        tears it down, and later binds the real cache. Prepared PrimTS plans
        retain K/V tensor maps, and workspaces allocated during the profiling
        capture belong to a throwaway graph pool. Neither may cross this
        rebinding boundary.
        """

        self._clear_q_token_kv_block_sparse_ts_prepared_storage()
        self.kv_cache = kv_cache

    def unbind_kv_cache(self) -> None:
        """Release the KV cache and every prepared object that refers to it."""
        self._clear_q_token_kv_block_sparse_ts_prepared_storage()
        self.kv_cache = torch.tensor([])

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in (
            "auto",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA supports BF16 and FP8-E4M3 KV caches"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        # Decode/verify batches have at most 1 + num_spec query tokens per
        # request; use_prefill_config (max_query_len > this) steers the
        # config table. Shorter batches take the decode profile — harmless,
        # the difference is tile-shape tuning, not correctness.
        self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.uint8):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16 or encoded FP8 cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        decode_group_size = vllm_config.uniform_decode_query_len
        # TODO: add a Q3 kernel configuration instead of falling back to Q1
        # when a framework configures MTP=2.
        if decode_group_size not in Q_TOKEN_KV_BLOCK_SPARSE_TS_GROUP_SIZES:
            decode_group_size = 1
        self.q_token_kv_block_sparse_ts_decode_group_size = int(decode_group_size)
        self.q_token_kv_block_sparse_ts_max_seq_len_kv = int(model_config.max_model_len)
        sparse_block_size = int(config.indexer_compress_ratio)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
            qsa_sparse_block_size=sparse_block_size,
            qsa_max_group_size=max(
                _Q_TOKEN_KV_BLOCK_SPARSE_PREFILL_GROUP_SIZE,
                self.q_token_kv_block_sparse_ts_decode_group_size,
            ),
        )
        if self.impl.use_q_token_kv_block_sparse_ts and parallel_config.use_ubatching:
            raise NotImplementedError(
                "PrimTS QToken-KvBlock-Sparse-Attention workspace sharing "
                "does not support DBO or microbatching"
            )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # Expanded Triton rows retain the trailing valid-count column during
        # MTP reuse; PrimTS rows contain only compact logical block IDs.
        self.qsa_indices_are_blocks = self.impl.use_q_token_kv_block_sparse_ts
        selection_width = (
            self.indexer.block_topk
            if self.qsa_indices_are_blocks
            else self.indexer.packed_output_width
        )
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                selection_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )
        # Only the MTP owner enables this buffer. Target attention uses the
        # current positions directly, without an extra copy per layer.
        self.register_buffer("topk_query_positions_buffer", None, persistent=False)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            self.register_buffer(
                "_qsa_fp8_query_buffer",
                torch.zeros(
                    max_tokens,
                    self.num_heads,
                    self.head_dim,
                    dtype=current_platform.fp8_dtype(),
                ),
                persistent=False,
            )
        self._q_token_kv_block_sparse_ts_graph_states = {}
        self._q_token_kv_block_sparse_ts_eager_states = OrderedDict()
        self._q_token_kv_block_sparse_ts_eager_workspace = None

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        self._q_token_kv_block_sparse_ts_workspaces = next(
            (
                layer._q_token_kv_block_sparse_ts_workspaces
                for layer in static_context.values()
                if isinstance(layer, Qwen4ExpQSAAttention)
            ),
            _QTokenKvBlockSparseTSWorkspaces(),
        )
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Finalize host descales used by the model-facing FP8 QSA path."""

        self.impl.process_weights_after_loading(act_dtype)
        for name in ("q", "k", "v"):
            scale = float(getattr(self, f"_{name}_scale").item())
            setattr(self, f"_{name}_scale_float", scale)
        self._k_scale_cpu.fill_(self._k_scale_float)
        self._v_scale_cpu.fill_(self._v_scale_float)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    @eager_break_during_capture
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        metadata = forward_context.attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
            compact_blocks=self.qsa_indices_are_blocks,
        )
        if selected.shape != (
            num_tokens,
            self.topk_indices_buffer.shape[1],
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        selected_positions = side_metadata.logical_positions
        if self.topk_query_positions_buffer is not None:
            if not self.indexer.skip_topk:
                self.topk_query_positions_buffer[:num_tokens].copy_(
                    selected_positions[:num_tokens]
                )
            selected_positions = self.topk_query_positions_buffer[:num_tokens]
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,
            logical_positions=selected_positions,
            query_start_offsets=side_metadata.query_start_offsets,
            qo_indptr_cache=side_metadata.q_token_kv_block_sparse_qo_indptr,
            has_prefill=side_metadata.has_prefill,
            uniform_decode_query_len=side_metadata.uniform_decode_query_len,
            persistent_plan=(
                forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                or side_metadata.prepare_cudagraph_plan
            ),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        # Keep the index projection outside the eager break.
        projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        self._run_qsa(
            projected_qk,
            positions,
            query,
            key,
            value,
            attn_output,
        )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
]
