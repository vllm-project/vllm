# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import ClassVar, cast

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.qwen3_vl import mrope_positions_factor
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.index_group import (
    HiSparseMLAIndexGroup,
    SparseMLAIndexGroupBuilder,
    get_sparse_mla_index_group_max_rows,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.hisparse.runtime import (
    HiSparsePrefillStagingPlan,
    build_hisparse_prefill_staging_plan,
    hisparse_staging_block_capacity,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayerConfig,
    SparseCacheRole,
    get_kv_quant_mode,
)
from vllm.v1.worker.workspace import current_workspace_manager

from ..common.qsa import Qwen4ExpQSAMetadata, Qwen4ExpQSAMetadataBuilder
from ..common.qsa_indexer import QSAIndexerMetadata, qsa_kv_cache_dtype
from . import model
from .ops.qsa_flashinfer import (
    GROUP_SIZE,
    get_workspace_size,
    is_flashinfer_qsa_supported,
    qsa_sparse_paged_prefill,
)
from .ops.qsa_indexer import _TOPK_WORKSPACE_BYTES
from .qsa_indexer import QSAIndexer


@dataclass
class Qwen4ExpQSAHiSparsePrefillMetadata:
    host_staging_plan: HiSparsePrefillStagingPlan


@dataclass(kw_only=True)
class Qwen4ExpQSAHiSparseMetadata(Qwen4ExpQSAMetadata):
    """The batch fields `HiSparseMLAIndexGroup` reads."""

    num_reqs: int
    num_decodes: int
    num_decode_tokens: int
    block_size: int
    req_id_per_token: torch.Tensor
    query_start_loc: torch.Tensor
    prefill: Qwen4ExpQSAHiSparsePrefillMetadata | None


class Qwen4ExpQSAQKVIndexerLinear(MergedColumnParallelLinear):
    """Pack sharded Q/gate, K, V and replicated indexer Q/K in one GEMM."""

    def __init__(
        self,
        qkv_proj: QKVParallelLinear,
        index_size: int,
        quant_config: QuantizationConfig | None,
    ) -> None:
        self.num_kv_head_replicas = qkv_proj.num_kv_head_replicas
        super().__init__(
            input_size=qkv_proj.input_size,
            output_sizes=[*qkv_proj.output_sizes, index_size * qkv_proj.tp_size],
            bias=False,
            params_dtype=qkv_proj.params_dtype,
            quant_config=quant_config,
            prefix=qkv_proj.prefix,
        )

    def _load_shard(
        self,
        loader: Callable[..., None],
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | tuple[int, ...] | None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == 3:
            shard_rank = 0
        elif loaded_shard_id in (1, 2):
            shard_rank = tp_rank // self.num_kv_head_replicas
        else:
            shard_rank = tp_rank
        self.tp_rank = shard_rank
        if param_tp_rank is not None:
            param.tp_rank = shard_rank
        try:
            loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(super().weight_loader, param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(
            super().weight_loader_v2, param, loaded_weight, loaded_shard_id
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        def remap_shards():
            for name, weight in weights:
                shard_id = getattr(weight, "shard_id", None)
                if isinstance(shard_id, str):
                    weight = weight.detach()
                    weight.shard_id = {"q": 0, "k": 1, "v": 2}[shard_id]
                yield name, weight

        return super().load_weights(remap_shards())


class Qwen4ExpQSAHiSparseMetadataBuilder(Qwen4ExpQSAMetadataBuilder):
    """Order decodes first and add the batch fields HiSparse reads."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.token_to_req_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Qwen4ExpQSAHiSparseMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.reorder_batch_threshold or 1
        )
        block_size = self.kv_cache_spec.block_size
        prefill = None
        if num_prefills:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            prefill = Qwen4ExpQSAHiSparsePrefillMetadata(
                build_hisparse_prefill_staging_plan(
                    common_attn_metadata.block_table_tensor[num_decodes:],
                    common_attn_metadata.seq_lens[num_decodes:],
                    block_size,
                    hisparse_staging_block_capacity(
                        seq_lens_cpu[num_decodes:], block_size
                    ),
                )
            )
        return Qwen4ExpQSAHiSparseMetadata(
            **vars(metadata),
            num_reqs=common_attn_metadata.num_reqs,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            block_size=block_size,
            req_id_per_token=common_attn_metadata.token_to_req_indices(
                self.token_to_req_buffer
            ),
            query_start_loc=common_attn_metadata.query_start_loc,
            prefill=prefill,
        )


@dataclass
class Qwen4ExpQSAFlashInferPrefillMetadata:
    """FlashInfer inputs for the prefill rows."""

    # [num_prefills]
    seq_lens: torch.Tensor
    # Offsets of the routes over the prefill rows.
    qo_indptr: torch.Tensor
    block_size: int
    max_seq_len_kv: int
    # The largest step's, so the caching allocator reuses one block.
    workspace_size: int


@dataclass
class Qwen4ExpQSANvidiaMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    # Decode requests lead the batch.
    num_decodes: int
    num_decode_tokens: int
    flashinfer_prefill: Qwen4ExpQSAFlashInferPrefillMetadata | None


class Qwen4ExpQSANvidiaMetadataBuilder(
    AttentionMetadataBuilder[Qwen4ExpQSANvidiaMetadata]
):
    """Order decodes first; prefill rows run FlashInfer where supported."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table = True
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        text_config = vllm_config.model_config.hf_text_config
        self.block_size = text_config.indexer_compress_ratio
        block_topk = text_config.indexer_budget // self.block_size
        num_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.use_flashinfer = is_flashinfer_qsa_supported(
            kv_cache_spec.dtype,
            self.block_size,
            block_topk,
            kv_cache_spec.head_size,
            num_heads,
            kv_cache_spec.num_kv_heads,
        )
        if not self.use_flashinfer:
            return
        page_size = kv_cache_spec.block_size
        self.max_seq_len_kv = (
            cdiv(vllm_config.model_config.max_model_len, page_size) * page_size
        )
        scheduler_config = vllm_config.scheduler_config
        self.workspace_size = get_workspace_size(
            scheduler_config.max_num_batched_tokens,
            scheduler_config.max_num_seqs,
            num_heads,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            page_size,
            self.max_seq_len_kv,
            self.block_size,
            block_topk,
            device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Qwen4ExpQSANvidiaMetadata:
        cm = common_attn_metadata
        # Exact under adaptive verification, which trims drafts only on device.
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                cm, decode_threshold=self.reorder_batch_threshold or 1
            )
        )
        metadata = Qwen4ExpQSANvidiaMetadata(
            num_actual_tokens=cm.num_actual_tokens,
            max_query_len=cm.max_query_len,
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            flashinfer_prefill=None,
        )
        if not self.use_flashinfer or fast_build or num_prefills == 0:
            return metadata
        query_start_loc = cm.query_start_loc_cpu[num_decodes : cm.num_reqs + 1]
        from flashinfer.attention.prims_ts import make_q_token_kv_block_sparse_qo_indptr

        qo_indptr = make_q_token_kv_block_sparse_qo_indptr(
            query_start_loc - num_decode_tokens,
            num_prefill_tokens,
            group_size=GROUP_SIZE,
            device="cpu",
        )
        # A fresh pinned tensor per step: async scheduling builds the next step
        # while this copy may still be in flight.
        qo_indptr = qo_indptr.pin_memory().to(self.device, non_blocking=True)
        return replace(
            metadata,
            flashinfer_prefill=Qwen4ExpQSAFlashInferPrefillMetadata(
                seq_lens=cm.seq_lens[num_decodes : cm.num_reqs],
                qo_indptr=qo_indptr,
                block_size=self.block_size,
                max_seq_len_kv=self.max_seq_len_kv,
                workspace_size=self.workspace_size,
            ),
        )

    def update_block_table(
        self,
        metadata: Qwen4ExpQSANvidiaMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Qwen4ExpQSANvidiaMetadata:
        return replace(metadata, block_table=blk_table, slot_mapping=slot_mapping)

    def update_draft_decode_metadata(self, metadata: Qwen4ExpQSANvidiaMetadata) -> None:
        """Decode batch bounds stay fixed and addresses reference runner buffers."""


class Qwen4ExpQSABackend(AttentionBackend):
    """Full-attention backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # fp8/fp8_e4m3: e4m3 bytes in a uint8 cache, written by reshape_and_cache
    # with the layer's per-tensor scales and dequantized on load inside the QSA
    # Triton kernel.
    # nvfp4: FlashInfer's NVFP4 pages, written and read by FlashInfer.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "nvfp4",
    ]

    # The owner writes K/V before calling the sparse attention implementation.
    forward_includes_kv_cache_update: bool = False

    @classmethod
    def customize_spec(cls, spec):
        from vllm.v1.attention.backends.flashinfer import FlashInferBackend

        return FlashInferBackend.customize_spec(spec)

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAImpl]:
        return Qwen4ExpQSAImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        return Qwen4ExpQSANvidiaMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAHiSparseBackend(Qwen4ExpQSABackend):
    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAHiSparseMetadataBuilder]:
        return Qwen4ExpQSAHiSparseMetadataBuilder


class Qwen4ExpQSAImpl(
    AttentionImplBase[Qwen4ExpQSANvidiaMetadata | Qwen4ExpQSAHiSparseMetadata]
):
    """Run paged sparse GQA with the QSA Triton kernel, or FlashInfer for
    prefill rows on Blackwell."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        kv_cache_dtype: str,
        index_group: HiSparseMLAIndexGroup | None = None,
    ) -> None:
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3", "nvfp4"):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, FP8-e4m3 or NVFP4 main KV cache"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.index_group = index_group
        if kv_cache_dtype == "nvfp4":
            from flashinfer.sparse import qsa_sparse_paged_attention

            self.nvfp4_attention = qsa_sparse_paged_attention

    def kv_view(self, rows: torch.Tensor) -> torch.Tensor:
        """View `[blocks, block_size, row_width]` rows as this impl's KV cache."""
        return rows.unflatten(-1, (self.num_kv_heads, 2 * self.head_size)).transpose(
            1, 2
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Scatter K/V using the same layout and scales as the fused writer."""
        if self.kv_cache_dtype == "nvfp4":
            from flashinfer import (
                nvfp4_quantize_append_paged_kv_cache_with_slot_mapping,
            )

            from vllm.v1.attention.backends.flashinfer import _nvfp4_kv_views

            data, scales, kv_layout = _nvfp4_kv_views(kv_cache, self.num_kv_heads)
            nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
                key,
                value,
                slot_mapping,
                data,
                scales,
                layer._k_scale,
                layer._v_scale,
                kv_layout=kv_layout,
            )
            return
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Qwen4ExpQSANvidiaMetadata | Qwen4ExpQSAHiSparseMetadata,
        output: torch.Tensor,
        *,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output_gate: torch.Tensor,
        block_indices: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run sparse attention over the owner's indexer-selected K/V rows."""
        num_tokens = attn_metadata.num_actual_tokens
        # No zero fill: the kernel writes every actual row (empty selections
        # included), and padding rows only feed row-wise ops, never sampled
        # outputs — same contract as the other attention backends.
        if num_tokens == 0:
            return output
        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        selected = topk_buffer[:num_tokens]
        query = query[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        out = output[:num_tokens]
        output_gate = output_gate[:num_tokens]
        if self.index_group is None:
            nvidia_metadata = cast(Qwen4ExpQSANvidiaMetadata, attn_metadata)
            num_decode_tokens = nvidia_metadata.num_decode_tokens
            if num_decode_tokens:
                self.attend(
                    layer,
                    query[:num_decode_tokens],
                    kv_cache,
                    selected[:num_decode_tokens],
                    nvidia_metadata.block_table,
                    token_to_req[:num_decode_tokens],
                    False,
                    out[:num_decode_tokens],
                    output_gate[:num_decode_tokens],
                )
            if num_decode_tokens == num_tokens:
                return output
            prefill_rows = slice(num_decode_tokens, num_tokens)
            flashinfer_prefill = nvidia_metadata.flashinfer_prefill
            if flashinfer_prefill is None:
                self.attend(
                    layer,
                    query[prefill_rows],
                    kv_cache,
                    selected[prefill_rows],
                    nvidia_metadata.block_table,
                    token_to_req[prefill_rows],
                    True,
                    out[prefill_rows],
                    output_gate[prefill_rows],
                )
            else:
                qsa_sparse_paged_prefill(
                    query[prefill_rows],
                    kv_cache,
                    nvidia_metadata.block_table,
                    nvidia_metadata.num_decodes,
                    flashinfer_prefill.seq_lens,
                    block_indices[prefill_rows],
                    token_to_req[prefill_rows],
                    positions[prefill_rows],
                    flashinfer_prefill.qo_indptr,
                    flashinfer_prefill.block_size,
                    flashinfer_prefill.max_seq_len_kv,
                    flashinfer_prefill.workspace_size,
                    self.scale,
                    output_gate[prefill_rows],
                    out[prefill_rows],
                )
            return output

        group = self.index_group
        metadata = cast(Qwen4ExpQSAHiSparseMetadata, attn_metadata)
        cache = group.cache(0)
        num_decode_tokens = metadata.num_decode_tokens
        if cache.all_context_pages_resident and num_decode_tokens < num_tokens:
            # The kernel resolves logical indices against the resident view.
            state_indices = cache.runtime.request_state_indices
            assert cache.view is not None and cache.block_table is not None
            assert state_indices is not None
            self.attend(
                layer,
                query,
                self.kv_view(cache.view.cache),
                selected,
                cache.block_table,
                state_indices[token_to_req],
                use_prefill_config,
                out,
                output_gate,
            )
            return output
        if num_decode_tokens:
            physical = group.convert_logical_to_physical_topk(
                0,
                selected[:num_decode_tokens, :-1],
                metadata,
                block_stride_rows=None,
                return_valid_counts=False,
            )
            assert isinstance(physical, torch.Tensor)
            self.attend(
                layer,
                query[:num_decode_tokens],
                self.kv_view(group.physical_kv_cache(0)),
                physical,
                metadata.block_table,
                token_to_req[:num_decode_tokens],
                use_prefill_config,
                out[:num_decode_tokens],
                output_gate[:num_decode_tokens],
                valid_counts=selected[:num_decode_tokens, -1],
            )
        if num_decode_tokens < num_tokens:
            host_rows = cache.runtime.host_cache.view(
                -1, metadata.block_size, cache.runtime.row_width
            )
            staged_cache, block_table, req_ids = group.stage_prefill_rows(
                0, host_rows, metadata
            )
            self.attend(
                layer,
                query[num_decode_tokens:],
                self.kv_view(staged_cache),
                selected[num_decode_tokens:],
                block_table,
                req_ids,
                use_prefill_config,
                out[num_decode_tokens:],
                output_gate[num_decode_tokens:],
            )
        return output

    def attend(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        selected_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output: torch.Tensor,
        output_gate: torch.Tensor,
        valid_counts: torch.Tensor | None = None,
    ) -> None:
        if self.kv_cache_dtype == "nvfp4":
            from vllm.v1.attention.backends.flashinfer import _nvfp4_kv_views

            (k_data, v_data), (k_sf, v_sf), kv_layout = _nvfp4_kv_views(
                kv_cache, self.num_kv_heads
            )
            if kv_layout == "HND":  # FlashInfer takes [blocks, tokens, heads, *] views
                k_data, v_data, k_sf, v_sf = (
                    x.transpose(1, 2) for x in (k_data, v_data, k_sf, v_sf)
                )
            self.nvfp4_attention(
                query,
                k_data,
                v_data,
                selected_indices,
                block_table,
                token_to_req,
                use_prefill_config,
                output,
                layer._k_scale_float,
                layer._v_scale_float,
                output_gate=output_gate,
                k_sf=k_sf,
                v_sf=v_sf,
            )
            return
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        k_scale = v_scale = None
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            # The cache is allocated as uint8; on sm_89+ reinterpret the e4m3
            # bytes (same itemsize, so shape and strides are preserved). Below
            # sm_89 the kernel byte-decodes the raw uint8 storage instead.
            if current_platform.has_device_capability(89):
                key_cache = key_cache.view(torch.float8_e4m3fn)
                value_cache = value_cache.view(torch.float8_e4m3fn)
            # Host-side per-tensor dequant scales (Python floats), as used by
            # other host-scale backends; folded into the kernel's scales.
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float
        if query.dtype != torch.bfloat16 or key_cache.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.uint8,
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16 Q and BF16 or FP8-e4m3 K/V"
            )

        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query,
            key_cache,
            value_cache,
            selected_indices,
            block_table,
            token_to_req,
            use_prefill_config,
            output,
            k_scale=k_scale,
            v_scale=v_scale,
            output_gate=output_gate,
            valid_counts=valid_counts,
        )


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA indexer branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        topk_indices_buffer: torch.Tensor,
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
            "nvfp4",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, FP8-e4m3 or NVFP4 main KV cache"
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
        self.head_dim = int(config.head_dim or self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
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
            mrope_positions_factor=mrope_positions_factor(vllm_config),
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

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
            and (not mrope_section or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = qsa_kv_cache_dtype(cache_config, prefix)
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.uint8):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16, FP8-e4m3 or NVFP4 (uint8) cache storage"
            )
        if (
            self.kv_cache_dtype == "nvfp4"
            and vllm_config.attention_config.hisparse_config is not None
        ):
            raise NotImplementedError("HiSparse does not support an NVFP4 QSA KV cache")
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        index_group: HiSparseMLAIndexGroup | None = None
        if vllm_config.attention_config.hisparse_config is not None:
            group, _ = SparseMLAIndexGroupBuilder(
                topk_indices_buffer[:, :-1],
                get_sparse_mla_index_group_max_rows(vllm_config),
            ).register_layer(
                True,
                vllm_config,
                row_width=self.num_kv_heads * 2 * self.head_dim,
                kv_cache_dtype=self.kv_cache_dtype,
            )
            assert isinstance(group, HiSparseMLAIndexGroup)
            index_group = group
        self.index_group = index_group
        self.hisparse_cache = index_group.cache(0) if index_group else None
        self.attn_backend = (
            Qwen4ExpQSAHiSparseBackend if index_group else Qwen4ExpQSABackend
        )
        self.impl: Qwen4ExpQSAImpl = Qwen4ExpQSAImpl(  # type: ignore[assignment]
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            self.kv_cache_dtype,
            index_group=index_group,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        self.block_topk = self.indexer.token_topk // self.indexer.compress_ratio
        self.flashinfer_qsa_supported = (
            index_group is None
            and is_flashinfer_qsa_supported(
                self.kv_cache_torch_dtype,
                self.indexer.compress_ratio,
                self.block_topk,
                self.head_dim,
                self.num_heads,
                self.num_kv_heads,
            )
        )
        # One launch does the indexer prepare, the main QK-norm/RoPE/gate and
        # the main K/V cache write (see QSAIndexer.forward); otherwise all of
        # them take the separate kernels. The fused writer receives an fp8
        # pointer, which Triton cannot lower below sm_89. NVFP4 runs it into a
        # BF16 staging buffer instead, which FlashInfer then quantizes into the
        # cache (QSAIndexer.forward).
        self.use_fused_qsa_prepare = (
            self.use_fused_qk_norm_rope_gate
            and self.indexer.use_fused_pre_indexer
            and (
                self.kv_cache_torch_dtype != torch.uint8
                or self.kv_cache_dtype == "nvfp4"
                or current_platform.has_device_capability(89)
            )
        )
        # Without a quant config here (BF16, or ModelOpt FP4, which keeps
        # attention in BF16) the fused layer is unquantized. Otherwise the
        # indexer shares the q/k/v GEMM only if it uses the same quant method.
        fused_quant_config = model.without_modelopt_fp4(quant_config)
        projection_method = type(self.qkv_proj.quant_method)
        self.fuse_indexer_projection = vllm_config.lora_config is None and (
            fused_quant_config is None
            or projection_method is type(self.indexer.index_qk_proj.quant_method)
        )
        if self.fuse_indexer_projection:
            self.index_qk_size = self.indexer.index_qk_proj.output_size
            self.qkv_proj = Qwen4ExpQSAQKVIndexerLinear(
                self.qkv_proj,
                self.index_qk_size,
                fused_quant_config,
            )
            del self.indexer.index_qk_proj

        # MTP skip_topk steps reuse rows frozen from step 0; the valid count
        # is a row column, so compaction/reuse keep it paired with the content.
        self.topk_indices_buffer = topk_indices_buffer
        # NVFP4 + fused prepare: the launch writes BF16 K|V rows to workspace
        # staging (row i at staging slot i) and FlashInfer quantizes them.
        self.stage_nvfp4_kv = (
            self.kv_cache_dtype == "nvfp4" and self.use_fused_qsa_prepare
        )
        if self.stage_nvfp4_kv:
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self.register_buffer(
                "staging_slots", torch.arange(max_tokens), persistent=False
            )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def process_weights_after_loading(self) -> None:
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = None
        if self.fuse_indexer_projection:
            mapper = WeightsMapper(
                orig_to_new_stacked={
                    "indexer.index_qk_proj.": ("qkv_proj.", 3),
                }
            )
        return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def nvfp4_staging(self, num_tokens: int) -> torch.Tensor:
        shape = (num_tokens, 1, self.num_kv_heads, 2 * self.head_dim)
        return current_workspace_manager().get_simultaneous((shape, torch.bfloat16))[0]

    def write_nvfp4_staging(
        self, staging: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        key, value = staging[:, 0].split(self.head_dim, dim=-1)
        self.impl.do_kv_cache_update(self, key, value, self.kv_cache, slot_mapping)

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        spec = AttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            cache_role=SparseCacheRole.SPARSE,
            topk_buffer_width=self.indexer.output_width,
            is_index_group_leader=True,
        )
        return FullAttentionManager.layer_config(spec)

    @eager_break_during_capture
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        output_gate: torch.Tensor | None,
        qkv: torch.Tensor,
    ) -> None:
        # query/key/value/output_gate are None when the fused prepare runs
        # inside the indexer launch.
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            # Profiling/dummy run: the indexer allocates its selection scratch
            # on demand, so touch its peak here for the memory profiler (as
            # sparse_attn_indexer does): the prefill logits budget or the
            # worst-case decode logits [decode tokens, compressed columns],
            # plus the top-k workspace.
            logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            cfg = get_current_vllm_config_or_none()
            if cfg is not None:
                spec = cfg.speculative_config
                num_spec = spec.num_speculative_tokens if spec is not None else 0
                decode_tokens = min(
                    cfg.scheduler_config.max_num_seqs * (num_spec + 1),
                    cfg.scheduler_config.max_num_batched_tokens,
                )
                block_size = cfg.cache_config.block_size
                columns = cdiv(cfg.model_config.max_model_len, block_size) * cdiv(
                    block_size, self.indexer.compress_ratio
                )
                logits_bytes = max(logits_bytes, decode_tokens * columns * 4)
                if self.flashinfer_qsa_supported:
                    logits_bytes += get_workspace_size(
                        cfg.scheduler_config.max_num_batched_tokens,
                        cfg.scheduler_config.max_num_seqs,
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        block_size,
                        cdiv(cfg.model_config.max_model_len, block_size) * block_size,
                        self.indexer.compress_ratio,
                        self.block_topk,
                        output.device,
                    )
            _ = torch.empty(
                logits_bytes + _TOPK_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=output.device,
            )
            if self.stage_nvfp4_kv:  # reserve the staging workspace while profiling
                self.nvfp4_staging(self.staging_slots.numel())
            output.zero_()
            return
        main_metadata = cast(
            Qwen4ExpQSANvidiaMetadata | Qwen4ExpQSAHiSparseMetadata,
            metadata[self.layer_name],
        )
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        indexer_metadata = cast(
            QSAIndexerMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        main_kv_cache = self.kv_cache
        slot_mapping = main_metadata.slot_mapping
        block_indices = torch.empty(
            num_tokens,
            self.block_topk,
            dtype=torch.int32,
            device=projected_qk.device,
        )
        if self.index_group is not None:
            self.index_group.prepare_for_batch(0, main_metadata)
            resident, slot_mapping, _ = self.index_group.cache(0).write_target(
                num_tokens, slot_mapping.shape[0]
            )
            main_kv_cache = self.impl.kv_view(resident)
        selected, main_outputs = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
            attn=self,
            qkv=qkv,
            slot_mapping=slot_mapping,
            main_kv_cache=main_kv_cache,
            block_indices=block_indices,
        )
        if selected.shape != (num_tokens, self.indexer.packed_output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        if self.index_group is not None:
            self.index_group.set_logical_topk_ready(0)
        if main_outputs is None:
            assert key is not None and value is not None
            num_rows = slot_mapping.shape[0]
            self.impl.do_kv_cache_update(
                self, key[:num_rows], value[:num_rows], main_kv_cache, slot_mapping
            )
        else:
            query, output_gate = main_outputs
        if self.index_group is not None:
            self.index_group.cache(0).finish_kv_update()
        assert query is not None and output_gate is not None
        self.impl.forward(
            self,
            query,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=indexer_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,
            output_gate=output_gate,
            block_indices=block_indices,
            positions=indexer_metadata.logical_positions,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.fuse_indexer_projection:
            qkv, projected_qk = qkv.split(
                [2 * self.q_size + 2 * self.kv_size, self.index_qk_size], dim=-1
            )
        else:
            projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        num_tokens = hidden_states.shape[0]
        if not self.use_fused_qsa_prepare:
            q, k, v, gate = self._project_qkv_gate(qkv, positions)
            assert gate is not None
            query = q.view(num_tokens, self.num_heads, self.head_dim)
            key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            # Norm/RoPE/gate and the K/V cache write happen inside _run_qsa.
            query = key = value = gate = None
        attn_output = qkv.new_empty(num_tokens, self.num_heads, self.head_dim)
        self._run_qsa(
            projected_qk,
            positions,
            query,
            key,
            value,
            attn_output,
            gate,
            qkv,
        )
        flat_output = attn_output.view(num_tokens, -1)
        output, _ = self.o_proj(flat_output)
        return output


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSABackend",
    "Qwen4ExpQSAImpl",
]
