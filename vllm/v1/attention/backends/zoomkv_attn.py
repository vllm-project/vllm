# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZoomKV V1 attention backend.

Dense prefill / short-context decode use FlashAttention over the Triton
paged KV layout. Long-context single-token decode runs hierarchical
Quest + KIVI retrieval over physical-block block_summaries, then
non-causal attention over sink + local + Top-K tokens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)

from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
)

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)

from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)

from vllm.v1.attention.ops.zoomkv import recall_probe as _zoomkv_recall
from vllm.v1.attention.ops.zoomkv import stage_timer as _zt
from vllm.v1.attention.ops.zoomkv.paged import (
    assemble_sparse_context_indices,
    assemble_sparse_context_indices_batch,
    gather_kv_by_logical_indices,
    gather_kv_by_logical_indices_batch,
    gather_kv_from_topk_batch,
    gather_kv_hybrid,
    sparse_decode_attention,
    sparse_decode_attention_batch,
)

from vllm.v1.attention.ops.zoomkv.retriever import (
    ZoomKVRetriever,
    ZoomKVRuntimeConfig,
    prepare_retrieval_query,
)

from vllm.v1.attention.ops.zoomkv.state import get_or_create_block_summary
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _needs_summary_update(
    *,
    num_prefills: int,
    max_query_len: int,
    num_decodes: int,
    seq_lens_cpu: torch.Tensor | None,
    num_reqs: int,
    block_size: int,
) -> bool:
    """Return whether this scheduler step can complete a summary block."""
    if (
        num_prefills != 0
        or max_query_len != 1
        or num_decodes <= 0
        or seq_lens_cpu is None
    ):
        return True
    return any(
        int(seq_len) > 0 and int(seq_len) % block_size == 0
        for seq_len in seq_lens_cpu[:num_reqs].tolist()
    )


def _decode_region_sparse_eligible(
    *,
    cfg: ZoomKVRuntimeConfig,
    num_decodes: int,
) -> bool:
    """True when the reordered decode prefix may use sparse decode.

    No seq_len threshold: every single-token decode uses sparse retrieval.
    Mixed batches keep a global ``max_query_len > 1`` from the prefill suffix,
    so callers must not require batch-wide query_len==1.
    """
    return (not cfg.dense_fallback) and num_decodes > 0


def _should_use_sparse_decode(
    *,
    cfg: ZoomKVRuntimeConfig,
    max_query_len: int,
    num_decodes: int,
    num_prefills: int,
    num_reqs: int,
    seq_lens_cpu: torch.Tensor | None,
    seq_lens: torch.Tensor | None = None,
) -> bool:
    """Pure-batch sparse flag (CUDA Graph / all-sparse path).

    Mixed batches keep ``use_sparse=False`` here; forward routes them via
    ``_should_use_mixed_sparse_decode`` plus a dense prefill suffix instead.
    """
    del num_reqs, seq_lens_cpu, seq_lens  # Length gate removed; decode count only.
    if cfg.dense_fallback:
        return False
    # Speculative / multi-token pure decode → dense (first release).
    if max_query_len != 1:
        return False
    if num_prefills > 0:
        return False
    return _decode_region_sparse_eligible(cfg=cfg, num_decodes=num_decodes)


def _should_use_mixed_sparse_decode(
    *,
    cfg: ZoomKVRuntimeConfig,
    num_decodes: int,
    num_prefills: int,
) -> bool:
    """GPU-only mixed batch: sparse decode prefix + dense prefill suffix."""
    if num_prefills <= 0 or cfg.enable_offload:
        return False
    return _decode_region_sparse_eligible(cfg=cfg, num_decodes=num_decodes)


def _load_zoomkv_runtime_config(vllm_config: VllmConfig | None) -> ZoomKVRuntimeConfig:
    if vllm_config is None:
        try:
            from vllm.config import get_current_vllm_config_or_none

            vllm_config = get_current_vllm_config_or_none()
        except Exception:
            vllm_config = None
    if vllm_config is None:
        return ZoomKVRuntimeConfig()
    attn = vllm_config.attention_config
    strict = bool(getattr(attn, "zoomkv_strict_kernels", False))
    if strict:
        # Propagate to kernel dispatch helpers that read the env flag.
        import os

        os.environ["VLLM_ZOOMKV_STRICT_KERNELS"] = "1"
    return ZoomKVRuntimeConfig(
        sink_size=getattr(attn, "zoomkv_sink_size", 64),
        local_size=getattr(attn, "zoomkv_local_size", 256),
        max_model_len=int(vllm_config.model_config.max_model_len),
        final_topk=getattr(attn, "zoomkv_final_topk", 100),
        quest_chunk=getattr(attn, "zoomkv_quest_chunk", 16),
        quest_large_chunk=getattr(attn, "zoomkv_quest_large_chunk", 256),
        quest_large_ratio=getattr(attn, "zoomkv_quest_large_ratio", 0.5),
        quest_small_ratio=getattr(attn, "zoomkv_quest_small_ratio", 0.3),
        dense_ratio=getattr(attn, "zoomkv_dense_ratio", 0.4),
        dense_topk=getattr(attn, "zoomkv_dense_topk", 8),
        sparse_topk=getattr(attn, "zoomkv_sparse_topk", 4),
        full_attention_threshold=getattr(attn, "zoomkv_full_attention_threshold", 2000),
        dense_fallback=getattr(attn, "zoomkv_dense_fallback", False),
        strict_kernels=strict,
        enable_offload=getattr(attn, "zoomkv_enable_offload", False),
    )


def _graph_chunk_bucket(cfg: ZoomKVRuntimeConfig, block_size: int) -> int:
    """Static retrieval capacity for full CUDA Graph capture."""
    start_block = cfg.sink_size // block_size
    local_start = max(cfg.sink_size, cfg.max_model_len - cfg.local_size)
    max_chunks = max(1, local_start // block_size - start_block)
    # Graph shapes are already static; unlike eager scratch buckets, they do
    # not need power-of-two rounding. Align only to a complete parent group.
    factor = cfg.hq_factor
    return max(factor, ((max_chunks + factor - 1) // factor) * factor)


@dataclass
class ZoomKVMetadata(FlashAttentionMetadata):
    num_reqs: int = 0
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0
    zoomkv: ZoomKVRuntimeConfig | None = None
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    # True when at least one request completes a physical block this step.
    # Pure decode skips summary updates on the other 15/16 tokens.
    need_summary_update: bool = True
    # Pure-batch sparse flag (CUDA Graph / all-sparse). Mixed batches keep this
    # False and use ``use_mixed_sparse`` for decode-prefix + prefill-suffix.
    use_sparse: bool = False
    # GPU-only mixed: sparse over reordered decode prefix, dense prefill tail.
    use_mixed_sparse: bool = False
    # True only for the static pure-decode metadata used while capturing a
    # full CUDA Graph.  Python executes once at capture time; replay updates
    # seq_lens/block_table/slot_mapping in-place.
    is_cudagraph_capture: bool = False
    graph_chunk_bucket: int | None = None
    # Preallocated physical Top-K / context index buffers (MLA-style).
    # Shape: [max_num_seqs, num_kv_heads, final_topk]
    topk_indices_buffer: torch.Tensor | None = None
    # Shape: [max_num_seqs, num_kv_heads, sink+local+final_topk]
    context_indices_buffer: torch.Tensor | None = None


class ZoomKVMetadataBuilder(FlashAttentionMetadataBuilder):
    """Build FlashAttention metadata plus ZoomKV knobs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        cfg = _load_zoomkv_runtime_config(vllm_config)
        # The first full-graph implementation deliberately excludes CPU
        # offload and dense-only/debug routing. Mixed/prefill batches continue
        # to use the normal piecewise/eager path.
        if cfg.enable_offload or cfg.dense_fallback:
            return AttentionCGSupport.NEVER
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.zoomkv = _load_zoomkv_runtime_config(vllm_config)
        if kv_cache_spec.block_size != 16:
            raise ValueError(
                f"ZoomKV requires --block-size 16 (got {kv_cache_spec.block_size})"
            )
        if self.zoomkv.quest_chunk != 16:
            raise ValueError(
                "ZoomKV first release requires zoomkv_quest_chunk=16 "
                f"(got {self.zoomkv.quest_chunk})"
            )
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self.graph_chunk_bucket = _graph_chunk_bucket(
            self.zoomkv, kv_cache_spec.block_size
        )
        logger.info_once(
            "ZoomKV GPU-only CUDA Graph retrieval capacity: %d child chunks "
            "(max_model_len=%d, max_small_candidates=%d)",
            self.graph_chunk_bucket,
            self.zoomkv.max_model_len,
            self.zoomkv.max_small_candidates,
        )
        # Preallocate once; sparse decode writes into these buffers instead of
        # allocating temporary Top-K / context index tensors per request.
        max_seqs = int(vllm_config.scheduler_config.max_num_seqs)
        num_kv = int(kv_cache_spec.num_kv_heads)
        final_topk = int(self.zoomkv.final_topk)
        ctx_width = (
            int(self.zoomkv.sink_size) + int(self.zoomkv.local_size) + final_topk
        )
        self.topk_indices_buffer = torch.full(
            (max_seqs, num_kv, final_topk),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.context_indices_buffer = torch.full(
            (max_seqs, num_kv, ctx_width),
            -1,
            dtype=torch.int64,
            device=device,
        )



    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> ZoomKVMetadata:
        metadata = self.build(0, common_attn_metadata)
        # Dummy capture sequence lengths are intentionally tiny in the generic
        # runner. Force the static sparse route; replay supplies real long
        # context lengths through persistent device buffers.
        metadata.use_sparse = True
        metadata.use_mixed_sparse = False
        metadata.need_summary_update = True
        metadata.is_cudagraph_capture = True
        # Capture against the configured maximum context, not the generic
        # runner's tiny dummy sequence lengths. Replay changes only device
        # values (seq_lens/block_table); all retrieval tensor shapes stay fixed.
        metadata.graph_chunk_bucket = self.graph_chunk_bucket
        ZoomKVAttentionImpl._step_need_summary_update = True
        return metadata



    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ZoomKVMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=1,
            )
        )
        # Host-side block-boundary check: after writing the new token,
        # seq_len % block_size == 0 means a child chunk just completed.
        seq_cpu = common_attn_metadata.seq_lens_cpu
        need_summary_update = _needs_summary_update(
            num_prefills=num_prefills,
            max_query_len=common_attn_metadata.max_query_len,
            num_decodes=num_decodes,
            seq_lens_cpu=seq_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            block_size=int(self.kv_cache_spec.block_size),
        )
        use_sparse = _should_use_sparse_decode(
            cfg=self.zoomkv,
            max_query_len=common_attn_metadata.max_query_len,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_reqs=common_attn_metadata.num_reqs,
            seq_lens_cpu=seq_cpu,
            seq_lens=common_attn_metadata.seq_lens,
        )
        use_mixed_sparse = _should_use_mixed_sparse_decode(
            cfg=self.zoomkv,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
        )
        # do_kv_cache_update runs without metadata; publish for this step.
        ZoomKVAttentionImpl._step_need_summary_update = need_summary_update
        fields = {
            "num_actual_tokens": base.num_actual_tokens,
            "max_query_len": base.max_query_len,
            "query_start_loc": base.query_start_loc,
            "max_seq_len": base.max_seq_len,
            "seq_lens": base.seq_lens,
            "block_table": base.block_table,
            "slot_mapping": base.slot_mapping,
            "use_cascade": base.use_cascade,
            "common_prefix_len": base.common_prefix_len,
            "cu_prefix_query_lens": base.cu_prefix_query_lens,
            "prefix_kv_lens": base.prefix_kv_lens,
            "suffix_kv_lens": base.suffix_kv_lens,
            "max_dcp_context_kv_len": base.max_dcp_context_kv_len,
            "dcp_context_kv_lens": base.dcp_context_kv_lens,
            "scheduler_metadata": base.scheduler_metadata,
            "prefix_scheduler_metadata": base.prefix_scheduler_metadata,
            "max_num_splits": base.max_num_splits,
            "causal": base.causal,
            "num_reqs": common_attn_metadata.num_reqs,
            "num_decodes": num_decodes,
            "num_prefills": num_prefills,
            "num_decode_tokens": num_decode_tokens,
            "num_prefill_tokens": num_prefill_tokens,
            "zoomkv": self.zoomkv,
            # Keep host copies in metadata so every full-attention layer does
            # not synchronize the GPU merely to recover scalar sequence
            # geometry.
            "query_start_loc_cpu": common_attn_metadata.query_start_loc_cpu,
            "seq_lens_cpu": common_attn_metadata.seq_lens_cpu,
            "need_summary_update": need_summary_update,
            "use_sparse": use_sparse,
            "use_mixed_sparse": use_mixed_sparse,
            "is_cudagraph_capture": False,
            "graph_chunk_bucket": None,
            "topk_indices_buffer": self.topk_indices_buffer,
            "context_indices_buffer": self.context_indices_buffer,
        }
        return ZoomKVMetadata(**fields)


class ZoomKVAttentionBackend(AttentionBackend):
    """Native ZoomKV sparse-retrieval attention backend."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "ZOOMKV"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size == 16

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128, 256]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (128, 256)

    @classmethod
    def is_sparse(cls) -> bool:
        # Retrieval sparsity is internal; selector treats us as dense decoder attn.
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type[ZoomKVAttentionImpl]:
        return ZoomKVAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[ZoomKVMetadataBuilder]:
        return ZoomKVMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return TritonAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return TritonAttentionBackend.get_kv_cache_stride_order(
            include_num_layers_dimension
        )

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


class ZoomKVAttentionImpl(AttentionImpl[ZoomKVMetadata]):
    # Published by ZoomKVMetadataBuilder.build() each step so
    # do_kv_cache_update (which has no metadata) can skip empty summary updates.
    _step_need_summary_update: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: AttentionType | str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
        chunk_lookback: int = -1,
    ) -> None:
        if head_size not in (128, 256):
            raise ValueError(
                f"ZoomKV requires head_size in {{128, 256}}, got {head_size}"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.block_size = 16
        # FlashAttention dense path config. ZoomKV keeps the Triton paged KV
        # layout for sparse gather/summary compatibility, so dense attention
        # calls flash_attn_varlen_func on the split K/V views rather than
        # FlashAttentionImpl.forward (which expects FA's (2, N, ...) layout).
        if use_alibi_sqrt:
            raise NotImplementedError(
                "ZoomKV dense FlashAttention path does not support use_alibi_sqrt"
            )
        if chunk_lookback is not None and chunk_lookback >= 0:
            raise NotImplementedError(
                "ZoomKV dense FlashAttention path does not support chunk_lookback"
            )
        self._fa = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=self.num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
        )
        self._retriever: ZoomKVRetriever | None = None
        self._layer_name: str | None = None
        # Cache at construction: do_kv_cache_update may run without metadata.
        try:
            from vllm.config import get_current_vllm_config

            self._runtime_cfg = _load_zoomkv_runtime_config(get_current_vllm_config())
        except Exception:
            self._runtime_cfg = _load_zoomkv_runtime_config(None)

    def _get_retriever(self, cfg: ZoomKVRuntimeConfig) -> ZoomKVRetriever:
        if self._retriever is None or self._retriever.cfg != cfg:
            self._retriever = ZoomKVRetriever(cfg)
        return self._retriever

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Support both Triton KV layouts used across vLLM versions.
        - Newer: (num_blocks, num_kv_heads, block_size, 2 * head_size)
        - v0.24: (num_blocks, 2, block_size, num_kv_heads, head_size)
        Returns key/value views as (num_blocks, block_size, num_kv_heads, head_size).
        """
        if kv_cache.ndim == 5 and kv_cache.shape[1] == 2:
            return kv_cache.unbind(1)
        if kv_cache.ndim == 4:
            return kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        raise ValueError(f"Unexpected ZoomKV kv_cache shape: {tuple(kv_cache.shape)}")

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        block_table: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        num_reqs: int = 0,
        scan_all_parents: bool = False,
        graph_chunk_bucket: int | None = None,
    ) -> None:
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        # Refresh ZoomKV block summaries for newly completed physical blocks.
        layer_name = getattr(layer, "layer_name", None) or getattr(
            layer, "name", f"zoomkv_{id(layer)}"
        )
        self._layer_name = str(layer_name)
        num_blocks = kv_cache.shape[0]
        dtype = (
            key.dtype
            if key.dtype in (torch.float16, torch.bfloat16)
            else torch.bfloat16
        )
        cfg = self._runtime_cfg
        block_summary = get_or_create_block_summary(
            layer_name=self._layer_name,
            num_blocks=num_blocks,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            block_size=self.block_size,
            device=kv_cache.device,
            dtype=dtype,
            blocks_per_parent=max(1, cfg.quest_large_chunk // cfg.quest_chunk),
        )
        slots = slot_mapping.flatten()
        # Pure decode: metadata builder publishes whether any request just
        # completed a physical block (seq_len % block_size == 0). Skip the
        # conditional Triton launch on the other 15/16 decode steps.
        if self._step_need_summary_update:
            start_b = cfg.sink_size // self.block_size
            max_parents = None
            if graph_chunk_bucket is not None:
                max_parents = max(
                    1, graph_chunk_bucket // block_summary.blocks_per_parent
                )
            parent_block_table = block_table
            parent_seq_lens = seq_lens
            if parent_seq_lens is not None and num_reqs > 0:
                parent_seq_lens = parent_seq_lens[:num_reqs]
                if parent_block_table is not None:
                    parent_block_table = parent_block_table[:num_reqs]
            else:
                parent_block_table = None
                parent_seq_lens = None
            with _zt.Stage("block_summary.update"):
                block_summary.update_completed_slots(
                    key_cache,
                    slots,
                    block_table=parent_block_table,
                    start_block=start_b,
                    seq_lens=parent_seq_lens,
                    scan_all_parents=scan_all_parents,
                    max_parents=max_parents,
                )

        # K+V offload: after block_summaries are built for completed blocks,
        # async D2H the Key and Value pages. GPU pages are NOT zeroed here —
        # this step's (and later prefill chunks') dense attention still reads
        # them. Zeroing happens lazily in the sparse decode path (mark_cold).
        if cfg.enable_offload:
            from vllm.v1.attention.ops.zoomkv.offload import get_cpu_key_pool

            pool = get_cpu_key_pool()
            if pool is not None:
                # Newly written slots may complete a child chunk.
                valid_slots = slots[slots >= 0]
                if valid_slots.numel():
                    block_ids = torch.div(
                        valid_slots, self.block_size, rounding_mode="floor"
                    )
                    offsets = torch.remainder(valid_slots, self.block_size)
                    complete = block_ids[offsets == (self.block_size - 1)].unique()
                    if complete.numel():
                        with _zt.Stage("block_summary.offload"):
                            pool.offload_blocks_bulk(
                                self._layer_name,
                                key_cache,
                                value_cache,
                                block_summary,
                                complete,
                            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ZoomKVMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)

        # Own the KV update inside this opaque attention op. This removes the
        # separate unified_kv_cache_update op and its dummy dependency while
        # preserving launch order before any dense or sparse cache read.
        if (
            layer.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            with _zt.Stage("sparse.kv_update"):
                self.do_kv_cache_update(
                    layer,
                    key,
                    value,
                    kv_cache,
                    attn_metadata.slot_mapping,
                    block_table=getattr(attn_metadata, "block_table", None),
                    seq_lens=getattr(attn_metadata, "seq_lens", None),
                    num_reqs=getattr(attn_metadata, "num_reqs", 0),
                    scan_all_parents=(
                        getattr(attn_metadata, "num_prefills", 0) > 0
                        or getattr(attn_metadata, "max_query_len", 1) != 1
                    ),
                    graph_chunk_bucket=getattr(
                        attn_metadata, "graph_chunk_bucket", None
                    ),
                )

        with _zt.Stage("sparse.route"):
            cfg = attn_metadata.zoomkv or ZoomKVRuntimeConfig()
            use_sparse = getattr(attn_metadata, "use_sparse", None)
            if use_sparse is None:
                use_sparse = self._should_sparse_decode(attn_metadata, cfg)
            use_mixed_sparse = getattr(attn_metadata, "use_mixed_sparse", None)
            if use_mixed_sparse is None:
                use_mixed_sparse = _should_use_mixed_sparse_decode(
                    cfg=cfg,
                    num_decodes=getattr(attn_metadata, "num_decodes", 0),
                    num_prefills=getattr(attn_metadata, "num_prefills", 0),
                )

        if use_mixed_sparse:
            # Reordered decode prefix → sparse; prefill suffix → dense FA.
            num_decodes = int(attn_metadata.num_decodes)
            num_decode_tokens = int(attn_metadata.num_decode_tokens)
            logger.info_once(
                "ZoomKV mixed batch path is active: sparse decode prefix + "
                "dense chunked-prefill suffix"
            )
            self._sparse_decode_forward_batched(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
                cfg,
                num_decode_reqs=num_decodes,
                num_decode_tokens=num_decode_tokens,
            )
            return self._dense_flash_forward(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
                req_start=num_decodes,
                tok_start=num_decode_tokens,
            )

        if not use_sparse:
            if cfg.enable_offload:
                # Dense attention reads the paged cache directly; any cold
                # (zeroed) block visible to this batch must be restored first.
                self._restore_cold_blocks_for_dense(layer, kv_cache, attn_metadata)
            return self._dense_flash_forward(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        # GPU-only long-context decode uses the batched sparse fast path.
        # K+V offload keeps the per-request loop (hybrid gather / cold-page
        # bookkeeping is still serial in the first release).
        if cfg.enable_offload:
            return self._sparse_decode_forward(
                layer, query, kv_cache, attn_metadata, output, cfg
            )
        return self._sparse_decode_forward_batched(
            layer, query, kv_cache, attn_metadata, output, cfg
        )

    def _dense_flash_forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ZoomKVMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        *,
        req_start: int = 0,
        tok_start: int = 0,
    ) -> torch.Tensor:
        """Dense prefill/short decode via FlashAttention on Triton KV pages.

        Keeps ZoomKV's Triton ``(num_blocks, 2, block_size, Hkv, D)`` cache
        layout so sparse gather and block_summary stay unchanged. K/V are
        split with ``_split_kv_cache`` and passed directly to FA's paged API.

        Optional ``req_start`` / ``tok_start`` select the prefill suffix of a
        reordered mixed batch (TurboQuant / FlashInfer style).
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for ZoomKV dense FA"
            )
        if attn_metadata is None:
            return output.fill_(0)
        if getattr(attn_metadata, "use_cascade", False):
            raise NotImplementedError(
                "Cascade attention is not supported by the ZoomKV dense FA path"
            )

        fa = self._fa
        assert fa.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )
        num_reqs = int(attn_metadata.num_reqs)
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        if req_start < 0 or req_start > num_reqs:
            raise ValueError(f"invalid dense req_start={req_start} num_reqs={num_reqs}")
        if tok_start < 0 or tok_start > num_actual_tokens:
            raise ValueError(
                f"invalid dense tok_start={tok_start} "
                f"num_actual_tokens={num_actual_tokens}"
            )
        region_reqs = num_reqs - req_start
        region_tokens = num_actual_tokens - tok_start
        if region_reqs <= 0 or region_tokens <= 0:
            return output

        key_cache, value_cache = self._split_kv_cache(kv_cache)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "ZoomKV dense FlashAttention path does not support quantized KV cache"
            )

        if req_start == 0 and tok_start == 0:
            query_start_loc = attn_metadata.query_start_loc
            seq_lens = attn_metadata.seq_lens
            block_table = attn_metadata.block_table
            max_query_len = int(attn_metadata.max_query_len)
            max_seq_len = int(attn_metadata.max_seq_len)
            scheduler_metadata = getattr(attn_metadata, "scheduler_metadata", None)
            num_splits = getattr(attn_metadata, "max_num_splits", 0)
        else:
            # Prefill suffix of a mixed batch. Rebase query_start_loc and use
            # prefill-local max lengths so FA fast paths stay correct.
            query_start_loc = (
                attn_metadata.query_start_loc[req_start : num_reqs + 1] - tok_start
            )
            seq_lens = attn_metadata.seq_lens[req_start:num_reqs]
            block_table = attn_metadata.block_table[req_start:num_reqs]
            qsl_cpu = attn_metadata.query_start_loc_cpu
            if qsl_cpu is None:
                qsl_cpu = attn_metadata.query_start_loc
            q_lens = qsl_cpu[req_start + 1 : num_reqs + 1] - qsl_cpu[req_start:num_reqs]
            max_query_len = int(q_lens.max().item()) if q_lens.numel() else 1
            seq_cpu = attn_metadata.seq_lens_cpu
            if seq_cpu is None:
                # Avoid GPU sync in the common path; fall back only if needed.
                seq_cpu = attn_metadata.seq_lens
            region_seq = seq_cpu[req_start:num_reqs]
            max_seq_len = (
                int(max(region_seq.tolist())) if region_seq.numel() else max_query_len
            )
            # Full-batch scheduler_metadata is invalid for a sliced call.
            scheduler_metadata = None
            num_splits = 0

        sliding_window_size = (
            list(fa.sliding_window) if fa.sliding_window is not None else None
        )
        descale_shape = (region_reqs, self.num_kv_heads)
        q_descale = (
            layer._q_scale.expand(descale_shape)
            if fa.supports_quant_query_input
            else None
        )
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)
        flash_attn_varlen_func(
            q=query[tok_start : tok_start + region_tokens],
            k=key_cache,
            v=value_cache,
            out=output[tok_start : tok_start + region_tokens],
            cu_seqlens_q=query_start_loc,
            max_seqlen_q=max_query_len,
            seqused_k=seq_lens,
            max_seqlen_k=max_seq_len,
            softmax_scale=self.scale,
            causal=getattr(attn_metadata, "causal", True),
            alibi_slopes=fa.alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table,
            softcap=fa.logits_soft_cap,
            scheduler_metadata=scheduler_metadata,
            fa_version=fa.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=num_splits,
            s_aux=fa.sinks,
        )
        return output

    @staticmethod
    def _block_table_cpu(attn_metadata: ZoomKVMetadata) -> torch.Tensor:
        """Host copy of the batch block table, fetched once per step.

        The metadata object is shared by every layer in the step, so caching
        the transfer on it turns per-layer GPU->CPU syncs into a single one.
        """
        bt_cpu = getattr(attn_metadata, "_zoomkv_block_table_cpu", None)
        if bt_cpu is None:
            bt_cpu = attn_metadata.block_table[: attn_metadata.num_reqs].cpu()
            attn_metadata._zoomkv_block_table_cpu = bt_cpu
        return bt_cpu

    def _restore_cold_blocks_for_dense(
        self,
        layer: AttentionLayer,
        kv_cache: torch.Tensor,
        attn_metadata: ZoomKVMetadata,
    ) -> None:
        """H2D-restore cold blocks in the batch's visible KV range.

        Runs before any dense read of the paged cache (prefill steps, mixed
        batches, dense decode). This is what makes prefix caching safe with
        offload: a cache-hit prefill sees fully materialized GPU pages.
        """
        from vllm.v1.attention.ops.zoomkv.offload import get_cpu_key_pool

        pool = get_cpu_key_pool()
        if pool is None or kv_cache.numel() == 0:
            return
        layer_name = str(
            self._layer_name
            or getattr(layer, "layer_name", None)
            or f"zoomkv_{id(layer)}"
        )
        if not pool.has_cold_blocks(layer_name):
            return
        # The visible physical block set is identical for every layer; compute
        # it once per step and cache it on the metadata object.
        block_ids = getattr(attn_metadata, "_zoomkv_batch_block_ids", None)
        if block_ids is None:
            seq_lens = (
                attn_metadata.seq_lens_cpu
                if attn_metadata.seq_lens_cpu is not None
                else attn_metadata.seq_lens.cpu()
            )[: attn_metadata.num_reqs]
            bt_cpu = self._block_table_cpu(attn_metadata)
            visible: set[int] = set()
            for req_i, seq_len in enumerate(seq_lens.tolist()):
                n_blocks = (int(seq_len) + self.block_size - 1) // self.block_size
                visible.update(bt_cpu[req_i, :n_blocks].tolist())
            visible.discard(-1)
            block_ids = list(visible)
            attn_metadata._zoomkv_batch_block_ids = block_ids
        if not block_ids:
            return
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        restored = pool.restore_blocks(layer_name, key_cache, value_cache, block_ids)
        if restored:
            logger.debug(
                "ZoomKV restored %d cold blocks for dense read (layer=%s)",
                restored,
                layer_name,
            )

    def _should_sparse_decode(
        self, attn_metadata: ZoomKVMetadata, cfg: ZoomKVRuntimeConfig
    ) -> bool:
        return _should_use_sparse_decode(
            cfg=cfg,
            max_query_len=attn_metadata.max_query_len,
            num_decodes=attn_metadata.num_decodes,
            num_prefills=attn_metadata.num_prefills,
            num_reqs=attn_metadata.num_reqs,
            seq_lens_cpu=attn_metadata.seq_lens_cpu,
            seq_lens=attn_metadata.seq_lens,
        )

    def _sparse_decode_forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ZoomKVMetadata,
        output: torch.Tensor,
        cfg: ZoomKVRuntimeConfig,
    ) -> torch.Tensor:
        with _zt.Stage("sparse.setup"):
            path = "K+V offload" if cfg.enable_offload else "GPU-only"
            logger.info_once("ZoomKV %s sparse decode path is active", path)
            layer_name = self._layer_name or getattr(
                layer, "layer_name", f"zoomkv_{id(layer)}"
            )
            num_blocks = kv_cache.shape[0]
            key_cache, value_cache = self._split_kv_cache(kv_cache)
            dtype = query.dtype
            block_summary = get_or_create_block_summary(
                layer_name=str(layer_name),
                num_blocks=num_blocks,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_size,
                block_size=self.block_size,
                device=kv_cache.device,
                dtype=dtype
                if dtype in (torch.float16, torch.bfloat16)
                else torch.bfloat16,
                blocks_per_parent=max(1, cfg.quest_large_chunk // cfg.quest_chunk),
            )
            retriever = self._get_retriever(cfg)
            q_start = (
                attn_metadata.query_start_loc_cpu
                if attn_metadata.query_start_loc_cpu is not None
                else attn_metadata.query_start_loc
            )
            seq_lens = (
                attn_metadata.seq_lens_cpu
                if attn_metadata.seq_lens_cpu is not None
                else attn_metadata.seq_lens
            )
            block_table = attn_metadata.block_table

            cpu_pool = None
            if cfg.enable_offload:
                from vllm.v1.attention.ops.zoomkv.offload import get_cpu_key_pool

                cpu_pool = get_cpu_key_pool()

        topk_buf = attn_metadata.topk_indices_buffer
        ctx_buf = attn_metadata.context_indices_buffer

        # Pull per-request scalar geometry to the host once. query_start_loc_cpu
        # and seq_lens_cpu are already host tensors, so this replaces three
        # GPU/CPU round-trips per request on every full-attention layer with a
        # single transfer per decode step.
        q_start_list = q_start.tolist()
        seq_lens_list = seq_lens.tolist()

        for req_i in range(attn_metadata.num_reqs):
            q0 = int(q_start_list[req_i])
            q1 = int(q_start_list[req_i + 1])
            if q1 - q0 != 1:
                continue
            seq_len = int(seq_lens_list[req_i])
            q = query[q0:q1]  # [1, Hq, D]
            with _zt.Stage("sparse.prep_q"):
                raw_q = prepare_retrieval_query(q, self.num_kv_heads)

            start_b, end_b = retriever.retrieval_block_range(seq_len, self.block_size)
            bt = block_table[req_i]

            if end_b > start_b:
                phys_ids = bt[start_b:end_b]
            else:
                phys_ids = torch.empty(0, dtype=torch.int64, device=q.device)

            # Host copy of this request's visible block ids (fetched once per
            # step for the whole batch, shared across layers).
            n_blocks_total = (seq_len + self.block_size - 1) // self.block_size
            bt_row_cpu = self._block_table_cpu(attn_metadata)[req_i]
            full_ids = bt_row_cpu[:n_blocks_total].tolist()
            # Content-anchored summary-cache key: the batch index alone is
            # not a request identity (requests reorder in the persistent
            # batch, and prefix-cache hits skip the big prefill that used to
            # flush the cache), so anchor on the physical ids as well.
            cache_key = (
                req_i,
                start_b,
                end_b,
                full_ids[start_b] if start_b < len(full_ids) else -1,
                full_ids[end_b - 1] if 0 < end_b <= len(full_ids) else -1,
            )

            # Offload bookkeeping (host-side, no extra GPU sync): map the
            # full visible range to CPU slots once, transition warm
            # retrieval-zone blocks to cold (pure GPU zeroing), and reuse the
            # slot tensor for both retrieval and the hybrid gather below.
            slots_full = None
            retrieval_has_slots = False
            if cpu_pool is not None and cfg.enable_offload:
                # Only retrieval-zone blocks may go cold: sink/local blocks
                # of *this* request stay warm, and cross-request sharing of
                # cold blocks is safe because the hybrid gather below covers
                # the whole visible range, not just the retrieval zone.
                cpu_pool.mark_cold(
                    str(layer_name),
                    key_cache,
                    value_cache,
                    full_ids[start_b:end_b],
                )
                slots_full, slots_list = cpu_pool.slots_from_block_ids(
                    str(layer_name), full_ids
                )
                retrieval_has_slots = any(s >= 0 for s in slots_list[start_b:end_b])

            # Prefer CPU-slot block_summaries when Keys have been offloaded.
            with _zt.Stage("sparse.retrieve"):
                if cpu_pool is not None and cfg.enable_offload and end_b > start_b:
                    assert slots_full is not None
                    cpu_slots = slots_full[start_b:end_b]
                    if retrieval_has_slots:
                        packed, cmin, cmax, centroid, valid = (
                            cpu_pool.gather_block_summaries_by_physical_ids(
                                str(layer_name), phys_ids
                            )
                        )
                        # Fall back to GPU block_summaries for blocks not yet offloaded.
                        gpu_packed, gpu_cmin, gpu_cmax, gpu_cent, gpu_valid = (
                            block_summary.gather_request_block_summaries(phys_ids)
                        )
                        on_cpu = (cpu_slots >= 0).to(device=q.device)
                        packed = torch.where(
                            on_cpu.view(1, 1, -1, 1, 1), packed, gpu_packed
                        )
                        cmin = torch.where(on_cpu.view(1, 1, -1, 1), cmin, gpu_cmin)
                        cmax = torch.where(on_cpu.view(1, 1, -1, 1), cmax, gpu_cmax)
                        centroid = torch.where(
                            on_cpu.view(1, 1, -1, 1), centroid, gpu_cent
                        )
                        valid = torch.where(on_cpu.view(1, 1, -1), valid, gpu_valid)
                        topk = retriever.retrieve_topk_from_block_summaries(
                            raw_q,
                            packed,
                            cmin,
                            cmax,
                            centroid,
                            valid,
                            seq_len,
                            self.block_size,
                            start_b,
                        )
                    else:
                        topk = retriever.retrieve_topk_tokens(
                            raw_q,
                            block_summary,
                            phys_ids,
                            seq_len,
                            cache_key=cache_key,
                        )
                else:
                    topk = retriever.retrieve_topk_tokens(
                        raw_q,
                        block_summary,
                        phys_ids,
                        seq_len,
                        cache_key=cache_key,
                    )

            # Materialize into the preallocated MLA-style buffer when present.
            topk_logical = topk[0]
            if _zoomkv_recall.enabled() and not cfg.enable_offload:
                # Debug-only: exact-attention recall of the retrieved Top-K.
                # Requires GPU-resident Keys, so offload mode is excluded.
                probe = _zoomkv_recall.get_probe()
                if probe is not None:
                    probe.record(
                        layer_name=str(layer_name),
                        req_idx=req_i,
                        query=q,
                        key_cache=key_cache,
                        block_table_row=bt,
                        block_size=self.block_size,
                        seq_len=seq_len,
                        start_block=start_b,
                        end_block=end_b,
                        topk_logical=topk_logical,
                        scale=self.scale,
                        retrieval_query=raw_q,
                    )
            if topk_buf is not None and req_i < topk_buf.shape[0]:
                dst = topk_buf[req_i]
                n = min(dst.shape[-1], topk_logical.shape[-1])
                dst.fill_(-1)
                dst[:, :n].copy_(topk_logical[:, :n])
                topk_logical = dst

            with _zt.Stage("sparse.assemble"):
                ctx_idx, _ctx_valid = assemble_sparse_context_indices(
                    seq_len,
                    topk_logical,
                    cfg.sink_size,
                    cfg.local_size,
                    device=q.device,
                    out=ctx_buf[req_i]
                    if ctx_buf is not None and req_i < ctx_buf.shape[0]
                    else None,
                )
            with _zt.Stage("sparse.gather"):
                if cfg.enable_offload and cpu_pool is not None:
                    # Cover the full visible range, not just the retrieval
                    # zone: with prefix sharing a block that is cold for one
                    # request may fall inside another request's sink/local
                    # window, and those tokens must also come from the CPU
                    # copy once the GPU page is zeroed.
                    gk, gv = gather_kv_hybrid(
                        key_cache,
                        value_cache,
                        bt,
                        ctx_idx,
                        self.block_size,
                        cpu_pool,
                        str(layer_name),
                        0,
                        n_blocks_total,
                        cpu_slots=slots_full,
                        any_offloaded=any(s >= 0 for s in slots_list),
                    )
                else:
                    gk, gv = gather_kv_by_logical_indices(
                        key_cache, value_cache, bt, ctx_idx, self.block_size
                    )
            with _zt.Stage("sparse.attn"):
                # The serial/materialized compatibility path has no host-side
                # validity guarantee. Keep its safe mask fallback; only the
                # explicit batched direct result may skip this synchronization.
                valid_mask = None if bool(_ctx_valid.all()) else _ctx_valid
                out = sparse_decode_attention(
                    q,
                    gk,
                    gv,
                    self.scale,
                    valid_mask=valid_mask,
                )
            output[q0:q1].copy_(out)
        return output

    def _sparse_decode_forward_batched(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ZoomKVMetadata,
        output: torch.Tensor,
        cfg: ZoomKVRuntimeConfig,
        *,
        num_decode_reqs: int | None = None,
        num_decode_tokens: int | None = None,
    ) -> torch.Tensor:
        """GPU-only batched sparse decode: one retrieve/gather/attn per layer.

        Pure decode and mixed-batch decode prefixes both use this path. Mixed
        callers pass ``num_decode_reqs`` / ``num_decode_tokens`` so only the
        reordered decode prefix is processed; the prefill suffix is handled by
        dense FA. CUDA Graph capture remains pure one-token decode only.
        """
        with _zt.Stage("sparse.setup"):
            logger.info_once(
                "ZoomKV GPU-only batched sparse decode path is active"
            )
            layer_name = self._layer_name or getattr(
                layer, "layer_name", f"zoomkv_{id(layer)}"
            )
            num_blocks = kv_cache.shape[0]
            key_cache, value_cache = self._split_kv_cache(kv_cache)
            dtype = query.dtype
            block_summary = get_or_create_block_summary(
                layer_name=str(layer_name),
                num_blocks=num_blocks,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_size,
                block_size=self.block_size,
                device=kv_cache.device,
                dtype=dtype
                if dtype in (torch.float16, torch.bfloat16)
                else torch.bfloat16,
                blocks_per_parent=max(1, cfg.quest_large_chunk // cfg.quest_chunk),
            )
            retriever = self._get_retriever(cfg)
            block_table = attn_metadata.block_table
            num_reqs = (
                int(attn_metadata.num_reqs)
                if num_decode_reqs is None
                else int(num_decode_reqs)
            )
            if num_reqs <= 0:
                return output
            tok_end = (
                num_reqs
                if num_decode_tokens is None
                else int(num_decode_tokens)
            )
            graph_capture = bool(
                getattr(attn_metadata, "is_cudagraph_capture", False)
            )
            if graph_capture and (
                num_decode_reqs is not None or num_decode_tokens is not None
            ):
                raise RuntimeError(
                    "ZoomKV CUDA Graph sparse path does not support mixed-batch slices"
                )

        if graph_capture:
            # Full-graph capture is restricted to pure one-token decode, whose
            # padded token layout is statically [0..B). Do not inspect CPU
            # metadata or synchronize a device tensor while capturing.
            q_start_list = None
            contiguous = True
            q_batch = query[:num_reqs]
        else:
            q_start = (
                attn_metadata.query_start_loc_cpu
                if attn_metadata.query_start_loc_cpu is not None
                else attn_metadata.query_start_loc
            )
            q_start_list = q_start.tolist()
            contiguous = (
                all(int(q_start_list[i]) == i for i in range(num_reqs))
                and int(q_start_list[num_reqs]) == tok_end
            )
            if contiguous:
                q_batch = query[:num_reqs]
            else:
                starts = [int(q_start_list[i]) for i in range(num_reqs)]
                q_batch = query[starts]

        with _zt.Stage("sparse.prep_q"):
            raw_q = prepare_retrieval_query(q_batch, self.num_kv_heads)

        seq_lens_t = attn_metadata.seq_lens[:num_reqs]
        seq_lens_host = (
            None
            if graph_capture or attn_metadata.seq_lens_cpu is None
            else attn_metadata.seq_lens_cpu[:num_reqs]
        )
        topk_buf = attn_metadata.topk_indices_buffer
        topk_out = (
            topk_buf[:num_reqs]
            if topk_buf is not None and topk_buf.shape[0] >= num_reqs
            else None
        )
        with _zt.Stage("sparse.retrieve"):
            retrieval = retriever.retrieve_topk_tokens_batch_result(
                raw_q,
                block_summary,
                block_table[:num_reqs],
                seq_lens_t,
                # Production sparse decode only exposes fully-completed
                # retrieval blocks. Summary lifecycle finalizes those blocks
                # before they enter the retrieval zone and preserves state on
                # CoW remaps; invalid-summary tests deliberately omit this.
                summaries_guaranteed_valid=True,
                topk_out=topk_out,
                assume_context_fully_valid=graph_capture,
                chunk_bucket=(
                    attn_metadata.graph_chunk_bucket
                    if graph_capture
                    else (
                        _graph_chunk_bucket(cfg, self.block_size)
                        if num_decode_reqs is not None
                        else None
                    )
                ),
                seq_lens_host=seq_lens_host,
                use_cudagraph=num_decode_reqs is not None,
            )
        if graph_capture and not retrieval.used_direct_physical:
            raise RuntimeError(
                "ZoomKV full CUDA Graph requires direct physical retrieval"
            )

        topk_logical = retrieval.topk
        fully_valid = graph_capture or retrieval.context_fully_valid
        if _zoomkv_recall.enabled() and not graph_capture:
            # Debug-only exact recall probe for the batched direct path. The
            # probe intentionally uses host sequence lengths and per-request
            # exact QK, so it remains outside production/full-graph execution.
            probe = _zoomkv_recall.get_probe()
            seq_lens_host = attn_metadata.seq_lens_cpu
            if probe is not None and seq_lens_host is not None:
                for req_i in range(num_reqs):
                    seq_len = int(seq_lens_host[req_i])
                    start_b, end_b = retriever.retrieval_block_range(
                        seq_len, self.block_size
                    )
                    probe.record(
                        layer_name=str(layer_name),
                        req_idx=req_i,
                        query=q_batch[req_i : req_i + 1],
                        key_cache=key_cache,
                        block_table_row=block_table[req_i],
                        block_size=self.block_size,
                        seq_len=seq_len,
                        start_block=start_b,
                        end_block=end_b,
                        topk_logical=topk_logical[req_i],
                        scale=self.scale,
                        retrieval_query=raw_q[req_i : req_i + 1],
                    )

        # Device seq_lens for Triton fused gather / assemble kernels.
        seq_lens_dev = attn_metadata.seq_lens[:num_reqs]
        if fully_valid:
            # Direct fully-valid path: fuse sink/local/topk assembly into the
            # paged gather and skip materializing ctx_idx / _ctx_valid.
            with _zt.Stage("sparse.gather"):
                gk, gv = gather_kv_from_topk_batch(
                    key_cache,
                    value_cache,
                    block_table[:num_reqs],
                    seq_lens_dev,
                    topk_logical,
                    self.block_size,
                    cfg.sink_size,
                    cfg.local_size,
                    output_bthd=True,
                    validate_mapping=(
                        num_decode_reqs is not None
                        and os.environ.get(
                            "VLLM_ZOOMKV_VALIDATE_GATHER", "0"
                        )
                        == "1"
                    ),
                )
            valid_mask = None
            kv_layout_bthd = True
        else:
            ctx_buf = attn_metadata.context_indices_buffer
            ctx_out = (
                ctx_buf[:num_reqs]
                if ctx_buf is not None and ctx_buf.shape[0] >= num_reqs
                else None
            )
            with _zt.Stage("sparse.assemble"):
                ctx_idx, _ctx_valid = assemble_sparse_context_indices_batch(
                    seq_lens_dev,
                    topk_logical,
                    cfg.sink_size,
                    cfg.local_size,
                    out=ctx_out,
                )
            with _zt.Stage("sparse.gather"):
                gk, gv = gather_kv_by_logical_indices_batch(
                    key_cache,
                    value_cache,
                    block_table[:num_reqs],
                    ctx_idx,
                    self.block_size,
                )
            # Safe fallback for short / padded contexts: one host sync.
            valid_mask = None if bool(_ctx_valid.all()) else _ctx_valid
            kv_layout_bthd = False

        attn_out = output[:num_reqs] if contiguous else None
        with _zt.Stage("sparse.attn"):
            out = sparse_decode_attention_batch(
                q_batch,
                gk,
                gv,
                self.scale,
                valid_mask=valid_mask,
                kv_layout_bthd=kv_layout_bthd,
                out=attn_out,
            )

        if contiguous:
            # The fully-valid FlashAttention path writes directly into the
            # model output. Reference fallbacks return a separate tensor.
            if out is not attn_out:
                output[:num_reqs].copy_(out)
        else:
            assert q_start_list is not None
            for i in range(num_reqs):
                q0 = int(q_start_list[i])
                output[q0 : q0 + 1].copy_(out[i : i + 1])
        return output
