# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZoomKV V1 attention backend.

Dense prefill / short-context decode reuse Triton paged attention.
Long-context single-token decode runs hierarchical Quest + KIVI retrieval
over physical-block block_summaries, then non-causal attention over
sink + local + Top-K tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.zoomkv import recall_probe as _zoomkv_recall
from vllm.v1.attention.ops.zoomkv import stage_timer as _zt
from vllm.v1.attention.ops.zoomkv.paged import (
    assemble_sparse_context_indices,
    gather_kv_by_logical_indices,
    gather_kv_hybrid,
    sparse_decode_attention,
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
        final_topk=getattr(attn, "zoomkv_final_topk", 100),
        quest_chunk=getattr(attn, "zoomkv_quest_chunk", 16),
        quest_large_chunk=getattr(attn, "zoomkv_quest_large_chunk", 256),
        quest_large_ratio=getattr(attn, "zoomkv_quest_large_ratio", 0.8),
        quest_small_ratio=getattr(attn, "zoomkv_quest_small_ratio", 0.5),
        dense_ratio=getattr(attn, "zoomkv_dense_ratio", 0.4),
        dense_topk=getattr(attn, "zoomkv_dense_topk", 16),
        sparse_topk=getattr(attn, "zoomkv_sparse_topk", 8),
        full_attention_threshold=getattr(attn, "zoomkv_full_attention_threshold", 2000),
        dense_fallback=getattr(attn, "zoomkv_dense_fallback", False),
        strict_kernels=strict,
        enable_offload=getattr(attn, "zoomkv_enable_offload", False),
        per_query_head=getattr(attn, "zoomkv_per_query_head", False),
    )


@dataclass
class ZoomKVMetadata(TritonAttentionMetadata):
    num_reqs: int = 0
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    zoomkv: ZoomKVRuntimeConfig | None = None
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    # Preallocated physical Top-K / context index buffers (MLA-style).
    # Shape: [max_num_seqs, num_kv_heads, final_topk]
    topk_indices_buffer: torch.Tensor | None = None
    # Shape: [max_num_seqs, num_kv_heads, sink+local+final_topk]
    context_indices_buffer: torch.Tensor | None = None


class ZoomKVMetadataBuilder(TritonAttentionMetadataBuilder):
    """Build Triton-compatible metadata plus ZoomKV knobs.

    CUDA Graphs are disabled for ZoomKV (eager-only first release).
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

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

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ZoomKVMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=1,
        )
        # Reconstruct from base fields that exist on this vLLM version.
        fields = {
            "num_actual_tokens": base.num_actual_tokens,
            "max_query_len": base.max_query_len,
            "query_start_loc": base.query_start_loc,
            "max_seq_len": base.max_seq_len,
            "seq_lens": base.seq_lens,
            "block_table": base.block_table,
            "slot_mapping": base.slot_mapping,
            "seq_threshold_3D": base.seq_threshold_3D,
            "num_par_softmax_segments": base.num_par_softmax_segments,
            "softmax_segm_output": base.softmax_segm_output,
            "softmax_segm_max": base.softmax_segm_max,
            "softmax_segm_expsum": base.softmax_segm_expsum,
            "use_cascade": base.use_cascade,
            "common_prefix_len": base.common_prefix_len,
            "cu_prefix_query_lens": base.cu_prefix_query_lens,
            "prefix_kv_lens": base.prefix_kv_lens,
            "suffix_kv_lens": base.suffix_kv_lens,
            "scheduler_metadata": base.scheduler_metadata,
            "prefix_scheduler_metadata": base.prefix_scheduler_metadata,
            "mm_prefix_range": base.mm_prefix_range,
            "mm_prefix_range_tensor": base.mm_prefix_range_tensor,
            "num_reqs": common_attn_metadata.num_reqs,
            "num_decodes": num_decodes,
            "num_prefills": num_prefills,
            "num_decode_tokens": num_decode_tokens,
            "zoomkv": self.zoomkv,
            # Keep host copies in metadata so every full-attention layer does
            # not synchronize the GPU merely to recover scalar sequence
            # geometry.
            "query_start_loc_cpu": common_attn_metadata.query_start_loc_cpu,
            "seq_lens_cpu": common_attn_metadata.seq_lens_cpu,
            "topk_indices_buffer": self.topk_indices_buffer,
            "context_indices_buffer": self.context_indices_buffer,
        }
        return ZoomKVMetadata(**fields)


class ZoomKVAttentionBackend(AttentionBackend):
    """Native ZoomKV sparse-retrieval attention backend."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

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
        self._dense = TritonAttentionImpl(
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
            use_alibi_sqrt=use_alibi_sqrt,
            chunk_lookback=chunk_lookback,
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
        # The decode fast path launches one conditional Triton kernel and never
        # evaluates a CUDA predicate on the host. Prefill retains the batched
        # PyTorch path for throughput.
        with _zt.Stage("block_summary.update"):
            block_summary.update_completed_slots(key_cache, slots)

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

        cfg = attn_metadata.zoomkv or ZoomKVRuntimeConfig()
        use_sparse = self._should_sparse_decode(attn_metadata, cfg)
        if not use_sparse:
            if cfg.enable_offload:
                # Dense attention reads the paged cache directly; any cold
                # (zeroed) block visible to this batch must be restored first.
                self._restore_cold_blocks_for_dense(layer, kv_cache, attn_metadata)
            return self._dense.forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        return self._sparse_decode_forward(
            layer, query, kv_cache, attn_metadata, output, cfg
        )

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
        if cfg.dense_fallback:
            return False
        # Speculative / multi-token decode → dense (first release).
        if attn_metadata.max_query_len != 1:
            return False
        if attn_metadata.num_decodes <= 0:
            return False
        if attn_metadata.num_prefills > 0:
            # Mixed batch: keep dense for correctness.
            return False
        # All decode requests must exceed the full-attention threshold.
        seq_lens = (
            attn_metadata.seq_lens_cpu
            if attn_metadata.seq_lens_cpu is not None
            else attn_metadata.seq_lens
        )[: attn_metadata.num_reqs]
        if seq_lens.numel() == 0:
            return False
        retriever = self._get_retriever(cfg)
        return bool(
            all(not retriever.should_use_dense(int(s)) for s in seq_lens.tolist())
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
                raw_q = prepare_retrieval_query(
                    q, self.num_kv_heads, per_query_head=cfg.per_query_head
                )

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
                # Invalid context slots (topk padding = -1, or overlap gaps)
                # are gathered as zeroed K/V but would still take softmax
                # weight and skew the output -- this bites short / near-
                # threshold sequences whose retrieval zone yields fewer than
                # final_topk tokens. Mask them out. The fast unmasked
                # FlashAttention path is kept whenever every slot is valid
                # (the common long-context case). This is an eager-only
                # backend, so the scalar validity check is acceptable; the
                # future batched CUDA-graph path must mask in-kernel instead.
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
