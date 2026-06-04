# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN attention backend.

KV-cache compression by Hadamard rotation + iterative variance-normalization
(Sinkhorn-like) + asymmetric RTN. K is quantized per-channel, V per-token —
KIVI orientation. The variance-normalization tile equals the vLLM
``block_size`` (default and only supported value in this PR: ``128``).

Cache layout (per block, per kv-head, ``head_dim=128, k_bits=4, v_bits=2``):
  13824 B = 8192 (K packed) + 256 + 256 + 256  (K absorbed scales + zp + s_row)
          + 4096 (V packed) + 256 + 512 + 256  (V s_col + absorbed s_row + zp)

vLLM-shape reinterpretation: ``(num_blocks, block_size=128, num_kv_heads, 140)``
where ``140 = 17920 / 128``. The 128-slot middle dim has no semantic per-token
meaning — KVarN treats each ``kv_cache[block, :, head, :].view(-1)`` as one
flat 17920-byte tile record. The slot dim is preserved only to satisfy
vLLM's KV-cache allocator (which expects 4D for non-MLA layouts) and to keep
``slot_mapping`` arithmetic uniform with other backends.

Implementation outline:
  - `do_kv_cache_update` buffers incoming fp16 K/V in a per-block staging dict
    (keyed by block_id). When a block fills to 128 tokens, it rotates by
    Hadamard, calls `kvarn_store_tile_{k,v}` (Stage-3a validated), and writes
    the packed 17920-byte record into the cache.
  - `forward` has three branches: pure-prefill first chunk (raw K/V →
    flash_attn_varlen), pure-decode (dequant cached blocks + un-rotate, concat
    with fp16 tail buffers, run SDPA), mixed batch (split decode / prefill).
  - The decode path is intentionally slow PyTorch — Stage 4 replaces it with
    a Triton split-KV decode mirroring `triton_turboquant_decode.py`.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.kvarn_decode import (
    kvarn_dequant_tile_k,
    kvarn_dequant_tile_v,
)
from vllm.v1.attention.ops.kvarn_store import (
    kvarn_store_tile_k,
    kvarn_store_tile_k_batch_from_sinkhorn,
    kvarn_store_tile_v,
    kvarn_store_tile_v_batch_from_sinkhorn,
)
from vllm.v1.attention.ops.triton_kvarn_decode import kvarn_decode_attention
from vllm.v1.attention.ops.triton_kvarn_sinkhorn import kvarn_sinkhorn_triton

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


# ──────────────────────────────────────────────────────────────────────────────
# Hadamard cache (one D×D matrix per (head_dim, device))
# ──────────────────────────────────────────────────────────────────────────────


@functools.cache
def _hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    """Sylvester Hadamard, normalised, cached per (d, device)."""
    H = torch.ones(1, 1)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str)).float()


def _build_hadamard(d: int, device: torch.device) -> torch.Tensor:
    return _hadamard_cached(d, str(torch.device(device)))


# ──────────────────────────────────────────────────────────────────────────────
# Backend metadata classes
# ──────────────────────────────────────────────────────────────────────────────


class KVarNAttentionBackend(AttentionBackend):
    """Attention backend using KVarN KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "kvarn_k4v2_g128",
    ]

    @staticmethod
    def get_name() -> str:
        return "KVARN"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["KVarNAttentionImpl"]:
        return KVarNAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["KVarNMetadataBuilder"]:
        return KVarNMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "kvarn_k4v2_g128",
    ) -> tuple[int, ...]:
        """3D shape: one contiguous ``tile_bytes_aligned`` record per (block, head).

        Unlike TurboQuant's per-token slot, KVarN's scales are tile-shared,
        so one block per head is a single 17920-byte record. The natural
        shape is therefore ``(num_blocks, num_kv_heads, tile_bytes_aligned)``
        — no leading 2 (K and V share the record), and no per-position dim.

        The total bytes per block (= ``num_kv_heads * tile_bytes_aligned``)
        equals ``block_size * num_kv_heads * slot_size`` from
        ``TQFullAttentionSpec.page_size_bytes`` when ``slot_size = tile_bytes
        / block_size``, so vLLM's memory accounting works unchanged.
        """
        from vllm.model_executor.layers.quantization.kvarn.config import (
            KVarNConfig,
        )

        cfg = KVarNConfig.from_cache_dtype(cache_dtype_str, head_size)
        assert block_size == cfg.group, (
            f"KVarN requires block_size ({block_size}) == group ({cfg.group})."
        )
        return (num_blocks, num_kv_heads, cfg.tile_bytes_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("kvarn_")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == 128


@dataclass
class KVarNMetadata(AttentionMetadata):
    """Metadata for KVarN attention (mirrors ``TurboQuantMetadata``)."""

    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False
    num_decodes: int = 0
    num_decode_tokens: int = 0
    # Precomputed once per batch in the metadata builder and reused across all
    # 28+ layer forward calls. Saves 28× .tolist() syncs per decode token.
    seq_lens_cpu: list[int] | None = None
    block_table_cpu: list[list[int]] | None = None
    slot_mapping_cpu: list[int] | None = None
    # Stage α-2 capture-correct decode metadata. The block_table-driven
    # build-packed-KV kernel reads block_table / seq_lens / fa_cu_seqlens_k
    # directly (all PERSISTENT buffers updated in-place by the builder), so a
    # captured CUDA graph sees fresh data on every replay.
    fa_cu_seqlens_q: torch.Tensor | None = None       # [B+1] int32 (persistent)
    fa_cu_seqlens_k: torch.Tensor | None = None       # [B+1] int32 (persistent prefix sum of seq_lens)
    fa_max_blocks_per_req: int = 0                    # ceil(max_model_len / group): grid dim
    fa_max_seqlen_k_fixed: int = 0                    # = max_model_len; fixed FA grid bound


class KVarNMetadataBuilder(AttentionMetadataBuilder[KVarNMetadata]):
    """Builds ``KVarNMetadata`` from scheduler output."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )  # Stage α-2: the decode forward + do_kv_cache_update are now pure
       # tensor ops (Triton scatter + matmul + dequant/gather + flash_attn,
       # all on pre-allocated scratch). All Python state mutation (slot
       # allocation, sink marking, tile-boundary flush) happens in
       # KVarNMetadataBuilder.build() between captured graph replays.

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        # Stage α-2: per-request fill tracking for flush detection, keyed by
        # the sink block id so request identity survives batch reordering.
        #   _prev_seq_len_by_sink[sink]  = seq_len at the previous step
        #                                  (= tokens currently in the pool)
        #   _flush_watermark_by_sink[sink] = next block index to flush
        self._prev_seq_len_by_sink: dict[int, int] = {}
        self._flush_watermark_by_sink: dict[int, int] = {}

        # Max model length (for the fixed FA grid bound + max_blocks_per_req).
        try:
            self._max_model_len = vllm_config.model_config.max_model_len
        except Exception:
            self._max_model_len = 4096

        # Persistent cu_seqlens buffers (allocated lazily in build()).
        self._cu_seqlens_q_buf: torch.Tensor | None = None
        self._cu_seqlens_k_buf: torch.Tensor | None = None
        self._cu_seqlens_q_host: torch.Tensor | None = None
        self._cu_seqlens_k_host: torch.Tensor | None = None

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> KVarNMetadata:
        return self.build(0, common_attn_metadata)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam = common_attn_metadata
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )
        # Pre-materialise CPU views ONCE per batch. Every layer's forward()
        # would otherwise re-issue these syncs (28+ syncs/token for Qwen3-0.6B).
        seq_lens_cpu = cam.seq_lens.tolist()
        block_table_cpu = cam.block_table_tensor.tolist()
        slot_mapping_cpu = cam.slot_mapping.tolist()
        device = cam.seq_lens.device

        # ── Stage α-2: capture-correct metadata ──────────────────────────
        # The decode driver uses ONE block_table-driven kernel that reads the
        # PERSISTENT block_table / seq_lens / cu_seqlens directly, so no
        # per-step derived task tensors (which would be stale under graph
        # replay). We only need cu_seqlens_k (prefix sum of seq_lens) and
        # cu_seqlens_q (= arange(B+1)), both kept in PERSISTENT buffers and
        # updated in place so captured graphs see fresh values.
        B = len(seq_lens_cpu)
        GROUP = 128                                    # KVarN cfg.group; fixed
        cu_seqlens_k_h = [0]
        for sl in seq_lens_cpu:
            cu_seqlens_k_h.append(cu_seqlens_k_h[-1] + sl)

        # ── Stage α-2: assign pool slots for every block_id touched this
        # step. The allocator state is class-level on KVarNAttentionImpl
        # and we mutate it here (in the builder, outside any captured
        # region). do_kv_cache_update then only READS block_to_slot_t.
        from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionImpl  # local import
        # Pool slots are needed ONLY for blocks that physically live in the fp16
        # tail pool: each request's sink (block_table[r][0], kept fp16 forever)
        # and the in-progress tail block(s) currently being written — which are
        # exactly the blocks named by slot_mapping this step (one tail per
        # decoding request; the full set of touched blocks during a prefill
        # chunk). Flushed history blocks (1..n_full-1) live in the int4 cache,
        # carry pool_slot=-1, and are dequantized in-kernel.
        #
        # Do NOT allocate slots for "every block up to seq_len": those history
        # blocks would be (a) re-allocated every step but flushed only once,
        # leaking the pool until it drains (RuntimeError: pool exhausted at long
        # context), and (b) read from empty pool slots instead of int4 by the
        # build kernel. slot_mapping + sink is necessary and sufficient.
        # blocks_needed = the exact set of blocks that must hold a pool slot
        # right now: per active request its sink (row[0]) + its in-progress tail
        # (the block holding token seq_len-1, i.e. row[seq_len//GROUP]), plus
        # every block named by slot_mapping this step (prefill chunks, and a
        # safety superset of the tails). Anything in the allocator NOT in this
        # set belongs to a completed request and is reclaimed below — sinks are
        # otherwise never flushed, so without reclamation every finished
        # request leaks its sink (and partial-tail) slot until the pool drains.
        blocks_needed: set[int] = set()
        for b in range(B):
            row = block_table_cpu[b] if b < len(block_table_cpu) else []
            sl = seq_lens_cpu[b]
            if not row or sl <= 0:
                continue
            if row[0] >= 0:
                blocks_needed.add(row[0])          # sink (kept fp16 forever)
            tail_idx = sl // GROUP                  # in-progress tail block
            if tail_idx < len(row) and row[tail_idx] >= 0:
                blocks_needed.add(row[tail_idx])
        for s in slot_mapping_cpu:                 # in-progress tail(s) / prefill
            if s >= 0:
                blocks_needed.add(s // GROUP)

        if KVarNAttentionImpl._all_impls:
            impl0 = KVarNAttentionImpl._all_impls[0]
            # Ensure pool + lookup tensors exist for this device.
            impl0._ensure_pool(device,
                num_blocks_hint=max(blocks_needed, default=0) + 1)
            b2s_t = KVarNAttentionImpl._block_to_slot_t_per_device[device]
            is_sink_t = KVarNAttentionImpl._is_sink_t_per_device[device]
            dict_map = KVarNAttentionImpl._block_to_slot_dict
            free_slots = KVarNAttentionImpl._free_slots
            sinks = KVarNAttentionImpl._global_sink_blocks

            # ORDER MATTERS: mark sinks → FLUSH (frees just-completed blocks'
            # slots) → ALLOCATE (the new tails, reusing the freed slots). Doing
            # the flush before allocation caps the live-slot peak at 2·B
            # (one sink + one in-progress tail per request). Allocating first
            # would transiently need 3·B when every request crosses a block
            # boundary in lockstep (sink + pending-flush full block + new tail)
            # → "pool exhausted" at large batch.

            # (1) Mark per-request sink blocks (block_table[r][0]).
            for b in range(B):
                row = block_table_cpu[b] if b < len(block_table_cpu) else []
                if row and row[0] >= 0:
                    sb = row[0]
                    if sb not in sinks:
                        sinks.add(sb)
                        if sb < is_sink_t.shape[0]:
                            is_sink_t[sb] = True

            # (2) Flush detection (Stage α-2 Step B).
            # CRITICAL timing: token (k+1)*GROUP-1 (the one that completes
            # block k) is written during THIS step's do_kv_cache_update, which
            # runs AFTER the builder. So at builder time the pool only holds
            # tokens written through the PREVIOUS step. We therefore flush
            # against `prev_sl` (= pool token count now), never `sl`.
            #
            # _prev_seq_len_by_sink[sink] holds the seq_len reported at the
            # previous step = exactly the number of tokens now in the pool.
            # _flush_watermark_by_sink[sink] = next block index to flush.
            flush_block_ids: list[int] = []
            seen_sinks: set[int] = set()
            for b in range(B):
                row = block_table_cpu[b] if b < len(block_table_cpu) else []
                sl = seq_lens_cpu[b]
                if not row or row[0] < 0 or sl <= 0:
                    continue
                sink_bid = row[0]
                seen_sinks.add(sink_bid)
                prev_sl = self._prev_seq_len_by_sink.get(sink_bid, 0)
                complete_in_pool = prev_sl // GROUP    # blocks 0..that-1 fully in pool
                watermark = self._flush_watermark_by_sink.get(sink_bid, 1)  # skip sink (k=0)
                for k in range(watermark, complete_in_pool):
                    if k < len(row):
                        bid = row[k]
                        if bid >= 0 and bid not in sinks:
                            flush_block_ids.append(bid)
                if complete_in_pool > watermark:
                    self._flush_watermark_by_sink[sink_bid] = complete_in_pool
                self._prev_seq_len_by_sink[sink_bid] = sl
            # Drop tracking for requests no longer present (completed).
            for stale in [s for s in self._prev_seq_len_by_sink if s not in seen_sinks]:
                del self._prev_seq_len_by_sink[stale]
                self._flush_watermark_by_sink.pop(stale, None)

            # Trigger the flush on every layer's pool. Each impl quantises its
            # own pool[slot] into its own kv_cache (ref cached on first
            # forward), then frees the slot below. Runs eagerly here, before
            # the captured forward replay.
            if flush_block_ids:
                # One batched Sinkhorn + RTN over ALL (layer, block) flush tiles
                # — replaces 48×N_blocks individual launches. Numerically
                # identical (per-tile-independent ops) → no accuracy change.
                flush_pairs = []
                for impl in KVarNAttentionImpl._all_impls:
                    kvc = getattr(impl, "_kv_cache_ref", None)
                    if kvc is None:
                        continue
                    for bid in flush_block_ids:
                        flush_pairs.append((impl, bid, kvc))
                KVarNAttentionImpl._batched_flush(flush_pairs)
                # Free the flushed blocks' slots so the allocation below can
                # reuse them (they now live in int4; pool_slot → -1).
                for bid in flush_block_ids:
                    slot = dict_map.pop(bid, None)
                    if slot is not None:
                        free_slots.append(slot)
                        if bid < b2s_t.shape[0]:
                            b2s_t[bid] = -1

            # (2b) Reclaim stale slots from COMPLETED requests. Any block still
            # holding a slot but not needed this step is a finished request's
            # sink or partial tail (its data is dead — discard, do not flush).
            # Without this, sinks (never flushed) leak one slot per finished
            # request and the pool exhausts across requests / over serving.
            for bid in [b for b in dict_map if b not in blocks_needed]:
                slot = dict_map.pop(bid)
                free_slots.append(slot)
                if bid < b2s_t.shape[0]:
                    b2s_t[bid] = -1
                if bid in sinks:
                    sinks.discard(bid)
                    if bid < is_sink_t.shape[0]:
                        is_sink_t[bid] = False
                # Reset flush tracking so a recycled block_id starts fresh.
                self._prev_seq_len_by_sink.pop(bid, None)
                self._flush_watermark_by_sink.pop(bid, None)

            # (3) Allocate slots for any new block_ids (sinks + new tails).
            for bid in blocks_needed:
                if bid not in dict_map:
                    if not free_slots:
                        raise RuntimeError(
                            f"KVarN pool exhausted "
                            f"({KVarNAttentionImpl._allocator_pool_size} slots)"
                        )
                    slot = free_slots.pop()
                    dict_map[bid] = slot
                    if bid < b2s_t.shape[0]:
                        b2s_t[bid] = slot
                    KVarNAttentionImpl._max_known_block_id = max(
                        KVarNAttentionImpl._max_known_block_id, bid
                    )

        # ── Persistent cu_seqlens buffers (in-place updated) ─────────────
        # A captured graph bakes in tensor addresses, so cu_seqlens MUST live
        # in fixed buffers updated in place — not recreated each step.
        cap = B + 1
        if self._cu_seqlens_q_buf is None or self._cu_seqlens_q_buf.shape[0] < cap:
            new_cap = max(cap, 257)   # default max_num_seqs headroom
            self._cu_seqlens_q_buf = torch.empty(new_cap, dtype=torch.int32, device=device)
            self._cu_seqlens_k_buf = torch.empty(new_cap, dtype=torch.int32, device=device)
            self._cu_seqlens_q_host = torch.empty(new_cap, dtype=torch.int32, pin_memory=True)
            self._cu_seqlens_k_host = torch.empty(new_cap, dtype=torch.int32, pin_memory=True)
        for i in range(B + 1):
            self._cu_seqlens_q_host[i] = i
            self._cu_seqlens_k_host[i] = cu_seqlens_k_h[i]
        fa_cu_seqlens_q = self._cu_seqlens_q_buf[:B + 1]
        fa_cu_seqlens_k = self._cu_seqlens_k_buf[:B + 1]
        fa_cu_seqlens_q.copy_(self._cu_seqlens_q_host[:B + 1], non_blocking=True)
        fa_cu_seqlens_k.copy_(self._cu_seqlens_k_host[:B + 1], non_blocking=True)

        max_blocks_per_req = (self._max_model_len + GROUP - 1) // GROUP

        return KVarNMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            seq_lens_cpu=seq_lens_cpu,
            block_table_cpu=block_table_cpu,
            slot_mapping_cpu=slot_mapping_cpu,
            fa_cu_seqlens_q=fa_cu_seqlens_q,
            fa_cu_seqlens_k=fa_cu_seqlens_k,
            fa_max_blocks_per_req=max_blocks_per_req,
            fa_max_seqlen_k_fixed=self._max_model_len,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Per-block fp16 tail buffer (in-progress tile staging)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _BlockTail:
    """In-progress fp16 K/V for one cache block.

    Reset whenever a token with ``position_in_block == 0`` arrives for this
    block_id (handles vLLM's block recycling on request preemption). Evicted
    immediately after a 128-token flush.
    """

    K: torch.Tensor  # [group, num_kv_heads, head_dim] fp16
    V: torch.Tensor  # [group, num_kv_heads, head_dim] fp16
    filled_mask: torch.Tensor = field(repr=False)  # [group] bool — which slots written
    filled_count: int = 0                          # CPU-side counter (avoid .all() sync)


# ──────────────────────────────────────────────────────────────────────────────
# Attention impl
# ──────────────────────────────────────────────────────────────────────────────


class KVarNAttentionImpl(AttentionImpl["KVarNMetadata"]):
    """KVarN attention implementation.

    Slow PyTorch decode for Stage 3b.2 — replaced by Triton in Stage 4.
    """

    supports_quant_query_input: bool = False

    # Shared decode scratch — these are per-step throwaway buffers used by
    # `kvarn_decode_attention`. Sharing across all impl instances (one set
    # per device) avoids 28× memory waste on the per-layer attention.
    # Lazily allocated by `_ensure_pool` on the first non-capture call.
    _shared_q_fp32_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_q_rot_fp32_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_q_rot_fp16_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_out_rot_fp32_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_output_fp32_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_fused_out_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_mid_o_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}    # split-K partials
    _shared_mid_lse_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_fa_K_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _shared_fa_V_buf: ClassVar[dict[torch.device, torch.Tensor]] = {}

    # ── Stage α-2: class-level shared sparse slot allocator ──────────────────
    # Single source of truth across all 28 KVarNAttentionImpl instances:
    #   _block_to_slot_dict[block_id] → slot      (Python, CPU)
    #   _block_to_slot_t_per_device[device][block_id] → slot int32    (GPU mirror)
    #   _is_sink_t_per_device[device][block_id] → bool                (GPU mirror)
    # All allocator mutations happen in KVarNMetadataBuilder.build(), which
    # runs once per step OUTSIDE any captured CUDA-graph region. The captured
    # do_kv_cache_update kernel just reads block_to_slot_t.
    _block_to_slot_dict: ClassVar[dict[int, int]] = {}
    _global_sink_blocks: ClassVar[set[int]] = set()
    _free_slots: ClassVar[list[int] | None] = None
    _allocator_pool_size: ClassVar[int] = 0
    _block_to_slot_t_per_device: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _is_sink_t_per_device: ClassVar[dict[torch.device, torch.Tensor]] = {}
    _max_known_block_id: ClassVar[int] = 0

    # Registry of impls so the builder can enumerate per-layer pools when
    # it needs to update sink markers / trigger flushes.
    _all_impls: ClassVar[list["KVarNAttentionImpl"]] = []

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
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        from vllm.model_executor.layers.quantization.kvarn.config import (
            KVarNConfig,
        )

        self.kvarn_config = KVarNConfig.from_cache_dtype(kv_cache_dtype, head_size)

        # Per-block fp16 tail buffer (in-progress tiles). Keyed by block_id.
        # Stage 3b uses a Python dict — small concurrent batch sizes only.
        # Stage 4 will move this into a dedicated GPU buffer.
        self._tails: dict[int, _BlockTail] = {}

        # Sink blocks (NEVER quantised, stay fp16 forever in self._tails).
        # Identified per-request as block_table[r][0]. Populated lazily during
        # ``forward()`` since ``do_kv_cache_update`` doesn't get block_table.
        # TODO(Stage 4.5.e): wire to vLLM's request-completion hook for eviction.
        self._sink_blocks: set[int] = set()

        # ── Stage α-2: deterministic per-block tail pool ─────────────────────
        # Each block_id maps to slot = block_id in the pool (no allocator,
        # no dict). Pool is sized to kv_cache.shape[0] = num_blocks at first
        # `_ensure_pool` call. Sink blocks stay in the pool permanently;
        # non-sink blocks have their slot's content quantised into the int4
        # cache at tile-boundary flushes (triggered from the metadata
        # builder, between captured graph replays).
        self._tail_K_pool: torch.Tensor | None = None   # [POOL_SIZE, group, Hk, D] fp16
        self._tail_V_pool: torch.Tensor | None = None
        # Per-instance shorthand views of the class-level per-device tensors
        # (so kernels can read without dict lookups). Re-bound on every
        # _ensure_pool call.
        self._is_sink_t: torch.Tensor | None = None        # [num_blocks] bool
        self._block_to_slot_t: torch.Tensor | None = None  # [num_blocks] int32
        self._block_lookup_size: int = 0

        # Cached fp16 Hadamard for the rotate-on-store matmul in
        # do_kv_cache_update (avoids a per-call .float() cast that allocates).
        self._H_fp16: torch.Tensor | None = None

        # Store-side rotation scratch (pre-allocated by _ensure_pool so the
        # captured forward never allocates). Shapes:
        #   _k_rot_scratch  [max_num_batched_tokens, Hk, D] fp16
        #   _v_rot_scratch  [max_num_batched_tokens, Hk, D] fp16
        self._k_rot_scratch: torch.Tensor | None = None
        self._v_rot_scratch: torch.Tensor | None = None

        # Reference to this layer's int4 kv_cache, captured on the first
        # forward(). The metadata builder uses it to drive tile-boundary
        # flushes into this layer's cache (outside the captured region).
        self._kv_cache_ref: torch.Tensor | None = None


        # Stage 5.a Step 7 — decode scratch. These instance attrs are
        # bound by `_ensure_pool` to per-device class-shared tensors so all
        # 28 attention layers reuse a single set of buffers.
        self._q_fp32_buf: torch.Tensor | None = None
        self._q_rot_fp32_buf: torch.Tensor | None = None
        self._q_rot_fp16_buf: torch.Tensor | None = None
        self._out_rot_fp32_buf: torch.Tensor | None = None
        self._output_fp32_buf: torch.Tensor | None = None
        self._fused_out_buf: torch.Tensor | None = None
        self._mid_o_buf: torch.Tensor | None = None
        self._mid_lse_buf: torch.Tensor | None = None
        self._fa_K_buf: torch.Tensor | None = None
        self._fa_V_buf: torch.Tensor | None = None

        self.fa_version = get_flash_attn_version(head_size=head_size)

        # Look up serving caps so scratch can be sized once, generously
        # enough that capture probes don't trigger a resize inside the
        # captured region. Falls back to conservative defaults if the
        # global config isn't available (unit tests, etc).
        try:
            from vllm.config import get_current_vllm_config
            _cfg = get_current_vllm_config()
            self._max_num_seqs = _cfg.scheduler_config.max_num_seqs
            self._max_num_batched_tokens = _cfg.scheduler_config.max_num_batched_tokens
            self._max_model_len = _cfg.model_config.max_model_len
            self._num_hidden_layers = getattr(_cfg.model_config.hf_config,
                                              "num_hidden_layers", 32)
        except Exception:
            self._max_num_seqs = 256
            self._max_num_batched_tokens = 8192
            self._num_hidden_layers = 32
            self._max_model_len = 4096

        # Register so the metadata builder can find us (slot allocation /
        # sink marking / flush triggers all enumerate _all_impls).
        type(self)._all_impls.append(self)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ensure_pool(self, device: torch.device, num_blocks_hint: int = 0) -> None:
        """Lazy-allocate the GPU tail pool + lookup tensors + decode scratch.

        Stage α-2: pool is a *fixed-size* sparse buffer
        ([POOL_SIZE, group, Hk, D]). A class-level allocator maps block_id →
        slot (with a GPU lookup tensor `_block_to_slot_t_per_device[device]`
        sized to num_blocks). Pool size = ~2 × max_num_seqs (covers
        sink + in-progress tail for the largest captured batch).

        All allocation happens BEFORE the captured forward, so
        do_kv_cache_update can be pure tensor ops.
        """
        if torch.cuda.is_current_stream_capturing():
            return
        cfg = self.kvarn_config
        cls = type(self)

        # Pool: fixed size, per-instance because each layer holds unique K/V.
        if self._tail_K_pool is None:
            # Size the pool to the structural peak for the *capped* concurrency:
            # sink + in-progress tail per active request, plus the full blocks a
            # chunked prefill can touch. max_num_seqs has already been clamped in
            # the platform's check_and_update_config so this peak fits the pool
            # memory budget — making the pool both exhaustion-safe (the scheduler
            # can never exceed it) and OOM-safe (it is <= the budget). No
            # per-model tuning; KVARN_POOL_SLOTS still pins the count exactly.
            env_slots = int(os.environ.get("KVARN_POOL_SLOTS", "0"))
            if env_slots > 0:
                pool_size = max(env_slots, 64)
            else:
                pool_size = cfg.pool_slots(
                    self._max_num_seqs, self._max_num_batched_tokens
                )
            self._tail_K_pool = torch.zeros(
                pool_size, cfg.group, self.num_kv_heads, cfg.head_dim,
                dtype=torch.float16, device=device,
            )
            self._tail_V_pool = torch.zeros_like(self._tail_K_pool)
            # Class-level allocator state — initialise once on the first
            # impl seen on this device.
            if cls._free_slots is None or cls._allocator_pool_size != pool_size:
                cls._free_slots = list(range(pool_size - 1, -1, -1))
                cls._allocator_pool_size = pool_size
                cls._block_to_slot_dict.clear()
                cls._global_sink_blocks.clear()

        # GPU lookup tensors (per-device, shared across impls on that device).
        num_blocks = max(num_blocks_hint, cls._max_known_block_id + 1, 1024)
        existing = cls._block_to_slot_t_per_device.get(device)
        if existing is None or existing.shape[0] < num_blocks:
            new_b2s = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
            new_is_sink = torch.zeros(num_blocks, dtype=torch.bool, device=device)
            # Re-sync from class-level state (rare, only on resize / first init).
            for bid, slot in cls._block_to_slot_dict.items():
                if bid < num_blocks:
                    new_b2s[bid] = slot
            for bid in cls._global_sink_blocks:
                if bid < num_blocks:
                    new_is_sink[bid] = True
            cls._block_to_slot_t_per_device[device] = new_b2s
            cls._is_sink_t_per_device[device] = new_is_sink
        # Per-instance shorthand pointers so the decode driver / kernels read
        # without dict lookups in the hot path.
        self._is_sink_t = cls._is_sink_t_per_device[device]
        self._block_to_slot_t = cls._block_to_slot_t_per_device[device]
        self._block_lookup_size = self._block_to_slot_t.shape[0]

        # Cached fp16 Hadamard for the rotate-on-store matmul.
        if self._H_fp16 is None:
            self._H_fp16 = self._hadamard(device).to(torch.float16).contiguous()

        # Store-side rotation scratch.
        if self._k_rot_scratch is None:
            q_rows = max(self._max_num_batched_tokens, 1)
            self._k_rot_scratch = torch.empty(
                q_rows, self.num_kv_heads, cfg.head_dim,
                dtype=torch.float16, device=device,
            )
            self._v_rot_scratch = torch.empty_like(self._k_rot_scratch)
        # Decode scratch sized from vllm_config, SHARED across all impl
        # instances on this device (one set per device).
        D = cfg.head_dim
        Hq = self.num_heads
        Hk = self.num_kv_heads
        # Decode scratch rows. The decode driver indexes these buffers by
        # N = B * Hq (decode batch * query heads), so they must hold the largest
        # decode N as well as any prefill token count. A decode step (incl. its
        # CUDA-graph capture batch) has at most max_num_seqs queries, so the
        # decode bound is max_num_seqs * Hq. Sizing to the max of that and
        # max_num_batched_tokens makes the buffers correct for ANY
        # max_num_batched_tokens (the old code silently assumed
        # max_num_batched_tokens >= max_num_seqs * Hq, which breaks when it is
        # set low — e.g. a small chunked-prefill budget on a wide model).
        q_rows = max(self._max_num_batched_tokens, self._max_num_seqs * Hq, 1)
        # FA packed K/V scratch holds the total KV tokens attended in ONE
        # decode step (= sum of the batch's context lengths). The theoretical
        # bound max_num_seqs * max_model_len is pathological (e.g. 256×8192 =
        # 2.1M tokens ≈ 8.6 GB) and would starve the actual KV cache. Cap it at
        # FA_SCRATCH_CAP tokens (~1 GB of fp16 K+V) — enough for typical
        # serving and for the bench (single request up to max_model_len). The
        # scratch is per-step, shared across all layers, allocated ONCE.
        FA_SCRATCH_CAP = 262144
        fa_rows = max(min(self._max_num_seqs * self._max_model_len,
                          FA_SCRATCH_CAP),
                      self._max_model_len, 4096)
        cls = type(self)
        if device not in cls._shared_q_fp32_buf:
            cls._shared_q_fp32_buf[device] = torch.empty(q_rows, D, dtype=torch.float32, device=device)
            cls._shared_q_rot_fp32_buf[device] = torch.empty(q_rows, D, dtype=torch.float32, device=device)
            cls._shared_q_rot_fp16_buf[device] = torch.empty(q_rows, D, dtype=torch.float16, device=device)
            cls._shared_out_rot_fp32_buf[device] = torch.empty(q_rows, D, dtype=torch.float32, device=device)
            cls._shared_output_fp32_buf[device] = torch.empty(q_rows, D, dtype=torch.float32, device=device)
            cls._shared_fused_out_buf[device] = torch.empty(q_rows, D, dtype=torch.float16, device=device)
            from vllm.v1.attention.ops.triton_kvarn_decode import KVARN_NUM_KV_SPLITS
            cls._shared_mid_o_buf[device] = torch.empty(q_rows, KVARN_NUM_KV_SPLITS, D, dtype=torch.float32, device=device)
            cls._shared_mid_lse_buf[device] = torch.empty(q_rows, KVARN_NUM_KV_SPLITS, dtype=torch.float32, device=device)
        if device not in cls._shared_fa_K_buf or cls._shared_fa_K_buf[device].shape[0] < fa_rows:
            cls._shared_fa_K_buf[device] = torch.zeros(fa_rows, Hk, D, dtype=torch.float16, device=device)
            cls._shared_fa_V_buf[device] = torch.zeros_like(cls._shared_fa_K_buf[device])
        # Mirror to instance attrs for fast access by the decode driver.
        self._q_fp32_buf = cls._shared_q_fp32_buf[device]
        self._q_rot_fp32_buf = cls._shared_q_rot_fp32_buf[device]
        self._q_rot_fp16_buf = cls._shared_q_rot_fp16_buf[device]
        self._out_rot_fp32_buf = cls._shared_out_rot_fp32_buf[device]
        self._output_fp32_buf = cls._shared_output_fp32_buf[device]
        self._fused_out_buf = cls._shared_fused_out_buf[device]
        self._mid_o_buf = cls._shared_mid_o_buf[device]
        self._mid_lse_buf = cls._shared_mid_lse_buf[device]
        self._fa_K_buf = cls._shared_fa_K_buf[device]
        self._fa_V_buf = cls._shared_fa_V_buf[device]
    def _batch_slot_mapping_cpu(self) -> list[int] | None:
        """Return the slot_mapping CPU list cached on this step's metadata, or
        None if unavailable. Looks up via the forward context so we don't need
        the caller to plumb it through."""
        try:
            from vllm.forward_context import get_forward_context
            ctx = get_forward_context()
        except Exception:
            return None
        md = getattr(ctx, "attn_metadata", None)
        if md is None:
            return None
        if isinstance(md, dict):
            for m in md.values():
                if isinstance(m, KVarNMetadata):
                    return m.slot_mapping_cpu
            return None
        if isinstance(md, list):
            for entry in md:
                if isinstance(entry, dict):
                    for m in entry.values():
                        if isinstance(m, KVarNMetadata):
                            return m.slot_mapping_cpu
            return None
        return getattr(md, "slot_mapping_cpu", None)

    def _hadamard(self, device: torch.device) -> torch.Tensor:
        return _build_hadamard(self.head_size, device)

    def _flat_block(self, kv_cache: torch.Tensor, block_id: int, head: int) -> torch.Tensor:
        """Contiguous ``[tile_bytes_aligned]`` uint8 view for one (block, head).

        ``kv_cache`` has shape ``(num_blocks, num_kv_heads, tile_bytes_aligned)``,
        so this selects a single contiguous row — no copy, writes propagate
        back to the cache tensor.
        """
        return kv_cache[block_id, head]

    def _write_packed(
        self, kv_cache: torch.Tensor, block_id: int, head: int,
        store_K: dict[str, torch.Tensor], store_V: dict[str, torch.Tensor],
    ) -> None:
        cfg = self.kvarn_config
        flat = self._flat_block(kv_cache, block_id, head)

        # K packed bytes
        ko = cfg.k_packed_offset
        flat[ko:ko + cfg.k_packed_bytes] = store_K["q_packed_uint8"].reshape(-1).to(torch.uint8)
        # K s_col, zp (per-channel, length D, fp16)
        flat[cfg.k_s_col_offset:cfg.k_s_col_offset + cfg.head_dim * 2].view(
            torch.float16
        )[:] = store_K["s_col_K"]
        flat[cfg.k_zp_offset:cfg.k_zp_offset + cfg.head_dim * 2].view(
            torch.float16
        )[:] = store_K["zp_K"]
        flat[cfg.k_s_row_offset:cfg.k_s_row_offset + cfg.group * 2].view(
            torch.float16
        )[:] = store_K["s_row_K"]

        # V packed bytes
        vo = cfg.v_packed_offset
        flat[vo:vo + cfg.v_packed_bytes] = store_V["q_packed_uint8"].reshape(-1).to(torch.uint8)
        flat[cfg.v_s_col_offset:cfg.v_s_col_offset + cfg.head_dim * 2].view(
            torch.float16
        )[:] = store_V["s_col_V"]
        flat[cfg.v_s_row_offset:cfg.v_s_row_offset + cfg.group * 2].view(
            torch.float16
        )[:] = store_V["s_row_V"]
        flat[cfg.v_zp_offset:cfg.v_zp_offset + cfg.group * 2].view(
            torch.float16
        )[:] = store_V["zp_V"]

    def _read_block_dequantized(
        self, kv_cache: torch.Tensor, block_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a full quantized block and return (K, V) in unrotated frame.

        Returns:
            K: [group, num_kv_heads, head_dim] fp16
            V: [group, num_kv_heads, head_dim] fp16
        """
        cfg = self.kvarn_config
        group = cfg.group
        D = cfg.head_dim
        device = kv_cache.device

        K_out = torch.empty(group, self.num_kv_heads, D, dtype=torch.float16, device=device)
        V_out = torch.empty(group, self.num_kv_heads, D, dtype=torch.float16, device=device)

        H = self._hadamard(device)  # [D, D] fp32

        for h in range(self.num_kv_heads):
            flat = self._flat_block(kv_cache, block_id, h)

            # K side
            k_packed = flat[cfg.k_packed_offset:cfg.k_packed_offset + cfg.k_packed_bytes
                            ].view(D, group // 2)
            s_col_K = flat[cfg.k_s_col_offset:cfg.k_s_col_offset + D * 2].view(torch.float16)
            zp_K = flat[cfg.k_zp_offset:cfg.k_zp_offset + D * 2].view(torch.float16)
            s_row_K = flat[cfg.k_s_row_offset:cfg.k_s_row_offset + group * 2].view(torch.float16)
            K_rot_DG = kvarn_dequant_tile_k(k_packed, s_col_K, zp_K, s_row_K, group=group)
            # Un-rotate: [D, group] → [group, D] (= K rows-tokens), then ⋅H to undo rotation
            K_unrot = K_rot_DG.T @ H  # [group, D]
            K_out[:, h, :] = K_unrot.to(torch.float16)

            # V side
            v_packed = flat[cfg.v_packed_offset:cfg.v_packed_offset + cfg.v_packed_bytes
                            ].view(group, D // 2)
            s_col_V = flat[cfg.v_s_col_offset:cfg.v_s_col_offset + D * 2].view(torch.float16)
            s_row_V = flat[cfg.v_s_row_offset:cfg.v_s_row_offset + group * 2].view(torch.float16)
            zp_V = flat[cfg.v_zp_offset:cfg.v_zp_offset + group * 2].view(torch.float16)
            V_rot_GD = kvarn_dequant_tile_v(v_packed, s_col_V, s_row_V, zp_V, head_dim=D)
            V_unrot = V_rot_GD @ H  # [group, D]
            V_out[:, h, :] = V_unrot.to(torch.float16)

        return K_out, V_out

    def _flush_tail(self, block_id: int, kv_cache: torch.Tensor) -> None:
        """Quantize a fully-filled tail buffer and write it into the cache.

        Stage α-2 sparse pool: pool slot = _block_to_slot_dict[block_id].
        Data is already rotated (rotation happens at do_kv_cache_update),
        so no `@ H` step here.
        """
        cfg = self.kvarn_config
        cls = type(self)
        slot = cls._block_to_slot_dict.get(block_id)
        if slot is None:
            # Block has no pool slot — nothing to flush.
            self._tails.pop(block_id, None)
            return
        K_rot = self._tail_K_pool[slot].float()                   # [group, Hk, D]
        V_rot = self._tail_V_pool[slot].float()                   # [group, Hk, D]
        self._tails.pop(block_id, None)                           # drop tracker entry

        # Build batched per-head tiles (rows = absorb axis for each)
        K_tiles = K_rot.permute(1, 2, 0).contiguous()             # [Hk, D, group]
        V_tiles = V_rot.permute(1, 0, 2).contiguous()             # [Hk, group, D]

        # One Triton launch for all 2*Hk tiles (same [R,C] shape).
        all_tiles = torch.cat([K_tiles, V_tiles], dim=0)          # [2*Hk, 128, 128]
        all_bal, all_sc, all_sr = kvarn_sinkhorn_triton(
            all_tiles, iterations=cfg.sinkhorn_iters,
        )
        Hk = self.num_kv_heads
        K_out = kvarn_store_tile_k_batch_from_sinkhorn(
            all_bal[:Hk], all_sc[:Hk], all_sr[:Hk], bits=cfg.key_bits,
        )
        V_out = kvarn_store_tile_v_batch_from_sinkhorn(
            all_bal[Hk:], all_sc[Hk:], all_sr[Hk:], bits=cfg.value_bits,
        )

        for h in range(Hk):
            store_K = {k: v[h] for k, v in K_out.items()}
            store_V = {k: v[h] for k, v in V_out.items()}
            self._write_packed(kv_cache, block_id, h, store_K, store_V)

        # NOTE: the pool slot is NOT freed here. The slot index addresses the
        # SAME row in every layer's pool, so it must stay allocated until ALL
        # layers have flushed their data into int4. The builder frees it once,
        # after iterating every impl (see "Free the flushed blocks' slots" in
        # build()). Freeing here would let layer 0's flush drop the slot, after
        # which layers 1..N find no slot (`.get()` → None) and silently skip
        # writing their int4 — corrupting all-but-the-first layer's history.

    @classmethod
    def _batched_flush(cls, flush_pairs: list) -> None:
        """Flush many (impl, block_id, kv_cache) tiles via batched Sinkhorn + RTN.

        Replaces the per-(layer, block) Python loop calling `_flush_tail`. At
        burst with many layers × many lockstep boundary crossings, the per-call
        kernel-launch + Python-iter overhead dominated; Sinkhorn and the RTN-
        pack are per-tile-independent, so stacking is numerically identical
        (no accuracy change).

        Chunked at CHUNK_PAIRS to bound the transient gather memory — at peak
        (48 layers × ~73 lockstep reqs = ~3.5k pairs), the unchunked stack hits
        >2 GB of fp32 working memory and OOMs on a memory-tight burst.
        """
        if not flush_pairs:
            return
        CHUNK_PAIRS = 256
        cfg = flush_pairs[0][0].kvarn_config
        Hk = flush_pairs[0][0].num_kv_heads
        # Pre-filter pairs that still have a pool slot (some may have been freed
        # by a sibling impl's flush already during this builder call).
        filt: list[tuple] = []
        for impl, bid, kvc in flush_pairs:
            slot = cls._block_to_slot_dict.get(bid)
            if slot is None:
                impl._tails.pop(bid, None)
                continue
            filt.append((impl, bid, kvc, slot))
            impl._tails.pop(bid, None)
        if not filt:
            return
        for c0 in range(0, len(filt), CHUNK_PAIRS):
            chunk = filt[c0:c0 + CHUNK_PAIRS]
            N = len(chunk)
            # Gather pool data for this chunk.
            K_list = [impl._tail_K_pool[slot].float() for impl, _, _, slot in chunk]   # [G, Hk, D]
            V_list = [impl._tail_V_pool[slot].float() for impl, _, _, slot in chunk]
            K_stack = torch.stack(K_list, dim=0)                                       # [N, G, Hk, D]
            V_stack = torch.stack(V_list, dim=0)
            # Optional: dump first chunk's raw (pre-Sinkhorn) tiles for outlier
            # analysis (KVARN_DUMP_TILES=/path/to/file.pt).
            dump_path = os.environ.get("KVARN_DUMP_TILES", "")
            if dump_path and not getattr(cls, "_tiles_dumped", False):
                cls._tiles_dumped = True
                # Capture per-tile (layer_idx, block_id) for per-layer analysis.
                # layer_idx pulled from impl.layer_name (e.g. "model.layers.7.self_attn")
                # via a regex fallback to enumerate index if name parsing fails.
                import re
                lyr_ids, blk_ids = [], []
                for impl, bid, _, _ in chunk:
                    name = getattr(impl, "layer_name", "") or ""
                    m = re.search(r"layers\.(\d+)\b", name)
                    lyr_ids.append(int(m.group(1)) if m else -1)
                    blk_ids.append(int(bid))
                torch.save({"K_stack": K_stack.detach().cpu(),
                            "V_stack": V_stack.detach().cpu(),
                            "layer_ids": lyr_ids,
                            "block_ids": blk_ids,
                            "Hk": flush_pairs[0][0].num_kv_heads,
                            "G": cfg.group, "D": cfg.head_dim,
                            "key_bits": cfg.key_bits, "value_bits": cfg.value_bits,
                            "sinkhorn_iters": cfg.sinkhorn_iters},
                           dump_path)
                print(f"[KVARN] dumped {N} (layer,block) pre-Sinkhorn tiles → {dump_path}",
                      flush=True)
                print(f"[KVARN] layer_ids in dump: {sorted(set(lyr_ids))}", flush=True)
            del K_list, V_list
            # K tile per Sinkhorn batch row: [D, G] (absorb = channel).
            K_tiles = K_stack.permute(0, 2, 3, 1).reshape(N * Hk, K_stack.shape[3], K_stack.shape[1])
            V_tiles = V_stack.permute(0, 2, 1, 3).reshape(N * Hk, V_stack.shape[1], V_stack.shape[3])
            del K_stack, V_stack
            all_tiles = torch.cat([K_tiles, V_tiles], dim=0)                           # [2*N*Hk, R, C]
            del K_tiles, V_tiles
            all_bal, all_sc, all_sr = kvarn_sinkhorn_triton(
                all_tiles, iterations=cfg.sinkhorn_iters,
            )
            del all_tiles
            nh = N * Hk
            K_out = kvarn_store_tile_k_batch_from_sinkhorn(
                all_bal[:nh], all_sc[:nh], all_sr[:nh], bits=cfg.key_bits,
            )
            V_out = kvarn_store_tile_v_batch_from_sinkhorn(
                all_bal[nh:], all_sc[nh:], all_sr[nh:], bits=cfg.value_bits,
            )
            del all_bal, all_sc, all_sr
            # Distribute packed results to each (layer, block, head) cache slot.
            for i, (impl, bid, kvc, _) in enumerate(chunk):
                for h in range(Hk):
                    idx = i * Hk + h
                    store_K = {k: v[idx] for k, v in K_out.items()}
                    store_V = {k: v[idx] for k, v in V_out.items()}
                    impl._write_packed(kvc, bid, h, store_K, store_V)
            del K_out, V_out

    # ── do_kv_cache_update ───────────────────────────────────────────────────

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Append incoming tokens to the per-block fp16 tail buffers.

        We DO NOT flush here. Flushing requires the block_table (to know
        which block IDs are "sink" blocks for each request), which is only
        available via ``attn_metadata`` in ``forward()``. The flush is
        therefore deferred to ``_flush_eligible_tails``, invoked at the top
        of ``forward()``.
        """
        # Stage α-2: fully tensorised store. rotate(k, v) by H_fp16 → scatter
        # into pool at slot=block_id directly. No Python loop, no allocator,
        # no dict mutation. Safe inside a captured CUDA graph.
        cfg = self.kvarn_config
        N = slot_mapping.shape[0]
        if N <= 0:
            return
        device = key.device
        Hk = self.num_kv_heads
        D = self.head_size

        # Ensure pool + lookup tensors + rotation scratch exist (no-op during
        # capture; first call before capture sizes pool to kv_cache num_blocks).
        self._ensure_pool(device, num_blocks_hint=kv_cache.shape[0])

        # Reshape to (N, Hk, D) — view, no copy (key/value already fp16).
        k_view = key[:N].view(N, Hk, D)
        v_view = value[:N].view(N, Hk, D)

        # Rotate via cached fp16 Hadamard. torch.matmul `out=` is
        # capture-friendly (uses the caching allocator's pool).
        k_rot = self._k_rot_scratch[:N]
        v_rot = self._v_rot_scratch[:N]
        torch.matmul(k_view, self._H_fp16, out=k_rot)
        torch.matmul(v_view, self._H_fp16, out=v_rot)

        # Scatter via the sparse pool indirection. Slot lookup (block_id →
        # pool slot) is done inside the kernel against the GPU
        # _block_to_slot_t tensor (mutated only by the metadata builder).
        from vllm.v1.attention.ops.triton_kvarn_decode import (
            _kvarn_scatter_store_kernel,
        )
        _kvarn_scatter_store_kernel[(N, Hk)](
            k_rot, v_rot, slot_mapping[:N],
            self._block_to_slot_t,
            self._tail_K_pool, self._tail_V_pool,
            k_rot.stride(0), k_rot.stride(1),
            self._tail_K_pool.stride(0),
            self._tail_K_pool.stride(1),
            self._tail_K_pool.stride(2),
            GROUP=cfg.group, D=D,
            NUM_BLOCKS_LOOKUP=self._block_lookup_size,
            num_warps=2, num_stages=2,
        )
        # No CPU bookkeeping here — fill tracking + flush triggering live in
        # KVarNMetadataBuilder.build() (outside the captured region). This
        # method is now pure tensor ops, safe inside a captured CUDA graph.

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "KVarNMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        device = query.device

        if output is None:
            output = torch.zeros(
                num_tokens, self.num_heads * self.head_size,
                dtype=query.dtype, device=device,
            )
        if attn_metadata is None:
            return output.fill_(0)

        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        # Make sure pool + block-lookup tensors exist and cover num_blocks.
        self._ensure_pool(kv_cache.device, num_blocks_hint=kv_cache.shape[0])
        # Cache the kv_cache ref so the metadata builder can drive flushes
        # into this layer's int4 cache (outside the captured region).
        self._kv_cache_ref = kv_cache

        # Flush is now triggered from KVarNMetadataBuilder.build() between
        # captured graph replays — nothing to do here at the top of forward.

        q = query[:N].view(N, self.num_heads, self.head_size)

        if not attn_metadata.is_prefill:
            attn_out = self._decode_path(q, kv_cache, attn_metadata)
        elif attn_metadata.num_decodes == 0:
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_first_chunk(q, k, v, attn_metadata, kv_cache)
        else:
            # Mixed batch — split into decode + prefill portions, same as TurboQuant.
            attn_out = self._mixed_batch_path(
                q, key[:N], value[:N], kv_cache, attn_metadata
            )

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ── attention sub-paths ──────────────────────────────────────────────────

    def _flash_varlen(
        self, q, k, v, cu_q, cu_k, max_q, max_k,
    ) -> torch.Tensor:
        if self.fa_version is None:
            return flash_attn_varlen_func(
                q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=max_q, max_seqlen_k=max_k,
                softmax_scale=self.scale, causal=True,
            )
        return flash_attn_varlen_func(
            q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_q, max_seqlen_k=max_k,
            softmax_scale=self.scale, causal=True, fa_version=self.fa_version,
        )

    def _prefill_first_chunk(
        self, q, k, v, attn_metadata: KVarNMetadata, kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        """First-chunk prefill: every request's full prompt is in the current
        batch, so attention runs on raw K/V via flash_attn_varlen. The K/V
        have already been written to the cache by `do_kv_cache_update`."""
        if _HAS_FLASH_ATTN:
            return self._flash_varlen(
                q, k, v,
                cu_q=attn_metadata.query_start_loc,
                cu_k=attn_metadata.query_start_loc,
                max_q=attn_metadata.max_query_len,
                max_k=attn_metadata.max_query_len,
            )
        # Per-request SDPA fallback (e.g. when flash_attn isn't available).
        outputs = []
        qsl = attn_metadata.query_start_loc.tolist()
        for r in range(len(qsl) - 1):
            qs, qe = qsl[r], qsl[r + 1]
            if qe <= qs:
                continue
            q_r = q[qs:qe].transpose(0, 1).unsqueeze(0)  # [1, Hq, q_len, D]
            k_r = k[qs:qe].transpose(0, 1).unsqueeze(0)
            v_r = v[qs:qe].transpose(0, 1).unsqueeze(0)
            o = F.scaled_dot_product_attention(
                q_r, k_r, v_r, is_causal=True, scale=self.scale,
                enable_gqa=self.num_kv_heads < self.num_heads,
            )
            outputs.append(o[0].transpose(0, 1))         # [q_len, Hq, D]
        return torch.cat(outputs, dim=0) if outputs else torch.empty(
            0, self.num_heads, self.head_size, device=q.device, dtype=q.dtype,
        )

    def _gather_request_kv(
        self, kv_cache: torch.Tensor, block_table_row: torch.Tensor, seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct full fp16 K and V for one request from cached blocks
        + tail buffers. Returns (K [seq_len, Hk, D], V [seq_len, Hk, D])."""
        cfg = self.kvarn_config
        group = cfg.group
        n_full = seq_len // group
        tail_len = seq_len % group
        device = kv_cache.device
        D = self.head_size

        # Stage α-2: pool stores ROTATED K/V indexed by block_id directly.
        # The slow fallback consumer (_decode_path_slow → SDPA) expects
        # un-rotated K/V, so apply H^-1 (= H^T for orthonormal H).
        H = self._hadamard(device)                                # [D, D] fp32

        def _unrot_pool(x: torch.Tensor) -> torch.Tensor:
            return (x.float() @ H.T).to(torch.float16)

        # Stage α-2: a block lives in the fp16 pool iff it has a slot
        # (sinks + in-progress tails). Flushed blocks have their slot freed
        # and live in the int4 cache.
        dict_map = type(self)._block_to_slot_dict

        K_parts: list[torch.Tensor] = []
        V_parts: list[torch.Tensor] = []

        for i in range(n_full):
            block_id = int(block_table_row[i].item())
            slot = dict_map.get(block_id)
            if slot is not None:
                K_parts.append(_unrot_pool(self._tail_K_pool[slot]))
                V_parts.append(_unrot_pool(self._tail_V_pool[slot]))
            else:
                K_blk, V_blk = self._read_block_dequantized(kv_cache, block_id)
                K_parts.append(K_blk)
                V_parts.append(V_blk)

        if tail_len > 0:
            block_id = int(block_table_row[n_full].item())
            slot = dict_map.get(block_id)
            if slot is not None:
                K_parts.append(_unrot_pool(self._tail_K_pool[slot, :tail_len]))
                V_parts.append(_unrot_pool(self._tail_V_pool[slot, :tail_len]))
            else:
                K_parts.append(torch.zeros(
                    tail_len, self.num_kv_heads, D,
                    dtype=torch.float16, device=device,
                ))
                V_parts.append(torch.zeros_like(K_parts[-1]))

        K = torch.cat(K_parts, dim=0) if K_parts else torch.empty(
            0, self.num_kv_heads, D, dtype=torch.float16, device=device,
        )
        V = torch.cat(V_parts, dim=0) if V_parts else torch.empty_like(K)
        return K, V

    def _decode_path(
        self, q: torch.Tensor, kv_cache: torch.Tensor,
        attn_metadata: KVarNMetadata,
    ) -> torch.Tensor:
        """Triton-driven decode: in-kernel dequant + scoring + weighted V,
        with the in-progress fp16 tail buffers combined via LSE in PyTorch.

        Assumes one query token per request (the standard decode regime).
        For mixed-query-length decode steps (e.g. speculative decoding), this
        path falls back to ``_decode_path_slow`` which materialises fp16 K/V
        and runs SDPA — retained for correctness in edge cases.
        """
        # If every request contributes exactly one query token, the Triton
        # kernel's (B, Hq) launch shape is valid; otherwise fall back.
        # Use the precomputed Python-int max_query_len (NOT a GPU reduction +
        # host branch — that would force a sync, forbidden during CUDA graph
        # capture).
        if attn_metadata.max_query_len > 1:
            return self._decode_path_slow(q, kv_cache, attn_metadata)

        # q shape: [num_decode_tokens, num_heads, head_dim]
        # num_decode_tokens == B (one token per request)
        return kvarn_decode_attention(
            query=q,
            kv_cache=kv_cache,
            hadamard=self._hadamard(q.device),
            scale=self.scale,
            cfg=self.kvarn_config,
            impl=self,
            md=attn_metadata,
        )

    def _decode_path_slow(
        self, q: torch.Tensor, kv_cache: torch.Tensor,
        attn_metadata: KVarNMetadata,
    ) -> torch.Tensor:
        """Fallback: full fp16 dequant + SDPA. Used for multi-query-per-request
        decode steps (e.g. speculative decoding) where the Triton kernel's
        ``(B, Hq)`` per-program shape would mis-handle the per-token mapping.
        """
        num_reqs = attn_metadata.block_table.shape[0]
        seq_lens = attn_metadata.seq_lens.tolist()
        qsl = attn_metadata.query_start_loc.tolist()
        out = torch.empty(q.shape[0], self.num_heads, self.head_size,
                          dtype=q.dtype, device=q.device)
        for r in range(num_reqs):
            q_start, q_end = qsl[r], qsl[r + 1]
            if q_end <= q_start:
                continue
            seq_len = seq_lens[r]
            K_full, V_full = self._gather_request_kv(
                kv_cache, attn_metadata.block_table[r], seq_len,
            )
            q_r = q[q_start:q_end].transpose(0, 1).unsqueeze(0).float()
            K_t = K_full.transpose(0, 1).unsqueeze(0).float()
            V_t = V_full.transpose(0, 1).unsqueeze(0).float()
            cached_len = seq_len - (q_end - q_start)
            q_pos = torch.arange(q_end - q_start, device=q.device).unsqueeze(1) + cached_len
            k_pos = torch.arange(seq_len, device=q.device).unsqueeze(0)
            mask = k_pos <= q_pos
            o = F.scaled_dot_product_attention(
                q_r, K_t, V_t, attn_mask=mask, scale=self.scale,
                enable_gqa=self.num_kv_heads < self.num_heads,
            )
            out[q_start:q_end] = o[0].transpose(0, 1).to(q.dtype)
        return out

    def _mixed_batch_path(
        self, q: torch.Tensor, k_all: torch.Tensor, v_all: torch.Tensor,
        kv_cache: torch.Tensor, attn_metadata: KVarNMetadata,
    ) -> torch.Tensor:
        """Split mixed batch into decode-then-prefill, mirroring TurboQuant."""
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        N = attn_metadata.num_actual_tokens

        out = torch.empty(N, self.num_heads, self.head_size,
                          dtype=q.dtype, device=q.device)

        # Build the Stage α-2 fa_* fields for the decode subset. This path
        # always runs eager (mixed batches aren't graph-captured), so fresh
        # (non-persistent) tensors are fine.
        group = self.kvarn_config.group
        dec_seq_lens = attn_metadata.seq_lens[:num_decodes].to(torch.int32)
        dec_cu_k = torch.nn.functional.pad(
            torch.cumsum(dec_seq_lens, dim=0), (1, 0)
        ).to(torch.int32)
        dec_cu_q = torch.arange(
            num_decodes + 1, dtype=torch.int32, device=q.device
        )
        mbpr = (self._max_model_len + group - 1) // group
        decode_meta = KVarNMetadata(
            seq_lens=attn_metadata.seq_lens[:num_decodes],
            slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
            block_table=attn_metadata.block_table[:num_decodes],
            query_start_loc=attn_metadata.query_start_loc[:num_decodes + 1],
            num_actual_tokens=num_decode_tokens,
            max_query_len=1, max_seq_len=attn_metadata.max_seq_len,
            is_prefill=False,
            fa_cu_seqlens_q=dec_cu_q,
            fa_cu_seqlens_k=dec_cu_k,
            fa_max_blocks_per_req=mbpr,
            fa_max_seqlen_k_fixed=self._max_model_len,
        )
        out[:num_decode_tokens] = self._decode_path(
            q[:num_decode_tokens], kv_cache, decode_meta,
        )

        prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
        prefill_qsl = attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
        prefill_meta = KVarNMetadata(
            seq_lens=prefill_seq_lens,
            slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
            block_table=attn_metadata.block_table[num_decodes:],
            query_start_loc=prefill_qsl,
            num_actual_tokens=N - num_decode_tokens,
            max_query_len=attn_metadata.max_query_len,
            max_seq_len=int(prefill_seq_lens.max().item()),
            is_prefill=True,
        )
        k_pref = k_all[num_decode_tokens:].view(-1, self.num_kv_heads, self.head_size)
        v_pref = v_all[num_decode_tokens:].view(-1, self.num_kv_heads, self.head_size)
        out[num_decode_tokens:] = self._prefill_first_chunk(
            q[num_decode_tokens:], k_pref, v_pref, prefill_meta, kv_cache,
        )
        return out
