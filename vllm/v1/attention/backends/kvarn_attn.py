# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN attention backend.

KV-cache compression by Hadamard rotation + iterative variance-normalization
(Sinkhorn-like) + asymmetric RTN. K is quantized per-channel, V per-token —
KIVI orientation. The variance-normalization tile equals the vLLM
``block_size`` (default and only supported value in this PR: ``128``).

Cache layout (per block, per kv-head, ``head_dim=128, k_bits=4, v_bits=4``):
  17920 B = 8192 (K packed) + 256 + 256 + 256  (K absorbed scales + zp + s_row)
          + 8192 (V packed) + 256 + 256 + 256  (V s_col + absorbed s_row + zp)

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
from typing import ClassVar

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
    kvarn_store_tile_k_batch_from_sinkhorn,
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


def _sinkhorn_pack_kv(K_tiles, V_tiles, cfg):
    """Sinkhorn-balance + pack a batch of K and V tiles into int4 stores.

    K_tiles is [N, D, group] (absorb axis = channel), V_tiles is [N, group, D]
    (absorb axis = token). When D == group (head_dim 128) the two have the same
    [R, C] shape, so we fuse them into ONE Triton Sinkhorn launch. When D != group
    (e.g. head_dim 256, group 128) the tiles are non-square and have different
    [R, C] — kvarn_sinkhorn_triton takes R, C as per-launch constexpr — so K and V
    must be balanced in SEPARATE launches. (A single torch.cat here assumed square
    and broke at head_dim=256.)"""
    if K_tiles.shape[1:] == V_tiles.shape[1:]:
        nk = K_tiles.shape[0]
        bal, sc, sr = kvarn_sinkhorn_triton(
            torch.cat([K_tiles, V_tiles], dim=0),
            iterations=cfg.sinkhorn_iters,
        )
        K_out = kvarn_store_tile_k_batch_from_sinkhorn(
            bal[:nk], sc[:nk], sr[:nk], bits=cfg.key_bits
        )
        V_out = kvarn_store_tile_v_batch_from_sinkhorn(
            bal[nk:], sc[nk:], sr[nk:], bits=cfg.value_bits
        )
    else:
        kbal, ksc, ksr = kvarn_sinkhorn_triton(K_tiles, iterations=cfg.sinkhorn_iters)
        vbal, vsc, vsr = kvarn_sinkhorn_triton(V_tiles, iterations=cfg.sinkhorn_iters)
        K_out = kvarn_store_tile_k_batch_from_sinkhorn(
            kbal, ksc, ksr, bits=cfg.key_bits
        )
        V_out = kvarn_store_tile_v_batch_from_sinkhorn(
            vbal, vsc, vsr, bits=cfg.value_bits
        )
    return K_out, V_out


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
        "kvarn_k4v4_g128",
        "kvarn_k4v2_g128",
        "kvarn_k4v4_g64",
        "kvarn_k4v2_g64",
    ]

    @staticmethod
    def get_name() -> str:
        return "KVARN"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # One vLLM block == one KVarN tile (cfg.group). Supported tile sizes are
        # the distinct `group` values across the registered presets (64, 128).
        from vllm.model_executor.layers.quantization.kvarn.config import (
            KVARN_PRESETS,
        )

        return sorted({p["group"] for p in KVARN_PRESETS.values()})

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # The active preset pins the tile size (..._g64 / ..._g128), and
        # get_kv_cache_shape asserts block_size == cfg.group. The generic
        # fallback returns the MINIMUM supported size (64) whenever the
        # framework default (16) is unsupported — which breaks any g128 preset
        # run without an explicit --block-size (a g128 deployment then builds
        # its cache with 64-token kernel blocks and dies on the assert, e.g.
        # hybrid models without spec decode). Prefer the preset's group.
        from vllm.config.vllm import get_current_vllm_config
        from vllm.model_executor.layers.quantization.kvarn.config import (
            KVARN_PRESETS,
        )

        try:
            cache_dtype = get_current_vllm_config().cache_config.cache_dtype
        except Exception:
            cache_dtype = None
        if isinstance(cache_dtype, str) and cache_dtype in KVARN_PRESETS:
            return KVARN_PRESETS[cache_dtype]["group"]
        return super().get_preferred_block_size(default_block_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_non_causal(cls) -> bool:
        # Causality is purely a masking choice; the KV quantization is
        # independent of it. The per-token verify path attends each query row
        # to [0, vq_seqlen[row]) — a flat full-context length per row gives
        # bidirectional attention (used by DFlash cross-attention drafting).
        return True

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type[KVarNAttentionImpl]:
        return KVarNAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[KVarNMetadataBuilder]:
        return KVarNMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "kvarn_k4v4_g128",
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
        return kv_cache_dtype.startswith("kvarn_") and not kv_cache_dtype.startswith(
            "kvarn_mla"
        )

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (128, 256, 512)

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Multimodal models (e.g. Gemma-4) set use_mm_prefix; text generation
        # never materializes mm tokens so KVarN decode is unaffected. (Image/audio
        # prefix full-attention correctness is unverified — text-only validated.)
        return True


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
    # True if any multi-query request (query_len > 1) also has cached context
    # (seq_len > query_len): a speculative-decode verify step or a chunked-
    # prefill continuation. Such steps MUST attend over the cached K/V, so they
    # route to the context-aware path rather than _prefill_first_chunk (which
    # assumes a fresh prompt, cached_len == 0). Computed once in build() from
    # CPU arrays (no GPU sync).
    has_cached_multiquery: bool = False
    # Precomputed once per batch in the metadata builder and reused across all
    # 28+ layer forward calls. Saves 28× .tolist() syncs per decode token.
    seq_lens_cpu: list[int] | None = None
    block_table_cpu: list[list[int]] | None = None
    slot_mapping_cpu: list[int] | None = None
    # Stage α-2 capture-correct decode metadata. The block_table-driven
    # build-packed-KV kernel reads block_table / seq_lens / fa_cu_seqlens_k
    # directly (all PERSISTENT buffers updated in-place by the builder), so a
    # captured CUDA graph sees fresh data on every replay.
    fa_cu_seqlens_q: torch.Tensor | None = None  # [B+1] int32 (persistent)
    fa_cu_seqlens_k: torch.Tensor | None = (
        None  # [B+1] int32 (persistent prefix sum of seq_lens)
    )
    fa_max_blocks_per_req: int = 0  # ceil(max_model_len / group): grid dim
    fa_max_seqlen_k_fixed: int = 0  # = max_model_len; fixed FA grid bound
    # Verify (spec-as-decode) plan: one virtual kernel row per decode-portion
    # query token. Persistent buffers (pointers baked into captured graphs),
    # filled CPU-side in build(). None when the decode portion is single-token.
    vq_req: torch.Tensor | None = None  # [num_decode_tokens] int32 block-table row
    vq_seqlen: torch.Tensor | None = None  # [num_decode_tokens] int32 causal length
    vq_qlen: int = 0  # uniform decode query len (>=2), else 0
    # Non-causal (bidirectional) attention: each query row attends to the full
    # context (no bottom-right causal staircase). Set by DFlash cross-attention
    # drafting via CommonAttentionMetadata.causal=False. When False, the verify
    # plan stores a flat full-context length per row instead of committed+j+1.
    causal: bool = True


class KVarNMetadataBuilder(AttentionMetadataBuilder[KVarNMetadata]):
    """Builds ``KVarNMetadata`` from scheduler output."""

    # UNIFORM_BATCH: spec-decode verify steps (uniform query length
    # 1 + num_spec) are graph-capturable via the fused verify kernel — the
    # whole MTP step replays as ONE full graph like vanilla FA, instead of
    # ~num_layers eager attention calls between piecewise segments per step
    # (the dominant MTP overhead once the materialize round-trip was gone;
    # the gap to vanilla was 0.65-0.85x and worse under TP). All Python
    # state mutation (slot allocation, sink marking, tile-boundary flush,
    # the vq verify plan) happens in KVarNMetadataBuilder.build() between
    # captured graph replays; the forward is pure tensor ops.
    # KVARN_FUSED_VERIFY=0 reverts to single-token-only support.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
        if os.environ.get("KVARN_FUSED_VERIFY", "1") == "1"
        else AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # spec-as-decode: verify steps (query_len <= 1 + num_spec) classify
        # as decodes and carry a vq plan (see build()) for the fused verify
        # kernel; the threshold is derived from speculative_config by the
        # base helper.
        self._init_reorder_batch_threshold(
            1,
            supports_spec_as_decode=(os.environ.get("KVARN_FUSED_VERIFY", "1") == "1"),
        )
        # KV-cache-group key, must match KVarNAttentionImpl._group_key for this
        # group's layers so the builder mutates the right group's slot allocator.
        # (head_size, num_kv_heads, sliding_window) — see impl._group_key.
        # TRUE per-group identity = this builder's exact layer set. A config
        # proxy (head,kv,sw) is NOT enough: vLLM splits same-config layers into
        # multiple groups (Gemma-4's repeating pattern -> 5 sliding groups all
        # head256/16kv/1024), each with its own block_id space. The builder tags
        # its impls with this key in build() (impls don't reliably carry a name).
        self._layer_names = list(layer_names)
        self._layer_names_set = set(self._layer_names)
        self._group_key = tuple(sorted(self._layer_names))
        # Stage α-2: per-block fill tracking — block_id -> tokens present in
        # the pool for that block after the current step. Keyed by PHYSICAL
        # block (never by request or by the sink block id): vLLM's prefix
        # caching shares physical blocks across live requests and recycles ids
        # across finished ones, so any request-identity proxy collides under
        # sharing (the repetition-collapse / stale-tile class). A
        # partial block has exactly one writer, so the value has a single
        # source. Drives flush-on-reclaim: a finished request's complete block
        # must be flushed (a future prefix-cache hit may read it), a partial
        # one is safe to discard (vLLM never prefix-caches partial blocks).
        self._block_fill: dict[int, int] = {}
        # Retired sinks: finished requests' sink blocks, kept RESIDENT in the
        # fp16 pool (insertion order = retirement order) instead of flushed on
        # reclaim. A prefix-cache hit re-adopts the block with its fp16 data
        # byte-identical — preserving KVarN's fp16-sink accuracy on multi-turn
        # traffic, where every follow-up turn reuses the previous turn's first
        # block. Evicted (flushed to int4, so later cache hits still find a
        # valid tile) lazily, oldest first, only when slot allocation runs
        # dry — residency therefore never shrinks live capacity.
        self._retired_sinks: dict[int, None] = {}

        # Max model length (for the fixed FA grid bound + max_blocks_per_req).
        try:
            self._max_model_len = vllm_config.model_config.max_model_len
        except Exception:
            self._max_model_len = 4096

        # KVarN tile / group size (= vLLM block size). Sourced from the configured
        # kv-cache dtype so non-128 groups (e.g. g64) drive the flush + slot math
        # in build() correctly. Every storage / kernel path already reads
        # cfg.group; this is the one place the builder needs it without an impl
        # handle. Falls back to 128 if it cannot be parsed.
        self._group = 128
        try:
            from vllm.model_executor.layers.quantization.kvarn.config import (
                KVarNConfig,
            )

            _cd = vllm_config.cache_config.cache_dtype
            _hd = vllm_config.model_config.get_head_size()
            self._group = KVarNConfig.from_cache_dtype(_cd, _hd).group
        except Exception:
            self._group = 128

        # Persistent cu_seqlens buffers (allocated lazily in build()).
        self._cu_seqlens_q_buf: torch.Tensor = None  # type: ignore[assignment]
        self._cu_seqlens_k_buf: torch.Tensor = None  # type: ignore[assignment]
        self._cu_seqlens_q_host: torch.Tensor = None  # type: ignore[assignment]
        self._cu_seqlens_k_host: torch.Tensor = None  # type: ignore[assignment]
        # Persistent verify-plan buffers (allocated lazily in build()).
        self._vq_req_buf: torch.Tensor = None  # type: ignore[assignment]
        self._vq_seqlen_buf: torch.Tensor = None  # type: ignore[assignment]
        self._vq_req_host: torch.Tensor = None  # type: ignore[assignment]
        self._vq_seqlen_host: torch.Tensor = None  # type: ignore[assignment]

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
        # Use the framework's cached CPU copy of seq_lens (cam.seq_lens_cpu) to
        # avoid an extra GPU->CPU sync per step (build-overhead reduction).
        _slc = getattr(cam, "seq_lens_cpu", None)
        seq_lens_cpu = _slc.tolist() if _slc is not None else cam.seq_lens.tolist()
        # Per-request query length this step (already on CPU; no extra sync).
        # query_len > 1 for prefill chunks and for speculative-decode verify
        # steps (MTP / draft). Used by flush detection to compute the COMMITTED
        # token count (seq_len - query_len) so speculative tokens that may still
        # be rejected are never quantized into the permanent int4 cache.
        _qsl = getattr(cam, "query_start_loc_cpu", None)
        if _qsl is not None:
            _qsl_l = _qsl.tolist()
            query_lens_cpu = [_qsl_l[i + 1] - _qsl_l[i] for i in range(len(_qsl_l) - 1)]
        else:
            query_lens_cpu = [1] * len(seq_lens_cpu)
        # block_table as a numpy 2-D array (C-backed, lazy element access) rather
        # than .tolist(): the full B×max_blocks nested-list build was ~7 ms/step
        # at B=256 and dominated build() once the flush was vectorized.
        # We only touch column 0 (sinks) + a few per-request entries, so numpy's
        # O(1) indexing avoids materializing ~8k Python ints every step.
        block_table_np = cam.block_table_tensor.cpu().numpy()
        slot_mapping_cpu = cam.slot_mapping.tolist()
        bt_rows = block_table_np.shape[0]
        bt_cols = block_table_np.shape[1] if block_table_np.ndim == 2 else 0
        device = cam.seq_lens.device

        # ── Stage α-2: capture-correct metadata ──────────────────────────
        # The decode driver uses ONE block_table-driven kernel that reads the
        # PERSISTENT block_table / seq_lens / cu_seqlens directly, so no
        # per-step derived task tensors (which would be stale under graph
        # replay). We only need cu_seqlens_k (prefix sum of seq_lens) and
        # cu_seqlens_q (= arange(B+1)), both kept in PERSISTENT buffers and
        # updated in place so captured graphs see fresh values.
        B = len(seq_lens_cpu)
        GROUP = self._group  # KVarN tile size (= block size); 64 or 128
        cu_seqlens_k_h = [0]
        for sl in seq_lens_cpu:
            cu_seqlens_k_h.append(cu_seqlens_k_h[-1] + sl)

        # ── Stage α-2: assign pool slots for every block_id touched this
        # step. The allocator state is class-level on KVarNAttentionImpl
        # and we mutate it here (in the builder, outside any captured
        # region). do_kv_cache_update then only READS block_to_slot_t.
        from vllm.v1.attention.backends.kvarn_attn import (
            KVarNAttentionImpl,
        )  # local import

        # Pool slots are needed ONLY for blocks that physically live in the fp16
        # tail pool: each request's sink (block_table[r][0], kept fp16 for the
        # request's lifetime) and the blocks receiving writes THIS step —
        # tokens committed..seq_len-1 land in do_kv_cache_update after the
        # builder. Flushed history blocks live in the int4 cache, carry
        # pool_slot=-1, and are dequantized in-kernel.
        #
        # Sharing-safe lifecycle (prefix caching + chunked prefill + spec
        # decode): everything below derives from per-step facts —
        #   committed = seq_len - query_len   (tokens written BEFORE this step)
        #   dict_map membership = "block is unflushed" (ground truth: a flush
        #     frees the slot, so a slot-holding block below the committed
        #     boundary is exactly a full-but-unflushed block)
        #   _block_fill[bid] = tokens the pool holds for bid after this step
        # A cache-hit request's context blocks receive no writes, so they
        # correctly need no slots (their tiles are already int4). Anything in
        # the allocator NOT needed this step belongs to a finished request and
        # is reclaimed below: complete blocks are FLUSHED (a future prefix-
        # cache hit must find a valid int4 tile), partial ones discarded.
        blocks_needed: set[int] = set()
        for b in range(B):
            if b >= bt_rows:
                break
            sl = seq_lens_cpu[b]
            if bt_cols == 0 or sl <= 0:
                continue
            row = block_table_np[b]
            q_len = query_lens_cpu[b] if b < len(query_lens_cpu) else 1
            committed = max(sl - q_len, 0)
            # Blocks written this step. Record how full each will be AFTER the
            # step: if its owner finishes on the step that fills it, the
            # reclaim below must flush it (not discard).
            for k in range(committed // GROUP, min((sl - 1) // GROUP, bt_cols - 1) + 1):
                bid = int(row[k])
                if bid >= 0:
                    blocks_needed.add(bid)
                    self._block_fill[bid] = min(sl, (k + 1) * GROUP) - k * GROUP
        for s in slot_mapping_cpu:  # safety superset of the above
            if s >= 0:
                blocks_needed.add(s // GROUP)

        gk = self._group_key
        # Claim THIS group's impls by layer name (set on the impl in
        # Attention.__init__) and tag them with the true group key, so their
        # _ensure_pool / store paths use this group's slot allocator + mirror.
        group_impls = [
            impl
            for impl in KVarNAttentionImpl._all_impls
            if getattr(impl, "layer_name", None) in self._layer_names_set
        ]
        for impl in group_impls:
            impl._group_key = gk
        if group_impls:
            impl0 = group_impls[0]
            # Ensure pool + lookup tensors exist for this device.
            impl0._ensure_pool(
                device, num_blocks_hint=max(blocks_needed, default=0) + 1
            )
            mkey = (device, gk)
            b2s_t = KVarNAttentionImpl._block_to_slot_t_per_device[mkey]
            is_sink_t = KVarNAttentionImpl._is_sink_t_per_device[mkey]
            dict_map = KVarNAttentionImpl._block_to_slot_dict[gk]
            free_slots = KVarNAttentionImpl._free_slots[gk]
            sinks = KVarNAttentionImpl._global_sink_blocks[gk]

            # ORDER MATTERS: mark sinks → FLUSH (frees just-completed blocks'
            # slots) → ALLOCATE (the new tails, reusing the freed slots). Doing
            # the flush before allocation caps the live-slot peak at 2·B
            # (one sink + one in-progress tail per request). Allocating first
            # would transiently need 3·B when every request crosses a block
            # boundary in lockstep (sink + pending-flush full block + new tail)
            # → "pool exhausted" at large batch.

            # (1) Mark per-request sink blocks (block_table[r][0]). A block is
            # an fp16 sink only while its data lives in the pool: a fresh
            # prefill writes block 0 this step (it is in blocks_needed) and
            # keeps it fp16 for the request's lifetime; an existing sink keeps
            # its slot via blocks_needed. A prefix-cache-hit request whose
            # block 0 was already reclaimed (flushed to int4) must NOT re-mark
            # it: its data lives in the int4 tile (slot -1) and every kernel
            # reads it there like any history block. Re-marking would allocate
            # an EMPTY pool slot that is never written (cache hits skip those
            # tokens) and attention would read garbage for the whole first
            # block — repetition-loops on multi-turn chat.
            row0_set: set[int] = set()
            for b in range(B):
                if b >= bt_rows or bt_cols == 0:
                    break
                s0 = int(block_table_np[b, 0])
                if s0 < 0:
                    continue
                row0_set.add(s0)
                if s0 in sinks:
                    blocks_needed.add(s0)  # live/retired sink keeps its slot
                elif s0 in blocks_needed:  # written this step → fresh sink
                    sinks.add(s0)
                    if s0 < is_sink_t.shape[0]:
                        is_sink_t[s0] = True

            # Un-retire any retired block named this step: a prefix-cache hit
            # re-adopting a retired sink (its fp16 data is intact and byte-
            # identical), or vLLM recycling the id for a fresh write. Either
            # way the block is live again and must not be evicted under it.
            # A recycled block that is no request's first block sheds its
            # stale sink label so the normal walk-back flush applies to it.
            for bid in [b for b in self._retired_sinks if b in blocks_needed]:
                self._retired_sinks.pop(bid, None)
                if bid not in row0_set and bid in sinks:
                    sinks.discard(bid)
                    if bid < is_sink_t.shape[0]:
                        is_sink_t[bid] = False

            # (2) Flush detection (Stage α-2 Step B).
            # CRITICAL timing: token (k+1)*GROUP-1 (the one that completes
            # block k) is written during THIS step's do_kv_cache_update, which
            # runs AFTER the builder. So at builder time the pool only holds
            # tokens already committed before this step. That committed count is
            # `seq_len - query_len` (this step's query tokens are written later),
            # i.e. exactly num_computed_tokens. We flush against THAT, never the
            # full `sl`.
            #
            # Why not the full `sl` (or the previous step's `sl`): under
            # speculative decoding (MTP / draft) a step appends `num_spec+1`
            # tokens at once and seq_len jumps by a VARIABLE accepted amount,
            # with later-rejected speculative tokens sitting in the pool until
            # they are overwritten next step. Quantizing a block to int4 is
            # PERMANENT, so flushing a block that still contains a speculative
            # (rejectable) token freezes wrong KV → progressive corruption →
            # repetition-collapse / garbage. Using the committed length means we
            # only ever quantize blocks whose tokens are all accepted.
            #
            # Walk each row BACKWARD from the committed boundary while blocks
            # still hold pool slots — those are exactly the full-but-unflushed
            # blocks. The walk stops at the first slotless block (flushes
            # happen in order, so everything earlier is already int4) and never
            # touches k=0 (a live request's sink stays fp16; finished requests'
            # sinks are handled by the reclaim below). Idempotent under prefix
            # sharing: a co-owner finds the block already queued (or slotless)
            # and stops — no per-request state to collide.
            flush_block_ids: list[int] = []
            flush_seen: set[int] = set()
            for b in range(B):
                if b >= bt_rows or bt_cols == 0:
                    break
                sl = seq_lens_cpu[b]
                row = block_table_np[b]
                if sl <= 0:
                    continue
                q_len = query_lens_cpu[b] if b < len(query_lens_cpu) else 1
                committed_len = max(sl - q_len, 0)  # tokens already in pool & accepted
                k = min(committed_len // GROUP - 1, bt_cols - 1)
                while k >= 1:
                    bid = int(row[k])
                    if (
                        bid < 0
                        or bid in flush_seen
                        or bid in sinks
                        or bid not in dict_map
                    ):
                        break
                    flush_seen.add(bid)
                    flush_block_ids.append(bid)
                    k -= 1

            # (2b) Reclaim slot-holding blocks neither written this step nor
            # queued above: they belong to finished (or preempted) requests.
            # A COMPLETE sink is RETIRED — kept fp16-resident so a prefix-
            # cache hit (every follow-up chat turn) re-adopts it byte-
            # identically; the old discard destroyed its fp16-only data
            # outright, garbling every multi-turn cache hit.
            # Any other COMPLETE block is FLUSHED — vLLM's prefix cache may
            # hand it to a future request, which must find a valid int4 tile
            # (the old discard left stale tile bytes). A PARTIAL block is
            # discarded: vLLM never prefix-caches partial blocks.
            discard_ids: list[int] = []
            for bid in [
                b for b in dict_map if b not in blocks_needed and b not in flush_seen
            ]:
                full = self._block_fill.get(bid, 0) >= GROUP
                if full and bid in sinks:
                    self._retired_sinks[bid] = None  # idempotent re-insert
                    continue
                if full:
                    flush_seen.add(bid)
                    flush_block_ids.append(bid)
                else:
                    discard_ids.append(bid)
                if bid in sinks:  # finished request's partial sink
                    sinks.discard(bid)
                    if bid < is_sink_t.shape[0]:
                        is_sink_t[bid] = False

            # Trigger the flush on every layer's pool. Each impl quantises its
            # own pool[slot] into its own kv_cache (ref cached on first
            # forward), then frees the slot below. Runs eagerly here, before
            # the captured forward replay.
            if flush_block_ids:
                # One batched Sinkhorn + RTN over ALL (layer, block) flush tiles
                # — replaces 48×N_blocks individual launches. Numerically
                # identical (per-tile-independent ops) → no accuracy change.
                flush_pairs = []
                for impl in group_impls:
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
                    self._block_fill.pop(bid, None)
                    if slot is not None:
                        free_slots.append(slot)
                        if bid < b2s_t.shape[0]:
                            b2s_t[bid] = -1

            # Free the discarded (partial, never-cacheable) blocks' slots.
            for bid in discard_ids:
                slot = dict_map.pop(bid)
                self._block_fill.pop(bid, None)
                free_slots.append(slot)
                if bid < b2s_t.shape[0]:
                    b2s_t[bid] = -1

            # (3) Allocate slots for any new block_ids (sinks + new tails).
            for bid in blocks_needed:
                if bid not in dict_map:
                    if not free_slots and self._retired_sinks:
                        # Evict the oldest retired sink: flush it to int4 so a
                        # later prefix-cache hit still finds a valid tile, then
                        # hand its slot to the live allocation.
                        old = next(iter(self._retired_sinks))
                        self._retired_sinks.pop(old)
                        evict_pairs = [
                            (impl, old, impl._kv_cache_ref)
                            for impl in group_impls
                            if getattr(impl, "_kv_cache_ref", None) is not None
                        ]
                        KVarNAttentionImpl._batched_flush(evict_pairs)
                        old_slot = dict_map.pop(old, None)
                        self._block_fill.pop(old, None)
                        sinks.discard(old)
                        if old < is_sink_t.shape[0]:
                            is_sink_t[old] = False
                        if old < b2s_t.shape[0]:
                            b2s_t[old] = -1
                        if old_slot is not None:
                            free_slots.append(old_slot)
                    if not free_slots:
                        raise RuntimeError(
                            f"KVarN pool exhausted "
                            f"({KVarNAttentionImpl._allocator_pool_size.get(gk)} slots)"
                        )
                    slot = free_slots.pop()
                    dict_map[bid] = slot
                    if bid < b2s_t.shape[0]:
                        b2s_t[bid] = slot
                    KVarNAttentionImpl._max_known_block_id[gk] = max(
                        KVarNAttentionImpl._max_known_block_id.get(gk, 0), bid
                    )

        # ── Persistent cu_seqlens buffers (in-place updated) ─────────────
        # A captured graph bakes in tensor addresses, so cu_seqlens MUST live
        # in fixed buffers updated in place — not recreated each step.
        cap = B + 1
        if self._cu_seqlens_q_buf is None or self._cu_seqlens_q_buf.shape[0] < cap:
            new_cap = max(cap, 257)  # default max_num_seqs headroom
            self._cu_seqlens_q_buf = torch.empty(
                new_cap, dtype=torch.int32, device=device
            )
            self._cu_seqlens_k_buf = torch.empty(
                new_cap, dtype=torch.int32, device=device
            )
            self._cu_seqlens_q_host = torch.empty(
                new_cap, dtype=torch.int32, pin_memory=True
            )
            self._cu_seqlens_k_host = torch.empty(
                new_cap, dtype=torch.int32, pin_memory=True
            )
        for i in range(B + 1):
            self._cu_seqlens_q_host[i] = i
            self._cu_seqlens_k_host[i] = cu_seqlens_k_h[i]
        fa_cu_seqlens_q = self._cu_seqlens_q_buf[: B + 1]
        fa_cu_seqlens_k = self._cu_seqlens_k_buf[: B + 1]
        fa_cu_seqlens_q.copy_(self._cu_seqlens_q_host[: B + 1], non_blocking=True)
        fa_cu_seqlens_k.copy_(self._cu_seqlens_k_host[: B + 1], non_blocking=True)

        # ── Verify (spec-as-decode) plan ─────────────────────────────────
        # When the decode portion carries multi-token queries (an MTP verify
        # step; query_len <= reorder threshold), build one virtual kernel row
        # per decode token: its block-table row and its bottom-right causal
        # length (committed + idx + 1). Persistent buffers, CPU-filled here,
        # so the fused verify kernel is CUDA-graph-capturable (pointers stay
        # stable across replays; only values change).
        vq_req_t = vq_seqlen_t = None
        vq_qlen = 0
        if num_decodes > 0 and num_decode_tokens > num_decodes:
            if (
                self._vq_req_buf is None
                or self._vq_req_buf.shape[0] < num_decode_tokens
            ):
                vq_cap = max(num_decode_tokens, 4096)
                self._vq_req_buf = torch.empty(vq_cap, dtype=torch.int32, device=device)
                self._vq_seqlen_buf = torch.empty(
                    vq_cap, dtype=torch.int32, device=device
                )
                self._vq_req_host = torch.empty(
                    vq_cap, dtype=torch.int32, pin_memory=True
                )
                self._vq_seqlen_host = torch.empty(
                    vq_cap, dtype=torch.int32, pin_memory=True
                )
            # Non-causal (DFlash cross-attention): every query row attends to
            # the full context, so its per-row limit is the whole seq_len, not
            # the bottom-right causal staircase committed+j+1.
            non_causal = not getattr(cam, "causal", True)
            i = 0
            uniform = query_lens_cpu[0] if num_decodes else 0
            for b in range(num_decodes):
                ql = query_lens_cpu[b] if b < len(query_lens_cpu) else 1
                if ql != uniform:
                    uniform = 0
                committed = max(seq_lens_cpu[b] - ql, 0)
                full = committed + ql
                for j in range(ql):
                    self._vq_req_host[i] = b
                    self._vq_seqlen_host[i] = full if non_causal else committed + j + 1
                    i += 1
            # Uniform query length -> the shared-dequant verify kernel (the
            # request's tokens share each block's dequant); this is always the
            # case under uniform-batch graph capture. The shared kernel bakes
            # the bottom-right causal staircase internally, so non-causal must
            # fall back to the per-token path (which honours the flat per-row
            # limit above) — force vq_qlen=0 in that case.
            vq_qlen = uniform if (uniform >= 2 and not non_causal) else 0
            vq_req_t = self._vq_req_buf[:num_decode_tokens]
            vq_seqlen_t = self._vq_seqlen_buf[:num_decode_tokens]
            vq_req_t.copy_(self._vq_req_host[:num_decode_tokens], non_blocking=True)
            vq_seqlen_t.copy_(
                self._vq_seqlen_host[:num_decode_tokens], non_blocking=True
            )

        max_blocks_per_req = (self._max_model_len + GROUP - 1) // GROUP

        # A multi-query request with cached context (seq_len > query_len) is a
        # speculative-decode verify step or a chunked-prefill continuation —
        # its query tokens must attend over the cached K/V, not just each other.
        # Detected here from CPU arrays (no GPU sync) so forward() can route it
        # to the context-aware path. Fresh first-chunk prefill has
        # seq_len == query_len on every row → flag stays False.
        has_cached_multiquery = any(
            query_lens_cpu[b] > 1 and seq_lens_cpu[b] > query_lens_cpu[b]
            for b in range(min(B, len(query_lens_cpu)))
        )

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
            has_cached_multiquery=has_cached_multiquery,
            seq_lens_cpu=seq_lens_cpu,
            # not consumed downstream; build() uses block_table_np
            block_table_cpu=None,
            slot_mapping_cpu=slot_mapping_cpu,
            fa_cu_seqlens_q=fa_cu_seqlens_q,
            fa_cu_seqlens_k=fa_cu_seqlens_k,
            fa_max_blocks_per_req=max_blocks_per_req,
            fa_max_seqlen_k_fixed=self._max_model_len,
            vq_req=vq_req_t,
            vq_seqlen=vq_seqlen_t,
            vq_qlen=vq_qlen,
            causal=getattr(cam, "causal", True),
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
    filled_count: int = 0  # CPU-side counter (avoid .all() sync)


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
    _shared_mid_o_buf: ClassVar[
        dict[torch.device, torch.Tensor]
    ] = {}  # split-K partials
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
    # Allocator state, scoped PER KV-CACHE-GROUP (key = group_key tuple), because
    # block_ids are only unique WITHIN a group. CPU dicts keyed by group_key; GPU
    # mirrors keyed by (device, group_key). See `self._group_key`.
    _block_to_slot_dict: ClassVar[dict[tuple, dict[int, int]]] = {}
    _global_sink_blocks: ClassVar[dict[tuple, set[int]]] = {}
    _free_slots: ClassVar[dict[tuple, list[int]]] = {}
    _allocator_pool_size: ClassVar[dict[tuple, int]] = {}
    _block_to_slot_t_per_device: ClassVar[dict[tuple, torch.Tensor]] = {}
    _is_sink_t_per_device: ClassVar[dict[tuple, torch.Tensor]] = {}
    _max_known_block_id: ClassVar[dict[tuple, int]] = {}
    # Keys (device, D, group, k_bits, v_bits) whose flush kernels (Sinkhorn +
    # int4 store) have already been JIT-compiled via the pool-init warmup.
    _kernel_warmed: ClassVar[set] = set()

    # Registry of impls so the builder can enumerate per-layer pools when
    # it needs to update sink markers / trigger flushes.
    _all_impls: ClassVar[list[KVarNAttentionImpl]] = []
    _tiles_dumped: ClassVar[bool] = False

    @classmethod
    def _impls_for_group(cls, group_key: tuple) -> list[KVarNAttentionImpl]:
        """Impls belonging to one KV-cache group (same group_key)."""
        return [i for i in cls._all_impls if i._group_key == group_key]

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
        # Sliding-window layers (e.g. Gemma-4: 50/60 layers, window 1024) only
        # attend to the last `sliding_window` keys. Stored so the decode kernel
        # can bound its block loop to the window — without this it reads the FULL
        # history every step (16x too much work + wrong output past the window).
        self.sliding_window = sliding_window or 0
        # KV-cache-group key. KVarN's slot allocator + GPU mirrors are keyed by
        # block_id, but vLLM gives each KV-cache group an INDEPENDENT block_id
        # space. Heterogeneous models put KVarN layers in >1 group (e.g. Gemma-4:
        # sliding head256/16kv + global head512/4kv), so a single global allocator
        # aliases the two groups' block_ids -> wrong slots -> garbage. Scope all
        # allocator state by this key so each group has its own slot space.
        # (head_size, num_kv_heads, sliding_window) uniquely identifies the group
        # and is computable identically by the per-group builder and each impl.
        self._group_key = (head_size, self.num_kv_heads, self.sliding_window)
        if os.environ.get("KVARN_DBG_LAYERS") == "1":
            print(
                f"[KVARN_LAYER] head_size={head_size} num_heads={num_heads} "
                f"num_kv_heads={self.num_kv_heads} "
                f"sliding_window={self.sliding_window}",
                flush=True,
            )

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
        # TODO: wire to vLLM's request-completion hook for eviction.
        self._sink_blocks: set[int] = set()

        # ── Stage α-2: deterministic per-block tail pool ─────────────────────
        # Each block_id maps to slot = block_id in the pool (no allocator,
        # no dict). Pool is sized to kv_cache.shape[0] = num_blocks at first
        # `_ensure_pool` call. Sink blocks stay in the pool permanently;
        # non-sink blocks have their slot's content quantised into the int4
        # cache at tile-boundary flushes (triggered from the metadata
        # builder, between captured graph replays).
        # [POOL_SIZE, group, Hk, D] fp16
        self._tail_K_pool: torch.Tensor = None  # type: ignore[assignment]
        self._tail_V_pool: torch.Tensor = None  # type: ignore[assignment]
        # Per-instance shorthand views of the class-level per-device tensors
        # (so kernels can read without dict lookups). Re-bound on every
        # _ensure_pool call.
        self._is_sink_t: torch.Tensor | None = None  # [num_blocks] bool
        self._block_to_slot_t: torch.Tensor | None = None  # [num_blocks] int32
        self._block_lookup_size: int = 0

        # Cached fp16 Hadamard for the rotate-on-store matmul in
        # do_kv_cache_update (avoids a per-call .float() cast that allocates).
        self._H_fp16: torch.Tensor | None = None

        # Store-side rotation scratch (pre-allocated by _ensure_pool so the
        # captured forward never allocates). Shapes:
        #   _k_rot_scratch  [max_num_batched_tokens, Hk, D] fp16
        #   _v_rot_scratch  [max_num_batched_tokens, Hk, D] fp16
        self._k_rot_scratch: torch.Tensor = None  # type: ignore[assignment]
        self._v_rot_scratch: torch.Tensor = None  # type: ignore[assignment]

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
        self._fa_K_buf: torch.Tensor = None  # type: ignore[assignment]
        self._fa_V_buf: torch.Tensor = None  # type: ignore[assignment]

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
            self._num_hidden_layers = getattr(
                _cfg.model_config.hf_config, "num_hidden_layers", 32
            )
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
                pool_size,
                cfg.group,
                self.num_kv_heads,
                cfg.head_dim,
                dtype=torch.float16,
                device=device,
            )
            self._tail_V_pool = torch.zeros_like(self._tail_K_pool)
        else:
            pool_size = self._tail_K_pool.shape[0]

        # Per-GROUP allocator state — ensure it exists for THIS group_key.
        # Decoupled from the per-impl pool allocation above: the impl's
        # _group_key is set to the proxy in __init__ and later RE-TAGGED to the
        # true (per-group) key by the builder, so the pool may already exist
        # under a stale key when this group_key is first seen. Idempotent.
        gk = self._group_key
        if gk not in cls._free_slots:
            cls._free_slots[gk] = list(range(pool_size - 1, -1, -1))
            cls._allocator_pool_size[gk] = pool_size
            cls._block_to_slot_dict[gk] = {}
            cls._global_sink_blocks[gk] = set()

        # GPU lookup tensors, keyed by (device, group_key): each KV-cache group
        # has its own block_id space, so the two groups must NOT share a mirror.
        gk = self._group_key
        mkey = (device, gk)
        num_blocks = max(num_blocks_hint, cls._max_known_block_id.get(gk, 0) + 1, 1024)
        existing = cls._block_to_slot_t_per_device.get(mkey)
        if existing is None or existing.shape[0] < num_blocks:
            new_b2s = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
            new_is_sink = torch.zeros(num_blocks, dtype=torch.bool, device=device)
            # Re-sync from this group's CPU state (rare, only on resize / first init).
            for bid, slot in cls._block_to_slot_dict.get(gk, {}).items():
                if bid < num_blocks:
                    new_b2s[bid] = slot
            for bid in cls._global_sink_blocks.get(gk, set()):
                if bid < num_blocks:
                    new_is_sink[bid] = True
            cls._block_to_slot_t_per_device[mkey] = new_b2s
            cls._is_sink_t_per_device[mkey] = new_is_sink
        # Per-instance shorthand pointers so the decode driver / kernels read
        # without dict lookups in the hot path.
        self._is_sink_t = cls._is_sink_t_per_device[mkey]
        self._block_to_slot_t = cls._block_to_slot_t_per_device[mkey]
        self._block_lookup_size = self._block_to_slot_t.shape[0]

        # Cached fp16 Hadamard for the rotate-on-store matmul.
        if self._H_fp16 is None:
            self._H_fp16 = self._hadamard(device).to(torch.float16).contiguous()

        # One-time flush-kernel warmup. The Sinkhorn + int4-store
        # kernels are exercised ONLY at a tile-boundary flush, which never
        # happens during vLLM's profiling/dummy run (no request crosses a block
        # boundary there). So they JIT-compile on the FIRST real flush DURING
        # serving — a multi-hundred-ms stall that surfaces as a latency spike
        # and a `jit_monitor` "JIT compilation during inference" warning, and
        # disproportionately hurts low-concurrency aggregate throughput (the
        # one-time cost lands inside a small measured window). Compile them here,
        # once per shape/config, at pool-init time (outside any captured region)
        # using the exact tile shapes the flush uses, so serving never pays it.
        warm_key = (device, cfg.head_dim, cfg.group, cfg.key_bits, cfg.value_bits)
        if warm_key not in cls._kernel_warmed:
            k_dummy = torch.zeros(
                1, cfg.head_dim, cfg.group, dtype=torch.float16, device=device
            )
            v_dummy = torch.zeros(
                1, cfg.group, cfg.head_dim, dtype=torch.float16, device=device
            )
            _sinkhorn_pack_kv(k_dummy, v_dummy, cfg)
            cls._kernel_warmed.add(warm_key)

        # Decode-kernel warmup. The DECODE kernels (fused
        # single-stage incl. its @triton.autotune sweep, split-K stage1/2, and
        # the packed-KV build kernel) never run during vLLM's prefill-shaped
        # profiling, so their one-time JIT + autotune cost (including the
        # autotuner's benchmark scratch) used to land in the FIRST real decode —
        # which, since v0.21, is the CUDA-graph memory estimation warmup. The
        # estimate then absorbed those one-time costs and over-charged "graph
        # memory" by GiBs, directly shrinking the derived KV-cache capacity.
        # Warm them here (profile time) on tiny synthetic state instead: the
        # cost is charged once to the memory profile, and the graph estimate
        # measures only real graph-pool memory. Keyed per (device, shape combo).
        dec_key = (
            "decode",
            device,
            cfg.head_dim,
            cfg.group,
            cfg.key_bits,
            cfg.value_bits,
            self.num_heads,
            self.num_kv_heads,
            int(getattr(self, "sliding_window", 0) or 0),
        )
        if dec_key not in cls._kernel_warmed:
            self._warm_decode_kernels(device)
            cls._kernel_warmed.add(dec_key)

        # Store-side rotation scratch.
        if self._k_rot_scratch is None:
            q_rows = max(self._max_num_batched_tokens, 1)
            self._k_rot_scratch = torch.empty(
                q_rows,
                self.num_kv_heads,
                cfg.head_dim,
                dtype=torch.float16,
                device=device,
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
        fa_rows = max(
            min(self._max_num_seqs * self._max_model_len, FA_SCRATCH_CAP),
            self._max_model_len,
            4096,
        )
        cls = type(self)
        # Key the shared decode scratch by (device, D, Hk), NOT device alone:
        # heterogeneous-head models (e.g. Gemma-4: 256-dim/16-kv sliding layers +
        # 512-dim/4-kv global layers) have multiple (head_dim, kv_heads) combos,
        # and a buffer sized for one combo's D/Hk is the wrong width for another
        # (caused a reshape(N,512)-on-256-wide-buffer crash). One scratch set per
        # combo (Gemma-4 = 2 sets; cost is small).
        bkey = (device, D, Hk)
        if bkey not in cls._shared_q_fp32_buf:
            cls._shared_q_fp32_buf[bkey] = torch.empty(
                q_rows, D, dtype=torch.float32, device=device
            )
            cls._shared_q_rot_fp32_buf[bkey] = torch.empty(
                q_rows, D, dtype=torch.float32, device=device
            )
            cls._shared_q_rot_fp16_buf[bkey] = torch.empty(
                q_rows, D, dtype=torch.float16, device=device
            )
            cls._shared_out_rot_fp32_buf[bkey] = torch.empty(
                q_rows, D, dtype=torch.float32, device=device
            )
            cls._shared_output_fp32_buf[bkey] = torch.empty(
                q_rows, D, dtype=torch.float32, device=device
            )
            cls._shared_fused_out_buf[bkey] = torch.empty(
                q_rows, D, dtype=torch.float16, device=device
            )
        from vllm.v1.attention.ops.triton_kvarn_decode import adaptive_num_kv_splits

        # Split-K partial buffers, sized to EXACTLY what the split-K decode path
        # can index: it runs ONLY on pure single-query decode steps, whose row
        # count is N = B*Hq with B <= max_num_seqs — NOT q_rows (which is
        # max_num_batched_tokens-driven and sized the buffer ~85x too big at
        # typical configs: 256 MiB instead of ~3 MiB for max_num_seqs=2/Hq=24/
        # 64 splits). Split count matches the driver's
        # adaptive helper (same max_model_len) or a larger adaptive count would
        # overflow a smaller buffer; the driver additionally falls back to the
        # single-stage kernel if N ever exceeds the buffer rows (defensive —
        # e.g. an oversized padded dummy batch).
        _splits = adaptive_num_kv_splits(
            (self._max_model_len + cfg.group - 1) // cfg.group
        )
        # Rows are bounded by the split-K REGIME, not max_num_seqs: the driver
        # only takes split-K when B*Hk <= sm_count (otherwise the single-stage
        # kernel runs), so the most rows it can ever index is (sm_count//Hk)*Hq
        # = sm_count*Q_PER_KV, independent of max_num_seqs. Sizing to
        # max_num_seqs*Hq over-reserved this fp32 partial buffer ~10-20x at high
        # concurrency, where it competes directly with the int4 KV cache. The
        # driver still falls back to single-stage if N ever exceeds these rows
        # (defensive), and split-K is never disabled for a batch it would take
        # (B*Hk<=sm_count => N=B*Hq <= (sm_count//Hk)*Hq).
        _sm = (
            getattr(self, "_sm_count", 0)
            or torch.cuda.get_device_properties(device).multi_processor_count
        )
        mid_rows = max((_sm // max(Hk, 1)) * Hq, Hq, 1)
        _ex_mid = cls._shared_mid_o_buf.get(bkey)
        if (
            _ex_mid is None
            or _ex_mid.shape[0] < mid_rows
            or _ex_mid.shape[1] != _splits
        ):
            cls._shared_mid_o_buf[bkey] = torch.empty(
                mid_rows, _splits, D, dtype=torch.float32, device=device
            )
            cls._shared_mid_lse_buf[bkey] = torch.empty(
                mid_rows, _splits, dtype=torch.float32, device=device
            )
        if (
            bkey not in cls._shared_fa_K_buf
            or cls._shared_fa_K_buf[bkey].shape[0] < fa_rows
        ):
            cls._shared_fa_K_buf[bkey] = torch.zeros(
                fa_rows, Hk, D, dtype=torch.float16, device=device
            )
            cls._shared_fa_V_buf[bkey] = torch.zeros_like(cls._shared_fa_K_buf[bkey])
        # Mirror to instance attrs for fast access by the decode driver.
        self._q_fp32_buf = cls._shared_q_fp32_buf[bkey]
        self._q_rot_fp32_buf = cls._shared_q_rot_fp32_buf[bkey]
        self._q_rot_fp16_buf = cls._shared_q_rot_fp16_buf[bkey]
        self._out_rot_fp32_buf = cls._shared_out_rot_fp32_buf[bkey]
        self._output_fp32_buf = cls._shared_output_fp32_buf[bkey]
        self._fused_out_buf = cls._shared_fused_out_buf[bkey]
        self._mid_o_buf = cls._shared_mid_o_buf[bkey]
        self._mid_lse_buf = cls._shared_mid_lse_buf[bkey]
        self._fa_K_buf = cls._shared_fa_K_buf[bkey]
        self._fa_V_buf = cls._shared_fa_V_buf[bkey]

    def _warm_decode_kernels(self, device: torch.device) -> None:
        """Compile + autotune every decode-path Triton kernel on tiny synthetic
        state (see the note at the call site in ``_ensure_pool``).
        Uses throwaway tensors only — never touches the real cache/pool."""
        from vllm.v1.attention.ops.triton_kvarn_decode import (
            _kvarn_build_packed_kv_kernel,
            _kvarn_fused_decode_kernel,
            _kvarn_fused_decode_stage1,
            _kvarn_fused_decode_stage2,
            adaptive_num_kv_splits,
        )

        cfg = self.kvarn_config
        D, G = cfg.head_dim, cfg.group
        Hq, Hk = self.num_heads, self.num_kv_heads
        B, n_blocks = 8, 4
        sw = int(getattr(self, "sliding_window", 0) or 0)

        cache = torch.zeros(
            B * n_blocks, Hk, cfg.tile_bytes_aligned, dtype=torch.uint8, device=device
        )
        pool_k = torch.zeros(1, G, Hk, D, dtype=torch.float16, device=device)
        pool_v = torch.zeros_like(pool_k)
        b2s = torch.full((B * n_blocks,), -1, dtype=torch.int32, device=device)
        bt = torch.arange(B * n_blocks, dtype=torch.int32, device=device).view(
            B, n_blocks
        )
        sl = torch.full((B,), n_blocks * G, dtype=torch.int32, device=device)
        q = torch.zeros(B, Hq, D, dtype=torch.float16, device=device)
        out = torch.zeros_like(q)

        qpk = Hq // Hk
        qpk_pad = 1 << (qpk - 1).bit_length() if qpk > 1 else 1
        common = dict(
            MAX_BLOCKS_PER_REQ=n_blocks,
            D=D,
            GROUP=G,
            Q_PER_KV=qpk,
            Q_PER_KV_PAD=qpk_pad,
            SLIDING_WINDOW=sw,
            K_BITS=cfg.key_bits,
            V_BITS=cfg.value_bits,
            NUM_BLOCKS_LOOKUP=B * n_blocks,
            K_PACKED_OFFSET=cfg.k_packed_offset,
            K_S_COL_OFFSET=cfg.k_s_col_offset,
            K_ZP_OFFSET=cfg.k_zp_offset,
            K_S_ROW_OFFSET=cfg.k_s_row_offset,
            V_PACKED_OFFSET=cfg.v_packed_offset,
            V_S_COL_OFFSET=cfg.v_s_col_offset,
            V_S_ROW_OFFSET=cfg.v_s_row_offset,
            V_ZP_OFFSET=cfg.v_zp_offset,
            VQ_INDIRECT=False,
        )
        # 1. Single-stage fused kernel — runs the @triton.autotune sweep.
        # (sl doubles as the unused Req_row_ptr dummy; see VQ_INDIRECT.)
        _kvarn_fused_decode_kernel[(B, Hk)](
            q,
            sl,
            bt,
            sl,
            b2s,
            cache,
            pool_k,
            pool_v,
            out,
            self.scale,
            Hq * D,
            D,
            bt.stride(0),
            cache.stride(0),
            cache.stride(1),
            pool_k.stride(0),
            pool_k.stride(1),
            pool_k.stride(2),
            Hq * D,
            D,
            **common,
        )
        # 2. Split-K stage1 + stage2, with the exact split count and launch
        # knobs the decode driver will use for this deployment.
        splits = adaptive_num_kv_splits((self._max_model_len + G - 1) // G)
        mid_o = torch.zeros(B * Hq, splits, D, dtype=torch.float32, device=device)
        mid_lse = torch.zeros(B * Hq, splits, dtype=torch.float32, device=device)
        # stage1 is @triton.autotune'd; this warmup launch triggers its sweep
        # here (pre-CUDA-graph-capture) so capture never benchmarks.
        _kvarn_fused_decode_stage1[(B, Hk, splits)](
            q,
            sl,
            bt,
            sl,
            b2s,
            cache,
            pool_k,
            pool_v,
            mid_o,
            mid_lse,
            self.scale,
            Hq * D,
            D,
            bt.stride(0),
            cache.stride(0),
            cache.stride(1),
            pool_k.stride(0),
            pool_k.stride(1),
            pool_k.stride(2),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_lse.stride(0),
            NUM_KV_SPLITS=splits,
            HQ=Hq,
            **common,
        )
        out2d = out.view(B * Hq, D)
        _kvarn_fused_decode_stage2[(B * Hq,)](
            mid_o,
            mid_lse,
            out2d,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_lse.stride(0),
            out2d.stride(0),
            D=D,
            NUM_KV_SPLITS=splits,
            num_warps=2,
        )
        # 2b. VQ_INDIRECT (fused spec-verify) specializations — separate
        # compiled variants; warm them so the FIRST MTP verify step doesn't
        # pay the Triton JIT mid-serving.
        vq_rows = torch.zeros(B, dtype=torch.int32, device=device)
        common_vq = dict(common, VQ_INDIRECT=True)
        _kvarn_fused_decode_kernel[(B, Hk)](
            q,
            vq_rows,
            bt,
            sl,
            b2s,
            cache,
            pool_k,
            pool_v,
            out,
            self.scale,
            Hq * D,
            D,
            bt.stride(0),
            cache.stride(0),
            cache.stride(1),
            pool_k.stride(0),
            pool_k.stride(1),
            pool_k.stride(2),
            Hq * D,
            D,
            **common_vq,
        )
        _kvarn_fused_decode_stage1[(B, Hk, splits)](
            q,
            vq_rows,
            bt,
            sl,
            b2s,
            cache,
            pool_k,
            pool_v,
            mid_o,
            mid_lse,
            self.scale,
            Hq * D,
            D,
            bt.stride(0),
            cache.stride(0),
            cache.stride(1),
            pool_k.stride(0),
            pool_k.stride(1),
            pool_k.stride(2),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_lse.stride(0),
            NUM_KV_SPLITS=splits,
            HQ=Hq,
            **common_vq,
        )
        # 2c. Shared-dequant verify kernel (uniform-QLEN spec verify) — runs
        # its @triton.autotune sweep here so capture never benchmarks. QLEN is
        # the deployment's 1 + num_speculative_tokens.
        try:
            from vllm.config import get_current_vllm_config

            _spec = get_current_vllm_config().speculative_config
            _qlen = 1 + int(_spec.num_speculative_tokens) if _spec else 0
        except Exception:
            _qlen = 0
        if _qlen >= 2:
            from vllm.v1.attention.ops.triton_kvarn_decode import (
                _kvarn_fused_verify_stage1,
            )

            nq = B * _qlen
            sl_vq = sl.repeat_interleave(_qlen)
            qv = torch.zeros(nq, Hq, D, dtype=torch.float16, device=device)
            mid_o_v = torch.zeros(
                nq * Hq, splits, D, dtype=torch.float32, device=device
            )
            mid_lse_v = torch.zeros(nq * Hq, splits, dtype=torch.float32, device=device)
            common_v = dict(common)
            _kvarn_fused_verify_stage1[(B, Hk, splits)](
                qv,
                bt,
                sl_vq,
                b2s,
                cache,
                pool_k,
                pool_v,
                mid_o_v,
                mid_lse_v,
                self.scale,
                Hq * D,
                D,
                bt.stride(0),
                cache.stride(0),
                cache.stride(1),
                pool_k.stride(0),
                pool_k.stride(1),
                pool_k.stride(2),
                mid_o_v.stride(0),
                mid_o_v.stride(1),
                mid_lse_v.stride(0),
                QLEN=_qlen,
                HQ=Hq,
                NUM_KV_SPLITS=splits,
                **common_v,
            )
        # 3. Packed-KV build kernel (materialize fallback + the cached-multiquery
        # spec-verify path).
        kp = torch.zeros(B * n_blocks * G, Hk, D, dtype=torch.float16, device=device)
        vp = torch.zeros_like(kp)
        cu_k = torch.arange(B + 1, dtype=torch.int32, device=device) * (n_blocks * G)
        _kvarn_build_packed_kv_kernel[(B * n_blocks, Hk)](
            bt,
            sl,
            cu_k,
            b2s,
            cache,
            pool_k,
            pool_v,
            kp,
            vp,
            bt.stride(0),
            cache.stride(0),
            cache.stride(1),
            pool_k.stride(0),
            pool_k.stride(1),
            pool_k.stride(2),
            kp.stride(0),
            kp.stride(1),
            MAX_BLOCKS_PER_REQ=n_blocks,
            D=D,
            GROUP=G,
            K_BITS=cfg.key_bits,
            V_BITS=cfg.value_bits,
            NUM_BLOCKS_LOOKUP=B * n_blocks,
            K_PACKED_OFFSET=cfg.k_packed_offset,
            K_S_COL_OFFSET=cfg.k_s_col_offset,
            K_ZP_OFFSET=cfg.k_zp_offset,
            K_S_ROW_OFFSET=cfg.k_s_row_offset,
            V_PACKED_OFFSET=cfg.v_packed_offset,
            V_S_COL_OFFSET=cfg.v_s_col_offset,
            V_S_ROW_OFFSET=cfg.v_s_row_offset,
            V_ZP_OFFSET=cfg.v_zp_offset,
            num_warps=4,
            num_stages=2,
        )
        torch.accelerator.synchronize(device)

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

    def _flat_block(
        self, kv_cache: torch.Tensor, block_id: int, head: int
    ) -> torch.Tensor:
        """Contiguous ``[tile_bytes_aligned]`` uint8 view for one (block, head).

        ``kv_cache`` has shape ``(num_blocks, num_kv_heads, tile_bytes_aligned)``,
        so this selects a single contiguous row — no copy, writes propagate
        back to the cache tensor.
        """
        return kv_cache[block_id, head]

    def _write_packed(
        self,
        kv_cache: torch.Tensor,
        block_id: int,
        head: int,
        store_K: dict[str, torch.Tensor],
        store_V: dict[str, torch.Tensor],
    ) -> None:
        cfg = self.kvarn_config
        flat = self._flat_block(kv_cache, block_id, head)

        # K packed bytes
        ko = cfg.k_packed_offset
        flat[ko : ko + cfg.k_packed_bytes] = (
            store_K["q_packed_uint8"].reshape(-1).to(torch.uint8)
        )
        # K s_col, zp (per-channel, length D, fp16)
        flat[cfg.k_s_col_offset : cfg.k_s_col_offset + cfg.head_dim * 2].view(
            torch.float16
        )[:] = store_K["s_col_K"]
        flat[cfg.k_zp_offset : cfg.k_zp_offset + cfg.head_dim * 2].view(torch.float16)[
            :
        ] = store_K["zp_K"]
        flat[cfg.k_s_row_offset : cfg.k_s_row_offset + cfg.group * 2].view(
            torch.float16
        )[:] = store_K["s_row_K"]

        # V packed bytes
        vo = cfg.v_packed_offset
        flat[vo : vo + cfg.v_packed_bytes] = (
            store_V["q_packed_uint8"].reshape(-1).to(torch.uint8)
        )
        flat[cfg.v_s_col_offset : cfg.v_s_col_offset + cfg.head_dim * 2].view(
            torch.float16
        )[:] = store_V["s_col_V"]
        flat[cfg.v_s_row_offset : cfg.v_s_row_offset + cfg.group * 2].view(
            torch.float16
        )[:] = store_V["s_row_V"]
        flat[cfg.v_zp_offset : cfg.v_zp_offset + cfg.group * 2].view(torch.float16)[
            :
        ] = store_V["zp_V"]

    def _read_block_dequantized(
        self,
        kv_cache: torch.Tensor,
        block_id: int,
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

        K_out = torch.empty(
            group, self.num_kv_heads, D, dtype=torch.float16, device=device
        )
        V_out = torch.empty(
            group, self.num_kv_heads, D, dtype=torch.float16, device=device
        )

        H = self._hadamard(device)  # [D, D] fp32

        for h in range(self.num_kv_heads):
            flat = self._flat_block(kv_cache, block_id, h)

            # K side. K is packed at cfg.key_bits (8 // bits values per byte),
            # so the per-channel row holds group // pack_k bytes — NOT a fixed
            # group // 2. (group // 2 only happens to be right for 4-bit K.)
            pack_k = 8 // cfg.key_bits
            k_packed = flat[
                cfg.k_packed_offset : cfg.k_packed_offset + cfg.k_packed_bytes
            ].view(D, group // pack_k)
            s_col_K = flat[cfg.k_s_col_offset : cfg.k_s_col_offset + D * 2].view(
                torch.float16
            )
            zp_K = flat[cfg.k_zp_offset : cfg.k_zp_offset + D * 2].view(torch.float16)
            s_row_K = flat[cfg.k_s_row_offset : cfg.k_s_row_offset + group * 2].view(
                torch.float16
            )
            K_rot_DG = kvarn_dequant_tile_k(
                k_packed, s_col_K, zp_K, s_row_K, group=group, bits=cfg.key_bits
            )
            # Un-rotate: [D, group] → [group, D] (= K rows-tokens),
            # then ⋅H to undo rotation
            K_unrot = K_rot_DG.T @ H  # [group, D]
            K_out[:, h, :] = K_unrot.to(torch.float16)

            # V side. V is packed at cfg.value_bits — for the default k4v2
            # preset that is 2-bit (4 values per byte), so each token row holds
            # D // pack_v bytes. The old fixed D // 2 assumed 4-bit V and broke
            # k4v2 (view size mismatch), which is why this slow gather path had
            # never worked for the default preset.
            pack_v = 8 // cfg.value_bits
            v_packed = flat[
                cfg.v_packed_offset : cfg.v_packed_offset + cfg.v_packed_bytes
            ].view(group, D // pack_v)
            s_col_V = flat[cfg.v_s_col_offset : cfg.v_s_col_offset + D * 2].view(
                torch.float16
            )
            s_row_V = flat[cfg.v_s_row_offset : cfg.v_s_row_offset + group * 2].view(
                torch.float16
            )
            zp_V = flat[cfg.v_zp_offset : cfg.v_zp_offset + group * 2].view(
                torch.float16
            )
            V_rot_GD = kvarn_dequant_tile_v(
                v_packed, s_col_V, s_row_V, zp_V, head_dim=D, bits=cfg.value_bits
            )
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
        slot = cls._block_to_slot_dict.get(self._group_key, {}).get(block_id)
        if slot is None:
            # Block has no pool slot — nothing to flush.
            self._tails.pop(block_id, None)
            return
        K_rot = self._tail_K_pool[slot].float()  # [group, Hk, D]
        V_rot = self._tail_V_pool[slot].float()  # [group, Hk, D]
        self._tails.pop(block_id, None)  # drop tracker entry

        # Build batched per-head tiles (rows = absorb axis for each)
        K_tiles = K_rot.permute(1, 2, 0).contiguous()  # [Hk, D, group]
        V_tiles = V_rot.permute(1, 0, 2).contiguous()  # [Hk, group, D]

        # Sinkhorn + pack (fused launch when square head_dim==group, else
        # separate K/V launches — see _sinkhorn_pack_kv).
        K_out, V_out = _sinkhorn_pack_kv(K_tiles, V_tiles, cfg)
        Hk = self.num_kv_heads

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
        """Flush many (impl, block_id, kv_cache) tiles to int4.

        Dispatches to the vectorized path (default) or the legacy per-tile path
        (KVARN_FAST_FLUSH=0, kept for A/B + the tile-dump debug hook). The
        vectorized path replaces the per-(layer,block,head) Python gather/write
        loops — which exploded into ~10^5 tiny GPU ops on a synchronized burst
        (prefill completion, lockstep decode boundary) and dominated build() at
        high concurrency (~44 ms/step at B=256) — with one
        index_select gather + one index_copy write per (layer, block-chunk).
        Numerically identical: same Sinkhorn, same RTN/pack math, same byte
        layout; only the data movement is batched."""
        if not flush_pairs:
            return
        if os.environ.get("KVARN_FAST_FLUSH", "1") != "1":
            return cls._batched_flush_legacy(flush_pairs)

        cfg = flush_pairs[0][0].kvarn_config
        Hk = flush_pairs[0][0].num_kv_heads
        D = cfg.head_dim
        G = cfg.group
        T = cfg.tile_bytes_aligned
        kpb = cfg.k_packed_bytes
        vpb = cfg.v_packed_bytes

        # Group by impl (layer); every impl flushes the SAME block set (the
        # builder cross-products flush_block_ids with group_impls) and pool slot
        # indices are shared across layers, so per impl we have (kvc, bids, slots).
        by_impl: dict = {}
        for impl, bid, kvc in flush_pairs:
            slot = cls._block_to_slot_dict.get(impl._group_key, {}).get(bid)
            if slot is None:
                impl._tails.pop(bid, None)
                continue
            e = by_impl.get(id(impl))
            if e is None:
                e = [impl, kvc, [], []]
                by_impl[id(impl)] = e
            e[2].append(bid)
            e[3].append(slot)
            impl._tails.pop(bid, None)
        if not by_impl:
            return

        # Block-chunk so one Sinkhorn launch stays bounded (~2k [R,C] tiles).
        CHUNK_BLOCKS = max(1, 2048 // max(Hk, 1))
        for impl, kvc, bids, slots in by_impl.values():
            if kvc is None:
                continue
            dev = impl._tail_K_pool.device
            # WSL fix: one H2D for the whole block set, slice on device
            # per chunk (a torch.as_tensor H2D per chunk is a sync, ~100x on WSL).
            slots_dev = torch.as_tensor(slots, dtype=torch.long, device=dev)
            bids_dev = torch.as_tensor(bids, dtype=torch.long, device=dev)
            for c0 in range(0, len(bids), CHUNK_BLOCKS):
                bchunk = bids[c0 : c0 + CHUNK_BLOCKS]
                nB = len(bchunk)
                slot_t = slots_dev[c0 : c0 + CHUNK_BLOCKS]
                bid_t = bids_dev[c0 : c0 + CHUNK_BLOCKS]
                # One gather per chunk (was nB tiny .float() ops).
                K_rot = impl._tail_K_pool.index_select(0, slot_t).float()  # [nB,G,Hk,D]
                V_rot = impl._tail_V_pool.index_select(0, slot_t).float()
                # Tiles: K [N, D, G] (absorb=channel), V [N, G, D] (absorb=token).
                K_tiles = K_rot.permute(0, 2, 3, 1).reshape(nB * Hk, D, G)
                V_tiles = V_rot.permute(0, 2, 1, 3).reshape(nB * Hk, G, D)
                K_out, V_out = _sinkhorn_pack_kv(K_tiles, V_tiles, cfg)
                # Assemble the packed cache record [nB*Hk, tile_bytes] by
                # concatenating fields in config-offset order (fp16 scales
                # byte-reinterpreted to uint8), then pad to tile_bytes_aligned.
                M = nB * Hk
                parts = [
                    K_out["q_packed_uint8"].reshape(M, kpb),
                    K_out["s_col_K"].contiguous().view(torch.uint8),
                    K_out["zp_K"].contiguous().view(torch.uint8),
                    K_out["s_row_K"].contiguous().view(torch.uint8),
                    V_out["q_packed_uint8"].reshape(M, vpb),
                    V_out["s_col_V"].contiguous().view(torch.uint8),
                    V_out["s_row_V"].contiguous().view(torch.uint8),
                    V_out["zp_V"].contiguous().view(torch.uint8),
                ]
                rec = torch.cat(parts, dim=1)  # [M, tile_bytes]
                if rec.shape[1] < T:
                    rec = torch.nn.functional.pad(rec, (0, T - rec.shape[1]))
                # One scatter per chunk (was nB*Hk _write_packed calls).
                kvc[bid_t] = rec.view(nB, Hk, T)

    @classmethod
    def _batched_flush_legacy(cls, flush_pairs: list) -> None:
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
            slot = cls._block_to_slot_dict.get(impl._group_key, {}).get(bid)
            if slot is None:
                impl._tails.pop(bid, None)
                continue
            filt.append((impl, bid, kvc, slot))
            impl._tails.pop(bid, None)
        if not filt:
            return
        for c0 in range(0, len(filt), CHUNK_PAIRS):
            chunk = filt[c0 : c0 + CHUNK_PAIRS]
            N = len(chunk)
            # Gather pool data for this chunk.
            K_list = [
                impl._tail_K_pool[slot].float() for impl, _, _, slot in chunk
            ]  # [G, Hk, D]
            V_list = [impl._tail_V_pool[slot].float() for impl, _, _, slot in chunk]
            K_stack = torch.stack(K_list, dim=0)  # [N, G, Hk, D]
            V_stack = torch.stack(V_list, dim=0)
            # Optional: dump first chunk's raw (pre-Sinkhorn) tiles for outlier
            # analysis (KVARN_DUMP_TILES=/path/to/file.pt).
            dump_path = os.environ.get("KVARN_DUMP_TILES", "")
            if dump_path and not getattr(cls, "_tiles_dumped", False):
                cls._tiles_dumped = True
                # Capture per-tile (layer_idx, block_id) for per-layer analysis.
                # layer_idx pulled from impl.layer_name
                # (e.g. "model.layers.7.self_attn")
                # via a regex fallback to enumerate index if name parsing fails.
                import regex as re

                lyr_ids, blk_ids = [], []
                for impl, bid, _, _ in chunk:
                    name = getattr(impl, "layer_name", "") or ""
                    m = re.search(r"layers\.(\d+)\b", name)
                    lyr_ids.append(int(m.group(1)) if m else -1)
                    blk_ids.append(int(bid))
                torch.save(
                    {
                        "K_stack": K_stack.detach().cpu(),
                        "V_stack": V_stack.detach().cpu(),
                        "layer_ids": lyr_ids,
                        "block_ids": blk_ids,
                        "Hk": flush_pairs[0][0].num_kv_heads,
                        "G": cfg.group,
                        "D": cfg.head_dim,
                        "key_bits": cfg.key_bits,
                        "value_bits": cfg.value_bits,
                        "sinkhorn_iters": cfg.sinkhorn_iters,
                    },
                    dump_path,
                )
                print(
                    f"[KVARN] dumped {N} (layer,block) pre-Sinkhorn tiles "
                    f"→ {dump_path}",
                    flush=True,
                )
                print(f"[KVARN] layer_ids in dump: {sorted(set(lyr_ids))}", flush=True)
            del K_list, V_list
            # K tile per Sinkhorn batch row: [D, G] (absorb = channel).
            K_tiles = K_stack.permute(0, 2, 3, 1).reshape(
                N * Hk, K_stack.shape[3], K_stack.shape[1]
            )
            V_tiles = V_stack.permute(0, 2, 1, 3).reshape(
                N * Hk, V_stack.shape[1], V_stack.shape[3]
            )
            del K_stack, V_stack
            # Sinkhorn + pack (fused when square head_dim==group, else separate
            # K/V launches for non-square head_dim=256 — see _sinkhorn_pack_kv).
            K_out, V_out = _sinkhorn_pack_kv(K_tiles, V_tiles, cfg)
            del K_tiles, V_tiles
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

        # bf16 boundary-cast (see forward): KVarN store/rotation is fp16.
        if key.dtype != torch.float16:
            key = key.to(torch.float16)
            value = value.to(torch.float16)

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
            k_rot,
            v_rot,
            slot_mapping[:N],
            self._block_to_slot_t,
            self._tail_K_pool,
            self._tail_V_pool,
            k_rot.stride(0),
            k_rot.stride(1),
            self._tail_K_pool.stride(0),
            self._tail_K_pool.stride(1),
            self._tail_K_pool.stride(2),
            GROUP=cfg.group,
            D=D,
            NUM_BLOCKS_LOOKUP=self._block_lookup_size,
            num_warps=2,
            num_stages=2,
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
        attn_metadata: KVarNMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        device = query.device

        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=device,
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

        # bf16 boundary-cast: KVarN's compute (rotation matmul, scratch buffers,
        # Triton stores) is fp16 internally. Cast bf16 activations to fp16 at this
        # entry point; the output write below casts back to output.dtype. fp16 is
        # untouched (byte-identical), and the cast is lossless for KVarN (fp16
        # mantissa > bf16, and the cache is 4-bit). Without this, bf16 q mixing
        # with fp16 KV buffers trips "Expected out BFloat16, got Half".
        if query.dtype != torch.float16:
            query = query.to(torch.float16)
            key = key.to(torch.float16)
            value = value.to(torch.float16)

        q = query[:N].view(N, self.num_heads, self.head_size)

        if not attn_metadata.is_prefill:
            attn_out = self._decode_path(q, kv_cache, attn_metadata)
        elif (
            attn_metadata.vq_seqlen is not None and attn_metadata.num_decode_tokens == N
        ):
            # Pure multi-token decode batch = a spec-decode verify step
            # (uniform query length under graph capture). One fused-kernel
            # pass over the vq plan — fully graph-capturable.
            attn_out = self._verify_decode_path(q, kv_cache, attn_metadata)
        elif attn_metadata.num_decodes == 0:
            if attn_metadata.has_cached_multiquery:
                # Speculative-decode verify (or chunked-prefill continuation):
                # the query tokens have cached history that must be attended.
                # _prefill_first_chunk would drop it; use the context-aware path.
                attn_out = self._cached_multiquery_path(q, kv_cache, attn_metadata)
            else:
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
        self,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        causal=True,
    ) -> torch.Tensor:
        if self.fa_version is None:
            return flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                softmax_scale=self.scale,
                causal=causal,
            )
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=self.scale,
            causal=causal,
            fa_version=self.fa_version,
        )

    def _prefill_first_chunk(
        self,
        q,
        k,
        v,
        attn_metadata: KVarNMetadata,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        """First-chunk prefill: every request's full prompt is in the current
        batch, so attention runs on raw K/V via flash_attn_varlen. The K/V
        have already been written to the cache by `do_kv_cache_update`."""
        causal = getattr(attn_metadata, "causal", True)
        # FlashAttention caps head_dim at 256; the head_dim-512 global layers of
        # Gemma-4 must use the SDPA path (handles arbitrary head_dim). Prefill is
        # a one-time cost (decode dominates at long context), so SDPA here is fine.
        if _HAS_FLASH_ATTN and self.head_size <= 256:
            return self._flash_varlen(
                q,
                k,
                v,
                cu_q=attn_metadata.query_start_loc,
                cu_k=attn_metadata.query_start_loc,
                max_q=attn_metadata.max_query_len,
                max_k=attn_metadata.max_query_len,
                causal=causal,
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
                q_r,
                k_r,
                v_r,
                is_causal=causal,
                scale=self.scale,
                enable_gqa=self.num_kv_heads < self.num_heads,
            )
            outputs.append(o[0].transpose(0, 1))  # [q_len, Hq, D]
        return (
            torch.cat(outputs, dim=0)
            if outputs
            else torch.empty(
                0,
                self.num_heads,
                self.head_size,
                device=q.device,
                dtype=q.dtype,
            )
        )

    def _gather_request_kv(
        self,
        kv_cache: torch.Tensor,
        block_table_row: torch.Tensor,
        seq_len: int,
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
        H = self._hadamard(device)  # [D, D] fp32

        def _unrot_pool(x: torch.Tensor) -> torch.Tensor:
            return (x.float() @ H.T).to(torch.float16)

        # Stage α-2: a block lives in the fp16 pool iff it has a slot
        # (sinks + in-progress tails). Flushed blocks have their slot freed
        # and live in the int4 cache.
        dict_map = type(self)._block_to_slot_dict.get(self._group_key, {})

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
                K_parts.append(
                    torch.zeros(
                        tail_len,
                        self.num_kv_heads,
                        D,
                        dtype=torch.float16,
                        device=device,
                    )
                )
                V_parts.append(torch.zeros_like(K_parts[-1]))

        K = (
            torch.cat(K_parts, dim=0)
            if K_parts
            else torch.empty(
                0,
                self.num_kv_heads,
                D,
                dtype=torch.float16,
                device=device,
            )
        )
        V = torch.cat(V_parts, dim=0) if V_parts else torch.empty_like(K)
        return K, V

    def _decode_path(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
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
            return self._cached_multiquery_path(q, kv_cache, attn_metadata)

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
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KVarNMetadata,
    ) -> torch.Tensor:
        """Fallback: full fp16 dequant + SDPA. Used for multi-query-per-request
        decode steps (e.g. speculative decoding) where the Triton kernel's
        ``(B, Hq)`` per-program shape would mishandle the per-token mapping.
        """
        num_reqs = attn_metadata.block_table.shape[0]
        seq_lens = attn_metadata.seq_lens.tolist()
        qsl = attn_metadata.query_start_loc.tolist()
        out = torch.empty(
            q.shape[0], self.num_heads, self.head_size, dtype=q.dtype, device=q.device
        )
        for r in range(num_reqs):
            q_start, q_end = qsl[r], qsl[r + 1]
            if q_end <= q_start:
                continue
            seq_len = seq_lens[r]
            K_full, V_full = self._gather_request_kv(
                kv_cache,
                attn_metadata.block_table[r],
                seq_len,
            )
            q_r = q[q_start:q_end].transpose(0, 1).unsqueeze(0).float()
            K_t = K_full.transpose(0, 1).unsqueeze(0).float()
            V_t = V_full.transpose(0, 1).unsqueeze(0).float()
            cached_len = seq_len - (q_end - q_start)
            q_pos = (
                torch.arange(q_end - q_start, device=q.device).unsqueeze(1) + cached_len
            )
            k_pos = torch.arange(seq_len, device=q.device).unsqueeze(0)
            mask = k_pos <= q_pos
            o = F.scaled_dot_product_attention(
                q_r,
                K_t,
                V_t,
                attn_mask=mask,
                scale=self.scale,
                enable_gqa=self.num_kv_heads < self.num_heads,
            )
            out[q_start:q_end] = o[0].transpose(0, 1).to(q.dtype)
        return out

    def _verify_decode_path(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KVarNMetadata,
    ) -> torch.Tensor:
        """Spec-as-decode verify using the builder's persistent vq plan.

        Capture-safe: the vq buffers are persistent (filled CPU-side in
        build() between replays), the block-table/seq-len bound is the
        deployment constant, and the driver's intermediates are created
        inside the captured region (graph-pool managed) — same pattern as
        the single-token fused decode.
        """
        md = attn_metadata
        group = self.kvarn_config.group
        max_ctx_blocks = max((self._max_model_len + group - 1) // group, 1)
        from vllm.v1.attention.ops.triton_kvarn_decode import (
            kvarn_verify_attention,
        )

        B = md.block_table.shape[0]
        return kvarn_verify_attention(
            q,
            kv_cache,
            md.block_table,
            self.scale,
            self.kvarn_config,
            self,
            md.vq_req,
            md.vq_seqlen,
            max_ctx_blocks,
            qlen=md.vq_qlen,
            seq_lens=md.seq_lens[:B],
        )

    def _fused_verify_path(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KVarNMetadata,
    ) -> torch.Tensor:
        """Speculative-decode verify via the fused dual-source kernel.

        Each query token becomes a virtual kernel row with its own
        bottom-right causal length (cached_len + idx + 1) and an indirection
        to its request's block-table row — so the verify step reads int4
        tiles + the fp16 pool directly instead of materializing the whole
        context to fp16 scratch every step (O(context)/step; the
        long-context MTP collapse: measured 88 -> 45 tok/s from 2K -> 32K
        with the materialize route vs near-flat without MTP).

        Eager-only (verify steps are not graph-captured): fresh small
        tensors per call are fine.
        """
        md = attn_metadata
        B = md.block_table.shape[0]
        n_tok = q.shape[0]
        device = q.device
        group = self.kvarn_config.group

        qsl = md.query_start_loc[: B + 1].to(torch.long)
        qlens = qsl[1:] - qsl[:-1]  # [B]
        vq_req_long = torch.repeat_interleave(
            torch.arange(B, device=device), qlens
        )  # [n_tok]
        pos_in_req = torch.arange(n_tok, device=device) - qsl[:-1][vq_req_long]
        committed = md.seq_lens[:B].to(torch.long) - qlens
        vq_seqlen = (committed[vq_req_long] + pos_in_req + 1).to(torch.int32)
        vq_req = vq_req_long.to(torch.int32)

        max_ctx_blocks = min(
            (int(md.max_seq_len) + group - 1) // group, md.block_table.shape[1]
        )
        max_ctx_blocks = max(max_ctx_blocks, 1)

        from vllm.v1.attention.ops.triton_kvarn_decode import (
            kvarn_verify_attention,
        )

        return kvarn_verify_attention(
            q,
            kv_cache,
            md.block_table,
            self.scale,
            self.kvarn_config,
            self,
            vq_req,
            vq_seqlen,
            max_ctx_blocks,
        )

    def _cached_multiquery_path(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KVarNMetadata,
    ) -> torch.Tensor:
        """Multi-query tokens with cached history (a speculative-decode verify
        step or a chunked-prefill continuation), batched.

        Builds the batch's rotated fp16 K/V with the ONE block_table-driven
        Triton kernel (``_kvarn_build_packed_kv_kernel``) and runs a single
        ``flash_attn_varlen`` call. FA's varlen causal mask is bottom-right
        aligned when ``seqlen_q < seqlen_k``, i.e. query token ``t`` attends
        keys ``<= cached_len + t`` — exactly the spec-verify / continuation
        semantics, so no explicit mask is needed.

        Replaces ``_decode_path_slow`` on this route: the per-request Python
        gather (per-block ``.item()`` syncs + Python dequant + fp32 SDPA, per
        layer, per step) made MTP decode unusably slow (< 5 tok/s) and its
        transient fp32 materializations inflated the CUDA-graph memory
        estimate by GiBs, collapsing the derived KV-cache capacity. The slow
        path remains the fallback for head_dim > 256 (FA's cap) or a batch
        whose total KV exceeds the shared materialize scratch.
        """
        md = attn_metadata
        B = md.block_table.shape[0]
        # Small-qlen multi-query (the spec-decode verify step: every decode
        # step under MTP) goes to the FUSED verify kernel: per-token virtual
        # rows over the dual-source decode kernel, no fp16 materialization.
        # Routed by CONTEXT depth (measured, Qwen3.6-27B AWQ single stream):
        # materialize wins short context (88 vs 81 tok/s @2K — its one
        # write+read is cheap there and the fused per-call overhead shows);
        # fused wins long context (51 vs 45 @32K, growing with depth — the
        # materialize round-trip is the O(context)/step MTP
        # slowdown). Crossover ~12K; default threshold 64 blocks (8K).
        # The materialize+FA route also keeps LARGE qlen (chunked-prefill
        # continuations), where one materialization amortizes over thousands
        # of query tokens. KVARN_FUSED_VERIFY=0 forces materialize always.
        _group = self.kvarn_config.group
        if (
            os.environ.get("KVARN_FUSED_VERIFY", "1") == "1"
            and md.max_query_len <= int(os.environ.get("KVARN_FUSED_VERIFY_MAXQ", "8"))
            and (int(md.max_seq_len) + _group - 1) // _group
            >= int(os.environ.get("KVARN_FUSED_VERIFY_MIN_BLOCKS", "64"))
            and B > 0
        ):
            return self._fused_verify_path(q, kv_cache, md)
        if not _HAS_FLASH_ATTN or self.head_size > 256 or self._fa_K_buf is None:
            return self._decode_path_slow(q, kv_cache, md)

        seq_lens = md.seq_lens[:B].to(torch.int32)
        cu_k = F.pad(torch.cumsum(seq_lens, 0, dtype=torch.int32), (1, 0))
        total_k = int(cu_k[-1].item())
        if total_k <= 0 or total_k > self._fa_K_buf.shape[0]:
            return self._decode_path_slow(q, kv_cache, md)

        cfg = self.kvarn_config
        group = cfg.group
        D = self.head_size
        Hk = self.num_kv_heads
        max_k = int(md.max_seq_len)
        max_blocks = min((max_k + group - 1) // group, md.block_table.shape[1])
        max_blocks = max(max_blocks, 1)

        from vllm.v1.attention.ops.triton_kvarn_decode import (
            _kvarn_build_packed_kv_kernel,
        )

        K_packed = self._fa_K_buf
        V_packed = self._fa_V_buf
        _kvarn_build_packed_kv_kernel[(B * max_blocks, Hk)](
            md.block_table,
            seq_lens,
            cu_k,
            self._block_to_slot_t,
            kv_cache,
            self._tail_K_pool,
            self._tail_V_pool,
            K_packed,
            V_packed,
            md.block_table.stride(0),
            kv_cache.stride(0),
            kv_cache.stride(1),
            self._tail_K_pool.stride(0),
            self._tail_K_pool.stride(1),
            self._tail_K_pool.stride(2),
            K_packed.stride(0),
            K_packed.stride(1),
            MAX_BLOCKS_PER_REQ=max_blocks,
            D=D,
            GROUP=group,
            K_BITS=cfg.key_bits,
            V_BITS=cfg.value_bits,
            NUM_BLOCKS_LOOKUP=self._block_lookup_size,
            K_PACKED_OFFSET=cfg.k_packed_offset,
            K_S_COL_OFFSET=cfg.k_s_col_offset,
            K_ZP_OFFSET=cfg.k_zp_offset,
            K_S_ROW_OFFSET=cfg.k_s_row_offset,
            V_PACKED_OFFSET=cfg.v_packed_offset,
            V_S_COL_OFFSET=cfg.v_s_col_offset,
            V_S_ROW_OFFSET=cfg.v_s_row_offset,
            V_ZP_OFFSET=cfg.v_zp_offset,
            num_warps=4,
            num_stages=2,
        )

        # The packed K/V are in the rotated frame (the store path rotates before
        # quantizing / pooling), so rotate q in and un-rotate the output — same
        # fp16 Hadamard as the store side, so QK^T is invariant.
        H16 = (
            self._H_fp16
            if self._H_fp16 is not None
            else self._hadamard(q.device).to(torch.float16)
        )
        n_tok = q.shape[0]
        q_rot = torch.mm(q.reshape(-1, D), H16).view(n_tok, self.num_heads, D)
        out_rot = self._flash_varlen(
            q_rot,
            K_packed[:total_k],
            V_packed[:total_k],
            cu_q=md.query_start_loc[: B + 1],
            cu_k=cu_k,
            max_q=md.max_query_len,
            max_k=max_k,
            causal=getattr(md, "causal", True),
        )
        return torch.mm(out_rot.reshape(-1, D), H16).view(n_tok, self.num_heads, D)

    def _mixed_batch_path(
        self,
        q: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KVarNMetadata,
    ) -> torch.Tensor:
        """Split mixed batch into decode-then-prefill, mirroring TurboQuant."""
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        N = attn_metadata.num_actual_tokens

        out = torch.empty(
            N, self.num_heads, self.head_size, dtype=q.dtype, device=q.device
        )

        # Build the Stage α-2 fa_* fields for the decode subset. This path
        # always runs eager (mixed batches aren't graph-captured), so fresh
        # (non-persistent) tensors are fine.
        group = self.kvarn_config.group
        dec_seq_lens = attn_metadata.seq_lens[:num_decodes].to(torch.int32)
        dec_cu_k = torch.nn.functional.pad(
            torch.cumsum(dec_seq_lens, dim=0), (1, 0)
        ).to(torch.int32)
        dec_cu_q = torch.arange(num_decodes + 1, dtype=torch.int32, device=q.device)
        mbpr = (self._max_model_len + group - 1) // group
        decode_meta = KVarNMetadata(
            seq_lens=attn_metadata.seq_lens[:num_decodes],
            slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
            block_table=attn_metadata.block_table[:num_decodes],
            query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
            num_actual_tokens=num_decode_tokens,
            max_query_len=1,
            max_seq_len=attn_metadata.max_seq_len,
            is_prefill=False,
            fa_cu_seqlens_q=dec_cu_q,
            fa_cu_seqlens_k=dec_cu_k,
            fa_max_blocks_per_req=mbpr,
            fa_max_seqlen_k_fixed=self._max_model_len,
            causal=getattr(attn_metadata, "causal", True),
        )
        if attn_metadata.vq_seqlen is not None:
            # Spec-as-decode: the decode portion carries multi-token verify
            # queries — use the vq plan (mixed batches run eager, slices ok).
            # vq_req is set together with vq_seqlen (never None when it is not).
            assert attn_metadata.vq_req is not None
            decode_meta.vq_req = attn_metadata.vq_req[:num_decode_tokens]
            decode_meta.vq_seqlen = attn_metadata.vq_seqlen[:num_decode_tokens]
            decode_meta.vq_qlen = attn_metadata.vq_qlen
            out[:num_decode_tokens] = self._verify_decode_path(
                q[:num_decode_tokens],
                kv_cache,
                decode_meta,
            )
        else:
            out[:num_decode_tokens] = self._decode_path(
                q[:num_decode_tokens],
                kv_cache,
                decode_meta,
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
            # avoid a per-step .item() D2H sync; the global max is a safe
            # upper bound for the prefill kernel
            max_seq_len=attn_metadata.max_seq_len,
            is_prefill=True,
            causal=getattr(attn_metadata, "causal", True),
        )
        if attn_metadata.has_cached_multiquery:
            # The multi-query (prefill-classified) requests here are speculative
            # -decode verify steps / chunked-prefill continuations with cached
            # history — attend over the cached K/V, not just the new tokens.
            out[num_decode_tokens:] = self._cached_multiquery_path(
                q[num_decode_tokens:],
                kv_cache,
                prefill_meta,
            )
        else:
            k_pref = k_all[num_decode_tokens:].view(
                -1, self.num_kv_heads, self.head_size
            )
            v_pref = v_all[num_decode_tokens:].view(
                -1, self.num_kv_heads, self.head_size
            )
            out[num_decode_tokens:] = self._prefill_first_chunk(
                q[num_decode_tokens:],
                k_pref,
                v_pref,
                prefill_meta,
                kv_cache,
            )
        return out
