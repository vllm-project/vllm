# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER ASM FP8 backend for MLA prefill on AMD gfx950 (MI350).

Dispatches through aiter.mla_prefill_ps_asm_fwd -> aiter.mla_reduce_v1.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.prefill.selector import (
        MLAPrefillSelectorConfig,
    )

logger = init_logger(__name__)

# Q-side tile size baked into the gfx950 mla_prefill_ps_asm_fwd kernel.
_FP8_PREFILL_TILE_Q = 256
# K-side tiling granularity required by the PS scheduler.
_KVLEN_GRANULARITY = 128


class AiterAsmPrefillBackend(MLAPrefillBackend):
    """FP8 MLA prefill backend built on AITER persistent-scheduling ASM on gfx950.

    The PS metadata is built once per batch for the new-tokens chunk and once per
    context chunk, then reused inside the kernel dispatch.

    Requires:
        - gfx950 (``DeviceCapability`` ``major=9, minor=5``)
        - AITER built with ``mla_prefill_ps_asm_fwd``, ``mla_reduce_v1``,
          ``get_ps_metadata_v1``, ``get_ps_metadata_info_v1`` exported
        - DeepSeek R1 MLA dimensions (qk_nope=128, qk_rope=64, v_head_dim=128)
        - FP8 KV cache (the FP8 ASM kernel produces wrong results otherwise)
    """

    supported_dtypes = [torch.float16, torch.bfloat16]
    requires_r1_mla_dimensions = True
    # mla_prefill_ps_asm_fwd only accepts FP8 Q/K/V; force the cast in the
    # parent regardless of attention_config.use_prefill_query_quantization.
    requires_fp8_query_quantization = True

    # Process-wide singleton caches keyed by (device, num_heads, max_num_reqs,
    # max_qlen, max_kvlen, kind). MLA backends are instantiated once per layer
    # (~30 per rank for DSV3), but the persistent PS scratch buffers depend
    # only on shape constants known at construction time. Sharing across all
    # layer instances drops allocation from ~30x to 1x per rank.
    _SHARED_BUFFERS: dict[tuple, dict] = {}
    # Process-wide cache of the kv_indices arange buffer, keyed by
    # (device, length, dtype). Reused across layers and chunks; sliced per call.
    _KV_INDICES_BUFFERS: dict[tuple, torch.Tensor] = {}
    # Process-wide gate for the noncausal JIT warmup. Keyed by
    # (device, num_heads, max_num_reqs, max_qlen, max_kvlen) so only the first
    # MLA layer instance on a given rank actually fires the warmup kernel.
    _WARMUP_DONE: set[tuple] = set()

    @staticmethod
    def get_name() -> str:
        return "AITER_ASM"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        return device_capability.major == 9 and device_capability.minor == 5

    @classmethod
    def is_available(cls) -> bool:
        try:
            from vllm.platforms.rocm import on_gfx950
        except Exception:  # noqa: BLE001
            return False
        if not on_gfx950():
            return False
        try:
            from aiter import (  # noqa: F401
                get_ps_metadata_info_v1,
                get_ps_metadata_v1,
                mla_prefill_ps_asm_fwd,
                mla_reduce_v1,
            )
        except Exception:  # noqa: BLE001
            return False
        return True

    @classmethod
    def validate_configuration(
        cls,
        device_capability: "DeviceCapability",
        selector_config: "MLAPrefillSelectorConfig",
    ) -> list[str]:
        invalid_reasons = super().validate_configuration(
            device_capability, selector_config
        )
        # Otherwise produces incorrect results
        if selector_config.cache_dtype not in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            invalid_reasons.append(
                f"cache_dtype {selector_config.cache_dtype!r} is not FP8 "
                "(requires fp8/fp8_e4m3/fp8_e5m2)"
            )
        return invalid_reasons

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

        from aiter import (
            get_ps_metadata_info_v1,
            get_ps_metadata_v1,
            mla_prefill_ps_asm_fwd,
            mla_reduce_v1,
        )

        self._mla_prefill_ps_asm_fwd = mla_prefill_ps_asm_fwd
        self._mla_reduce_v1 = mla_reduce_v1
        self._get_ps_metadata_v1 = get_ps_metadata_v1
        self._get_ps_metadata_info_v1 = get_ps_metadata_info_v1

        # Populated by prepare_metadata before each forward pass.
        self._new_tokens_ps: dict | None = None

        # Per-forward cache of int(reduce_indptr[-1].item()) keyed by
        # chunk_idx. The PS scheduler is deterministic in chunk geometry
        # (qo_indptr, kv_indptr, seq_lens) and that geometry is identical
        # across all layers in a forward, so the live partial-tile count
        # for chunk i is the same on every layer. Caching turns the sync
        # from O(num_layers * num_chunks) per forward into O(num_chunks).
        # Reset in prepare_metadata at the start of each forward.
        self._context_num_partial_tiles: dict[int, int] = {}

        # Worst-case sizes used to pre-allocate persistent PS buffers.
        #
        # AITER's get_ps_metadata_info_v1 sizes work buffers as
        # qo_tile_cnt = batch_size * ceil(max_qlen / qlen_granularity).
        # The real upper bound on Q-tiles a forward can emit is
        # sum_i ceil(q_i / qlen_granularity) <= total_Q / qlen_granularity + B,
        # where total_Q <= max_num_batched_tokens and B <= max_num_seqs. The
        # buffer must dominate that. Sizing with batch_size=max_num_seqs and
        # max_qlen=max_num_batched_tokens gives
        # qo_tile_cnt = max_num_seqs * ceil(max_num_batched_tokens / 256),
        # which dominates the bound and matches the original PR (#42509).
        self._ps_max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self._ps_max_qlen = vllm_config.scheduler_config.max_num_batched_tokens
        # Worst-case K length for a context chunk per sequence. Bounded by
        # the chunked-prefill workspace size, since the scheduler sets
        # max_context_chunk = chunked_prefill_workspace_size //
        # num_prefills_with_context (1 in the worst case). Needed by AITER
        # PS scheduler to size noncausal context-chunk work buffers; without
        # it, the PS metadata is sized for causal-only (max_kvlen ==
        # max_qlen) and can under-allocate work slots for long contexts.
        from vllm.model_executor.layers.attention.mla_attention import (
            MLACommonMetadataBuilder,
        )

        self._ps_max_context_kvlen = (
            MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
                vllm_config
            )
        )
        # Used by _get_kv_indices_buf to size the shared arange buffer; the
        # scheduler never emits more than max_num_batched_tokens new tokens
        # per forward.
        self._ps_max_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        # Persistent buffers are looked up from the class-level _SHARED_BUFFERS
        # cache the first time prepare_metadata / run_prefill_context_chunk
        # runs (when we have a device). Two separate buffer sets exist per
        # rank, one for the new-tokens chunk and one shared across context
        # chunks: the new-tokens dict is built once in prepare_metadata and
        # consumed later by run_prefill_new_tokens, while context chunks
        # build and consume within run_prefill_context_chunk. Sharing one
        # buffer set would let the context loop overwrite the new-tokens PS
        # metadata before the new-tokens kernel reads it.

    def _get_kv_indices_buf(self, device: torch.device, length: int) -> torch.Tensor:
        """Return a [0, 1, ..., length-1] int32 view into a shared arange buffer.

        kv_indices feeds the PS kernel as a contiguous identity map. Allocating
        a fresh ``torch.arange`` per layer per chunk shows up in profiles as
        an HtoD DtoD copy + arange kernel right before each prefill launch.
        We allocate one buffer per (device, length-class) on first use, sized
        to ``max(length, max_num_batched_tokens)`` so subsequent calls slice
        into it instead of allocating.
        """
        size = max(length, self._ps_max_batched_tokens)
        key = (str(device), size)
        buf = type(self)._KV_INDICES_BUFFERS.get(key)
        if buf is None:
            buf = torch.arange(size, device=device, dtype=torch.int32)
            type(self)._KV_INDICES_BUFFERS[key] = buf
        return buf[:length]

    def _ensure_persistent_buffers(self, device: torch.device, kind: str) -> dict:
        """Lazily allocate max-size PS scratch buffers for one chunk kind.

        ``kind`` is ``"new_tokens"`` or ``"context"``. Two independent buffer
        sets exist so the context-chunk loop never overwrites new-tokens
        metadata that ``run_prefill_new_tokens`` still needs.

        Buffers live in the class-level ``_SHARED_BUFFERS`` cache keyed by
        (device, num_heads, max_num_reqs, max_qlen, max_kvlen, kind) so all
        layer instances on the same rank share one allocation per kind.
        """
        # Noncausal context chunks need max_kvlen passed explicitly so the PS
        # scheduler sizes per-q-tile KV-split slots correctly. Causal new-
        # tokens chunks have K == Q per sequence; the default (max_kvlen
        # falls back to max_qlen) is correct.
        max_kvlen = self._ps_max_context_kvlen if kind == "context" else None
        cache_key = (
            str(device),
            self.num_heads,
            self._ps_max_num_reqs,
            self._ps_max_qlen,
            max_kvlen,
            kind,
        )
        cached = type(self)._SHARED_BUFFERS.get(cache_key)
        if cached is not None:
            return cached
        (
            (work_metadata_size, work_metadata_dtype),
            (work_indptr_size, work_indptr_dtype),
            (work_info_size, work_info_dtype),
            (reduce_indptr_size, reduce_indptr_dtype),
            (reduce_final_map_size, reduce_final_map_dtype),
            (reduce_partial_map_size, reduce_partial_map_dtype),
        ) = self._get_ps_metadata_info_v1(
            batch_size=self._ps_max_num_reqs,
            num_head_k=self.num_heads,
            max_qlen=self._ps_max_qlen,
            qlen_granularity=_FP8_PREFILL_TILE_Q,
            max_kvlen=max_kvlen,
            kvlen_granularity=_KVLEN_GRANULARITY,
        )
        buffers = {
            "work_metadata": torch.empty(
                work_metadata_size, dtype=work_metadata_dtype, device=device
            ),
            "work_indptr": torch.empty(
                work_indptr_size, dtype=work_indptr_dtype, device=device
            ),
            "work_info": torch.empty(
                *work_info_size, dtype=work_info_dtype, device=device
            ),
            "reduce_indptr": torch.empty(
                reduce_indptr_size, dtype=reduce_indptr_dtype, device=device
            ),
            "reduce_final_map": torch.empty(
                *reduce_final_map_size,
                dtype=reduce_final_map_dtype,
                device=device,
            ),
            "reduce_partial_map": torch.empty(
                reduce_partial_map_size,
                dtype=reduce_partial_map_dtype,
                device=device,
            ),
        }
        type(self)._SHARED_BUFFERS[cache_key] = buffers
        logger.info(
            "AITER_ASM persistent PS buffers allocated (kind=%s, "
            "max_num_reqs=%d, max_qlen=%d, num_head_k=%d) "
            "work_metadata=%s work_indptr=%s work_info=%s "
            "reduce_indptr=%s reduce_final_map=%s reduce_partial_map=%s",
            kind,
            self._ps_max_num_reqs,
            self._ps_max_qlen,
            self.num_heads,
            work_metadata_size,
            work_indptr_size,
            tuple(work_info_size),
            reduce_indptr_size,
            tuple(reduce_final_map_size),
            reduce_partial_map_size,
        )
        return buffers

    def _build_ps_for_chunk(
        self,
        qo_indptr_cpu: torch.Tensor,
        kv_indptr_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        is_causal: bool,
        device: torch.device,
        kind: str,
        max_qlen: int,
        chunk_idx: int | None = None,
    ) -> dict:
        """Build PS metadata in the persistent scratch buffers of one kind.

        ``kind`` is ``"new_tokens"`` or ``"context"``; selects which
        persistent buffer set to write into. ``seq_lens_cpu`` drives the PS
        scheduler's K-side work split. For the new-tokens chunk (Q == K)
        Q-side and K-side lengths are identical. ``max_qlen`` is Q-side
        (kernel uses it to size per-tile workspace); caller supplies it from
        prefill_metadata.max_query_len to avoid a per-build host sync.

        The kernel writes into pre-allocated max-size persistent scratch
        buffers; the returned dict holds direct references to them. Safe
        because vLLM serializes build -> forward -> next build on the same
        stream, so the previous forward's kernels have already been queued
        (and read the buffers in stream order) before the next build writes.
        Mirrors the original PR #42509 pattern.
        """
        num_head_k = self.num_heads

        buffers = self._ensure_persistent_buffers(device, kind)

        self._get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_cpu,
            1,  # gqa_ratio: K is decompressed to num_heads, matching Q.
            num_head_k,
            buffers["work_metadata"],
            buffers["work_indptr"],
            buffers["work_info"],
            buffers["reduce_indptr"],
            buffers["reduce_final_map"],
            buffers["reduce_partial_map"],
            qhead_granularity=1,
            qlen_granularity=_FP8_PREFILL_TILE_Q,
            kvlen_granularity=_KVLEN_GRANULARITY,
            block_size=1,
            is_causal=is_causal,
        )

        # reduce_indptr[-1] counts the K-split partial tiles the PS
        # scheduler emitted. _run_kernel uses it to size the float32
        # logits/attn_lse scratch tensors, so it must be the live count
        # (the static upper bound from get_ps_metadata_info_v1 is sized
        # for worst-case batch * max_qlen and produces multi-TB
        # allocations on real shapes). The scheduler is deterministic in
        # chunk geometry and that geometry is identical across all layers
        # in a forward, so we cache the count per chunk_idx: first layer
        # syncs, subsequent layers hit the cache. New-tokens (chunk_idx
        # is None) runs once per forward from prepare_metadata and always
        # syncs directly.
        if chunk_idx is None:
            num_partial_tiles = int(buffers["reduce_indptr"][-1].item())
        else:
            cached = self._context_num_partial_tiles.get(chunk_idx)
            if cached is None:
                cached = int(buffers["reduce_indptr"][-1].item())
                self._context_num_partial_tiles[chunk_idx] = cached
            num_partial_tiles = cached

        return {
            "work_indptr": buffers["work_indptr"],
            "work_info": buffers["work_info"],
            "reduce_indptr": buffers["reduce_indptr"],
            "reduce_final_map": buffers["reduce_final_map"],
            "reduce_partial_map": buffers["reduce_partial_map"],
            "num_partial_tiles": num_partial_tiles,
            "max_q_len": max_qlen,
        }

    def _warmup_noncausal(self, device: torch.device) -> None:
        """JIT-compile the noncausal PS pipeline before the first real chunk.

        vLLM's profile_run only builds pure-prefill batches (context_len=0),
        so it exercises causal new-tokens but never the noncausal context
        chunk path. Without this, the first chunked-context request blocks
        all TP ranks on hipModuleLoad of the noncausal hsaco (and on
        module_ps_metadata's first build if the cache is cold), long enough
        to trigger shm_broadcast timeouts on collectives.

        Class-gated so only the first AiterAsmPrefillBackend instance on a
        (device, shape) tuple fires the kernel; later layer instances and
        later forward passes short-circuit.
        """
        warmup_key = (
            str(device),
            self.num_heads,
            self._ps_max_num_reqs,
            self._ps_max_qlen,
            self._ps_max_context_kvlen,
        )
        if warmup_key in type(self)._WARMUP_DONE:
            return
        type(self)._WARMUP_DONE.add(warmup_key)

        from vllm.platforms import current_platform

        fp8_dtype = current_platform.fp8_dtype()
        nhead = self.num_heads
        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        v_head_dim = self.v_head_dim
        tile_q = _FP8_PREFILL_TILE_Q
        # Minimum valid launch: 1 request, 1 Q-tile, 1 KV-granularity.
        total_q = tile_q
        total_k = _KVLEN_GRANULARITY

        qo_indptr_cpu = torch.tensor([0, total_q], dtype=torch.int32)
        kv_indptr_cpu = torch.tensor([0, total_k], dtype=torch.int32)
        seq_lens_cpu = torch.tensor([total_k], dtype=torch.int32)

        # Use the same _ps_max_* values production uses so the persistent
        # buffer set this allocates is the same one run_prefill_context_chunk
        # reuses later (no duplicate allocation).
        ps = self._build_ps_for_chunk(
            qo_indptr_cpu=qo_indptr_cpu,
            kv_indptr_cpu=kv_indptr_cpu,
            seq_lens_cpu=seq_lens_cpu,
            is_causal=False,
            device=device,
            kind="context",
            max_qlen=total_q,
        )
        ps["qo_indptr"] = qo_indptr_cpu.to(device)
        ps["kv_indptr"] = kv_indptr_cpu.to(device)
        ps["kv_indices"] = self._get_kv_indices_buf(device, total_k)

        q = torch.zeros((total_q, nhead, head_dim), dtype=fp8_dtype, device=device)
        k = torch.zeros((total_k, nhead, head_dim), dtype=fp8_dtype, device=device)
        v = torch.zeros((total_k, nhead, v_head_dim), dtype=fp8_dtype, device=device)
        out = torch.empty(
            (total_q, nhead, v_head_dim), dtype=torch.bfloat16, device=device
        )
        num_partial_tiles = ps["num_partial_tiles"]
        logits = torch.empty(
            (num_partial_tiles * tile_q, nhead, v_head_dim),
            dtype=torch.float32,
            device=device,
        )
        attn_lse = torch.empty(
            (num_partial_tiles * tile_q, nhead),
            dtype=torch.float32,
            device=device,
        )
        final_lse = torch.empty((total_q, nhead), dtype=torch.float32, device=device)
        one_scale = torch.ones((), dtype=torch.float32, device=device)

        self._mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            ps["qo_indptr"],
            ps["kv_indptr"],
            ps["kv_indices"],
            ps["work_indptr"],
            ps["work_info"],
            ps["max_q_len"],
            self.scale,
            False,
            logits,
            attn_lse,
            out,
            one_scale,
            one_scale,
            one_scale,
        )
        self._mla_reduce_v1(
            logits,
            attn_lse,
            ps["reduce_indptr"],
            ps["reduce_final_map"],
            ps["reduce_partial_map"],
            tile_q,
            out,
            final_lse,
        )
        logger.info(
            "AITER_ASM noncausal warmup complete (num_heads=%d, max_qlen=%d, "
            "max_kvlen=%d)",
            self.num_heads,
            self._ps_max_qlen,
            self._ps_max_context_kvlen,
        )

    def prepare_metadata(self, prefill_metadata: "MLACommonPrefillMetadata") -> None:
        super().prepare_metadata(prefill_metadata)

        # New forward: invalidate the per-chunk partial-tile cache. The
        # cached counts are only valid for the chunk geometry built into
        # the shared context buffers during this forward.
        self._context_num_partial_tiles.clear()

        qo_indptr = prefill_metadata.query_start_loc  # device int32 [bs+1]
        device = qo_indptr.device
        self._warmup_noncausal(device)
        # Use the CPU mirror the metadata builder already populated; avoids
        # one DtoH copy + host sync per layer per forward (~30x with DSV3).
        qo_indptr_cpu = prefill_metadata.query_start_loc_cpu.to(torch.int32)
        q_seq_lens_cpu = (qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).to(torch.int32)
        total_q = int(qo_indptr_cpu[-1].item())

        # New-tokens chunk (causal): K == Q. Context chunks are built
        # on-the-fly inside run_prefill_context_chunk; they share the context
        # buffer set and consume it before the next chunk builds over it.
        ps = self._build_ps_for_chunk(
            qo_indptr_cpu=qo_indptr_cpu,
            kv_indptr_cpu=qo_indptr_cpu,
            seq_lens_cpu=q_seq_lens_cpu,
            is_causal=True,
            device=device,
            kind="new_tokens",
            max_qlen=prefill_metadata.max_query_len,
        )
        ps["qo_indptr"] = qo_indptr
        ps["kv_indptr"] = qo_indptr
        ps["kv_indices"] = self._get_kv_indices_buf(device, total_q)
        self._new_tokens_ps = ps

        # Stash Q-side metadata for run_prefill_context_chunk to reuse.
        self._qo_indptr = qo_indptr
        self._qo_indptr_cpu = qo_indptr_cpu

    def _run_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ps: dict,
        is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the PS ASM kernel + reduce, returning `(out, lse)`.

        Output dtype matches `prefill_metadata.output_dtype` so the caller
        can feed it directly to `merge_attn_states` or copy it into the
        final `output` buffer.
        """
        # AITER ASM kernel requires V contiguous in (seq, head, v_head_dim).
        # cp_gather_cache produces V as a slice of a wider nope+rope buffer
        # (stride[1] = qk_head_dim, not v_head_dim), so force a copy here.
        v = v.contiguous()

        total_q = q.shape[0]
        nhead = self.num_heads
        v_head_dim = self.v_head_dim
        tile_q = _FP8_PREFILL_TILE_Q
        num_partial_tiles = ps["num_partial_tiles"]

        out_dtype = self._prefill_metadata.output_dtype
        assert out_dtype is not None
        out = torch.empty(
            (total_q, nhead, v_head_dim),
            dtype=out_dtype,
            device=q.device,
        )

        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

        # Allocate scratch directly instead of going through the workspace
        # manager. The manager is locked after profile_run + graph capture,
        # but profile_run never exercises a noncausal chunked-context shape,
        # so the lock-time size is too small and a later long-context request
        # silently gets an undersized view, causing GPU OOB writes. Direct
        # torch.empty hits the caching allocator, which amortizes reuse for
        # the common shapes after warmup.
        logits = torch.empty(
            (num_partial_tiles * tile_q, nhead, v_head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        attn_lse = torch.empty(
            (num_partial_tiles * tile_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        final_lse = torch.empty(
            (total_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )

        self._mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            ps["qo_indptr"],
            ps["kv_indptr"],
            ps["kv_indices"],
            ps["work_indptr"],
            ps["work_info"],
            ps["max_q_len"],
            self.scale,
            is_causal,
            logits,
            attn_lse,
            out,
            # Per-tensor dequant scales for Q/K/V passed to the AITER FP8
            # ASM kernel. We hardcode 1.0 because this backend currently
            # assumes inputs are in real units (no per-tensor FP8 dequant
            # scale to undo). Supporting q_scale/k_scale/v_scale != 1 would
            # require plumbing the layer's _q_scale/_k_scale/_v_scale tensors
            # through prepare_metadata or a bind step. The test suite is
            # expected to skip non-unit scales for this backend.
            one_scale,
            one_scale,
            one_scale,
        )

        self._mla_reduce_v1(
            logits,
            attn_lse,
            ps["reduce_indptr"],
            ps["reduce_final_map"],
            ps["reduce_partial_map"],
            tile_q,
            out,
            final_lse,
        )

        # AITER mla_reduce_v1 writes final_lse as (total_q, num_heads) with a
        # hardcoded byte offset (seq_idx * num_heads + head_idx) and no stride
        # parameter, so we cannot ask it to write transposed. The downstream
        # consumer on ROCm is triton_merge_attn_states, which indexes
        # (head_idx * num_tokens + token_idx) and requires the buffer
        # contiguous in (num_heads, total_q) layout. Hence transpose +
        # contiguous; can't be elided without changing one of the kernels.
        return out, final_lse.transpose(0, 1).contiguous()

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self._new_tokens_ps is not None, (
            "prepare_metadata must be called before run_prefill_new_tokens"
        )
        out, lse = self._run_kernel(q, k, v, self._new_tokens_ps, is_causal=True)
        if return_softmax_lse:
            return out, lse
        return out

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-causal ASM prefill over one cached-context chunk.

        Q is the full new-tokens query (same across chunks). K/V are the
        chunk's cached context, already loaded into contiguous tensors by
        the parent ``_compute_prefill_context`` loop. Per-chunk PS metadata
        is built into the shared context buffer set; sequential chunk
        execution on the default stream means the previous chunk's kernels
        have already read the buffer before the next chunk overwrites it.
        """
        assert self._prefill_metadata.chunked_context is not None
        cc = self._prefill_metadata.chunked_context

        # Use the CPU mirror surfaced by the metadata builder; avoids one
        # DtoH copy + host sync per layer per chunk.
        kv_indptr = cc.cu_seq_lens[chunk_idx]
        kv_indptr_cpu = cc.cu_seq_lens_cpu[chunk_idx].to(torch.int32)
        k_seq_lens_cpu = (kv_indptr_cpu[1:] - kv_indptr_cpu[:-1]).to(torch.int32)
        total_k = int(kv_indptr_cpu[-1].item())

        ps = self._build_ps_for_chunk(
            qo_indptr_cpu=self._qo_indptr_cpu,
            kv_indptr_cpu=kv_indptr_cpu,
            seq_lens_cpu=k_seq_lens_cpu,
            is_causal=False,
            device=q.device,
            kind="context",
            max_qlen=self._prefill_metadata.max_query_len,
            chunk_idx=chunk_idx,
        )
        ps["qo_indptr"] = self._qo_indptr
        ps["kv_indptr"] = kv_indptr
        ps["kv_indices"] = self._get_kv_indices_buf(q.device, total_k)

        return self._run_kernel(q, k, v, ps, is_causal=False)
