# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER ASM FP8 backend for MLA prefill on AMD gfx950 (MI350).

Dispatches through aiter.mla_prefill_ps_asm_fwd -> aiter.mla_reduce_v1.
"""

import math
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
# Compute-unit count on gfx950/MI350 (multi_processor_count is always 256).
# Hardcoded because torch.cuda.get_device_properties() is slow to call on the
# per-forward metadata path; the value is fixed for the only supported arch.
_GFX950_CU_NUM = 256


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

        # Populated by prepare_metadata before each forward pass: the
        # new-tokens (causal) PS metadata dict, and the list of per-context-
        # chunk (noncausal) PS metadata dicts indexed by chunk_idx. Each is
        # built once per forward, right-sized to actual geometry, and reused
        # by every MLA layer that forward.
        self._new_tokens_ps: dict | None = None
        self._context_ps: list[dict] = []

        # Warmup-gate shape constants. These no longer size any persistent
        # buffer (PS scratch is now allocated per-forward, right-sized to
        # actual geometry in _build_owned_ps_for_chunk); they only key and
        # log the one-time noncausal JIT warmup. _ps_max_context_kvlen is the
        # chunked-prefill workspace size, the upper bound on a context chunk's
        # per-sequence K length.
        self._ps_max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self._ps_max_qlen = vllm_config.scheduler_config.max_num_batched_tokens
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

    def _build_owned_ps_for_chunk(
        self,
        qo_indptr_cpu: torch.Tensor,
        kv_indptr_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        is_causal: bool,
        device: torch.device,
        max_qlen: int,
        max_kvlen: int | None,
    ) -> dict:
        """Build PS metadata into freshly allocated, right-sized buffers.

        Unlike a persistent worst-case buffer shared across the whole run,
        each call allocates scratch sized to THIS chunk's actual geometry
        (batch_size, max_qlen, max_kvlen) via get_ps_metadata_info_v1, then
        fills it with get_ps_metadata_v1. The caller is expected to invoke
        this once per (new-tokens / context-chunk) per forward, in
        prepare_metadata, and stash the returned dict so all ~30 MLA layers
        reuse it. That collapses the C++ get_ps_metadata_v1 call (and its 5
        pageable H2D copies, each of which syncs the stream) from once per
        layer per chunk to once per chunk per forward.

        ``seq_lens_cpu`` drives the PS scheduler's K-side work split.
        ``max_kvlen`` is the max KV length per sequence in this chunk; pass
        None for the causal new-tokens chunk (K == Q, so it falls back to
        max_qlen).

        num_partial_tiles (which sizes the per-call logits/attn_lse scratch in
        _run_kernel) is computed from a tight host-side bound, NOT from the
        info call's reduce_partial_map_size. The info bound is
        qo_tile_cnt * (max_kv_split + cus_per_cluster); its max_kv_split =
        ceil(max_kvlen/128) factor assumes every q-tile splits into max_kv_split
        partials, but the scheduler distributes total KV units across thread
        groups and emits at most one carryover work-item per TG, so the true
        per-cluster partial count is QT + tgs_per_cluster, independent of KV
        length (QT = sum over sequences of ceil(qlen/qlen_granularity)). For a
        long-context chunk at large batch the info bound over-counts by ~256x,
        which made logits balloon to hundreds of GB and fault on allocation at
        high concurrency. The tight bound keeps it proportional to the query
        side only. Both QT and tgs_per_cluster are pure host arithmetic, so the
        sizing path stays sync-free.

        Buffers are read-only after the build: the kernel reads work_indptr/
        work_info/reduce_* and writes only the per-call logits/attn_lse/out
        allocated in _run_kernel. Sharing the same metadata across all layers
        in a forward is therefore safe; vLLM runs prepare_metadata once before
        any layer executes.
        """
        num_head_k = self.num_heads

        (
            (work_metadata_size, work_metadata_dtype),
            (work_indptr_size, work_indptr_dtype),
            (work_info_size, work_info_dtype),
            (reduce_indptr_size, reduce_indptr_dtype),
            (reduce_final_map_size, reduce_final_map_dtype),
            (reduce_partial_map_size, reduce_partial_map_dtype),
        ) = self._get_ps_metadata_info_v1(
            batch_size=qo_indptr_cpu.numel() - 1,
            num_head_k=num_head_k,
            max_qlen=max_qlen,
            qlen_granularity=_FP8_PREFILL_TILE_Q,
            max_kvlen=max_kvlen,
            kvlen_granularity=_KVLEN_GRANULARITY,
        )

        work_metadata = torch.empty(
            work_metadata_size, dtype=work_metadata_dtype, device=device
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_dtype, device=device
        )
        work_info = torch.empty(*work_info_size, dtype=work_info_dtype, device=device)
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_dtype, device=device
        )
        reduce_final_map = torch.empty(
            *reduce_final_map_size, dtype=reduce_final_map_dtype, device=device
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_dtype, device=device
        )

        self._get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_cpu,
            1,  # gqa_ratio: K is decompressed to num_heads, matching Q.
            num_head_k,
            work_metadata,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            qhead_granularity=1,
            qlen_granularity=_FP8_PREFILL_TILE_Q,
            kvlen_granularity=_KVLEN_GRANULARITY,
            block_size=1,
            is_causal=is_causal,
        )

        # Tight, sync-free upper bound on the per-cluster partial-tile count:
        # QT (sum over sequences of ceil(qlen/qlen_granularity)) plus one
        # carryover per thread group in a cluster. partial_o_loc resets per
        # cluster and is identical across clusters, so this bounds the highest
        # partial slot the kernel writes; logits/attn_lse are sized to
        # num_partial_tiles * tile_q rows. Computed on the CPU qo_indptr (no
        # device read). See the method docstring for why this replaces the
        # info call's loose reduce_partial_map_size.
        q_seq_lens = qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]
        qt = int(
            ((q_seq_lens + (_FP8_PREFILL_TILE_Q - 1)) // _FP8_PREFILL_TILE_Q)
            .sum()
            .item()
        )
        tgs_per_cluster = _GFX950_CU_NUM // math.gcd(num_head_k, _GFX950_CU_NUM)
        num_partial_tiles = qt + tgs_per_cluster

        # Temporary: compare the tight bound against the info call's loose
        # reduce_partial_map_size to confirm the old sizing was overly
        # pessimistic. Remove once validated.
        logger.debug(
            "ps num_partial_tiles: tight=%d (qt=%d + tgs_per_cluster=%d) "
            "vs loose reduce_partial_map_size=%d",
            num_partial_tiles,
            qt,
            tgs_per_cluster,
            int(reduce_partial_map_size),
        )

        return {
            "work_indptr": work_indptr,
            "work_info": work_info,
            "reduce_indptr": reduce_indptr,
            "reduce_final_map": reduce_final_map,
            "reduce_partial_map": reduce_partial_map,
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

        # Build right-sized noncausal metadata just to exercise the JIT/hsaco
        # load path; the buffers are discarded after this warmup call.
        ps = self._build_owned_ps_for_chunk(
            qo_indptr_cpu=qo_indptr_cpu,
            kv_indptr_cpu=kv_indptr_cpu,
            seq_lens_cpu=seq_lens_cpu,
            is_causal=False,
            device=device,
            max_qlen=total_q,
            max_kvlen=total_k,
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

        qo_indptr = prefill_metadata.query_start_loc  # device int32 [bs+1]
        device = qo_indptr.device
        self._warmup_noncausal(device)
        # Use the CPU mirror the metadata builder already populated; avoids
        # one DtoH copy + host sync per layer per forward (~30x with DSV3).
        qo_indptr_cpu = prefill_metadata.query_start_loc_cpu.to(torch.int32)
        q_seq_lens_cpu = (qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).to(torch.int32)
        total_q = int(qo_indptr_cpu[-1].item())
        max_query_len = prefill_metadata.max_query_len

        # New-tokens chunk (causal): K == Q. Built once here and reused by
        # every MLA layer's run_prefill_new_tokens this forward.
        ps = self._build_owned_ps_for_chunk(
            qo_indptr_cpu=qo_indptr_cpu,
            kv_indptr_cpu=qo_indptr_cpu,
            seq_lens_cpu=q_seq_lens_cpu,
            is_causal=True,
            device=device,
            max_qlen=max_query_len,
            max_kvlen=None,
        )
        ps["qo_indptr"] = qo_indptr
        ps["kv_indptr"] = qo_indptr
        ps["kv_indices"] = self._get_kv_indices_buf(device, total_q)
        self._new_tokens_ps = ps

        # Context chunks (noncausal): build every chunk's PS metadata once
        # here, right-sized to that chunk's geometry, and stash by chunk index.
        # run_prefill_context_chunk(chunk_idx=i) then just reads the prebuilt
        # entry. Both _compute_prefill_context paths iterate chunk indices
        # 0..len(seq_tot)-1 with the same cu_seq_lens[i] geometry used here,
        # so per-index reuse is exact. Building all chunks upfront collapses
        # the per-layer-per-chunk get_ps_metadata_v1 syncs to one per chunk.
        self._context_ps = []
        cc = prefill_metadata.chunked_context
        if cc is not None:
            for chunk_idx in range(len(cc.seq_tot)):
                kv_indptr = cc.cu_seq_lens[chunk_idx]
                kv_indptr_cpu = cc.cu_seq_lens_cpu[chunk_idx].to(torch.int32)
                k_seq_lens_cpu = (kv_indptr_cpu[1:] - kv_indptr_cpu[:-1]).to(
                    torch.int32
                )
                total_k = int(kv_indptr_cpu[-1].item())
                chunk_ps = self._build_owned_ps_for_chunk(
                    qo_indptr_cpu=qo_indptr_cpu,
                    kv_indptr_cpu=kv_indptr_cpu,
                    seq_lens_cpu=k_seq_lens_cpu,
                    is_causal=False,
                    device=device,
                    max_qlen=max_query_len,
                    max_kvlen=cc.max_seq_lens[chunk_idx],
                )
                chunk_ps["qo_indptr"] = qo_indptr
                chunk_ps["kv_indptr"] = kv_indptr
                chunk_ps["kv_indices"] = self._get_kv_indices_buf(device, total_k)
                self._context_ps.append(chunk_ps)

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
        the parent ``_compute_prefill_context`` loop. The chunk's PS metadata
        was built once in prepare_metadata (right-sized to this chunk's
        geometry) and stashed at ``self._context_ps[chunk_idx]``; all MLA
        layers in this forward reuse it, so the per-layer get_ps_metadata_v1
        sync is gone.
        """
        ps = self._context_ps[chunk_idx]
        return self._run_kernel(q, k, v, ps, is_causal=False)
