# SPDX-License-Identifier: Apache-2.0
"""TurboQuant shared dequantization buffer manager (Patch 22 + 23 as class).

Problem (root cause drilled to 5-WHY):
    SYMPTOM: CUDA OOM at ~234k context: "tried 88 MiB, 42 MiB free".

    WHY-1: Engine allocates ~900 MiB dequant buffer at long-context prefill.
    WHY-2: This allocation happens INSIDE `_continuation_prefill` forward path.
    WHY-3: Forward-path allocations are invisible to vLLM's memory profiler.
    WHY-4: Profiler runs uniform dummy batch during warmup — never triggers
           the `cached_len > THRESHOLD` branch that allocates the buffer.
    WHY-5: KV cache sizing formula: (total_vram × gpu_util) - max_memory_allocated().
           Profiler-invisible allocations aren't counted → KV cache sized too
           generously → production OOM when real 234k request arrives.

Root cause location:
    vllm/v1/attention/backends/turboquant_attn.py:_continuation_prefill
    Lines that lazily `torch.empty(buf_shape, ...)` on first long-context hit.

Fix (this module):
    1. Pre-allocate K/V dequant buffers in `_ensure_on_device` (which fires
       during profile_run warmup → profiler-visible → KV cache sized correctly).
    2. Share single buffer pair across all attention layers (sequential forward
       per layer = one buffer sufficient, avoids N_layers × 256MB over-alloc).
    3. Pre-allocate cu_seqlens scratch tensors (Patch 23 bundled) to avoid
       per-call `torch.tensor([0, q_len])` host→device creation.

Prior art & credits:
    - @JartX (TurboQuant author, JartX/vllm#11) — FP16 rotation fix prerequisite
      that flattened the OOM cliff from 185k → 234k, making residual spike findable
    - @jhsmith409 (vllm contributor) — endorsed investigation: "Impressive work.
      Thank you. Happy to proceed however the maintainers like."
    - @youkaichao (vLLM core team) — memory profiler invariant canonical pattern

Platform compatibility:
    - NVIDIA CUDA:  ✅ Primary target (TurboQuant is CUDA-only upstream)
    - AMD ROCm:     💤 Skip (TurboQuant not ported to ROCm)
    - Intel XPU:    💤 Skip
    - CPU:          💤 Skip

Memory math (Qwen3.6-35B-A3B production config):
    TP=2 → num_kv_heads per rank = 2 (from total 4)
    head_size = 128
    max_model_len = 262144
    bf16 → 2 bytes per element
    Per buffer: 2 × 128 × 262144 × 2 = 128 MiB
    K+V combined: 256 MiB (shared across 10 attention layers, NOT per-layer)

    Without Patch 22: 10 layers × lazy 128 MiB = could peak at any point →
        1.28 GB invisible to profiler → 88 MiB OOM when colliding with
        KV cache allocation.

    With Patch 22: 256 MiB visible at profile_run → KV cache sized accounting
        for it → no OOM.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.0 IMPLEMENTED
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

log = logging.getLogger("genesis.dequant_buffer")


# P26: env-driven budget override. MUST be resolved at module-import time
# because `acquire_prefill_output` is reachable from TurboQuant's
# `_prefill_attention` forward path, and `torch.dynamo` rejects `os.environ`
# access inside a traced region with "can't handle functions not
# implemented in python". Caching once here is dynamo-safe.
_ENV_TQ_MAX_BT = "GENESIS_TQ_MAX_BATCHED_TOKENS"


def _read_tq_env_budget() -> Optional[int]:
    env = os.environ.get(_ENV_TQ_MAX_BT, "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    return None


_TQ_ENV_BUDGET: Optional[int] = _read_tq_env_budget()


# ═══════════════════════════════════════════════════════════════════════════
#                            HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def estimate_buffer_bytes(
    num_kv_heads: int,
    head_size: int,
    max_alloc_len: int,
    dtype: torch.dtype,
) -> int:
    """Estimate bytes for K+V buffer pair.

    Used for warnings on large footprint configs (e.g. TP=1 + 256k context).

    Args:
        num_kv_heads: Per-rank after TP split.
        head_size: Per-head feature dimension.
        max_alloc_len: Maximum sequence length buffer needs.
        dtype: Element type.

    Returns:
        Estimated combined K+V buffer bytes.
    """
    # PyTorch element_size requires a tensor instance; use lookup table
    dtype_bytes = {
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
    }
    bytes_per = dtype_bytes.get(dtype, 2)  # default 2 for unknown

    per_buffer = num_kv_heads * head_size * max_alloc_len * bytes_per
    return 2 * per_buffer  # K + V


def _humanize_bytes(n: int) -> str:
    """Convert byte count to human-readable string."""
    for unit, power in [("B", 0), ("KiB", 10), ("MiB", 20), ("GiB", 30)]:
        if n < (1 << (power + 10)):
            if power == 0:
                return f"{n} B"
            return f"{n / (1 << power):.2f} {unit}"
    return f"{n / (1 << 30):.2f} GiB"


# ═══════════════════════════════════════════════════════════════════════════
#                      TurboQuantBufferManager
# ═══════════════════════════════════════════════════════════════════════════

class TurboQuantBufferManager:
    """Class-level shared buffer cache for TQ K/V dequantization.

    Thread-safety: each worker process has its own class state. TP ranks
    each initialize their own cache. Not shared across ranks.

    Lifecycle:
      1. Engine init → Attention.__init__ → _ensure_on_device() [fires
         during profile_run warmup]
      2. First call → allocate fresh buffer via GenesisPreallocBuffer
         (profiler sees it)
      3. Subsequent layers with same (Hk, D, device) → cached buffer returned
         (pointer-stable → CUDA graph safe)
      4. Process shutdown → GC cleans up (no explicit cleanup needed)

    Anti-patterns (WILL break):
      - Do NOT call from inside forward() for new namespaces (not
        profiler-visible, leads to #40420 class of OOM).
      - Do NOT modify existing tensor shapes (allocate new namespace instead).
    """

    # Cache maps (num_kv_heads, head_size, max_alloc_len, device_str) → tensors
    _K_BUFFERS: dict[tuple, torch.Tensor] = {}
    _V_BUFFERS: dict[tuple, torch.Tensor] = {}
    _CU_Q_BUFFERS: dict[str, torch.Tensor] = {}
    _CU_K_BUFFERS: dict[str, torch.Tensor] = {}
    # P32: _cu_2 scratch — reused 2-element int32 tensor for
    # flash_attn_varlen_func second-hop cu_seqlens.
    _CU_2_BUFFERS: dict[str, torch.Tensor] = {}
    # P33: synth_seq_lens — synthetic per-batch seq_lens (int32) used by
    # the TQ decode path when the real seq_lens are CPU-only and need a
    # stable device-side mirror. Keyed by (max_batch, device).
    _SYNTH_SEQ_LENS_BUFFERS: dict[tuple, torch.Tensor] = {}
    # P26: prefill output buffer — pre-allocated per (max_batched_tokens,
    # num_q_heads, head_size, dtype, device), reused across prefill calls.
    # Saves per-call torch.zeros(N, Hq, D) allocator churn + zero-fill.
    _PREFILL_OUT_BUFFERS: dict[tuple, torch.Tensor] = {}
    # P36: DECODE intermediate buffers — shared across ALL TurboQuant
    # attention layers. Upstream (pre-PR #40655) registers these per-layer
    # via `register_buffer` → 30+ allocator calls on our 10-layer hybrid,
    # 180+ on Qwen3-32B dense (see PR #40655 body: "~16 GiB direct +
    # ~45 GiB allocator fragmentation" on that config). Since all TQ
    # layers execute SEQUENTIALLY per step, one shared set is safe.
    #   - _tq_mid_o_buf: (B, Hq, S, D+1) fp32 — KV-split accumulator
    #   - _tq_output_buf: (B, Hq, D) fp32   — final per-head output
    #   - _tq_lse_buf:    (B, Hq) fp32      — logsumexp scratch
    _DECODE_MID_O_BUFFERS: dict[tuple, torch.Tensor] = {}
    _DECODE_OUTPUT_BUFFERS: dict[tuple, torch.Tensor] = {}
    _DECODE_LSE_BUFFERS: dict[tuple, torch.Tensor] = {}
    # P38: CONTINUATION-PREFILL K/V dequant buffers (4-D, matching dev134
    # `turboquant_attn.py:_continuation_prefill` internal shape
    # `(1, Hk, alloc_len, D)` exactly — supersedes P22's legacy 3-D
    # prealloc, which dev134 silently ignores because of the shape check
    # `k_buf.shape[2] < alloc_len` on a 3-D vs a 4-D tensor).
    #   Key: (Hk, D, max_alloc_len, device_str, dtype_str)
    _P38_K_DEQUANT_4D_BUFFERS: dict[tuple, torch.Tensor] = {}
    _P38_V_DEQUANT_4D_BUFFERS: dict[tuple, torch.Tensor] = {}
    # P38: K_full / V_full workspace — persistent (cached_len + q_len, Hk, D)
    # FP16 buffers that REPLACE the per-call `torch.cat([...trim..., chunk])`
    # in `_continuation_prefill`. One buffer pair per (Hk, D, max_seq_cap,
    # device, dtype) shared across ALL TQ layers since layers run
    # sequentially in a forward pass. Eliminates the ~500 MiB transient
    # peak that saturates GPU memory at deep prefix prefill on dev134.
    _P38_K_FULL_BUFFERS: dict[tuple, torch.Tensor] = {}
    _P38_V_FULL_BUFFERS: dict[tuple, torch.Tensor] = {}

    @classmethod
    def should_apply(cls) -> bool:
        """Platform guard — TurboQuant is NVIDIA CUDA + SM 8.0+ only.

        Returns False on:
          - AMD ROCm (no TurboQuant kernel port)
          - Intel XPU (no TurboQuant kernel port)
          - CPU (no GPU memory path)
          - NVIDIA pre-Ampere (SM < 8.0, TurboQuant not supported)

        Returns True on:
          - NVIDIA Ampere (8.0, 8.6) — primary target
          - NVIDIA Ada (8.9) — works
          - NVIDIA Hopper (9.0) — works
          - NVIDIA Blackwell (10.0) — works
        """
        from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least
        if not is_nvidia_cuda():
            return False
        if not is_sm_at_least(8, 0):
            return False
        return True

    @classmethod
    def get_or_create_kv_buffers(
        cls,
        num_kv_heads: int,
        head_size: int,
        max_alloc_len: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (K, V) shared dequant buffers for this layer's shape.

        Args:
            num_kv_heads: Per-rank (after TP split).
            head_size: Per-head feature dimension.
            max_alloc_len: Max sequence length we'll ever need — use
                `vllm_config.model_config.max_model_len` rounded up to
                1024 alignment.
            device: Target GPU device.
            dtype: Typically bf16 matching model compute dtype.

        Returns:
            Tuple (K_buffer, V_buffer), both of shape
            (num_kv_heads, head_size, max_alloc_len), or (None, None) if
            platform incompatible.
        """
        if not cls.should_apply():
            return None, None

        key = (num_kv_heads, head_size, max_alloc_len, str(device))

        if key not in cls._K_BUFFERS:
            # Footprint warning for tight-VRAM configs
            buf_bytes = estimate_buffer_bytes(
                num_kv_heads, head_size, max_alloc_len, dtype
            )
            if buf_bytes > (2 << 30):  # > 2 GiB
                try:
                    world_size = torch.distributed.get_world_size()
                except RuntimeError:
                    world_size = 1
                if world_size == 1:
                    log.warning(
                        "[TQ buffer] Large K+V buffer pair %s on TP=1. "
                        "Consider --tensor-parallel-size>=2 or smaller "
                        "max-model-len. Anyway proceeding.",
                        _humanize_bytes(buf_bytes),
                    )

            shape = (num_kv_heads, head_size, max_alloc_len)

            # Use GenesisPreallocBuffer framework (consistent tracking across
            # all Genesis prealloc — appears in get_registry_info())
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

            cls._K_BUFFERS[key] = GPB.get_or_create(
                namespace=f"tq_k_dequant|{key}",
                shape=shape, dtype=dtype, device=device,
            )
            cls._V_BUFFERS[key] = GPB.get_or_create(
                namespace=f"tq_v_dequant|{key}",
                shape=shape, dtype=dtype, device=device,
            )

            log.info(
                "[TQ buffer] Allocated shared K+V dequant buffers: "
                "num_kv_heads=%d head_size=%d max_len=%d dtype=%s device=%s "
                "total=%s",
                num_kv_heads, head_size, max_alloc_len, dtype, device,
                _humanize_bytes(buf_bytes),
            )

        return cls._K_BUFFERS[key], cls._V_BUFFERS[key]

    @classmethod
    def get_or_create_cu_seqlens(
        cls,
        device: torch.device | str,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return reusable cu_seqlens scratch tensors for flash_attn_varlen_func.

        These int32[2] tensors store [0, q_len] and [0, seq_len]. Pre-allocated
        once per device to avoid per-call `torch.tensor([0, q_len])` which
        creates a fresh host→device transfer × 2 per attention layer per
        continuation call.

        Usage in forward path:
            cu_q, cu_k = TurboQuantBufferManager.get_or_create_cu_seqlens(device)
            cu_q[1] = q_len  # in-place, pointer-stable
            cu_k[1] = seq_len
            flash_attn_varlen_func(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, ...)

        Returns:
            Tuple (cu_q, cu_k) of int32 tensors shape (2,), zero-initialized.
            Both None if platform incompatible.
        """
        if not cls.should_apply():
            return None, None

        dev_key = str(device)

        if dev_key not in cls._CU_Q_BUFFERS:
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

            cls._CU_Q_BUFFERS[dev_key] = GPB.get_or_create(
                namespace=f"tq_cu_q|{dev_key}",
                shape=(2,), dtype=torch.int32, device=device,
                zero_init=True,
            )
            cls._CU_K_BUFFERS[dev_key] = GPB.get_or_create(
                namespace=f"tq_cu_k|{dev_key}",
                shape=(2,), dtype=torch.int32, device=device,
                zero_init=True,
            )

        return cls._CU_Q_BUFFERS[dev_key], cls._CU_K_BUFFERS[dev_key]

    @classmethod
    def get_or_create_cu_2(
        cls,
        device: torch.device | str,
    ) -> Optional[torch.Tensor]:
        """P32: scratch cu_seqlens tensor (shape (2,), int32) for second-hop
        varlen-attn calls. Pre-allocating dodges a per-call host→device
        transfer; pointer-stable so CUDA-graph-safe.

        Returns None if platform incompatible (CPU / ROCm / XPU).
        """
        if not cls.should_apply():
            return None

        dev_key = str(device)
        if dev_key not in cls._CU_2_BUFFERS:
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
            cls._CU_2_BUFFERS[dev_key] = GPB.get_or_create(
                namespace=f"tq_cu_2|{dev_key}",
                shape=(2,), dtype=torch.int32, device=device,
                zero_init=True,
            )
        return cls._CU_2_BUFFERS[dev_key]

    @classmethod
    def get_or_create_synth_seq_lens(
        cls,
        max_batch: int,
        device: torch.device | str,
    ) -> Optional[torch.Tensor]:
        """P33: scratch synthetic seq_lens tensor (int32, shape (max_batch,)).

        Used by the TurboQuant decode path when the scheduler's seq_lens
        tensor is on CPU and needs a device-side mirror. Pre-allocating it
        avoids per-iteration `.to(device)` and fresh allocations inside the
        forward, which are invisible to vLLM's memory profiler.

        Args:
            max_batch: Upper bound on batch size at decode time. Typically
                `vllm_config.scheduler_config.max_num_seqs` or rounded up.
            device: Target GPU device.

        Returns None if platform incompatible.
        """
        if not cls.should_apply():
            return None
        # Round up to 8 for alignment-friendly size (scheduler may request
        # batches up to max_num_seqs; small over-alloc is fine).
        rounded = ((max_batch + 7) // 8) * 8
        key = (rounded, str(device))
        if key not in cls._SYNTH_SEQ_LENS_BUFFERS:
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
            cls._SYNTH_SEQ_LENS_BUFFERS[key] = GPB.get_or_create(
                namespace=f"tq_synth_seq_lens|{key}",
                shape=(rounded,), dtype=torch.int32, device=device,
                zero_init=True,
            )
        return cls._SYNTH_SEQ_LENS_BUFFERS[key]

    @classmethod
    def get_or_create_prefill_output(
        cls,
        max_batched_tokens: int,
        num_q_heads: int,
        head_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """P26: prefill output scratch buffer (N_max, Hq, D).

        Pre-allocates the prefill path's `torch.zeros(N, Hq, D)` target
        so each prefill call reuses the same memory (pointer-stable) and
        only zeros the active slice `[:N]` via in-place `zero_()`.

        Args:
            max_batched_tokens: Upper bound on N for any single prefill.
                Should be `scheduler_config.max_num_batched_tokens`.
            num_q_heads: Per-rank query head count (after TP split).
            head_size: Per-head feature dim.
            device: Target GPU.
            dtype: Model compute dtype.

        Returns None if platform incompatible.
        """
        if not cls.should_apply():
            return None
        key = (max_batched_tokens, num_q_heads, head_size, str(device), str(dtype))
        if key not in cls._PREFILL_OUT_BUFFERS:
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
            cls._PREFILL_OUT_BUFFERS[key] = GPB.get_or_create(
                namespace=f"tq_prefill_out|{key}",
                shape=(max_batched_tokens, num_q_heads, head_size),
                dtype=dtype, device=device, zero_init=True,
            )
        return cls._PREFILL_OUT_BUFFERS[key]

    @classmethod
    def acquire_prefill_output(
        cls,
        num_tokens: int,
        num_q_heads: int,
        head_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
        max_batched_tokens: int | None = None,
    ) -> torch.Tensor:
        """P26 forward-path entry: get a zeroed (num_tokens, Hq, D) view.

        Replaces the per-call `torch.zeros(N, Hq, D)` in TurboQuant
        `_prefill_attention`. Returns a slice of a pointer-stable pool
        (CUDA graph safe) on the first allocation; later calls reuse the
        same memory and only zero the live `[:num_tokens]` prefix.

        If platform incompatible OR num_tokens exceeds the pool budget,
        falls back to a fresh allocation to preserve correctness. The
        budget can be raised by passing `max_batched_tokens` explicitly
        (usually `scheduler_config.max_num_batched_tokens`).

        Args:
            num_tokens: Actual batch size for this call.
            num_q_heads, head_size: Per-rank model dims.
            device, dtype: target tensor spec.
            max_batched_tokens: Optional override for the pool size;
                defaults to 4096 if neither env nor impl config is set.
        """
        # NO `os.environ` access here — dynamo rejects it inside traced
        # regions. The env var is read once at module import into
        # `_TQ_ENV_BUDGET` above.
        if max_batched_tokens is not None and max_batched_tokens > 0:
            max_n = int(max_batched_tokens)
        elif _TQ_ENV_BUDGET is not None:
            max_n = _TQ_ENV_BUDGET
        else:
            # [Genesis P73 fix v7.42] Was hardcoded 4096 — caused chunk-overflow
            # at runtime when scheduler dispatched chunk > 4096 (P28 incident).
            # Now consult central resolver (which probes vllm scheduler_config).
            try:
                from vllm._genesis.prealloc_budget import resolve_token_budget
                max_n = resolve_token_budget(domain_env=_ENV_TQ_MAX_BT)
            except Exception:
                max_n = 4096  # final safety net

        if num_tokens > max_n or not cls.should_apply():
            # Fallback: fresh allocation (correctness over throughput)
            return torch.zeros(
                (num_tokens, num_q_heads, head_size),
                device=device, dtype=dtype,
            )

        buf = cls.get_or_create_prefill_output(
            max_batched_tokens=max_n,
            num_q_heads=num_q_heads,
            head_size=head_size,
            device=device, dtype=dtype,
        )
        if buf is None:
            return torch.zeros(
                (num_tokens, num_q_heads, head_size),
                device=device, dtype=dtype,
            )
        slice_ = buf[:num_tokens]
        slice_.zero_()
        return slice_

    # ═══════════════════════════════════════════════════════════════════
    # P36 — Shared TurboQuant decode buffers (mirrors upstream PR #40655)
    # ═══════════════════════════════════════════════════════════════════

    @classmethod
    def get_shared_decode_mid_o(
        cls,
        max_num_seqs: int,
        num_q_heads: int,
        tq_max_kv_splits: int,
        head_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """Shared `_tq_mid_o_buf` across all TQ attention layers.

        Shape: `(max_num_seqs, num_q_heads, tq_max_kv_splits, head_size+1)`,
        dtype fp32 (required by TQ decode stage1 numerical stability).
        Returns None on non-NVIDIA / pre-Ampere (caller falls back to
        upstream per-layer `register_buffer` path).

        Invariant: all TQ attention layers in a given model share the
        exact same `(B, Hq, S, D)` config (same max_num_seqs, same head
        count, same S cap, same head_size). Sequential execution across
        layers makes sharing safe — no race, no CUDA-graph invalidation
        (pointer-stable).
        """
        if not cls.should_apply():
            return None
        key = (max_num_seqs, num_q_heads, tq_max_kv_splits, head_size,
               str(device), str(dtype))
        if key not in cls._DECODE_MID_O_BUFFERS:
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
            cls._DECODE_MID_O_BUFFERS[key] = GPB.get_or_create(
                namespace=f"tq_shared_decode_mid_o|{key}",
                shape=(max_num_seqs, num_q_heads, tq_max_kv_splits, head_size + 1),
                dtype=dtype, device=device, zero_init=False,
            )
        return cls._DECODE_MID_O_BUFFERS[key]

    @classmethod
    def get_shared_decode_output(
        cls,
        max_num_seqs: int,
        num_q_heads: int,
        head_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """Shared `_tq_output_buf` across all TQ attention layers.

        Shape: `(max_num_seqs, num_q_heads, head_size)`, fp32.
        """
        if not cls.should_apply():
            return None
        key = (max_num_seqs, num_q_heads, head_size, str(device), str(dtype))
        if key not in cls._DECODE_OUTPUT_BUFFERS:
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
            cls._DECODE_OUTPUT_BUFFERS[key] = GPB.get_or_create(
                namespace=f"tq_shared_decode_output|{key}",
                shape=(max_num_seqs, num_q_heads, head_size),
                dtype=dtype, device=device, zero_init=False,
            )
        return cls._DECODE_OUTPUT_BUFFERS[key]

    @classmethod
    def get_shared_decode_lse(
        cls,
        max_num_seqs: int,
        num_q_heads: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """Shared `_tq_lse_buf` across all TQ attention layers.

        Shape: `(max_num_seqs, num_q_heads)`, fp32.
        """
        if not cls.should_apply():
            return None
        key = (max_num_seqs, num_q_heads, str(device), str(dtype))
        if key not in cls._DECODE_LSE_BUFFERS:
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
            cls._DECODE_LSE_BUFFERS[key] = GPB.get_or_create(
                namespace=f"tq_shared_decode_lse|{key}",
                shape=(max_num_seqs, num_q_heads),
                dtype=dtype, device=device, zero_init=False,
            )
        return cls._DECODE_LSE_BUFFERS[key]

    @classmethod
    def acquire_cu_2(
        cls,
        device: torch.device | str,
    ) -> torch.Tensor:
        """P32 forward-path entry: get the reusable cu_seqlens-2 int32 scratch.

        Replaces the per-call `torch.zeros(2, ..., int32)` pattern. Pool
        is pointer-stable for CUDA-graph safety; callers write via
        `buf[1] = q_len` in-place. If platform incompatible, falls back
        to a fresh zero tensor.
        """
        if cls.should_apply():
            buf = cls.get_or_create_cu_2(device)
            if buf is not None:
                # Reset second slot — first slot is always 0 (cu_seqlens invariant)
                buf.zero_()
                return buf
        return torch.zeros(2, device=device, dtype=torch.int32)

    # ═══════════════════════════════════════════════════════════════════
    # P44 — Mixed-batch (decode + prefill) attn_out zero pool
    # ═══════════════════════════════════════════════════════════════════
    # Upstream `turboquant_attn.py:438` does
    #   attn_out = torch.zeros(N, num_heads, head_size, device, dtype=q.dtype)
    # in the decode+prefill mixed-batch branch. N up to
    # `max_num_batched_tokens` = 80 MB zero-init per forward in that
    # branch. P26 covers the PREFILL-ONLY path (line 566) but not this
    # mixed branch. Adding a per-(N_max, Hq, D, dtype) pool via the
    # same infra.

    _MIXED_ATTN_OUT_BUFFERS: dict[tuple, torch.Tensor] = {}

    @classmethod
    def get_or_create_mixed_attn_out(
        cls,
        max_num_batched_tokens: int,
        num_q_heads: int,
        head_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Persistent `(max_num_batched_tokens, Hq, D)` zero-init pool for
        the mixed-batch branch of `_forward`.

        Returns None on platform skip (caller falls back to per-call
        `torch.zeros`).
        """
        if not cls.should_apply():
            return None
        key = (
            max_num_batched_tokens, num_q_heads, head_size,
            str(device), str(dtype),
        )
        if key not in cls._MIXED_ATTN_OUT_BUFFERS:
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
            cls._MIXED_ATTN_OUT_BUFFERS[key] = GPB.get_or_create(
                namespace=f"tq_mixed_attn_out|{key}",
                shape=(max_num_batched_tokens, num_q_heads, head_size),
                dtype=dtype, device=device, zero_init=True,
            )
        return cls._MIXED_ATTN_OUT_BUFFERS[key]

    @classmethod
    def acquire_mixed_attn_out(
        cls,
        num_tokens: int,
        num_q_heads: int,
        head_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
        max_batched_tokens: int | None = None,
    ) -> torch.Tensor:
        """P44 forward-path entry for the MIXED-BATCH attn_out tensor.

        Replaces `torch.zeros(N, Hq, D, device, dtype)` in the mixed
        decode+prefill branch of `TurboQuantAttentionImpl._forward`.
        Returns a zeroed `[:num_tokens]` slice of a pointer-stable pool.

        On platform incompatibility OR overflow (num_tokens > budget),
        falls back to fresh `torch.zeros` — correctness preserved.
        """
        if max_batched_tokens is not None and max_batched_tokens > 0:
            max_n = int(max_batched_tokens)
        elif _TQ_ENV_BUDGET is not None:
            max_n = _TQ_ENV_BUDGET
        else:
            # [Genesis P73 fix v7.42] Was hardcoded 4096 — caused chunk-overflow
            # at runtime when scheduler dispatched chunk > 4096 (P28 incident).
            # Now consult central resolver (which probes vllm scheduler_config).
            try:
                from vllm._genesis.prealloc_budget import resolve_token_budget
                max_n = resolve_token_budget(domain_env=_ENV_TQ_MAX_BT)
            except Exception:
                max_n = 4096  # final safety net

        if num_tokens > max_n or not cls.should_apply():
            return torch.zeros(
                (num_tokens, num_q_heads, head_size),
                device=device, dtype=dtype,
            )
        buf = cls.get_or_create_mixed_attn_out(
            max_num_batched_tokens=max_n,
            num_q_heads=num_q_heads,
            head_size=head_size,
            device=device, dtype=dtype,
        )
        if buf is None:
            return torch.zeros(
                (num_tokens, num_q_heads, head_size),
                device=device, dtype=dtype,
            )
        slice_ = buf[:num_tokens]
        slice_.zero_()
        return slice_

    # ═══════════════════════════════════════════════════════════════════
    # P38 — Continuation-prefill K/V dequant + full-workspace buffers
    # ═══════════════════════════════════════════════════════════════════

    @classmethod
    def get_or_create_p38_dequant_4d(
        cls,
        num_kv_heads: int,
        head_size: int,
        max_alloc_len: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Persistent (1, Hk, max_alloc_len, D) K & V dequant buffers.

        These MATCH dev134 `_continuation_prefill`'s internal shape
        exactly, so the engine's `k_buf.shape[2] < alloc_len` check sees
        shape[2] == max_alloc_len and reuses our prealloc instead of
        falling back to fresh `torch.empty`.
        """
        if not cls.should_apply():
            return None, None
        key = (num_kv_heads, head_size, max_alloc_len, str(device), str(dtype))
        if key not in cls._P38_K_DEQUANT_4D_BUFFERS:
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
            shape = (1, num_kv_heads, max_alloc_len, head_size)
            cls._P38_K_DEQUANT_4D_BUFFERS[key] = GPB.get_or_create(
                namespace=f"tq_k_dequant_4d|{key}",
                shape=shape, dtype=dtype, device=device, zero_init=False,
            )
            cls._P38_V_DEQUANT_4D_BUFFERS[key] = GPB.get_or_create(
                namespace=f"tq_v_dequant_4d|{key}",
                shape=shape, dtype=dtype, device=device, zero_init=False,
            )
            buf_bytes = 2 * num_kv_heads * max_alloc_len * head_size * 2
            log.info(
                "[P38 dequant 4D] allocated K+V buffers (1,%d,%d,%d) %s on %s",
                num_kv_heads, max_alloc_len, head_size,
                _humanize_bytes(buf_bytes), device,
            )
        return (
            cls._P38_K_DEQUANT_4D_BUFFERS[key],
            cls._P38_V_DEQUANT_4D_BUFFERS[key],
        )

    @classmethod
    def get_or_create_p38_full(
        cls,
        num_kv_heads: int,
        head_size: int,
        max_seq_cap: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Persistent (max_seq_cap, Hk, D) K_full & V_full workspace.

        These buffers REPLACE the per-call
        `torch.cat([...cached_trim..., key_chunk], dim=0)` in
        `_continuation_prefill` with in-place `.copy_()` into pre-
        allocated memory. Shared across all TQ layers (sequential
        execution), so one pair is sufficient regardless of layer count.

        `max_seq_cap` should be `max_model_len + max_num_batched_tokens`
        so the last chunk (prefix ~= max_model_len - chunk) fits too.

        Returns (k_full, v_full) of shape (max_seq_cap, Hk, D), or
        (None, None) on platform incompatibility.
        """
        if not cls.should_apply():
            return None, None
        key = (num_kv_heads, head_size, max_seq_cap, str(device), str(dtype))
        if key not in cls._P38_K_FULL_BUFFERS:
            from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
            shape = (max_seq_cap, num_kv_heads, head_size)
            cls._P38_K_FULL_BUFFERS[key] = GPB.get_or_create(
                namespace=f"tq_k_full|{key}",
                shape=shape, dtype=dtype, device=device, zero_init=False,
            )
            cls._P38_V_FULL_BUFFERS[key] = GPB.get_or_create(
                namespace=f"tq_v_full|{key}",
                shape=shape, dtype=dtype, device=device, zero_init=False,
            )
            buf_bytes = 2 * max_seq_cap * num_kv_heads * head_size * 2
            log.info(
                "[P38 full] allocated K_full+V_full (%d,%d,%d) %s on %s",
                max_seq_cap, num_kv_heads, head_size,
                _humanize_bytes(buf_bytes), device,
            )
        return cls._P38_K_FULL_BUFFERS[key], cls._P38_V_FULL_BUFFERS[key]

    @classmethod
    def get_registry_info(cls) -> dict:
        """Diagnostic info for observability / logging.

        Usage:
            import json
            log.info("TQ buffers: %s",
                     json.dumps(TurboQuantBufferManager.get_registry_info(),
                                indent=2, default=str))
        """
        total_bytes = 0
        kv_entries = []
        for key, k_buf in cls._K_BUFFERS.items():
            v_buf = cls._V_BUFFERS.get(key)
            k_bytes = k_buf.element_size() * k_buf.numel()
            v_bytes = v_buf.element_size() * v_buf.numel() if v_buf else 0
            total_bytes += k_bytes + v_bytes
            kv_entries.append({
                "num_kv_heads": key[0],
                "head_size": key[1],
                "max_alloc_len": key[2],
                "device": key[3],
                "k_bytes": k_bytes,
                "v_bytes": v_bytes,
                "combined_human": _humanize_bytes(k_bytes + v_bytes),
            })

        # Aggregate auxiliary scratch-pool bytes (P32 _cu_2, P33
        # synth_seq_lens, P26 prefill output, P36 decode mid_o/output/lse)
        # so the "total_bytes" number reflects EVERYTHING this manager
        # keeps alive — not just K/V dequant.
        aux_bytes = 0
        for t in cls._CU_Q_BUFFERS.values():
            aux_bytes += t.element_size() * t.numel()
        for t in cls._CU_K_BUFFERS.values():
            aux_bytes += t.element_size() * t.numel()
        for t in cls._CU_2_BUFFERS.values():
            aux_bytes += t.element_size() * t.numel()
        for t in cls._SYNTH_SEQ_LENS_BUFFERS.values():
            aux_bytes += t.element_size() * t.numel()
        for t in cls._PREFILL_OUT_BUFFERS.values():
            aux_bytes += t.element_size() * t.numel()
        decode_bytes = 0
        decode_entries = []
        for key, t in cls._DECODE_MID_O_BUFFERS.items():
            b = t.element_size() * t.numel()
            decode_bytes += b
            decode_entries.append({
                "pool": "mid_o", "key": key, "bytes": b,
            })
        for key, t in cls._DECODE_OUTPUT_BUFFERS.items():
            b = t.element_size() * t.numel()
            decode_bytes += b
            decode_entries.append({"pool": "output", "key": key, "bytes": b})
        for key, t in cls._DECODE_LSE_BUFFERS.items():
            b = t.element_size() * t.numel()
            decode_bytes += b
            decode_entries.append({"pool": "lse", "key": key, "bytes": b})

        # P38: continuation-prefill 4-D dequant + K_full/V_full pools
        p38_bytes = 0
        p38_entries = []
        for key, t in cls._P38_K_DEQUANT_4D_BUFFERS.items():
            b = t.element_size() * t.numel()
            p38_bytes += b
            p38_entries.append({"pool": "k_dequant_4d", "key": key, "bytes": b})
        for key, t in cls._P38_V_DEQUANT_4D_BUFFERS.items():
            b = t.element_size() * t.numel()
            p38_bytes += b
            p38_entries.append({"pool": "v_dequant_4d", "key": key, "bytes": b})
        for key, t in cls._P38_K_FULL_BUFFERS.items():
            b = t.element_size() * t.numel()
            p38_bytes += b
            p38_entries.append({"pool": "k_full", "key": key, "bytes": b})
        for key, t in cls._P38_V_FULL_BUFFERS.items():
            b = t.element_size() * t.numel()
            p38_bytes += b
            p38_entries.append({"pool": "v_full", "key": key, "bytes": b})

        total_all = total_bytes + aux_bytes + decode_bytes + p38_bytes

        return {
            "total_buffers": len(kv_entries) * 2,
            "total_bytes": total_all,
            "total_bytes_kv": total_bytes,
            "total_bytes_aux_scratch": aux_bytes,
            "total_bytes_decode_shared": decode_bytes,
            "total_bytes_p38_continuation": p38_bytes,
            "total_human": _humanize_bytes(total_all),
            "kv_entries": kv_entries,
            "cu_seqlens_devices": list(cls._CU_Q_BUFFERS.keys()),
            "cu_2_devices": list(cls._CU_2_BUFFERS.keys()),
            "synth_seq_lens_entries": [
                {"max_batch": k[0], "device": k[1],
                 "bytes": t.element_size() * t.numel()}
                for k, t in cls._SYNTH_SEQ_LENS_BUFFERS.items()
            ],
            "prefill_out_entries": [
                {"max_batched_tokens": k[0], "num_q_heads": k[1],
                 "head_size": k[2], "device": k[3], "dtype": k[4],
                 "bytes": t.element_size() * t.numel()}
                for k, t in cls._PREFILL_OUT_BUFFERS.items()
            ],
            "decode_shared_entries": decode_entries,
            "p38_continuation_entries": p38_entries,
        }

    @classmethod
    def clear_for_tests(cls):
        """Clear cache — TESTS ONLY. Calling at runtime breaks CUDA graphs."""
        cls._K_BUFFERS.clear()
        cls._V_BUFFERS.clear()
        cls._CU_Q_BUFFERS.clear()
        cls._CU_K_BUFFERS.clear()
        cls._CU_2_BUFFERS.clear()
        cls._SYNTH_SEQ_LENS_BUFFERS.clear()
        cls._PREFILL_OUT_BUFFERS.clear()
        cls._DECODE_MID_O_BUFFERS.clear()
        cls._DECODE_OUTPUT_BUFFERS.clear()
        cls._DECODE_LSE_BUFFERS.clear()
        cls._P38_K_DEQUANT_4D_BUFFERS.clear()
        cls._P38_V_DEQUANT_4D_BUFFERS.clear()
        cls._P38_K_FULL_BUFFERS.clear()
        cls._P38_V_FULL_BUFFERS.clear()
        cls._MIXED_ATTN_OUT_BUFFERS.clear()


# ═══════════════════════════════════════════════════════════════════════════
#                     UPSTREAM INTEGRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def ensure_turboquant_buffers(impl, layer, device: torch.device) -> None:
    """Drop-in call site for `_ensure_on_device` monkey-patch.

    This is THE function that patches/apply_all.py arranges to be called
    from within vLLM's `_ensure_on_device` method. It attaches pre-allocated
    buffers to the layer object so subsequent forward passes can use them.

    Args:
        impl: The TurboQuantAttentionImpl instance (self).
        layer: The Attention layer module.
        device: Target GPU device.

    Side effects on `layer`:
        layer._tq_k_dequant_buf     — K buffer view
        layer._tq_v_dequant_buf     — V buffer view
        layer._tq_cu_q              — cu_seqlens Q scratch
        layer._tq_cu_k              — cu_seqlens K scratch
        layer._tq_cu_2              — P32: second-hop cu_seqlens scratch
        layer._tq_synth_seq_lens    — P33: synthetic seq_lens mirror scratch
        layer._tq_prefill_output    — P26: prefill output reusable buffer
    """
    if not TurboQuantBufferManager.should_apply():
        return  # Graceful skip — layer falls back to upstream behavior

    # P51 (v7.9): TQ-active runtime detection.
    # Skip preallocations entirely when this impl is not actually running
    # TurboQuant KV compression (kv_cache_dtype != turboquant_*). Saves
    # ~516 MiB per rank on FP16-KV / auto deployments where TQ patches
    # graceful-skip but preallocs still fire.
    kv_cache_dtype = getattr(impl, "kv_cache_dtype", None)
    if isinstance(kv_cache_dtype, str) and not kv_cache_dtype.startswith("turboquant_"):
        layer_id = getattr(layer, "layer_name", id(layer))
        if not getattr(impl, "_p51_logged", False):
            log.info(
                "[P51 TQ-active] skipping TQ preallocs on layer=%s: "
                "kv_cache_dtype=%s is non-TurboQuant (saves ~516 MiB)",
                layer_id, kv_cache_dtype,
            )
            impl._p51_logged = True
        return

    # Pull scheduler budget knobs from the current vLLM config. This lets
    # the P22/P26/P32/P33 pools size themselves correctly without relying
    # on per-call env vars. Safe to fail — we stamp attributes on impl so
    # other patches (P26 wiring, etc.) can read them via getattr.
    max_model_len = getattr(impl, "_max_model_len", None)
    max_num_batched_tokens = getattr(impl, "_max_num_batched_tokens", None)
    max_num_seqs = getattr(impl, "_max_num_seqs", None)
    if max_model_len is None or max_num_batched_tokens is None or max_num_seqs is None:
        try:
            from vllm.config import get_current_vllm_config
            cfg = get_current_vllm_config()
            if max_model_len is None:
                max_model_len = getattr(cfg.model_config, "max_model_len", None)
                if max_model_len is not None:
                    impl._max_model_len = max_model_len
            if max_num_batched_tokens is None:
                max_num_batched_tokens = getattr(
                    cfg.scheduler_config, "max_num_batched_tokens", None,
                )
                if max_num_batched_tokens is not None:
                    impl._max_num_batched_tokens = max_num_batched_tokens
            if max_num_seqs is None:
                max_num_seqs = getattr(cfg.scheduler_config, "max_num_seqs", None)
                if max_num_seqs is not None:
                    impl._max_num_seqs = max_num_seqs
        except Exception as e:
            log.debug(
                "[TQ buffer] could not resolve current vLLM config: %s "
                "(will use env/defaults)", e,
            )

    # Stamp on layer too — P26 text-patch reads `self._max_num_batched_tokens`
    # from within TurboQuantAttentionImpl, and impl is the *same* object that
    # `self` refers to inside _prefill_attention.
    if max_num_batched_tokens is not None:
        layer._tq_max_num_batched_tokens = max_num_batched_tokens

    if max_model_len is None:
        # Config-context fetch failed (common: `_ensure_on_device` runs
        # from `forward()` in dev134+ — OUTSIDE the `set_current_vllm_config`
        # context manager that was active during __init__). Fall back
        # to environment override → layer attribute → conservative default.
        env_mml = os.environ.get("GENESIS_TQ_MAX_MODEL_LEN", "")
        if env_mml.isdigit() and int(env_mml) > 0:
            max_model_len = int(env_mml)
            log.info(
                "[TQ buffer] resolved max_model_len=%d from "
                "GENESIS_TQ_MAX_MODEL_LEN env",
                max_model_len,
            )
        else:
            layer_mml = getattr(layer, "max_model_len", None)
            if layer_mml is None:
                # Conservative default: dev134's hybrid Qwen3.6-35B-A3B prod
                # config uses 262144. Over-allocating here is cheap (just
                # ~256 MiB extra if real max_model_len is smaller) vs the
                # alternative of falling back to the 500 MiB transient peak.
                max_model_len = 262144
                log.warning(
                    "[TQ buffer] impl._max_model_len unset AND "
                    "get_current_vllm_config() returned unusable AND no "
                    "GENESIS_TQ_MAX_MODEL_LEN env AND no "
                    "layer.max_model_len attr. Falling back to 262144 "
                    "(prod default). Set GENESIS_TQ_MAX_MODEL_LEN to "
                    "override if your deployment uses a different size."
                )
            else:
                max_model_len = int(layer_mml)
                log.info(
                    "[TQ buffer] resolved max_model_len=%d from layer",
                    max_model_len,
                )
        impl._max_model_len = max_model_len

    # Round up to 1024 for efficient block-aligned allocation
    max_alloc_len = ((max_model_len + 1023) // 1024) * 1024

    # Legacy 3-D K/V dequant prealloc (P22 original). On dev134+ the
    # engine's `_continuation_prefill` uses 4-D slicing
    # (`k_buf[:, :, :alloc_len, :]`) which CANNOT consume a 3-D tensor —
    # so this 3-D allocation was dead weight on dev134. We keep it ONLY
    # for legacy vLLM versions where `_continuation_prefill` uses 3-D
    # shape `(Hk, D, max_alloc_len)`. Skip it when P38 will succeed to
    # avoid 256 MiB/rank of wasted memory on dev134.
    k_buf, v_buf = None, None
    if not TurboQuantBufferManager.should_apply():
        # Legacy path falls through; our 3-D helper will no-op anyway.
        pass
    cu_q, cu_k = TurboQuantBufferManager.get_or_create_cu_seqlens(device)
    cu_2 = TurboQuantBufferManager.get_or_create_cu_2(device)

    # P33: synth_seq_lens sized from max_num_seqs (scheduler batch cap);
    # fall back to a conservative default if the impl doesn't carry it.
    max_batch = getattr(impl, "_max_num_seqs", None) or 32
    synth_seq_lens = TurboQuantBufferManager.get_or_create_synth_seq_lens(
        max_batch=max_batch, device=device,
    )

    # P26: prefill output buffer. Size to scheduler's max_num_batched_tokens
    # when known; fall back to `max_num_batched_tokens` resolved above
    # (from vLLM config) or ultimately 4096.
    max_bt = max_num_batched_tokens or getattr(
        impl, "_max_num_batched_tokens", None,
    ) or 4096
    num_q_heads = getattr(impl, "num_heads", None) or getattr(
        impl, "num_kv_heads", 0,
    )
    model_dtype = getattr(impl, "_model_dtype", None) or torch.bfloat16
    prefill_out = None
    if num_q_heads and num_q_heads > 0:
        prefill_out = TurboQuantBufferManager.get_or_create_prefill_output(
            max_batched_tokens=max_bt,
            num_q_heads=num_q_heads,
            head_size=impl.head_size,
            device=device,
            dtype=model_dtype,
        )

    # P38: 4-D dequant buffers matching dev134 shape convention exactly.
    # These OVERRIDE the 3-D P22 buffers because dev134's
    # `_continuation_prefill` does `k_buf[:, :, :alloc_len, :]` (4-D
    # slicing) — a 3-D prealloc there raises IndexError or gets
    # ignored via the `k_buf.shape[2] < alloc_len` fallback path.
    #
    # Budget rationale:
    #   max_alloc_len = max_model_len rounded up to 1024 → 262144 for prod config
    #   bytes per buffer = Hk × max_alloc_len × D × 2 (fp16) = 2 × 262144 × 128 × 2
    #                    = 128 MiB per K, same for V → 256 MiB pair
    #   Shared across ALL layers (sequential forward), so 256 MiB total
    #   NOT 256 MiB × num_layers.
    k_buf_4d, v_buf_4d = TurboQuantBufferManager.get_or_create_p38_dequant_4d(
        num_kv_heads=impl.num_kv_heads,
        head_size=impl.head_size,
        max_alloc_len=max_alloc_len,
        device=device,
        dtype=torch.float16,
    )

    # P38: K_full / V_full (cached + chunk) workspace — replaces torch.cat
    # peak transient. Sized to max_model_len + max_num_batched_tokens so
    # the last chunk (prefix ≈ max_model_len - chunk) fits with room for
    # the new chunk appended. Shared across layers.
    max_bt_for_full = max_num_batched_tokens or 4096
    max_seq_cap = max_alloc_len + max_bt_for_full
    # Align to 1024 for allocator-friendly slab size.
    max_seq_cap = ((max_seq_cap + 1023) // 1024) * 1024
    k_full_buf, v_full_buf = TurboQuantBufferManager.get_or_create_p38_full(
        num_kv_heads=impl.num_kv_heads,
        head_size=impl.head_size,
        max_seq_cap=max_seq_cap,
        device=device,
        dtype=torch.float16,
    )

    # Attach (skip if None — graceful degrade)
    # P38 4-D buffers TAKE PRECEDENCE over 3-D P22 buffers: dev134
    # engine code reads the 4-D shape convention, so we must attach the
    # 4-D pair to the canonical attribute names.
    if k_buf_4d is not None and v_buf_4d is not None:
        layer._tq_k_dequant_buf = k_buf_4d
        layer._tq_v_dequant_buf = v_buf_4d
    elif k_buf is not None and v_buf is not None:
        # Fallback path for non-CUDA/non-SM80 where P38 allocator refuses.
        layer._tq_k_dequant_buf = k_buf
        layer._tq_v_dequant_buf = v_buf
    if k_full_buf is not None:
        layer._tq_k_full_buf = k_full_buf
    if v_full_buf is not None:
        layer._tq_v_full_buf = v_full_buf
    if cu_q is not None:
        layer._tq_cu_q = cu_q
    if cu_k is not None:
        layer._tq_cu_k = cu_k
    if cu_2 is not None:
        layer._tq_cu_2 = cu_2
    if synth_seq_lens is not None:
        layer._tq_synth_seq_lens = synth_seq_lens
    if prefill_out is not None:
        layer._tq_prefill_output = prefill_out
