# SPDX-License-Identifier: Apache-2.0
"""GDN window-scratch pool — config/env gate + future pool primitive.

⚠ AUDIT P2.5 honesty 2026-05-05 (genesis_deep_cross_audit):
This module DEFINES `acquire_h_window`, `acquire_v_new_window`,
`acquire_state` registries — but the production `streaming_gdn_driver.py`
runtime does NOT call them. The `_streaming_path` lets the inner
`chunk_gated_delta_rule_fwd_h` kernel allocate its own `h_w` and pays
for the smaller window allocation rather than reusing a pool slot.

Net production behavior: the WINDOWING is real (h_w shrinks from full
(B,NT,H,V,K) to (B,WINDOW_NT,H,V,K)) — that's the −142 MiB/GPU saving
PN59 measured on PROD. The POOL/cudagraph-reuse facet of this module
is currently INFRASTRUCTURE FOR FUTURE WORK, used by tests and the
reference utility at `utils/streaming_gdn_reference.py` only.

If you came here looking for runtime pool acquire calls in the live
streaming path: there are none yet. Audit-tracked for future sprint.

Cliff 2b multi-turn OOM fix (Issue #19, true root cause).

Background
----------
`vllm/model_executor/layers/fla/ops/chunk_delta_h.py:332` GDN
`chunk_gated_delta_rule_fwd_h` allocates a fresh `(B, NT, H, V, K)` BF16
tensor PER LAYER PER FORWARD:

    h = k.new_empty(B, NT, H, V, K)

For Lorbus Qwen3.6-27B-int4 hybrid (NT=T/64, H=24, K=V=128, fp16 at T=64K):
**805 MiB single allocation × N hybrid layers** → multi-turn allocator
fragmentation → Cliff 2b OOM after 4-5 turns.

**Independent confirmation** (issue #20, 2026-05-05): noonghunna
(reproducer owner): "the limitation is the triton kernel for cliff 2.
The problem doesn't appear with llama.cpp and specifically with vllm."

Cross-engine: llama.cpp / MLX-LM use pure-streaming (register-resident
state), no `(B, NT, ...)` materialization → survive multi-turn.

Variant D fix
-------------
Window-iterative driver: process WINDOW_NT chunks at a time. Pool holds
single `(B, WINDOW_NT, H, V, K)` window buffer + reusable `v_new` and
`state` buffers, all shape-keyed.

Pattern lifted from FFNIntermediateCache (PN12), proven safe:
1. Class-level singleton registry per (shape_key)
2. Pointer-stable across same-key acquires (cudagraph-safe)
3. Grow-once on size increase
4. Slice-on-acquire when fits
5. Per-process (per-rank) — no cross-rank IPC
6. Static env gate

Window memory budget
--------------------
At Genesis 27B Lorbus shapes (B=1, H=24, K=V=128, fp16):
  WINDOW_NT=4: 1 × 4 × 24 × 128 × 128 × 2 = 3.0 MiB
  WINDOW_NT=8: 6.0 MiB
  WINDOW_NT=16: 12.0 MiB
vs. baseline 805 MiB at T=64K → **~270x reduction at WINDOW_NT=4**.

Why this is safe
----------------
1. **Sequential layer execution.** Same as FFN PN12 — vLLM transformer
   forward calls GDN layers in strict sequence. Layer N's window buffer
   is fully consumed by `chunk_fwd_o` BEFORE layer N+1 acquires.
2. **Pointer-stable.** Same shape_key → same data_ptr (cudagraph reuse).
3. **Numerical equivalence proven.** Triton kernel `fwd_h` uses internal
   register state `b_h1..b_h4` accumulation; window-iterative driver
   produces bit-identical output (TDD-verified).
4. **No state corruption between requests.** State buffer (B, H, V, K)
   is always overwritten by `output_final_state` writeback or restored
   from `initial_state` parameter on next request.

How to use
----------
Operator opt-in via env:

    GENESIS_ENABLE_PN59_STREAMING_GDN=1
    GENESIS_VARIANT_D_WINDOW_NT=4   # tune per shape

Genesis text-patch (PN59) wraps `chunk_gated_delta_rule_fwd_h` to use
this pool when the request meets streaming criteria (single-seq, T >
window threshold). Falls through to vanilla `_orig_fwd` otherwise.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Pattern: FFNIntermediateCache (PN12)
Cross-engine inspirations: llama.cpp ssm-scan.cu (register-streaming),
  Mamba2 ssd_combined (3-stage chunk split), FLA RFC #485 (Songlin Yang
  memory_efficient flag direction)
"""
from __future__ import annotations

import logging
import os

import torch

from vllm._genesis.guards import is_nvidia_cuda

log = logging.getLogger("genesis.kernels.gdn_scratch_pool")


_ENV_FLAG_MASTER = "GENESIS_ENABLE_PN59_STREAMING_GDN"
_ENV_WINDOW_NT = "GENESIS_VARIANT_D_WINDOW_NT"
_DEFAULT_WINDOW_NT = 4


# Shape keys (5D for h, 4D for v_new, 4D for state, 4D for o output)
HKey = tuple[int, int, int, int, int, torch.dtype, torch.device]
VKey = tuple[int, int, int, int, torch.dtype, torch.device]
SKey = tuple[int, int, int, int, torch.dtype, torch.device]
OKey = tuple[int, int, int, int, torch.dtype, torch.device]  # (B, H, V, T_max_bin)


class GdnScratchPool:
    """Class-level registry of GDN window-scratch buffers.

    Per-process singleton. Each TP rank has its own registry — buffers
    are NOT shared across ranks (no IPC).

    Three buffer types
    ------------------
    1. **h_window** — `(B, WINDOW_NT, H, V, K)` chunk hidden states for
       window's worth of chunks. Replaces `(B, NT, H, V, K)` baseline alloc.
    2. **v_new_window** — `(B, WINDOW_NT * BT, H, V)` corrected V values
       for the window. Allocated once, reused window-by-window.
    3. **state** — `(B, H, V, K)` recurrent state float32. Persistent
       across windows within one forward call; reset between forward calls.

    Lifecycle
    ---------
    1. First call with unique shape key → alloc at requested max
    2. Subsequent same-shape acquires → return slice (size <= cached_max)
       OR grow-once (size > cached_max)
    3. Process shutdown → tensors freed via Python GC
    """

    _H_REGISTRY: dict[HKey, torch.Tensor] = {}
    _V_REGISTRY: dict[VKey, torch.Tensor] = {}
    _S_REGISTRY: dict[SKey, torch.Tensor] = {}
    _O_REGISTRY: dict[OKey, torch.Tensor] = {}  # club-3090#22 Level 2C

    # ──────────────────────────────────────────────────────────────────
    # Platform / env gate
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def should_apply() -> bool:
        """Env gate. Operator must explicitly opt in.

        Audit P3 fix 2026-05-05: unified bool parser to match the rest of
        the Genesis env-flag conventions ("1", "true", "yes", "y", "on" —
        case-insensitive). Was previously rejecting "yes"/"on"/"True " etc.
        which made the wrapper return True from `apply()` while the runtime
        pool still rejected eligibility — silent no-op trap.
        """
        return os.environ.get(_ENV_FLAG_MASTER, "").strip().lower() in (
            "1", "true", "yes", "y", "on",
        )

    @staticmethod
    def is_production_eligible() -> bool:
        """Stricter: env + NVIDIA CUDA."""
        return GdnScratchPool.should_apply() and is_nvidia_cuda()

    @staticmethod
    def get_window_nt() -> int:
        """Read tunable WINDOW_NT from env (default 4 chunks = 256 tokens at BT=64)."""
        try:
            v = int(os.environ.get(_ENV_WINDOW_NT, str(_DEFAULT_WINDOW_NT)))
            return max(1, min(64, v))
        except ValueError:
            return _DEFAULT_WINDOW_NT

    # ──────────────────────────────────────────────────────────────────
    # Core API: h_window
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def acquire_h_window(
        cls,
        B: int,
        window_nt: int,
        H: int,
        V: int,
        K: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return `(B, window_nt, H, V, K)` view of pooled h-window buffer.

        Caller writes Triton kernel output IN PLACE; consumer `chunk_fwd_o`
        reads same memory.
        """
        if min(B, window_nt, H, V, K) <= 0:
            raise ValueError(
                f"GdnScratchPool.acquire_h_window: dims must be > 0, "
                f"got (B={B}, W={window_nt}, H={H}, V={V}, K={K})"
            )
        # Key WITHOUT window_nt — buffer is sized at MAX seen window
        key: HKey = (B, _DEFAULT_WINDOW_NT, H, V, K, dtype, device)
        cached = cls._H_REGISTRY.get(key)
        if cached is None:
            buf = torch.empty(
                (B, window_nt, H, V, K), dtype=dtype, device=device,
            )
            cls._H_REGISTRY[key] = buf
            log.info(
                "[PN59] first acquire h_window: alloc (%d, %d, %d, %d, %d) "
                "%s on %s (%.2f MiB)",
                B, window_nt, H, V, K, dtype, device,
                B * window_nt * H * V * K * _dtype_byte_size(dtype) / 1024 / 1024,
            )
            return buf
        # Existing buffer — fits if cached's window dim >= requested
        cached_w = cached.shape[1]
        if window_nt <= cached_w:
            return cached[:, :window_nt]
        # Grow once
        new_buf = torch.empty(
            (B, window_nt, H, V, K), dtype=dtype, device=device,
        )
        cls._H_REGISTRY[key] = new_buf
        log.info(
            "[PN59] grew h_window: window %d → %d for shape (B=%d H=%d V=%d K=%d)",
            cached_w, window_nt, B, H, V, K,
        )
        return new_buf

    # ──────────────────────────────────────────────────────────────────
    # Core API: v_new_window
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def acquire_v_new_window(
        cls,
        B: int,
        window_T: int,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return `(B, window_T, H, V)` view for corrected V values."""
        if min(B, window_T, H, V) <= 0:
            raise ValueError(
                f"GdnScratchPool.acquire_v_new_window: dims must be > 0, "
                f"got (B={B}, T={window_T}, H={H}, V={V})"
            )
        key: VKey = (B, H, V, _DEFAULT_WINDOW_NT * 64, dtype, device)
        cached = cls._V_REGISTRY.get(key)
        if cached is None:
            buf = torch.empty(
                (B, window_T, H, V), dtype=dtype, device=device,
            )
            cls._V_REGISTRY[key] = buf
            log.info(
                "[PN59] first acquire v_new_window: alloc (%d, %d, %d, %d) "
                "%s on %s (%.2f MiB)",
                B, window_T, H, V, dtype, device,
                B * window_T * H * V * _dtype_byte_size(dtype) / 1024 / 1024,
            )
            return buf
        cached_T = cached.shape[1]
        if window_T <= cached_T:
            return cached[:, :window_T]
        new_buf = torch.empty(
            (B, window_T, H, V), dtype=dtype, device=device,
        )
        cls._V_REGISTRY[key] = new_buf
        log.info(
            "[PN59] grew v_new_window: T %d → %d for shape (B=%d H=%d V=%d)",
            cached_T, window_T, B, H, V,
        )
        return new_buf

    # ──────────────────────────────────────────────────────────────────
    # Core API: state (recurrent state, shared across windows)
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def acquire_state(
        cls,
        B: int,
        H: int,
        V: int,
        K: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return `(B, H, V, K)` state buffer.

        State semantics: caller MUST overwrite via initial_state param OR
        explicit `.zero_()`. Pool only provides storage, not initialization.
        """
        if min(B, H, V, K) <= 0:
            raise ValueError(
                f"GdnScratchPool.acquire_state: dims must be > 0, "
                f"got (B={B}, H={H}, V={V}, K={K})"
            )
        key: SKey = (B, H, V, K, dtype, device)
        cached = cls._S_REGISTRY.get(key)
        if cached is not None:
            return cached
        buf = torch.empty(
            (B, H, V, K), dtype=dtype, device=device,
        )
        cls._S_REGISTRY[key] = buf
        log.info(
            "[PN59] first acquire state: alloc (%d, %d, %d, %d) %s on %s (%.2f MiB)",
            B, H, V, K, dtype, device,
            B * H * V * K * _dtype_byte_size(dtype) / 1024 / 1024,
        )
        return buf

    # ──────────────────────────────────────────────────────────────────
    # Core API: o_output (chunk_o output, full T-dim per call)
    # ──────────────────────────────────────────────────────────────────
    #
    # club-3090#22 Level 2C+D fix: chunk_o.py:161 `o = torch.empty_like(v)`
    # is the OOM site noonghunna hit — 50 MiB request from a 56 MiB-frag-
    # mented free pool. Routing this allocation through the pool means
    # the chunk_o output buffer is allocated ONCE at first FLA call and
    # reused across:
    #   - all 64 GDN layers in the model
    #   - all WINDOW_NT = 4-16 windows per call (PN59 streaming path)
    #   - all forward passes for the request lifetime
    #
    # vs the per-window-per-layer churn (~1024 allocations per forward
    # at T=60K) under the current code.
    #
    # Sharing across all layers + windows is safe because GDN layers are
    # processed sequentially (no cross-layer overlap) and PN59's window
    # loop has zero overlap (one window's o is written into o_full and
    # the next window's o re-uses the same buffer).
    #
    # Bin-by-power-of-2 T to avoid grow-once on every slightly-larger
    # request — caller passes `T` actual, we cache a buffer of size
    # `next_pow2(T)` and slice down. Eliminates 99% of grow events on
    # chunked-prefill schedules where T varies modestly.

    @classmethod
    def acquire_o_output(
        cls,
        B: int,
        T: int,
        H: int,
        V: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return `(B, T, H, V)` view for chunk_o output.

        Shape semantics: caller writes into the slice; pool guarantees
        contiguous storage along T-dim. Buffer is shared across ALL
        layers + windows + forward passes for the process lifetime —
        sequential GDN layer execution guarantees no overlap.

        T is bin-rounded to next power of 2 internally to amortize
        grow-events across requests of varying length.
        """
        if min(B, T, H, V) <= 0:
            raise ValueError(
                f"GdnScratchPool.acquire_o_output: dims must be > 0, "
                f"got (B={B}, T={T}, H={H}, V={V})"
            )

        # Bin T to next power of 2 (min 512 to avoid pathological churn
        # on very-short prompts).
        T_binned = max(512, 1 << (T - 1).bit_length())

        key: OKey = (B, H, V, T_binned, dtype, device)
        cached = cls._O_REGISTRY.get(key)
        if cached is not None:
            return cached[:, :T]

        buf = torch.empty(
            (B, T_binned, H, V), dtype=dtype, device=device,
        )
        cls._O_REGISTRY[key] = buf
        log.info(
            "[PN59] first acquire o_output: alloc (%d, %d_binned_from_%d, %d, %d) "
            "%s on %s (%.2f MiB) — closes club-3090#22 chunk_o.py:161 OOM",
            B, T_binned, T, H, V, dtype, device,
            B * T_binned * H * V * _dtype_byte_size(dtype) / 1024 / 1024,
        )
        return buf[:, :T]

    # ──────────────────────────────────────────────────────────────────
    # Introspection / lifecycle
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def total_pooled_bytes(cls) -> int:
        """Sum of bytes held across all 4 registries."""
        total = 0
        for reg in (cls._H_REGISTRY, cls._V_REGISTRY,
                    cls._S_REGISTRY, cls._O_REGISTRY):
            for buf in reg.values():
                total += buf.numel() * _dtype_byte_size(buf.dtype)
        return total

    @classmethod
    def num_pools(cls) -> dict[str, int]:
        """Pool count breakdown."""
        return {
            "h_window": len(cls._H_REGISTRY),
            "v_new_window": len(cls._V_REGISTRY),
            "state": len(cls._S_REGISTRY),
            "o_output": len(cls._O_REGISTRY),
            "total": (len(cls._H_REGISTRY)
                      + len(cls._V_REGISTRY)
                      + len(cls._S_REGISTRY)
                      + len(cls._O_REGISTRY)),
        }

    @classmethod
    def stats(cls) -> dict:
        """Detailed introspection — used by genesis doctor."""
        return {
            "enabled": cls.should_apply(),
            "window_nt": cls.get_window_nt(),
            "pools": cls.num_pools(),
            "total_pooled_mib": cls.total_pooled_bytes() / 1024 / 1024,
        }

    @classmethod
    def reset(cls) -> None:
        """Clear all registries. Test-only — DO NOT use in prod."""
        cls._H_REGISTRY.clear()
        cls._V_REGISTRY.clear()
        cls._S_REGISTRY.clear()
        cls._O_REGISTRY.clear()


def _dtype_byte_size(dtype: torch.dtype) -> int:
    """Bytes per element. Same helper as PN12 FFN cache."""
    if dtype in (torch.float32, torch.int32):
        return 4
    if dtype in (torch.float16, torch.bfloat16, torch.int16):
        return 2
    if dtype in (torch.float64, torch.int64):
        return 8
    if dtype in (torch.uint8, torch.int8, torch.bool):
        return 1
    if dtype == torch.float8_e4m3fn:
        return 1
    if dtype == torch.float8_e5m2:
        return 1
    try:
        return torch.tensor([], dtype=dtype).element_size()
    except Exception:
        return 4
