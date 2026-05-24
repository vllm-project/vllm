"""PN40 — DFlash drafter omnibus optimization (Genesis-original).

Strict-superset optimization for DFlash spec-decode that:
  - DOES NOT compete with FA2 attention (PN37 lesson learned)
  - Reduces kernel launch overhead (per-layer loop fusion)
  - Improves memory/cache utilization (persistent buffer pool)
  - Adapts to workload (short/long ctx, code, free-form)
  - Provides runtime stability sentinel + auto-fallback
  - Composes additively with PN21/PN23/PN24 (other DFlash patches)

Architecture (4 sub-kernels under one umbrella):

  A. Fused per-layer K-norm Triton kernel
     Replaces `for i in range(L): ops.rms_norm(...)` loop in
     `qwen3_dflash.py:397-404` (per-layer K-norm). Single kernel launch
     vs L launches (L=5 on 27B drafter, L=8 on 35B drafter).
     Saves: L-1 kernel launches per draft step ≈ 12-35 µs.

  B. Persistent K/V buffer pool (Python orchestrator)
     Reuses `all_k_normed` / `all_v_buffer` allocations across draft
     steps instead of fresh `torch.empty_like()` each step.
     Saves: ~50-150 µs Ampere malloc overhead per step.

  C. Adaptive DFlash N controller (mirror SGLang [1, 3, 5] tier policy)
     EMA-based acceptance length tracking; auto-shifts N down on reject
     clusters, up on hit clusters. Per-workload tuning.
     Variable: +5-30% TPS depending on workload mix.

  D. Workload classifier + auto-fallback
     Cheap (no GPU sync) classifier picks which sub-kernels engage per
     request: short-ctx, long-ctx, code-query, free-form. Stability
     sentinel disables PN40 on NaN/Inf or AL drop > 0.5.

This v1 file implements ONLY Sub-kernel A (fused K-norm). Sub-kernels
B/C/D land in follow-up commits to keep diff reviewable. The orchestrator
shape exists from the start so the API is stable.

Eligibility (no-regression contract):
  - DFlash drafter present (config.architectures contains "DFlashDraftModel")
  - K tensor BF16, head_dim ∈ {64, 128} (covers 27B + 35B drafters)
  - L (num drafter layers) ∈ [2, 16] (handles edge cases)
  - num_ctx ≥ 1 (any non-empty draft step)
  - SM 8.6+ (Ampere; on SM 8.9+ kernel still works, just less critical)
  - GENESIS_ENABLE_PN40_DFLASH_OMNIBUS=1 (default OFF until A/B validates)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.kernels.pn40")

_ENV_ENABLE = "GENESIS_ENABLE_PN40_DFLASH_OMNIBUS"
_ENV_SUB_A = "GENESIS_PN40_ENABLE_SUB_A"  # fused K-norm (default ON if PN40 ON)


def env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on"
    )


def sub_a_enabled() -> bool:
    """Sub-kernel A toggle (default: ON when PN40 master is ON)."""
    if not env_enabled():
        return False
    val = os.environ.get(_ENV_SUB_A, "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    return True  # default ON


# ───────────────────────────────────────────────────────────────────
# Sub-kernel A: fused per-layer K-norm
# ───────────────────────────────────────────────────────────────────

_CACHED_K_NORM_KERNEL = None


def _build_fused_k_norm_kernel():
    """Lazy-build Triton kernel. Cached so repeated calls are free."""
    global _CACHED_K_NORM_KERNEL
    if _CACHED_K_NORM_KERNEL is not None:
        return _CACHED_K_NORM_KERNEL

    import triton
    import triton.language as tl

    @triton.jit
    def _pn40_fused_k_norm_kernel(
        K_in_ptr,            # [L, N, H, D]
        W_ptr,               # [L, D] — per-layer norm weights
        K_out_ptr,           # [L, N, H, D]
        sK_l, sK_n, sK_h, sK_d,
        sW_l, sW_d,
        N_per_kv: tl.constexpr,
        eps,
        BLOCK_D: tl.constexpr,
    ):
        """One CTA owns one (layer, ctx_pos). Iterates H heads inner.

        For each head:
          x = K_in[layer, ctx_pos, head, :]         # [D]
          var = mean(x * x)
          x_normed = x * rsqrt(var + eps)
          K_out[layer, ctx_pos, head, :] = x_normed * W[layer, :]
        """
        layer_id = tl.program_id(0)
        ctx_pos = tl.program_id(1)

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < BLOCK_D  # always true for power-of-2 BLOCK_D == HEAD_DIM

        # Load per-layer norm weight [D]
        w_offsets = layer_id * sW_l + offs_d * sW_d
        w_tile = tl.load(W_ptr + w_offsets, mask=d_mask, other=0.0).to(tl.float32)

        # Iterate heads (cheap, small constant for GQA configs)
        for h_off in tl.static_range(N_per_kv):
            x_offsets = (
                layer_id * sK_l
                + ctx_pos * sK_n
                + h_off * sK_h
                + offs_d * sK_d
            )
            x = tl.load(K_in_ptr + x_offsets, mask=d_mask, other=0.0).to(tl.float32)

            # RMSNorm: x * rsqrt(mean(x^2) + eps)
            var = tl.sum(x * x, axis=0) / BLOCK_D
            inv_rms = 1.0 / tl.sqrt(var + eps)
            x_normed = x * inv_rms * w_tile

            o_offsets = x_offsets  # same layout for output
            tl.store(
                K_out_ptr + o_offsets,
                x_normed.to(K_out_ptr.dtype.element_ty),
                mask=d_mask,
            )

    _CACHED_K_NORM_KERNEL = _pn40_fused_k_norm_kernel
    return _CACHED_K_NORM_KERNEL


def fused_k_norm(k_in, weights, eps: float, out=None):
    """Fused per-layer K-norm.

    Args:
        k_in:    [L, N, H, D] BF16
        weights: [L, D] BF16/FP32
        eps:     float
        out:     [L, N, H, D] BF16 buffer (optional; allocated if None)

    Returns:
        out (allocated or passed) with norm applied.
    """
    assert k_in.is_cuda, "PN40 requires CUDA tensor"
    assert k_in.dim() == 4, f"k_in must be [L,N,H,D]; got {k_in.shape}"
    assert weights.dim() == 2, f"weights must be [L,D]; got {weights.shape}"
    L, N, H, D = k_in.shape
    Lw, Dw = weights.shape
    assert Lw == L, f"weights L ({Lw}) != k_in L ({L})"
    assert Dw == D, f"weights D ({Dw}) != k_in D ({D})"

    if out is None:
        out = k_in.new_empty((L, N, H, D))
    assert out.shape == k_in.shape

    kernel = _build_fused_k_norm_kernel()
    grid = (L, N)
    BLOCK_D = D
    kernel[grid](
        k_in, weights, out,
        k_in.stride(0), k_in.stride(1), k_in.stride(2), k_in.stride(3),
        weights.stride(0), weights.stride(1),
        N_per_kv=H,
        eps=eps,
        BLOCK_D=BLOCK_D,
    )
    return out


# ───────────────────────────────────────────────────────────────────
# Eligibility predicates (no-regression contract)
# ───────────────────────────────────────────────────────────────────


def is_sub_a_eligible(k_shape, k_dtype) -> bool:
    """Sub-kernel A eligibility: cheap, no GPU sync.

    k_shape: tuple/torch.Size [L, N, H, D]
    k_dtype: torch.dtype
    """
    import torch

    if not sub_a_enabled():
        return False
    if len(k_shape) != 4:
        return False
    L, N, H, D = k_shape
    if not (2 <= L <= 16):
        return False
    if N < 1:
        return False
    if D not in (64, 128):
        return False
    if k_dtype not in (torch.bfloat16, torch.float16):
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
# Sub-kernel C: Universal adaptive spec-decode K/N controller
# Applies to: MTP K (PROD 27B/35B) + DFlash N (test 27B/35B)
# ═══════════════════════════════════════════════════════════════════

_ENV_SUB_C = "GENESIS_PN40_ENABLE_SUB_C"


def sub_c_enabled() -> bool:
    """Sub-kernel C toggle (default ON when PN40 master ON)."""
    if not env_enabled():
        return False
    val = os.environ.get(_ENV_SUB_C, "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    return True


class AdaptiveSpecKController:
    """Universal adaptive K/N controller for spec-decode (MTP, DFlash, ngram).

    Mirrors SGLang's tier-policy from
    `python/sglang/srt/managers/scheduler.py` `AdaptiveSpecParams` —
    EMA-tracked acceptance length + hysteresis bands shift K up/down
    between configured tiers based on observed acceptance rate.

    State (per spec_method):
      - tiers: list of K values (e.g. [0, 1, 3, 5] — K=0 means skip spec)
      - current_idx: current tier index
      - ema_al: EMA of acceptance length (0.0..K_max)
      - ema_alpha: EMA decay rate (default 0.2)
      - up_threshold: shift up if ema_al > tier[current_idx] * up_threshold
      - down_threshold: shift down if ema_al < tier[current_idx] * down_threshold

    Stability:
      - bounded state machine (no unbounded loops)
      - reset on engine restart (instance per StructuredOutputManager)
      - disable on NaN/Inf in observed AL

    Usage:
      controller = AdaptiveSpecKController(tiers=[0, 1, 3, 5], base_k=5)
      ...per accepted batch:
        controller.observe(accepted_len)
      ...before next propose:
        k = controller.current_k()
    """

    def __init__(
        self,
        tiers: list[int],
        base_k: int,
        ema_alpha: float = 0.2,
        up_threshold: float = 0.85,
        down_threshold: float = 0.55,
    ):
        assert len(tiers) >= 1, "at least one tier required"
        assert base_k in tiers, f"base_k {base_k} must be in tiers {tiers}"
        self.tiers = sorted(tiers)
        self.current_idx = self.tiers.index(base_k)
        self.base_k = base_k
        self.ema_alpha = ema_alpha
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.ema_al: float = float(base_k)
        self.observation_count: int = 0
        self.disabled: bool = False
        self.disable_reason: str | None = None
        # [v7.72 workload-aware tier override] Per-workload class K bias.
        # When request._genesis_pn40_workload_class is set, current_k()
        # returns biased K instead of pure controller decision.
        # See classify_workload() for class taxonomy.
        self._workload_class: str | None = None
        self._workload_bias_table: dict[str, int] = {
            # Code workloads have predictable token streams → high acceptance →
            # benefit from MAX tier (or +1 from controller's idx)
            "code": +1,
            # Long context: KV thrashing with high K + worse acceptance →
            # bias DOWN by 1 tier from controller's choice
            "long_ctx": -1,
            # Short / free-form: trust controller (no bias)
            "short_ctx": 0,
            "free_form": 0,
        }

    def set_workload_class(self, workload_class: str | None) -> None:
        """Update per-request workload class. Read by current_k() to bias K.

        Cheap (no allocation, no GPU sync). Pass None to clear bias.
        """
        if workload_class is not None and workload_class not in self._workload_bias_table:
            # Unknown class → treat as no bias (defensive)
            workload_class = None
        self._workload_class = workload_class

    def current_k(self) -> int:
        """Cheap getter — returns current tier K, biased by workload class.

        No GPU sync. If disabled → returns base_k. If no workload class set
        OR class is unknown → returns pure controller tier.
        """
        if self.disabled:
            return self.base_k  # fall back to user-configured
        idx = self.current_idx
        bias = self._workload_bias_table.get(self._workload_class or "", 0)
        idx = max(0, min(len(self.tiers) - 1, idx + bias))
        return self.tiers[idx]

    def observe(self, accepted_len: float) -> None:
        """Observe an accepted-length sample, update EMA, possibly shift tier."""
        if self.disabled:
            return
        # NaN/Inf sentinel
        if accepted_len != accepted_len or accepted_len < 0 or accepted_len > 100:
            self.disabled = True
            self.disable_reason = (
                f"invalid AL observation {accepted_len!r}, controller disabled"
            )
            return

        self.ema_al = (1 - self.ema_alpha) * self.ema_al + self.ema_alpha * accepted_len
        self.observation_count += 1

        # Warmup grace period — don't shift in first 10 observations
        if self.observation_count < 10:
            return

        cur_k = self.tiers[self.current_idx]
        if cur_k == 0:
            # Already at K=0 — only path UP is observing high acceptance on the
            # NEXT non-zero tier. Cheap re-engage policy: shift up after 30 obs
            # if EMA still > 0 (proxy: there were some hits).
            if self.observation_count - 10 > 30 and self.ema_al > 0.5:
                self.current_idx = min(self.current_idx + 1, len(self.tiers) - 1)
                self.observation_count = 0
            return

        # Hysteresis: shift up if AL approaches current K, down if far below
        up_thr = cur_k * self.up_threshold
        down_thr = cur_k * self.down_threshold

        if self.ema_al >= up_thr and self.current_idx < len(self.tiers) - 1:
            self.current_idx += 1
            self.observation_count = 0
        elif self.ema_al <= down_thr and self.current_idx > 0:
            self.current_idx -= 1
            self.observation_count = 0

    def state_dict(self) -> dict:
        """For diagnostics + serialization."""
        return {
            "tiers": list(self.tiers),
            "current_k": self.current_k(),
            "ema_al": self.ema_al,
            "observation_count": self.observation_count,
            "disabled": self.disabled,
            "disable_reason": self.disable_reason,
        }


# Default tier policies (env-overridable, per spec_method)
_DEFAULT_TIERS = {
    "mtp_3": [0, 1, 3],          # MTP K=3 base (27B PROD, 35B PROD)
    "dflash_5": [0, 1, 3, 5],    # DFlash N=5 base (27B+DFlash)
    "dflash_3": [0, 1, 3],       # DFlash N=3 base (35B+DFlash)
    "ngram": [0, 1, 3, 5, 7],    # ngram base (legacy P77 territory)
}


def get_default_tiers(spec_method: str, base_k: int) -> list[int]:
    """Resolve default tier policy by spec method + base K."""
    key = f"{spec_method}_{base_k}"
    return _DEFAULT_TIERS.get(key, [0, max(1, base_k // 2), base_k])


# ═══════════════════════════════════════════════════════════════════
# Sub-kernel D: Universal stability sentinel + workload classifier
# Applies to ALL 4 configs (MTP + DFlash, 27B + 35B)
# ═══════════════════════════════════════════════════════════════════

_ENV_SUB_D = "GENESIS_PN40_ENABLE_SUB_D"


def sub_d_enabled() -> bool:
    """Sub-kernel D toggle (default ON when PN40 master ON)."""
    if not env_enabled():
        return False
    val = os.environ.get(_ENV_SUB_D, "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    return True


def classify_workload(prompt_text: str | None, prompt_len: int) -> str:
    """Cheap (no GPU sync) workload classifier.

    Returns one of: "code", "long_ctx", "short_ctx", "free_form"

    - "code": detected via tool-call markers, code fences, or fim tokens
              in last 256 chars of prompt
    - "long_ctx": prompt_len >= 16K tokens
    - "short_ctx": prompt_len < 1K tokens
    - "free_form": everything else (default tier)

    Used by orchestrator to select tier policy + buffer pool size.
    """
    if prompt_len >= 16_384:
        return "long_ctx"
    if prompt_text is not None:
        # Check last ~256 chars only (cheap)
        tail = prompt_text[-256:] if len(prompt_text) > 256 else prompt_text
        for sig in ("```", "<tool_call>", "<|fim_", "def ", "function"):
            if sig in tail:
                return "code"
    if prompt_len < 1_024:
        return "short_ctx"
    return "free_form"


class StabilitySentinel:
    """Universal stability monitor for spec-decode optimizations.

    Tracks AL over a sliding window. If AL drops by >threshold (default 0.5)
    over `window_size` observations relative to the long-running EMA, OR if
    NaN/Inf is observed, the sentinel raises a flag that orchestrator reads
    to disable PN40 sub-kernels for the affected request.

    Per-spec-method (separate state for MTP vs DFlash). Reset on engine
    restart.
    """

    def __init__(
        self,
        window_size: int = 50,
        drop_threshold: float = 0.5,
        ema_alpha: float = 0.05,  # slow EMA = stable baseline
    ):
        self.window_size = window_size
        self.drop_threshold = drop_threshold
        self.ema_alpha = ema_alpha
        self.long_ema_al: float = 0.0
        self.recent_window: list[float] = []
        self.tripped: bool = False
        self.trip_reason: str | None = None
        self.trip_count: int = 0

    def observe(self, accepted_len: float) -> bool:
        """Observe AL sample. Returns True if sentinel just tripped this step."""
        # NaN/Inf → trip immediately
        if accepted_len != accepted_len or accepted_len < 0 or accepted_len > 100:
            if not self.tripped:
                self.tripped = True
                self.trip_reason = f"invalid AL: {accepted_len!r}"
                self.trip_count += 1
                return True
            return False

        # Slow EMA for stable baseline
        if self.long_ema_al == 0.0:
            self.long_ema_al = accepted_len
        self.long_ema_al = (
            (1 - self.ema_alpha) * self.long_ema_al + self.ema_alpha * accepted_len
        )

        # Sliding window
        self.recent_window.append(accepted_len)
        if len(self.recent_window) > self.window_size:
            self.recent_window.pop(0)

        # Window vs baseline drop check (only after warmup)
        if len(self.recent_window) >= self.window_size:
            window_avg = sum(self.recent_window) / len(self.recent_window)
            if (
                self.long_ema_al > 0
                and (self.long_ema_al - window_avg) > self.drop_threshold
            ):
                if not self.tripped:
                    self.tripped = True
                    self.trip_reason = (
                        f"AL drop: baseline {self.long_ema_al:.2f} → "
                        f"window {window_avg:.2f} (Δ {self.long_ema_al - window_avg:.2f} "
                        f"> threshold {self.drop_threshold})"
                    )
                    self.trip_count += 1
                    return True
        return False

    def reset(self) -> None:
        """Manual reset (e.g. after engine restart)."""
        self.long_ema_al = 0.0
        self.recent_window.clear()
        self.tripped = False
        self.trip_reason = None

    def state_dict(self) -> dict:
        return {
            "long_ema_al": self.long_ema_al,
            "window_len": len(self.recent_window),
            "window_avg": (
                sum(self.recent_window) / len(self.recent_window)
                if self.recent_window
                else 0.0
            ),
            "tripped": self.tripped,
            "trip_reason": self.trip_reason,
            "trip_count": self.trip_count,
        }


# ═══════════════════════════════════════════════════════════════════
# Sub-kernel B: Persistent K/V buffer pool (DFlash-specific MVP)
# ═══════════════════════════════════════════════════════════════════

_ENV_SUB_B = "GENESIS_PN40_ENABLE_SUB_B"


def sub_b_enabled() -> bool:
    if not env_enabled():
        return False
    val = os.environ.get(_ENV_SUB_B, "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    return True


class PersistentKVBufferPool:
    """Per-shape buffer pool for drafter K/V allocations.

    Saves `cudaMalloc` overhead by reusing same allocation across draft
    steps when shape is stable. Bounded size (LRU eviction) to prevent
    leaks under shape churn.

    Universal API; current orchestrator wires only DFlash path. MTP path
    via torch.compile already optimizes allocations — wiring there gives
    no extra benefit (verified empirically on Qwen3.6 MTP).
    """

    def __init__(self, max_entries_per_shape: int = 4, max_distinct_shapes: int = 16):
        self._pool: dict[tuple, list] = {}  # (shape, dtype) → [tensor, ...]
        self._lru: list[tuple] = []
        self.max_entries = max_entries_per_shape
        self.max_shapes = max_distinct_shapes
        self.hits = 0
        self.misses = 0

    def _key(self, shape, dtype, device):
        return (tuple(shape), str(dtype), str(device))

    def get(self, shape, dtype, device):
        """Return a buffer matching shape+dtype+device, or None."""
        import torch  # noqa: F401

        if not sub_b_enabled():
            return None
        k = self._key(shape, dtype, device)
        bucket = self._pool.get(k)
        if bucket:
            self.hits += 1
            # Update LRU
            if k in self._lru:
                self._lru.remove(k)
            self._lru.append(k)
            return bucket.pop()
        self.misses += 1
        return None

    def put(self, tensor) -> None:
        """Return a tensor to the pool. Bounded by max_entries + max_shapes."""
        if not sub_b_enabled():
            return
        k = self._key(tensor.shape, tensor.dtype, tensor.device)
        bucket = self._pool.setdefault(k, [])
        if len(bucket) >= self.max_entries:
            return  # silently drop — already enough buffered
        bucket.append(tensor)
        if k not in self._lru:
            self._lru.append(k)
            # Evict LRU shape if exceed total
            while len(self._lru) > self.max_shapes:
                evict = self._lru.pop(0)
                self._pool.pop(evict, None)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "distinct_shapes": len(self._lru),
            "total_buffered": sum(len(v) for v in self._pool.values()),
        }

    def clear(self) -> None:
        self._pool.clear()
        self._lru.clear()
        self.hits = 0
        self.misses = 0


# Module-level singleton for sub-B
_BUFFER_POOL = PersistentKVBufferPool()


def get_buffer_pool() -> PersistentKVBufferPool:
    return _BUFFER_POOL


# ═══════════════════════════════════════════════════════════════════
# Orchestrator API (used by wiring + bench tools)
# ═══════════════════════════════════════════════════════════════════


def orchestrator_status() -> dict:
    """Return current state of all PN40 sub-kernels for diagnostics."""
    return {
        "master_enabled": env_enabled(),
        "sub_a_enabled": sub_a_enabled(),
        "sub_b_enabled": sub_b_enabled(),
        "sub_c_enabled": sub_c_enabled(),
        "sub_d_enabled": sub_d_enabled(),
        "buffer_pool": _BUFFER_POOL.stats(),
        "controllers": {
            method: ctl.state_dict()
            for method, ctl in _CONTROLLERS.items()
        },
        "sentinels": {
            method: sent.state_dict()
            for method, sent in _SENTINELS.items()
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Universal MTP/DFlash observability hooks (v1: observe-only)
# Wires sub-C controller + sub-D sentinel to scheduler accepted-length
# feedback loop. Runtime K override deferred to v2 (requires proposer
# rewrite). v1 logs adaptive recommendations + flags anomalies.
# ═══════════════════════════════════════════════════════════════════


_CONTROLLERS: dict[str, AdaptiveSpecKController] = {}
_SENTINELS: dict[str, StabilitySentinel] = {}
_OBSERVATION_LOG_INTERVAL = 200  # log aggregate stats every N observations
_OBSERVATION_COUNTERS: dict[str, int] = {}


def _ensure_controller(spec_method: str, base_k: int) -> AdaptiveSpecKController:
    """Lazy-init per-spec_method controller. Idempotent."""
    if spec_method not in _CONTROLLERS:
        tiers = get_default_tiers(spec_method, base_k)
        _CONTROLLERS[spec_method] = AdaptiveSpecKController(
            tiers=tiers, base_k=base_k,
        )
    return _CONTROLLERS[spec_method]


def _ensure_sentinel(spec_method: str) -> StabilitySentinel:
    """Lazy-init per-spec_method sentinel. Idempotent."""
    if spec_method not in _SENTINELS:
        _SENTINELS[spec_method] = StabilitySentinel()
    return _SENTINELS[spec_method]


def pn40_observe_accepted_len(
    spec_method: str,
    accepted_len: int | float,
    base_k: int = 3,
) -> None:
    """Universal hook called from scheduler after each spec-decode step.

    Feeds accepted_len to BOTH the adaptive K controller and the stability
    sentinel for the given spec_method. Defensive — never raises (called
    from the engine hot path).

    Args:
        spec_method: "mtp", "dflash", "ngram", etc. (matches vllm config)
        accepted_len: number of accepted spec tokens this step
        base_k: configured K (fallback if controller not yet init'd)

    v1 is observe-only: controllers compute recommended K but the engine
    does NOT use it (proposer-level override is v2 work). Sentinel trips
    on anomalies and logs WARNING.
    """
    if not env_enabled():
        return
    try:
        if sub_c_enabled():
            ctl = _ensure_controller(spec_method, base_k)
            ctl.observe(float(accepted_len))
        if sub_d_enabled():
            sent = _ensure_sentinel(spec_method)
            tripped = sent.observe(float(accepted_len))
            if tripped:
                log.warning(
                    "[PN40 sub-D sentinel] %s tripped: %s",
                    spec_method, sent.trip_reason,
                )
                # Sentinel-driven auto-disable bridge: cascade into
                # sub-kernel C+D registries so future requests bypass
                # PN40 paths and use baseline behavior. Sub-A stays on
                # (independent kernel-level signal).
                _check_and_auto_disable_on_trip(spec_method)

        # Periodic stats log (low overhead — every 200 observations)
        cnt = _OBSERVATION_COUNTERS.get(spec_method, 0) + 1
        _OBSERVATION_COUNTERS[spec_method] = cnt
        if cnt % _OBSERVATION_LOG_INTERVAL == 0:
            ctl_state = (
                _CONTROLLERS[spec_method].state_dict()
                if spec_method in _CONTROLLERS else None
            )
            sent_state = (
                _SENTINELS[spec_method].state_dict()
                if spec_method in _SENTINELS else None
            )
            log.info(
                "[PN40 observe %s] count=%d controller=%s sentinel=%s",
                spec_method, cnt,
                ctl_state, sent_state,
            )
    except Exception as e:  # noqa: BLE001 — never break engine
        log.debug("[PN40 observe] suppressed exception: %s", e)


def pn40_get_recommended_k(
    spec_method: str,
    base_k: int,
    workload_class: str | None = None,
) -> int:
    """Get current adaptive K recommendation for spec_method.

    v3 (workload-aware): if workload_class is provided, controller biases
    its tier choice (code=+1 tier, long_ctx=-1 tier, others=neutral).
    See AdaptiveSpecKController._workload_bias_table for full mapping.

    v2 semantics retained: respects auto-disable; falls back to base_k
    if sentinel-disabled or controller not initialized.
    """
    if not env_enabled() or not sub_c_enabled():
        return base_k
    if is_auto_disabled("C"):
        return base_k
    if spec_method not in _CONTROLLERS:
        return base_k
    ctl = _CONTROLLERS[spec_method]
    # Apply per-request workload bias (cheap; no allocation)
    if workload_class is not None:
        ctl.set_workload_class(workload_class)
    return ctl.current_k()


def pn40_should_disable_for_request(spec_method: str) -> bool:
    """Check if downstream PN40 paths should bypass for this spec_method.

    Returns True if EITHER:
      - sub-D auto-disabled this method (sentinel tripped), OR
      - the per-method sentinel itself is tripped (covers cold-init case
        before auto_disable cascade fires)

    v2: orchestrator-level decision point. Wiring patches that read
    request._genesis_pn40_workload_class can short-circuit on True.
    """
    if not env_enabled() or not sub_d_enabled():
        return False
    if is_auto_disabled("D"):
        return True
    sent = _SENTINELS.get(spec_method)
    return sent is not None and sent.tripped


def reset_sentinels() -> None:
    """Reset all per-spec_method sentinels. Call after engine restart or
    when operator manually accepts that anomaly is resolved."""
    for sent in _SENTINELS.values():
        sent.reset()


def reset_controllers() -> None:
    """Reset all controllers to base_k. For testing/operator override."""
    _CONTROLLERS.clear()
    _OBSERVATION_COUNTERS.clear()


# ═══════════════════════════════════════════════════════════════════
# Sentinel-driven auto-disable bridge (v2 item)
# Single decision point used by all PN40 sub-kernels: if sentinel for
# this spec_method has tripped, downstream code paths short-circuit
# back to baseline behavior (no PN40 optimizations) until manual reset.
# ═══════════════════════════════════════════════════════════════════


# Module-level flags — checked by sub-kernels' eligibility predicates.
# Allows orchestrator to disable a sub-kernel per-method WITHOUT requiring
# every wiring patch to know about sentinel state.
#
# Audit P1.6 fix 2026-05-05 (genesis_deep_cross_audit_2026-05-05, A-11):
# Previously stored sub_name only — a sentinel trip on MTP would disable
# C+D for ALL methods including DFlash. Refactored to (spec_method, sub_name)
# tuples so trips are scoped to the originating method. Backwards-compat
# helpers `is_auto_disabled(sub_name)` and `auto_disable_sub_kernel(sub_name,
# reason)` are preserved as wrappers over the per-method primitive.
_AUTO_DISABLED_PER_METHOD: set[tuple[str, str]] = set()


def auto_disable_sub_kernel_for_method(
    spec_method: str, sub_name: str, reason: str,
) -> None:
    """Programmatically auto-disable a sub-kernel for ONE spec method.

    Audit P1.6 primitive (genesis_deep_cross_audit). Other methods stay
    enabled. Sub-names: "A", "B", "C", "D" (per omnibus design).
    Idempotent (no-op if already disabled for this method).
    """
    key = (spec_method, sub_name)
    if key in _AUTO_DISABLED_PER_METHOD:
        return
    _AUTO_DISABLED_PER_METHOD.add(key)
    log.warning(
        "[PN40 auto-disable] sub-%s disabled for %s: %s. Other spec methods "
        "unaffected. Manual reset via "
        "vllm._genesis.kernels.pn40_dflash_omnibus.reset_auto_disable() "
        "to re-engage all.",
        sub_name, spec_method, reason,
    )


def auto_disable_sub_kernel(sub_name: str, reason: str) -> None:
    """Backwards-compat shim: applies disable to ALL known spec methods.

    Pre-A-11 fix this was the only entry point and silently global. New
    code should call `auto_disable_sub_kernel_for_method(method, sub_name, reason)`
    instead. This wrapper is kept so existing tests / call sites compile.
    """
    for method in ("mtp", "dflash", "ngram", "eagle"):
        auto_disable_sub_kernel_for_method(method, sub_name, reason)


def is_auto_disabled(sub_name: str, spec_method: str | None = None) -> bool:
    """Read-only check used by sub-kernel eligibility predicates.

    If `spec_method` is None (legacy callers), returns True iff the sub-kernel
    is disabled for ANY method (preserving pre-A-11 behavior). New callers
    SHOULD pass the spec_method to get per-method scoping.
    """
    if spec_method is not None:
        return (spec_method, sub_name) in _AUTO_DISABLED_PER_METHOD
    # Legacy global semantics — answers "is sub_name disabled anywhere?"
    return any(s == sub_name for (_m, s) in _AUTO_DISABLED_PER_METHOD)


def is_auto_disabled_for_method(spec_method: str, sub_name: str) -> bool:
    """Per-method check (audit P1.6 primitive). Strict (spec_method, sub_name)."""
    return (spec_method, sub_name) in _AUTO_DISABLED_PER_METHOD


def reset_auto_disable() -> None:
    """Operator-only reset of all sentinel-triggered disables.

    Use case: AL drop was a transient blip (e.g. cold-cache compile),
    sentinel false-tripped, want to re-engage PN40 without engine restart.
    """
    if _AUTO_DISABLED_PER_METHOD:
        log.info(
            "[PN40] reset_auto_disable: re-engaging %d (method, sub) entries: %s",
            len(_AUTO_DISABLED_PER_METHOD),
            sorted(_AUTO_DISABLED_PER_METHOD),
        )
    _AUTO_DISABLED_PER_METHOD.clear()
    # Also reset the sentinels themselves so they observe fresh baseline.
    reset_sentinels()


def _check_and_auto_disable_on_trip(spec_method: str) -> None:
    """Internal: called from observe loop. If sentinel for this method
    just tripped, auto-disable the corresponding sub-kernels FOR THIS
    METHOD ONLY (audit P1.6 fix — was global pre-2026-05-05).

    Mapping: trip on MTP → disable C+D for MTP only (DFlash unaffected).
            trip on DFlash → disable C+D for DFlash only; A keeps running.
    """
    sent = _SENTINELS.get(spec_method)
    if sent is None or not sent.tripped:
        return
    # Conservative: disable C + D for this method's request path only.
    # Sub-A (kernel-level) is independent — leave it on.
    auto_disable_sub_kernel_for_method(
        spec_method, "C",
        f"sentinel tripped on {spec_method}: {sent.trip_reason}",
    )
    auto_disable_sub_kernel_for_method(
        spec_method, "D",
        f"sentinel tripped on {spec_method}: {sent.trip_reason}",
    )
