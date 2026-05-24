# SPDX-License-Identifier: Apache-2.0
"""Genesis v7.12 — universal runtime configuration profiler.

Companion to `model_detect.py`. Where `model_detect.py` answers
"what kind of MODEL are we running" (MoE / hybrid / TQ),
this module answers "what kind of RUNTIME does it run in":

  * scheduler knobs: `max_num_seqs`, `max_num_batched_tokens`,
    `enable_chunked_prefill`, `enable_prefix_caching`
  * spec-decode: enabled / method (ngram / mtp / eagle / draft) / N
  * compilation: cudagraph_mode (NONE / FULL_AND_PIECEWISE / ...),
    torch.compile mode
  * upstream-fix presence: did PR #40384 / #40798 / #40792 land in
    this vLLM build, detected by structural code probes?

Each Genesis patch can use this profile to make a **smart self-decision**
about whether it adds value on the current runtime. Examples:

  * P36 (shared decode buffers) — skip if `pr40798_active=True`
    (workspace manager already does it natively) OR if
    `max_num_seqs < 8` (memory savings marginal: ~60 MiB).
  * P40 (TQ grouped decode) — skip if `pr40792_active=True`.
  * P56 (spec-decode safe-path guard) — skip if `cudagraph_mode == NONE`
    (workaround already in effect; P56 redundant) OR always
    (deprecated by noonghunna's six-probe ladder; routing fix
    is partial and #40831 root cause is in cudagraph capture itself).
  * P22 / P38 / P44 — TQ infrastructure foundation, not skipped.

The cache lives in the same singleton style as `model_detect`:
one process-wide query, immutable after first call.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger("genesis.config_detect")


# ──────────────────────────────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────────────────────────────

_CACHED_PROFILE: Optional[dict[str, Any]] = None


def clear_for_tests() -> None:
    """TESTS ONLY. Reset cached profile so next query re-probes."""
    global _CACHED_PROFILE
    _CACHED_PROFILE = None


def _try_get_vllm_config() -> Optional[Any]:
    """Return current vllm config or None if not yet set."""
    try:
        from vllm.config import get_current_vllm_config
        return get_current_vllm_config()
    except Exception as e:
        log.debug("[config_detect] get_current_vllm_config unavailable: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────
# Scheduler / runtime knobs
# ──────────────────────────────────────────────────────────────────────

def _probe_scheduler(cfg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sc = getattr(cfg, "scheduler_config", None)
    if sc is None:
        return out
    for attr in (
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "enable_prefix_caching",
    ):
        v = getattr(sc, attr, None)
        if v is not None:
            out[attr] = v
    return out


# ──────────────────────────────────────────────────────────────────────
# Speculative decoding
# ──────────────────────────────────────────────────────────────────────

def _probe_spec_decode_from_argv() -> dict[str, Any] | None:
    """Fallback: probe argv (own + container PID 1 cmdline + env override)
    for --speculative-config when called BEFORE vllm config is constructed
    (e.g. apply_all phase). Returns None if no spec-decode flag detected.

    Sources tried in order:
      1. GENESIS_FORCE_SPEC_DECODE env (operator override) — value used as method
      2. own sys.argv (covers `python -m vllm` invocations directly)
      3. /proc/1/cmdline (covers Docker entrypoint that runs apply_all then vllm serve)

    Patterns parsed:
      --speculative-config '{"method":"mtp",...}'
      --speculative-config={"method":"ngram",...}
      --speculative-model <path>
    """
    import os
    # 1. Explicit operator override (escape hatch)
    forced = os.environ.get("GENESIS_FORCE_SPEC_DECODE", "").strip()
    if forced and forced.lower() not in ("0", "false", "no", "off"):
        return {"spec_decode_enabled": True, "spec_decode_method": forced,
                "spec_decode_method_kind": (
                    "mtp" if "mtp" in forced.lower()
                    else "ngram" if "ngram" in forced.lower()
                    else "eagle" if "eagle" in forced.lower()
                    else forced)}
    # 2. + 3. Build a combined argv string from sys.argv + /proc/1/cmdline
    import sys
    argv_parts = [" ".join(sys.argv)]
    try:
        with open("/proc/1/cmdline", "rb") as f:
            # cmdline args are NUL-separated
            pid1_cmd = f.read().replace(b"\x00", b" ").decode(errors="replace")
            argv_parts.append(pid1_cmd)
    except (OSError, IOError):
        pass  # not on Linux or no /proc — fine, no extra signal
    argv = " ".join(argv_parts)
    if "--speculative-config" not in argv and "--speculative-model" not in argv:
        return None
    out: dict[str, Any] = {"spec_decode_enabled": True}
    # Try to extract method (match outside /proc shell-quote noise)
    import re
    m = re.search(r'"method"\s*:\s*\\?"(\w+)\\?"', argv)
    if m:
        method = m.group(1)
        out["spec_decode_method"] = method
        out["spec_decode_method_kind"] = (
            "ngram" if "ngram" in method.lower()
            else "mtp" if "mtp" in method.lower()
            else "eagle" if "eagle" in method.lower()
            else "draft_model" if "draft" in method.lower()
            else method
        )
    return out


def _probe_spec_decode(cfg: Any) -> dict[str, Any]:
    """Detect whether spec-decode is configured + which method + N."""
    out: dict[str, Any] = {"spec_decode_enabled": False}
    spec = getattr(cfg, "speculative_config", None)
    if spec is None:
        # Fallback: check sys.argv for --speculative-config (apply_all phase
        # runs BEFORE vllm config is built, so cfg.speculative_config is
        # always None at that time).
        argv_probe = _probe_spec_decode_from_argv()
        if argv_probe is not None:
            return argv_probe
        return out
    out["spec_decode_enabled"] = True
    for attr in ("method", "num_speculative_tokens", "model"):
        v = getattr(spec, attr, None)
        if v is not None:
            out[f"spec_decode_{attr}"] = v
    # Convenience aliases
    method = out.get("spec_decode_method") or ""
    out["spec_decode_method_kind"] = (
        "ngram" if "ngram" in str(method).lower()
        else "mtp" if "mtp" in str(method).lower()
        else "eagle" if "eagle" in str(method).lower()
        else "draft_model" if "draft" in str(method).lower()
        else str(method)
    )
    return out


# ──────────────────────────────────────────────────────────────────────
# Compilation / CUDA graph mode
# ──────────────────────────────────────────────────────────────────────

def _probe_compilation(cfg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cc = getattr(cfg, "compilation_config", None)
    if cc is None:
        return out
    # cudagraph_mode is sometimes an enum, sometimes string
    cgm = getattr(cc, "cudagraph_mode", None)
    if cgm is not None:
        cgm_str = getattr(cgm, "name", str(cgm))
        out["cudagraph_mode"] = cgm_str
        out["cudagraph_capture_active"] = ("NONE" not in cgm_str.upper())
    # torch.compile mode
    tcm = getattr(cc, "mode", None) or getattr(cc, "compile_mode", None)
    if tcm is not None:
        out["compile_mode"] = getattr(tcm, "name", str(tcm))
    return out


# ──────────────────────────────────────────────────────────────────────
# Cache config
# ──────────────────────────────────────────────────────────────────────

def _probe_cache(cfg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cc = getattr(cfg, "cache_config", None)
    if cc is None:
        return out
    for attr in ("kv_cache_dtype", "block_size", "cache_dtype", "gpu_memory_utilization"):
        v = getattr(cc, attr, None)
        if v is not None:
            out[attr] = v
    return out


# ──────────────────────────────────────────────────────────────────────
# Upstream-fix presence detection (structural probes)
# ──────────────────────────────────────────────────────────────────────

def _probe_pr40798_active() -> bool:
    """PR #40798 ('Share decode scratch workspace across layers') is
    detected by absence of the per-layer register_buffer calls in
    `attention.py::_init_turboquant_buffers`.

    Returns True if upstream has merged or local backport applied
    (markers identical from caller's perspective).
    """
    try:
        from vllm.model_executor.layers.attention import attention as attn_mod
        import inspect
        src = inspect.getsource(attn_mod)
        # Pre-merge: contains register_buffer("_tq_mid_o_buf"
        # Post-merge: that string is removed; comment about workspace_manager appears
        return (
            'register_buffer("_tq_mid_o_buf"' not in src
            or "v1 workspace manager" in src
            or "[PR #40798 backport]" in src
        )
    except Exception as e:
        log.debug("[config_detect] pr40798 probe failed: %s", e)
        return False


def _probe_pr40792_active() -> bool:
    """PR #40792 ('Optimize k8v4 decode attention with GQA head grouping')
    detected by presence of `_tq_grouped_decode_stage1` symbol in the
    Triton module.
    """
    try:
        from vllm.v1.attention.ops import triton_turboquant_decode as tt
        return hasattr(tt, "_tq_grouped_decode_stage1")
    except Exception as e:
        log.debug("[config_detect] pr40792 probe failed: %s", e)
        return False


def _probe_pr40384_active() -> bool:
    """PR #40384 ('Exclude O(1) Mamba groups from hybrid KV cache token capacity')
    — Sander co-author. Detect via the helper function it adds to
    kv_cache_utils.
    """
    try:
        from vllm.v1.core import kv_cache_utils as kvu
        return hasattr(kvu, "token_capacity_kv_cache_groups") or hasattr(
            kvu, "exclude_o1_mamba_groups"
        )
    except Exception as e:
        log.debug("[config_detect] pr40384 probe failed: %s", e)
        return False


def _probe_pr40074_active() -> bool:
    """PR #40074 ('Fix TurboQuant KV cache index-out-of-bounds in Triton
    decode kernel') detected by `safe_page_idx` symbol in the Triton kernel
    source.
    """
    try:
        from vllm.v1.attention.ops import triton_turboquant_decode as tt
        import inspect
        src = inspect.getsource(tt)
        return "safe_page_idx" in src
    except Exception as e:
        log.debug("[config_detect] pr40074 probe failed: %s", e)
        return False


def _probe_workspace_manager_present() -> bool:
    """Check if v1 WorkspaceManager exists at all (orthogonal to #40798
    using it for TQ specifically)."""
    try:
        from vllm.v1.worker.workspace import (
            current_workspace_manager,
            is_workspace_manager_initialized,
        )
        _ = current_workspace_manager, is_workspace_manager_initialized
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# Recommendation engine
# ──────────────────────────────────────────────────────────────────────

# Marginal-benefit threshold for P36-class shared-buffer patches.
# Below this max_num_seqs, the per-layer buffer cost is < ~50 MiB total
# on a 40-layer model, so the shared pool's payoff is small.
P36_MIN_BENEFIT_MAX_NUM_SEQS = 8

# Marginal-benefit threshold for P37 MoE intermediate cache.
P37_MIN_BENEFIT_MAX_NUM_SEQS = 8


def _recommend_for_patches(profile: dict[str, Any]) -> dict[str, Any]:
    """Cost/benefit advisory for runtime-aware patches. Each entry maps
    a patch ID to (recommendation, reason) tuple. Patches still get the
    final say — they call `recommend(patch_id)` and decide whether to
    honor the recommendation.

    Recommendations:
      "apply"       — clear positive signal
      "skip:<r>"    — skip; reason after colon (caller can override via env)
      "redundant:<r>" — upstream already does this; safe to skip
      "deprecated:<r>" — empirically known not to help; use alternative
      "neutral"     — no strong signal either way; default to opt-in flag
    """
    rec: dict[str, str] = {}

    max_num_seqs = profile.get("max_num_seqs") or 0
    pr40798 = profile.get("pr40798_active", False)
    pr40792 = profile.get("pr40792_active", False)
    pr40384 = profile.get("pr40384_active", False)
    pr40074 = profile.get("pr40074_active", False)
    cudagraph_active = profile.get("cudagraph_capture_active", True)
    spec_decode = profile.get("spec_decode_enabled", False)

    # P36: shared decode buffers
    if pr40798:
        rec["P36"] = "redundant:upstream PR #40798 (workspace manager) active — skip"
    elif max_num_seqs and max_num_seqs < P36_MIN_BENEFIT_MAX_NUM_SEQS:
        rec["P36"] = (
            f"skip:max_num_seqs={max_num_seqs} < {P36_MIN_BENEFIT_MAX_NUM_SEQS} "
            f"(memory benefit ~< 50 MiB; not worth maintaining)"
        )
    else:
        rec["P36"] = "apply"

    # P40 (opt-in): TQ grouped decode
    if pr40792:
        rec["P40"] = "redundant:upstream PR #40792 (native tl.dot kernel) active — skip"
    else:
        rec["P40"] = "neutral"  # opt-in env flag still required

    # P67 (opt-in): TurboQuant multi-query kernel for spec-decode K+1 verify
    # 2026-04-27 v756 bisect: P67's `max_query_len > 1` heuristic ALSO matches
    # chunked-prefill batches (not just spec-verify K+1). Without spec-decode,
    # P67 misroutes prefill batches through the multi-query kernel which
    # assumes uniform K+1 layout per request, causing scrambled output and
    # downstream `hidden_states[logits_indices]` overflow under sustained
    # burst. v756 reproducer 100% reliable; B5 confirmed P67=0 stable.
    # See docs/reference/V756_STABILITY_INVESTIGATION_20260427.md.
    if not spec_decode:
        rec["P67"] = (
            "skip:no speculative_config — P67 multi-query kernel is for "
            "spec-decode K+1 verify only. Without spec-decode the dispatch "
            "heuristic (`max_query_len > 1`) misroutes chunked-prefill "
            "batches and causes IndexKernel overflow under sustained burst. "
            "See V756_STABILITY_INVESTIGATION_20260427.md."
        )
    else:
        rec["P67"] = "neutral"  # opt-in env flag still required

    # P56 (opt-in): spec-decode safe-path guard
    # **Empirically deprecated** — see Genesis_Doc/spec_decode_investigation/
    if not spec_decode:
        rec["P56"] = "skip:no speculative_config — guard would never fire"
    elif not cudagraph_active:
        rec["P56"] = "redundant:cudagraph_mode=NONE already in effect — guard noop"
    else:
        rec["P56"] = (
            "deprecated:noonghunna's Probe 4 disproved routing-layer hypothesis. "
            "Use --compilation-config '{\"cudagraph_mode\":\"NONE\"}' for the real "
            "(and only known) #40831 fix until upstream lands a proper graph-correct "
            "spec-decode + TurboQuant path."
        )

    # P9 (in legacy monolith): KV-cache token capacity
    if pr40384:
        rec["P9"] = "redundant:upstream PR #40384 (token_capacity_kv_cache_groups) merged"
    else:
        rec["P9"] = "apply"

    # PR #40074 IOOB clamp — note for P40 area
    if pr40074:
        rec["IOOB_clamp"] = "redundant:upstream PR #40074 already merged — no need to backport"
    else:
        rec["IOOB_clamp"] = "neutral"

    # P37 MoE intermediate cache
    if max_num_seqs and max_num_seqs < P37_MIN_BENEFIT_MAX_NUM_SEQS:
        rec["P37"] = (
            f"skip:max_num_seqs={max_num_seqs} < {P37_MIN_BENEFIT_MAX_NUM_SEQS} "
            f"(MoE pool benefit marginal at low concurrency)"
        )
    else:
        rec["P37"] = "neutral"  # already opt-in via GENESIS_ENABLE_P37

    return rec


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def get_runtime_profile() -> dict[str, Any]:
    """Cached probe of vllm runtime config + upstream fix presence.

    Returns a dict that's safe to log (no secrets, no large objects).
    Schema is stable across calls; missing keys mean probe failed.

    Conservative fallback: if `get_current_vllm_config()` is unavailable,
    returns `{"resolved": False, ...}` with empty-default values. Callers
    should treat unresolved as "no signal" and apply patches as their own
    guards see fit.
    """
    global _CACHED_PROFILE
    if _CACHED_PROFILE is not None:
        # H3 fix (research agent aae2c26c, 2026-04-28): re-probe if previously
        # cached as UNRESOLVED but vllm config is now available. Avoids
        # stale unresolved profile masking subsequent resolved data after
        # apply_all phase ends. Cheap: just one is-not-None check on cache hit.
        if not _CACHED_PROFILE.get("resolved", False):
            if _try_get_vllm_config() is not None:
                _CACHED_PROFILE = None  # fall through to fresh probe
            else:
                return _CACHED_PROFILE
        else:
            return _CACHED_PROFILE

    cfg = _try_get_vllm_config()
    if cfg is None:
        # 2026-04-27 v756 fix: even without vllm_config (apply_all phase),
        # try to detect spec-decode from sys.argv / /proc/1/cmdline /
        # GENESIS_FORCE_SPEC_DECODE env. Critical for P67 safety gate to
        # work correctly during apply_all (which runs BEFORE vllm config
        # is constructed but AFTER the launch script has set up everything).
        argv_spec = _probe_spec_decode_from_argv()
        spec_decode_enabled = bool(argv_spec and argv_spec.get("spec_decode_enabled"))
        profile: dict[str, Any] = {
            "resolved": False,
            "spec_decode_enabled": spec_decode_enabled,
            "cudagraph_capture_active": True,  # conservative default
            "max_num_seqs": None,
            "pr40384_active": _probe_pr40384_active(),
            "pr40074_active": _probe_pr40074_active(),
            "pr40798_active": _probe_pr40798_active(),
            "pr40792_active": _probe_pr40792_active(),
            "workspace_manager_present": _probe_workspace_manager_present(),
        }
        if argv_spec:
            for k, v in argv_spec.items():
                if k != "spec_decode_enabled":
                    profile[k] = v
        profile["recommendations"] = _recommend_for_patches(profile)
        # H3 fix: cache the unresolved profile too. Saves ~33 patches × 5
        # source-file-read probes = ~100 ms boot time. Re-probe trigger at
        # cache-hit time when vllm config becomes available (above).
        _CACHED_PROFILE = profile
        return profile

    profile = {"resolved": True}
    profile.update(_probe_scheduler(cfg))
    profile.update(_probe_spec_decode(cfg))
    profile.update(_probe_compilation(cfg))
    profile.update(_probe_cache(cfg))
    profile["pr40384_active"] = _probe_pr40384_active()
    profile["pr40074_active"] = _probe_pr40074_active()
    profile["pr40798_active"] = _probe_pr40798_active()
    profile["pr40792_active"] = _probe_pr40792_active()
    profile["workspace_manager_present"] = _probe_workspace_manager_present()
    profile["recommendations"] = _recommend_for_patches(profile)

    _CACHED_PROFILE = profile
    log.info(
        "[Genesis v7.12 config_detect] profile: max_num_seqs=%s spec_decode=%s "
        "cudagraph_active=%s kv=%s | upstream: pr40798=%s pr40792=%s pr40384=%s pr40074=%s",
        profile.get("max_num_seqs"),
        profile.get("spec_decode_enabled"),
        profile.get("cudagraph_capture_active"),
        profile.get("kv_cache_dtype"),
        profile.get("pr40798_active"),
        profile.get("pr40792_active"),
        profile.get("pr40384_active"),
        profile.get("pr40074_active"),
    )
    return profile


def recommend(patch_id: str) -> tuple[str, str]:
    """Return (recommendation, reason) for a specific patch.

    Recommendation values:
      "apply", "neutral"
      "skip:<reason>", "redundant:<reason>", "deprecated:<reason>"

    A patch should typically:
      * apply if rec == "apply" or rec == "neutral"
      * skip cleanly with the given reason if rec starts with skip/redundant/deprecated
      * always honor an env flag override (`GENESIS_FORCE_APPLY_<P>=1`) so
        operators can re-enable a patch even when the recommendation says skip
    """
    profile = get_runtime_profile()
    raw = profile.get("recommendations", {}).get(patch_id, "neutral")
    if ":" in raw:
        rec, reason = raw.split(":", 1)
    else:
        rec, reason = raw, ""
    return rec, reason


def is_force_applied(patch_id: str) -> bool:
    """Operator override: GENESIS_FORCE_APPLY_<patch_id_uppercased>=1
    re-enables a patch even when recommendation says skip."""
    env = f"GENESIS_FORCE_APPLY_{patch_id.upper()}"
    return os.environ.get(env, "").strip().lower() in ("1", "true", "yes", "on")


def should_apply(patch_id: str) -> tuple[bool, str]:
    """Single decision call for a patch's apply().

    Returns (should_apply: bool, reason: str). Use as:

        ok, reason = config_detect.should_apply("P36")
        if not ok:
            return "skipped", reason

    Force-apply env override is respected.
    """
    rec, reason = recommend(patch_id)
    if rec in ("apply", "neutral"):
        return True, reason or f"recommendation={rec}"
    if is_force_applied(patch_id):
        return True, f"forced via GENESIS_FORCE_APPLY_{patch_id.upper()} (orig {rec}: {reason})"
    return False, f"{rec}: {reason}"
