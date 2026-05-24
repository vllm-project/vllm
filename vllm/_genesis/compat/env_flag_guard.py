# SPDX-License-Identifier: Apache-2.0
"""Environment-flag typo guard — silent-no-op shield.

Inspired by vllm#41310 root-cause class: when a feature flag is silently
disabled because of typo / wrong macro / case mismatch, the bug is
**performance** (or correctness) regression that doesn't trigger any
test. Hardest class of bugs to debug — by definition the fix is "look
at config more carefully".

Genesis equivalent: ~110 `GENESIS_ENABLE_*` env flags. Typo in flag
name (`GENESIS_ENABLE_P67c` vs `_P67C`, `_P67_TQ_MULTI_QUERY_KERNEL`
vs `_P67_TQ_MULTI_QUERY_KENRNEL`, etc.) → patch silently doesn't
fire → boot logs say "OK" but performance/feature missing.

This module:
1. Compares `os.environ` keys starting with `GENESIS_ENABLE_` against
   the set of env_flag values registered in `PATCH_REGISTRY`
2. Reports any environment variable that LOOKS like a Genesis flag
   but does NOT match a registered patch
3. Suggests closest match (Levenshtein < 3 chars)
4. Either WARN (default) or RAISE (`GENESIS_TYPO_GUARD_STRICT=1`)

Hooked into apply_all on boot so typos are caught immediately.

Memory precedent: `feedback_p105_dequant_num_stages_noise` —
P105 had dispatcher entry but no apply_all wiring → silently dead → we
measured CV noise instead of real effect. Same class of bug as #41310.

Author: Sandermage 2026-05-04, pattern from vllm#41310.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

log = logging.getLogger("genesis.compat.env_flag_guard")


@dataclass(frozen=True)
class TypoFinding:
    env_var: str             # User-set (mistyped) variable
    closest_known: str | None  # Closest known flag (or None if no good match)
    distance: int | None       # Levenshtein distance to closest_known


def _levenshtein(a: str, b: str) -> int:
    """Pure-Python Levenshtein for short strings (no external deps)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def collect_known_flags() -> set[str]:
    """Walk PATCH_REGISTRY and return all `env_flag` values."""
    try:
        from vllm._genesis.dispatcher import PATCH_REGISTRY
    except Exception:
        return set()
    flags: set[str] = set()
    for meta in PATCH_REGISTRY.values():
        flag = meta.get("env_flag")
        if isinstance(flag, str) and flag:
            flags.add(flag)
    # Genesis also has many tuning knobs (not patch toggles) and observability
    # vars — recognize their prefix to avoid false positives.
    return flags


# Allowlist prefixes/suffixes that LOOK like Genesis flags but are
# tuning/observability knobs, not patch toggles.
_ALLOWLIST_PREFIXES = (
    "GENESIS_ENABLE_",     # all patch toggles share this prefix
    "GENESIS_DISABLE_",    # disable inverse pattern — also typo-scanned
                           # (Audit P2 fix 2026-05-05: was unreachable
                           # because find_typos() iterated only ENABLE_)
)
_ALLOWLIST_SUFFIXES = (
    "_DEBUG", "_VERBOSE", "_LOG_EVERY",
    "_THRESHOLD", "_THRESHOLD_SINGLE",
    "_BLOCK_KV", "_NUM_WARPS", "_NUM_STAGES",
    "_RAM_GB", "_DISK_GB", "_TTL_DAYS",
    "_K_MAX", "_BASE_K",
    "_PIN_POLICY", "_TQ_MAX_MODEL_LEN", "_PROFILE_RUN_CAP_M",
    "_PREALLOC_TOKEN_BUDGET", "_BUFFER_MODE",
    "_PN26_SPARSE_V_THRESHOLD", "_PN26_SPARSE_V_BLOCK_KV",
    "_PN26_SPARSE_V_NUM_WARPS", "_PN26_SPARSE_V_DEBUG",
    "_API_KEY", "_ENDPOINT", "_MODEL",
    "_TYPO_GUARD_STRICT",
    "_VLLM_PIN_POLICY",
)
# Special "disable inverse" pattern: GENESIS_DISABLE_<X>=1 to disable a
# default-ON patch. These are valid even though they don't appear in
# env_flag values.
_DISABLE_PREFIX = "GENESIS_DISABLE_"


def _is_known_or_allowed(env_var: str, known_flags: set[str]) -> bool:
    if env_var in known_flags:
        return True
    if env_var.startswith(_DISABLE_PREFIX):
        return True  # GENESIS_DISABLE_<X> is a valid pattern
    if any(env_var.endswith(s) for s in _ALLOWLIST_SUFFIXES):
        return True
    return False


def find_typos(environ: dict | None = None) -> list[TypoFinding]:
    """Scan environ for likely Genesis flag typos. Returns empty list if clean."""
    if environ is None:
        environ = dict(os.environ)
    known = collect_known_flags()
    findings: list[TypoFinding] = []
    for env_var in environ:
        if not env_var.startswith(_ALLOWLIST_PREFIXES):
            continue
        if _is_known_or_allowed(env_var, known):
            continue
        # Find closest match
        if known:
            closest = min(known, key=lambda k: _levenshtein(env_var, k))
            distance = _levenshtein(env_var, closest)
        else:
            closest, distance = None, None
        # Only flag if reasonably close (avoid false positives on totally
        # unrelated user vars). Distance ≤ 4 = likely typo. Distance > 4
        # = probably custom knob, skip.
        if distance is None or distance <= 4:
            findings.append(TypoFinding(env_var=env_var,
                                        closest_known=closest,
                                        distance=distance))
    return findings


def assert_no_typos(strict: bool | None = None) -> int:
    """Run typo scan + log/raise per `strict` setting.

    Returns the number of findings (0 = clean). When `strict` is None,
    reads `GENESIS_TYPO_GUARD_STRICT` env (default OFF — log only).
    """
    if strict is None:
        strict = os.environ.get(
            "GENESIS_TYPO_GUARD_STRICT", ""
        ).strip().lower() in ("1", "true", "yes", "on")

    findings = find_typos()
    if not findings:
        return 0

    msg_lines = [
        f"[Genesis env-flag guard] Found {len(findings)} suspicious "
        "GENESIS_ENABLE_* variable(s) — possible typos:",
    ]
    for f in findings:
        if f.closest_known and f.distance is not None:
            msg_lines.append(
                f"  ⚠ {f.env_var:50} → did you mean {f.closest_known}? "
                f"(distance={f.distance})"
            )
        else:
            msg_lines.append(
                f"  ⚠ {f.env_var:50} (no close match in PATCH_REGISTRY)"
            )
    msg = "\n".join(msg_lines)

    if strict:
        raise RuntimeError(msg)
    log.warning(msg)
    return len(findings)


def main():
    """CLI entry — `python -m vllm._genesis.compat.env_flag_guard`."""
    import sys
    n = assert_no_typos(strict=False)
    if n == 0:
        print("✓ Genesis env-flag guard: no typos detected")
        return 0
    print(f"⚠ {n} suspicious env var(s); see warnings above")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
