# SPDX-License-Identifier: Apache-2.0
"""Genesis compat — richer applies_to predicate evaluator.

Existing PATCH_REGISTRY entries declare `applies_to` as a flat dict
of accepted values per profile key:

    "P67": {
        "applies_to": {"is_turboquant": [True]},
    },

This is fine for binary gates but doesn't express common scenarios:

  - "this patch matters when (model is qwen3-family) AND
     (quant in {fp8, autoround_int4})" → needs OR over quant values
  - "this patch is only relevant on Ampere or newer (sm_86+)
     but NOT on Blackwell consumer" → needs range AND negation
  - "this patch fires on TQ + spec-decode but NOT on TQ alone" →
     needs explicit AND of two boolean flags

Without DSL we end up sprinkling if/elif/else into wiring modules,
which makes patches harder to introspect (`genesis doctor` can't tell
why a patch will skip if the gate is opaque Python code).

This module provides:

  evaluate(rule, profile) -> (matched: bool, why: str)

Supported rule shapes (recursive):

  Leaf forms (single key/value gate):
    {"is_turboquant": True}                  # exact match
    {"is_turboquant": [True, False]}         # value in list
    {"model_class": "qwen3"}                 # str equality
    {"model_class": ["qwen3", "qwen3_5"]}    # str in list
    {"vllm_version_range": ">=0.20.0,<0.21.0"}  # PEP 440 specifier

  Compound forms:
    {"all_of": [<rule>, <rule>, ...]}        # AND
    {"any_of": [<rule>, <rule>, ...]}        # OR
    {"not": <rule>}                          # NOT
    {"none_of": [<rule>, ...]}               # NOR (all must be false)

Backwards compatible: a flat dict of leaf forms is treated as `all_of`
implicitly, so existing PATCH_REGISTRY entries keep working without
edits. Migration is opt-in per patch.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from typing import Any

# Reserved compound keys that can't be misinterpreted as leaf gate keys.
_COMPOUND_KEYS = ("all_of", "any_of", "not", "none_of")

# Leaf keys that delegate to version_check rather than profile-dict lookup.
_VERSION_KEYS = (
    "vllm_version_range",
    "torch_version_min", "triton_version_min",
    "cuda_runtime_min", "nvidia_driver_min", "python_version_min",
    "compute_capability_min", "compute_capability_max",
)


def evaluate(rule: Any, profile: dict[str, Any]) -> tuple[bool, str]:
    """Evaluate `rule` against `profile`. Returns (matched, reason).

    `profile` is a flat dict — typically merged from
    `model_detect.get_model_profile()` + selected fields from
    `version_check.detect_versions()`. Keys absent from `profile` are
    treated as 'unknown' → conservative pass (the patch is allowed to
    apply on imperfect detection).
    """
    if rule is None or rule == {}:
        # Empty rule = always matches
        return True, "no constraints declared"

    if not isinstance(rule, dict):
        return False, f"rule must be dict, got {type(rule).__name__}"

    # Compound forms
    if "all_of" in rule:
        for i, sub in enumerate(rule["all_of"]):
            ok, why = evaluate(sub, profile)
            if not ok:
                return False, f"all_of[{i}] failed: {why}"
        return True, f"all_of (n={len(rule['all_of'])}) satisfied"

    if "any_of" in rule:
        reasons = []
        for sub in rule["any_of"]:
            ok, why = evaluate(sub, profile)
            if ok:
                return True, f"any_of matched: {why}"
            reasons.append(why)
        return False, f"any_of: no branch matched ({'; '.join(reasons)})"

    if "not" in rule:
        ok, why = evaluate(rule["not"], profile)
        if ok:
            return False, f"not: inner matched ({why})"
        return True, f"not: inner did not match"

    if "none_of" in rule:
        for i, sub in enumerate(rule["none_of"]):
            ok, why = evaluate(sub, profile)
            if ok:
                return False, f"none_of[{i}] matched (forbidden): {why}"
        return True, f"none_of (n={len(rule['none_of'])}) all rejected"

    # Leaf form — version-related keys delegate to version_check
    version_constraints = {k: v for k, v in rule.items() if k in _VERSION_KEYS}
    profile_constraints = {k: v for k, v in rule.items() if k not in _VERSION_KEYS}

    if version_constraints:
        try:
            from vllm._genesis.compat.version_check import (
                check_version_constraints,
            )
            vc_ok, vc_results = check_version_constraints(version_constraints)
            if not vc_ok:
                failed = [r for r in vc_results if r.matched is False]
                if failed:
                    return False, f"version: {failed[0].reason}"
                return False, "version: constraint violation"
        except Exception:
            # Don't fail the patch because version checking failed
            pass

    # Leaf profile-dict gates — exact match or value in list
    for key, expected in profile_constraints.items():
        if key in _COMPOUND_KEYS:
            return False, (
                f"compound key {key!r} mixed with leaf keys at same level "
                f"— wrap in {{'{key}': ...}} alone"
            )
        actual = profile.get(key)
        if actual is None:
            # Conservative: detector couldn't resolve → don't block patch
            continue
        if isinstance(expected, (list, tuple, set)):
            allowed = list(expected)
            if actual not in allowed:
                return False, f"{key}={actual!r} not in {allowed!r}"
        else:
            if actual != expected:
                return False, f"{key}={actual!r} != expected {expected!r}"

    return True, "all leaf gates satisfied"


def normalize_legacy_rule(rule: Any) -> dict:
    """Convert a flat legacy-style applies_to (e.g. {'is_turboquant': [True]})
    into an explicit `all_of` shape. The output evaluates identically.

    Already-explicit rules (with `all_of` / `any_of` / `not` / `none_of`
    at top level) pass through unchanged.
    """
    if not isinstance(rule, dict) or not rule:
        return rule
    if any(k in _COMPOUND_KEYS for k in rule):
        # Already compound — pass through
        return rule
    # Legacy flat dict → wrap each leaf as its own gate, AND them
    return {"all_of": [{k: v} for k, v in rule.items()]}


def explain(rule: Any, profile: dict[str, Any]) -> list[str]:
    """Render an indented explanation tree of how `rule` evaluated
    against `profile`. Useful for `genesis doctor` and `genesis explain`.
    """
    return _explain_recurse(rule, profile, indent=0)


def _explain_recurse(rule: Any, profile, indent: int) -> list[str]:
    pad = "  " * indent
    if rule is None or rule == {}:
        return [f"{pad}(empty rule → always match)"]
    if not isinstance(rule, dict):
        return [f"{pad}<malformed: {type(rule).__name__}>"]

    if "all_of" in rule:
        lines = [f"{pad}all_of:"]
        for sub in rule["all_of"]:
            lines.extend(_explain_recurse(sub, profile, indent + 1))
        return lines
    if "any_of" in rule:
        lines = [f"{pad}any_of:"]
        for sub in rule["any_of"]:
            lines.extend(_explain_recurse(sub, profile, indent + 1))
        return lines
    if "not" in rule:
        lines = [f"{pad}not:"]
        lines.extend(_explain_recurse(rule["not"], profile, indent + 1))
        return lines
    if "none_of" in rule:
        lines = [f"{pad}none_of:"]
        for sub in rule["none_of"]:
            lines.extend(_explain_recurse(sub, profile, indent + 1))
        return lines

    # Leaf
    ok, why = evaluate(rule, profile)
    mark = "✓" if ok else "✗"
    return [f"{pad}{mark} {rule}  ({why})"]
