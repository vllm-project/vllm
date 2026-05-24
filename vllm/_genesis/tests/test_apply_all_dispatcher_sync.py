# SPDX-License-Identifier: Apache-2.0
"""Pin the apply_all.py ↔ dispatcher.py PATCH_REGISTRY sync contract.

For every patch that has an `@register_patch` + `apply_patch_<id>_*`
function in `apply_all.py`, there must be a corresponding entry in
`PATCH_REGISTRY` in `dispatcher.py` (and vice versa, with one
documented exception for P68/P69 sharing one apply function).

Why this matters: without this gate, the legacy P1–P46 patches drifted
out of the registry for an entire phase of development. New patches
can land in apply_all without dispatcher metadata (no env_flag, no
schema validation, invisible to `genesis explain` and `genesis list`).
This test catches that drift on every commit.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
APPLY_ALL = REPO_ROOT / "vllm" / "_genesis" / "patches" / "apply_all.py"


# Documented exceptions where one `apply_patch_*` function registers
# multiple registry entries (or vice versa). Update this set ONLY when
# adding a new intentional asymmetry — every entry here is a deviation
# from the 1:1 contract and should have a clear reason.
_KNOWN_REGISTRY_ONLY = frozenset({
    # P68 and P69 share `apply_patch_68_long_ctx_tool_adherence` — both
    # patches modify the same long-context tool-adherence middleware so
    # they ship as one wiring function. Registry tracks them separately
    # for `genesis explain` / docs disambiguation.
    "P69",
    # PN40-classifier is a sub-component of PN40 sub-D — it is wired
    # inside `apply_patch_N40_dflash_omnibus()` (the PN40 omnibus apply
    # path). Registered separately so the dispatcher v2 validator does
    # not warn when scheduler.py reports `PN40-classifier` as applied,
    # but no standalone wiring function exists.
    "PN40-classifier",
    # P51 is a runtime layer-level TQ-active library guard living in
    # kernels/dequant_buffer.py — no env toggle, no apply_patch_* needed.
    # Registered for visibility in `genesis explain` and audit tooling.
    "P51",
    # P102 is a diagnostic-only spec-decode metadata module
    # (vllm/_genesis/spec_meta.py). Activated by direct call from spec
    # paths when GENESIS_ENABLE_P102=1, not via apply_all wiring.
    # Registered for `genesis explain` visibility.
    "P102",
    # PN60 is a preflight DX validator wired into compat/doctor.py — runs
    # BEFORE vLLM loads, not via apply_all. Registered for visibility +
    # operator search-ability in `genesis explain PN60`.
    "PN60",
    # PN61/PN62 are now WIRED (apply_patch_N61_*, apply_patch_N62_*)
    # — left commented here in case future refactor decouples them from
    # apply_all. As of 2026-05-05 they have full wiring + opt-in env flag
    # + class-rebind apply() with idempotency markers; cross-rig validation
    # pending an actual qwen3_vl checkpoint reachable from a Genesis test rig.
    # PN63 is a gpu_profile advisory rule (lives in gpu_profile.py
    # PATCH_RECOMMENDATIONS). Suggest-only, not a runtime patch — so no
    # apply_patch_* function exists by design.
    "PN63",
    # PN64 is a Marlin MoE per-SM tuning placeholder for SM 12.0. The
    # actual config entry lives in kernels/marlin_tuning.py table; the
    # registry entry is for `genesis explain` visibility. Real wiring
    # only matters when SM 12.0 hardware is detected at boot.
    "PN64",
})

_KNOWN_APPLY_ONLY: frozenset[str] = frozenset({
    # No documented exceptions yet — every apply_patch_* should have a
    # registry entry. Add an ID here with a comment if you intentionally
    # ship a wiring function without dispatcher metadata.
})


def _extract_apply_patch_ids() -> set[str]:
    """Parse apply_all.py and return all patch IDs from
    `apply_patch_<id>_*` function names. Format: prefix-letter `P` plus
    the captured ID, so `apply_patch_67b_*` → `P67b`,
    `apply_patch_N32_*` → `PN32`."""
    tree = ast.parse(APPLY_ALL.read_text(encoding="utf-8"))
    ids: set[str] = set()
    pattern = re.compile(r"^apply_patch_([NM]?\d+[a-zA-Z]?)(?:_|$)")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith(
            "apply_patch_"
        ):
            m = pattern.match(node.name)
            if m:
                ids.add("P" + m.group(1))
    return ids


def _load_registry_ids() -> set[str]:
    """Load PATCH_REGISTRY from dispatcher.py without importing the
    full vllm package (so the test runs in CI without GPU)."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    return set(PATCH_REGISTRY.keys())


@pytest.fixture(scope="module")
def apply_ids() -> set[str]:
    return _extract_apply_patch_ids()


@pytest.fixture(scope="module")
def registry_ids() -> set[str]:
    return _load_registry_ids()


def test_no_registry_entries_without_apply_function(
    apply_ids: set[str], registry_ids: set[str]
) -> None:
    """Every PATCH_REGISTRY entry must have a corresponding
    `apply_patch_<id>_*` function in apply_all.py — except for
    documented sharing cases like P68/P69."""
    registry_only = (registry_ids - apply_ids) - _KNOWN_REGISTRY_ONLY
    assert registry_only == set(), (
        f"PATCH_REGISTRY has {len(registry_only)} entries with no "
        f"apply_patch_* function: {sorted(registry_only)}\n"
        "Either add the wiring function in apply_all.py, or document "
        "the asymmetry in _KNOWN_REGISTRY_ONLY in this test."
    )


def test_no_apply_functions_without_registry_entry(
    apply_ids: set[str], registry_ids: set[str]
) -> None:
    """Every `apply_patch_<id>_*` function in apply_all.py must have a
    corresponding PATCH_REGISTRY entry. Without metadata the patch is
    invisible to `genesis explain` / schema validation / lifecycle
    audit / opt-in env discovery."""
    apply_only = (apply_ids - registry_ids) - _KNOWN_APPLY_ONLY
    assert apply_only == set(), (
        f"apply_all.py has {len(apply_only)} apply_patch_* functions "
        f"with no PATCH_REGISTRY entry: {sorted(apply_only)}\n"
        "Add an entry to PATCH_REGISTRY in dispatcher.py with at least "
        "title / env_flag / default_on / category / lifecycle, or "
        "document the asymmetry in _KNOWN_APPLY_ONLY in this test."
    )


def test_documented_exceptions_actually_present(
    registry_ids: set[str], apply_ids: set[str]
) -> None:
    """Sanity check: every ID in _KNOWN_REGISTRY_ONLY must really be in
    the registry but NOT in apply (else the exception is stale)."""
    for pid in _KNOWN_REGISTRY_ONLY:
        assert pid in registry_ids, (
            f"_KNOWN_REGISTRY_ONLY contains {pid!r} which is no longer "
            f"in PATCH_REGISTRY — remove the stale exception"
        )
        assert pid not in apply_ids, (
            f"_KNOWN_REGISTRY_ONLY contains {pid!r} but apply_all.py "
            f"now has an apply_patch_* function — remove the exception"
        )
    for pid in _KNOWN_APPLY_ONLY:
        assert pid in apply_ids, (
            f"_KNOWN_APPLY_ONLY contains {pid!r} which is no longer "
            f"in apply_all.py — remove the stale exception"
        )
        assert pid not in registry_ids, (
            f"_KNOWN_APPLY_ONLY contains {pid!r} but PATCH_REGISTRY "
            f"now has an entry — remove the exception"
        )
