# SPDX-License-Identifier: Apache-2.0
"""TDD tests for PN35 — skip inputs_embeds buffer for text-only models.

Verifies:
  - Wiring module imports cleanly + exposes apply()
  - Dispatcher registry + apply_all integration
  - Default-on (env-flag GENESIS_DISABLE_PN35_INPUTS_EMBEDS_OPTIONAL
    not set means ENABLED)
  - Anchor matches the EXACT existing vllm allocation block
  - Replacement preserves the original allocation behind a guard
  - Marker uniqueness (idempotency)
  - Drift-marker specificity (won't false-positive on sibling PN
    insertions — see club-3090#19 finding 1 lesson)
  - Sub-patch soft-fail tolerance: if one site fails, the other
    still applies (~64 MiB savings still meaningful)

Author: Sandermage (Sander) Barzov Aleksandr.
Backport: vllm#35975 by AjAnubolu.
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────
# Module imports + dispatcher integration
# ─────────────────────────────────────────────────────────────────


def test_pn35_wiring_module_imports():
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N35_inputs_embeds_optional as mod,
    )
    assert hasattr(mod, "apply")
    assert hasattr(mod, "GENESIS_PN35_MARKER")
    assert hasattr(mod, "PN35_PART1_ANCHOR")
    assert hasattr(mod, "PN35_PART2_ANCHOR")


def test_pn35_in_dispatcher_registry():
    from vllm._genesis.dispatcher import PATCH_REGISTRY

    assert "PN35" in PATCH_REGISTRY
    e = PATCH_REGISTRY["PN35"]
    assert e["env_flag"] == "GENESIS_ENABLE_PN35_INPUTS_EMBEDS_OPTIONAL"
    # Default ON — strict savings, no regression possible
    assert e["default_on"] is True, (
        "PN35 must default ON — text-only is the common case + the "
        "patch is a guard, no regression possible"
    )
    assert e["upstream_pr"] == 35975


def test_pn35_registered_in_apply_all():
    from vllm._genesis.patches.apply_all import (
        PATCH_REGISTRY as APPLY_REGISTRY,
    )

    names = [name for name, _ in APPLY_REGISTRY]
    found = [n for n in names if "PN35" in n]
    assert len(found) == 1, (
        f"PN35 not registered in apply_all (or duplicated). "
        f"Names matching 'PN35': {found}"
    )


def test_pn35_credit_mentions_upstream_author_and_noonghunna():
    """Attribution chain: AjAnubolu (UPSTREAM PR author) + noonghunna
    (sidecar pattern) + club-3090#32 (issue origin)."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY

    credit = PATCH_REGISTRY["PN35"]["credit"]
    assert "AjAnubolu" in credit, "missing UPSTREAM PR author credit"
    assert "noonghunna" in credit, "missing sidecar pattern credit"
    assert "35975" in credit, "missing PR number"


def test_pn35_credit_mentions_club_3090_issue_32_origin():
    """The patch was prompted by club-3090#32 (RossNE99 + GuiPerPT
    WSL2 OOM reports). Credit must capture this."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY

    credit = PATCH_REGISTRY["PN35"]["credit"]
    assert "club-3090#32" in credit or "WSL2" in credit, (
        "credit must reference the cross-rig issue that prompted "
        "the absorption (club-3090#32)"
    )


# ─────────────────────────────────────────────────────────────────
# Anchor + replacement structural correctness
# ─────────────────────────────────────────────────────────────────


def test_pn35_part1_anchor_targets_gpu_model_runner_make_buffer():
    """part1 must match the EXACT _make_buffer() call in
    gpu_model_runner.py:713 (current vllm pin)."""
    from vllm._genesis.wiring.perf_hotfix.patch_N35_inputs_embeds_optional import (
        PN35_PART1_ANCHOR,
    )
    assert "self.inputs_embeds = self._make_buffer(" in PN35_PART1_ANCHOR
    assert "self.max_num_tokens" in PN35_PART1_ANCHOR
    assert "self.inputs_embeds_size" in PN35_PART1_ANCHOR
    assert "numpy=False" in PN35_PART1_ANCHOR


def test_pn35_part2_anchor_targets_llm_base_proposer_torch_zeros():
    """part2 must match torch.zeros(...) in llm_base_proposer.py:205."""
    from vllm._genesis.wiring.perf_hotfix.patch_N35_inputs_embeds_optional import (
        PN35_PART2_ANCHOR,
    )
    assert "self.inputs_embeds = torch.zeros(" in PN35_PART2_ANCHOR
    assert "self.max_num_tokens, self.inputs_embeds_size" in PN35_PART2_ANCHOR
    assert "device=device," in PN35_PART2_ANCHOR


def test_pn35_part1_replacement_preserves_original_under_guard():
    """The patch must be a strict ADDITION — if multimodal/prompt_embeds
    is True, original allocation runs as before. Zero regression path."""
    from vllm._genesis.wiring.perf_hotfix.patch_N35_inputs_embeds_optional import (
        PN35_PART1_REPLACEMENT,
    )
    # Default value: None
    assert "self.inputs_embeds = None" in PN35_PART1_REPLACEMENT
    # Guard re-allocates for multimodal OR prompt_embeds
    assert "self.supports_mm_inputs or self.enable_prompt_embeds" in PN35_PART1_REPLACEMENT
    # Original allocation preserved inside the guard
    assert "self._make_buffer(" in PN35_PART1_REPLACEMENT


def test_pn35_part2_replacement_preserves_original_under_guard():
    from vllm._genesis.wiring.perf_hotfix.patch_N35_inputs_embeds_optional import (
        PN35_PART2_REPLACEMENT,
    )
    assert "self.inputs_embeds = None" in PN35_PART2_REPLACEMENT
    assert "self.supports_mm_inputs" in PN35_PART2_REPLACEMENT
    # Original torch.zeros call preserved under guard
    assert "torch.zeros(" in PN35_PART2_REPLACEMENT


# ─────────────────────────────────────────────────────────────────
# Marker + drift-marker specificity (lesson from club-3090#19 F1)
# ─────────────────────────────────────────────────────────────────


def test_pn35_marker_unique_and_versioned():
    from vllm._genesis.wiring.perf_hotfix.patch_N35_inputs_embeds_optional import (
        GENESIS_PN35_MARKER,
    )
    assert "PN35" in GENESIS_PN35_MARKER
    assert "v7.69" in GENESIS_PN35_MARKER
    assert "vllm#35975" in GENESIS_PN35_MARKER


def test_pn35_drift_markers_specific_no_generic_collision_risk():
    """Drift markers must be SPECIFIC. Generic '[Genesis PN35' would
    risk collision if any future PN3X patch inserts text into the same
    file. See club-3090#19 finding 1 (PN30 part3 false-positive)."""
    import os
    import tempfile

    from vllm._genesis.wiring.perf_hotfix import (
        patch_N35_inputs_embeds_optional as mod,
    )
    import vllm._genesis.guards as guards

    with tempfile.TemporaryDirectory() as td:
        for sub in ["v1/worker", "v1/spec_decode"]:
            os.makedirs(os.path.join(td, sub))
        for fname in [
            "v1/worker/gpu_model_runner.py",
            "v1/spec_decode/llm_base_proposer.py",
        ]:
            with open(os.path.join(td, fname), "w") as f:
                f.write("# placeholder for resolve_vllm_file\n")

        orig = guards.vllm_install_root
        guards.vllm_install_root = lambda: td
        try:
            p1 = mod._make_patcher_part1()
            p2 = mod._make_patcher_part2()
        finally:
            guards.vllm_install_root = orig

    for patcher in [p1, p2]:
        assert patcher is not None
        for m in patcher.upstream_drift_markers:
            # Must NOT be a bare '[Genesis PN35' prefix that could
            # collide. Either it's specific to part-N's own insertion,
            # or it's an upstream-merge signal (vllm#35975).
            assert (
                "v7.69" in m
                or "inputs_embeds_optional" in m
                or "vllm#35975" in m
            ), (
                f"drift marker {m!r} too generic — risk of "
                f"false-positive collision on sibling patches"
            )


# ─────────────────────────────────────────────────────────────────
# apply() runtime semantics
# ─────────────────────────────────────────────────────────────────


def test_pn35_apply_skips_when_env_disabled(monkeypatch):
    """Operators can opt-out via GENESIS_ENABLE_PN35_INPUTS_EMBEDS_OPTIONAL=0."""
    monkeypatch.setenv("GENESIS_ENABLE_PN35_INPUTS_EMBEDS_OPTIONAL", "0")
    from vllm._genesis.wiring.perf_hotfix.patch_N35_inputs_embeds_optional import (
        apply,
    )
    status, reason = apply()
    assert status == "skipped"


def test_pn35_apply_skips_when_vllm_install_missing(monkeypatch):
    """If vllm install root not resolvable, soft-skip cleanly."""
    monkeypatch.setenv("GENESIS_ENABLE_PN35_INPUTS_EMBEDS_OPTIONAL", "1")
    import vllm._genesis.wiring.perf_hotfix.patch_N35_inputs_embeds_optional as mod

    # Force vllm_install_root to return None
    monkeypatch.setattr(mod, "vllm_install_root", lambda: None)

    status, reason = mod.apply()
    assert status == "skipped"
    assert "vllm install root" in reason.lower()


def test_pn35_two_sub_patches_independent():
    """Soft-fail tolerance: if one of the two patcher targets is
    missing/drifted, the other should still apply for partial savings.
    apply() reports based on what landed, not all-or-nothing."""
    import inspect

    from vllm._genesis.wiring.perf_hotfix import (
        patch_N35_inputs_embeds_optional as mod,
    )

    src = inspect.getsource(mod.apply)
    # Implementation should iterate over the two patches, log warnings
    # for failures, and report based on `any_applied or any_idempotent`.
    assert "any_applied" in src or "any_idempotent" in src or "log.warning" in src, (
        "apply() should be tolerant of single-site failures; the "
        "other half still gives ~64 MiB savings"
    )


# ─────────────────────────────────────────────────────────────────
# Composition + documentation requirements
# ─────────────────────────────────────────────────────────────────


def test_pn35_module_docstring_explains_savings_math():
    """Docstring must show the actual MiB saved per buffer (~64 MiB on
    Qwen3.6-27B at default config) so operators understand the impact."""
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N35_inputs_embeds_optional as mod,
    )
    doc = mod.__doc__ or ""
    assert "64 MiB" in doc, (
        "module docstring must show the per-buffer savings math"
    )
    assert "max_num_tokens" in doc and "hidden_size" in doc.lower() or "inputs_embeds_size" in doc


def test_pn35_module_docstring_mentions_composition_with_cliff_2_stack():
    """PN35 is particularly useful WITH P103 + PN32 (Cliff 2 stack).
    Docstring should mention this composition."""
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N35_inputs_embeds_optional as mod,
    )
    doc = mod.__doc__ or ""
    assert "Cliff 2" in doc or "P103" in doc or "PN32" in doc, (
        "must document composition with Cliff 2 stack — that's the "
        "primary motivating use case"
    )


def test_pn35_module_docstring_credits_upstream_author():
    """vllm#35975 author (AjAnubolu) must be credited in the module
    docstring per Sander's no-AI-credit / always-credit-humans policy."""
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N35_inputs_embeds_optional as mod,
    )
    doc = mod.__doc__ or ""
    assert "AjAnubolu" in doc, "UPSTREAM author must be credited"
    assert "35975" in doc


def test_pn35_module_docstring_mentions_wsl2_use_case():
    """The patch was prompted by WSL2 OOM reports (RossNE99 + GuiPerPT
    on club-3090#32). Docstring must mention WSL2 so operators searching
    for 'WSL OOM' find this patch."""
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N35_inputs_embeds_optional as mod,
    )
    doc = mod.__doc__ or ""
    assert "WSL" in doc, "must mention WSL2 use case for discoverability"


# ─────────────────────────────────────────────────────────────────
# Sync gate
# ─────────────────────────────────────────────────────────────────


def test_pn35_in_patches_md():
    """PATCHES.md must list PN35 — operators browse PATCHES.md to
    find available patches."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    patches_md = (repo_root / "docs" / "PATCHES.md").read_text()
    # Sync gate test (test_patches_md_sync) enforces this generically;
    # this is the explicit per-patch pin
    assert "PN35" in patches_md, (
        "PATCHES.md missing PN35 — add a row in the perf_hotfix table"
    )
