# SPDX-License-Identifier: Apache-2.0
"""TDD test for P87 v7.62.10 — text-patch rewrite of Marlin sub-tile pad.

The v7.62 implementation used class-rebind (capture original methods +
monkey-patch new ones). Empirically observed under torch.compile +
FULL cudagraph capture: dynamo refused to trace through the wrapper
indirection and crashed with `Attempted to call function marked as
skipped`. The wrapper closure over `_ORIGINAL_APPLY_WEIGHTS` global
was the trigger.

This test enforces the v7.62.10 invariants: the patch IS a text-patch
(no class-rebind / no monkey-patch globals), the marker is versioned,
all 5 sub-patches are present and required, and the anchors line up
with what the patch replaces.
"""
from __future__ import annotations

import pytest

from vllm._genesis.wiring.kernels.patch_87_marlin_pad_sub_tile import (
    GENESIS_P87_MARKER,
    P87_APPLY_NEW,
    P87_APPLY_OLD,
    P87_CAN_IMPLEMENT_NEW,
    P87_CAN_IMPLEMENT_OLD,
    P87_IMPORTS_NEW,
    P87_IMPORTS_OLD,
    P87_LOGGER_NEW,
    P87_LOGGER_OLD,
    P87_PWA_NEW,
    P87_PWA_OLD,
    _make_patcher,
)


# ─── Marker invariants ───────────────────────────────────────────────────


def test_p87_marker_versioned_textpatch():
    """The marker should embed v7.62.10 + textpatch tag so re-applies
    don't no-op against a stale class-rebind marker.
    """
    assert "v7.62.10_textpatch" in GENESIS_P87_MARKER, (
        f"P87 marker {GENESIS_P87_MARKER!r} should embed v7.62.10_textpatch"
    )
    assert "vllm#40361" in GENESIS_P87_MARKER, (
        "P87 marker should reference upstream PR for drift detection"
    )


# ─── No class-rebind residue ─────────────────────────────────────────────


def test_p87_module_has_no_class_rebind_globals():
    """The class-rebind variant defined module-level globals to capture
    the original methods (e.g. _ORIGINAL_APPLY_WEIGHTS). Those must be
    gone — they are the dynamo crash trigger.
    """
    import vllm._genesis.wiring.kernels.patch_87_marlin_pad_sub_tile as mod

    forbidden_names = [
        "_ORIGINAL_APPLY_WEIGHTS",
        "_ORIGINAL_PROCESS_WEIGHTS_AFTER_LOADING",
        "_ORIGINAL_CAN_IMPLEMENT",
    ]
    for name in forbidden_names:
        assert not hasattr(mod, name), (
            f"P87 module still exposes {name} — class-rebind residue. "
            "v7.62.10 must be pure text-patch."
        )


def test_p87_uses_text_patcher_not_class_rebind():
    """Confirm the apply path goes through TextPatcher, not class-rebind."""
    patcher = _make_patcher()
    if patcher is None:
        pytest.skip("vllm not installed (resolve_vllm_file returned None)")
    from vllm._genesis.wiring.text_patch import TextPatcher

    assert isinstance(patcher, TextPatcher), (
        "P87 _make_patcher must return TextPatcher instance"
    )


# ─── Sub-patch structure ─────────────────────────────────────────────────


def test_p87_has_five_required_sub_patches():
    """v7.62.10 specifies exactly 5 sub-patches, all required."""
    patcher = _make_patcher()
    if patcher is None:
        pytest.skip("vllm not installed")
    assert len(patcher.sub_patches) == 5, (
        f"Expected 5 sub-patches, got {len(patcher.sub_patches)}"
    )
    for sp in patcher.sub_patches:
        assert sp.required, f"sub-patch {sp.name!r} must be required"


def test_p87_sub_patch_names_complete():
    """Sub-patches must cover: imports, logger+round_up, can_implement,
    process_weights_after_loading prelude, apply_weights output slice.
    """
    patcher = _make_patcher()
    if patcher is None:
        pytest.skip("vllm not installed")
    names = {sp.name for sp in patcher.sub_patches}
    expected = {
        "p87_imports",
        "p87_logger_round_up_imports",
        "p87_can_implement_padded",
        "p87_pwa_with_maybe_pad_n",
        "p87_apply_weights_slice",
    }
    assert names == expected, (
        f"Sub-patch name set mismatch.\nGot:      {names}\nExpected: {expected}"
    )


# ─── Anchor / replacement integrity ──────────────────────────────────────


@pytest.mark.parametrize("old,new,label", [
    (P87_IMPORTS_OLD, P87_IMPORTS_NEW, "imports"),
    (P87_LOGGER_OLD, P87_LOGGER_NEW, "logger_round_up"),
    (P87_CAN_IMPLEMENT_OLD, P87_CAN_IMPLEMENT_NEW, "can_implement"),
    (P87_PWA_OLD, P87_PWA_NEW, "pwa_maybe_pad_n"),
    (P87_APPLY_OLD, P87_APPLY_NEW, "apply_slice"),
])
def test_p87_anchors_nonempty_and_replacements_differ(old, new, label):
    """Every sub-patch must have a non-empty anchor and a replacement
    that actually differs from the anchor (otherwise it's a no-op
    that hides drift).
    """
    assert old.strip(), f"{label}: anchor is empty"
    assert new.strip(), f"{label}: replacement is empty"
    assert old != new, f"{label}: replacement equals anchor (no-op patch)"


@pytest.mark.parametrize("new,label", [
    (P87_IMPORTS_NEW, "imports"),
    (P87_LOGGER_NEW, "logger_round_up"),
    (P87_CAN_IMPLEMENT_NEW, "can_implement"),
    (P87_PWA_NEW, "pwa_maybe_pad_n"),
    (P87_APPLY_NEW, "apply_slice"),
])
def test_p87_each_replacement_carries_genesis_breadcrumb(new, label):
    """Every modified region must carry a `[Genesis P87` breadcrumb so
    `git diff` and on-disk forensics can trace which patch authored each
    edit (drift detection relies on this).
    """
    assert "[Genesis P87" in new, (
        f"{label}: replacement missing `[Genesis P87` breadcrumb"
    )


# ─── Semantic invariants of the rewrite ──────────────────────────────────


def test_p87_can_implement_uses_round_up():
    """can_implement must wrap partition_weight_shape[1] with round_up
    so sub-tile shards report supported.
    """
    assert "_genesis_p87_round_up" in P87_CAN_IMPLEMENT_NEW, (
        "can_implement replacement must call round_up helper"
    )
    assert "_GENESIS_P87_MIN_THREAD_N" in P87_CAN_IMPLEMENT_NEW, (
        "can_implement replacement must reference MIN_THREAD_N constant"
    )


def test_p87_pwa_inserts_maybe_pad_n_method():
    """The PWA sub-patch must INSERT a new `_maybe_pad_n` method and
    call it as the very first statement of process_weights_after_loading.
    """
    assert "def _maybe_pad_n(self, layer:" in P87_PWA_NEW, (
        "PWA replacement must define _maybe_pad_n method"
    )
    # Ensure the call site is BEFORE the device = ... line (i.e. first stmt).
    pwa_call_idx = P87_PWA_NEW.find("self._maybe_pad_n(layer)")
    device_idx = P87_PWA_NEW.find("device = getattr(layer, self.w_q_name)")
    assert pwa_call_idx > 0, (
        "PWA replacement must call self._maybe_pad_n(layer) inside PWA body"
    )
    assert pwa_call_idx < device_idx, (
        "self._maybe_pad_n(layer) must be called BEFORE the device = ... line "
        "(first statement of process_weights_after_loading)"
    )


def test_p87_pwa_stores_marlin_orig_n():
    """_maybe_pad_n must record orig_n on the layer so apply_weights can
    slice correctly. The early-return no-op path must also set it.
    """
    assert "layer._marlin_orig_n = orig_n" in P87_PWA_NEW, (
        "_maybe_pad_n must set layer._marlin_orig_n"
    )
    # Defense: the assignment must come BEFORE the early-return so that
    # the no-op case (already aligned) still gets the attribute.
    assign_idx = P87_PWA_NEW.find("layer._marlin_orig_n = orig_n")
    early_return_idx = P87_PWA_NEW.find("if padded_n == orig_n:")
    assert assign_idx < early_return_idx, (
        "layer._marlin_orig_n must be set BEFORE the `if padded_n == orig_n` "
        "early-return so the aligned no-op path still records orig_n"
    )


def test_p87_apply_weights_slices_output():
    """apply_weights replacement must slice the output back to orig_n
    and pad bias if caller supplied at orig_n.
    """
    assert "_marlin_orig_n" in P87_APPLY_NEW, (
        "apply_weights replacement must read _marlin_orig_n from layer"
    )
    # Slice may be split across lines for line-length; collapse whitespace
    # and check the slice expression.
    import re
    collapsed = re.sub(r"\s+", " ", P87_APPLY_NEW)
    assert "[ ..., :_genesis_p87_orig_n ]" in collapsed or \
        "[..., :_genesis_p87_orig_n]" in collapsed, (
        "apply_weights replacement must slice output back to orig_n"
    )
    # Bias-pad guard:
    assert "F.pad(" in P87_APPLY_NEW, (
        "apply_weights replacement must F.pad bias when caller supplied "
        "at orig_n but kernel was loaded at padded_n"
    )


def test_p87_imports_add_dataclasses_and_F():
    """The imports sub-patch must add dataclasses (for dataclasses.replace
    of self.config) and torch.nn.functional as F (for F.pad).
    """
    assert "import dataclasses" in P87_IMPORTS_NEW, (
        "imports must add `import dataclasses` for self.config replace"
    )
    assert "import torch.nn.functional as F" in P87_IMPORTS_NEW, (
        "imports must add `import torch.nn.functional as F` for F.pad"
    )


# ─── Drift detection — anchors must be specific ──────────────────────────


def test_p87_anchors_have_enough_context():
    """Anchors must be long enough to be unique against the full file.
    Short anchors risk matching multiple sites and patching the wrong one.
    Heuristic: at least 80 chars per anchor.
    """
    for label, anchor in [
        ("imports", P87_IMPORTS_OLD),
        ("logger", P87_LOGGER_OLD),
        ("can_implement", P87_CAN_IMPLEMENT_OLD),
        ("pwa", P87_PWA_OLD),
        ("apply", P87_APPLY_OLD),
    ]:
        assert len(anchor) >= 80, (
            f"{label}: anchor too short ({len(anchor)} chars). Risk of "
            "matching multiple sites in marlin.py."
        )
