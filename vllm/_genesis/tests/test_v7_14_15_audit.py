# SPDX-License-Identifier: Apache-2.0
"""TDD audit suite for v7.14 + v7.15 patch additions and drift checks.

Validates THREE properties:

1. **Anchor presence**: Every new patch (P64, P65 v2, P66, P68/P69, P70)
   has its OLD anchor present in the pinned vLLM source. If upstream
   drifts, these tests fail loudly.

2. **Drift safety for legacy patches**: P14 (BlockTable tail zero), P28
   (GDN core_attn_out prealloc), P38 (TQ continuation_prefill workspace),
   P39a (FLA persistent A pool) — verify their anchors still match the
   pinned vLLM source. These patches don't have their own per-patch tests
   and could silently break on pin upgrade.

3. **Dispatcher consistency**: every PATCH_REGISTRY entry has a matching
   apply_patch_NN function in apply_all and the env_flag/marker structure
   is sane.

These are CPU-side AST + text-anchor tests. Behavioral validation (does
the patch actually fix the bug under load) requires GPU + full vLLM
stack — covered by integration suite at Genesis_Doc/MTP_TEST_RESULTS_*
and chunked_ladder.py / extended_ladder.py.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


# ────────────────────────────────────────────────────────────────────────
#   Pinned vLLM source resolution
# ────────────────────────────────────────────────────────────────────────

# Path to the pinned vLLM source tree used for anchor checks.
# In CI this can be set via env GENESIS_VLLM_PIN_PATH; defaults to the
# /tmp sparse-checkout of fe9c3d6c5 used during v7.14 development.
_PINNED_VLLM_ROOT = Path("/tmp/vllm_pin/vllm")


def _pinned_file(rel: str) -> Path:
    """Return path to a pinned vLLM source file. Skip test if missing."""
    p = _PINNED_VLLM_ROOT / rel
    if not p.exists():
        pytest.skip(f"pinned vLLM source not available: {p}")
    return p


def _load_wiring_module(name: str):
    """Load a Genesis wiring module without executing the vllm package
    (avoids requiring torch/triton on test host)."""
    # Stub heavy deps if not present
    for stub in [
        "vllm",
        "vllm._genesis",
        "vllm._genesis.guards",
        "vllm._genesis.wiring",
        "vllm._genesis.wiring.text_patch",
    ]:
        if stub not in sys.modules:
            sys.modules[stub] = types.ModuleType(stub)

    # Ensure callable stubs
    g = sys.modules["vllm._genesis.guards"]
    if not hasattr(g, "resolve_vllm_file"):
        g.resolve_vllm_file = lambda x: None
        g.vllm_install_root = lambda: None

    tp = sys.modules["vllm._genesis.wiring.text_patch"]
    if not hasattr(tp, "TextPatcher"):

        class _Stub:
            pass

        for n in ("TextPatcher", "TextPatchResult", "TextPatch"):
            setattr(tp, n, _Stub)

    repo_root = Path(__file__).resolve().parents[3]
    file_path = repo_root / "vllm" / "_genesis" / "wiring" / f"{name}.py"
    if not file_path.exists():
        pytest.skip(f"wiring module not present: {file_path}")
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # py3.13 dataclass introspection needs this
    spec.loader.exec_module(mod)
    return mod


# ════════════════════════════════════════════════════════════════════════
#   v7.14 / v7.15 NEW PATCHES — anchor presence
# ════════════════════════════════════════════════════════════════════════


class TestNewPatchAnchors:
    """Each new patch's OLD anchor must be present in pinned vLLM source."""

    def test_p64_qwen3coder_anchors(self):
        m = _load_wiring_module("patch_64_qwen3coder_mtp_streaming")
        parser_path = _pinned_file("tool_parsers/qwen3coder_tool_parser.py")
        serving_path = _pinned_file(
            "entrypoints/openai/chat_completion/serving.py"
        )
        parser_content = parser_path.read_text()
        serving_content = serving_path.read_text()
        assert m.QWEN3CODER_OLD in parser_content, (
            "P64 sub-patch A anchor missing in qwen3coder_tool_parser.py"
        )
        assert m.QWEN3COD_FNEND_OLD in parser_content, (
            "P64 sub-patch B anchor missing in qwen3coder_tool_parser.py"
        )
        assert m.SERVING_SHOULD_OLD in serving_content, (
            "P64 sub-patch C anchor missing in serving.py"
        )

    def test_p65_v2_turboquant_anchor(self):
        m = _load_wiring_module("patch_65_turboquant_spec_cg_downgrade")
        path = _pinned_file("v1/attention/backends/turboquant_attn.py")
        content = path.read_text()
        assert m.TQ_CG_SUPPORT_OLD in content, (
            "P65 v2 anchor missing in turboquant_attn.py"
        )

    def test_p66_cudagraph_filter_anchor(self):
        m = _load_wiring_module("patch_66_cudagraph_size_divisibility_filter")
        path = _pinned_file("config/vllm.py")
        content = path.read_text()
        assert m.P66_OLD in content, "P66 anchor missing in config/vllm.py"

    def test_p68_p69_serving_anchor(self):
        m = _load_wiring_module("patch_68_69_long_ctx_tool_adherence")
        path = _pinned_file("entrypoints/openai/chat_completion/serving.py")
        content = path.read_text()
        assert m.P6869_OLD in content, "P68/P69 anchor missing in serving.py"

    def test_p70_strict_ngram_anchor(self):
        m = _load_wiring_module("patch_70_auto_strict_ngram")
        path = _pinned_file("config/speculative.py")
        content = path.read_text()
        assert m.P70_OLD in content, (
            "P70 anchor missing in config/speculative.py"
        )


# ════════════════════════════════════════════════════════════════════════
#   LEGACY PATCHES — drift safety check (× spec-decode mixed batches)
# ════════════════════════════════════════════════════════════════════════
# These patches predate v7.14 and don't have per-patch unit tests. They
# pre-allocate buffers / hook into hot paths that interact with spec-decode.
# Each test verifies the anchor is still present in the pinned source —
# if upstream drifts, the patch should fail to apply (NOT silently corrupt).
#
# Behavioral validation under spec-decode mixed batches requires GPU.


class TestLegacyPatchDriftSafety:
    """Verify P14, P28, P38, P39a anchors still match pinned vLLM."""

    def test_p14_block_table_anchor(self):
        m = _load_wiring_module("patch_14_block_table")
        path = _pinned_file("v1/worker/block_table.py")
        content = path.read_text()
        # P14 may have multiple anchors; check at least one is present
        # by looking for known field name pre-patch state
        assert "append_row" in content, (
            "BlockTable.append_row symbol missing — P14 target moved"
        )
        # If the patch defines explicit OLD anchor strings, check them.
        for attr in dir(m):
            if attr.endswith("_OLD") and isinstance(
                getattr(m, attr), str
            ):
                anchor = getattr(m, attr)
                if len(anchor) > 50:
                    assert anchor in content, (
                        f"P14 anchor {attr} missing in block_table.py"
                    )

    def test_p28_gdn_core_attn_anchor(self):
        m = _load_wiring_module("patch_28_gdn_core_attn")
        path = _pinned_file("model_executor/layers/mamba/gdn_linear_attn.py")
        content = path.read_text()
        # Same defensive pattern — verify anchor fragments
        for attr in dir(m):
            if attr.endswith("_OLD") and isinstance(
                getattr(m, attr), str
            ):
                anchor = getattr(m, attr)
                if len(anchor) > 50:
                    assert anchor in content, (
                        f"P28 anchor {attr} missing in gdn_linear_attn.py"
                    )

    def test_p38_tq_continuation_anchor(self):
        m = _load_wiring_module("patch_38_tq_continuation_memory")
        path = _pinned_file("v1/attention/backends/turboquant_attn.py")
        content = path.read_text()
        for attr in dir(m):
            if attr.endswith("_OLD") and isinstance(
                getattr(m, attr), str
            ):
                anchor = getattr(m, attr)
                if len(anchor) > 50:
                    assert anchor in content, (
                        f"P38 anchor {attr} missing in turboquant_attn.py"
                    )

    def test_p39_fla_kkt_anchor(self):
        m = _load_wiring_module("patch_39_fla_kkt_buffer")
        path = _pinned_file("model_executor/layers/fla/ops/chunk.py")
        if not path.exists():
            # FLA chunk path varies — try alternate
            alt = _pinned_file("model_executor/layers/fla/ops")
            pytest.skip(f"FLA chunk source structure varies: {alt}")
        content = path.read_text()
        for attr in dir(m):
            if attr.endswith("_OLD") and isinstance(
                getattr(m, attr), str
            ):
                anchor = getattr(m, attr)
                if len(anchor) > 50:
                    assert anchor in content, (
                        f"P39a anchor {attr} missing in chunk.py"
                    )


# ════════════════════════════════════════════════════════════════════════
#   DISPATCHER REGISTRY consistency check
# ════════════════════════════════════════════════════════════════════════


class TestDispatcherRegistry:
    """Every PATCH_REGISTRY entry must have well-formed metadata."""

    def test_registry_has_required_fields(self):
        # Load dispatcher without importing vllm
        for stub in ["vllm", "vllm._genesis"]:
            if stub not in sys.modules:
                sys.modules[stub] = types.ModuleType(stub)
        repo_root = Path(__file__).resolve().parents[3]
        path = repo_root / "vllm" / "_genesis" / "dispatcher.py"
        spec = importlib.util.spec_from_file_location("dispatcher", str(path))
        d = importlib.util.module_from_spec(spec)
        sys.modules["dispatcher"] = d  # py3.13 dataclass introspection needs this
        spec.loader.exec_module(d)
        for pid, meta in d.PATCH_REGISTRY.items():
            assert "title" in meta, f"{pid} missing title"
            assert "env_flag" in meta, f"{pid} missing env_flag"
            assert "default_on" in meta, f"{pid} missing default_on"
            assert "category" in meta, f"{pid} missing category"
            assert isinstance(meta["env_flag"], str)
            # Schema validator pattern: ^GENESIS_[A-Z][A-Z0-9_]*$
            # Most patches use GENESIS_ENABLE_*, but legacy patches use
            # GENESIS_LEGACY_* (placeholder for pre-dispatcher patches that
            # don't actually read an env var). Both are schema-clean.
            assert meta["env_flag"].startswith("GENESIS_"), (
                f"{pid} env_flag must start with GENESIS_"
            )

    def test_v7_14_v7_15_patches_in_registry(self):
        for stub in ["vllm", "vllm._genesis"]:
            if stub not in sys.modules:
                sys.modules[stub] = types.ModuleType(stub)
        repo_root = Path(__file__).resolve().parents[3]
        path = repo_root / "vllm" / "_genesis" / "dispatcher.py"
        spec = importlib.util.spec_from_file_location("dispatcher", str(path))
        d = importlib.util.module_from_spec(spec)
        sys.modules["dispatcher"] = d  # py3.13 dataclass introspection needs this
        spec.loader.exec_module(d)
        # All v7.14 + v7.15 patches must be present
        for pid in ("P64", "P65", "P66", "P68", "P69", "P70"):
            assert pid in d.PATCH_REGISTRY, (
                f"{pid} missing from PATCH_REGISTRY"
            )
        # P63 must be marked deprecated
        assert d.PATCH_REGISTRY["P63"].get("deprecated") is True

    def test_should_apply_default_skips_all(self):
        # Without env flags, should_apply must return False for all opt-in
        # patches (default_on=False).
        for stub in ["vllm", "vllm._genesis", "vllm._genesis.config_detect"]:
            if stub not in sys.modules:
                sys.modules[stub] = types.ModuleType(stub)
        # Stub config_detect.recommend if not present
        if not hasattr(sys.modules["vllm._genesis.config_detect"], "recommend"):
            sys.modules["vllm._genesis.config_detect"].recommend = (
                lambda pid: ("neutral", "no profile")
            )
        repo_root = Path(__file__).resolve().parents[3]
        path = repo_root / "vllm" / "_genesis" / "dispatcher.py"
        spec = importlib.util.spec_from_file_location("dispatcher2", str(path))
        d = importlib.util.module_from_spec(spec)
        sys.modules["dispatcher2"] = d  # py3.13 dataclass introspection needs this
        spec.loader.exec_module(d)

        import os as _os
        # Save + clear ALL env flags
        saved = {
            k: _os.environ.pop(k, None)
            for k in list(_os.environ.keys())
            if k.startswith("GENESIS_ENABLE_")
        }
        try:
            for pid, meta in d.PATCH_REGISTRY.items():
                if not meta.get("default_on"):
                    decision, reason = d.should_apply(pid)
                    assert decision is False, (
                        f"{pid} should default-skip without env: {reason}"
                    )
        finally:
            for k, v in saved.items():
                if v is not None:
                    _os.environ[k] = v


# ════════════════════════════════════════════════════════════════════════
#   APPLY_ALL ↔ DISPATCHER consistency
# ════════════════════════════════════════════════════════════════════════


class TestApplyAllConsistency:
    def test_v7_14_v7_15_register_decorators_present(self):
        repo_root = Path(__file__).resolve().parents[3]
        apply_all = (
            repo_root / "vllm" / "_genesis" / "patches" / "apply_all.py"
        )
        content = apply_all.read_text()
        # Each new patch must be registered
        for pid, fn_name_part in (
            ("P64", "qwen3coder_mtp_streaming"),
            ("P65", "turboquant_spec_cg_downgrade"),
            ("P66", "cudagraph_size_filter"),
            ("P68/P69", "long_ctx_tool_adherence"),
            ("P70", "auto_strict_ngram"),
        ):
            assert f"@register_patch(\"{pid}" in content or (
                f"@register_patch(\"{pid.split('/')[0]}" in content
            ), f"{pid} missing @register_patch decorator in apply_all.py"
            assert fn_name_part in content, (
                f"apply_patch_*_{fn_name_part}* function missing"
            )
