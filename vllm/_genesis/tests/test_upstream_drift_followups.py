# SPDX-License-Identifier: Apache-2.0
"""Tests pinning upstream-PR drift detection for active backports.

These regressions ensure the PATCH_REGISTRY-tracked upstream PRs that
matter to Genesis are wired correctly into each patch's drift markers
or auto-retire probes. If an upstream PR merges and we don't notice,
we either ship duplicate code or block on stale anchors. These tests
catch that class of drift at CI time.

Coverage:
- P3 drift markers include both FP16 (Genesis form) and FP32 (PR #39988
  form) cast staircases.
- P82 drift markers include the PR #40819 block-verify symbols
  (`use_block_verify`, `verify_method`, `SpecVerifyMethod`).
- P5 has an auto-retire probe that detects PR #39931 merge by looking
  for `TQFullAttentionSpec` + `_get_full_attention_layer_indices` in
  the upstream `vllm.model_executor.layers.quantization.turboquant.config`
  module.
"""
from __future__ import annotations

import sys
import types

import pytest


class TestP3DriftBothCastForms:
    def test_p3_drift_markers_cover_fp16_and_fp32(self):
        from vllm._genesis.wiring.legacy.patch_3_tq_bf16_cast import (
            UPSTREAM_DRIFT_MARKERS,
        )
        markers = "\n".join(UPSTREAM_DRIFT_MARKERS)
        assert "tl.float16).to(tl.float8e4b15)" in markers, (
            "P3 must drift-detect Genesis-original FP16 staircase form"
        )
        assert "tl.float32).to(tl.float8e4b15)" in markers, (
            "P3 must drift-detect PR #39988 FP32 staircase form so the "
            "detector fires correctly when upstream merges with FP32"
        )
        assert "PR #39988" in markers


class TestP82DriftCovers40819:
    def test_p82_drift_markers_cover_block_verify(self):
        # P82's drift markers live inline in _make_patcher; read them
        # via the patcher object.
        from vllm._genesis.wiring.spec_decode.patch_82_sglang_acceptance_threshold import (
            _make_patcher,
        )
        # The patcher is constructed with a threshold value; pass any
        # valid float since we only need the markers list.
        patcher = _make_patcher(0.5)
        if patcher is None:
            pytest.skip("rejection_sampler.py not present in this env "
                        "— marker structure still in source but patcher "
                        "needs the file")

        markers_str = "\n".join(patcher.upstream_drift_markers)
        assert "use_block_verify" in markers_str, (
            "P82 must drift-detect PR #40819's `use_block_verify` flag "
            "so we know when the canonical SGLang block-verify rule "
            "lands in upstream rejection_sampler.py"
        )
        assert "verify_method" in markers_str
        assert "SpecVerifyMethod" in markers_str, (
            "PR #40819 adds SpecVerifyMethod to vllm/config/speculative.py "
            "— P82 must watch for that symbol too"
        )

    def test_p82_drift_markers_source_includes_40819_context(self):
        """The marker block must include a comment explaining why we
        watch these strings — operator + future contributor context."""
        import inspect
        from vllm._genesis.wiring.spec_decode import patch_82_sglang_acceptance_threshold
        src = inspect.getsource(patch_82_sglang_acceptance_threshold)
        # Comment explaining why these markers exist
        assert "#40819" in src or "PR #40819" in src
        assert "complementary" in src.lower() or "block-verify" in src


class TestP5AutoRetireProbe:
    """Genesis P5 should auto-skip when PR #39931 (JartX) is detected
    in the upstream vllm install. PR #39931's canonical symbols are
    `TQFullAttentionSpec` + `_get_full_attention_layer_indices` in
    `vllm.model_executor.layers.quantization.turboquant.config`.
    """

    def test_p5_skips_when_pr39931_symbols_present(self, monkeypatch):
        """Inject a fake upstream tq config module containing the
        PR #39931 symbols. P5 must SKIP and explain why."""
        # Build a fake module with the PR #39931 attestation
        fake_tq_cfg = types.ModuleType(
            "vllm.model_executor.layers.quantization.turboquant.config"
        )
        # Sentinel: the two symbols #39931 introduces
        fake_tq_cfg.TQFullAttentionSpec = type("TQFullAttentionSpec", (), {})
        fake_tq_cfg._get_full_attention_layer_indices = lambda *a, **kw: []
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.layers.quantization.turboquant.config",
            fake_tq_cfg,
        )

        # Also stub out the early-exit checks so we hit the probe
        from vllm._genesis.wiring.legacy import patch_5_page_size
        # GENESIS_DISABLE_P5 must NOT be set
        monkeypatch.delenv("GENESIS_DISABLE_P5", raising=False)

        status, reason = patch_5_page_size.apply()
        assert status == "skipped", (
            f"P5 must skip when PR #39931 symbols are present; got "
            f"{status} ({reason})"
        )
        assert "#39931" in reason, (
            "skip reason must reference PR #39931 so operators can find "
            "the institutional reasoning"
        )
        assert "TQFullAttentionSpec" in reason, (
            "skip reason must name the probed symbol"
        )

    def test_p5_proceeds_when_pr39931_symbols_absent(self, monkeypatch):
        """Without the PR #39931 symbols, P5 must NOT auto-skip on the
        retire probe (it may skip for other reasons, but not this
        one)."""
        # Build a fake module WITHOUT the PR #39931 symbols
        fake_tq_cfg = types.ModuleType(
            "vllm.model_executor.layers.quantization.turboquant.config"
        )
        # Note: deliberately DO NOT set TQFullAttentionSpec or the helper
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.layers.quantization.turboquant.config",
            fake_tq_cfg,
        )

        from vllm._genesis.wiring.legacy import patch_5_page_size
        monkeypatch.delenv("GENESIS_DISABLE_P5", raising=False)

        _status, reason = patch_5_page_size.apply()
        # Should NOT mention the auto-retire reason text
        assert "#39931" not in reason, (
            "When #39931 symbols absent, P5 must not return the "
            "auto-retire reason"
        )

    def test_p5_probe_failure_is_non_fatal(self, monkeypatch):
        """If the probe itself raises (e.g. import infrastructure
        broken), P5 must NOT fail — fall through to normal apply."""
        from vllm._genesis.wiring.legacy import patch_5_page_size

        # Force the probe import to raise
        import importlib
        original_im = importlib.import_module

        def _raising(name, *a, **kw):
            if name == "vllm.model_executor.layers.quantization.turboquant.config":
                raise RuntimeError("simulated infra failure")
            return original_im(name, *a, **kw)

        monkeypatch.setattr(importlib, "import_module", _raising)
        monkeypatch.delenv("GENESIS_DISABLE_P5", raising=False)

        # Should not raise, should reach a skip-or-applied status
        status, reason = patch_5_page_size.apply()
        assert status in ("applied", "skipped", "failed")
        # The probe failure must NOT show up as the reason — we should
        # have fallen through to normal apply path.
        assert "simulated infra failure" not in reason
