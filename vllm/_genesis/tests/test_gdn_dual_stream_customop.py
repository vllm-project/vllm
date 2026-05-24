# SPDX-License-Identifier: Apache-2.0
"""TDD tests for P7b — GDN dual-stream via `torch.library.custom_op`.

Covers:
- Module imports cleanly on CPU-only
- `is_p7b_enabled` respects env (default OFF, honours "1"/"true"/"yes")
- `should_apply` respects env + platform
- Public `dual_linear_parallel` fallback path on CPU returns correct
  shapes/dtypes (serial F.linear)
- Custom-op registration is deferred (not fired at module import)
- Fake (meta) impl returns correct shape for 2-D and 3-D inputs
- Wiring surface: `apply` / `is_applied` / `revert` / `should_apply`
- Wiring skips when env off
- Wiring skips when target missing
- Upstream drift markers present in `UPSTREAM_DRIFT_MARKERS`

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    """Default: P7B disabled. Individual tests flip on via setenv."""
    monkeypatch.delenv("GENESIS_ENABLE_P7B", raising=False)
    yield


class TestP7bModule:
    def test_import_on_cpu(self):
        """Module imports on CPU-only without triggering custom-op
        registration (which needs torch.library + is lazy)."""
        from vllm._genesis.kernels import gdn_dual_stream_customop as m
        assert hasattr(m, "should_apply")
        assert hasattr(m, "is_p7b_enabled")
        assert hasattr(m, "dual_linear_parallel")
        # Custom op registration flag should be False until first real use
        assert m._op_registered is False

    def test_is_p7b_enabled_default_off(self):
        from vllm._genesis.kernels import gdn_dual_stream_customop as m
        assert m.is_p7b_enabled() is False

    def test_is_p7b_enabled_truthy_values(self, monkeypatch):
        from vllm._genesis.kernels import gdn_dual_stream_customop as m
        for val in ("1", "true", "TRUE", "yes", "Yes", "on", "ON"):
            monkeypatch.setenv("GENESIS_ENABLE_P7B", val)
            assert m.is_p7b_enabled() is True, f"Should accept {val!r}"

    def test_is_p7b_enabled_falsy_values(self, monkeypatch):
        from vllm._genesis.kernels import gdn_dual_stream_customop as m
        for val in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("GENESIS_ENABLE_P7B", val)
            assert m.is_p7b_enabled() is False, f"Should reject {val!r}"


class TestP7bShouldApply:
    def test_disabled_without_env(self, monkeypatch):
        from vllm._genesis.kernels import gdn_dual_stream_customop as m
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            guards, "is_sm_at_least", lambda major, minor=0: True,
        )
        assert m.should_apply() is False

    def test_disabled_non_nvidia(self, monkeypatch):
        from vllm._genesis.kernels import gdn_dual_stream_customop as m
        from vllm._genesis import guards
        monkeypatch.setenv("GENESIS_ENABLE_P7B", "1")
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        assert m.should_apply() is False


class TestP7bFallback:
    def test_cpu_fallback_returns_correct_shape_2d(self):
        """On CPU, dual_linear_parallel falls through to serial F.linear."""
        from vllm._genesis.kernels.gdn_dual_stream_customop import (
            dual_linear_parallel,
        )
        hidden = torch.randn(8, 16)
        w1 = torch.randn(32, 16)
        w2 = torch.randn(24, 16)
        out1, out2 = dual_linear_parallel(hidden, w1, None, w2, None)
        assert out1.shape == (8, 32)
        assert out2.shape == (8, 24)

    def test_cpu_fallback_matches_serial(self):
        """Fallback path must be numerically identical to two F.linear."""
        import torch.nn.functional as F
        from vllm._genesis.kernels.gdn_dual_stream_customop import (
            dual_linear_parallel,
        )
        torch.manual_seed(0)
        hidden = torch.randn(4, 16)
        w1 = torch.randn(8, 16)
        w2 = torch.randn(6, 16)
        b1 = torch.randn(8)
        b2 = torch.randn(6)
        out1, out2 = dual_linear_parallel(hidden, w1, b1, w2, b2)
        ref1 = F.linear(hidden, w1, b1)
        ref2 = F.linear(hidden, w2, b2)
        assert torch.equal(out1, ref1)
        assert torch.equal(out2, ref2)

    def test_cpu_fallback_3d_input(self):
        """Shape-polymorphic: (B, N, K) input should work."""
        from vllm._genesis.kernels.gdn_dual_stream_customop import (
            dual_linear_parallel,
        )
        hidden = torch.randn(2, 4, 16)
        w1 = torch.randn(32, 16)
        w2 = torch.randn(24, 16)
        out1, out2 = dual_linear_parallel(hidden, w1, None, w2, None)
        assert out1.shape == (2, 4, 32)
        assert out2.shape == (2, 4, 24)


class TestP7bWiringSurface:
    def test_public_surface(self):
        from vllm._genesis.wiring.legacy import patch_7b_gdn_dual_stream_customop as p7b
        assert callable(p7b.apply)
        assert callable(p7b.is_applied)
        assert callable(p7b.revert)
        assert callable(p7b.should_apply)

    def test_apply_skips_without_env(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_7b_gdn_dual_stream_customop as p7b
        monkeypatch.setattr(p7b, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            p7b, "is_sm_at_least", lambda major, minor=0: True,
        )
        monkeypatch.delenv("GENESIS_ENABLE_P7B", raising=False)
        status, reason = p7b.apply()
        assert status == "skipped"
        assert "opt-in" in reason.lower() or "GENESIS_ENABLE_P7B" in reason

    def test_apply_skips_on_non_nvidia(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_7b_gdn_dual_stream_customop as p7b
        monkeypatch.setenv("GENESIS_ENABLE_P7B", "1")
        monkeypatch.setattr(p7b, "is_nvidia_cuda", lambda: False)
        status, reason = p7b.apply()
        assert status == "skipped"
        assert "NVIDIA" in reason or "platform" in reason.lower()

    def test_revert_always_false(self):
        """P7b is a text-patch — no runtime revert (need compose down)."""
        from vllm._genesis.wiring.legacy import patch_7b_gdn_dual_stream_customop as p7b
        assert p7b.revert() is False

    def test_upstream_drift_markers(self):
        from vllm._genesis.wiring.legacy import patch_7b_gdn_dual_stream_customop as p7b
        assert "dual_linear_parallel" in p7b.UPSTREAM_DRIFT_MARKERS
        assert any("genesis::" in m for m in p7b.UPSTREAM_DRIFT_MARKERS)


class TestP7bVsP7Coexistence:
    def test_p7_and_p7b_marker_strings_differ(self):
        """P7 and P7b must use different markers so text-patch detection
        doesn't false-match."""
        from vllm._genesis.wiring.legacy import patch_7_gdn_dual_stream as p7
        from vllm._genesis.wiring.legacy import patch_7b_gdn_dual_stream_customop as p7b
        assert p7.GENESIS_P7_MARKER != p7b.GENESIS_P7B_MARKER
        assert "P7 " in p7.GENESIS_P7_MARKER
        assert "P7b " in p7b.GENESIS_P7B_MARKER
