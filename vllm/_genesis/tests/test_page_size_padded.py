# SPDX-License-Identifier: Apache-2.0
"""TDD tests for vllm._genesis.kernels.page_size_padded (Patch 5b helpers).

P5b is scaffolding for the future pad-smaller-to-max KV unification
strategy. These tests verify the helpers behave correctly BEFORE we wire
them into the TurboQuant kernel.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest


class TestIsP5bEnabled:
    def test_disabled_by_default(self, monkeypatch):
        from vllm._genesis.kernels.page_size_padded import is_p5b_enabled
        monkeypatch.delenv("GENESIS_ENABLE_P5B", raising=False)
        assert is_p5b_enabled() is False

    @pytest.mark.parametrize("v", ["1", "true", "yes", "on", "TRUE", "Yes"])
    def test_enabled_values(self, monkeypatch, v):
        from vllm._genesis.kernels.page_size_padded import is_p5b_enabled
        monkeypatch.setenv("GENESIS_ENABLE_P5B", v)
        assert is_p5b_enabled() is True

    @pytest.mark.parametrize("v", ["0", "false", "no", "", "maybe", "off"])
    def test_disabled_values(self, monkeypatch, v):
        from vllm._genesis.kernels.page_size_padded import is_p5b_enabled
        monkeypatch.setenv("GENESIS_ENABLE_P5B", v)
        assert is_p5b_enabled() is False


class _FakeSpec:
    """Minimal layer-spec stand-in."""
    def __init__(self, page_size_bytes, **extra):
        self.page_size_bytes = page_size_bytes
        for k, v in extra.items():
            setattr(self, k, v)


class TestComputeRealPageSizeBytes:
    def test_uses_real_attr_when_present(self):
        from vllm._genesis.kernels.page_size_padded import (
            compute_real_page_size_bytes,
        )
        s = _FakeSpec(page_size_bytes=1073152, real_page_size_bytes=813248)
        assert compute_real_page_size_bytes(s) == 813248

    def test_uses_natural_alias(self):
        from vllm._genesis.kernels.page_size_padded import (
            compute_real_page_size_bytes,
        )
        s = _FakeSpec(
            page_size_bytes=1073152, page_size_bytes_natural=900000,
        )
        assert compute_real_page_size_bytes(s) == 900000

    def test_falls_back_to_page_size_bytes(self):
        from vllm._genesis.kernels.page_size_padded import (
            compute_real_page_size_bytes,
        )
        s = _FakeSpec(page_size_bytes=42)
        assert compute_real_page_size_bytes(s) == 42

    def test_real_takes_precedence_over_natural(self):
        from vllm._genesis.kernels.page_size_padded import (
            compute_real_page_size_bytes,
        )
        s = _FakeSpec(
            page_size_bytes=1073152,
            real_page_size_bytes=813248,
            page_size_bytes_natural=999999,
        )
        assert compute_real_page_size_bytes(s) == 813248


class TestClampToRealShape:
    def test_unchanged_when_no_padding(self):
        from vllm._genesis.kernels.page_size_padded import clamp_to_real_shape
        s = _FakeSpec(page_size_bytes=1024)  # no real_* attr → natural==stated
        assert clamp_to_real_shape((4, 16, 1024), s) == (4, 16, 1024)

    def test_scales_last_dim_down(self):
        from vllm._genesis.kernels.page_size_padded import clamp_to_real_shape
        # Padded 1073152, natural 813248 → ratio 813248/1073152
        s = _FakeSpec(
            page_size_bytes=1073152, real_page_size_bytes=813248,
        )
        scaled = clamp_to_real_shape((524, 16, 1, 512), s)
        # Last dim 512 × 813248/1073152 → 387.9... → int() = 387
        assert scaled[:-1] == (524, 16, 1)
        assert scaled[-1] == int(512 * 813248 / 1073152)

    def test_zero_stated_returns_unchanged(self):
        from vllm._genesis.kernels.page_size_padded import clamp_to_real_shape
        s = _FakeSpec(page_size_bytes=0, real_page_size_bytes=42)
        assert clamp_to_real_shape((4, 16, 1024), s) == (4, 16, 1024)


class TestP5bEnvGatedRegistration:
    """P5b is registered in v7.4 but ENV-GATED.

    Until VM 100 GSM8K + long-context regression bench closes, the
    registered entry must SKIP by default (no env var set). This
    protects operators from accidentally changing KV allocator sizing
    semantics in prod.
    """

    def test_is_registered(self):
        """v7.4 change: P5b IS registered (was deliberately absent in
        v7.3). Guards against accidental removal on refactors."""
        from vllm._genesis.patches import apply_all
        names = [n for n, _ in apply_all.PATCH_REGISTRY]
        p5b_names = [n for n in names if n.lower().startswith("p5b")]
        assert len(p5b_names) == 1, (
            f"Expected exactly one P5b registration; found: {p5b_names}"
        )
        assert "opt-in" in p5b_names[0].lower(), (
            "P5b registration name must make the env-gate explicit "
            "(operators should see 'opt-in' in apply_all summary)"
        )

    def test_apply_skips_without_env(self, monkeypatch):
        """Default-OFF behaviour: even with NVIDIA+SM8.0, skip until
        GENESIS_ENABLE_P5B env is set."""
        monkeypatch.delenv("GENESIS_ENABLE_P5B", raising=False)
        from vllm._genesis.wiring.legacy import patch_5b_page_size_pad_smaller as p5b
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            guards, "is_sm_at_least", lambda major, minor=0: True,
        )
        status, reason = p5b.apply()
        assert status == "skipped"
        assert "opt-in" in reason.lower()
        assert "GENESIS_ENABLE_P5B" in reason
