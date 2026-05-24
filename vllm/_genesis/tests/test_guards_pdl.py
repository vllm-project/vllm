# SPDX-License-Identifier: Apache-2.0
"""TDD for PDL-support guard + issue #40742 awareness.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations



class TestPdlSupportExpected:
    def test_false_on_non_nvidia(self, monkeypatch):
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        assert guards.pdl_support_expected() is False

    def test_false_on_ampere(self, monkeypatch):
        """SM 8.6 (A5000) should return False — PDL not supported."""
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: False)
        assert guards.pdl_support_expected() is False

    def test_false_on_ada(self, monkeypatch):
        """SM 8.9 (Ada) — PDL unsupported (need >=9.0)."""
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        # is_sm_at_least(9, 0) -> False on Ada
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: (major, minor) <= (8, 9))
        assert guards.pdl_support_expected() is False

    def test_true_on_hopper(self, monkeypatch):
        """SM 9.0 (H100) — PDL expected."""
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: (major, minor) <= (9, 0))
        assert guards.pdl_support_expected() is True

    def test_true_on_blackwell(self, monkeypatch):
        """SM 10.0 (Blackwell) — PDL expected."""
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)
        assert guards.pdl_support_expected() is True


class TestDetectPdlEnvMisconfig:
    def test_none_when_no_env_set(self, monkeypatch):
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "pdl_support_expected", lambda: False)
        monkeypatch.delenv("TRTLLM_ENABLE_PDL", raising=False)
        monkeypatch.delenv("TORCHINDUCTOR_ENABLE_PDL", raising=False)
        assert guards.detect_pdl_env_misconfig() == []

    def test_detects_trtllm_pdl_on_ampere(self, monkeypatch):
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "pdl_support_expected", lambda: False)
        monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
        monkeypatch.delenv("TORCHINDUCTOR_ENABLE_PDL", raising=False)
        bad = guards.detect_pdl_env_misconfig()
        assert "TRTLLM_ENABLE_PDL" in bad
        assert "TORCHINDUCTOR_ENABLE_PDL" not in bad

    def test_detects_both_when_set(self, monkeypatch):
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "pdl_support_expected", lambda: False)
        monkeypatch.setenv("TRTLLM_ENABLE_PDL", "true")
        monkeypatch.setenv("TORCHINDUCTOR_ENABLE_PDL", "yes")
        bad = guards.detect_pdl_env_misconfig()
        assert set(bad) == {"TRTLLM_ENABLE_PDL", "TORCHINDUCTOR_ENABLE_PDL"}

    def test_no_warning_on_hopper(self, monkeypatch):
        """On PDL-supporting hardware, env vars should NOT be flagged."""
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "pdl_support_expected", lambda: True)
        monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
        monkeypatch.setenv("TORCHINDUCTOR_ENABLE_PDL", "1")
        assert guards.detect_pdl_env_misconfig() == []

    def test_falsy_values_are_not_misconfig(self, monkeypatch):
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "pdl_support_expected", lambda: False)
        for falsy in ["0", "false", "no", "off", "", "  "]:
            monkeypatch.setenv("TRTLLM_ENABLE_PDL", falsy)
            monkeypatch.delenv("TORCHINDUCTOR_ENABLE_PDL", raising=False)
            assert guards.detect_pdl_env_misconfig() == [], (
                f"value {falsy!r} should not flag"
            )
