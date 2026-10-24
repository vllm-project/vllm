# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_ROCM_USE_AITER defaults on only where AITER can run."""

import vllm.envs as envs


def test_explicit_off_wins(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    assert envs._vllm_rocm_use_aiter_enabled() is False


def test_explicit_on(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    assert envs._vllm_rocm_use_aiter_enabled() is True


def test_unset_without_aiter_stays_off(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_USE_AITER", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert envs._vllm_rocm_use_aiter_enabled() is False


def test_unset_on_cdna3_with_aiter(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_USE_AITER", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

    import vllm.platforms.rocm as rocm

    monkeypatch.setattr(rocm, "get_cdna_version", lambda: 3)
    assert envs._vllm_rocm_use_aiter_enabled() is True


def test_unset_on_mi250_stays_off(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_USE_AITER", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

    import vllm.platforms.rocm as rocm

    monkeypatch.setattr(rocm, "get_cdna_version", lambda: 2)
    assert envs._vllm_rocm_use_aiter_enabled() is False
