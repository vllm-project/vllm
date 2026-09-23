# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.envs import disable_envs_cache
from vllm.models.deepseek_v41.nvidia import fewhead_prefill as fh
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability


@pytest.fixture(autouse=True)
def _reset_env_and_loader(monkeypatch: pytest.MonkeyPatch):
    disable_envs_cache()
    fh._FWD = None
    monkeypatch.delenv("VLLM_DSV41_FEWHEAD_PREFILL", raising=False)
    monkeypatch.delenv("VLLM_DSV41_FEWHEAD_MIN_SQ", raising=False)
    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(9, 0)),
    )
    yield
    fh._FWD = None
    disable_envs_cache()


def test_defaults_enable_fewhead_above_min_sq():
    assert fh.fewhead_prefill_enabled() is True
    assert fh.fewhead_min_sq() == 2048
    assert fh.should_use_fewhead_prefill(n_local_heads=8, padded_heads=64, s_q=8192)
    assert fh.should_use_fewhead_prefill(n_local_heads=8, padded_heads=64, s_q=2048)
    assert not fh.should_use_fewhead_prefill(n_local_heads=8, padded_heads=64, s_q=2047)
    assert not fh.should_use_fewhead_prefill(n_local_heads=8, padded_heads=64, s_q=512)


def test_disable_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_DSV41_FEWHEAD_PREFILL", "0")
    assert fh.fewhead_prefill_enabled() is False
    assert not fh.should_use_fewhead_prefill(n_local_heads=8, padded_heads=64, s_q=8192)


def test_min_sq_zero_always_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_DSV41_FEWHEAD_MIN_SQ", "0")
    assert fh.fewhead_min_sq() == 0
    assert fh.should_use_fewhead_prefill(n_local_heads=8, padded_heads=64, s_q=15)


def test_no_pad_skips_kernel():
    assert not fh.should_use_fewhead_prefill(
        n_local_heads=64, padded_heads=64, s_q=8192
    )


@pytest.mark.parametrize("n_local_heads", [0, 1, 8, 16, 17, 32, 64])
def test_only_heads_within_kernel_tile_are_dispatched(n_local_heads: int):
    # The Triton kernel has one 16-head tile; larger widths leave output unwritten.
    assert fh.should_use_fewhead_prefill(
        n_local_heads=n_local_heads, padded_heads=64, s_q=2048
    ) == (0 < n_local_heads <= 16)


@pytest.mark.parametrize("major,minor", [(8, 0), (8, 9), (10, 0), (12, 0)])
def test_non_sm90_keeps_flashmla(monkeypatch: pytest.MonkeyPatch, major, minor):
    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(major, minor)),
    )
    assert not fh.should_use_fewhead_prefill(n_local_heads=8, padded_heads=64, s_q=8192)


def test_invalid_min_sq_falls_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_DSV41_FEWHEAD_MIN_SQ", "not-int")
    assert fh.fewhead_min_sq() == 2048


def test_load_once(monkeypatch: pytest.MonkeyPatch):
    calls = []
    sentinel = object()

    def fake_import():
        calls.append(1)
        return sentinel

    monkeypatch.setattr(fh, "_import_kernel", fake_import)
    assert fh.load_fewhead_kernel() is sentinel
    assert fh.load_fewhead_kernel() is sentinel
    assert calls == [1]
