# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static (no-GPU) routing check for Gemma 4 NVFP4 KV -> FLASHINFER.

On consumer Blackwell (CC 12.x) the FlashInfer FA2 asymmetric paged
kernel serves Gemma 4 heterogeneous-head full-attention layers
(head_dim_qk=512, head_dim_vo=256) directly, so Gemma4Config must route
NVFP4-KV configs to FLASHINFER instead of the TRITON_ATTN fallback.

Runs under a mocked platform/capability; no CUDA required. Mocking
pattern adapted from the campaign triton-retirement selection test.
"""

from types import SimpleNamespace

import pytest

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum

ALL_KNOBS = ("VLLM_NVFP4_KV_VOSPLIT",)

CC12_0 = DeviceCapability(12, 0)
CC12_1 = DeviceCapability(12, 1)
CC9_0 = DeviceCapability(9, 0)


@pytest.fixture(autouse=True)
def _clear_knobs(monkeypatch):
    for name in ALL_KNOBS:
        monkeypatch.delenv(name, raising=False)
    yield


def _mock_vllm_config(*, backend=None, cache_dtype="nvfp4",
                      head_dim=256, global_head_dim=512):
    return SimpleNamespace(
        attention_config=SimpleNamespace(backend=backend,
                                         flash_attn_version=None),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype,
                                     kv_cache_dtype_skip_layers=None),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                head_dim=head_dim, global_head_dim=global_head_dim,
                layer_types=["sliding_attention", "sliding_attention",
                             "sliding_attention", "sliding_attention",
                             "sliding_attention", "full_attention"],
            )
        ),
    )


class _FakePlatform:
    """Delegates everything to the real current_platform except the
    compute-capability accessors, which are pinned to ``capability``."""

    def __init__(self, capability):
        self._cap = capability
        from vllm.platforms import current_platform as real
        self._real = real

    def is_cuda(self):
        return True

    def get_device_capability(self, device_id=0):
        return self._cap

    def is_device_capability_family(self, cap, device_id=0):
        return (self._cap.to_int() // 10) == (cap // 10)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def fake_cc(monkeypatch):
    def _set(capability):
        import vllm.platforms as platforms_mod
        import vllm.v1.attention.backends.fa_utils as fa_utils_mod
        fake = _FakePlatform(capability)
        monkeypatch.setattr(platforms_mod, "current_platform", fake,
                            raising=False)
        monkeypatch.setattr(fa_utils_mod, "is_fa_version_supported",
                            lambda v: False)
        return fake
    return _set


def _gemma4_route(vllm_config):
    from vllm.model_executor.models.config import Gemma4Config
    Gemma4Config.verify_and_update_config(vllm_config)
    return vllm_config.attention_config.backend


@pytest.mark.parametrize("capability", [CC12_0, CC12_1])
def test_nvfp4_cc12_routes_to_flashinfer(fake_cc, capability):
    fake_cc(capability)
    cfg = _mock_vllm_config()
    backend = _gemma4_route(cfg)
    assert backend == AttentionBackendEnum.FLASHINFER, backend


def test_nvfp4_cc12_disabled_knob_falls_back_to_triton(fake_cc, monkeypatch):
    monkeypatch.setenv("VLLM_NVFP4_KV_VOSPLIT", "0")
    fake_cc(CC12_0)
    cfg = _mock_vllm_config()
    assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN


def test_nvfp4_hopper_does_not_route(fake_cc):
    fake_cc(CC9_0)
    cfg = _mock_vllm_config()
    assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN


def test_nvfp4_explicit_user_backend_wins(fake_cc):
    fake_cc(CC12_0)
    cfg = _mock_vllm_config(backend=AttentionBackendEnum.TRITON_ATTN)
    assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN


def test_bf16_kv_keeps_triton_fallback(fake_cc):
    fake_cc(CC12_0)
    cfg = _mock_vllm_config(cache_dtype="auto")
    assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN
