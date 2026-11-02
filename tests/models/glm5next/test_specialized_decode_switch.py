# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared specialized-decode switch: off is a no-op on every platform, and a
request off ROCm is refused without importing the AMD runtime."""

import sys
from types import SimpleNamespace

import pytest

import vllm.models.glm5next.common.model as glm_model

AMD_RUNTIME = "vllm.models.glm5next.amd.specialized_decode.runtime"


def _config(monokernel=False, backend=None):
    return SimpleNamespace(
        kernel_config=SimpleNamespace(monokernel=monokernel),
        additional_config={} if backend is None else {"decode_backend": backend},
    )


@pytest.fixture
def not_rocm(monkeypatch):
    monkeypatch.setattr(glm_model.current_platform, "is_rocm", lambda: False)
    monkeypatch.delitem(sys.modules, AMD_RUNTIME, raising=False)


@pytest.mark.parametrize("backend", [None, "auto"])
def test_off_is_none_and_imports_no_amd_code(not_rocm, backend):
    assert glm_model._specialized_decode_config(_config(backend=backend)) is None
    assert AMD_RUNTIME not in sys.modules


@pytest.mark.parametrize(
    "cfg", [_config(monokernel=True), _config(backend="latency_specialized")]
)
def test_request_off_rocm_is_refused(not_rocm, cfg):
    with pytest.raises(ValueError, match="only on ROCm"):
        glm_model._specialized_decode_config(cfg)
    assert AMD_RUNTIME not in sys.modules


def test_unknown_decode_backend_raises(not_rocm):
    with pytest.raises(ValueError, match="decode_backend must be"):
        glm_model._specialized_decode_config(_config(backend="fast"))
