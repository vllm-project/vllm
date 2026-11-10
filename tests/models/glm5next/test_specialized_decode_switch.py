# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared specialized-decode switch: off is a no-op on every platform. Off ROCm
the AMD runtime is never imported: `--monokernel` is refused, and
`decode_backend=latency_specialized` is left to the NVIDIA executor."""

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


def test_monokernel_off_rocm_is_refused(not_rocm):
    with pytest.raises(ValueError, match="only on ROCm"):
        glm_model._specialized_decode_config(_config(monokernel=True))
    assert AMD_RUNTIME not in sys.modules


def test_decode_backend_off_rocm_is_left_to_nvidia(not_rocm):
    cfg = _config(backend="latency_specialized")
    assert glm_model._specialized_decode_config(cfg) is None
    assert AMD_RUNTIME not in sys.modules


def test_unknown_decode_backend_raises(not_rocm):
    with pytest.raises(ValueError, match="decode_backend must be"):
        glm_model._specialized_decode_config(_config(backend="fast"))
