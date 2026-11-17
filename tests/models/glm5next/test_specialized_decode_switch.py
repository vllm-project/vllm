# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The vendor switch in `common/model.py`: the platform picks the specialized decode's
config, executor and model state, and off is a no-op on every platform. Off ROCm the
AMD runtime is never imported: `--monokernel` is refused, and
`decode_backend=latency_specialized` goes to the NVIDIA executor. On ROCm the NVIDIA
executor is never imported."""

import importlib
import sys
from types import SimpleNamespace

import pytest

import vllm.models.glm5next.common.model as glm_model

AMD_RUNTIME = "vllm.models.glm5next.amd.specialized_decode.runtime"
NVIDIA_RUNTIME = "vllm.models.glm5next.nvidia.specialized_decode.runtime"
VENDOR = {True: "amd", False: "nvidia"}


def _config(monokernel=False, backend=None):
    return SimpleNamespace(
        kernel_config=SimpleNamespace(monokernel=monokernel),
        additional_config={} if backend is None else {"decode_backend": backend},
    )


@pytest.fixture
def load(monkeypatch):
    """`common/model.py` imported on ROCm (True) or off it (False), with neither
    vendor's runtime imported yet."""

    def _load(rocm):
        monkeypatch.setattr(glm_model.current_platform, "is_rocm", lambda: rocm)
        for name in (AMD_RUNTIME, NVIDIA_RUNTIME):
            monkeypatch.delitem(sys.modules, name, raising=False)
        return importlib.reload(glm_model)

    yield _load
    monkeypatch.undo()
    importlib.reload(glm_model)


@pytest.mark.parametrize("rocm", [True, False])
@pytest.mark.parametrize("backend", [None, "auto"])
def test_off_is_none_and_imports_no_other_vendor(load, rocm, backend):
    model = load(rocm)
    assert model.specialized_decode_config(_config(backend=backend)) is None
    assert (NVIDIA_RUNTIME if rocm else AMD_RUNTIME) not in sys.modules


def test_monokernel_off_rocm_is_refused(load):
    model = load(False)
    with pytest.raises(ValueError, match="only on ROCm"):
        model.specialized_decode_config(_config(monokernel=True))
    assert AMD_RUNTIME not in sys.modules


def test_decode_backend_off_rocm_goes_to_nvidia(load, monkeypatch):
    model = load(False)
    monkeypatch.setattr(model, "_nvidia_specialized_decode_config", lambda _: "nv")
    assert model.specialized_decode_config(_config(backend="latency_specialized")) == (
        "nv"
    )
    assert AMD_RUNTIME not in sys.modules


@pytest.mark.parametrize("rocm", [True, False])
def test_unknown_decode_backend_raises(load, rocm):
    model = load(rocm)
    with pytest.raises(ValueError, match="decode_backend"):
        model.specialized_decode_config(_config(backend="fast"))


def test_rocm_never_imports_the_nvidia_executor(load):
    from vllm.models.glm5next.amd.specialized_decode.runtime import spec

    model = load(True)
    assert model.specialized_decode_config is spec.specialized_decode_config
    assert NVIDIA_RUNTIME not in sys.modules


@pytest.mark.parametrize("rocm", [True, False])
def test_model_state_follows_the_platform(load, rocm):
    """Each vendor's model state builds its own decode; the other vendor's would
    leave it unbuilt and serve stock decode silently."""
    model = load(rocm)
    cls = model.Glm5NextForCausalLM.get_model_state_cls(SimpleNamespace())
    assert cls.__module__ == f"vllm.models.glm5next.{VENDOR[rocm]}.model_state"
    assert cls.__name__ == "Glm5NextModelState"
    decode = model._specialized_decode_cls()
    assert decode.__module__.startswith(f"vllm.models.glm5next.{VENDOR[rocm]}.")
    assert decode.__name__ == "SpecializedDecode"


def test_amd_executor_has_the_nvidia_interface():
    """The hooks in `common/model.py` call both vendors' `SpecializedDecode` the
    same way."""
    from vllm.models.glm5next.amd.specialized_decode.runtime.decode import (
        SpecializedDecode,
    )

    for name in ("plan", "run", "prepare_kv", "build_logits", "compute_logits"):
        assert callable(getattr(SpecializedDecode, name)), name
