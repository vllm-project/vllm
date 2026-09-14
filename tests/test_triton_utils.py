# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types
from unittest import mock

import pytest

from vllm.triton_utils import importing as triton_importing
from vllm.triton_utils.importing import TritonLanguagePlaceholder, TritonPlaceholder


def _has_triton_for_backends(cpu_build: bool = False, **drivers: bool) -> bool:
    """Re-evaluate ``HAS_TRITON`` against a synthetic ``triton.backends`` map.

    ``drivers`` maps backend name to whether its driver reports itself active.
    """
    backends = {}
    for name, is_active in drivers.items():
        driver = mock.Mock()
        driver.is_active.return_value = is_active
        backend = mock.Mock()
        backend.driver = driver
        backends[name] = backend

    triton_mod = types.ModuleType("triton")
    triton_mod.__spec__ = importlib.machinery.ModuleSpec("triton", None)
    backends_mod = types.ModuleType("triton.backends")
    backends_mod.__dict__["backends"] = backends
    triton_mod.__dict__["backends"] = backends_mod

    patched_modules = {"triton": triton_mod, "triton.backends": backends_mod}

    def vllm_version(package: str) -> str:
        assert package == "vllm"
        return "0.0.0+cpu" if cpu_build else "0.0.0"

    try:
        with (
            mock.patch.dict(sys.modules, patched_modules),
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch(
                "importlib.metadata.version",
                side_effect=vllm_version,
            ),
        ):
            return importlib.reload(triton_importing).HAS_TRITON
    finally:
        # Restore the module state derived from the real environment.
        importlib.reload(triton_importing)


def test_triton_placeholder_is_module():
    triton = TritonPlaceholder()
    assert isinstance(triton, types.ModuleType)
    assert triton.__name__ == "triton"


def test_triton_language_placeholder_is_module():
    triton_language = TritonLanguagePlaceholder()
    assert isinstance(triton_language, types.ModuleType)
    assert triton_language.__name__ == "triton.language"


def test_triton_placeholder_decorators():
    triton = TritonPlaceholder()

    @triton.jit
    def foo(x):
        return x

    @triton.autotune
    def bar(x):
        return x

    @triton.heuristics
    def baz(x):
        return x

    assert foo(1) == 1
    assert bar(2) == 2
    assert baz(3) == 3


def test_triton_placeholder_decorators_with_args():
    triton = TritonPlaceholder()

    @triton.jit(debug=True)
    def foo(x):
        return x

    @triton.autotune(configs=[], key="x")
    def bar(x):
        return x

    @triton.heuristics({"BLOCK_SIZE": lambda args: 128 if args["x"] > 1024 else 64})
    def baz(x):
        return x

    assert foo(1) == 1
    assert bar(2) == 2
    assert baz(3) == 3


def test_triton_placeholder_language():
    lang = TritonLanguagePlaceholder()
    assert isinstance(lang, types.ModuleType)
    assert lang.__name__ == "triton.language"
    assert lang.constexpr is None
    assert lang.dtype is None
    assert lang.int64 is None
    assert lang.int32 is None
    assert lang.tensor is None


def test_triton_placeholder_language_from_parent():
    triton = TritonPlaceholder()
    lang = triton.language
    assert isinstance(lang, TritonLanguagePlaceholder)


@pytest.mark.parametrize(
    ("cpu_build", "drivers", "expected"),
    [
        pytest.param(False, {"amd": True, "cpu": True}, True, id="gpu-plus-cpu"),
        pytest.param(False, {"amd": True}, True, id="single-gpu"),
        pytest.param(
            False,
            {"amd": True, "nvidia": True},
            False,
            id="multiple-gpus",
        ),
        pytest.param(False, {"cpu": True}, False, id="non-cpu-build"),
        pytest.param(True, {"cpu": True}, True, id="cpu-build"),
    ],
)
def test_triton_backend_selection(
    cpu_build: bool, drivers: dict[str, bool], expected: bool
) -> None:
    assert _has_triton_for_backends(cpu_build=cpu_build, **drivers) is expected


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        pytest.param("cpu", True, id="cpu"),
        pytest.param("cuda", False, id="cuda"),
    ],
)
def test_active_triton_cpu_backend(backend: str, expected: bool) -> None:
    target = types.SimpleNamespace(backend=backend)
    driver = types.SimpleNamespace(
        active=types.SimpleNamespace(get_current_target=lambda: target)
    )
    runtime = types.ModuleType("triton.runtime")
    runtime.__dict__["driver"] = driver

    with (
        mock.patch.object(triton_importing, "HAS_TRITON", True),
        mock.patch.dict(sys.modules, {"triton.runtime": runtime}),
    ):
        assert triton_importing.has_active_triton_cpu_backend() is expected


def test_unavailable_triton_cpu_backend_fails_closed():
    with mock.patch.object(triton_importing, "HAS_TRITON", False):
        assert not triton_importing.has_active_triton_cpu_backend()

    driver = types.SimpleNamespace(
        active=types.SimpleNamespace(
            get_current_target=mock.Mock(side_effect=RuntimeError("no active driver"))
        )
    )
    runtime = types.ModuleType("triton.runtime")
    runtime.__dict__["driver"] = driver
    with (
        mock.patch.object(triton_importing, "HAS_TRITON", True),
        mock.patch.dict(sys.modules, {"triton.runtime": runtime}),
    ):
        assert not triton_importing.has_active_triton_cpu_backend()


def test_no_triton_fallback():
    # clear existing triton modules
    sys.modules.pop("triton", None)
    sys.modules.pop("triton.language", None)
    sys.modules.pop("vllm.triton_utils", None)
    sys.modules.pop("vllm.triton_utils.importing", None)

    # mock triton not being installed
    with mock.patch.dict(sys.modules, {"triton": None}):
        from vllm.triton_utils import HAS_TRITON, tl, triton

        assert HAS_TRITON is False
        assert triton.__class__.__name__ == "TritonPlaceholder"
        assert triton.language.__class__.__name__ == "TritonLanguagePlaceholder"
        assert tl.__class__.__name__ == "TritonLanguagePlaceholder"
