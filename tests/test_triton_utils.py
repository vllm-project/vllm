# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from importlib.util import find_spec
from unittest import mock

import pytest

from vllm.triton_utils.importing import TritonLanguagePlaceholder, TritonPlaceholder


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


def test_known_backends_skip_driver_is_active():
    if find_spec("triton") is None:
        pytest.skip("triton not installed")

    def _poison():
        drv = mock.MagicMock()
        drv.is_active.side_effect = AssertionError(
            "driver.is_active() must not be called for known backends"
        )
        b = mock.MagicMock()
        b.driver = drv
        return b

    fake_backends = {"nvidia": _poison(), "amd": _poison(), "cpu": _poison()}

    sys.modules.pop("vllm.triton_utils", None)
    sys.modules.pop("vllm.triton_utils.importing", None)

    with (
        mock.patch("triton.backends.backends", fake_backends),
        mock.patch("vllm.platforms.current_platform.is_cuda", return_value=True),
        mock.patch("vllm.platforms.current_platform.is_rocm", return_value=False),
        mock.patch("vllm.platforms.current_platform.is_cpu", return_value=False),
    ):
        import vllm.triton_utils.importing  # noqa: F401

    for b in fake_backends.values():
        b.driver.is_active.assert_not_called()


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
