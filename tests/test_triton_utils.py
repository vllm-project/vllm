# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from unittest import mock

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


def test_configure_triton_ptxas_respects_existing_env():
    """Test that _configure_triton_ptxas_for_new_gpus doesn't override
    user-set TRITON_PTXAS_PATH."""
    import os

    from vllm.triton_utils.importing import _configure_triton_ptxas_for_new_gpus

    # Save original value
    original = os.environ.get("TRITON_PTXAS_PATH")

    try:
        # Set a custom path
        os.environ["TRITON_PTXAS_PATH"] = "/custom/path/to/ptxas"

        # Call the function - it should not override
        _configure_triton_ptxas_for_new_gpus()

        # Verify it wasn't changed
        assert os.environ.get("TRITON_PTXAS_PATH") == "/custom/path/to/ptxas"
    finally:
        # Restore original value
        if original is None:
            os.environ.pop("TRITON_PTXAS_PATH", None)
        else:
            os.environ["TRITON_PTXAS_PATH"] = original


def test_configure_triton_ptxas_detects_new_gpu():
    """Test that _configure_triton_ptxas_for_new_gpus sets TRITON_PTXAS_PATH
    for GPUs with compute capability >= 11.0 using Triton's native detection."""
    import os
    import tempfile

    from vllm.triton_utils.importing import _configure_triton_ptxas_for_new_gpus

    # Save original values
    original_ptxas = os.environ.get("TRITON_PTXAS_PATH")
    original_cuda_home = os.environ.get("CUDA_HOME")

    try:
        # Clear TRITON_PTXAS_PATH
        os.environ.pop("TRITON_PTXAS_PATH", None)

        # Create a mock ptxas executable
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ptxas = os.path.join(tmpdir, "bin", "ptxas")
            os.makedirs(os.path.dirname(mock_ptxas))
            with open(mock_ptxas, "w") as f:
                f.write("#!/bin/sh\necho 'ptxas mock'\n")
            os.chmod(mock_ptxas, 0o755)

            # Set CUDA_HOME to our temp dir
            os.environ["CUDA_HOME"] = tmpdir

            # Mock Triton's native GPU detection to return arch=110 (Thor, CC 11.0)
            mock_target = mock.MagicMock()
            mock_target.arch = 110  # CC 11.0

            mock_driver_instance = mock.MagicMock()
            mock_driver_instance.get_current_target.return_value = mock_target

            mock_driver_class = mock.MagicMock(return_value=mock_driver_instance)
            mock_driver_class.is_active.return_value = True

            mock_nvidia_backend = mock.MagicMock()
            mock_nvidia_backend.driver = mock_driver_class

            mock_backends = mock.MagicMock()
            mock_backends.get.return_value = mock_nvidia_backend

            with mock.patch("vllm.triton_utils.importing.backends", mock_backends):
                _configure_triton_ptxas_for_new_gpus()

            # Verify TRITON_PTXAS_PATH was set
            assert os.environ.get("TRITON_PTXAS_PATH") == mock_ptxas

    finally:
        # Restore original values
        if original_ptxas is None:
            os.environ.pop("TRITON_PTXAS_PATH", None)
        else:
            os.environ["TRITON_PTXAS_PATH"] = original_ptxas
        if original_cuda_home is None:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = original_cuda_home


def test_configure_triton_ptxas_skips_older_gpus():
    """Test that _configure_triton_ptxas_for_new_gpus does not set
    TRITON_PTXAS_PATH for GPUs with compute capability < 11.0."""
    import os
    import tempfile

    from vllm.triton_utils.importing import _configure_triton_ptxas_for_new_gpus

    # Save original values
    original_ptxas = os.environ.get("TRITON_PTXAS_PATH")
    original_cuda_home = os.environ.get("CUDA_HOME")

    try:
        # Clear TRITON_PTXAS_PATH
        os.environ.pop("TRITON_PTXAS_PATH", None)

        # Create a mock ptxas executable
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ptxas = os.path.join(tmpdir, "bin", "ptxas")
            os.makedirs(os.path.dirname(mock_ptxas))
            with open(mock_ptxas, "w") as f:
                f.write("#!/bin/sh\necho 'ptxas mock'\n")
            os.chmod(mock_ptxas, 0o755)

            # Set CUDA_HOME to our temp dir
            os.environ["CUDA_HOME"] = tmpdir

            # Mock Triton's native GPU detection to return arch=90 (Hopper, CC 9.0)
            mock_target = mock.MagicMock()
            mock_target.arch = 90  # CC 9.0

            mock_driver_instance = mock.MagicMock()
            mock_driver_instance.get_current_target.return_value = mock_target

            mock_driver_class = mock.MagicMock(return_value=mock_driver_instance)
            mock_driver_class.is_active.return_value = True

            mock_nvidia_backend = mock.MagicMock()
            mock_nvidia_backend.driver = mock_driver_class

            mock_backends = mock.MagicMock()
            mock_backends.get.return_value = mock_nvidia_backend

            with mock.patch("vllm.triton_utils.importing.backends", mock_backends):
                _configure_triton_ptxas_for_new_gpus()

            # Verify TRITON_PTXAS_PATH was NOT set
            assert os.environ.get("TRITON_PTXAS_PATH") is None

    finally:
        # Restore original values
        if original_ptxas is None:
            os.environ.pop("TRITON_PTXAS_PATH", None)
        else:
            os.environ["TRITON_PTXAS_PATH"] = original_ptxas
        if original_cuda_home is None:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = original_cuda_home


def test_configure_triton_ptxas_detects_gb10():
    """Test that _configure_triton_ptxas_for_new_gpus sets TRITON_PTXAS_PATH
    for NVIDIA GB10 (DGX Spark) with compute capability 12.1 (arch=121)."""
    import os
    import tempfile

    from vllm.triton_utils.importing import _configure_triton_ptxas_for_new_gpus

    # Save original values
    original_ptxas = os.environ.get("TRITON_PTXAS_PATH")
    original_cuda_home = os.environ.get("CUDA_HOME")

    try:
        # Clear TRITON_PTXAS_PATH
        os.environ.pop("TRITON_PTXAS_PATH", None)

        # Create a mock ptxas executable
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ptxas = os.path.join(tmpdir, "bin", "ptxas")
            os.makedirs(os.path.dirname(mock_ptxas))
            with open(mock_ptxas, "w") as f:
                f.write("#!/bin/sh\necho 'ptxas mock'\n")
            os.chmod(mock_ptxas, 0o755)

            # Set CUDA_HOME to our temp dir
            os.environ["CUDA_HOME"] = tmpdir

            # Mock Triton's native GPU detection to return arch=121 (GB10, CC 12.1)
            mock_target = mock.MagicMock()
            mock_target.arch = 121  # CC 12.1 (GB10 / DGX Spark)

            mock_driver_instance = mock.MagicMock()
            mock_driver_instance.get_current_target.return_value = mock_target

            mock_driver_class = mock.MagicMock(return_value=mock_driver_instance)
            mock_driver_class.is_active.return_value = True

            mock_nvidia_backend = mock.MagicMock()
            mock_nvidia_backend.driver = mock_driver_class

            mock_backends = mock.MagicMock()
            mock_backends.get.return_value = mock_nvidia_backend

            with mock.patch("vllm.triton_utils.importing.backends", mock_backends):
                _configure_triton_ptxas_for_new_gpus()

            # Verify TRITON_PTXAS_PATH was set
            assert os.environ.get("TRITON_PTXAS_PATH") == mock_ptxas

    finally:
        # Restore original values
        if original_ptxas is None:
            os.environ.pop("TRITON_PTXAS_PATH", None)
        else:
            os.environ["TRITON_PTXAS_PATH"] = original_ptxas
        if original_cuda_home is None:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = original_cuda_home
