# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

from vllm.utils import flashinfer


def _clear_cache() -> None:
    flashinfer.has_flashinfer_cutlass_fused_moe_fp4.cache_clear()


def test_fp4_capability_accepts_precompiled_cubin():
    _clear_cache()
    with (
        patch.object(flashinfer, "has_flashinfer_cutlass_fused_moe", return_value=True),
        patch.object(flashinfer, "has_flashinfer_cubin", return_value=True),
        patch.object(flashinfer.shutil, "which", return_value=None),
    ):
        assert flashinfer.has_flashinfer_cutlass_fused_moe_fp4()
    _clear_cache()


def test_fp4_capability_requires_supported_nvcc_without_cubin():
    for version, expected in [
        ("Cuda compilation tools, release 12.7, V12.7.99", False),
        ("Cuda compilation tools, release 12.8, V12.8.99", True),
    ]:
        _clear_cache()
        with (
            patch.object(
                flashinfer, "has_flashinfer_cutlass_fused_moe", return_value=True
            ),
            patch.object(flashinfer, "has_flashinfer_cubin", return_value=False),
            patch.object(flashinfer.importlib.util, "find_spec", return_value=None),
            patch.object(flashinfer.shutil, "which", return_value="/usr/bin/nvcc"),
            patch.object(
                flashinfer.subprocess,
                "run",
                return_value=SimpleNamespace(stdout=version),
            ),
        ):
            assert flashinfer.has_flashinfer_cutlass_fused_moe_fp4() is expected
    _clear_cache()
