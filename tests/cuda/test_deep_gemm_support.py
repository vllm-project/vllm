# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType
from unittest.mock import patch

import pytest

# CPU development builds may omit the stable CUDA extension imported by
# ``vllm.platforms.cuda``.  The method under test only contains capability
# selection logic, so provide a stub when the extension is unavailable.  Try
# the real extension first so CUDA test processes are never masked by the stub.
try:
    import vllm._C_stable_libtorch  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "vllm._C_stable_libtorch":
        raise
    sys.modules["vllm._C_stable_libtorch"] = ModuleType("vllm._C_stable_libtorch")

from vllm.platforms.cuda import CudaPlatformBase  # noqa: E402


@pytest.mark.parametrize(
    ("device_capability", "expected"),
    [(90, True), (100, True), (120, True), (121, False)],
)
def test_support_deep_gemm_excludes_sm121(
    device_capability: int, expected: bool
) -> None:
    with (
        patch.object(
            CudaPlatformBase,
            "is_device_capability",
            side_effect=lambda capability: capability == device_capability,
        ),
        patch.object(
            CudaPlatformBase,
            "is_device_capability_family",
            side_effect=lambda capability: capability == device_capability,
        ) as is_device_capability_family,
    ):
        assert CudaPlatformBase.support_deep_gemm() is expected

    if device_capability == 121:
        is_device_capability_family.assert_not_called()
