# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import types

import pytest

from vllm.platforms import current_platform


def _test_oink_availability_impl(
    device_capability: tuple[int, int],
    has_rmsnorm: bool,
    has_fused_add_rms_norm: bool,
    expected_available: bool,
    expected_fused: bool,
) -> None:
    """Test OINK support detection with mocked state."""
    import torch

    from vllm import platforms

    # Mock device capability (class method, override on class)
    dc = platforms.interface.DeviceCapability(*device_capability)
    platforms.current_platform.__class__.get_device_capability = lambda device_id=0: dc

    # Mock oink ops
    oink_ops = types.SimpleNamespace()
    if has_rmsnorm:
        oink_ops.rmsnorm = lambda x, w, eps: x
    if has_fused_add_rms_norm:
        oink_ops.fused_add_rms_norm = lambda x, residual, w, eps: None

    torch.ops.oink = oink_ops

    # Now import vllm modules with mocks in place (fresh import with mocked platform)
    import vllm.kernels.oink_ops  # noqa: F401
    from vllm.ir.ops import fused_add_rms_norm, rms_norm

    # Verify support checks
    assert rms_norm.impls["oink"].supported is expected_available
    assert fused_add_rms_norm.impls["oink"].supported is expected_fused


@pytest.mark.parametrize(
    "device_capability,has_rmsnorm,has_fused_add_rms_norm,expected_available,expected_fused",
    [
        # Case 1: < SM100, ops not supported
        ((9, 0), True, False, False, False),
        # Case 2: CUDA available and SM100, rmsnorm op registered
        ((10, 0), True, False, True, False),
        # Case 3: SM100 with both rmsnorm and fused_add_rms_norm
        ((10, 0), True, True, True, True),
    ],
)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test on CUDA")
def test_oink_availability_checks(
    device_capability: tuple[int, int],
    has_rmsnorm: bool,
    has_fused_add_rms_norm: bool,
    expected_available: bool,
    expected_fused: bool,
):
    """Test OINK support detection with clean import state for each parameter set."""

    # Use spawn to run function in fresh process with clean imports
    # TODO migrate to spawn utility:
    # https://github.com/vllm-project/vllm/issues/41415
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(
        target=_test_oink_availability_impl,
        args=(
            device_capability,
            has_rmsnorm,
            has_fused_add_rms_norm,
            expected_available,
            expected_fused,
        ),
    )
    process.start()
    process.join()

    if process.exitcode != 0:
        raise AssertionError(
            f"Subprocess test failed with exit code {process.exitcode}"
        )


def test_can_view_as_2d_stride_guard():
    # No global import
    import torch

    # Import the helper from the kernels module.
    from vllm.kernels.oink_ops import _can_view_as_2d

    x = torch.zeros((2, 3, 4))
    assert _can_view_as_2d(x) is True

    # Size-1 dims should be ignored by the viewability check.
    # Create a tensor where stride(0) != stride(1) * size(1) due to padding,
    # but view(-1, H) is still valid because dim 1 has size 1.
    base = torch.zeros((2, 10, 4))
    x_singleton = base[:, :1, :]
    x_singleton.view(-1, x_singleton.shape[-1])
    assert _can_view_as_2d(x_singleton) is True

    # Middle-dimension stride break: view(-1, hidden) should be invalid.
    x2 = x[:, ::2, :]
    with pytest.raises(RuntimeError):
        x2.view(-1, x2.shape[-1])
    assert _can_view_as_2d(x2) is False
