# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for scoping the ``VllmRunner`` memory-settle wait to the
devices an engine actually occupies.

These exercise :func:`resolve_engine_devices` across the four dimensions
that determine which visible-set device indices the memory-query path must
wait on: platform (CUDA vs XPU), presence of a visibility mask
(``CUDA_VISIBLE_DEVICES`` / ``ZE_AFFINITY_MASK``), and ``device_ids`` (unset,
contiguous, non-contiguous, or UUID strings). Integer ``device_ids`` are
visible ordinals on *both* platforms, so the selected indices are independent
of the platform and of any mask -- the mask only relabels which physical GPU
each visible ordinal points at, downstream of this selection.
"""

import pytest

from tests.utils import resolve_engine_devices


@pytest.mark.parametrize(
    ("device_ids", "num_devices", "expected"),
    [
        # No pin -> contiguous fallback [0, num_devices).
        (None, 4, [0, 1, 2, 3]),
        ([], 2, [0, 1]),
        # Contiguous integer pin is returned as-is.
        ([0, 1], 2, [0, 1]),
        # Non-contiguous integer pin (the regression the reviewer flagged).
        ([2, 3, 5, 7], 4, [2, 3, 5, 7]),
        ([1, 3], 2, [1, 3]),
    ],
)
def test_resolve_engine_devices(device_ids, num_devices, expected):
    assert resolve_engine_devices(device_ids, num_devices) == expected


def test_uuid_device_ids_fall_back_to_range():
    # UUID strings are not visible ordinals; fall back to the contiguous range
    # (same as the pre-existing behavior, no regression) rather than guessing.
    assert resolve_engine_devices(["GPU-abcd", "GPU-ef01"], 2) == [0, 1]


def test_returns_plain_int_list():
    # numpy-style ints or other integer subclasses should be normalized.
    result = resolve_engine_devices([1, 3], 4)
    assert result == [1, 3]
    assert all(type(d) is int for d in result)


# ---------------------------------------------------------------------------
# Four-dimension combination: platform x mask x device_ids -> queried device.
#
# ``resolve_engine_devices`` yields *visible* ordinals. What those ordinals
# resolve to downstream differs per platform:
#   * CUDA: the memory query maps visible -> physical via CUDA_VISIBLE_DEVICES
#     (``get_physical_device_indices``) before hitting NVML, which always
#     indexes by physical id. CUDA preserves the *listed* order of the mask, so
#     a non-contiguous / reordered CVD maps ``visible[i] -> int(cvd_list[i])``.
#   * XPU: the query indexes ``torch.xpu.mem_get_info`` by visible ordinal
#     directly; ``get_physical_device_indices`` is a no-op because it only
#     reads CUDA_VISIBLE_DEVICES, not ZE_AFFINITY_MASK. The waited-on indices
#     are therefore the visible ordinals themselves, independent of the mask's
#     contiguity *and* of its listed order. (Empirically Level Zero even sorts
#     the mask ascending, e.g. ``ZE_AFFINITY_MASK=3,1`` binds visible 0 ->
#     physical 1; this is irrelevant here because parent and worker share the
#     same runtime and both address devices by visible ordinal.)
# The cases below assert the *engine-occupied physical GPUs* each platform ends
# up waiting on, which must match the GPUs the engine actually binds.
# ---------------------------------------------------------------------------

_COMBINATIONS = [
    # (platform, mask, device_ids, num_devices, expected_physical)
    # -- CUDA, no mask: visible == physical.
    ("cuda", None, None, 2, [0, 1]),
    ("cuda", None, [2, 3, 5, 7], 4, [2, 3, 5, 7]),
    # -- CUDA + contiguous CUDA_VISIBLE_DEVICES.
    ("cuda", "4,5,6,7", None, 2, [4, 5]),
    ("cuda", "4,5,6,7", [1, 3], 2, [5, 7]),
    # -- CUDA + non-contiguous CVD: visible ordinal -> listed physical id.
    ("cuda", "2,3,5,7", None, 2, [2, 3]),
    ("cuda", "2,3,5,7", [0, 3], 2, [2, 7]),
    ("cuda", "1,4,6,7", [0, 2], 2, [1, 6]),
    # -- CUDA + reordered (descending) non-contiguous CVD: order preserved.
    ("cuda", "7,5,3,1", [0, 1], 2, [7, 5]),
    # -- XPU, no mask: visible == physical.
    ("xpu", None, None, 2, [0, 1]),
    ("xpu", None, [1, 3], 2, [1, 3]),
    # -- XPU + ZE_AFFINITY_MASK: query stays in the visible namespace, so the
    #    visible ordinals themselves are what gets waited on regardless of the
    #    mask's contiguity or order. get_physical_device_indices must NOT remap.
    ("xpu", "4,5,6,7", None, 2, [0, 1]),
    ("xpu", "4,5,6,7", [1, 3], 2, [1, 3]),
    ("xpu", "2,3,5,7", None, 2, [0, 1]),
    ("xpu", "2,3,5,7", [1, 3], 2, [1, 3]),
    ("xpu", "7,3", [0, 1], 2, [0, 1]),
]


@pytest.mark.parametrize(
    ("platform", "mask", "device_ids", "num_devices", "expected_physical"),
    _COMBINATIONS,
)
def test_platform_mask_device_ids_combination(
    monkeypatch, platform, mask, device_ids, num_devices, expected_physical
):
    from tests import utils as test_utils

    # This drives the real ``get_physical_device_indices``, which hardcodes
    # ``CUDA_VISIBLE_DEVICES``. The env var name below is deliberately that
    # literal, matching the function under test -- do NOT switch it to
    # ``current_platform.device_control_env_var`` (that would be
    # ``ZE_AFFINITY_MASK`` on an XPU host, which the function ignores, so the
    # "cuda" cases would stop mapping and the test would break). ``platform``
    # here is a logical label, not the host platform.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if platform == "cuda" and mask is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    visible = resolve_engine_devices(device_ids, num_devices)

    if platform == "cuda":
        # CUDA memory query maps visible -> physical via CUDA_VISIBLE_DEVICES.
        queried_physical = test_utils.get_physical_device_indices(visible)
    else:
        # XPU query indexes by visible ordinal; get_physical_device_indices is
        # a no-op for ZE_AFFINITY_MASK, so the visible ordinals are used as-is.
        queried_physical = test_utils.get_physical_device_indices(visible)

    assert queried_physical == expected_physical
