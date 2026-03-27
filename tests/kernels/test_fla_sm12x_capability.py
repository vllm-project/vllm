# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that FLA capability and SMEM checks correctly classify SM12x.

SM12x (desktop Blackwell: RTX 5090/5080, DGX Spark GB10) has capability
major=12, which previously triggered >= 9 checks designed for Hopper (SM90).
SM12x needs different NUM_WARPS tuning and has insufficient shared memory
for TMA code paths.
"""

import importlib
from unittest.mock import patch

import pytest


def _reload_utils(capability, device_name, max_shared_mem):
    """Reload fla/ops/utils.py with mocked platform values.

    Returns the reloaded module so callers can inspect is_nvidia_hopper
    and is_tma_supported.
    """
    import vllm.model_executor.layers.fla.ops.utils as utils_mod

    mock_cap = (capability[0], capability[1])

    def fake_has_device_capability(cap, device_id=0):
        if isinstance(cap, tuple):
            return mock_cap >= cap
        return (mock_cap[0] * 10 + mock_cap[1]) >= cap

    def fake_is_device_capability_family(cap, device_id=0):
        return (mock_cap[0] * 10 + mock_cap[1]) // 10 == cap // 10

    def fake_get_device_name(device_id=0):
        return device_name

    patches = [
        patch.object(
            type(utils_mod.current_platform),
            "has_device_capability",
            classmethod(lambda cls, cap, device_id=0:
                        fake_has_device_capability(cap, device_id)),
        ),
        patch.object(
            type(utils_mod.current_platform),
            "is_device_capability_family",
            classmethod(lambda cls, cap, device_id=0:
                        fake_is_device_capability_family(cap, device_id)),
        ),
        patch.object(
            type(utils_mod.current_platform),
            "get_device_name",
            classmethod(lambda cls, device_id=0:
                        fake_get_device_name(device_id)),
        ),
        patch(
            "vllm.model_executor.layers.fla.ops.utils.get_all_max_shared_mem",
            return_value=[max_shared_mem],
        ),
    ]
    for p in patches:
        p.start()
    try:
        importlib.reload(utils_mod)
        return utils_mod
    finally:
        for p in patches:
            p.stop()


@pytest.mark.parametrize(
    "capability,device_name,smem_bytes,expected_hopper,expected_tma",
    [
        # Ampere (SM80) — not Hopper, no TMA (also low SMEM for this test)
        ((8, 0), "NVIDIA A100", 166912, False, False),
        # Hopper (SM90) — is "Hopper-like", has TMA (228KB SMEM)
        ((9, 0), "NVIDIA H100", 232448, True, True),
        # Datacenter Blackwell (SM100) — "Hopper-like" per the range,
        # has TMA with sufficient SMEM
        ((10, 0), "NVIDIA B200", 232448, True, True),
        ((11, 0), "NVIDIA B300", 232448, True, True),
        # Desktop Blackwell RTX 5090 (SM120) — NOT Hopper-like, no TMA
        # (101KB SMEM is below 128KB threshold)
        ((12, 0), "NVIDIA GeForce RTX 5090", 101376, False, False),
        # DGX Spark GB10 (SM121) — NOT Hopper-like, no TMA
        ((12, 1), "Orin (nvgpu)", 101376, False, False),
        # Hypothetical SM12x with datacenter-class SMEM — NOT Hopper-like,
        # but TMA should be enabled since SMEM is sufficient
        ((12, 0), "NVIDIA RTX Pro 6000", 232448, False, True),
    ],
)
def test_fla_capability_classification(
    capability: tuple[int, int],
    device_name: str,
    smem_bytes: int,
    expected_hopper: bool,
    expected_tma: bool,
):
    """Verify is_nvidia_hopper and is_tma_supported for various architectures.

    Tests the actual module-level variables by reloading utils.py with mocked
    platform values, rather than reimplementing the logic.
    """
    utils = _reload_utils(capability, device_name, smem_bytes)

    assert utils.is_nvidia_hopper == expected_hopper, (
        f"capability {capability} ({device_name}): is_nvidia_hopper should be "
        f"{expected_hopper}, got {utils.is_nvidia_hopper}"
    )
    assert utils.is_tma_supported == expected_tma, (
        f"capability {capability} ({device_name}): is_tma_supported should be "
        f"{expected_tma}, got {utils.is_tma_supported}"
    )
