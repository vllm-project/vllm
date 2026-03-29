# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for _ROCM_DEVICE_ID_NAME_MAP in vllm.platforms.rocm.

These tests are fully offline (no GPU required) and verify the structure
and content of the device ID → name mapping used by get_device_name().
"""

from vllm.platforms.rocm import _ROCM_DEVICE_ID_NAME_MAP


def test_rocm_device_id_map_format():
    """All keys must be lowercase hex strings; all values must be non-empty."""
    for key, val in _ROCM_DEVICE_ID_NAME_MAP.items():
        assert key.startswith("0x"), f"Key {key!r} must start with '0x'"
        assert key == key.lower(), f"Key {key!r} must be lowercase"
        assert val, f"Value for {key!r} must be non-empty"
        assert " " not in val, f"Value {val!r} must use underscores, not spaces"


def test_rocm_device_id_map_known_entries():
    """Spot-check that known device IDs are present and map to expected names."""
    # MI300 Instinct series
    assert _ROCM_DEVICE_ID_NAME_MAP["0x74a1"] == "AMD_Instinct_MI300X"
    assert _ROCM_DEVICE_ID_NAME_MAP["0x74a0"] == "AMD_Instinct_MI300A"
    # RDNA 3.5 APUs — amdsmi reports generic "AMD Radeon Graphics" for these,
    # so explicit entries are required for stable name resolution.
    assert _ROCM_DEVICE_ID_NAME_MAP["0x150e"] == "AMD_Radeon_890M"  # gfx1150
    assert _ROCM_DEVICE_ID_NAME_MAP["0x1586"] == "AMD_Radeon_8060S"  # gfx1151
    # RDNA 4 discrete (Navi 48)
    assert _ROCM_DEVICE_ID_NAME_MAP["0x7550"] == "AMD_Radeon_RX9070XT"  # gfx1201


def test_rocm_device_id_map_no_duplicate_keys():
    """Each device ID must appear only once (dict enforces this, but be explicit)."""
    keys = list(_ROCM_DEVICE_ID_NAME_MAP.keys())
    assert len(keys) == len(set(keys)), "Duplicate device ID keys found"
