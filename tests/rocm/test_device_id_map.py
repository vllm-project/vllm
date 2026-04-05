# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for _ROCM_DEVICE_ID_NAME_MAP in vllm.platforms.rocm.

These tests are fully offline (no GPU required) and verify the structure
and content of the device ID → name mapping used by get_device_name().
"""

import ast
import importlib.util

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
    """Parse the source file with ast to detect duplicate keys in the dict literal.

    Python silently drops duplicate keys at runtime, so a runtime check is
    always true and provides no signal.  Inspecting the AST of the source
    literal catches any accidental duplicate before it reaches the interpreter.
    """
    spec = importlib.util.find_spec("vllm.platforms.rocm")
    assert spec is not None and spec.origin is not None, (
        "Could not locate vllm/platforms/rocm.py"
    )

    with open(spec.origin) as fh:
        source = fh.read()
    tree = ast.parse(source, filename=spec.origin)

    # Walk assignments to find _ROCM_DEVICE_ID_NAME_MAP = {...}.
    # The variable uses an annotated assignment (ast.AnnAssign), e.g.:
    #   _ROCM_DEVICE_ID_NAME_MAP: dict[str, str] = { ... }
    # Plain ast.Assign is also handled for future-proofing.
    literal_keys: list[str] = []
    found = False
    for node in ast.walk(tree):
        dict_node: ast.Dict | None = None
        if isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "_ROCM_DEVICE_ID_NAME_MAP"
                and isinstance(node.value, ast.Dict)
            ):
                dict_node = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "_ROCM_DEVICE_ID_NAME_MAP"
                    and isinstance(node.value, ast.Dict)
                ):
                    dict_node = node.value
                    break
        if dict_node is not None:
            for key_node in dict_node.keys:
                if isinstance(key_node, ast.Constant) and isinstance(
                    key_node.value, str
                ):
                    literal_keys.append(key_node.value)
            found = True
            break

    assert found, "_ROCM_DEVICE_ID_NAME_MAP dict literal not found in source"

    duplicates = {k for k in literal_keys if literal_keys.count(k) > 1}
    assert not duplicates, (
        f"Duplicate device ID keys in _ROCM_DEVICE_ID_NAME_MAP source: "
        f"{sorted(duplicates)}"
    )
