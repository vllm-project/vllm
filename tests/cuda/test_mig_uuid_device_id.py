# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for device_id_to_physical_device_id handling MIG device UUIDs.

When CUDA_VISIBLE_DEVICES is set to a MIG device UUID (e.g.
"MIG-377e0049-554c-540b-93c6-d0976f8426cb"), the function must not raise
ValueError and must return the logical device_id unchanged, because CUDA
already remaps the UUID to device 0.

Regression test for https://github.com/vllm-project/vllm/issues/41848.
"""

import pytest


def test_device_id_integer_passthrough():
    """Standard integer CUDA_VISIBLE_DEVICES still works."""
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2"}):
        assert CudaPlatform.device_id_to_physical_device_id(0) == 2


def test_device_id_mig_uuid_returns_logical_index():
    """MIG UUID in CUDA_VISIBLE_DEVICES must return device_id, not raise."""
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    mig_uuid = "MIG-377e0049-554c-540b-93c6-d0976f8426cb"
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": mig_uuid}):
        # Before the fix this raised:
        #   ValueError: invalid literal for int() with base 10: 'MIG-377e0049-...'
        result = CudaPlatform.device_id_to_physical_device_id(0)
        assert result == 0, (
            f"Expected logical device_id=0, got {result!r}. "
            "CUDA maps MIG UUIDs to logical device 0 transparently."
        )


def test_device_id_multiple_mig_uuids():
    """Multiple MIG UUIDs comma-separated: each maps to its logical index."""
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    uuid0 = "MIG-377e0049-554c-540b-93c6-d0976f8426cb"
    uuid1 = "MIG-f39aa248-6f1b-5769-b119-8d650bb34b27"
    with patch.dict("os.environ",
                    {"CUDA_VISIBLE_DEVICES": f"{uuid0},{uuid1}"}):
        assert CudaPlatform.device_id_to_physical_device_id(0) == 0
        assert CudaPlatform.device_id_to_physical_device_id(1) == 1


def test_device_id_unset_env_returns_device_id():
    """When CUDA_VISIBLE_DEVICES is unset, device_id is returned as-is."""
    import os
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    env = {k: v for k, v in os.environ.items()
           if k != "CUDA_VISIBLE_DEVICES"}
    with patch.dict("os.environ", env, clear=True):
        assert CudaPlatform.device_id_to_physical_device_id(3) == 3


def test_device_id_empty_env_returns_device_id():
    """Empty CUDA_VISIBLE_DEVICES is treated as unset (Ray compatibility)."""
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
        assert CudaPlatform.device_id_to_physical_device_id(2) == 2
