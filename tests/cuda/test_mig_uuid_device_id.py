# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for device_id_to_physical_device_id handling MIG device UUIDs.

When CUDA_VISIBLE_DEVICES is set to a MIG device UUID (e.g.
"MIG-377e0049-554c-540b-93c6-d0976f8426cb"), the function must return the
UUID string so that callers can use nvmlDeviceGetHandleByUUID and get
accurate per-partition hardware info (e.g. 20 GiB for a 2g.20gb slice,
not 80 GiB for the parent GPU).

Regression test for https://github.com/vllm-project/vllm/issues/41848.
"""


def test_device_id_integer_passthrough():
    """Standard integer CUDA_VISIBLE_DEVICES still works."""
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2"}):
        assert CudaPlatform.device_id_to_physical_device_id(0) == 2


def test_device_id_mig_uuid_returns_uuid_string():
    """MIG UUID in CUDA_VISIBLE_DEVICES must return the UUID string, not raise.

    Before the fix: raised ValueError: invalid literal for int() with base 10.
    After the fix:  returns the UUID string for use with nvmlDeviceGetHandleByUUID.
    """
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    mig_uuid = "MIG-377e0049-554c-540b-93c6-d0976f8426cb"
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": mig_uuid}):
        result = CudaPlatform.device_id_to_physical_device_id(0)
        assert result == mig_uuid, (
            f"Expected UUID string {mig_uuid!r}, got {result!r}. "
            "The UUID must be returned so callers can query MIG partition info."
        )


def test_device_id_multiple_mig_uuids():
    """Multiple MIG UUIDs comma-separated: each returns its own UUID string."""
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    uuid0 = "MIG-377e0049-554c-540b-93c6-d0976f8426cb"
    uuid1 = "MIG-f39aa248-6f1b-5769-b119-8d650bb34b27"
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": f"{uuid0},{uuid1}"}):
        assert CudaPlatform.device_id_to_physical_device_id(0) == uuid0
        assert CudaPlatform.device_id_to_physical_device_id(1) == uuid1


def test_device_id_unset_env_returns_device_id():
    """When CUDA_VISIBLE_DEVICES is unset, device_id is returned as-is."""
    import os
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    with patch.dict("os.environ", env, clear=True):
        assert CudaPlatform.device_id_to_physical_device_id(3) == 3


def test_device_id_empty_env_returns_device_id():
    """Empty CUDA_VISIBLE_DEVICES is treated as unset (Ray compatibility)."""
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
        assert CudaPlatform.device_id_to_physical_device_id(2) == 2


def test_device_id_mixed_integer_and_mig_uuid():
    """Mix of integer index and MIG UUID in CUDA_VISIBLE_DEVICES.

    Covers hosts that expose both a full GPU (integer) and a MIG partition
    (UUID) in the same CUDA_VISIBLE_DEVICES string, e.g.:
        CUDA_VISIBLE_DEVICES="0,MIG-377e0049-554c-540b-93c6-d0976f8426cb"
    """
    from unittest.mock import patch

    from vllm.platforms.cuda import CudaPlatform

    mig_uuid = "MIG-377e0049-554c-540b-93c6-d0976f8426cb"
    with patch.dict("os.environ",
                    {"CUDA_VISIBLE_DEVICES": f"0,{mig_uuid}"}):
        assert CudaPlatform.device_id_to_physical_device_id(0) == 0
        assert CudaPlatform.device_id_to_physical_device_id(1) == mig_uuid
