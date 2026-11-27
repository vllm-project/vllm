# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FSOffloadingSpec's access_mode configuration."""

import pytest

from tests.v1.kv_offload.conftest import make_offloading_config
from vllm.v1.kv_offload.base import ReqContext, StoreTrigger, make_offload_key
from vllm.v1.kv_offload.config import OffloadAccessMode
from vllm.v1.kv_offload.fs.spec import FSOffloadingSpec


def _make_spec(root_dir, **extra_config) -> FSOffloadingSpec:
    return FSOffloadingSpec(
        make_offloading_config(
            spec_name="FSOffloadingSpec",
            cpu_bytes_to_use=None,
            extra_config={"root_dir": str(root_dir), **extra_config},
        )
    )


@pytest.mark.parametrize(
    "access_mode, can_read, can_write",
    [
        (None, True, True),
        ("readwrite", True, True),
        ("read", True, False),
        ("write", False, True),
    ],
)
def test_access_mode_sets_direction(tmp_path, access_mode, can_read, can_write):
    """FSOffloadingSpec takes direction from access_mode alone, defaulting to
    readwrite, and exposes the flags the connector scheduler gates lookups
    and store jobs on."""
    extra_config = {} if access_mode is None else {"access_mode": access_mode}
    spec = _make_spec(tmp_path, **extra_config)

    assert spec.access_mode is OffloadAccessMode(access_mode or "readwrite")
    assert spec.access_mode.can_read is can_read
    assert spec.access_mode.can_write is can_write


@pytest.mark.parametrize("access_mode", ["both", "ro", ""])
def test_invalid_access_mode_raises_at_construction(tmp_path, access_mode):
    with pytest.raises(ValueError, match="Invalid access_mode"):
        _make_spec(tmp_path, access_mode=access_mode)


def test_read_mode_does_not_gate_the_manager(tmp_path):
    """Direction is enforced by the connector scheduler, which issues no store
    job for a manager that cannot write. The manager keeps no direction state
    of its own, so it still accepts a store if called directly."""
    spec = _make_spec(tmp_path, access_mode="read")
    manager = spec.get_manager()
    key = make_offload_key(b"block-hash", 0)

    output = manager.prepare_store(
        [key], ReqContext(req_id="r1"), trigger=StoreTrigger.ON_COMPUTE
    )

    assert output is not None
    assert output.keys_to_store == [key]
