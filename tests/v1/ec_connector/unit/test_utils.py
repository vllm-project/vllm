# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECCPUConnector scheduler utilities."""

import contextlib
import uuid
from unittest.mock import MagicMock

import msgspec
import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    create_ec_shared_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    build_block_descs,
    deserialize_mem_descriptor,
    serialize_mem_descriptor,
)

# ── build_block_descs ────────────────────────────────────────────────────────


@pytest.mark.parametrize("device_id,expected_dev", [(7, 7), (None, 0)])
def test_build_block_descs(device_id, expected_dev):
    kwargs = dict(base_ptr=1000, num_blocks=4, block_size_bytes=256)
    if device_id is not None:
        kwargs["device_id"] = device_id
    descs = build_block_descs(**kwargs)
    assert len(descs) == 4
    for i, (addr, size, dev) in enumerate(descs):
        assert addr == 1000 + i * 256
        assert size == 256
        assert dev == expected_dev


def test_build_block_descs_zero_blocks_returns_empty():
    assert build_block_descs(base_ptr=100, num_blocks=0, block_size_bytes=64) == []


# ── serialize / deserialize_mem_descriptor ───────────────────────────────────


def test_mem_descriptor_roundtrip():
    descs = [(100, 64, 0), (164, 64, 0), (228, 64, 1)]
    assert deserialize_mem_descriptor(serialize_mem_descriptor(descs)) == descs


@pytest.mark.parametrize(
    "bad_value",
    [
        [(1, 2)],  # 2-tuple instead of 3
        [("a", "b", "c")],  # strings, not ints
    ],
)
def test_mem_descriptor_rejects_malformed_payload(bad_value):
    """Malformed descriptor lists must fail to decode."""
    encoder = msgspec.msgpack.Encoder()
    bad_payload = encoder.encode(bad_value)
    with pytest.raises((msgspec.DecodeError, msgspec.ValidationError)):
        deserialize_mem_descriptor(bad_payload)


# ── create_ec_shared_region ──────────────────────────────────────────────────


def _make_vllm_config(
    *,
    instance_id: str | None = None,
    dp_rank: int = 0,
    dtype: torch.dtype = torch.float16,
    hidden_dim: int = 32,
    ec_cpu_bytes: object = 100000 * 32 * 2,
    ec_transfer_config_present: bool = True,
) -> MagicMock:
    """Build a mock `VllmConfig` shaped enough for `create_ec_shared_region`."""
    cfg = MagicMock()
    cfg.instance_id = instance_id if instance_id is not None else str(uuid.uuid4())
    cfg.parallel_config.data_parallel_rank = dp_rank
    cfg.model_config.dtype = dtype
    # No vision config → _get_encoder_cache_hidden_dim falls back to
    # get_inputs_embeds_size() rather than the deepstack branch.
    cfg.model_config.hf_config = None
    cfg.model_config.get_inputs_embeds_size.return_value = hidden_dim

    if ec_transfer_config_present:
        ec_cfg = MagicMock()
        ec_cfg.ec_connector_extra_config = {"ec_cpu_bytes": ec_cpu_bytes}
        cfg.ec_transfer_config = ec_cfg
    else:
        cfg.ec_transfer_config = None
    return cfg


@pytest.fixture
def cleanup_regions():
    """Track regions created by tests and unlink them at teardown."""
    regions: list[ECSharedRegion] = []
    yield regions
    for r in regions:
        try:
            r.cleanup()
        except Exception:
            contextlib.suppress(Exception)


def test_create_ec_shared_region_returns_region_with_correct_shape(cleanup_regions):
    cfg = _make_vllm_config(dtype=torch.float16, hidden_dim=32, ec_cpu_bytes=64 * 64)
    region = create_ec_shared_region(cfg)
    cleanup_regions.append(region)

    assert region.block_size_bytes == 32 * 2  # hidden_dim * float16 element_size
    assert region.num_blocks == (64 * 64) // (32 * 2)


@pytest.mark.parametrize(
    "dtype,hidden_dim,expected_element_size",
    [
        (torch.bfloat16, 128, 2),
        (torch.float32, 128, 4),
    ],
)
def test_create_ec_shared_region_block_size_uses_dtype(
    dtype, hidden_dim, expected_element_size, cleanup_regions
):
    """block_size_bytes = hidden_dim * dtype.element_size()."""
    block_size_bytes = hidden_dim * expected_element_size
    cfg = _make_vllm_config(
        dtype=dtype, hidden_dim=hidden_dim, ec_cpu_bytes=block_size_bytes * 4
    )
    region = create_ec_shared_region(cfg)
    cleanup_regions.append(region)
    assert region.block_size_bytes == block_size_bytes


def test_create_ec_shared_region_uses_instance_id_dp_rank_for_mmap_path(
    cleanup_regions,
):
    instance_id = f"unit-{uuid.uuid4()}"
    cfg = _make_vllm_config(instance_id=instance_id, dp_rank=3)
    region = create_ec_shared_region(cfg)
    cleanup_regions.append(region)
    assert f"{instance_id}_dp3" in region._mmap_path


def test_create_ec_shared_region_asserts_on_missing_ec_config():
    cfg = _make_vllm_config(ec_transfer_config_present=False)
    with pytest.raises(AssertionError):
        create_ec_shared_region(cfg)


@pytest.mark.parametrize("missing_bytes", [None, 0])
def test_create_ec_shared_region_raises_when_ec_cpu_bytes_missing(missing_bytes):
    cfg = _make_vllm_config(ec_cpu_bytes=missing_bytes)
    with pytest.raises(ValueError, match="ec_cpu_bytes"):
        create_ec_shared_region(cfg)


def test_create_ec_shared_region_ec_cpu_bytes_coerced_to_int(cleanup_regions):
    """`int(ec_config.ec_connector_extra_config[...])` — strings must coerce."""
    cfg = _make_vllm_config(dtype=torch.float16, hidden_dim=32, ec_cpu_bytes="128")
    region = create_ec_shared_region(cfg)
    cleanup_regions.append(region)
    assert region.num_blocks == 128 // (32 * 2)
