# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECCPUConnector scheduler utilities."""

import contextlib
import uuid
from unittest.mock import MagicMock

import msgspec
import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    build_block_descs,
    deserialize_mem_descriptor,
    serialize_mem_descriptor,
    setup_ec_region,
)
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    ECSharedRegion,
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


# ── setup_ec_region ──────────────────────────────────────────────────────────


def _make_vllm_config(
    *,
    engine_id: str | None = None,
    dtype: torch.dtype = torch.float16,
    hidden_dim: int = 32,
    extra_config: dict | None = None,
    ec_transfer_config_present: bool = True,
) -> MagicMock:
    """Build a mock `VllmConfig` shaped enough for `setup_ec_region`."""
    cfg = MagicMock()
    cfg.model_config.dtype = dtype
    cfg.model_config.get_inputs_embeds_size.return_value = hidden_dim

    if ec_transfer_config_present:
        ec_cfg = MagicMock()
        ec_cfg.engine_id = engine_id if engine_id is not None else str(uuid.uuid4())
        ec_cfg.get_from_extra_config.side_effect = lambda key, default: (
            (extra_config or {}).get(key, default)
        )
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


def test_setup_ec_region_returns_layout_with_correct_shape(cleanup_regions):
    cfg = _make_vllm_config(dtype=torch.float16, hidden_dim=32)
    layout = setup_ec_region(cfg)
    cleanup_regions.append(layout.region)

    assert layout.dtype == torch.float16
    assert layout.hidden_dim == 32
    assert layout.element_size == 2  # float16 is 2 bytes
    assert layout.block_size_bytes == 32 * 2
    assert layout.num_blocks == 100000  # default


def test_setup_ec_region_extra_config_overrides_num_blocks(cleanup_regions):
    cfg = _make_vllm_config(extra_config={"num_ec_blocks": 64})
    layout = setup_ec_region(cfg)
    cleanup_regions.append(layout.region)
    assert layout.num_blocks == 64


@pytest.mark.parametrize(
    "dtype,hidden_dim,expected_element_size",
    [
        (torch.bfloat16, 128, 2),
        (torch.float32, 128, 4),
    ],
)
def test_setup_ec_region_block_size_uses_dtype(
    dtype, hidden_dim, expected_element_size, cleanup_regions
):
    """block_size_bytes = hidden_dim * dtype.element_size()."""
    cfg = _make_vllm_config(dtype=dtype, hidden_dim=hidden_dim)
    layout = setup_ec_region(cfg)
    cleanup_regions.append(layout.region)
    assert layout.element_size == expected_element_size
    assert layout.block_size_bytes == hidden_dim * expected_element_size


def test_setup_ec_region_uses_engine_id_for_mmap_path(cleanup_regions):
    engine_id = f"unit-{uuid.uuid4()}"
    cfg = _make_vllm_config(engine_id=engine_id)
    layout = setup_ec_region(cfg)
    cleanup_regions.append(layout.region)
    assert engine_id in layout.region.mmap_path


@pytest.mark.parametrize("missing", ["ec_config", "engine_id"])
def test_setup_ec_region_asserts_on_missing_config(missing):
    if missing == "ec_config":
        cfg = _make_vllm_config(ec_transfer_config_present=False)
    else:
        cfg = _make_vllm_config()
        cfg.ec_transfer_config.engine_id = None
    with pytest.raises(AssertionError):
        setup_ec_region(cfg)


def test_setup_ec_region_extra_config_value_coerced_to_int(cleanup_regions):
    """`int(ec_config.get_from_extra_config(...))` — string values must coerce."""
    cfg = _make_vllm_config(extra_config={"num_ec_blocks": "32"})
    layout = setup_ec_region(cfg)
    cleanup_regions.append(layout.region)
    assert layout.num_blocks == 32
