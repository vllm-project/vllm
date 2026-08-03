# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for pure NIXL member-transfer planning."""

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.member_transfer import (
    plan_member_transfer,
    validate_region_members,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)


def _metadata(
    region_members: list[list[str]],
    base_addresses: list[int],
    block_lens: list[int],
    *,
    packed_block_stride: int = 0,
    packed_member_layouts: dict[str, tuple[int, int]] | None = None,
) -> NixlAgentMetadata:
    return NixlAgentMetadata(
        engine_id="remote-engine",
        agent_metadata=b"agent",
        kv_caches_base_addr=base_addresses,
        device_id=7,
        num_blocks=2,
        block_lens=block_lens,
        kv_cache_layout="HND",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASH_ATTN",
        physical_blocks_per_logical_kv_block=1,
        region_members=region_members,
        packed_block_stride=packed_block_stride,
        packed_member_layouts=packed_member_layouts or {},
    )


def test_member_metadata_round_trip():
    metadata = _metadata([["L0", "L1"]], [0x10000], [256])

    encoded = msgspec.msgpack.encode(metadata)
    assert msgspec.msgpack.Decoder(NixlAgentMetadata).decode(encoded) == metadata


def test_packed_member_metadata_round_trip():
    metadata = _metadata(
        [["L0", "L1"]],
        [0x10000],
        [256],
        packed_block_stride=256,
        packed_member_layouts={"L0": (0, 128), "L1": (128, 128)},
    )

    encoded = msgspec.msgpack.encode(metadata)
    assert msgspec.msgpack.Decoder(NixlAgentMetadata).decode(encoded) == metadata


def test_plan_member_transfer_expands_pooled_regions():
    metadata = _metadata(
        [["a", "a.swa"], ["b"]],
        [0xA000, 0xB000],
        [128, 128],
    )

    prepared, plan = plan_member_transfer(
        metadata,
        [["a", "a.swa"], ["b"]],
        {"a": 0, "a.swa": 1, "b": 0},
    )

    assert plan.member_names == ("a", "a.swa", "b")
    assert plan.local_regions == (0, 0, 1)
    assert plan.group_ids == (0, 1, 0)
    assert prepared.kv_caches_base_addr == [0xA000, 0xA000, 0xB000]
    assert prepared.block_lens == [128, 128, 128]
    assert prepared.region_members == []


def test_plan_member_transfer_filters_and_reorders_a_pp_stage():
    metadata = _metadata(
        [["l2"], ["l0"], ["l3"], ["l1"]],
        [0xC000, 0xA000, 0xD000, 0xB000],
        [65536, 65536, 32768, 32768],
    )

    prepared, plan = plan_member_transfer(
        metadata,
        [["l2"], ["l3"]],
        {"l2": 0, "l3": 1},
    )

    assert plan.local_regions == (0, 1)
    assert plan.group_ids == (0, 1)
    assert prepared.kv_caches_base_addr == [0xC000, 0xD000]
    assert prepared.block_lens == [65536, 32768]


def test_plan_member_transfer_rejects_missing_local_member():
    metadata = _metadata([["l0"]], [0xA000], [128])

    with pytest.raises(RuntimeError, match="missing locally owned"):
        plan_member_transfer(
            metadata,
            [["l0"], ["l1"]],
            {"l0": 0, "l1": 1},
        )


def test_plan_member_transfer_rejects_duplicate_remote_member():
    metadata = _metadata([["a"], ["a"]], [0xA000, 0xB000], [128, 128])

    with pytest.raises(RuntimeError, match="multiple NIXL regions"):
        plan_member_transfer(metadata, [["a"]], {"a": 0})


def test_plan_member_transfer_is_canonical_across_remote_orderings():
    rank0 = _metadata([["x"], ["y"]], [0x1000, 0x2000], [64, 128])
    rank1 = _metadata([["y"], ["x"]], [0x2000, 0x1000], [128, 64])
    local_members = [["x"], ["y"]]
    layer_to_group = {"x": 0, "y": 1}

    _, plan0 = plan_member_transfer(rank0, local_members, layer_to_group)
    prepared1, plan1 = plan_member_transfer(rank1, local_members, layer_to_group)

    assert plan0 == plan1
    assert prepared1.kv_caches_base_addr == [0x1000, 0x2000]
    assert prepared1.block_lens == [64, 128]


def test_validate_region_members_rejects_duplicate_local_member():
    with pytest.raises(RuntimeError, match="spans multiple NIXL regions"):
        validate_region_members([["a"], ["a"]])


def test_plan_packed_member_transfer_keeps_local_and_remote_strides():
    remote = _metadata(
        [["L0", "L1"]],
        [0x10000],
        [256],
        packed_block_stride=256,
        packed_member_layouts={"L0": (0, 128), "L1": (128, 128)},
    )

    prepared, plan = plan_member_transfer(
        remote,
        [["L1"]],
        {"L1": 0},
        local_packed_layouts={"L1": (0, 128)},
        local_block_stride=128,
    )

    assert prepared.kv_caches_base_addr == [0x10080]
    assert prepared.block_lens == [128]
    assert prepared.packed_block_stride == 0
    assert prepared.packed_member_layouts == {}
    assert plan.member_names == ("L1",)
    assert plan.local_layouts == ((0, 128),)
    assert plan.local_block_stride == 128
    assert plan.remote_block_stride == 256
    assert plan.is_packed


@pytest.mark.parametrize(
    ("local_packed", "remote_packed"),
    [(True, False), (False, True)],
)
def test_plan_member_transfer_rejects_mixed_packed_layouts(
    local_packed: bool,
    remote_packed: bool,
):
    remote = (
        _metadata(
            [["L0"]],
            [0x10000],
            [128],
            packed_block_stride=128,
            packed_member_layouts={"L0": (0, 128)},
        )
        if remote_packed
        else _metadata([["L0"]], [0x10000], [128])
    )

    with pytest.raises(RuntimeError, match="cannot be mixed"):
        plan_member_transfer(
            remote,
            [["L0"]],
            {"L0": 0},
            local_packed_layouts={"L0": (0, 128)} if local_packed else None,
            local_block_stride=128 if local_packed else 0,
        )
