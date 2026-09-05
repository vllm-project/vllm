# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.eplb.expert_backup import (
    ExpertBackupDescriptor,
    ExpertBackupRegion,
    build_expert_backup_region,
)


def test_expert_backup_region_layout_and_views():
    weights = {
        "layers.0.experts.w13": torch.arange(12, dtype=torch.float32).view(3, 4),
        "layers.0.experts.w2": torch.arange(8, dtype=torch.bfloat16).view(2, 4),
    }

    region = build_expert_backup_region(
        weights,
        owner_node_rank=2,
        nixl_agent_metadata=b"agent-metadata",
        alignment=64,
    )

    assert region.buffer.device.type == "cpu"
    assert region.buffer.is_contiguous()
    assert region.descriptor.owner_node_rank == 2
    for name, expected in weights.items():
        location = region.descriptor.weight_pointer_map[name]
        assert (location.addr - region.descriptor.backup_region_base) % 64 == 0
        torch.testing.assert_close(region.tensor(name), expected)


def test_expert_backup_descriptor_round_trip():
    region = build_expert_backup_region(
        {
            "z": torch.arange(4, dtype=torch.int32),
            "a": torch.arange(6, dtype=torch.float16).view(2, 3),
        },
        owner_node_rank=1,
        nixl_agent_metadata=b"\x00nixl\xff",
    )

    encoded = region.descriptor.to_bytes()
    decoded = ExpertBackupDescriptor.from_bytes(encoded)

    assert decoded == region.descriptor
    assert encoded == decoded.to_bytes()
    remote_view = ExpertBackupRegion(region.buffer, decoded)
    torch.testing.assert_close(
        remote_view.tensor("a"),
        torch.arange(6, dtype=torch.float16).view(2, 3),
    )


@pytest.mark.parametrize("alignment", [0, -1, 3, 63])
def test_expert_backup_rejects_invalid_alignment(alignment):
    with pytest.raises(ValueError, match="positive power of two"):
        build_expert_backup_region(
            {"weight": torch.ones(1)},
            owner_node_rank=0,
            alignment=alignment,
        )


def test_expert_backup_rejects_corrupt_descriptor():
    with pytest.raises(ValueError, match="Invalid expert backup descriptor"):
        ExpertBackupDescriptor.from_bytes(b'{"owner_node_rank": 0}')
