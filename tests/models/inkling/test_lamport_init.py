# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.inkling.nvidia.ops import lamport


@pytest.fixture
def fake_tp(monkeypatch):
    """Stub the TP group so init runs without distributed state."""

    def make(world_size: int, fully_connected: bool) -> list[tuple]:
        tp = SimpleNamespace(
            world_size=world_size,
            cpu_group=object(),
            device=torch.device("cuda:0"),
        )
        constructed: list[tuple] = []

        def fake_all_gather_object(output, obj, group):
            output[:] = list(range(world_size))

        monkeypatch.setattr(lamport, "_STATE", None)
        monkeypatch.setattr(lamport, "_STATE_FAILED", False)
        monkeypatch.delenv("LAMPORT_RS_SCONV", raising=False)
        monkeypatch.setattr(lamport, "get_tp_group", lambda: tp)
        monkeypatch.setattr(
            lamport, "in_the_same_node_as", lambda group: [True] * world_size
        )
        monkeypatch.setattr(lamport.dist, "all_gather_object", fake_all_gather_object)
        monkeypatch.setattr(
            lamport.current_platform,
            "is_fully_connected",
            lambda ids: fully_connected,
            raising=False,
        )
        monkeypatch.setattr(
            lamport,
            "LamportRSConv",
            lambda *args, **kwargs: constructed.append((args, kwargs)),
        )
        return constructed

    return make


@pytest.mark.parametrize(
    ("world_size", "fully_connected", "expect_lamport"),
    [(4, False, False), (8, False, False), (4, True, True), (2, False, True)],
)
def test_lamport_is_skipped_on_pcie_only_tp_groups(
    fake_tp, world_size, fully_connected, expect_lamport
) -> None:
    """Beyond two GPUs Lamport needs NVLink; PCIe-only TP must use NCCL."""
    constructed = fake_tp(world_size, fully_connected)

    lamport.initialize_lamport_rs_conv(
        hidden_size=1024, window_size=4, max_num_batched_tokens=8192
    )

    assert bool(constructed) == expect_lamport
