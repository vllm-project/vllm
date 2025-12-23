# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.eplb.policy.swift_balancer import SwiftBalancerPolicy


def test_rebalance():
    """Test rebalancing functionality"""

    old_global_expert_indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15, 0],
            [0, 1, 2, 3, 6, 4, 5, 6, 7, 9, 8, 9, 10, 11, 6, 12, 13, 14, 15, 6],
        ]
    )

    weight = torch.tensor(
        [
            [29, 11, 41, 27, 19, 37, 21, 33, 13, 25, 39, 7, 15, 43, 47, 53],
            [25, 30, 12, 18, 22, 28, 32, 8, 14, 26, 38, 40, 42, 45, 50, 100],
        ]
    )

    num_layers = weight.shape[0]
    num_replicas = 20
    num_groups = 1
    num_nodes = 1
    num_rank = 4

    policy = SwiftBalancerPolicy()

    phy2log, log2phy, logcnt = policy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_rank, old_global_expert_indices
    )

    # Verify output shapes
    assert phy2log.shape == (
        2,
        20,
    ), f"Expected `phy2log` shape (2, 20), got {phy2log.shape}"
    assert log2phy.shape[0] == 2, (
        f"Expected `log2phy` first dimension 2, got {log2phy.shape[0]}"
    )
    assert log2phy.shape[1] == 16, (
        f"Expected `log2phy` second dimension 16, got {log2phy.shape[1]}"
    )
    assert logcnt.shape == (
        2,
        16,
    ), f"Expected `logcnt` shape (2, 16), got {logcnt.shape}"

    # Verify physical to logical expert mapping range is correct
    assert torch.all(phy2log >= 0) and torch.all(phy2log < 16), (
        "Physical to logical mapping should be in range [0, 16)"
    )

    # Verify expert count reasonableness
    assert torch.all(logcnt >= 1), "Each logical expert should have at least 1 replica"
    assert torch.sum(logcnt, dim=1).sum() == num_replicas * num_layers, (
        f"Total replicas should be {num_replicas * num_layers}"
    )

    # Verify expected output
    expected_phy2log = torch.tensor(
        [
            [0, 1, 9, 3, 14, 11, 5, 6, 7, 13, 8, 2, 10, 4, 15, 12, 13, 14, 15,
             2],
            [0, 1, 2, 3, 15, 4, 5, 6, 7, 15, 8, 9, 10, 11, 14, 12, 13, 14, 15,
             11]
        ]
    )
    assert torch.all(phy2log == expected_phy2log)

    expected_logcnt = torch.tensor(
        [
            [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3]
        ]
    )
    assert torch.all(logcnt == expected_logcnt)
