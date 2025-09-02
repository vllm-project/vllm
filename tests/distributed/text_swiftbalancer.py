import pytest
import torch

from vllm.distributed.eplb.policy.policy_swift_balancer import SwiftBalancer

def test_rebalance():
    """Test rebalancing functionality"""

    old_global_expert_indices = torch.tensor([
        [0, 1, 2, 3, 4,
         4, 5, 6, 7, 8,
         8, 9, 10, 11, 12,
         12, 13, 14, 15, 0],
        [0, 1, 2, 3, 6,
         4, 5, 6, 7, 9,
         8, 9, 10, 11, 6,
         12, 13, 14, 15, 6],
    ])

    weight = torch.tensor([
        [90, 132, 40, 61,
         104, 165, 39, 4,
         73, 56, 183, 86,
         22, 98, 65, 120],
        [20, 107, 104, 64,
         19, 197, 187, 157,
         172, 86, 16, 27,
         45, 85, 150, 164],
    ])

    num_layers = weight.shape[0]
    num_replicas = 20
    num_groups = 1
    num_nodes = 1
    num_rank = 4

    policy = SwiftBalancer()
    phy2log, log2phy, logcnt = policy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes,
        num_rank, old_global_expert_indices)

    # Verify output shapes
    assert phy2log.shape == (
        2,
        20,
    ), f"Expected `phy2log` shape (2, 20), got {phy2log.shape}"
    assert (log2phy.shape[0] == 2
            ), f"Expected `log2phy` first dimension 2, got {log2phy.shape[0]}"
    assert (
        log2phy.shape[1] == 16
    ), f"Expected `log2phy` second dimension 16, got {log2phy.shape[1]}"
    assert logcnt.shape == (
        2,
        16,
    ), f"Expected `logcnt` shape (2, 16), got {logcnt.shape}"

    # Verify physical to logical expert mapping range is correct
    assert torch.all(phy2log >= 0) and torch.all(
        phy2log < 16), "Physical to logical mapping should be in range [0, 16)"

    # Verify expert count reasonableness
    assert torch.all(
        logcnt >= 1), "Each logical expert should have at least 1 replica"
    assert (
        torch.sum(logcnt, dim=1).sum() == num_replicas *
        num_layers), f"Total replicas should be {num_replicas * num_layers}"

    # Verify expected output
    expected_phy2log = torch.tensor([[0, 1, 2, 3, 4,
                                      10, 5, 6, 7, 8,
                                      1, 9, 10, 11, 14,
                                      12, 13, 4, 15, 5],
                                     [0, 1, 2, 3, 6,
                                      4, 5, 8, 7, 9,
                                      8, 14, 10, 5, 6,
                                      12, 13, 11, 15, 7]])
    assert torch.all(phy2log == expected_phy2log)

    expected_logcnt = torch.tensor([[1, 2, 1, 1,
                                     2, 2, 1, 1,
                                     1, 1, 2, 1,
                                     1, 1, 1, 1],
                                    [1, 1, 1, 1,
                                     1, 2, 2, 2,
                                     2, 1, 1, 1,
                                     1, 1, 1, 1]])
    assert torch.all(logcnt == expected_logcnt)
