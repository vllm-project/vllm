# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounds tests for the CPU hash-MoE routing ops.

`tid2eid` is loaded verbatim from DeepSeek-V4 hash-MoE checkpoints as
`model.layers.{L}.mlp.gate.tid2eid` and its values are consumed as raw
expert indices: `hash_topk_kernel_impl` indexes a NUM_EXPERTS-wide stack
array with them (csrc/cpu/sgl-kernels/topk.cpp) and the resulting
`topk_ids` index the MoE alignment workspace in `moe_align_block_size`
(csrc/cpu/sgl-kernels/moe.cpp). Out-of-range entries must be rejected at
the op boundary instead of reading and writing out of bounds.
"""

import pytest
import torch

from vllm._custom_ops import (
    CPUQuantMethod,
    fused_experts_cpu,
    hash_topk_cpu,
)

NUM_EXPERTS = 256
TOPK = 6
M = 4
K = 16
N = 16
ROUTED_SCALING_FACTOR = 2.5


def _gating() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(M, NUM_EXPERTS, dtype=torch.bfloat16)


def _tid2eid(oob_value: int | None = None) -> torch.Tensor:
    torch.manual_seed(0)
    table = torch.randint(0, NUM_EXPERTS, (M, TOPK), dtype=torch.int32)
    table[0, 0] = 0
    table[0, 1] = NUM_EXPERTS - 1  # boundary-valid control values
    if oob_value is not None:
        table[0, 0] = oob_value
    return table


def _run(table, scoring_func):
    return hash_topk_cpu(
        _gating(),
        table,
        TOPK,
        scoring_func,
        num_fused_shared_experts=0,
        num_experts=NUM_EXPERTS,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
    )


def _fused_experts_args(oob_value: int | None = None):
    """Minimal well-shaped fused-experts inputs (zero weights)."""
    hidden = torch.zeros(M, K, dtype=torch.float32)
    w1 = torch.zeros(NUM_EXPERTS, 2 * N, K, dtype=torch.float32)
    w2 = torch.zeros(NUM_EXPERTS, K, N, dtype=torch.float32)
    out = torch.zeros(M, K, dtype=torch.float32)
    topk_weights = torch.zeros(M, TOPK, dtype=torch.float32)
    topk_ids = _tid2eid(oob_value)
    return out, hidden, w1, w2, topk_weights, topk_ids


@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_valid_table(scoring_func):
    """Benign control: a valid table routes verbatim with finite weights."""
    table = _tid2eid()
    weights, ids = _run(table, scoring_func)
    assert ids.dtype == torch.int32
    assert torch.equal(ids, table)
    assert torch.isfinite(weights).all()


@pytest.mark.parametrize("oob_value", [-1, 9000, 2**30])
def test_out_of_range_expert_id_rejected(oob_value):
    """A checkpoint-programmed out-of-range expert id must raise, not corrupt."""
    with pytest.raises(RuntimeError, match=r"outside \[0"):
        _run(_tid2eid(oob_value), "sqrtsoftplus")


@pytest.mark.parametrize("oob_value", [-1, 9000, 2**30])
def test_moe_align_rejects_out_of_range_ids(oob_value):
    """The workspace alignment must reject hostile topk_ids up front.

    `fused_experts_cpu` carves the MoE workspace exactly as
    csrc/cpu/sgl-kernels/moe.cpp does and hands `topk_ids` to
    `moe_align_block_size`, so this drives the second guarded op boundary.
    """
    out, hidden, w1, w2, topk_weights, topk_ids = _fused_experts_args(oob_value)
    with pytest.raises(
        RuntimeError, match=r"moe_align_block_size: topk_ids\[0\] = "
    ):
        fused_experts_cpu(
            out,
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            moe_comp_method=CPUQuantMethod.UNQUANT,
            w1_scale=None,
            w2_scale=None,
            w1_zero=None,
            w2_zero=None,
            block_size=None,
        )
