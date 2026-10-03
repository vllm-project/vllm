# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock


class _Experts:
    def __call__(self, hidden_states, router_logits):
        return hidden_states.clone()


def _make_block() -> Qwen3MoeSparseMoeBlock:
    block = Qwen3MoeSparseMoeBlock.__new__(Qwen3MoeSparseMoeBlock)
    block.experts = _Experts()
    block.is_sequence_parallel = False
    return block


def test_sparse_moe_block_supports_1d_input():
    block = _make_block()
    hidden_states = torch.randn(8)

    output = Qwen3MoeSparseMoeBlock.forward(block, hidden_states)

    assert output.shape == hidden_states.shape
    assert torch.equal(output, hidden_states)


def test_sparse_moe_block_preserves_2d_input():
    block = _make_block()
    hidden_states = torch.randn(4, 8)

    output = Qwen3MoeSparseMoeBlock.forward(block, hidden_states)

    assert output.shape == hidden_states.shape
    assert torch.equal(output, hidden_states)
