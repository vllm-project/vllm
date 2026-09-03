# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models.conformer_encoder import RelPosMultiHeadAttention

pytestmark = pytest.mark.cpu_test


class _ReferenceRelPosMultiHeadAttention(RelPosMultiHeadAttention):
    """Pre-baddbmm relative attention implementation."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_len = q.size(0), q.size(1)
        residual = q
        q, k, v = self.forward_qkv(q, k, v)

        q = q.transpose(1, 2)
        position_batch = pos_emb.size(0)
        p = self.linear_pos(pos_emb)[0].view(position_batch, -1, self.n_head, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self._reference_rel_shift(matrix_bd)

        attention_scores = matrix_ac + matrix_bd
        attention_scores.mul_(self.scale)

        output, attention = self.forward_attention(attention_scores, v, mask=mask)
        output = self.forward_output(output, residual, batch_size, query_len)
        return output, attention

    @staticmethod
    def _reference_rel_shift(x: torch.Tensor) -> torch.Tensor:
        batch, heads, query_len, position_len = x.shape
        zero_pad = torch.zeros(
            (batch, heads, query_len, 1), device=x.device, dtype=x.dtype
        )
        x = torch.cat([zero_pad, x], dim=-1)
        x = x.view(batch, heads, position_len + 1, query_len)
        x = x[:, :, 1:].view(batch, heads, query_len, position_len)
        return x[:, :, :, : position_len // 2 + 1]


@pytest.fixture
def tp1_cpu(tmp_path, default_vllm_config):
    """Initialize a temporary single-rank CPU tensor-parallel group."""
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp_path / 'distributed'}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        yield
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("sequence", [2, 8])
def test_relative_attention_matches_reference(tp1_cpu, sequence: int) -> None:
    torch.manual_seed(7)
    batch, heads, model_dim = 2, 4, 32
    optimized = RelPosMultiHeadAttention(heads, model_dim)
    reference = _ReferenceRelPosMultiHeadAttention(heads, model_dim)

    with torch.no_grad():
        for parameter in optimized.parameters():
            parameter.uniform_(-0.2, 0.2)
    reference.load_state_dict(optimized.state_dict())

    query = torch.randn(batch, sequence, model_dim)
    key = torch.randn(batch, sequence, model_dim)
    value = torch.randn(batch, sequence, model_dim)
    pos_emb = torch.randn(1, 2 * sequence - 1, model_dim)
    mask = torch.ones(batch, 1, sequence, dtype=torch.bool)
    mask[:, :, -1] = False

    expected_output, expected_attention = reference(
        query, key, value, pos_emb, mask=mask
    )
    output, attention = optimized(query, key, value, pos_emb, mask=mask)

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(attention, expected_attention)
    assert output.shape == (batch, sequence, model_dim)
    assert attention.shape == (batch, heads, sequence, sequence)
    assert output.is_contiguous()
    assert attention.is_contiguous()
