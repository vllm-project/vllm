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


@pytest.mark.parametrize("sequence", [2, 8, 130])
def test_relative_attention_matches_reference(
    tp1_cpu, monkeypatch, sequence: int
) -> None:
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
    # Key mask only. Queries past input_lengths still attend to valid keys.
    valid_keys = max(sequence - 5, 1)
    mask = torch.ones(batch, 1, sequence, dtype=torch.bool)
    mask[:, :, valid_keys:] = False

    expected_output, expected_attention = reference(
        query, key, value, pos_emb, mask=mask
    )
    original = torch.matmul
    tiles = {"n": 0}

    def tracked(left, right, *args, **kwargs):
        if torch.is_tensor(left) and left.dim() == 4:
            query_tile = left.size(-2)
            if query_tile > 128:
                raise RuntimeError(f"score tile has {query_tile} queries")
            tiles["n"] += 1
        return original(left, right, *args, **kwargs)

    fill_values: list[float] = []
    original_fill = torch.Tensor.masked_fill

    def tracked_fill(tensor, mask, value):
        if isinstance(value, (int, float)):
            fill_values.append(float(value))
        return original_fill(tensor, mask, value)

    monkeypatch.setattr(torch, "matmul", tracked)
    monkeypatch.setattr(torch.Tensor, "masked_fill", tracked_fill)
    output, attention = optimized(
        query, key, value, pos_emb, mask=mask, need_weights=True
    )

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(attention, expected_attention)
    assert output.shape == (batch, sequence, model_dim)
    assert attention.shape == (batch, heads, sequence, sequence)
    assert torch.count_nonzero(attention[..., valid_keys:]) == 0
    kept = attention[..., :valid_keys].sum(dim=-1)
    torch.testing.assert_close(kept, torch.ones_like(kept))
    assert float("-inf") in fill_values
    assert 0.0 in fill_values
    assert tiles["n"] >= 1
    assert output.is_contiguous()
    assert attention.is_contiguous()


def test_relative_attention_skips_weight_matrix(tp1_cpu) -> None:
    torch.manual_seed(3)
    batch, heads, model_dim, sequence = 2, 4, 32, 6
    optimized = RelPosMultiHeadAttention(heads, model_dim)
    query = torch.randn(batch, sequence, model_dim)
    pos_emb = torch.randn(1, 2 * sequence - 1, model_dim)
    output, attention = optimized(
        query, query, query, pos_emb, mask=None, need_weights=False
    )
    assert attention is None
    assert output.shape == (batch, sequence, model_dim)


def test_firered_fully_masked_keys_are_zero(tp1_cpu) -> None:
    torch.manual_seed(5)
    batch, heads, model_dim, sequence = 2, 4, 32, 6
    optimized = RelPosMultiHeadAttention(heads, model_dim)
    query = torch.randn(batch, sequence, model_dim)
    pos_emb = torch.randn(1, 2 * sequence - 1, model_dim)
    mask = torch.zeros(batch, 1, sequence, dtype=torch.bool)
    output, attention = optimized(
        query, query, query, pos_emb, mask=mask, need_weights=True
    )
    assert attention is not None
    assert torch.isfinite(attention).all()
    assert torch.count_nonzero(attention) == 0
    assert torch.isfinite(output).all()
