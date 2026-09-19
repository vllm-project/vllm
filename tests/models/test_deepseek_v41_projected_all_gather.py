# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.models.common.ops import sequence_parallel as sp_ops
from vllm.models.deepseek_v41.attention import DeepseekV4Attention


@pytest.mark.parametrize("num_tokens", [255, 256, 259])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "has_compressor,has_indexer",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_projected_gather_preserves_rows_and_optional_inputs(
    monkeypatch, num_tokens, enabled, has_compressor, has_indexer
):
    """Gather projected shards, trim padding, and retain the small-batch path."""
    torch.manual_seed(42)
    hidden = torch.randn(num_tokens, 5, dtype=torch.bfloat16)
    shard_size = (num_tokens + 3) // 4
    padded = torch.nn.functional.pad(hidden, (0, 0, 0, shard_size * 4 - num_tokens))
    weights = [
        torch.randn(5, width, dtype=torch.float32 if width == 2 else torch.bfloat16)
        for width in (3, 2, 1)
    ]
    expected = [padded.to(weight.dtype) @ weight for weight in weights]
    present = (True, has_compressor, has_indexer)
    gathers = []
    project_locally = enabled and num_tokens >= 256
    local_rows = slice(3 * shard_size, 4 * shard_size)

    def gather(projected):
        index = 3 - projected.shape[1]
        torch.testing.assert_close(projected, expected[index][local_rows])
        gathers.append(index)
        return expected[index]

    monkeypatch.setattr(sp_ops, "sp_all_gather", gather)

    def project(x):
        torch.testing.assert_close(x, padded[local_rows] if project_locally else hidden)
        return tuple(
            x.to(weight.dtype) @ weight if exists else None
            for weight, exists in zip(weights, present)
        )

    def prepare(
        hidden_states, qr, kv, qr_scale, kv_score, indexer_weights, positions, out
    ):
        assert hidden_states.shape[0] == num_tokens
        assert out.shape == (num_tokens, 1, 3)
        for actual, reference, exists in zip(
            (qr, kv_score, indexer_weights), expected, present
        ):
            if exists:
                torch.testing.assert_close(actual, reference[:num_tokens])
            else:
                assert actual is None
        out[:, 0].copy_(qr)

    attn = SimpleNamespace(
        project_before_all_gather=enabled,
        padded_heads=1,
        head_dim=3,
        _run_parallel_input_projections=project,
        _split_qkv_and_norm=lambda qr_kv: (qr_kv, None, qr_kv),
        _prepare_and_attn_fn=prepare,
        _o_proj=lambda out, positions: out[:, 0],
    )
    attn.use_projected_all_gather = MethodType(
        DeepseekV4Attention.use_projected_all_gather, attn
    )
    attn._alloc_attn_out = MethodType(DeepseekV4Attention._alloc_attn_out, attn)
    actual = DeepseekV4Attention.forward(
        attn,
        torch.arange(num_tokens),
        padded[local_rows] if project_locally else hidden,
    )
    torch.testing.assert_close(actual, expected[0][:num_tokens])
    assert gathers == (
        [i for i, exists in enumerate(present) if exists] if project_locally else []
    )
