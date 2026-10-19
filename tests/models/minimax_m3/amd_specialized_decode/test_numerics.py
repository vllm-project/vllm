# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.models.minimax_m3.amd_specialized_decode._stages import (
    fp32_router_stage,
    gemma_norm_stage,
    select_index_blocks,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.models.minimax_m3.amd.ops import gemma_rmsnorm
from vllm.platforms import current_platform


def _require_gfx950():
    if not current_platform.is_rocm():
        pytest.skip("ROCm-only")
    from vllm.platforms.rocm import get_cdna_version

    if get_cdna_version() != 4:
        pytest.skip("requires gfx950")


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f = actual.float().flatten()
    expected_f = expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(actual_f, expected_f, dim=0)
    relative = (actual_f - expected_f).norm() / expected_f.norm().clamp_min(1e-12)
    return float(cosine), float(relative)


def test_index_topk_is_independent_per_head_and_fixed_cardinality():
    scores = torch.zeros(4, 2, 24, 128)
    for head in range(4):
        scores[head, :, head : head + 16] = 10
    forced = torch.zeros(4, 2, 24, dtype=torch.bool)
    forced[:, :, 23] = True
    selected = select_index_blocks(scores, forced)
    assert selected.shape == (4, 2, 16)
    for head in range(4):
        chosen = set(selected[head, 0].tolist())
        assert len(chosen) == 16
        assert 23 in chosen
        assert chosen != set(selected[(head + 1) % 4, 0].tolist())


def test_index_topk_uses_max_over_block_tokens():
    scores = torch.zeros(1, 1, 17, 128)
    scores[0, 0, 16, 73] = 2
    selected = select_index_blocks(scores, torch.zeros(1, 1, 17, dtype=torch.bool))
    assert 16 in selected[0, 0].tolist()


@pytest.mark.parametrize("rows", [1, 4, 16])
def test_gemma_norm_stage_matches_native(rows: int):
    _require_gfx950()
    generator = torch.Generator(device="cuda").manual_seed(20261015 + rows)
    hidden = torch.randn(
        rows, 6144, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    weight = torch.randn(6144, dtype=torch.bfloat16, device="cuda", generator=generator)
    expected = gemma_rmsnorm(hidden, weight, 1e-6)
    actual = gemma_norm_stage(hidden, weight, 1e-6)
    cosine, relative = _metrics(actual, expected)
    assert cosine >= 0.9999
    assert relative <= 0.005


@pytest.mark.parametrize("rows", [1, 4, 16])
def test_fp32_router_stage_matches_native(rows: int):
    _require_gfx950()
    generator = torch.Generator(device="cuda").manual_seed(20261016 + rows)
    hidden = torch.randn(
        rows, 6144, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    weight = torch.randn(
        128, 6144, dtype=torch.float32, device="cuda", generator=generator
    )
    bias = torch.randn(128, dtype=torch.float32, device="cuda", generator=generator)

    logits, actual_weights, actual_ids = fp32_router_stage(
        hidden, weight, bias, top_k=4, routed_scale=2.0
    )
    expected_logits = torch.mm(hidden.float(), weight.t())
    expected_weights, expected_ids = fused_topk_bias(
        hidden_states=hidden,
        gating_output=expected_logits,
        topk=4,
        renormalize=True,
        indices_type=torch.int32,
        scoring_func="sigmoid",
        routed_scaling_factor=2.0,
        e_score_correction_bias=bias,
    )
    torch.testing.assert_close(logits, expected_logits)
    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-6, atol=1e-6)
