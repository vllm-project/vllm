# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from aiter.fused_moe import (
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.enum import ActivationType, QuantType

from tests.kernels.quantization.nvfp4_utils import (
    quantize_nvfp4_weight_for_moe,
    shuffle_nvfp4_weight_for_flydsl,
)
from vllm.kernels.flydsl.nvfp4_moe_2stages import (
    nvfp4_moe_stage1,
    nvfp4_moe_stage2,
)

DTYPE = torch.bfloat16
MODEL_DIMS = [128, 192, 256, 512, 1024, 1536, 2048]
INTER_DIMS = [128, 192, 256, 512, 1024, 1536, 2048]
TOKENS = [1, 16, 234]


@dataclass
class Nvfp4MoeCase:
    hidden: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    w1_packed_flydsl: torch.Tensor
    w1_scale_flydsl: torch.Tensor
    w1_global_scale: torch.Tensor
    w1_bf16_qdq: torch.Tensor
    w2_packed_flydsl: torch.Tensor
    w2_scale_flydsl: torch.Tensor
    w2_global_scale: torch.Tensor
    w2_bf16_qdq: torch.Tensor
    experts: int
    model_dim: int
    inter_dim: int
    topk: int


def _assert_close(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected_f.abs().clamp_min(1e-12)).max().item()
    print(f"{label}: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}")
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _make_case(
    tokens: int = 16,
    experts: int = 8,
    model_dim: int = 256,
    inter_dim: int = 128,
    topk: int = 2,
) -> Nvfp4MoeCase:
    torch.manual_seed(123)
    device = torch.device("cuda")

    hidden = (
        torch.randn((tokens, model_dim), device=device, dtype=DTYPE) * 0.2
    ).contiguous()
    router_logits = torch.randn((tokens, experts), device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=1)
    topk_weights = torch.softmax(topk_weights, dim=1).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()

    w1 = (
        torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=DTYPE)
        * 0.1
    ).contiguous()
    w2 = (
        torch.randn((experts, model_dim, inter_dim), device=device, dtype=DTYPE) * 0.1
    ).contiguous()

    w1_packed, w1_scale, w1_global_scale, w1_bf16_qdq = quantize_nvfp4_weight_for_moe(
        w1
    )
    w2_packed, w2_scale, w2_global_scale, w2_bf16_qdq = quantize_nvfp4_weight_for_moe(
        w2
    )

    w1_packed_flydsl = shuffle_nvfp4_weight_for_flydsl(w1_packed).contiguous()
    w2_packed_flydsl = shuffle_nvfp4_weight_for_flydsl(w2_packed).contiguous()
    w1_packed_flydsl.is_shuffled = True
    w2_packed_flydsl.is_shuffled = True

    return Nvfp4MoeCase(
        hidden=hidden,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w1_packed_flydsl=w1_packed_flydsl,
        w1_scale_flydsl=w1_scale.permute(0, 2, 1).contiguous(),
        w1_global_scale=w1_global_scale.contiguous(),
        w1_bf16_qdq=w1_bf16_qdq,
        w2_packed_flydsl=w2_packed_flydsl,
        w2_scale_flydsl=w2_scale.permute(0, 2, 1).contiguous(),
        w2_global_scale=w2_global_scale.contiguous(),
        w2_bf16_qdq=w2_bf16_qdq,
        experts=experts,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
    )


def _has_invalid_nvfp4_flydsl_weight_layout(model_dim: int, inter_dim: int) -> bool:
    w1_n_out = 2 * inter_dim
    w1_packed_k = model_dim // 2
    w2_n_out = model_dim
    w2_packed_k = inter_dim // 2
    return (
        w1_n_out % 16 != 0
        or w1_packed_k % 32 != 0
        or w2_n_out % 16 != 0
        or w2_packed_k % 32 != 0
    )


def _make_case_or_assert_invalid_layout(
    tokens: int,
    experts: int,
    model_dim: int,
    inter_dim: int,
    topk: int,
) -> Nvfp4MoeCase:
    if _has_invalid_nvfp4_flydsl_weight_layout(model_dim, inter_dim):
        with pytest.raises(ValueError, match="NVFP4 FlyDSL weight requires"):
            _make_case(
                tokens=tokens,
                experts=experts,
                model_dim=model_dim,
                inter_dim=inter_dim,
                topk=topk,
            )
        pytest.skip("unsupported case")

    return _make_case(
        tokens=tokens,
        experts=experts,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
    )


def _route(case: Nvfp4MoeCase, block_m: int):
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        case.topk_ids,
        case.topk_weights,
        case.experts,
        case.model_dim,
        DTYPE,
        block_m,
    )
    num_valid_ids = num_valid_ids[:1].contiguous()
    valid_count = int(num_valid_ids.item())
    valid_blocks = (valid_count + block_m - 1) // block_m
    valid_elems = valid_blocks * block_m
    return (
        sorted_ids[:valid_elems].contiguous(),
        sorted_weights[:valid_elems].contiguous(),
        sorted_expert_ids[:valid_blocks].contiguous(),
        num_valid_ids,
    )


TILE_M = 32
TILE_N = 64
TILE_K = 64


@pytest.mark.parametrize("model_dim", MODEL_DIMS)
@pytest.mark.parametrize("inter_dim", INTER_DIMS)
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("experts", [128])
@pytest.mark.parametrize("tokens", TOKENS)
def test_stage1_correctness(
    tokens: int,
    experts: int,
    model_dim: int,
    inter_dim: int,
    topk: int,
) -> None:
    case = _make_case_or_assert_invalid_layout(
        tokens=tokens,
        experts=experts,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
    )

    if case.model_dim // TILE_K % 2:
        pytest.skip("stage-one requires an even number of K tiles")
    sorted_ids, _, sorted_expert_ids, num_valid_ids = _route(case, TILE_M)
    out = torch.empty(
        (case.hidden.shape[0], case.topk, case.inter_dim),
        dtype=DTYPE,
        device=case.hidden.device,
    )
    ref = torch_moe_stage1(
        case.hidden,
        case.w1_bf16_qdq,
        case.w2_bf16_qdq,
        case.topk_weights,
        case.topk_ids,
        dtype=DTYPE,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
        doweight=False,
    )

    out = nvfp4_moe_stage1(
        case.hidden,
        case.w1_packed_flydsl,
        case.w1_scale_flydsl,
        case.w1_global_scale,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk=case.topk,
        inter_dim=case.inter_dim,
        tile_m=TILE_M,
        tile_n=TILE_N,
        tile_k=TILE_K,
        output=out,
    )
    torch.accelerator.synchronize()
    _assert_close("stage1", out, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("model_dim", MODEL_DIMS)
@pytest.mark.parametrize("inter_dim", INTER_DIMS)
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("experts", [128])
@pytest.mark.parametrize("tokens", TOKENS)
def test_stage2_correctness(
    tokens: int,
    experts: int,
    model_dim: int,
    inter_dim: int,
    topk: int,
) -> None:
    case = _make_case_or_assert_invalid_layout(
        tokens=tokens,
        experts=experts,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
    )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = _route(case, TILE_M)
    torch.manual_seed(456)
    inter_states = (
        torch.randn(
            (case.hidden.shape[0], case.topk, case.inter_dim),
            device=case.hidden.device,
            dtype=DTYPE,
        )
        * 0.2
    ).contiguous()
    out = torch.zeros(
        (case.hidden.shape[0], case.model_dim),
        dtype=DTYPE,
        device=case.hidden.device,
    )
    ref = torch_moe_stage2(
        inter_states,
        case.w1_bf16_qdq,
        case.w2_bf16_qdq,
        case.topk_weights,
        case.topk_ids,
        dtype=DTYPE,
        quant_type=QuantType.No,
        doweight=True,
    )

    out = torch.zeros_like(ref)
    nvfp4_moe_stage2(
        inter_states,
        case.w2_packed_flydsl,
        case.w2_scale_flydsl,
        case.w2_global_scale,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk=case.topk,
        model_dim=case.model_dim,
        tile_m=TILE_M,
        tile_n=TILE_N,
        tile_k=TILE_K,
        output=out,
        sorted_weights=sorted_weights,
    )
    torch.accelerator.synchronize()
    _assert_close("stage2", out, ref, atol=1.5e-1, rtol=1.5e-1)
