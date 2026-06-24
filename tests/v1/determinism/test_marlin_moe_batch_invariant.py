# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance tests for the WNA16 Marlin MoE GEMM.

``moe_wna16_marlin_gemm`` is shared by all Marlin MoE schemes (AWQ-INT4,
GPTQ-INT4/INT8, MXFP4), so its batch-invariant path is exercised across schemes
by calling ``fused_marlin_moe`` directly.
"""

from dataclasses import dataclass

import pytest
import torch
from utils import skip_unsupported

import vllm.envs as envs
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_mxfp4_like,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    awq_marlin_quantize,
    marlin_quantize,
)
from vllm.scalar_type import ScalarType, scalar_types


@dataclass(frozen=True)
class Scheme:
    name: str
    b_type: ScalarType
    group_size: int
    dtype: torch.dtype
    ref_atol: float


SCHEMES: list[Scheme] = [
    Scheme("awq_int4", scalar_types.uint4, 128, torch.float16, 4e-2),
    Scheme("gptq_int4", scalar_types.uint4b8, 128, torch.float16, 4e-2),
    Scheme("gptq_int8", scalar_types.uint8b128, 128, torch.float16, 4e-2),
    Scheme("mxfp4", scalar_types.float4_e2m1f, 32, torch.bfloat16, 1e-1),
]


def _stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(tensors, dim=0)


def _quantize_experts(
    w: torch.Tensor, quant_type: ScalarType, group_size: int
) -> dict[str, torch.Tensor | None]:
    """Quantize per-expert weights into Marlin layout (see
    ``MarlinMoEWeightData.make`` in ``tests/kernels/moe/test_moe.py``)."""
    has_zp = quant_type in (scalar_types.uint4, scalar_types.uint8)
    k = w.shape[-1]

    w_ref_l: list[torch.Tensor] = []
    qweight_l: list[torch.Tensor] = []
    scales_l: list[torch.Tensor] = []
    zeros_l: list[torch.Tensor] = []
    g_idx_l: list[torch.Tensor] = []
    sort_l: list[torch.Tensor] = []

    for i in range(w.shape[0]):
        if quant_type == scalar_types.float4_e2m1f:
            w_ref, qweight, scales = rand_marlin_weight_mxfp4_like(w[i], group_size)
            qweight_l.append(qweight)
            scales_l.append(scales)
        elif has_zp:
            w_ref, qweight, scales, zeros = awq_marlin_quantize(
                w[i].transpose(1, 0), quant_type, group_size
            )
            qweight_l.append(qweight)
            scales_l.append(scales)
            zeros_l.append(zeros)
        else:
            test_perm = torch.randperm(k)
            w_ref, qweight, scales, g_idx, sort_indices, _ = marlin_quantize(
                w[i].transpose(1, 0), quant_type, group_size, False, test_perm
            )
            qweight_l.append(qweight)
            scales_l.append(scales)
            g_idx_l.append(g_idx)
            sort_l.append(sort_indices)
        w_ref_l.append(w_ref.T)

    return {
        "w_ref": _stack(w_ref_l),
        "qweight": _stack(qweight_l).contiguous(),
        "scales": _stack(scales_l),
        "zeros": _stack(zeros_l) if zeros_l else None,
        "g_idx": _stack(g_idx_l) if g_idx_l else None,
        "sort_indices": _stack(sort_l) if sort_l else None,
    }


# (n, k) shapes mirror the small/large matrices in MARLIN_MOE_SCENARIOS
# (tests/kernels/moe/test_moe.py). The large shape exercises the multi-tile
# K reduction that the batch-invariant ``use_full_k`` path pins.
SHAPES: list[tuple[int, int]] = [(512, 512), (1024, 2048)]


@skip_unsupported
@pytest.mark.parametrize("scheme", SCHEMES, ids=[s.name for s in SCHEMES])
@pytest.mark.parametrize("n,k", SHAPES, ids=["small", "large"])
@pytest.mark.parametrize("batch_size", [4, 16, 64, 257])
def test_marlin_moe_kernel_is_batch_invariant(
    scheme: Scheme, n: int, k: int, batch_size: int
):
    """A token's Marlin MoE output is bitwise identical regardless of batch
    size or its position in the batch, and matches a dequantized reference."""
    assert envs.VLLM_BATCH_INVARIANT

    torch.manual_seed(0)
    e, topk = 8, 2
    dtype = scheme.dtype

    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    w1q = _quantize_experts(w1, scheme.b_type, scheme.group_size)
    w2q = _quantize_experts(w2, scheme.b_type, scheme.group_size)

    token = torch.randn((1, k), device="cuda", dtype=dtype) / 10
    token_score = torch.randn((1, e), device="cuda", dtype=dtype)

    def run(a: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)
        return fused_marlin_moe(
            a,
            w1q["qweight"],
            w2q["qweight"],
            None,
            None,
            w1q["scales"],
            w2q["scales"],
            topk_weights,
            topk_ids,
            quant_type_id=scheme.b_type.id,
            global_num_experts=e,
            g_idx1=w1q["g_idx"],
            g_idx2=w2q["g_idx"],
            sort_indices1=w1q["sort_indices"],
            sort_indices2=w2q["sort_indices"],
            w1_zeros=w1q["zeros"],
            w2_zeros=w2q["zeros"],
            input_dtype=dtype,
            is_k_full=True,
        )

    with set_current_vllm_config(VllmConfig()):
        baseline = run(token, token_score)[0]
        ref_weights, ref_ids, _ = fused_topk(token, token_score, topk, False)
        ref = torch_experts(
            token,
            w1q["w_ref"],
            w2q["w_ref"],
            topk_weight=ref_weights,
            topk_ids=ref_ids,
            global_num_experts=e,
        )
        torch.testing.assert_close(baseline, ref[0], rtol=0.0, atol=scheme.ref_atol)

        filler_a = torch.randn((batch_size - 1, k), device="cuda", dtype=dtype) / 10
        filler_score = torch.randn((batch_size - 1, e), device="cuda", dtype=dtype)

        front = run(
            torch.cat([token, filler_a], dim=0),
            torch.cat([token_score, filler_score], dim=0),
        )[0]
        back = run(
            torch.cat([filler_a, token], dim=0),
            torch.cat([filler_score, token_score], dim=0),
        )[-1]

    torch.testing.assert_close(front, baseline, rtol=0.0, atol=0.0)
    torch.testing.assert_close(back, baseline, rtol=0.0, atol=0.0)
