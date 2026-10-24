# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance tests for the WNA16 Marlin MoE GEMM.

``moe_wna16_marlin_gemm`` is shared by all Marlin MoE schemes (AWQ-INT4,
GPTQ-INT4/INT8, MXFP4, NVFP4), so its batch-invariant path is exercised across
schemes by calling ``fused_marlin_moe`` directly.
"""

from dataclasses import dataclass

import pytest
import torch
from utils import skip_if_not_cuda

from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_mxfp4_like,
    rand_marlin_weight_nvfp4_like,
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
    Scheme("nvfp4", scalar_types.float4_e2m1f, 16, torch.bfloat16, 1e-1),
]


def _stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(tensors, dim=0)


def _quantize_experts(
    w: torch.Tensor, quant_type: ScalarType, group_size: int
) -> dict[str, torch.Tensor | None]:
    """Quantize per-expert weights into Marlin layout (see
    ``MarlinMoEWeightData.make`` in ``tests/kernels/moe/test_moe.py``)."""
    has_zp = quant_type in (scalar_types.uint4, scalar_types.uint8)

    w_ref_l: list[torch.Tensor] = []
    qweight_l: list[torch.Tensor] = []
    scales_l: list[torch.Tensor] = []
    global_scale_l: list[torch.Tensor] = []
    zeros_l: list[torch.Tensor] = []

    for i in range(w.shape[0]):
        if quant_type == scalar_types.float4_e2m1f and group_size == 16:
            w_ref, qweight, scales, global_scale = rand_marlin_weight_nvfp4_like(
                w[i], group_size
            )
            qweight_l.append(qweight)
            scales_l.append(scales)
            global_scale_l.append(global_scale)
        elif quant_type == scalar_types.float4_e2m1f:
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
            w_ref, qweight, scales = marlin_quantize(
                w[i].transpose(1, 0), quant_type, group_size
            )
            qweight_l.append(qweight)
            scales_l.append(scales)
        w_ref_l.append(w_ref.T)

    return {
        "w_ref": _stack(w_ref_l),
        "qweight": _stack(qweight_l).contiguous(),
        "scales": _stack(scales_l),
        "global_scale": _stack(global_scale_l) if global_scale_l else None,
        "zeros": _stack(zeros_l) if zeros_l else None,
    }


# (n, k, e, topk). small/large mirror MARLIN_MOE_SCENARIOS
# (tests/kernels/moe/test_moe.py) and exercise the multi-tile K reduction that
# the batch-invariant ``use_full_k`` path pins. xlarge approximates a real NVFP4
# MoE layer.
SHAPES: list[tuple[int, int, int, int]] = [
    (512, 512, 8, 2),
    (1024, 2048, 8, 2),
    (1024, 4096, 64, 8),
]


@skip_if_not_cuda
@pytest.mark.parametrize("scheme", SCHEMES, ids=[s.name for s in SCHEMES])
@pytest.mark.parametrize("n,k,e,topk", SHAPES, ids=["small", "large", "xlarge"])
@pytest.mark.parametrize("batch_size", [4, 16, 64, 257])
def test_marlin_moe_kernel_is_batch_invariant(
    scheme: Scheme, n: int, k: int, e: int, topk: int, batch_size: int
):
    """Every token's Marlin MoE output is bitwise identical regardless of batch
    size or its position in the batch, and matches a dequantized reference."""
    torch.manual_seed(0)
    dtype = scheme.dtype

    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    w1q = _quantize_experts(w1, scheme.b_type, scheme.group_size)
    w2q = _quantize_experts(w2, scheme.b_type, scheme.group_size)

    tokens = torch.randn((batch_size, k), device="cuda", dtype=dtype) / 10
    scores = torch.randn((batch_size, e), device="cuda", dtype=dtype)

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
            w1_zeros=w1q["zeros"],
            w2_zeros=w2q["zeros"],
            global_scale1=w1q["global_scale"],
            global_scale2=w2q["global_scale"],
            input_dtype=dtype,
        )

    with set_current_vllm_config(VllmConfig()):
        alone = torch.cat(
            [run(tokens[i : i + 1], scores[i : i + 1]) for i in range(batch_size)]
        )
        ref_weights, ref_ids, _ = fused_topk(tokens, scores, topk, False)
        ref = torch_experts(
            tokens,
            w1q["w_ref"],
            w2q["w_ref"],
            topk_weight=ref_weights,
            topk_ids=ref_ids,
            global_num_experts=e,
        )
        torch.testing.assert_close(alone, ref, rtol=0.0, atol=scheme.ref_atol)

        batched = run(tokens, scores)
        flipped = run(tokens.flip(0), scores.flip(0)).flip(0)

    torch.testing.assert_close(batched, alone, rtol=0.0, atol=0.0)
    torch.testing.assert_close(flipped, alone, rtol=0.0, atol=0.0)
