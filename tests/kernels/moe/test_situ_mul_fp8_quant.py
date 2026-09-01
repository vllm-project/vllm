# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the persistent cp.async-pipelined SITU + block-FP8
quant kernel (``situ_and_mul_quant``).

Mirrors ``test_silu_mul_fp8_quant_deep_gemm.py`` in structure: build a reference
activation + block-FP8 quant, drive random per-row scale magnitudes, exercise
the masked contiguous layout, and compare per valid row.

The kernel computes SITU in fp32 and converts to fp8 with hardware rounding,
which torch cannot reproduce bit-exactly. It is therefore validated against the
*ideal* FP8 quantizer of the exact fp64 activation: the difference must stay
within the unavoidable FP8 block-quant error (within 1 fp8 code for ~all
elements, never off by more than 2).
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import situ_and_mul_quant
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

fp8_dtype = current_platform.fp8_dtype()
DEVICE = current_platform.device_type
FP8_MAX = torch.finfo(fp8_dtype).max

# The kernel uses fp8 e4m3 conversion, so it is CUDA-only and needs SM89+
# (Ada/Hopper). There is no Triton fallback.
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="situ_and_mul_quant persistent kernel requires CUDA SM89+ (fp8 e4m3).",
)

# Kimi-K3 baked SITU params. Only (d=3072, beta=4.0, linear_beta=25.0,
# group_size=128, fp16/bf16) routes to the persistent cp.async-pipelined kernel;
# anything else falls back to the scalar group kernel.
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
SITU_D = 3072
GROUP_SIZE = 128

# (num_tokens, num_valid_tokens). None means all rows are valid. 2048/4096 rows
# span multiple grid-stride waves of the persistent grid (GRID_DIM = 132*8).
CASES = [
    (1, None),
    (17, None),
    (128, None),
    (2048, None),
    (4096, None),
    (2048, 1500),  # DeepEP v2 contiguous padding
    (2048, 1),
    (4096, 4095),
    (512, 0),  # whole tensor is padding
]


def token_random(num_tokens: int, twod: int, dtype: torch.dtype) -> torch.Tensor:
    """Per-row random exponent so groups span many scale magnitudes."""
    base = torch.randn(num_tokens, twod, dtype=torch.float32, device=DEVICE)
    exps = torch.randint(1, 13, (num_tokens, 1), device=DEVICE).float()
    return (base * (2.0**exps)).to(dtype)


def ref_situ_block_fp8_quant(
    inp: torch.Tensor, beta: float, linear_beta: float, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ideal FP8 block-quant of the exact fp64 SITU activation.

    The kernel rounds the activation to ``scalar_t`` before quantizing (it stores
    ``(float)(scalar_t)situ(...)``), so the reference does the same before taking
    the per-group absmax.
    """
    d = inp.shape[-1] // 2
    gate, up = inp.double().chunk(2, dim=-1)
    gate_out = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up_out = linear_beta * torch.tanh(up / linear_beta) if linear_beta > 0 else up
    act = (gate_out * up_out).to(inp.dtype).float()

    m = act.shape[0]
    ng = d // group_size
    a = act.view(m, ng, group_size)
    scale = a.abs().amax(dim=2).clamp_min(1e-30) / FP8_MAX
    q = (a / scale.unsqueeze(2)).clamp(-FP8_MAX, FP8_MAX).to(fp8_dtype)
    return q.view(m, d), scale


def fp8_code_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Signed-magnitude fp8 code distance (valid for e4m3fn and e4m3fnuz)."""

    def ordered(x: torch.Tensor) -> torch.Tensor:
        x = x.view(torch.uint8).to(torch.int32)
        return torch.where((x >> 7) == 1, -(x & 0x7F), x & 0x7F)

    return (ordered(a) - ordered(b)).abs()


@pytest.mark.parametrize("num_tokens,valid", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@torch.inference_mode()
def test_situ_and_mul_quant_pipelined(
    num_tokens: int, valid: int | None, dtype: torch.dtype
) -> None:
    set_random_seed(42)
    d = SITU_D
    inp = token_random(num_tokens, 2 * d, dtype)

    # Sentinels: padding rows must keep their bytes (out) and get scale 1.0.
    out = torch.full((num_tokens, d), 17.0, device=DEVICE).to(fp8_dtype)
    scale = torch.full(
        (num_tokens, d // GROUP_SIZE), -1.0, dtype=torch.float32, device=DEVICE
    )
    out_sentinel = out.clone()
    num_valid_tokens = (
        None if valid is None else torch.tensor(valid, dtype=torch.int32, device=DEVICE)
    )

    situ_and_mul_quant(
        out,
        scale,
        inp,
        beta=SITU_BETA,
        linear_beta=SITU_LINEAR_BETA,
        group_size=GROUP_SIZE,
        num_valid_tokens=num_valid_tokens,
    )

    # Padding contract: skipped rows keep their bytes; their scale becomes 1.0.
    if valid is not None and valid < num_tokens:
        assert torch.equal(scale[valid:], torch.ones_like(scale[valid:]))
        assert torch.equal(
            out[valid:].view(torch.uint8), out_sentinel[valid:].view(torch.uint8)
        )

    rows = num_tokens if valid is None else valid
    if rows == 0:
        return

    ref_q, ref_s = ref_situ_block_fp8_quant(
        inp[:rows], SITU_BETA, SITU_LINEAR_BETA, GROUP_SIZE
    )

    assert torch.isfinite(scale[:rows]).all()
    assert not torch.isnan(out[:rows].float()).any()

    # Scales: the kernel's fp32 SITU shifts the per-group absmax only
    # marginally vs the fp64 reference.
    torch.testing.assert_close(scale[:rows], ref_s, rtol=3e-2, atol=1e-5)

    # FP8 codes: must land within 1 code of the ideal quantizer for ~all
    # elements, never off by more than 2.
    dist = fp8_code_dist(out[:rows], ref_q)
    assert dist.max().item() <= 2
    assert (dist <= 1).float().mean().item() >= 0.99


@pytest.mark.parametrize("tokens,topk", [(300, 8), (1, 4), (256, 6)])
@torch.inference_mode()
def test_situ_and_mul_quant_topk_row_bound(tokens: int, topk: int) -> None:
    """The kernel expands num_valid_tokens by topk to bound rows."""
    set_random_seed(7)
    d = SITU_D
    num_tokens = tokens * topk + 37  # padding tail past the valid rows
    inp = token_random(num_tokens, 2 * d, torch.bfloat16)

    out = torch.full((num_tokens, d), 17.0, device=DEVICE).to(fp8_dtype)
    scale = torch.full(
        (num_tokens, d // GROUP_SIZE), -1.0, dtype=torch.float32, device=DEVICE
    )
    out_sentinel = out.clone()
    num_valid_tokens = torch.tensor(tokens, dtype=torch.int32, device=DEVICE)

    situ_and_mul_quant(
        out,
        scale,
        inp,
        beta=SITU_BETA,
        linear_beta=SITU_LINEAR_BETA,
        group_size=GROUP_SIZE,
        num_valid_tokens=num_valid_tokens,
        topk=topk,
    )

    rows = tokens * topk
    assert torch.equal(scale[rows:], torch.ones_like(scale[rows:]))
    assert torch.equal(
        out[rows:].view(torch.uint8), out_sentinel[rows:].view(torch.uint8)
    )

    ref_q, ref_s = ref_situ_block_fp8_quant(
        inp[:rows], SITU_BETA, SITU_LINEAR_BETA, GROUP_SIZE
    )
    torch.testing.assert_close(scale[:rows], ref_s, rtol=3e-2, atol=1e-5)
    assert fp8_code_dist(out[:rows], ref_q).max().item() <= 2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@torch.inference_mode()
def test_situ_and_mul_quant_pipelined_deterministic(dtype: torch.dtype) -> None:
    """Two runs on the same input must be byte-identical."""
    set_random_seed(0)
    d = SITU_D
    num_tokens = 2048
    inp = token_random(num_tokens, 2 * d, dtype)

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty(num_tokens, d, dtype=fp8_dtype, device=DEVICE)
        scale = torch.empty(
            num_tokens, d // GROUP_SIZE, dtype=torch.float32, device=DEVICE
        )
        situ_and_mul_quant(
            out,
            scale,
            inp,
            beta=SITU_BETA,
            linear_beta=SITU_LINEAR_BETA,
            group_size=GROUP_SIZE,
        )
        return out, scale

    out1, s1 = run()
    out2, s2 = run()
    assert torch.equal(out1.view(torch.uint8), out2.view(torch.uint8))
    assert torch.equal(s1, s2)
