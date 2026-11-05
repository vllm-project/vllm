# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 sequence-parallel prefill kernels, bit for bit against the kernels
they replace."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm sequence-parallel prefill kernels", allow_module_level=True)

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    _mxfp8_e4m3_quantize_triton,
)
from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (  # noqa: E402
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
)
from vllm.models.minimax_m3.amd.ops.sp_norm_quant import (  # noqa: E402
    MOE_MIN_SCALE_BYTE,
    gemma_rmsnorm_mx,
)
from vllm.models.minimax_m3.amd.ops.sp_pack import (  # noqa: E402
    mxfp8_quant_pack,
    mxfp8_unpack,
)


def _rows(m: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, n, device="cuda", generator=g)
    x = x * torch.rand(m, 1, device="cuda", generator=g) * 40
    x[1] = 0  # an all-zero row: the scale floor
    x[2, :32] = 2.0**-36  # a tiny block
    x[3, :64] = 3e4  # a large block
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("rows", [64, 1032])
@pytest.mark.parametrize("world", [4])
def test_quant_pack_unpack(rows: int, world: int):
    d = 2048
    x = _rows(world * rows, d, 0)
    codes, scales = _mxfp8_e4m3_quantize_triton(x)
    packed = mxfp8_quant_pack(x)
    assert torch.equal(packed[:, :d], codes.view(torch.uint8))
    assert torch.equal(packed[:, d:], scales)
    got_codes, got_scales = mxfp8_unpack(packed, world, rows, d)
    blocks = packed.view(world, rows, -1).transpose(0, 1)
    assert torch.equal(got_codes.view(torch.uint8), blocks[:, :, :d].reshape(rows, -1))
    assert torch.equal(got_scales, blocks[:, :, d:].reshape(rows, -1))


@pytest.mark.parametrize("rows", [7, 2048])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("moe", [False, True])
def test_rmsnorm_mx(rows: int, with_residual: bool, moe: bool):
    """The linear input = gemma norm + ``mxfp8_e4m3_quantize``; the MoE input =
    gemma norm + aiter's ``per_1x32_mx_quant`` (the AITER MXFP8 MoE's quant)."""
    hidden = 6144
    x = _rows(rows, hidden, 1)
    residual = _rows(rows, hidden, 2) * 4 if with_residual else None
    weight = torch.randn(hidden, device="cuda") * 0.3
    if residual is None:
        norm_ref, res_ref = gemma_rmsnorm(x, weight, 1e-6), x
    else:
        norm_ref, res_ref = gemma_fused_add_rmsnorm(x, residual, weight, 1e-6)
    if moe:
        from aiter import dtypes
        from aiter.ops.quant import per_1x32_mx_quant_hip

        q_ref, s_ref = per_1x32_mx_quant_hip(
            norm_ref,
            scale=None,
            quant_dtype=dtypes.fp8,
            scale_type=dtypes.fp8_e8m0,
            shuffle=False,
        )
    else:
        q_ref, s_ref = _mxfp8_e4m3_quantize_triton(norm_ref)
    norm, res, q, s = gemma_rmsnorm_mx(
        x,
        residual,
        weight,
        1e-6,
        min_scale_byte=MOE_MIN_SCALE_BYTE if moe else 0,
        store_out=moe,
    )
    if moe:
        assert torch.equal(norm, norm_ref)
    else:
        assert norm is None
    assert torch.equal(res, res_ref)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s, s_ref.view(torch.uint8))
