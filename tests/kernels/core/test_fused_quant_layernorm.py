# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8,
)

DTYPES = [torch.bfloat16, torch.float]
QUANT_DTYPES = [torch.int8, torch.float8_e4m3fn]
VEC_HIDDEN_SIZES = [1024, 1025, 1027, 1029]
# Avoid combinatorial explosion with full Cartesian product
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, i) for i in [1, 64, *VEC_HIDDEN_SIZES, 5120, 5137]],
    *[(2048, i) for i in [1, 64, *VEC_HIDDEN_SIZES, 5137]],
    *[(4096, i) for i in [1, 64, 5137]],
]

ADD_RESIDUAL = [False, True]
SCALE_UBS = [True, False]
GROUP_SIZES = [None, [1, 64], [1, 128]]
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]

EPS = 1e-6

## Helpers


def as_float32_tensor(x: float | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device="cuda")


def ref_rms_norm(
    rms_norm_layer: RMSNorm, x: torch.Tensor, residual: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if residual is not None:
        residual = residual.clone()
        out, residual = rms_norm_layer.forward_native(x, residual)
    else:
        out = rms_norm_layer.forward_native(x)

    return out, residual


def ref_dynamic_per_token_or_block_quant(
    rms_norm_layer: RMSNorm,
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    residual: torch.Tensor | None,
    scale_ub: torch.Tensor | None,
    group_size: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if scale_ub is not None:
        assert quant_dtype == torch.float8_e4m3fn

    # Norm
    torch_out, residual = ref_rms_norm(rms_norm_layer, x, residual)

    # Quant
    if group_size is not None:
        if quant_dtype == torch.float8_e4m3fn:
            torch_out, scales = per_token_group_quant_fp8(
                torch_out, group_size=group_size[1], use_ue8m0=False
            )
        else:
            assert quant_dtype == torch.int8
            torch_out, scales = per_token_group_quant_int8(
                torch_out, group_size=group_size[1]
            )
    else:
        if quant_dtype == torch.float8_e4m3fn:
            torch_out, scales = ops.scaled_fp8_quant(
                torch_out, scale_ub=scale_ub, use_per_token_if_dynamic=True
            )
        else:
            assert quant_dtype == torch.int8
            torch_out, scales, _ = ops.scaled_int8_quant(torch_out)

    return torch_out, scales, residual


def ref_impl(
    rms_norm_layer: RMSNorm,
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    residual: torch.Tensor | None,
    scale_ub: torch.Tensor | None,
    group_size: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return ref_dynamic_per_token_or_block_quant(
        rms_norm_layer, x, quant_dtype, residual, scale_ub, group_size
    )


def ops_dynamic_per_token_or_block_quant(
    weight: torch.Tensor,
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    residual: torch.Tensor | None,
    scale_ub: torch.Tensor | None,
    group_size: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if residual is not None:
        residual = residual.clone()
    if group_size is not None:
        out, scales = ops.rms_norm_per_block_quant(
            x, weight, EPS, quant_dtype, group_size, scale_ub, residual, True
        )
        scales = scales.contiguous()
    else:
        out, scales = ops.rms_norm_dynamic_per_token_quant(
            x, weight, EPS, quant_dtype, scale_ub, residual
        )
    return out, scales, residual


def ops_impl(
    weight: torch.Tensor,
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    residual: torch.Tensor | None,
    scale_ub: torch.Tensor | None,
    group_size: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return ops_dynamic_per_token_or_block_quant(
        weight, x, quant_dtype, residual, scale_ub, group_size
    )


@pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("has_scale_ub", SCALE_UBS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    has_scale_ub: bool,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    group_size: list[int] | None,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    if group_size is not None and hidden_size % group_size[1] != 0:
        # skip
        return

    if group_size is not None and has_scale_ub:
        # blockwise baseline doesn't support scale_ub
        return

    if has_scale_ub and quant_dtype != torch.float8_e4m3fn:
        # skip
        return

    layer = RMSNorm(hidden_size, EPS).to(dtype=dtype)

    # Make weights
    layer.weight.data.normal_(mean=1.0, std=0.1)

    # Make inputs
    scale = 1 / (hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
    residual = torch.randn_like(x) * scale if add_residual else None
    if has_scale_ub:
        rms_x, _ = ref_rms_norm(layer, x, residual)
        scale_ub = torch.mean(rms_x).to(dtype=torch.float32, device="cuda")
    else:
        scale_ub = None

    ref_out, ref_scales, ref_residual = ref_impl(
        layer, x, quant_dtype, residual, scale_ub, group_size
    )
    ops_out, ops_scales, ops_residual = ops_impl(
        layer.weight, x, quant_dtype, residual, scale_ub, group_size
    )

    assert ref_out.dtype == quant_dtype
    assert ops_out.dtype == quant_dtype
    if quant_dtype == torch.int8:
        assert torch.allclose(ref_scales, ops_scales, atol=1e-6)
        # big atol to account for round-off errors.
        assert torch.allclose(ref_out, ops_out, atol=1)
    else:
        assert torch.allclose(ref_scales, ops_scales)
        a = ref_out.to(dtype=torch.float32)
        b = ops_out.to(dtype=torch.float32)
        ok = torch.allclose(a, b, atol=1e-6)
        if not ok:
            # fallback: compare dequantized values with relaxed tolerance
            if group_size is None:
                a_deq = a * ref_scales.view(-1, 1)
                b_deq = b * ops_scales.view(-1, 1)
            else:
                a_deq = a * ref_scales.repeat_interleave(group_size[1], dim=1)
                b_deq = b * ops_scales.repeat_interleave(group_size[1], dim=1)
            # NOTE: It is possible that some future test cases trigger this
            # max diff due to precision issues. If such an error is
            # encountered, it's recommended to inspect the differences between
            # all corresponding elements from each tensor (e.g. by looping over
            # them) and checking how many the max diff error shows up on (just
            # a few bad elements should still be considered acceptable).
            ok = torch.allclose(a_deq, b_deq, rtol=5e-2, atol=5e-2)
        assert ok
    if add_residual:
        assert torch.allclose(ref_residual, ops_residual)

    output = torch.empty_like(x, dtype=quant_dtype)
    scales = torch.empty(
        (x.numel() // x.shape[-1], 1), device=x.device, dtype=torch.float32
    )

    opcheck(
        torch.ops._C.rms_norm_dynamic_per_token_quant,
        (output, x, layer.weight, scales, 1e-5, scale_ub, residual),
    )
