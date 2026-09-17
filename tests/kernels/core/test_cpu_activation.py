# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from tests.kernels.utils import opcheck
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

from vllm.model_executor.layers.activation import (
    GELU,
    FastGELU,
    GeluAndMul,
    NewGELU,
    QuickGELU,
    SiluAndMul,
)

DTYPES = [torch.bfloat16, torch.float32]
NUM_TOKENS = [7, 83]
D = [512, 2048]
SEEDS = [0]


@pytest.mark.parametrize(
    ("activation_cls", "fn"),
    [
        (SiluAndMul, torch.ops._C.silu_and_mul),
        (GeluAndMul, torch.ops._C.gelu_and_mul),
        (GeluAndMul, torch.ops._C.gelu_tanh_and_mul),
    ],
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_cpu_act_and_mul(
    default_vllm_config,
    activation_cls: type[torch.nn.Module],
    fn: object,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    set_random_seed(seed)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)

    layer = activation_cls()
    out = layer(x)
    ref_out = layer.forward_native(x)

    torch.testing.assert_close(
        out, ref_out, atol=get_default_atol(out), rtol=get_default_rtol(out)
    )

    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    raw_out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    opcheck(fn, (raw_out, x))


@pytest.mark.parametrize(
    ("activation_cls", "fn", "op_args"),
    [
        (NewGELU, torch.ops._C.gelu_new, ()),
        (FastGELU, torch.ops._C.gelu_fast, ()),
        (QuickGELU, torch.ops._C.gelu_quick, ()),
        pytest.param(
            GELU,
            getattr(torch.ops._C, "activation_lut_bf16", None),
            ("gelu",),
            marks=pytest.mark.skipif(
                current_platform.get_cpu_architecture() != CpuArchEnum.ARM,
                reason="activation_lut_bf16 is only built on Arm CPU",
            ),
        ),
    ],
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_cpu_unary_activation(
    default_vllm_config,
    activation_cls: type[torch.nn.Module],
    fn: object,
    op_args: tuple[str, ...],
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    set_random_seed(seed)
    x = torch.randn(num_tokens, d, dtype=dtype)
    layer = activation_cls()
    out = layer(x)
    ref_out = layer.forward_native(x)
    torch.testing.assert_close(
        out, ref_out, atol=get_default_atol(out), rtol=get_default_rtol(out)
    )
    # gelu with activation_lut_bf16 only makes sense for BF16
    if not (activation_cls is GELU and dtype != torch.bfloat16):
        raw_out = torch.empty_like(x)
        opcheck(fn, (raw_out, x, *op_args))


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_cpu_gelu_tanh_and_mul(
    default_vllm_config,
    dtype: torch.dtype,
) -> None:
    gate = torch.tensor(
        [
            [
                -12.0,
                -10.0,
                -9.01,
                -5.0,
                -2.0,
                -1.0,
                -0.0,
                0.0,
                0.5,
                1.0,
                2.0,
                5.0,
                9.01,
                10.0,
                12.0,
                11.0,
            ],
            [
                -7.5,
                -4.5,
                -3.0,
                -1.5,
                -0.75,
                -0.25,
                0.25,
                0.75,
                1.5,
                3.0,
                4.5,
                7.5,
                -11.0,
                11.0,
                8.75,
                -8.75,
            ],
        ],
        dtype=dtype,
    )
    val = torch.tensor(
        [
            [
                0.25,
                -0.5,
                0.75,
                -1.0,
                1.25,
                -1.5,
                1.75,
                -2.0,
                2.25,
                -2.5,
                2.75,
                -3.0,
                3.25,
                -3.5,
                3.75,
                -4.0,
            ],
            [
                -0.4,
                0.6,
                -0.8,
                1.0,
                -1.2,
                1.4,
                -1.6,
                1.8,
                -2.0,
                2.2,
                -2.4,
                2.6,
                -2.8,
                3.0,
                -3.2,
                3.4,
            ],
        ],
        dtype=dtype,
    )

    x = torch.cat((val, gate), dim=-1).contiguous()
    kernel_out = torch.empty_like(val)
    torch.ops._C.gelu_tanh_and_mul(kernel_out, x)

    torch_ref = torch.nn.functional.gelu(val, approximate="tanh") * gate

    atol = get_default_atol(kernel_out)
    rtol = get_default_rtol(kernel_out)
    torch.testing.assert_close(kernel_out, torch_ref, atol=atol, rtol=rtol)
