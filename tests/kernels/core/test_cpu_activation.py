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
