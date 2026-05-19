# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.activation import (
    FastGELU,
    FatreluAndMul,
    GeluAndMul,
    MulAndSilu,
    NewGELU,
    QuickGELU,
    SiluAndMul,
    SwigluOAIAndMul,
    SwigluStepAndMul,
    swiglustep_and_mul_triton,
)
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 13824]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
INT32_MAX = torch.iinfo(torch.int32).max


@pytest.mark.parametrize(
    "activation",
    [
        "silu_and_mul",
        "mul_and_silu",
        "gelu",
        "gelu_tanh",
        "fatrelu",
        "swigluoai_and_mul",
        "swiglustep_and_mul",
    ],
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_act_and_mul(
    default_vllm_config,
    activation: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)
    if activation == "silu_and_mul":
        layer = SiluAndMul(compile_native=False)
        fn = torch.ops._C.silu_and_mul
    if activation == "mul_and_silu":
        layer = MulAndSilu()
        fn = torch.ops._C.mul_and_silu
    elif activation == "gelu":
        layer = GeluAndMul(approximate="none")
        fn = torch.ops._C.gelu_and_mul
    elif activation == "gelu_tanh":
        layer = GeluAndMul(approximate="tanh")
        fn = torch.ops._C.gelu_tanh_and_mul
    elif activation == "fatrelu":
        threshold = random.uniform(0, 1)
        layer = FatreluAndMul(threshold)
        fn = torch.ops._C.fatrelu_and_mul
    elif activation == "swigluoai_and_mul":
        layer = SwigluOAIAndMul()
        fn = torch.ops._C.swigluoai_and_mul
    elif activation == "swiglustep_and_mul":
        layer = SwigluStepAndMul()
        fn = swiglustep_and_mul_triton
    out = layer(x)
    ref_out = layer.forward_native(x)
    if activation in ["swigluoai_and_mul", "swiglustep_and_mul"]:
        rtol = {
            # For fp16, change the relative tolerance from 1e-3 to 2e-3
            torch.float16: 2e-3,
            torch.bfloat16: 2e-2,
            torch.float: 1.3e-6,
        }

        def _get_rtol(output) -> float:
            return rtol[output.dtype]

        torch.testing.assert_close(
            out, ref_out, atol=get_default_atol(out), rtol=_get_rtol(out)
        )
    else:
        # The SiluAndMul, MulAndSilu, GELU and FatReLU implementations are
        # equivalent to the native PyTorch implementations, so we can do exact
        # comparison.
        torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)

    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    if activation == "fatrelu":
        opcheck(fn, (out, x, threshold))
    elif activation == "swigluoai_and_mul":
        opcheck(fn, (out, x, layer.alpha, layer.limit))
    elif activation != "swiglustep_and_mul":
        opcheck(fn, (out, x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_silu_and_mul_large_row_offset_over_int32(default_vllm_config) -> None:
    num_tokens = 149910
    d = 14336
    input_cols = 2 * d
    dtype = torch.float16
    device = torch.device("cuda:0")

    assert num_tokens * input_cols == 4_298_219_520
    assert num_tokens * input_cols > INT32_MAX

    element_size = torch.empty((), dtype=dtype).element_size()
    required_bytes = num_tokens * (input_cols + d) * element_size
    reserve_bytes = 1 << 30

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < required_bytes + reserve_bytes:
        pytest.skip(
            "Not enough free CUDA memory for large activation overflow test: "
            f"need {(required_bytes + reserve_bytes) / 2**30:.1f} GiB, "
            f"have {free_bytes / 2**30:.1f} GiB"
        )

    x = torch.empty((num_tokens, input_cols), dtype=dtype, device=device)
    out = torch.empty((num_tokens, d), dtype=dtype, device=device)

    idx = torch.arange(d, dtype=torch.float32, device=device)
    gate = ((idx % 251) - 125) / 32
    up = ((idx % 127) - 63) / 16
    x[-1, :d].copy_(gate.to(dtype))
    x[-1, d:].copy_(up.to(dtype))

    # The old 32-bit row offset for the final token wrapped into this window.
    wrapped_offset = ((num_tokens - 1) * input_cols) & 0xFFFFFFFF
    assert wrapped_offset != (num_tokens - 1) * input_cols
    x.flatten()[wrapped_offset : wrapped_offset + 2 * d].fill_(0.25)

    torch.ops._C.silu_and_mul(out, x)

    ref_out = torch.nn.functional.silu(x[-1, :d]) * x[-1, d:]
    torch.testing.assert_close(out[-1], ref_out, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "activation",
    [
        (FastGELU, torch.ops._C.gelu_fast),
        (NewGELU, torch.ops._C.gelu_new),
        (QuickGELU, torch.ops._C.gelu_quick),
    ],
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_activation(
    default_vllm_config,
    activation: type[torch.nn.Module],
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, d, dtype=dtype)
    layer = activation[0]()
    fn = activation[1]
    out = layer(x)
    ref_out = layer.forward_native(x)
    torch.testing.assert_close(
        out, ref_out, atol=get_default_atol(out), rtol=get_default_rtol(out)
    )

    out = torch.empty_like(x)
    opcheck(fn, (out, x))
