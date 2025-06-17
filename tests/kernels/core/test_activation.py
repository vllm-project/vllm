# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.activation import (FastGELU, FatreluAndMul,
                                                   GeluAndMul, MulAndSilu,
                                                   NewGELU, QuickGELU,
                                                   SiluAndMul)
from vllm.platforms import current_platform


def ref_batched_silu_mul(x, out, valid_tokens_array):
    """
    Reference implementation of batched silu_and_mul
    """
    valid_tokens_array = valid_tokens_array.to("cpu")
    batch_size = x.size(0)
    for b in range(batch_size):
        # num valid tokens
        n = valid_tokens_array[b]
        if n == 0:
            continue
        torch.ops._C.silu_and_mul(out[b, :n, :], x[b, :n, :])


DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 13824]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


@pytest.mark.parametrize(
    "activation",
    ["silu_and_mul", "mul_and_silu", "gelu", "gelu_tanh", "fatrelu"])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_act_and_mul(
    activation: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)
    if activation == "silu_and_mul":
        layer = SiluAndMul()
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
    out = layer(x)
    ref_out = layer.forward_native(x)
    # The SiluAndMul, MulAndSilu, GELU and FatReLU implementations are
    # equivalent to the native PyTorch implementations, so we can do exact
    # comparison.
    torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)

    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d, ))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    if activation == "fatrelu":
        opcheck(fn, (out, x, threshold))
    else:
        opcheck(fn, (out, x))


@pytest.mark.parametrize("activation", [(FastGELU, torch.ops._C.gelu_fast),
                                        (NewGELU, torch.ops._C.gelu_new),
                                        (QuickGELU, torch.ops._C.gelu_quick)])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_activation(
    activation: type[torch.nn.Module],
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, d, dtype=dtype)
    layer = activation[0]()
    fn = activation[1]
    out = layer(x)
    ref_out = layer.forward_native(x)
    torch.testing.assert_close(out,
                               ref_out,
                               atol=get_default_atol(out),
                               rtol=get_default_rtol(out))

    out = torch.empty_like(x)
    opcheck(fn, (out, x))


## Test Batched Implementaion ####

BATCH_SIZES = [1, 13, 26, 32]
NUM_TOKENS = [7, 37, 64, 4096]
D = [128, 256, 384, 512, 1024, 13824]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_batched_silu_mul(batch_size: int, num_tokens: int, d: int,
                          dtype: torch.dtype):

    input = torch.randn(
        (batch_size, num_tokens, d), device="cuda", dtype=dtype) / 10.0

    out = torch.empty((batch_size, num_tokens, d // 2),
                      device="cuda",
                      dtype=dtype)

    ref_out = out.clone()

    # valid num_tokens per batch
    valid_num_tokens = torch.randint(low=0,
                                     high=num_tokens + 1,
                                     size=(batch_size, ),
                                     device="cuda").to(dtype=torch.int32)

    # reference
    ref_batched_silu_mul(input, ref_out, valid_num_tokens)

    # impl
    torch.ops._C.batched_silu_and_mul(out, input, valid_num_tokens)

    torch.testing.assert_close(ref_out, out)
