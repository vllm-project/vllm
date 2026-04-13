# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from types import SimpleNamespace
from unittest.mock import patch

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
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


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


@pytest.mark.parametrize(
    "activation,approximate",
    [
        ("gelu_and_mul", "none"),
        ("gelu_and_mul", "tanh"),
        ("new_gelu", None),
        ("fast_gelu", None),
        ("quick_gelu", None),
    ],
)
@pytest.mark.parametrize("num_tokens", [7, 83])
@pytest.mark.parametrize("d", [512])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@torch.inference_mode()
def test_activation_cuda_alike_no_c_ext(
    default_vllm_config,
    activation: str,
    approximate,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
) -> None:
    """Regression test: activations must not crash on cuda_alike platforms
    (e.g. ROCm) when the vLLM C++ extension was built without GPU support
    and the _C ops are absent."""
    # Simulate a cuda_alike platform whose _C extension has no custom ops.
    empty_C = SimpleNamespace()  # no attributes → hasattr(...) is always False

    with (
        patch(
            "vllm.model_executor.layers.activation.current_platform.is_cuda_alike",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.layers.activation.current_platform.is_rocm",
            return_value=False,
        ),
        patch(
            "vllm.model_executor.layers.activation.current_platform.is_xpu",
            return_value=False,
        ),
        patch("vllm.model_executor.layers.activation.torch.ops._C", empty_C),
    ):
        if activation == "gelu_and_mul":
            x = torch.randn(num_tokens, 2 * d, dtype=dtype)
            layer = GeluAndMul(approximate=approximate)
        elif activation == "new_gelu":
            x = torch.randn(num_tokens, d, dtype=dtype)
            layer = NewGELU()
        elif activation == "fast_gelu":
            x = torch.randn(num_tokens, d, dtype=dtype)
            layer = FastGELU()
        elif activation == "quick_gelu":
            x = torch.randn(num_tokens, d, dtype=dtype)
            layer = QuickGELU()

    out = layer(x)
    ref_out = layer.forward_native(x)
    torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)


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
