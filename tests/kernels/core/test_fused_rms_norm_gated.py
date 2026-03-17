# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests that FusedRMSNormGated decomposes correctly under torch.compile,
matching the eager triton kernel output."""

import pytest
import torch

from vllm.model_executor.layers.fla.ops.kda import FusedRMSNormGated
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16]
DEVICE_TYPE = current_platform.device_type
HIDDEN_SIZES = [128, 512]
NUM_TOKENS = [64, 128]
ACTIVATIONS = ["swish", "sigmoid"]
ELEMENTWISE_AFFINE = [True, False]
SEEDS = [0]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("elementwise_affine", ELEMENTWISE_AFFINE)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_compiled_vs_eager(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    activation: str,
    elementwise_affine: bool,
    dtype: torch.dtype,
    seed: int,
) -> None:
    """forward_native decomposition matches forward_cuda triton kernel."""
    torch._dynamo.reset()
    set_random_seed(seed)
    device = torch.device(f"{DEVICE_TYPE}:0")

    module = FusedRMSNormGated(
        hidden_size,
        elementwise_affine=elementwise_affine,
        eps=1e-5,
        activation=activation,
        device=device,
        dtype=dtype,
    )
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    g = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # forward_cuda may modify x in-place, so clone inputs
    cuda_out = module.forward_cuda(x.clone(), g.clone())
    compiled_native = torch.compile(module.forward_native, fullgraph=True)
    native_out = compiled_native(x.clone(), g.clone())

    torch.testing.assert_close(native_out, cuda_out, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 16, 32, 128),
        (2, 8, 16, 64),
    ],
)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("elementwise_affine", ELEMENTWISE_AFFINE)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_compiled_vs_eager_multidim(
    default_vllm_config,
    shape: tuple,
    activation: str,
    elementwise_affine: bool,
    dtype: torch.dtype,
    seed: int,
) -> None:
    """forward_native decomposition handles multi-dimensional inputs."""
    torch._dynamo.reset()
    set_random_seed(seed)
    device = torch.device(f"{DEVICE_TYPE}:0")
    head_dim = shape[-1]

    module = FusedRMSNormGated(
        head_dim,
        elementwise_affine=elementwise_affine,
        eps=1e-5,
        activation=activation,
        device=device,
        dtype=dtype,
    )
    x = torch.randn(*shape, dtype=dtype, device=device)
    g = torch.randn(*shape, dtype=dtype, device=device)

    # forward_cuda may modify x in-place, so clone inputs
    cuda_out = module.forward_cuda(x.clone(), g.clone())
    compiled_native = torch.compile(module.forward_native, fullgraph=True)
    native_out = compiled_native(x.clone(), g.clone())

    torch.testing.assert_close(native_out, cuda_out, atol=1e-3, rtol=1e-2)
