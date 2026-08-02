# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests that FusedRMSNormGated decomposes correctly under torch.compile,
matching the eager triton kernel output."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16]
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
    device = torch.device("cuda:0")

    module = FusedRMSNormGated(
        hidden_size,
        elementwise_affine=elementwise_affine,
        eps=1e-5,
        activation=activation,
        device=device,
        dtype=dtype,
    )
    # Model parameters use torch.empty because checkpoint loading overwrites
    # them. Initialize the standalone test module so allocator contents cannot
    # introduce NaNs and make this comparison flaky.
    if module.weight is not None:
        module.weight.uniform_(-1, 1)
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
    device = torch.device("cuda:0")
    head_dim = shape[-1]

    module = FusedRMSNormGated(
        head_dim,
        elementwise_affine=elementwise_affine,
        eps=1e-5,
        activation=activation,
        device=device,
        dtype=dtype,
    )
    if module.weight is not None:
        module.weight.uniform_(-1, 1)
    x = torch.randn(*shape, dtype=dtype, device=device)
    g = torch.randn(*shape, dtype=dtype, device=device)

    # forward_cuda may modify x in-place, so clone inputs
    cuda_out = module.forward_cuda(x.clone(), g.clone())
    compiled_native = torch.compile(module.forward_native, fullgraph=True)
    native_out = compiled_native(x.clone(), g.clone())

    torch.testing.assert_close(native_out, cuda_out, atol=1e-3, rtol=1e-2)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize("num_tokens", [1, 8, 64])
@torch.inference_mode()
def test_kimi_k3_gate_uses_fused_rocm_kernel(num_tokens: int) -> None:
    """Kimi-K3's packed gate selects the fused ROCm implementation."""
    set_random_seed(0)
    device = torch.device(current_platform.device_type)
    config = VllmConfig()
    config.compilation_config.custom_ops = ["none"]
    with set_current_vllm_config(config):
        module = FusedRMSNormGated(
            hidden_size=128,
            activation="sigmoid",
            device=device,
            dtype=torch.bfloat16,
            enforce_enable=True,
        )
    module.weight.normal_()

    x = torch.randn((1, num_tokens, 12, 128), dtype=torch.bfloat16, device=device)
    packed_gate = torch.randn((num_tokens, 6288), dtype=torch.bfloat16, device=device)
    gate = packed_gate[:, 4608:6144].view(num_tokens, 12, 128)
    assert gate.stride()[-2:] == (128, 1), "each gate head must remain contiguous"
    assert num_tokens == 1 or gate.stride(0) == packed_gate.stride(0), (
        "a multi-token gate must retain the packed projection buffer's token stride"
    )

    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    reference = (
        x_float
        * torch.rsqrt(variance + module.eps)
        * module.weight.float()
        * torch.sigmoid(gate.float())
    ).to(x.dtype)

    output = module(x, gate)

    assert output.data_ptr() == x.data_ptr()
    torch.testing.assert_close(output, reference, atol=2e-4, rtol=1e-2)
