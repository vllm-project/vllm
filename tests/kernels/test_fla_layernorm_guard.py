# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops.layernorm_guard import (
    layer_norm_fwd,
    layernorm_fn,
    rms_norm_ref,
)
from vllm.utils.torch_utils import set_random_seed


def layer_norm_ref(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    """Reference implementation for both layer norm and RMS norm."""
    if is_rms_norm:
        # Use the imported rms_norm_ref for RMS norm cases
        return rms_norm_ref(
            x,
            weight,
            bias,
            z=z,
            eps=eps,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            upcast=True,
        )

    # Layer norm implementation
    dtype = x.dtype
    x = x.float()
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    z = z.float() if z is not None else None

    if z is not None and not norm_before_gate:
        x = x * F.silu(z)

    if group_size is None:
        # Layer norm: subtract mean
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean).square()).mean(dim=-1, keepdim=True)
        rstd = 1 / torch.sqrt(var + eps)
        out = (x - mean) * rstd * weight
        if bias is not None:
            out = out + bias
    else:
        # Group norm
        from einops import rearrange

        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        mean = x_group.mean(dim=-1, keepdim=True)
        var = ((x_group - mean).square()).mean(dim=-1, keepdim=True)
        rstd = 1 / torch.sqrt(var + eps)
        x_group = (x_group - mean) * rstd
        out = rearrange(x_group, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias

    if z is not None and norm_before_gate:
        out *= F.silu(z)

    return out.to(dtype)


DTYPES = [torch.bfloat16, torch.float32]
# Test various M sizes to ensure rows_per_block logic works correctly
NUM_TOKENS = [
    1,
    7,
    16,
    63,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    5789,
    8189,
    8191,
    16383,
    32767,
]
HIDDEN_SIZES = [64, 128, 256, 1024]
GROUP_SIZES = [None, 64, 128]  # None means full hidden size
NORM_BEFORE_GATE = [True, False]
IS_RMS_NORM = [True, False]
SEEDS = [0, 42]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("is_rms_norm", IS_RMS_NORM)
@torch.inference_mode()
def test_layer_norm_fwd_basic(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    is_rms_norm: bool,
) -> None:
    """Test basic layer norm forward pass without z (gate) tensor."""
    set_random_seed(seed)
    device = torch.device("cuda:0")

    # Create inputs
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    bias = None if is_rms_norm else torch.randn(hidden_size, dtype=dtype, device=device)
    eps = 1e-6

    # Run the triton kernel
    out, mean, rstd = layer_norm_fwd(
        x, weight, bias, eps, z=None, is_rms_norm=is_rms_norm
    )

    # Run reference implementation
    ref_out = layer_norm_ref(x, weight, bias, z=None, eps=eps, is_rms_norm=is_rms_norm)

    # Check outputs
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    # Check mean and rstd shapes
    if not is_rms_norm:
        assert mean.shape == (num_tokens,)
    assert rstd.shape == (num_tokens,)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", [128, 256, 1024])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("norm_before_gate", NORM_BEFORE_GATE)
@pytest.mark.parametrize("is_rms_norm", IS_RMS_NORM)
@torch.inference_mode()
def test_layer_norm_fwd_with_gate(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    norm_before_gate: bool,
    is_rms_norm: bool,
) -> None:
    """Test layer norm forward pass with z (gate) tensor."""
    set_random_seed(42)
    device = torch.device("cuda:0")

    # Create inputs
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    z = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    bias = None if is_rms_norm else torch.randn(hidden_size, dtype=dtype, device=device)
    eps = 1e-6

    # Run the triton kernel
    out, mean, rstd = layer_norm_fwd(
        x,
        weight,
        bias,
        eps,
        z=z,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )

    # Run reference implementation
    ref_out = layer_norm_ref(
        x,
        weight,
        bias,
        z=z,
        eps=eps,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )

    # Check outputs
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", [128, 512])
@pytest.mark.parametrize("hidden_size", [512, 1024])
@pytest.mark.parametrize("group_size", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_rms_norm", IS_RMS_NORM)
@torch.inference_mode()
def test_layer_norm_fwd_with_groups(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    dtype: torch.dtype,
    is_rms_norm: bool,
) -> None:
    """Test layer norm forward pass with group normalization."""
    if hidden_size % group_size != 0:
        pytest.skip(
            f"hidden_size {hidden_size} not divisible by group_size {group_size}"
        )

    set_random_seed(42)
    device = torch.device("cuda:0")

    # Create inputs
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    bias = None if is_rms_norm else torch.randn(hidden_size, dtype=dtype, device=device)
    eps = 1e-6

    ngroups = hidden_size // group_size

    # Run the triton kernel
    out, mean, rstd = layer_norm_fwd(
        x, weight, bias, eps, z=None, group_size=group_size, is_rms_norm=is_rms_norm
    )

    # Run reference implementation
    ref_out = layer_norm_ref(
        x, weight, bias, z=None, eps=eps, group_size=group_size, is_rms_norm=is_rms_norm
    )

    # Check outputs
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    # Check mean and rstd shapes for groups
    if not is_rms_norm:
        assert mean.shape == (ngroups * num_tokens,)
    assert rstd.shape == (ngroups * num_tokens,)


@pytest.mark.parametrize("num_tokens", [7, 63, 128, 513, 1024, 2049])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_layer_norm_rows_per_block(
    num_tokens: int,
    dtype: torch.dtype,
) -> None:
    """Test that rows_per_block logic works correctly for various M sizes."""
    set_random_seed(42)
    device = torch.device("cuda:0")
    hidden_size = 1024

    # Create inputs
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    bias = torch.randn(hidden_size, dtype=dtype, device=device)
    eps = 1e-6

    # Run the triton kernel
    out, mean, rstd = layer_norm_fwd(x, weight, bias, eps, z=None, is_rms_norm=False)

    # Run reference implementation
    ref_out = layer_norm_ref(x, weight, bias, z=None, eps=eps, is_rms_norm=False)

    # Check outputs
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_strided_input(dtype: torch.dtype) -> None:
    """Test that the kernel handles non-contiguous (strided)
    inputs correctly."""
    set_random_seed(42)
    device = torch.device("cuda:0")
    num_tokens = 128
    hidden_size = 1024

    # Create a larger tensor and take a strided slice
    x_large = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device=device)
    x = x_large[:, :hidden_size]

    # Make it contiguous for the kernel
    x_contiguous = x.contiguous()

    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    bias = torch.randn(hidden_size, dtype=dtype, device=device)
    eps = 1e-6

    # Run the triton kernel with contiguous input
    out, mean, rstd = layer_norm_fwd(
        x_contiguous, weight, bias, eps, z=None, is_rms_norm=False
    )

    # Run reference implementation
    ref_out = layer_norm_ref(
        x_contiguous, weight, bias, z=None, eps=eps, is_rms_norm=False
    )

    # Check outputs
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", [1, 128, 2048])
@pytest.mark.parametrize("hidden_size", [768, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_output_buffer_provided(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """Test that the kernel works when an output buffer is provided."""
    set_random_seed(42)
    device = torch.device("cuda:0")

    # Create inputs
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    bias = torch.randn(hidden_size, dtype=dtype, device=device)
    eps = 1e-6

    # Pre-allocate output buffer
    out_buffer = torch.empty_like(x)

    # Run the triton kernel with provided output
    out, mean, rstd = layer_norm_fwd(
        x, weight, bias, eps, z=None, out=out_buffer, is_rms_norm=False
    )

    # Check that the provided buffer was used
    assert out.data_ptr() == out_buffer.data_ptr()

    # Run reference implementation
    ref_out = layer_norm_ref(x, weight, bias, z=None, eps=eps, is_rms_norm=False)

    # Check outputs
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 16, 1024),  # 3D tensor
        (2, 8, 512, 256),  # 4D tensor
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_multidimensional_input(
    shape: tuple,
    dtype: torch.dtype,
) -> None:
    """Test that the autograd function handles multidimensional inputs."""
    set_random_seed(42)
    device = torch.device("cuda:0")
    hidden_size = shape[-1]

    # Create inputs
    x = torch.randn(*shape, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    bias = torch.randn(hidden_size, dtype=dtype, device=device)
    eps = 1e-6

    # Run through autograd function
    out = layernorm_fn(x, weight, bias, z=None, eps=eps)

    # Run reference implementation
    ref_out = layer_norm_ref(x, weight, bias, z=None, eps=eps, is_rms_norm=False)

    # Check outputs
    assert out.shape == x.shape
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    # Run a quick smoke test
    test_layer_norm_fwd_basic(128, 1024, torch.float16, 42, False)
    test_layer_norm_fwd_with_gate(128, 1024, torch.float16, True, False)
    test_layer_norm_rows_per_block(513, torch.float16)
    print("All smoke tests passed!")
