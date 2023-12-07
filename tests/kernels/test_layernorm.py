import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm._C import ops

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [768, 5120, 8192]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True]
SEEDS = [0]
SCALE = [0.1, 0.5, 0.8, 1.2, 2.1]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    layer = RMSNorm(hidden_size).to(dtype).cuda()
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out = layer._forward(x, residual)
    out = layer(x, residual)
    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    if add_residual:
        assert torch.allclose(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
        assert torch.allclose(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
    else:
        assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_rms_norm_quant(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    layer = RMSNorm(hidden_size).to(dtype).cuda()
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    x *= scale
    residual1 = torch.randn_like(x) * scale if add_residual else None
    residual2 = residual1.clone() if add_residual else None

    out1 = torch.empty_like(x)
    if add_residual:
        ops.fused_add_rms_norm.add(
            out1,
            x,
            residual1,
            layer.weight.data,
            layer.variance_epsilon,
        )
    else:
        ops.rms_norm(
            out1,
            x,
            layer.weight.data,
            layer.variance_epsilon,
        )
    out1 = out1.clamp(-128, 127).round().to(torch.int8)
    out2 = torch.empty_like(x, dtype=torch.int8)
    if add_residual:
        ops.ops.fused_add_rms_norm.add(
            out2,
            x,
            residual2,
            layer.weight.data,
            layer.variance_epsilon,
            True
        )
    else:
        ops.rms_norm(
            out2,
            x,
            layer.weight.data,
            layer.variance_epsilon,
            True
        )
    
    assert torch.allclose(out1, out2, atol=1.0)
    if add_residual:
        assert torch.allclose(residual1, residual2, atol=0.001)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale", SCALE)
@torch.inference_mode()
def test_dequant_add_residual_rms_norm_quant(num_tokens: int, hidden_size: int,
                                             dtype: torch.dtype, seed: int,
                                             scale: float) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    s = float(hidden_size**-0.5)
    residual = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    x = torch.randint(-1000,
                      1000, (num_tokens, hidden_size),
                      dtype=torch.int32,
                      device="cuda")
    residual.uniform_(-s, s)
    ref = RMSNorm(hidden_size).to(dtype).cuda()
    x_ = (x * scale + residual).to(dtype)

    out1 = torch.empty_like(x_)
    ops.rms_norm(
        out1,
        x_,
        ref.weight.data,
        ref.variance_epsilon,
    )
    out1 = out1.round().clamp(-128, 127).to(torch.int8)
    out2 = torch.empty_like(x, dtype=torch.int8)
    ops.dequant_add_residual_rms_norm_quant(
        out2, x, residual, ref.weight.data, scale, ref.variance_epsilon)
    assert torch.allclose(out1, out2, atol=1.0)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_per_token_dequant_add_residual_rms_norm_quant(num_tokens: int,
                                                       hidden_size: int,
                                                       dtype: torch.dtype,
                                                       seed: int) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    s = float(hidden_size**-0.5)
    residual = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    x = torch.randint(-1000,
                      1000, (num_tokens, hidden_size),
                      dtype=torch.int32,
                      device="cuda")
    scale = torch.rand(num_tokens, 1, dtype=torch.float32, device="cuda")
    residual.uniform_(-s, s)
    ref = RMSNorm(hidden_size).to(dtype).cuda()
    x_ = (x * scale + residual).to(dtype)
    out1 = torch.empty_like(x_)
    ops.rms_norm(
        out1,
        x_,
        ref.weight.data,
        ref.variance_epsilon,
    )
    out1 = out1.round().clamp(-128, 127).to(torch.int8)
    out2 = torch.empty_like(x, dtype=torch.int8)
    scale = torch.squeeze(scale)
    ops.dequant_add_residual_rms_norm_quant(
        out2, x, residual, ref.weight.data, scale, ref.variance_epsilon)
    assert torch.allclose(out1, out2, atol=1.0)
    