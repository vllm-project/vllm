import pytest
import torch

from vllm.model_executor.layers.activation import FastGELU, NewGELU, SiluAndMul
from vllm._C import ops

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 4096, 5120, 13824]  # Arbitrary values for testing
SEEDS = [0]
SCALE_UP = [0.09, 1.2, 1.9]
SCALE_GATE = [2.17, 1.2, 1.9]
SCALE_OUT = [1.2, 1.9, 0.17]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_silu_and_mul(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype, device="cuda")
    layer = SiluAndMul()
    out = layer(x)
    ref_out = layer._forward(x)
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale_gate", SCALE_GATE)
@pytest.mark.parametrize("scale_up", SCALE_UP)
@pytest.mark.parametrize("scale_out", SCALE_OUT)
@torch.inference_mode()
def test_dequant_silu_and_mul_quant(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    scale_gate: float,
    scale_up: float,
    scale_out: float,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # x = torch.randn(num_tokens, 2 * d, dtype=dtype, device='cuda')
    x = torch.randint(-1000,
                      1000, (num_tokens, 2 * d),
                      dtype=torch.int32,
                      device="cuda")
    x_ = torch.empty_like(x, dtype=dtype)
    x_[:, :d] = x[:, :d] * scale_gate
    x_[:, d:] = x[:, d:] * scale_up
    out1 = torch.empty(num_tokens, d, dtype=dtype, device="cuda")
    ops.silu_and_mul(out1, x_)
    out1 = (out1 / scale_out).round().clamp(-128, 127).to(torch.int8)
    out2 = torch.empty(num_tokens, d, dtype=torch.int8, device="cuda")
    ops.dequant_silu_and_mul_quant(out2, x, scale_up, scale_out, scale_gate)
    assert torch.allclose(out1, out2, atol=2)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale_gate", SCALE_GATE)
@pytest.mark.parametrize("scale_up", SCALE_UP)
@torch.inference_mode()
def test_dequant_silu_and_mul_per_token_quant(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    scale_gate: float,
    scale_up: float,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.randint(-1000,
                      1000, (num_tokens, 2 * d),
                      dtype=torch.int32,
                      device="cuda")

    x_ = torch.empty_like(x, dtype=dtype)
    x_[:, :d] = x[:, :d] * scale_gate
    x_[:, d:] = x[:, d:] * scale_up
    out1 = torch.empty(num_tokens, d, dtype=dtype, device="cuda")
    ops.silu_and_mul(out1, x_)
    scale1 = torch.max(out1, dim=1)[0].to(torch.float32) / 127.0
    out1 = (out1 / scale1.view(-1, 1)).round().clamp(-128, 127).to(torch.int8)
    out2 = torch.empty(num_tokens, d, dtype=torch.int8, device="cuda")
    tmp = torch.empty(num_tokens, d, dtype=torch.float32, device="cuda")
    scale2 = torch.empty(num_tokens, dtype=torch.float32, device="cuda")
    ops.dequant_silu_and_mul_quant(out2, x, scale_gate, scale_up, scale2, tmp)
    assert torch.allclose(out1, out2, atol=2)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_gelu_new(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.randn(num_tokens, d, dtype=dtype, device="cuda")
    layer = NewGELU()
    out = layer(x)
    ref_out = layer._forward(x)
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_gelu_fast(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.randn(num_tokens, d, dtype=dtype, device="cuda")
    layer = FastGELU()
    out = layer(x)
    ref_out = layer._forward(x)
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)
