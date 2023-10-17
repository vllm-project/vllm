import pytest
import torch
import torch.nn.functional as F
from transformers.activations import get_activation

from vllm import activation_ops

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 4096, 5120, 13824]  # Arbitrary values for testing
SEEDS = [0]
SCALE_UP = [0.09, 1.2, 1.9]
SCALE_GATE = [2.17, 1.2, 1.9]
SCALE_OUT = [1.2, 1.9, 0.17]


def ref_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(chunks=2, dim=1)
    return F.silu(x1) * x2


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
    out = torch.empty(num_tokens, d, dtype=dtype, device="cuda")
    activation_ops.silu_and_mul(out, x)
    ref_out = ref_silu_and_mul(x)
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
    x = torch.randint(
        -1000, 1000, (num_tokens, 2 * d), dtype=torch.int32, device="cuda"
    )
    x_ = torch.empty_like(x, dtype=dtype)
    x_[:, :d] = x[:, :d] * scale_gate
    x_[:, d:] = x[:, d:] * scale_up
    out1 = torch.empty(num_tokens, d, dtype=dtype, device="cuda")
    activation_ops.silu_and_mul(out1, x_)
    out1 = (out1 / scale_out).round().clamp(-128, 127).to(torch.int8)
    # ref_out = ref_silu_and_mul(x)

    out2 = torch.empty(num_tokens, d, dtype=torch.int8, device="cuda")
    activation_ops.invoke_dequant_silu_and_mul_quant(
        out2, x, scale_gate, scale_up, scale_out
    )
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
    out = torch.empty(num_tokens, d, dtype=dtype, device="cuda")
    activation_ops.gelu_new(out, x)
    ref_out = get_activation("gelu_new")(x)
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
    out = torch.empty(num_tokens, d, dtype=dtype, device="cuda")
    activation_ops.gelu_fast(out, x)
    ref_out = get_activation("gelu_fast")(x)
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)
