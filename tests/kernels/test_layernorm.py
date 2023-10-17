import pytest
import torch
import torch.nn as nn

from vllm import layernorm_ops, fused_kernels

DTYPES = [torch.half, torch.bfloat16, torch.float]
HIDDEN_SIZES = [67, 768, 2048, 5120, 8192]  # Arbitrary values for testing
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
SEEDS = [0]
SCALE = [0.1, 0.5, 0.8, 1.2, 2.1]


class RefRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        weight = torch.empty(hidden_size)
        weight.normal_(mean=1.0, std=0.1)
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(hidden_size**-0.5)
    x = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    x.uniform_(-scale, scale)
    ref = RefRMSNorm(hidden_size).to(dtype).cuda()

    out = torch.empty_like(x)
    layernorm_ops.rms_norm(
        out,
        x,
        ref.weight.data,
        ref.variance_epsilon,
    )
    ref_out = ref(x)
    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-5)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_rms_norm_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(hidden_size**-0.5)
    x = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    x.uniform_(-scale, scale)
    ref = RefRMSNorm(hidden_size).to(dtype).cuda()

    out1 = torch.empty_like(x)
    layernorm_ops.rms_norm(
        out1,
        x,
        ref.weight.data,
        ref.variance_epsilon,
    )
    out1 = out1.clamp(-128, 127).round().to(torch.int8)
    out2 = torch.empty_like(x, dtype=torch.int8)
    layernorm_ops.invoke_rms_norm_quant(out2, x, ref.weight.data, ref.variance_epsilon)
    assert torch.allclose(out1, out2, atol=1.0)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale", SCALE)
@torch.inference_mode()
def test_dequant_add_residual_rms_norm_quant(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, seed: int, scale: float
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    s = float(hidden_size**-0.5)
    residual = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    # x = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, (num_tokens, hidden_size), dtype=torch.int32, device="cuda")
    x = torch.randint(
        -1000, 1000, (num_tokens, hidden_size), dtype=torch.int32, device="cuda"
    )
    residual.uniform_(-s, s)
    ref = RefRMSNorm(hidden_size).to(dtype).cuda()
    x_ = (x * scale + residual).to(dtype)

    out1 = torch.empty_like(x_)
    layernorm_ops.rms_norm(
        out1,
        x_,
        ref.weight.data,
        ref.variance_epsilon,
    )
    out1 = out1.round().clamp(-128, 127).to(torch.int8)
    out2 = torch.empty_like(x, dtype=torch.int8)
    layernorm_ops.invoke_dequant_add_residual_rms_norm_quant(
        out2, x, residual, ref.weight.data, ref.variance_epsilon, scale
    )

    assert torch.allclose(out1, out2, atol=1.0)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale", SCALE)
@torch.inference_mode()
def test_dequant_add_residual(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, seed: int, scale: float
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    s = float(hidden_size**-0.5)
    residual = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    x = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (num_tokens, hidden_size),
        dtype=torch.int32,
        device="cuda",
    )
    residual.uniform_(-s, s)
    out1 = (x * scale + residual).to(dtype)

    out2 = torch.empty_like(x, dtype=dtype)
    fused_kernels.invoke_dequant_add_residual(out2, x, residual, scale)

    assert torch.allclose(out1, out2, atol=0.001)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale", SCALE)
@torch.inference_mode()
def test_dequant(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, seed: int, scale: float
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    s = float(hidden_size**-0.5)
    # residual = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    x = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (num_tokens, hidden_size),
        dtype=torch.int32,
        device="cuda",
    )
    # residual.uniform_(-s, s)
    out1 = (x * scale).to(dtype)

    out2 = torch.empty_like(x, dtype=dtype)
    fused_kernels.invoke_dequant(out2, x, scale)
    assert torch.allclose(out1, out2, atol=0.001)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale", SCALE)
@torch.inference_mode()
def test_quant(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, seed: int, scale: float
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    s = float(hidden_size**-0.5)
    # residual = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") * 1000
    # residual.uniform_(-s, s)
    out1 = (x / scale).round().clamp(-128, 127).to(torch.int8)

    out2 = torch.empty_like(x, dtype=torch.int8)
    fused_kernels.invoke_quant(out2, x, scale)
    assert torch.allclose(out1, out2, atol=1)
