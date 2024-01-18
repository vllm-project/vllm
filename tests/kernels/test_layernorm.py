import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNorm

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [768, 5120, 8192]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True]
SEEDS = [0]
DEVICES = [i for i in range(1 if torch.cuda.device_count() == 1 else 2)]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    gpu_id = f"cuda:{device}"
    layer = RMSNorm(hidden_size).to(dtype=dtype, device=gpu_id)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=gpu_id)
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
