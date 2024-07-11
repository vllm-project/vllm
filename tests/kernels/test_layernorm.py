import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNorm
import vllm._custom_ops as ops
DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [3, 7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [2048, 768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192,
                8199]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm_quant(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    layer = RMSNorm(hidden_size).to(dtype).cuda()
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    x *= scale
    x_ = x.clone()
    residual1 = torch.randn_like(x) * scale if add_residual else None
    residual2 = residual1.clone() if add_residual else None

    out1 = torch.empty_like(x)
    if add_residual:
        ops.fused_add_rms_norm(
            x,
            residual1,
            layer.weight.data,
            layer.variance_epsilon,
        )
        out1 = x
    else:
        ops.rms_norm(
            out1,
            x,
            layer.weight.data,
            layer.variance_epsilon,
        )
    scale1 = out1.abs().max(dim=-1)[0].div(127.0).to(torch.float)
    out1 = (out1 / scale1.view(-1, 1)).round().clamp(-128, 127).to(torch.int8)
    out2 = torch.empty_like(x, dtype=torch.int8)
    scale2 = torch.empty(num_tokens, dtype=torch.float32)
    tmp = torch.empty_like(x_, dtype=torch.float32)
    if add_residual:
        ops.add_residual_rms_norm_quant(out2, x_, residual2, tmp,
                                        layer.weight.data, scale2,
                                        layer.variance_epsilon)
    else:
        ops.rms_norm_quant(out2, x_, tmp, layer.weight.data, scale2,
                           layer.variance_epsilon)

    assert torch.allclose(out1, out2, atol=2.0)
    assert torch.allclose(scale1, scale2, atol=1e-3)
    if add_residual:
        assert torch.allclose(residual1, residual2, atol=1e-3)

@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm_quant2(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    layer = RMSNorm(hidden_size).to(dtype).cuda()
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    x *= scale
    x_ = x.clone()
    residual1 = torch.randn_like(x) * scale if add_residual else None
    residual2 = residual1.clone() if add_residual else None

    out1 = torch.empty_like(x)
    if add_residual:
        ops.fused_add_rms_norm(
            x,
            residual1,
            layer.weight.data,
            layer.variance_epsilon,
        )
        out1 = x
    else:
        ops.rms_norm(
            out1,
            x,
            layer.weight.data,
            layer.variance_epsilon,
        )
    scale1 = out1.abs().max(dim=-1)[0].div(127.0).to(torch.float)
    out1 = (out1 / scale1.view(-1, 1)).round().clamp(-128, 127).to(torch.int8)
    out2 = torch.empty_like(x, dtype=torch.int8)
    tmp = torch.empty_like(x_, dtype=torch.float32)
    if add_residual:
        ops.add_residual_rms_norm_quant(out2, x_, residual2, tmp,
                                        layer.weight.data, scale1,
                                        layer.variance_epsilon)
    else:
        ops.rms_norm_quant(out2, x_, tmp, layer.weight.data, scale1,
                           layer.variance_epsilon)

    assert torch.allclose(out1, out2, atol=2.0)
    if add_residual:
        assert torch.allclose(residual1, residual2, atol=1e-3)

