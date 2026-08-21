# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float16]
QUANT_DTYPES = [current_platform.fp8_dtype()]
NUM_TOKENS = [1, 17, 86, 1234, 3045]  # Arbitrary values for testing
HIDDEN_SIZES = [16, 48, 128, 1562, 4096]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


def ref_impl(
    silu_and_mul: SiluAndMul, x: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    silu_and_mul_out = silu_and_mul.forward_native(x)
    out, scales = ops.scaled_fp8_quant(silu_and_mul_out, scale)
    return out


def ops_impl(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_shape = (x.shape[0], x.shape[1] // 2)
    out = torch.empty(out_shape, dtype=current_platform.fp8_dtype(), device=x.device)
    torch.ops._C.silu_and_mul_quant(out, x, scale)
    return out


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_silu_and_mul(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)

    layer = SiluAndMul()

    # Make inputs
    scale = torch.randn((1), device=device, dtype=torch.float32)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    ref_out = ref_impl(layer, x, scale)
    ops_out = ops_impl(x, scale)

    assert ref_out.dtype == quant_dtype
    assert ops_out.dtype == quant_dtype
    assert ref_out.shape == ops_out.shape
    assert torch.allclose(
        ref_out.to(dtype=torch.float32), ops_out.to(dtype=torch.float32)
    )
    opcheck(torch.ops._C.silu_and_mul_quant, (ops_out, x, scale))


@pytest.mark.parametrize(
    ("num_tokens", "hidden_size"),
    [(1, 1), (7, 15), (17, 128), (3, 2048), (8, 3072)],
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("swiglu_limit", [None, 10.0])
@torch.inference_mode()
def test_humming_silu_and_mul_dynamic_per_token_quant(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    swiglu_limit: float | None,
) -> None:
    pytest.importorskip("humming")
    if not current_platform.has_device_capability(89):
        pytest.skip("FP8 activation quantization requires SM89+")

    from humming import ops as humming_ops

    x = torch.randn(
        num_tokens,
        hidden_size * 2,
        dtype=dtype,
        device="cuda",
    )
    activation = torch.empty((num_tokens, hidden_size), dtype=dtype, device=x.device)
    if swiglu_limit is None:
        torch.ops._C.silu_and_mul(activation, x)
    else:
        swiglu_limit_func(activation, x, swiglu_limit)

    ref_output = torch.empty_like(activation, dtype=torch.float8_e4m3fn)
    ref_output, ref_scale = humming_ops.quant_input(
        inputs=activation,
        outputs=ref_output,
        dtype="float8e4m3",
        group_size=None,
        scale_dtype="float32",
    )

    output = torch.empty_like(ref_output)
    scale = torch.empty((num_tokens, 1), dtype=torch.float32, device=x.device)
    torch.ops._C.silu_and_mul_per_token_quant(
        output,
        x,
        scale,
        None,
        swiglu_limit,
        1e-30,
        True,
    )

    torch.testing.assert_close(scale, ref_scale, atol=0.0, rtol=0.0)
    torch.testing.assert_close(output.float(), ref_output.float(), atol=0.0, rtol=0.0)

    if num_tokens == 7 and hidden_size == 15 and dtype == torch.bfloat16:
        opcheck(
            torch.ops._C.silu_and_mul_per_token_quant,
            (output, x, scale, None, swiglu_limit, 1e-30, True),
        )


@pytest.mark.parametrize("swiglu_limit", [None, 10.0])
@torch.inference_mode()
def test_humming_silu_and_mul_dynamic_per_token_quant_corner_values(
    default_vllm_config,
    swiglu_limit: float | None,
) -> None:
    pytest.importorskip("humming")
    if not current_platform.has_device_capability(89):
        pytest.skip("FP8 activation quantization requires SM89+")

    from humming import ops as humming_ops

    limit = 10.0
    edge = torch.tensor(
        [
            0.0,
            1e-20,
            -1e-20,
            limit - 1e-2,
            limit,
            limit + 1e-2,
            -limit - 1e-2,
            1e4,
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    gate = torch.stack((torch.zeros_like(edge), edge, -edge, edge))
    up = torch.stack((torch.zeros_like(edge), edge.flip(0), edge, -edge))
    x = torch.cat((gate, up), dim=-1)

    activation = torch.empty_like(gate)
    if swiglu_limit is None:
        torch.ops._C.silu_and_mul(activation, x)
    else:
        swiglu_limit_func(activation, x, swiglu_limit)
    ref_output = torch.empty_like(activation, dtype=torch.float8_e4m3fn)
    ref_output, ref_scale = humming_ops.quant_input(
        inputs=activation,
        outputs=ref_output,
        dtype="float8e4m3",
        group_size=None,
        scale_dtype="float32",
    )

    output = torch.empty_like(ref_output)
    scale = torch.empty((x.size(0), 1), dtype=torch.float32, device=x.device)
    torch.ops._C.silu_and_mul_per_token_quant(
        output,
        x,
        scale,
        None,
        swiglu_limit,
        1e-30,
        True,
    )

    torch.testing.assert_close(scale, ref_scale, atol=0.0, rtol=0.0)
    torch.testing.assert_close(output.float(), ref_output.float(), atol=0.0, rtol=0.0)
