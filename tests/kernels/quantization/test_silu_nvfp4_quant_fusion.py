# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

if not current_platform.has_device_capability(100):
    pytest.skip(reason="Nvfp4 Requires compute capability of 10 or above.",
                allow_module_level=True)

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(128, 64), (128, 128), (256, 64), (256, 128)]
SEEDS = [42]
CUDA_DEVICES = ['cuda:0']

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

BLOCK_SIZE = 16


def ref_impl(silu_and_mul: SiluAndMul, x: torch.Tensor,
             global_scale: torch.Tensor,
             ref_output_scale: torch.Tensor) -> torch.Tensor:
    silu_and_mul_out = silu_and_mul.forward_native(x)
    assert not current_platform.is_rocm()
    assert silu_and_mul_out.ndim >= 1, (
        f'input.ndim needs to be >= 1, but got {silu_and_mul_out.ndim}.')
    other_dims = 1 if silu_and_mul_out.ndim == 1 else -1
    silu_and_mul_out = silu_and_mul_out.reshape(other_dims,
                                                silu_and_mul_out.shape[-1])
    m, n = silu_and_mul_out.shape
    device = silu_and_mul_out.device

    # Two fp4 values will be packed into an uint8.
    out = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    output_scale = ref_output_scale

    torch.ops._C.scaled_fp4_quant(out, silu_and_mul_out, output_scale,
                                  global_scale)

    return out, output_scale


def ops_impl(x: torch.Tensor, global_scale: torch.Tensor,
             ref_output_scale: torch.Tensor) -> torch.Tensor:
    out_shape = (x.shape[0], x.shape[1] // 4)
    output_scale = ref_output_scale
    out = torch.empty(out_shape, dtype=torch.uint8, device=x.device)
    torch.ops._C.silu_and_mul_nvfp4_quant(out, output_scale, x, global_scale)
    return out, output_scale


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_quantize_to_fp4(
    dtype: torch.dtype,
    shape: tuple[int, int],
    seed: int,
    device: str,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)

    m, n = shape

    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    block_size = 16

    assert n % block_size == 0, (
        f'last dim has to be multiple of 16, but got {n}.')
    assert x.dtype in (torch.float16, torch.bfloat16), (
        f'input.dtype needs to be fp16 or bf16 but got {x.dtype}.')

    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(x.shape[0], 128)
    scale_n = x.shape[1] // (2 * block_size)
    rounded_n = round_up(scale_n, 4)
    output_scale = torch.empty((rounded_m, rounded_n // 4),
                               device=x.device,
                               dtype=torch.int32)

    layer = SiluAndMul()

    ref_out, ref_out_scale = ref_impl(layer, x, global_scale, output_scale)

    fusion_out, fusion_out_scale = ops_impl(x, global_scale, output_scale)

    assert ref_out.dtype == torch.uint8
    assert fusion_out.dtype == torch.uint8
    assert ref_out.shape == fusion_out.shape

    assert ref_out_scale.dtype == torch.int32
    assert fusion_out_scale.dtype == torch.int32
    assert ref_out_scale.shape == fusion_out_scale.shape

    # Allow up to 2% of mismatched values since BF16 has accuracy issues.
    mis_threshold = 0.02
    atol = 0.4
    rtol = 0.4
    ref_logits = ref_out[-1]
    fusion_logits = fusion_out[-1]

    mis_count = torch.sum(
        torch.abs(fusion_logits - ref_logits) > (atol +
                                                 rtol * torch.abs(ref_logits)))
    mis_ratio = mis_count / fusion_logits.numel()

    assert mis_ratio < mis_threshold, \
        f"Mismatch ratio {mis_ratio} exceeds threshold {mis_threshold}"

    torch.testing.assert_close(ref_out_scale, fusion_out_scale)

    opcheck(torch.ops._C.silu_and_mul_nvfp4_quant,
            (fusion_out, fusion_out_scale, x, global_scale))
