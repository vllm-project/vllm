# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types

import pytest
import torch
from nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    convert_swizzled_to_linear,
)

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    NvFp4LinearBackend,
    apply_nvfp4_linear,
    convert_to_nvfp4_linear_kernel_format,
    cutlass_fp4_supported,
    select_nvfp4_linear_backend,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

CUDA_DEVICES = ["cuda:0"]
BLOCK_SIZE = 16
SEEDS = [42]

# (n, k, m) — output_size, input_size, batch_size
SHAPES = [(128, 128, 64), (256, 128, 128)]

DTYPES = [torch.bfloat16, torch.float16]

BACKENDS = [NvFp4LinearBackend.VLLM_CUTLASS]
if has_flashinfer():
    BACKENDS += [
        NvFp4LinearBackend.FLASHINFER_CUTLASS,
        NvFp4LinearBackend.FLASHINFER_CUDNN,
        NvFp4LinearBackend.FLASHINFER_TRTLLM,
    ]


def make_nvfp4_layer(
    n: int,
    k: int,
    m: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[types.SimpleNamespace, torch.Tensor, torch.Tensor]:
    weight_float = torch.randn((n, k), dtype=dtype, device=device)
    x = torch.randn((m, k), dtype=dtype, device=device)

    wgs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / weight_float.abs().max().float()
    igs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / x.abs().max().float()

    # ops.scaled_fp4_quant returns swizzled scales by default.
    # convert_to_nvfp4_linear_kernel_format expects linear (un-swizzled)
    # scales, so unswizzle before storing on the layer.
    w_fp4, w_sf_swizzled = ops.scaled_fp4_quant(weight_float, wgs)
    w_sf_linear = convert_swizzled_to_linear(w_sf_swizzled, n, k, BLOCK_SIZE).to(
        torch.float8_e4m3fn
    )

    alpha = torch.tensor(
        1.0 / (igs.item() * wgs.item()), dtype=torch.float32, device=device
    )

    layer = types.SimpleNamespace()
    layer.weight = torch.nn.Parameter(w_fp4, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(w_sf_linear, requires_grad=False)
    layer.weight_global_scale = wgs
    layer.input_global_scale_inv = igs
    layer.alpha = alpha
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    return layer, weight_float, x


def test_select_backend_auto() -> None:
    backend = select_nvfp4_linear_backend()
    assert isinstance(backend, NvFp4LinearBackend)
    if has_flashinfer():
        assert backend == NvFp4LinearBackend.FLASHINFER_CUTLASS
    else:
        assert backend == NvFp4LinearBackend.VLLM_CUTLASS


def test_select_backend_env_cutlass(monkeypatch) -> None:
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_USE_FBGEMM", False)
    monkeypatch.setattr(envs, "VLLM_USE_NVFP4_CT_EMULATIONS", False)
    monkeypatch.setattr(envs, "VLLM_NVFP4_GEMM_BACKEND", "cutlass")
    assert select_nvfp4_linear_backend() == NvFp4LinearBackend.VLLM_CUTLASS


def test_select_backend_emulation(monkeypatch) -> None:
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_USE_FBGEMM", False)
    monkeypatch.setattr(envs, "VLLM_USE_NVFP4_CT_EMULATIONS", True)
    assert select_nvfp4_linear_backend() == NvFp4LinearBackend.EMULATION


def test_cutlass_fp4_supported() -> None:
    assert cutlass_fp4_supported() is True


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n,k,m", SHAPES)
@pytest.mark.parametrize("backend", BACKENDS)
@torch.inference_mode()
def test_apply_nvfp4_linear(
    backend: NvFp4LinearBackend,
    n: int,
    k: int,
    m: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    if backend == NvFp4LinearBackend.FLASHINFER_TRTLLM and dtype == torch.float16:
        pytest.skip("TRTLLM backend only supports bfloat16")

    set_random_seed(seed)
    layer, _, x = make_nvfp4_layer(n, k, m, dtype, device)

    convert_to_nvfp4_linear_kernel_format(backend, layer)
    out = apply_nvfp4_linear(backend, layer, x)

    assert out.shape == (m, n)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    ref_scale = x.float().abs().mean() * layer.weight_global_scale.float() * k
    assert out.float().abs().mean() < ref_scale * 10


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("backend", BACKENDS)
@torch.inference_mode()
def test_apply_nvfp4_linear_with_bias(
    backend: NvFp4LinearBackend,
    seed: int,
    device: str,
) -> None:
    dtype, n, k, m = torch.bfloat16, 128, 128, 64
    set_random_seed(seed)
    layer, _, x = make_nvfp4_layer(n, k, m, dtype, device)
    convert_to_nvfp4_linear_kernel_format(backend, layer)

    bias = torch.randn(n, dtype=dtype, device=device)
    out_no_bias = apply_nvfp4_linear(backend, layer, x)
    out_with_bias = apply_nvfp4_linear(backend, layer, x, bias=bias)

    torch.testing.assert_close(out_with_bias, out_no_bias + bias, atol=1e-3, rtol=0.0)
