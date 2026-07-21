# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the FlashInfer W4A16 NVFP4 linear kernel (mm_bf16_fp4)."""

from types import SimpleNamespace

import pytest
import torch
from nvfp4_utils import (
    break_fp4_bytes,
    convert_swizzled_to_linear,
    quant_nvfp4_tensor,
)

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import (
    FlashInferW4A16NvFp4LinearKernel,
    MarlinNvFp4LinearKernel,
    NvFp4LinearLayerConfig,
    init_nvfp4_linear_kernel,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not FlashInferW4A16NvFp4LinearKernel.is_supported()[0]:
    pytest.skip(
        reason="FlashInfer W4A16 NVFP4 requires SM100/SM110/SM12x and "
        "FlashInfer with mm_bf16_fp4.",
        allow_module_level=True,
    )

# m, n, k
SHAPES = [
    (1, 128, 256),
    (16, 256, 512),
    (128, 512, 2048),
    (4, 2048, 512),
    # n not a multiple of the 64-wide repack tile (exercises N padding).
    (8, 200, 512),
    # k not a multiple of the 64-column tile (exercises K padding).
    (8, 64, 1120),
]

SEEDS = [42]
DEVICE = "cuda:0"
BLOCK_SIZE = 16


def _assert_close_to_reference(out: torch.Tensor, ref: torch.Tensor) -> None:
    # Elementwise tolerances do not fit a bf16-output GEMM, whose rounding
    # error grows with K, so compare relative L2 error and cosine
    # similarity against the dequantized-weight reference.
    out_f = out.float().flatten()
    ref_f = ref.float().flatten()
    rel_l2 = (torch.linalg.vector_norm(out_f - ref_f) / ref_f.norm()).item()
    cos = torch.nn.functional.cosine_similarity(out_f, ref_f, dim=0).item()
    assert rel_l2 < 2e-2, f"relative L2 error {rel_l2:.4f} exceeds 2e-2"
    assert cos > 0.999, f"cosine similarity {cos:.6f} below 0.999"


def _make_w4a16_layer(n: int, k: int, device: str):
    """Build a layer holding checkpoint-format NVFP4 weights plus the
    dequantized reference weight."""
    w = torch.randn((n, k), dtype=torch.bfloat16, device=device)
    w_fp4, w_sf_swizzled, w_global_scale_encode = quant_nvfp4_tensor(w)

    # Checkpoints store linear (N, K // 16) fp8 block scales. The
    # unswizzle helper keeps its column padding, so trim to K // 16.
    sf_linear = (
        convert_swizzled_to_linear(w_sf_swizzled, n, k, BLOCK_SIZE)[
            :, : k // BLOCK_SIZE
        ]
        .view(torch.float8_e4m3fn)
        .contiguous()
    )

    w_vals = break_fp4_bytes(w_fp4, torch.float32).reshape(
        n, k // BLOCK_SIZE, BLOCK_SIZE
    )
    sf = sf_linear.to(torch.float32) / w_global_scale_encode
    w_ref = (w_vals * sf.unsqueeze(-1)).reshape(n, k).to(torch.bfloat16)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_fp4, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(sf_linear, requires_grad=False)
    # Both quant methods store the global scale in its dequant form.
    layer.weight_global_scale = torch.nn.Parameter(
        1.0 / w_global_scale_encode, requires_grad=False
    )
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    return layer, w_ref


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("autotune", [False, True])
@torch.inference_mode()
def test_flashinfer_w4a16_nvfp4_gemm(
    shape: tuple[int, int, int],
    seed: int,
    use_bias: bool,
    autotune: bool,
) -> None:
    import flashinfer

    set_random_seed(seed)
    m, n, k = shape

    layer, w_ref = _make_w4a16_layer(n, k, DEVICE)
    kernel = FlashInferW4A16NvFp4LinearKernel(NvFp4LinearLayerConfig())
    kernel.process_weights_after_loading(layer)

    x = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    bias = torch.randn((n,), dtype=torch.bfloat16, device=DEVICE) if use_bias else None

    with flashinfer.autotune(autotune):
        out = kernel.apply_weights(layer, x, bias)

    expected = x @ w_ref.t()
    if use_bias:
        expected = expected + bias
    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16
    _assert_close_to_reference(out, expected)


@torch.inference_mode()
def test_flashinfer_w4a16_nvfp4_gemm_3d_input() -> None:
    set_random_seed(0)
    n, k = 256, 512
    layer, w_ref = _make_w4a16_layer(n, k, DEVICE)
    kernel = FlashInferW4A16NvFp4LinearKernel(NvFp4LinearLayerConfig())
    kernel.process_weights_after_loading(layer)

    x = torch.randn((2, 8, k), dtype=torch.bfloat16, device=DEVICE)
    out = kernel.apply_weights(layer, x)

    assert out.shape == (2, 8, n)
    _assert_close_to_reference(out, x @ w_ref.t())


def test_w4a16_kernel_selection_auto():
    kernel = init_nvfp4_linear_kernel(use_a16=True)
    if current_platform.is_device_capability(121):
        assert isinstance(kernel, FlashInferW4A16NvFp4LinearKernel)
    else:
        assert isinstance(kernel, MarlinNvFp4LinearKernel)


def test_w4a16_kernel_selection_flashinfer_cutedsl():
    vllm_config = VllmConfig()
    vllm_config.kernel_config.linear_backend = "flashinfer_cutedsl"
    with set_current_vllm_config(vllm_config):
        kernel = init_nvfp4_linear_kernel(use_a16=True)
    assert isinstance(kernel, FlashInferW4A16NvFp4LinearKernel)


def test_w4a16_kernel_selection_marlin_backend():
    vllm_config = VllmConfig()
    vllm_config.kernel_config.linear_backend = "marlin"
    with set_current_vllm_config(vllm_config):
        kernel = init_nvfp4_linear_kernel(use_a16=True)
    assert isinstance(kernel, MarlinNvFp4LinearKernel)


def test_w4a16_kernel_selection_batch_invariant(monkeypatch):
    """Batch-invariant mode must not route W4A16 onto a W4A4 kernel."""
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True, raising=False)
    kernel = init_nvfp4_linear_kernel(use_a16=True)
    assert isinstance(kernel, MarlinNvFp4LinearKernel)


def test_w4a16_kernel_selection_rejects_w4a4_backend():
    """Backends with only W4A4 kernels cannot serve W4A16 layers."""
    vllm_config = VllmConfig()
    vllm_config.kernel_config.linear_backend = "cutlass"
    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(ValueError, match="W4A16"),
    ):
        init_nvfp4_linear_kernel(use_a16=True)


def test_w4a16_kernel_requires_bf16(monkeypatch):
    """fp16 models must not select the FlashInfer W4A16 kernel."""
    import vllm.config as vllm_config_mod

    fake_config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float16))
    monkeypatch.setattr(
        vllm_config_mod, "get_current_vllm_config_or_none", lambda: fake_config
    )
    ok, reason = FlashInferW4A16NvFp4LinearKernel.can_implement(
        NvFp4LinearLayerConfig()
    )
    assert not ok
    assert "bfloat16" in reason
