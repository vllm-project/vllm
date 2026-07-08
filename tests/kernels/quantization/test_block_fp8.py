# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/pull/2575
import itertools
from unittest.mock import patch

import pytest
import torch

from tests.kernels.quant_utils import (
    native_per_token_group_quant_fp8,
    native_w8a8_block_matmul,
)
from tests.kernels.utils import fp8_ulp_distance
from vllm.config import VllmConfig
from vllm.model_executor.kernels.linear.scaled_mm import (
    cutlass as cutlass_kernel_module,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
    cutlass_scaled_mm,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    w8a8_triton_block_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_gemm_nt,
    get_tma_aligned_size,
    per_block_cast_to_fp8,
    should_use_deepgemm_for_fp8_linear,
)
from vllm.utils.flashinfer import (
    flashinfer_fp8_blockscale_gemm,
    has_flashinfer_fp8_blockscale_gemm,
)
from vllm.utils.import_utils import has_deep_gemm

if current_platform.get_device_capability() < (9, 0):
    pytest.skip("FP8 Triton requires CUDA 9.0 or higher", allow_module_level=True)

vllm_config = VllmConfig()

# Test configurations
DTYPES = [torch.bfloat16]  # [torch.half, torch.bfloat16, torch.float32]
# Quantization test configs
NUM_TOKENS = [7, 2050]
D = [512, 4096, 5120, 13824]
GROUP_SIZE = [64, 128, 512]
COLUMN_MAJOR_SCALES = [True, False]
TMA_ALIGNED_SCALES = [True, False]
# Matmul test configs
M = [1, 7, 8, 83, 4096]
N = [128, 512, 576, 7168, 13824]
K = [256, 3884, 4096, 13824, 16384]
# Deepseek-V3's intermediate size 18432, so N is 18432*2/8=4608 at TP8
# and its hidden size is 7168.
BLOCK_SIZE = [[128, 128]]
OUT_DTYPES = [torch.bfloat16]  # [torch.float32, torch.half, torch.bfloat16]
SEEDS = [0]

# Skip all tests if CUDA is not available
pytest.importorskip("torch.cuda")


@pytest.fixture(autouse=True)
def setup_cuda():
    torch.set_default_device("cuda")


@pytest.mark.skipif(
    current_platform.is_fp8_fnuz(),
    reason="This platform supports e4m3fnuz, not e4m3fn.",
)
@pytest.mark.parametrize(
    "num_tokens,d,dtype,group_size,column_major_scales,tma_aligned_scales,seed",
    itertools.product(
        NUM_TOKENS,
        D,
        DTYPES,
        GROUP_SIZE,
        COLUMN_MAJOR_SCALES,
        TMA_ALIGNED_SCALES,
        SEEDS,
    ),
)
@torch.inference_mode()
def test_per_token_group_quant_fp8(
    num_tokens, d, dtype, group_size, column_major_scales, tma_aligned_scales, seed
):
    torch.manual_seed(seed)
    x = torch.rand(num_tokens, d, dtype=dtype)

    ref_out, ref_scale = native_per_token_group_quant_fp8(x, group_size)
    out, scale = per_token_group_quant_fp8(
        x,
        group_size,
        column_major_scales=column_major_scales,
        tma_aligned_scales=tma_aligned_scales,
    )

    if current_platform.is_rocm():
        # On gfx950 the Triton and PyTorch FP8 kernels can round in opposite
        # directions when an element lands at the midpoint between two adjacent
        # e4m3fn values (1-ULP tie-breaking). Verify: (1) no element is more
        # than 1 FP8 ULP away, and (2) fewer than 0.05% of elements have any
        # mismatch. Observed worst case across all parameter combos: 0.049%,
        # max ULP = 1.
        ulp = fp8_ulp_distance(out, ref_out)
        assert (ulp <= 1).all(), (
            f"FP8 mismatch > 1 ULP: {int((ulp > 1).sum())} elements"
        )
        assert float((ulp > 0).float().mean()) < 5e-4, (
            f"Too many 1-ULP mismatches: {int((ulp > 0).sum())}/{ulp.numel()}"
        )
    else:
        assert torch.allclose(
            out.to(torch.float32), ref_out.to(torch.float32), rtol=0.15
        )
    assert torch.allclose(scale, ref_scale)

    if column_major_scales:
        assert scale.stride()[-2] == 1
        if tma_aligned_scales:
            assert scale.stride()[-1] == get_tma_aligned_size(num_tokens, 4)


@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, OUT_DTYPES, SEEDS),
)
@torch.inference_mode()
def test_w8a8_block_fp8_matmul(M, N, K, block_size, out_dtype, seed):
    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(current_platform.fp8_dtype())
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(current_platform.fp8_dtype())

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(current_platform.fp8_dtype())

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32) * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)
    out = w8a8_triton_block_scaled_mm(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    rel_diff = torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))
    ) / torch.mean(torch.abs(ref_out.to(torch.float32)))
    assert rel_diff < 0.001


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUTLASS only supported on CUDA platform."
)
@torch.inference_mode()
def test_w8a8_block_fp8_cutlass_matmul():
    # Test simple case where weight.shape % 128 != 0,
    # like in DSV3 kv_a_proj_with_mqa
    M = 32
    N = 576
    K = 7168
    block_size = [128, 128]
    out_dtype = torch.bfloat16
    seed = 0

    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale

    A_fp8, As = per_token_group_quant_fp8(
        A_fp32, block_size[1], column_major_scales=False
    )
    # CUTLASS uses column-major format for scales
    A_fp8_cutlass, As_cutlass = per_token_group_quant_fp8(
        A_fp32, block_size[1], column_major_scales=True
    )

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)
    out = cutlass_scaled_mm(A_fp8_cutlass, B_fp8, As_cutlass, Bs, block_size, out_dtype)

    rel_diff = torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))
    ) / torch.mean(torch.abs(ref_out.to(torch.float32)))
    assert rel_diff < 0.001


@torch.inference_mode()
def test_w8a8_block_fp8_matmul_e8m0_scales():
    # DeepSeek-V4-style checkpoints store block scales in exponent-only
    # E8M0, which Triton cannot bind directly; the kernel upcasts them to
    # fp32 before launch. Regression test for
    # https://github.com/vllm-project/vllm/issues/47818.
    M, N, K = 83, 512, 7168
    block_size = [128, 128]
    out_dtype = torch.bfloat16
    torch.manual_seed(0)

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    n_tiles = (N + block_size[0] - 1) // block_size[0]
    k_tiles = (K + block_size[1] - 1) // block_size[1]

    # Power-of-two scales round-trip E8M0 <-> fp32 exactly.
    As = torch.exp2(torch.randint(-8, 0, (M, k_tiles)).to(torch.float32))
    Bs = torch.exp2(torch.randint(-8, 0, (n_tiles, k_tiles)).to(torch.float32))

    ref_out = w8a8_triton_block_scaled_mm(A_fp8, B_fp8, As, Bs, block_size, out_dtype)
    out = w8a8_triton_block_scaled_mm(
        A_fp8,
        B_fp8,
        As.to(torch.float8_e8m0fnu),
        Bs.to(torch.float8_e8m0fnu),
        block_size,
        out_dtype,
    )
    assert torch.equal(out, ref_out)


def _block_fp8_linear_config(n: int, k: int) -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=create_fp8_quant_key(
            static=True, group_shape=GroupShape(128, 128)
        ),
        activation_quant_key=create_fp8_quant_key(
            static=False, group_shape=GroupShape(1, 128)
        ),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=(n, k),
    )


def test_cutlass_block_fp8_sm12x_declines_unaligned_n():
    # The SM12x CUTLASS blockwise kernels cannot serve layers whose weight
    # output dim is not a multiple of 128 (e.g. kv_a_proj N=576 in DeepSeek
    # models), so can_implement must fall through to the next kernel.
    with patch.object(
        cutlass_kernel_module.current_platform,
        "is_device_capability_family",
        new=lambda family: family == 120,
    ):
        ok, _ = CutlassFp8BlockScaledMMKernel.can_implement(
            _block_fp8_linear_config(512, 7168)
        )
        assert ok
        ok, reason = CutlassFp8BlockScaledMMKernel.can_implement(
            _block_fp8_linear_config(576, 7168)
        )
        assert not ok
        assert "divisible by 128" in reason

    with patch.object(
        cutlass_kernel_module.current_platform,
        "is_device_capability_family",
        new=lambda family: False,
    ):
        ok, _ = CutlassFp8BlockScaledMMKernel.can_implement(
            _block_fp8_linear_config(576, 7168)
        )
        assert ok


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUTLASS only supported on CUDA platform."
)
@torch.inference_mode()
def test_cutlass_block_fp8_e8m0_weight_scale_upcast(default_vllm_config):
    # cutlass_scaled_mm rejects E8M0 scales at dispatch, so the kernel must
    # upcast them to fp32 when processing weights. Regression test for
    # https://github.com/vllm-project/vllm/issues/47818.
    is_supported, reason = CutlassFp8BlockScaledMMKernel.is_supported()
    if not is_supported:
        pytest.skip(reason)

    M, N, K = 32, 512, 7168
    torch.manual_seed(0)
    fp8_info = torch.finfo(torch.float8_e4m3fn)

    weight = (
        ((torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_info.max)
        .clamp(min=fp8_info.min, max=fp8_info.max)
        .to(torch.float8_e4m3fn)
    )
    scale = torch.exp2(torch.randint(-8, 0, (N // 128, K // 128)).to(torch.float32))

    config = _block_fp8_linear_config(N, K)
    kernel = CutlassFp8BlockScaledMMKernel(config)

    def make_layer(weight_scale: torch.Tensor) -> torch.nn.Module:
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(weight_scale, requires_grad=False)
        kernel.process_weights_after_loading(layer)
        return layer

    e8m0_layer = make_layer(scale.to(torch.float8_e8m0fnu))
    assert e8m0_layer.weight_scale_inv.dtype == torch.float32
    assert torch.equal(e8m0_layer.weight_scale_inv, scale)

    fp32_layer = make_layer(scale.clone())
    x = torch.randn(M, K, dtype=torch.bfloat16)
    assert torch.equal(
        kernel.apply_weights(e8m0_layer, x), kernel.apply_weights(fp32_layer, x)
    )


@pytest.mark.skipif(
    current_platform.is_fp8_fnuz(),
    reason="This platform supports e4m3fnuz, not e4m3fn.",
)
@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, OUT_DTYPES, SEEDS),
)
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGemm kernels not available.")
@torch.inference_mode()
def test_w8a8_block_fp8_deep_gemm_matmul(M, N, K, block_size, out_dtype, seed):
    torch.manual_seed(seed)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max = fp8_info.max

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max

    # only aligned sizes are supported by deepgemm
    if not should_use_deepgemm_for_fp8_linear(
        output_dtype=out_dtype, weight_shape=B_fp32.shape, supports_deep_gemm=True
    ):
        pytest.skip(f"Skipping test; invalid size {M}, {N}, {K}")

    A_fp8, As_fp8 = per_token_group_quant_fp8(
        A_fp32, block_size[1], column_major_scales=True, tma_aligned_scales=True
    )
    B_fp8, Bs_fp8 = per_block_cast_to_fp8(B_fp32, block_size=block_size)

    As = As_fp8.to(torch.float32)
    Bs = Bs_fp8.to(torch.float32)

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    out = torch.zeros((M, N), device="cuda", dtype=out_dtype)

    assert As_fp8.shape == (M, (K + 127) // 128), (
        f"{As_fp8.shape} != {(M, (K + 127) // 128)}"
    )

    fp8_gemm_nt((A_fp8, As_fp8), (B_fp8, Bs_fp8), out)

    rel_diff = torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))
    ) / torch.mean(torch.abs(ref_out.to(torch.float32)))
    assert rel_diff < 0.001


@pytest.mark.skipif(
    current_platform.is_fp8_fnuz(),
    reason="This platform supports e4m3fnuz, not e4m3fn.",
)
@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, OUT_DTYPES, SEEDS),
)
@torch.inference_mode()
def test_w8a8_block_fp8_flashinfer_matmul(M, N, K, block_size, out_dtype, seed):
    if not has_flashinfer_fp8_blockscale_gemm():
        pytest.skip(
            "FlashInfer block GEMM not available (requires SM90+ and FlashInfer)"
        )
    # only aligned sizes
    if K % 128 != 0 or N % 64 != 0:
        pytest.skip(f"Skipping test; invalid size {M}, {N}, {K}")

    torch.manual_seed(seed)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max = fp8_info.max

    A_bf16 = (torch.rand(M, K, dtype=torch.bfloat16) - 0.5) * 2 * fp8_max
    B_bf16 = (torch.rand(N, K, dtype=torch.bfloat16) - 0.5) * 2 * fp8_max

    A_fp8, As_fp8 = per_token_group_quant_fp8(A_bf16, block_size[1], use_ue8m0=False)
    B_fp8, Bs_fp8 = per_block_cast_to_fp8(B_bf16, block_size, use_ue8m0=False)

    As = As_fp8.to(torch.float32)
    Bs = Bs_fp8.to(torch.float32)

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    out = flashinfer_fp8_blockscale_gemm(
        input=A_bf16,
        weight=B_fp8,
        input_scale=None,
        weight_scale=Bs,
        out_dtype=out_dtype,
    )

    rel_diff = torch.mean(
        torch.abs(out.to(torch.bfloat16) - ref_out.to(torch.bfloat16))
    ) / torch.mean(torch.abs(ref_out.to(torch.bfloat16)))
    assert rel_diff < 0.001
