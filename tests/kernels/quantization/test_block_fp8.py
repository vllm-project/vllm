# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/pull/2575
import itertools
import types

import pytest
import torch

from tests.kernels.quant_utils import (
    native_per_token_group_quant_fp8,
    native_w8a8_block_matmul,
)
from tests.kernels.utils import fp8_ulp_distance
from vllm.config import VllmConfig
from vllm.model_executor.kernels.linear.scaled_mm.b12x import (
    B12xFp8BlockScaledMMKernel,
    _run_b12x_fp8_block_scaled_mm,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import cutlass_scaled_mm
from vllm.model_executor.kernels.linear.scaled_mm.rdna4 import (
    RDNA4Fp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    w8a8_triton_block_scaled_mm,
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


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(90)),
    reason="torch._scaled_mm DeepSeek-style block scaling only supports SM90.",
)
def test_w8a8_block_fp8_torch_scaled_mm_matmul():
    # BlockWiseTorchFP8ScaledMMLinearKernel: 1x128 activation + 128x128 weight
    # block scaling routed through torch._scaled_mm. M=83 is not a multiple of
    # 4 so this also exercises the M-padding path.
    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
        BlockWiseTorchFP8ScaledMMLinearKernel,
    )

    M = 83
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

    # Reference uses row-major activation scales.
    A_fp8, As = per_token_group_quant_fp8(
        A_fp32, block_size[1], column_major_scales=False
    )
    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    # The kernel expects column-major activation scales on CUDA.
    A_fp8_cuda, As_cuda = per_token_group_quant_fp8(
        A_fp32, block_size[1], column_major_scales=True
    )

    stub = BlockWiseTorchFP8ScaledMMLinearKernel.__new__(
        BlockWiseTorchFP8ScaledMMLinearKernel
    )
    stub.config = types.SimpleNamespace(out_dtype=out_dtype)
    out = stub.apply_block_scaled_mm(
        A_fp8_cuda.cuda(), B_fp8.cuda(), As_cuda.cuda(), Bs.cuda()
    )

    ref_out = ref_out.cuda()
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

    assert As_fp8.shape == (
        M,
        (K + 127) // 128,
    ), f"{As_fp8.shape} != {(M, (K + 127) // 128)}"

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


@pytest.mark.parametrize(
    "M,N,K",
    [(1, 128, 256), (8, 256, 512), (129, 256, 256), (2, 4096, 4096)],
)
@torch.inference_mode()
def test_w8a8_block_fp8_b12x_matmul(M, N, K):
    supported, reason = B12xFp8BlockScaledMMKernel.is_supported()
    if not supported:
        pytest.skip(reason)

    torch.manual_seed(M)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    A_bf16 = (torch.rand(M, K, dtype=torch.bfloat16) - 0.5) * 2 * fp8_max
    B_bf16 = (torch.rand(N, K, dtype=torch.bfloat16) - 0.5) * 2 * fp8_max
    A_fp8, As = per_token_group_quant_fp8(A_bf16, 128, use_ue8m0=False)
    B_fp8, Bs = per_block_cast_to_fp8(
        B_bf16,
        block_size=[128, 128],
        use_ue8m0=False,
    )
    As = As.float()
    Bs = Bs.float()

    ref_out = native_w8a8_block_matmul(
        A_fp8,
        B_fp8,
        As,
        Bs,
        [128, 128],
        torch.bfloat16,
    )
    out = _run_b12x_fp8_block_scaled_mm(
        A_fp8,
        B_fp8,
        As,
        Bs,
        torch.bfloat16,
    )

    rel_diff = torch.mean(torch.abs(out.float() - ref_out.float())) / torch.mean(
        torch.abs(ref_out.float())
    )
    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(),
        ref_out.float().flatten(),
        dim=0,
    )
    # Four-way split-K uses atomic BF16 reductions, so nondeterministic atomic
    # ordering can flap an output between adjacent BF16 values one ULP apart.
    assert rel_diff < 0.003
    assert cosine >= 0.99999


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 128, 128),
        (2, 128, 128),
        (4, 128, 256),
        (16, 128, 384),
        (17, 5120, 3072),
        (33, 17408, 5120),
        (39, 16384, 8192),
        (48, 17408, 5120),
        (64, 5120, 3072),
        (65, 128, 256),
        (72, 128, 256),
        (129, 128, 256),
        (249, 128, 256),
        (256, 8192, 5120),
        (523, 5120, 8704),
        (784, 7168, 5120),
        (1024, 8192, 5120),
    ],
)
@torch.inference_mode()
def test_rdna4_block_fp8_flydsl_matches_triton(m, n, k):
    supported, reason = RDNA4Fp8BlockScaledMMKernel.is_supported()
    if not supported:
        pytest.skip(reason)

    generator = torch.Generator(device="cuda").manual_seed(m + k)
    a = (torch.randn((m, k), device="cuda", generator=generator) * 0.25).to(
        torch.float8_e4m3fn
    )
    weight = (torch.randn((n, k), device="cuda", generator=generator) * 0.25).to(
        torch.float8_e4m3fn
    )
    a_scale = (
        torch.rand((m, k // 128), device="cuda", generator=generator) * 0.05 + 0.001
    )
    weight_scale = (
        torch.rand((n // 128, k // 128), device="cuda", generator=generator) * 0.05
        + 0.001
    )

    reference = w8a8_triton_block_scaled_mm(
        a, weight, a_scale, weight_scale, [128, 128], torch.bfloat16
    )
    output = torch.ops.vllm.rdna4_fp8_block_scaled_mm(a, weight, a_scale, weight_scale)
    torch.testing.assert_close(output, reference, rtol=0.01, atol=0.01)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (2, 2560, 1280),
        (1, 12288, 384),
        (16, 12160, 640),
        (16, 12288, 640),
        (17, 4096, 1280),
        (24, 4224, 640),
        (31, 8192, 3072),
        (32, 1792, 1280),
        (39, 16384, 8192),
        (48, 5376, 3840),
        (64, 384, 384),
        (65, 512, 1024),
        (65, 1024, 1280),
        (129, 768, 3840),
        (256, 384, 1280),
        (257, 1024, 1280),
        (129, 1152, 1024),
        (129, 3072, 512),
        (129, 5376, 640),
        # Full-B-prefetch K boundary and odd rotated K-block count.
        (129, 2176, 896),
        (129, 2176, 1024),
        (129, 2176, 1152),
        (513, 3072, 512),
        (1025, 384, 1280),
        (255, 1024, 512),
        (256, 256, 1024),
        (256, 512, 4096),
        (256, 512, 4224),
        (127, 1024, 1152),
        (65, 2048, 128),
        (512, 3072, 512),
        (512, 3072, 640),
        (32, 5120, 128),
        (33, 5120, 640),
        # Both M-tile sizes at the inclusive 64MiB boundary and above it.
        (32, 8192, 8192),
        (33, 8192, 8192),
        (32, 8320, 8192),
        (33, 8320, 8192),
        # Split-K ownership and CTA-count boundaries, including rotation-free M.
        (256, 1024, 2816),
        (256, 1024, 3072),
        (257, 1024, 3072),
        (65, 2048, 3072),
        (65, 2176, 3072),
        (1024, 256, 2816),
        (1024, 256, 3072),
        (1025, 256, 3072),
        # Wave32 CU tiles: 64-column scale ownership, split K and LDS lookahead.
        (128, 1536, 2048),
        (128, 1536, 4096),
        (512, 1536, 4096),
        (4097, 4096, 384),
    ],
)
@torch.inference_mode()
def test_rdna4_block_fp8_rotated_k_graph_replay(m, n, k):
    """Every rotated K block must use current scales on each graph replay."""
    supported, reason = RDNA4Fp8BlockScaledMMKernel.is_supported()
    if not supported:
        pytest.skip(reason)

    generator = torch.Generator(device="cuda").manual_seed(934)
    a = (torch.randn((m, k), device="cuda", generator=generator) * 0.25).to(
        torch.float8_e4m3fn
    )
    weight = (torch.randn((n, k), device="cuda", generator=generator) * 0.25).to(
        torch.float8_e4m3fn
    )
    a_scale = torch.ones((m, k // 128), device="cuda", dtype=torch.float32)
    weight_scale = torch.ones((n // 128, k // 128), device="cuda", dtype=torch.float32)
    for _ in range(2):
        torch.ops.vllm.rdna4_fp8_block_scaled_mm(a, weight, a_scale, weight_scale)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = torch.ops.vllm.rdna4_fp8_block_scaled_mm(
            a, weight, a_scale, weight_scale
        )

    scale_blocks = k // 128
    for kb in dict.fromkeys(
        (0, min(1, scale_blocks - 1), scale_blocks // 2, scale_blocks - 1)
    ):
        a_scale.zero_()
        a_scale[:, kb] = 1.0
        weight_scale.fill_(0.5)
        output.fill_(float("nan"))
        graph.replay()
        reference = (
            a[:, kb * 128 : (kb + 1) * 128].float()
            @ weight[:, kb * 128 : (kb + 1) * 128].float().t()
        ) * 0.5
        torch.testing.assert_close(
            output, reference.to(torch.bfloat16), rtol=0.01, atol=0.0005
        )


@pytest.mark.parametrize(
    "m,n,k,padding",
    [
        (1, 384, 384, 128),
        (3, 12160, 384, 128),
        (15, 12288, 640, 256),
        (17, 4096, 640, 128),
        (24, 4224, 1280, 256),
        (31, 1792, 1280, 128),
        (31, 8192, 384, 128),
        (33, 5120, 3072, 256),
        (49, 5376, 3840, 128),
        (63, 3584, 5120, 256),
        (65, 384, 384, 128),
        (65, 1024, 512, 128),
        (96, 1024, 1280, 128),
        (129, 768, 1152, 128),
        (129, 768, 768, 128),
        (255, 384, 5376, 128),
        (127, 1792, 1280, 128),
        (129, 1024, 1024, 128),
        (255, 256, 1024, 256),
        (257, 128, 1024, 128),
        (257, 5376, 3840, 128),
        (129, 5376, 1280, 128),
        (512, 10752, 5376, 128),
        (1025, 384, 640, 128),
        (255, 512, 4096, 128),
        (255, 512, 4224, 128),
        (127, 1024, 1152, 256),
        (127, 384, 128, 4),
        (512, 3072, 512, 128),
        (513, 3072, 512, 128),
        (129, 5376, 640, 4),
        (129, 2176, 1152, 4),
        (127, 1536, 4096, 4),
        (513, 1536, 4224, 128),
        (4097, 4096, 384, 4),
    ],
)
@torch.inference_mode()
def test_rdna4_block_fp8_ragged_routes_with_padded_weight_rows(m, n, k, padding):
    """Routing must preserve actual weight strides and partial M-tile masks."""
    supported, reason = RDNA4Fp8BlockScaledMMKernel.is_supported()
    if not supported:
        pytest.skip(reason)

    generator = torch.Generator(device="cuda").manual_seed(m + n + k)
    a = (torch.randn((m, k), device="cuda", generator=generator) * 0.25).to(
        torch.float8_e4m3fn
    )
    backing = (
        torch.randn((n, k + padding), device="cuda", generator=generator) * 0.25
    ).to(torch.float8_e4m3fn)
    weight = backing[:, :k]
    a_scale = torch.rand((m, k // 128), device="cuda", generator=generator)
    weight_scale = torch.rand((n // 128, k // 128), device="cuda", generator=generator)
    reference = w8a8_triton_block_scaled_mm(
        a, weight, a_scale, weight_scale, [128, 128], torch.bfloat16
    )
    output = torch.ops.vllm.rdna4_fp8_block_scaled_mm(a, weight, a_scale, weight_scale)
    torch.testing.assert_close(output, reference, rtol=0.02, atol=0.0625)


@pytest.mark.parametrize("row_stride", [0, 124, 129])
@torch.inference_mode()
def test_rdna4_block_fp8_rejects_unsupported_weight_row_strides(row_stride):
    """Overlapping or unaligned weight rows must fail before kernel execution."""
    supported, reason = RDNA4Fp8BlockScaledMMKernel.is_supported()
    if not supported:
        pytest.skip(reason)

    m, n, k = 65, 128, 128
    a = torch.empty((m, k), device="cuda", dtype=torch.float8_e4m3fn)
    backing = torch.empty(
        (n - 1) * row_stride + k, device="cuda", dtype=torch.float8_e4m3fn
    )
    weight = backing.as_strided((n, k), (row_stride, 1))
    a_scale = torch.ones((m, 1), device="cuda", dtype=torch.float32)
    weight_scale = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    with pytest.raises(
        ValueError, match="row stride must be at least K and divisible by 4"
    ):
        torch.ops.vllm.rdna4_fp8_block_scaled_mm(a, weight, a_scale, weight_scale)


@pytest.mark.parametrize(
    "m,n,k,row_stride,accepted",
    [
        pytest.param(8388607, 128, 512, 512, True, id="a-below-4gib"),
        pytest.param(8388608, 128, 512, 512, False, id="a-at-4gib"),
        pytest.param(8388609, 128, 512, 512, False, id="a-above-4gib"),
        pytest.param(1, 128, 128, 33554428, True, id="weight-below-4gib"),
        pytest.param(1, 128, 128, 33554432, False, id="weight-at-4gib"),
        pytest.param(1, 128, 128, 33554436, False, id="weight-above-4gib"),
        pytest.param(65535, 32768, 128, 128, True, id="output-below-4gib"),
        pytest.param(65536, 32768, 128, 128, False, id="output-at-4gib"),
        pytest.param(65537, 32768, 128, 128, False, id="output-above-4gib"),
        pytest.param(65, 128, 128, 132, True, id="weight-padding4"),
        pytest.param(65, 128, 128, 148, True, id="weight-padding20"),
    ],
)
def test_rdna4_block_fp8_buffer_byte_span_contract(
    monkeypatch, m, n, k, row_stride, accepted
):
    """Reject descriptor wraparound without allocating multi-GiB tensors."""
    pytest.importorskip("flydsl")
    from vllm.model_executor.kernels.linear.scaled_mm.flydsl_kernels import (
        rdna4_fp8_blockscale,
    )

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: types.SimpleNamespace(gcnArchName="gfx1201"),
    )

    def metadata(shape, dtype, stride=None):
        strides = stride or (shape[1], 1)
        return types.SimpleNamespace(
            shape=shape,
            ndim=2,
            dtype=dtype,
            device=torch.device("cuda:0"),
            is_contiguous=lambda: strides == (shape[1], 1),
            stride=lambda dim: strides[dim],
        )

    a = metadata((m, k), torch.float8_e4m3fn)
    weight = metadata((n, k), torch.float8_e4m3fn, (row_stride, 1))
    a_scale = metadata((m, k // 128), torch.float32)
    weight_scale = metadata((n // 128, k // 128), torch.float32)
    if accepted:
        assert rdna4_fp8_blockscale.validate_tensors(
            a, weight, a_scale, weight_scale
        ) == (m, n, k)
    else:
        with pytest.raises(ValueError, match="smaller than 4 GiB"):
            rdna4_fp8_blockscale.validate_tensors(a, weight, a_scale, weight_scale)
