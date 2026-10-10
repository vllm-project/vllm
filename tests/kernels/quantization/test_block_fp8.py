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
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_gemm_nt,
    get_tma_aligned_size,
    is_deep_gemm_e8m0_used,
    per_block_cast_to_fp8,
    should_use_deepgemm_for_fp8_linear,
)
from vllm.utils.flashinfer import (
    flashinfer_fp8_blockscale_gemm,
    has_flashinfer_fp8_blockscale_gemm,
)
from vllm.utils.import_utils import has_deep_gemm

capability = current_platform.get_device_capability()
if capability is None or capability < (9, 0):
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
@pytest.mark.parametrize(
    # 65/66/67 cover all M%4 residue classes above the SM100 swapAB
    # threshold (m <= 64); 1026 crosses multiple 128-row SF atoms.
    "M",
    [32, 65, 66, 67, 1026],
)
@torch.inference_mode()
def test_w8a8_block_fp8_cutlass_matmul(M):
    # Test simple case where weight.shape % 128 != 0,
    # like in DSV3 kv_a_proj_with_mqa
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
    B_fp8, Bs_fp8 = per_block_cast_to_fp8(
        B_fp32,
        block_size=block_size,
        use_ue8m0=is_deep_gemm_e8m0_used(),
    )

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


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(90),
    reason="Triton MXFP8 kernel requires SM90",
)
@pytest.mark.parametrize(
    "m,n,k",
    [
        (m, 129, 96)
        for m in [0, 1, 2, 4, 8, 9, 32, 33, 63, 64, 65, 1023, 1024, 1025, 8193]
    ]
    + [(1, n, k) for n, k in [(576, 5120), (1792, 5120), (4096, 1280), (5120, 288)]]
    + [(8, 576, 5120), (32, 129, 288)]
    + [(16, 1792, 5120), (16, 4096, 1280), (32, 576, 5120)]
    + [(8, 4096, 1280), (8, 5120, 288), (8, 2049, 2048), (8, 2049, 2080)]
    + [(m, 576, 5120) for m in [33, 128, 129, 512, 1024, 8192]]
    + [(33, 1792, 5120), (129, 1792, 5120), (33, 4096, 1280)]
    + [(9, 4993, 320), (16, 5120, 288), (32, 5120, 288), (33, 4993, 320)],
)
@torch.inference_mode()
def test_triton_mxfp8_matches_quantized_reference(m, n, k):
    """Keep tail rows/columns and per-output-row MX32 scales independent."""
    from vllm.model_executor.layers.quantization.utils.triton_mxfp8 import (
        triton_mxfp8_linear,
    )

    torch.manual_seed(1729)
    x = torch.randn((m, k), dtype=torch.bfloat16)
    weight = torch.randn((n, k)).to(torch.float8_e4m3fn)
    scales = torch.exp2(torch.randint(-10, 2, (n, k // 32)).float())
    actual = triton_mxfp8_linear(x, weight, scales)
    assert actual.shape == (m, n)
    assert actual.dtype == x.dtype
    if not m:
        return

    # Independently reproduce the E8M0 activation grid in torch. Compare the
    # contraction after quantization, separately from W8A16 model-quality tests.
    groups = x.float().reshape(m, k // 32, 32)
    amax = groups.abs().amax(dim=-1).clamp_min(1e-10)
    a_scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    a_fp8 = (groups / a_scale[..., None]).to(torch.float8_e4m3fn)
    a_dequant = (a_fp8.float() * a_scale[..., None]).reshape(m, k)
    w_dequant = weight.float() * scales.repeat_interleave(32, dim=1)
    expected = a_dequant @ w_dequant.T
    assert torch.isfinite(actual).all()
    relative_rmse = (actual.float() - expected).square().mean().sqrt()
    relative_rmse /= expected.square().mean().sqrt()
    assert relative_rmse < 0.006

    x.zero_()
    assert torch.count_nonzero(triton_mxfp8_linear(x, weight, scales)) == 0


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(90),
    reason="Triton MXFP8 requires SM90",
)
@pytest.mark.parametrize("m", [0, 1, 1023, 1024, 8193])
@torch.inference_mode()
def test_triton_mxfp8_backend_owns_weights_for_all_rows(m):
    """Load padded scales without Marlin packing, including decode and large M."""
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mxfp8.triton import TritonMxfp8LinearKernel
    from vllm.model_executor.layers.quantization.utils.triton_mxfp8 import (
        triton_mxfp8_linear,
    )

    torch.manual_seed(1729)
    n, k = 129, 96
    weight = torch.randn((n, k)).to(torch.float8_e4m3fn)
    encoded = torch.randint(120, 126, (n + 1, k // 32 + 1), dtype=torch.uint8)
    scales = encoded[:n, : k // 32].contiguous().view(torch.float8_e8m0fnu).float()
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(encoded, requires_grad=False)
    kernel = TritonMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    weight_ptr = weight.data_ptr()
    kernel.process_weights_after_loading(layer)
    assert layer.weight.data_ptr() == weight_ptr
    assert set(dict(layer.named_parameters())) == {"weight", "weight_scale"}
    assert not list(layer.buffers())
    torch.testing.assert_close(layer.weight_scale, scales, atol=0, rtol=0)
    # Exercise leading dimensions, non-contiguous activations, and bias.
    x = torch.randn((2, m, k * 2), dtype=torch.bfloat16)[..., ::2]
    bias = torch.randn(n, dtype=torch.bfloat16)
    expected = triton_mxfp8_linear(x, weight, scales) + bias
    actual = kernel.apply_weights(layer, x, bias)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if m == 1:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = kernel.apply_weights(layer, x, bias)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(90),
    reason="Triton MXFP8 requires SM90",
)
@pytest.mark.parametrize("n,k", [(576, 5120), (129, 1024), (4993, 320)])
@torch.inference_mode()
def test_triton_mxfp8_compiles_across_dispatch_boundaries(n, k):
    """Dynamic fullgraph compilation must preserve GEMV/GEMM dispatch results."""
    from vllm.model_executor.layers.quantization.utils.triton_mxfp8 import (
        triton_mxfp8_linear,
    )

    torch.manual_seed(1729)
    weight = torch.randn(n, k).to(torch.float8_e4m3fn)
    scales = torch.exp2(torch.randint(-8, 0, (n, k // 32)).float())
    torch.compiler.reset()
    compiled = torch.compile(triton_mxfp8_linear, fullgraph=True, dynamic=True)
    for m in (1, 8, 16, 33, 129):
        x = torch.randn(m, k, dtype=torch.bfloat16)
        expected = triton_mxfp8_linear(x, weight, scales)
        torch.testing.assert_close(
            compiled(x, weight, scales), expected, rtol=0, atol=0
        )


@pytest.mark.parametrize("backend", ["auto", "marlin", "triton"])
def test_triton_mxfp8_backend_selection(backend):
    """An explicit MXFP8 override selects W8A8 without changing auto/Marlin."""
    from vllm.config import set_current_vllm_config
    from vllm.config.kernel import KernelConfig
    from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.mxfp8.triton import TritonMxfp8LinearKernel

    if not current_platform.is_device_capability(90):
        pytest.skip("Backend selection is validated on SM90")
    config = VllmConfig(
        kernel_config=KernelConfig(linear_backend_per_quant={"mxfp8": backend})
    )
    with set_current_vllm_config(config):
        kernel = init_mxfp8_linear_kernel()
    expected = (
        TritonMxfp8LinearKernel if backend == "triton" else MarlinMxfp8LinearKernel
    )
    assert isinstance(kernel, expected)


@torch.inference_mode()
def test_triton_mxfp8_rejects_unsafe_offsets_before_allocation():
    """A broadcast view must not allow an overflowing output pointer offset."""
    from vllm.model_executor.layers.quantization.utils.triton_mxfp8 import (
        triton_mxfp8_linear,
    )

    x = torch.zeros((1, 32), dtype=torch.bfloat16).expand(2**24, 32)
    weight = torch.zeros((128, 32), dtype=torch.float8_e4m3fn)
    scales = torch.ones((128, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="offsets below"):
        triton_mxfp8_linear(x, weight, scales)


@torch.inference_mode()
def test_triton_mxfp8_rejects_mismatched_input_features():
    """Do not silently reshape incompatible features into a different batch."""
    from vllm.model_executor.layers.quantization.utils.triton_mxfp8 import (
        triton_mxfp8_linear,
    )

    x = torch.zeros((2, 64), dtype=torch.bfloat16)
    weight = torch.zeros((128, 32), dtype=torch.float8_e4m3fn)
    scales = torch.ones((128, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="matching positive K"):
        triton_mxfp8_linear(x, weight, scales)
