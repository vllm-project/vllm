# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear import (
    FlashInferTrtllmMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.flashinfer import has_flashinfer

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_flashinfer()
    ),
    reason="FlashInfer TRTLLM MXFP8 requires an SM100-family GPU",
)


def _make_layer(weight: torch.Tensor) -> torch.nn.Module:
    from flashinfer import SfLayout, mxfp8_quantize

    weight_mxfp8, weight_scale = mxfp8_quantize(
        weight,
        sf_swizzle_layout=SfLayout.layout_linear,
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight_mxfp8, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        weight_scale.view(weight.shape[0], weight.shape[1] // 32),
        requires_grad=False,
    )
    return layer


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("scale_rows", [1, 32])
@torch.inference_mode()
def test_cuda_k32_mma_tail_strides_and_batch_composition(
    grouped: bool, packed: bool, scale_rows: int
) -> None:
    """K32 products retain row semantics for strided and packed E8M0 packets."""
    from vllm.model_executor.kernels.batch_invariant import block32_scaled_mm

    generator = torch.Generator().manual_seed(1931)
    m, n, k, groups = 7, 49, 128, 2 if grouped else 1
    a = torch.randint(-8, 9, (m, groups, k), generator=generator).to(
        torch.float8_e4m3fn
    )
    weight = torch.randint(-8, 9, (groups, n, k), generator=generator).to(a.dtype)
    sa = torch.randint(126, 130, (m, groups, k // 32), generator=generator).byte()
    sw = torch.randint(
        126,
        130,
        (groups, (n + scale_rows - 1) // scale_rows, k // 32),
        generator=generator,
    ).byte()
    expected = torch.zeros(m, groups, n)
    a_factor = torch.ldexp(torch.ones_like(sa, dtype=torch.float32), sa.int() - 127)
    w_factor = torch.ldexp(torch.ones_like(sw, dtype=torch.float32), sw.int() - 127)
    w_factor = w_factor.repeat_interleave(scale_rows, dim=1)[:, :n]
    for block in range(k // 32):
        start, end = block * 32, (block + 1) * 32
        product = (
            torch.bmm(
                a[:, :, start:end].double().transpose(0, 1),
                weight[:, :, start:end].double().transpose(1, 2),
            )
            .transpose(0, 1)
            .float()
        )
        expected += product * a_factor[:, :, block, None] * w_factor[:, :, block]
    if packed:
        sa = sa.contiguous().view(torch.int32)
        sw = sw.contiguous().view(torch.int32)
    # Preserve logical values in a physically strided packet.
    a_device = torch.empty(m, groups, k * 2, device="cuda", dtype=a.dtype)[:, :, ::2]
    a_device.copy_(a)
    w_device = weight.cuda()
    sa_device = sa.cuda().transpose(0, 1).contiguous().transpose(0, 1)
    sw_device = sw.cuda()
    if not grouped:
        a_device, sa_device = a_device[:, 0], sa_device[:, 0]
        w_device, sw_device, expected = w_device[0], sw_device[0], expected[:, 0]
    actual = block32_scaled_mm(
        a_device,
        sa_device,
        w_device,
        sw_device,
        torch.float32,
        grouped=grouped,
        weight_scale_group_rows=scale_rows,
    )
    assert torch.equal(actual.cpu(), expected)
    single = block32_scaled_mm(
        a_device[:1],
        sa_device[:1],
        w_device,
        sw_device,
        torch.float32,
        grouped=grouped,
        weight_scale_group_rows=scale_rows,
    )
    assert torch.equal(actual[:1], single)
    torch.library.opcheck(
        torch.ops.vllm_batch_invariant.block32_scaled_mm.default,
        (
            a_device,
            sa_device,
            w_device,
            sw_device,
            torch.empty_like(actual),
            grouped,
            scale_rows,
        ),
        test_utils=("test_faketensor", "test_aot_dispatch_dynamic"),
    )
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = block32_scaled_mm(
            a_device,
            sa_device,
            w_device,
            sw_device,
            torch.float32,
            grouped=grouped,
            weight_scale_group_rows=scale_rows,
        )
    a_device.zero_()
    replayed.fill_(float("nan"))
    graph.replay()
    assert torch.count_nonzero(replayed).item() == 0


@pytest.mark.parametrize("shape", [(1, 130, 256), (7, 256, 512), (128, 130, 768)])
@torch.inference_mode()
def test_flashinfer_trtllm_mxfp8_linear_numerics(
    shape: tuple[int, int, int],
) -> None:
    torch.manual_seed(0)
    m, n, k = shape
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    layer = _make_layer(weight)
    kernel = FlashInferTrtllmMxfp8LinearKernel(
        Mxfp8LinearLayerConfig(weight_shape=(n, k))
    )
    kernel.process_weights_after_loading(layer)

    output = kernel.apply_weights(layer, x)
    reference = torch.mm(x, weight.t())
    similarity = F.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    )

    assert output.shape == (m, n)
    assert output.is_contiguous()
    assert similarity.item() > 0.98


@torch.inference_mode()
def test_flashinfer_trtllm_mxfp8_custom_ops() -> None:
    x = torch.randn((7, 512), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda")
    layer = _make_layer(weight)
    kernel = FlashInferTrtllmMxfp8LinearKernel(
        Mxfp8LinearLayerConfig(weight_shape=(256, 512))
    )
    kernel.process_weights_after_loading(layer)

    torch.library.opcheck(
        torch.ops.vllm.flashinfer_mxfp8_quantize_8x4.default,
        (x,),
    )
    x_mxfp8, x_scale = vllm_flashinfer.flashinfer_mxfp8_quantize_8x4(x)
    # SchemaCheckMode compares inputs with allclose, which CUDA does not
    # implement for float8. The numerical tests above guard input mutation.
    torch.library.opcheck(
        torch.ops.vllm.mm_mxfp8.default,
        (
            x_mxfp8,
            layer.weight.t(),
            x_scale,
            layer.weight_scale,
            torch.bfloat16,
            "trtllm",
            True,
        ),
        test_utils=(
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ),
    )


@torch.inference_mode()
def test_flashinfer_trtllm_mxfp8_linear_cuda_graph() -> None:
    torch.manual_seed(0)
    m, n, k = 7, 130, 512
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    layer = _make_layer(weight)
    kernel = FlashInferTrtllmMxfp8LinearKernel(
        Mxfp8LinearLayerConfig(weight_shape=(n, k))
    )
    kernel.process_weights_after_loading(layer)

    static_x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    kernel.apply_weights(layer, static_x)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = kernel.apply_weights(layer, static_x)

    new_x = torch.randn_like(static_x)
    static_x.copy_(new_x)
    graph.replay()
    eager_output = kernel.apply_weights(layer, new_x)

    torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)


def _k32_operands(generator, m, n, k, groups, scale_lo, scale_hi):
    def codes(*shape):
        x = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
        x[(x & 0x7F) == 0x7F] = 0x3C  # drop the two E4M3 NaN codes
        return x.cuda().view(torch.float8_e4m3fn)

    def scales(*shape):
        return torch.randint(
            scale_lo, scale_hi, shape, generator=generator, dtype=torch.uint8
        ).cuda()

    if groups is None:
        return codes(m, k), scales(m, k // 32), codes(n, k), scales(n, k // 32)

    # DeepGEMM-style MN-major packed INT32 scales for the grouped (wo_a) path.
    def packed(rows, extra):
        s = scales(rows, groups, k // 32).view(rows, groups, k // 128, 4).int()
        words = s[..., 0] | (s[..., 1] << 8) | (s[..., 2] << 16) | (s[..., 3] << 24)
        store = torch.zeros(groups, k // 128, rows + extra, dtype=torch.int32).cuda()
        store[:, :, :rows] = words.permute(1, 2, 0)
        return store.permute(2, 0, 1)[:rows]

    return (
        codes(m, groups, k),
        packed(m, 3),
        codes(groups, n, k),
        packed(n, 0).permute(1, 0, 2),
    )


K32_TILED_MS = sorted(
    set(range(1, 49)) | set(range(6, 385, 6)) | {127, 129, 255, 257, 1000, 4096, 8192}
)


@pytest.mark.parametrize(
    ("n", "k", "groups", "scale_range"),
    [
        (1792, 5120, None, (110, 135)),  # fused_wqa_wkv
        (4096, 1280, None, (0, 256)),  # indexer wq_b, incl. 2^-127 and NaN scales
        (5120, 2304, None, (73, 180)),  # shared down, edge of the packed FP32 path
        (1024, 4096, 8, (110, 135)),  # grouped wo_a, packed INT32 scales
    ],
)
@torch.inference_mode()
def test_cuda_k32_tiled_kernels_match_reference(n, k, groups, scale_range):
    """Every tiled K32 kernel equals the reference kernel bit for bit at every M,
    including spec-verify sizes (multiples of 6), and a row's output does not
    depend on which other rows share the batch."""
    import vllm.model_executor.kernels.batch_invariant as bi

    bi.require_batch_invariant_kernel()
    op = torch.ops.vllm_batch_invariant.block32_scaled_mm_variant
    generator = torch.Generator().manual_seed(7 + n + k)
    a, sa, w, sw = _k32_operands(
        generator, max(K32_TILED_MS), n, k, groups, *scale_range
    )
    grouped = groups is not None

    def run(variant, rows):
        # Slices keep the strided (MN-major) scale views; index tensors gather.
        a_rows, sa_rows = a[rows], sa[rows]
        shape = (a_rows.shape[0], groups, n) if grouped else (a_rows.shape[0], n)
        out = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        op(a_rows, sa_rows, w, sw, out, grouped, 1, variant)
        return out.view(torch.int16)

    total = max(K32_TILED_MS)
    reference = run(1, slice(0, total))
    for m in K32_TILED_MS:
        for variant in (0, 2, 3, 4):  # auto, skinny, large tile, stream-K gemv
            if (variant == 2 and m > 384) or (variant == 4 and m > 48):
                continue
            assert torch.equal(run(variant, slice(0, m)), reference[:m]), (m, variant)
    for m in (1, 6, 37, 300, 4097):
        rows = torch.randperm(total, generator=generator)[:m].cuda()
        assert torch.equal(run(0, rows), reference[rows]), m


@pytest.mark.parametrize(
    ("n", "k"),
    [(1792, 5120), (8192, 1280), (5120, 2048), (1152, 5120), (5120, 576), (4096, 1280)],
)
@torch.inference_mode()
def test_bi_pinned_mxfp8_mm_rows_independent_of_batch(n, k):
    """The tensor-core BI GEMM (VLLM_BI_MXFP8_GEMM=tensor_core) gives every row
    the same bits at any M, including spec-verify sizes, and any batch-mates."""
    import vllm.model_executor.kernels.linear.mxfp8.batch_invariant  # noqa: F401
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
        swizzle_mxfp8_scale,
    )

    torch.manual_seed(n + k)
    total = max(K32_TILED_MS)
    x = torch.randn(total, k, device="cuda", dtype=torch.bfloat16)
    x[::7] *= 64
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    w_codes, w_scale = mxfp8_e4m3_quantize(weight)
    w_scale = swizzle_mxfp8_scale(w_scale.view(n, k // 32), M=n, K=k)

    def run(rows):
        codes, scale = mxfp8_e4m3_quantize(x[rows], is_sf_swizzled_layout=True)
        out = torch.ops.vllm.bi_pinned_mxfp8_mm(
            codes, w_codes.t(), scale, w_scale, torch.bfloat16
        )
        return out.view(torch.int16)

    reference = run(slice(0, total))
    for m in K32_TILED_MS:
        assert torch.equal(run(slice(0, m)), reference[:m]), m
    for m in (1, 6, 37, 300, 4097):
        rows = torch.randperm(total)[:m].cuda()
        assert torch.equal(run(rows), reference[rows]), m


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="cute-dsl MXFP8 GEMM requires SM100",
)
@pytest.mark.parametrize("m", [1, 6, 96, 384, 8192])
@pytest.mark.parametrize("n,k", [(1792, 5120), (5120, 2048), (1152, 5120)])
@torch.inference_mode()
def test_bi_pinned_mxfp8_mm_tactics_share_bits(m, n, k):
    """The BI GEMM autotunes over every non-split-K tactic, so each one must
    give the same bits as the untuned fallback."""
    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE, _get_cache_buf

    from vllm.model_executor.kernels.linear.mxfp8.batch_invariant import (
        _cute_dsl_mxfp8_runner,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
        swizzle_mxfp8_scale,
    )

    torch.manual_seed(m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    w_codes, w_scale = mxfp8_e4m3_quantize(weight)
    w_scale = swizzle_mxfp8_scale(w_scale.view(n, k // 32), M=n, K=k)
    codes, scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    runner = _cute_dsl_mxfp8_runner(torch.bfloat16)
    workspace = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, x.device)

    def inputs():
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        return [codes, w_codes.t(), scale, w_scale, torch.bfloat16, out, workspace]

    def run(tactic):
        args = inputs()
        runner.forward(args, tactic=tactic)
        return args[5].view(torch.int16)

    tactics = runner.get_valid_tactics(inputs(), None)
    assert tactics and all(t[-1] == 1 for t in tactics)
    reference = run(-1)
    for tactic in tactics:
        assert torch.equal(run(tactic), reference), tactic


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="cute-dsl MXFP8 GEMM requires SM100",
)
@torch.inference_mode()
def test_bi_prequant_qnorm_rows_ignore_batch():
    """VLLM_BI_MXFP8_PREQUANT feeds the fused Q/KV RMSNorm + MXFP8 quant into
    the pinned BI GEMM; a row's GEMM output and KV norm must not depend on how
    many or which rows share the batch."""
    from vllm.model_executor.kernels.linear.mxfp8.batch_invariant import (
        _pinned_mxfp8_mm,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
        swizzle_mxfp8_scale,
    )
    from vllm.models.deepseek_v41.common.ops.query_quant import (
        fused_q_kv_rmsnorm_quant,
    )

    torch.manual_seed(0)
    m, q_size, kv_size, n = 1024, 1536, 512, 2048
    qr = torch.randn(m, q_size, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(m, kv_size, device="cuda", dtype=torch.bfloat16)
    q_weight = (1 + 0.1 * torch.randn(q_size, device="cuda")).bfloat16()
    kv_weight = (1 + 0.1 * torch.randn(kv_size, device="cuda")).bfloat16()
    weight = torch.randn(n, q_size, device="cuda", dtype=torch.bfloat16)
    w_codes, w_scale = mxfp8_e4m3_quantize(weight)
    w_scale = swizzle_mxfp8_scale(w_scale.view(n, -1), M=n, K=q_size)

    def run(rows):
        q, kv_out = fused_q_kv_rmsnorm_quant(
            qr[rows].contiguous(), kv[rows].contiguous(), q_weight, kv_weight, 1e-6
        )
        out = _pinned_mxfp8_mm(
            q.data.reshape(-1, q_size), w_codes.t(), q.scale, w_scale, torch.bfloat16
        )
        return out, kv_out

    full_out, full_kv = run(slice(None))
    perm = torch.randperm(m, device="cuda")
    for rows in (slice(0, 1), slice(0, 6), slice(7, 136), slice(-129, None), perm):
        out, kv_out = run(rows)
        assert torch.equal(out, full_out[rows])
        assert torch.equal(kv_out, full_kv[rows])
