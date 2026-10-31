# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm._custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterFp8BlockScaledMMKernel,
)
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from vllm.platforms.rocm import on_gfx1151

pytestmark = pytest.mark.skipif(not on_gfx1151(), reason="gfx1151 only")


def _fp8_dtype() -> torch.dtype:
    return current_platform.fp8_dtype()


def _make_case(m: int, n: int, k: int = 4096, pad_cols: int = 0):
    device = "cuda"
    fp8 = _fp8_dtype()

    # Keep values safely representable and avoid synthetic quant-path NaNs.
    a_f32 = torch.randn(m, k, device=device, dtype=torch.float32) * 0.1
    w_f32 = torch.randn(n, k, device=device, dtype=torch.float32) * 0.1

    # Explicitly build a block-scaled payload with the exact logical layouts
    # used by the single-weight design.
    a_scale = torch.rand(m, k // 128, device=device, dtype=torch.float32) * 0.05 + 0.01
    w_scale = (
        torch.rand(n // 128, k // 128, device=device, dtype=torch.float32) * 0.05 + 0.01
    )

    a_q = torch.clamp(a_f32 / a_scale.repeat_interleave(128, dim=1), -200.0, 200.0).to(
        fp8
    )
    w_q = torch.clamp(
        w_f32 / w_scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1),
        -200.0,
        200.0,
    ).to(fp8)

    weight = w_q.contiguous()
    if pad_cols:
        # Mirror _maybe_pad_fp8_weight in the real kpack-fix ROCm loader.
        # The returned [N,K] view retains the padded physical row pitch.
        weight = torch.nn.functional.pad(weight, (0, pad_cols))[:, :k]
        assert weight.shape == (n, k)
        assert weight.stride() == (k + pad_cols, 1)
        assert not weight.is_contiguous()

    return a_q.contiguous(), weight, a_scale.contiguous(), w_scale.contiguous()


@pytest.mark.parametrize("pad_cols", [0, 256])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("n", [1536, 4096])
def test_wvsplitk_blockscale_matches_plain_aiter(m: int, n: int, pad_cols: int):
    a, w, a_scale, w_scale = _make_case(m, n, pad_cols=pad_cols)
    cu = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Exact known-good RDNA AITER Triton blockscale fallback contract.
    ref = rocm_aiter_ops.triton_gemm_a8w8_blockscale(
        a,
        w,
        a_scale,
        w_scale,
        [128, 128],
        output_dtype=torch.bfloat16,
    )
    got = ops.wvSplitKQBlockScale(w, a, a_scale, w_scale, cu)

    torch.testing.assert_close(got, ref, rtol=5e-2, atol=1e-1)
    assert torch.isfinite(got).all()
    # Both physically contiguous and production-padded native AITER layouts.
    assert w.stride() == (4096 + pad_cols, 1)
    assert a_scale.shape == (m, 32)


@pytest.mark.parametrize("pad_cols", [0, 256])
@pytest.mark.parametrize("n", [1536, 4096])
def test_wvsplitk_blockscale_m6_graph_replay_changes_inputs(n: int, pad_cols: int):
    m, k = 6, 4096
    a, w, a_scale, w_scale = _make_case(m, n, k, pad_cols=pad_cols)
    cu = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Capture with persistent input buffers, then mutate them before replay.
    a_buf = a.clone()
    as_buf = a_scale.clone()

    graph = torch.cuda.CUDAGraph()
    # Initialize native dispatch before capture, on a side stream.
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        ops.wvSplitKQBlockScale(w, a_buf, as_buf, w_scale, cu)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        out = ops.wvSplitKQBlockScale(w, a_buf, as_buf, w_scale, cu)

    graph.replay()
    first = out.clone()

    a2, _, as2, _ = _make_case(m, n, k)
    a_buf.copy_(a2)
    as_buf.copy_(as2)
    graph.replay()
    second = out.clone()

    assert not torch.equal(first, second)
    ref2 = rocm_aiter_ops.triton_gemm_a8w8_blockscale(
        a2,
        w,
        as2,
        w_scale,
        [128, 128],
        output_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(second, ref2, rtol=5e-2, atol=1e-1)


@pytest.mark.parametrize("pad_cols", [0, 256])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("n", [1536, 4096])
def test_native_aiter_fallback_non_skinny_sizes(m: int, n: int, pad_cols: int):
    # Original AITER remains callable on native weights for all shapes,
    # including M=8 which must never use the new skinny route.
    a, w, a_scale, w_scale = _make_case(m, n, pad_cols=pad_cols)
    ref = rocm_aiter_ops.triton_gemm_a8w8_blockscale(
        a, w, a_scale, w_scale, [128, 128], output_dtype=torch.bfloat16
    )
    assert ref.shape == (m, n)
    assert torch.isfinite(ref).all()


@pytest.mark.parametrize("pad_cols", [0, 256])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("n", [1536, 4096])
def test_only_marked_native_aiter_instances_dispatch_skinny(
    monkeypatch: pytest.MonkeyPatch, m: int, n: int, pad_cols: int
):
    a, w, a_scale, w_scale = _make_case(m, n, pad_cols=pad_cols)
    cu = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Exercise the REAL AITER backend method with minimal instance state.
    # This establishes that unmarked kernels and M=8 retain the original
    # tuned Triton fallback and that marked M<=6 kernels call skinny.
    native = SimpleNamespace(
        config=SimpleNamespace(out_dtype=torch.bfloat16),
        use_triton=True,
        weight_group_shape=(128, 128),
    )
    expected = AiterFp8BlockScaledMMKernel.apply_block_scaled_mm(
        native, a, w, a_scale, w_scale
    )
    calls = []
    original = ops.wvSplitKQBlockScale

    def traced_skinny(*args):
        calls.append((args[1].shape[0], args[0].shape[0]))
        return original(*args)

    monkeypatch.setattr(ops, "wvSplitKQBlockScale", traced_skinny)
    opted_in = SimpleNamespace(
        config=SimpleNamespace(out_dtype=torch.bfloat16),
        use_triton=True,
        weight_group_shape=(128, 128),
        _gfx1151_skinny_blockscale_n=n,
        _gfx1151_skinny_cu_count=cu,
    )
    actual = AiterFp8BlockScaledMMKernel.apply_block_scaled_mm(
        opted_in, a, w, a_scale, w_scale
    )
    expected_calls = [(m, n)] if m <= 6 else []
    assert calls == expected_calls
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=1e-1)


@pytest.mark.parametrize("n", [1536, 4096])
def test_invalid_weight_layout_is_rejected(n: int):
    """HIP must reject unsupported strides, not misinterpret their bytes."""
    a, w, a_scale, w_scale = _make_case(1, n)
    cu = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Unit K-stride is required for 16-byte weight vector loads.
    nonunit_k = torch.empty((n, 8192), device=w.device, dtype=w.dtype)[:, ::2]
    assert nonunit_k.shape == w.shape
    assert nonunit_k.stride(1) == 2
    with pytest.raises(RuntimeError, match="unit K stride"):
        ops.wvSplitKQBlockScale(nonunit_k, a, a_scale, w_scale, cu)

    # A row pitch smaller than K cannot describe valid non-overlapping rows.
    overlapping = torch.as_strided(w, (n, 4096), (4080, 1))
    with pytest.raises(RuntimeError, match="unit K stride"):
        ops.wvSplitKQBlockScale(overlapping, a, a_scale, w_scale, cu)

    # Non-vector-aligned row pitch is similarly unsupported.
    unaligned = torch.nn.functional.pad(w, (0, 1))[:, :4096]
    assert unaligned.stride() == (4097, 1)
    with pytest.raises(RuntimeError, match="unit K stride"):
        ops.wvSplitKQBlockScale(unaligned, a, a_scale, w_scale, cu)
