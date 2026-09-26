# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.models.inkling.nvidia.ops.sconv as sconv
from vllm.models.inkling.nvidia.ops import qkvr_prep
from vllm.platforms import current_platform

_cap = current_platform.get_device_capability() if current_platform.is_cuda() else None


def _make_inputs(*, is_local: bool, tokens: int = 33, tp_size: int = 4):
    torch.manual_seed(0)
    heads = 64 // tp_size
    kv_heads = (16 if is_local else 8) // tp_size
    head_dim = 128
    d_rel = 16
    rel_extent = 512 if is_local else 1024
    page_size = 16
    num_blocks = (tokens + page_size - 1) // page_size
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    r_width = heads * d_rel
    device = "cuda"

    qkvr = torch.randn(
        tokens,
        q_width + 2 * kv_width + r_width,
        device=device,
        dtype=torch.bfloat16,
    )
    k_weight = torch.randn(kv_width, 4, device=device, dtype=torch.bfloat16)
    v_weight = torch.randn_like(k_weight)
    q_norm_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
    k_norm_weight = torch.randn_like(q_norm_weight)
    rel_proj = torch.randn(d_rel, rel_extent, device=device, dtype=torch.bfloat16)
    conv_cache = torch.zeros(
        num_blocks,
        kv_heads,
        page_size,
        2 * head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    key_cache = torch.empty(
        num_blocks,
        page_size,
        kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    value_cache = torch.empty_like(key_cache)
    positions = torch.arange(tokens, device=device, dtype=torch.int64)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)[None]
    seq_idx = torch.zeros(tokens, device=device, dtype=torch.int32)
    slots = torch.arange(tokens, device=device, dtype=torch.int64)
    query_start = torch.zeros(tokens, device=device, dtype=torch.int32)
    log_scaling_n_floor = None if is_local else 128000
    log_scaling = None
    if log_scaling_n_floor is not None:
        log_scaling = torch.linspace(
            1.0,
            1.1,
            tokens,
            device=device,
            dtype=torch.float32,
        )
    return (
        qkvr,
        k_weight,
        v_weight,
        q_norm_weight,
        k_norm_weight,
        rel_proj,
        1e-6,
        heads,
        kv_heads,
        head_dim,
        d_rel,
        conv_cache,
        key_cache,
        value_cache,
        positions,
        block_table,
        seq_idx,
        slots,
        query_start,
        slots,
        0,
        head_dim,
        page_size,
        log_scaling,
    )


def _reference(args):
    qkvr = args[0]
    heads, kv_heads, head_dim, d_rel = args[7:11]
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    q, k, v, r = qkvr.split((q_width, kv_width, kv_width, heads * d_rel), dim=1)
    conv_cache = args[11].clone()
    k = sconv.fused_sconv(
        k.contiguous(),
        args[1],
        conv_cache,
        args[14],
        args[15],
        args[16],
        args[17],
        args[18],
        args[20],
        head_dim,
        args[22],
    )
    v = sconv.fused_sconv(
        v.contiguous(),
        args[2],
        conv_cache,
        args[14],
        args[15],
        args[16],
        args[17],
        args[18],
        args[21],
        head_dim,
        args[22],
    )

    def rms_norm(x, weight):
        x = x.reshape(-1, head_dim).float()
        rstd = torch.rsqrt(x.square().mean(1, keepdim=True) + args[6])
        return (x * rstd * weight.float()).to(qkvr.dtype)

    q = rms_norm(q, args[3]).view(qkvr.shape[0], heads, head_dim)
    k = rms_norm(k, args[4]).view(qkvr.shape[0], kv_heads, head_dim)
    v = v.view(qkvr.shape[0], kv_heads, head_dim)
    rel = torch.mm(r.reshape(-1, d_rel), args[5]).view(qkvr.shape[0], heads, -1)
    if args[23] is not None:
        q = (q.float() * args[23][:, None, None]).to(q.dtype)
        rel = (rel.float() * args[23][:, None, None]).to(rel.dtype)

    key_cache = args[12].clone()
    value_cache = args[13].clone()
    slots = args[19]
    valid = slots >= 0
    key_cache.view(-1, kv_heads, head_dim)[slots[valid]] = k[valid]
    value_cache.view(-1, kv_heads, head_dim)[slots[valid]] = v[valid]
    return q.flatten(1), rel, key_cache, value_cache


@pytest.mark.skipif(
    _cap is None,
    reason="Inkling QKVR prep kernels require CUDA",
)
@pytest.mark.parametrize(
    ("is_local", "tokens", "tp_size"),
    [
        (True, 9, 4),
        (False, 33, 4),
        (True, 33, 8),
        (True, 128, 4),
        (False, 128, 8),
        (True, 512, 4),
        (False, 640, 8),
    ],
)
@torch.inference_mode()
def test_qkvr_prep_matches_reference(is_local, tokens, tp_size):
    args = _make_inputs(is_local=is_local, tokens=tokens, tp_size=tp_size)
    ref_q, ref_rel, ref_key, ref_value = _reference(args)

    q, rel = qkvr_prep.fused_qkvr_prep(*args)

    torch.testing.assert_close(q, ref_q, rtol=0.01, atol=0.02)
    torch.testing.assert_close(rel, ref_rel, rtol=0.02, atol=0.125)
    torch.testing.assert_close(args[12], ref_key, rtol=0.01, atol=0.01)
    torch.testing.assert_close(args[13], ref_value, rtol=0, atol=0)


@pytest.mark.skipif(
    _cap is None,
    reason="Inkling QKVR prep kernels require CUDA",
)
@pytest.mark.parametrize("tokens", [9, 128])
@torch.inference_mode()
def test_qkvr_log_scaling_preserves_bf16_norm_output(tokens):
    args = list(_make_inputs(is_local=False, tokens=tokens, tp_size=8))
    tau = args[23]
    assert tau is not None

    scaled_q, scaled_rel = qkvr_prep.fused_qkvr_prep(*args)
    args[23] = None
    unscaled_q, unscaled_rel = qkvr_prep.fused_qkvr_prep(*args)

    expected_q = (unscaled_q.float() * tau[:, None]).to(unscaled_q.dtype)
    expected_rel = (unscaled_rel.float() * tau[:, None, None]).to(unscaled_rel.dtype)
    torch.testing.assert_close(scaled_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(scaled_rel, expected_rel, rtol=0, atol=0)


@pytest.mark.skipif(
    _cap is None,
    reason="Inkling QKVR prep kernels require CUDA",
)
@pytest.mark.parametrize("tokens", [9, 128])
@torch.inference_mode()
def test_qkvr_prep_negative_slots_skip_cache_writes(tokens):
    args = list(_make_inputs(is_local=True, tokens=tokens))
    args[17].fill_(-1)
    args[19].fill_(-1)
    conv_cache = args[11].clone()
    key_cache = args[12].clone()
    value_cache = args[13].clone()

    qkvr_prep.fused_qkvr_prep(*args)

    torch.testing.assert_close(args[11], conv_cache, rtol=0, atol=0)
    torch.testing.assert_close(args[12], key_cache, rtol=0, atol=0)
    torch.testing.assert_close(args[13], value_cache, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_fused_sconv_negative_slots_skip_cache_writes():
    args = list(_make_inputs(is_local=True, tokens=9))
    args[17].fill_(-1)
    cache = args[11].clone()

    sconv.fused_sconv(
        args[0][:, 16 * 128 : 16 * 128 + 4 * 128].contiguous(),
        args[1],
        args[11],
        args[14],
        args[15],
        args[16],
        args[17],
        args[18],
        0,
        128,
        args[22],
    )

    torch.testing.assert_close(args[11], cache, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("rel_extent", "last_latency_rows", "first_throughput_rows"),
    [(512, 8191, 8192), (1024, 2047, 2048)],
)
def test_rel_projection_schedule_crossover(
    rel_extent, last_latency_rows, first_throughput_rows
):
    assert not qkvr_prep.use_rel_proj_throughput(last_latency_rows, rel_extent)
    assert qkvr_prep.use_rel_proj_throughput(first_throughput_rows, rel_extent)


FP8_DTYPE = torch.float8_e4m3fn


@pytest.mark.skipif(
    _cap is None,
    reason="Inkling QKVR prep kernels require CUDA",
)
@pytest.mark.parametrize(
    ("is_local", "tokens"),
    [
        (True, 9),  # fused small-batch path
        (False, 9),
        (True, 128),  # tiled path
        (False, 640),
    ],
)
@pytest.mark.parametrize(("k_scale_val", "v_scale_val"), [(1.0, 1.0), (0.5, 2.0)])
@torch.inference_mode()
def test_qkvr_prep_fp8_cache_write(is_local, tokens, k_scale_val, v_scale_val):
    """fp8 quant-on-store: dequantized cache contents match the bf16 run,
    q/rel outputs are unaffected, and the conv cache stays bf16-identical."""
    args = list(_make_inputs(is_local=is_local, tokens=tokens, tp_size=4))
    kv_heads = args[8]
    head_dim = args[9]

    bf16_args = list(args)
    bf16_args[11] = args[11].clone()
    bf16_args[12] = args[12].clone()
    bf16_args[13] = args[13].clone()
    q_bf16, rel_bf16 = qkvr_prep.fused_qkvr_prep(*bf16_args)

    args[12] = torch.zeros_like(bf16_args[12], dtype=FP8_DTYPE)
    args[13] = torch.zeros_like(bf16_args[13], dtype=FP8_DTYPE)
    k_scale = torch.tensor(k_scale_val, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(v_scale_val, dtype=torch.float32, device="cuda")
    q_fp8, rel_fp8 = qkvr_prep.fused_qkvr_prep(*args, k_scale=k_scale, v_scale=v_scale)

    # The q/rel path does not depend on the cache dtype.
    torch.testing.assert_close(q_fp8, q_bf16, rtol=0, atol=0)
    torch.testing.assert_close(rel_fp8, rel_bf16, rtol=0, atol=0)

    # Conv state is pinned to the model dtype and must be byte-identical.
    torch.testing.assert_close(args[11], bf16_args[11], rtol=0, atol=0)

    # Dequantized fp8 contents vs the bf16 stores at the written slots.
    # e4m3 half-ulp relative error is 2**-4; atol covers the subnormal
    # granularity floor after descaling.
    written = slice(0, tokens)
    for cache_fp8, cache_bf16, scale in (
        (args[12], bf16_args[12], k_scale_val),
        (args[13], bf16_args[13], v_scale_val),
    ):
        dequant = cache_fp8.view(-1, kv_heads, head_dim)[written].float() * scale
        expected = cache_bf16.view(-1, kv_heads, head_dim)[written].float()
        torch.testing.assert_close(dequant, expected, rtol=0.07, atol=0.02)


@pytest.mark.skipif(
    _cap is None,
    reason="Inkling QKVR prep kernels require CUDA",
)
def test_qkvr_prep_fp8_requires_scales():
    args = list(_make_inputs(is_local=True, tokens=9))
    args[12] = torch.zeros_like(args[12], dtype=FP8_DTYPE)
    args[13] = torch.zeros_like(args[13], dtype=FP8_DTYPE)
    with pytest.raises(AssertionError):
        qkvr_prep.fused_qkvr_prep(*args)
