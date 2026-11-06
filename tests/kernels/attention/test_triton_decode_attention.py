# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.v1.attention.ops.triton_decode_attention as triton_decode_attention
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

pytestmark = pytest.mark.skip_global_cleanup

DEVICE_TYPE = current_platform.device_type


@pytest.mark.parametrize("B", [3, 5])
@pytest.mark.parametrize("L", [1027, 1025])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D_QK", [128, 192, 576])
@pytest.mark.parametrize("D_V", [128, 512])
@pytest.mark.parametrize("CACHE_SIZE", [16384])
@pytest.mark.parametrize("PAGE_SIZE", [1, 16])
def test_decode_attention(B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE):
    assert CACHE_SIZE % PAGE_SIZE == 0
    dtype = torch.bfloat16
    seq_len = L  # This represents the number of tokens already in the sequence
    sm_scale = 1.0 / (D_QK**0.5)
    num_kv_splits = 8

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.randint(
        0, CACHE_SIZE // PAGE_SIZE, (B, num_pages_per_batch, 1), device=DEVICE_TYPE
    )
    req_to_token = req_to_page * PAGE_SIZE
    req_to_token = req_to_token.expand(B, num_pages_per_batch, PAGE_SIZE)
    req_to_token = req_to_token + torch.arange(PAGE_SIZE, device=DEVICE_TYPE).view(
        1, 1, -1
    )
    req_to_token = req_to_token.view(B, -1)
    req_to_token = req_to_token[:, :seq_len].contiguous()

    # q represents the new token being generated, one per batch
    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE)

    # k_buffer and v_buffer represent all previous tokens
    # Page size is 1.
    k_buffer = torch.randn(CACHE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE)
    v_buffer = torch.randn(CACHE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE)

    # o will have the same shape as q
    o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)

    lse = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)

    b_seq_len = torch.full((B,), seq_len, device=DEVICE_TYPE)

    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1),
        dtype=torch.float32,
        device=DEVICE_TYPE,
    )

    # Call the original implementation.
    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o,
        lse,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
    )

    # Page size can be larger than 1.
    k_buffer = k_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
    v_buffer = v_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)

    o1 = torch.zeros_like(o)
    lse1 = torch.zeros_like(lse)

    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o1,
        lse1,
        req_to_page,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        PAGE_SIZE,
    )

    assert torch.allclose(o, o1)


def _quantize_to_fp8(tensor: torch.Tensor):
    """Quantize a BF16 tensor to FP8 e4m3fn with per-tensor scale.

    Returns (fp8_tensor, scale) where:
        fp8_tensor ≈ tensor / scale  (stored as float8_e4m3fn)
        tensor ≈ fp8_tensor.to(float32) * scale  (dequantized)
    """
    amax = tensor.abs().amax()
    # float8_e4m3fn max representable value is 448.0
    scale = (amax / 448.0).clamp(min=1e-12).to(torch.float32)
    fp8_tensor = (
        (tensor.to(torch.float32) / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    return fp8_tensor, scale


def _make_mla_single_split_case(dtype, fp8_kv=False):
    batch, heads = 2, 16
    # Existing 576/512 FP8 exceeds SM120's shared-memory limit, so exercise the
    # other production MLA geometry for FP8.
    d_qk, d_v = (288, 256) if fp8_kv else (576, 512)
    page_size, seq_len = 16, 129
    pages_per_batch = cdiv(seq_len, page_size)
    num_pages = batch * pages_per_batch

    q = torch.randn(batch, heads, d_qk, dtype=dtype, device=DEVICE_TYPE)
    kv = torch.randn(num_pages, page_size, 1, d_qk, dtype=dtype, device=DEVICE_TYPE)
    if fp8_kv:
        kv, scale = _quantize_to_fp8(kv)
    else:
        scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE_TYPE)
    v = kv[..., :d_v]
    block_table = torch.arange(num_pages, device=DEVICE_TYPE, dtype=torch.int32).view(
        batch, pages_per_batch
    )
    seq_lens = torch.tensor([seq_len, 0], device=DEVICE_TYPE, dtype=torch.int32)
    o = torch.empty(batch, heads, d_v, dtype=dtype, device=DEVICE_TYPE)
    lse = torch.empty(batch, heads, dtype=dtype, device=DEVICE_TYPE)
    attn_logits = torch.empty(
        batch, heads, 1, d_v + 1, dtype=torch.float32, device=DEVICE_TYPE
    )
    return {
        "q": q,
        "kv": kv,
        "v": v,
        "block_table": block_table,
        "seq_lens": seq_lens,
        "o": o,
        "lse": lse,
        "attn_logits": attn_logits,
        "scale": scale,
        "page_size": page_size,
        "d_qk": d_qk,
        "seq_len": seq_len,
    }


def _run_mla_single_split_legacy(case):
    """Run the pre-optimization two-stage path as an exact oracle."""
    o_ref = torch.empty_like(case["o"])
    lse_ref = torch.empty_like(case["lse"])
    ref_logits = torch.empty_like(case["attn_logits"])
    triton_decode_attention._decode_grouped_att_m_fwd(
        case["q"],
        case["kv"],
        case["v"],
        ref_logits,
        case["block_table"],
        case["seq_lens"],
        1,
        case["d_qk"] ** -0.5,
        case["page_size"],
        0.0,
        case["scale"],
        case["scale"],
        is_mla=True,
        o=o_ref,
        lse=lse_ref,
        write_direct=False,
    )
    triton_decode_attention._decode_softmax_reducev_fwd(
        ref_logits,
        case["q"],
        o_ref,
        lse_ref,
        case["v"],
        case["seq_lens"],
        1,
    )
    return o_ref, lse_ref


def _run_mla_single_split_direct(case):
    decode_attention_fwd(
        case["q"],
        case["kv"],
        case["v"],
        case["o"],
        case["lse"],
        case["block_table"],
        case["seq_lens"],
        case["attn_logits"],
        num_kv_splits=1,
        sm_scale=case["d_qk"] ** -0.5,
        page_size=case["page_size"],
        k_scale=case["scale"],
        v_scale=case["scale"],
        is_mla=True,
        write_direct=True,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp8_kv", [False, True])
def test_mla_single_split_skips_stage2(monkeypatch, dtype, fp8_kv):
    """A single MLA split needs no second-stage identity reduction."""
    if fp8_kv and not current_platform.has_device_capability(89):
        pytest.skip("FP8 KV cache requires compute capability 8.9 or newer")
    case = _make_mla_single_split_case(dtype, fp8_kv)
    o_ref, lse_ref = _run_mla_single_split_legacy(case)

    stage2_calls = 0
    original_stage2 = triton_decode_attention._decode_softmax_reducev_fwd

    def count_stage2(*args, **kwargs):
        nonlocal stage2_calls
        stage2_calls += 1
        return original_stage2(*args, **kwargs)

    monkeypatch.setattr(
        triton_decode_attention, "_decode_softmax_reducev_fwd", count_stage2
    )
    _run_mla_single_split_direct(case)

    assert stage2_calls == 0
    assert torch.equal(case["o"][0], o_ref[0])
    assert torch.equal(case["lse"][0], lse_ref[0])
    assert torch.count_nonzero(case["o"][1]) == 0
    assert torch.isneginf(case["lse"][1]).all()
    assert torch.isfinite(case["o"]).all()
    assert torch.isfinite(case["lse"][0]).all()


def test_mla_single_split_cuda_graph_replay():
    """Replay reads dynamic sequence lengths and rewrites all output rows."""
    case = _make_mla_single_split_case(torch.bfloat16)
    case["seq_lens"].copy_(torch.tensor([1, 0], device=DEVICE_TYPE, dtype=torch.int32))
    _run_mla_single_split_direct(case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_mla_single_split_direct(case)

    for lengths in ([case["seq_len"], 17], [64, 0]):
        case["seq_lens"].copy_(
            torch.tensor(lengths, device=DEVICE_TYPE, dtype=torch.int32)
        )
        case["o"].fill_(float("nan"))
        case["lse"].fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        ref_o, ref_lse = _run_mla_single_split_legacy(case)
        assert torch.equal(case["o"][0], ref_o[0])
        assert torch.equal(case["lse"][0], ref_lse[0])
        if lengths[1] > 0:
            assert torch.equal(case["o"][1], ref_o[1])
            assert torch.equal(case["lse"][1], ref_lse[1])
        else:
            assert torch.count_nonzero(case["o"][1]) == 0
            assert torch.isneginf(case["lse"][1]).all()


def test_mla_multi_split_keeps_stage2(monkeypatch):
    """A caller cannot enable direct output for multiple MLA splits."""
    case = _make_mla_single_split_case(torch.bfloat16)
    num_splits = 2
    case["attn_logits"] = torch.empty(
        *case["attn_logits"].shape[:2],
        num_splits,
        case["attn_logits"].shape[-1],
        dtype=torch.float32,
        device=DEVICE_TYPE,
    )
    stage2_calls = 0
    original_stage2 = triton_decode_attention._decode_softmax_reducev_fwd

    def count_stage2(*args, **kwargs):
        nonlocal stage2_calls
        stage2_calls += 1
        return original_stage2(*args, **kwargs)

    monkeypatch.setattr(
        triton_decode_attention, "_decode_softmax_reducev_fwd", count_stage2
    )
    decode_attention_fwd(
        case["q"],
        case["kv"],
        case["v"],
        case["o"],
        case["lse"],
        case["block_table"],
        case["seq_lens"],
        case["attn_logits"],
        num_splits,
        case["d_qk"] ** -0.5,
        case["page_size"],
        k_scale=case["scale"],
        v_scale=case["scale"],
        is_mla=True,
        write_direct=True,
    )
    assert stage2_calls == 1


@pytest.mark.parametrize("B", [3])
@pytest.mark.parametrize("L", [1025])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D_QK", [128, 576])
@pytest.mark.parametrize("D_V", [128, 512])
@pytest.mark.parametrize("CACHE_SIZE", [16384])
@pytest.mark.parametrize("PAGE_SIZE", [1, 16])
def test_decode_attention_fp8(B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE):
    """Test FP8 KV cache path: quantize K/V to FP8, run kernel with scales,
    and compare against BF16 reference output."""
    assert CACHE_SIZE % PAGE_SIZE == 0
    dtype = torch.bfloat16
    seq_len = L
    sm_scale = 1.0 / (D_QK**0.5)
    num_kv_splits = 8

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.randint(
        0, CACHE_SIZE // PAGE_SIZE, (B, num_pages_per_batch, 1), device=DEVICE_TYPE
    )
    req_to_token = req_to_page * PAGE_SIZE
    req_to_token = req_to_token.expand(B, num_pages_per_batch, PAGE_SIZE)
    req_to_token = req_to_token + torch.arange(PAGE_SIZE, device=DEVICE_TYPE).view(
        1, 1, -1
    )
    req_to_token = req_to_token.view(B, -1)
    req_to_token = req_to_token[:, :seq_len].contiguous()

    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE)

    # Create BF16 K/V as reference
    k_bf16 = torch.randn(CACHE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE)
    v_bf16 = torch.randn(CACHE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE)

    # --- BF16 reference ---
    o_ref = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
    lse_ref = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)
    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
    )

    if PAGE_SIZE == 1:
        decode_attention_fwd(
            q,
            k_bf16,
            v_bf16,
            o_ref,
            lse_ref,
            req_to_token,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
        )
    else:
        k_paged = k_bf16.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
        v_paged = v_bf16.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)
        decode_attention_fwd(
            q,
            k_paged,
            v_paged,
            o_ref,
            lse_ref,
            req_to_page,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            page_size=PAGE_SIZE,
        )

    # --- FP8 path ---
    k_fp8, k_scale = _quantize_to_fp8(k_bf16)
    v_fp8, v_scale = _quantize_to_fp8(v_bf16)

    o_fp8 = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
    lse_fp8 = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)
    attn_logits_fp8 = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
    )

    if PAGE_SIZE == 1:
        decode_attention_fwd(
            q,
            k_fp8,
            v_fp8,
            o_fp8,
            lse_fp8,
            req_to_token,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits_fp8,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    else:
        k_fp8_paged = k_fp8.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
        v_fp8_paged = v_fp8.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)
        decode_attention_fwd(
            q,
            k_fp8_paged,
            v_fp8_paged,
            o_fp8,
            lse_fp8,
            req_to_page,
            b_seq_len=torch.full((B,), seq_len, device=DEVICE_TYPE),
            attn_logits=attn_logits_fp8,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            page_size=PAGE_SIZE,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    # FP8 tolerances match test_mla_backends.py test_backend_correctness.
    torch.testing.assert_close(o_ref, o_fp8, atol=5e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "H_Q,H_KV,D_QK,D_V,is_mla",
    [
        (16, 1, 576, 512, True),  # MLA path (grouped kernel, v = trans(k))
        (32, 8, 128, 128, False),  # GQA path (grouped kernel)
        (32, 32, 128, 128, False),  # MHA path (normal kernel)
    ],
)
@pytest.mark.parametrize("PAGE_SIZE", [16])
def test_decode_attention_cross_layer_view(H_Q, H_KV, D_QK, D_V, is_mla, PAGE_SIZE):
    """The kernel must honor the cache's page-dim stride, not assume pages are
    packed back-to-back. A per-layer view into a cross-layer (block-major)
    cache has stride(0) inflated by num_layers; outputs must match a
    contiguous cache holding the same data exactly."""
    B = 3
    seq_len = 1027
    CACHE_SIZE = 16384
    NUM_LAYERS = 3
    LAYER_IDX = 1
    dtype = torch.bfloat16
    sm_scale = 1.0 / (D_QK**0.5)
    num_kv_splits = 8
    num_pages = CACHE_SIZE // PAGE_SIZE

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.randint(
        0, num_pages, (B, num_pages_per_batch), device=DEVICE_TYPE
    )

    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE)
    b_seq_len = torch.full((B,), seq_len, device=DEVICE_TYPE)

    # Reference: contiguous paged cache.
    k_ref = torch.randn(
        num_pages, PAGE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE
    )
    if is_mla:
        v_ref = k_ref[..., :D_V]
    else:
        v_ref = torch.randn(
            num_pages, PAGE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE
        )

    # Cross-layer cache: all layers' pages for a block are adjacent. The
    # per-layer view has the same shape as the contiguous cache but
    # stride(0) is NUM_LAYERS x larger. Neighbor layers hold random data so
    # any packed-pages addressing reads garbage rather than zeros.
    k_xl = torch.randn(
        num_pages, NUM_LAYERS, PAGE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE
    )
    k_view = k_xl[:, LAYER_IDX]
    k_view.copy_(k_ref)
    assert k_view.stride(0) == NUM_LAYERS * PAGE_SIZE * H_KV * D_QK
    if is_mla:
        v_view = k_view[..., :D_V]
    else:
        v_xl = torch.randn(
            num_pages, NUM_LAYERS, PAGE_SIZE, H_KV, D_V, dtype=dtype, device=DEVICE_TYPE
        )
        v_view = v_xl[:, LAYER_IDX]
        v_view.copy_(v_ref)

    def run(k_buffer, v_buffer):
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
        lse = torch.zeros(B, H_Q, dtype=dtype, device=DEVICE_TYPE)
        attn_logits = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
        )
        decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_page,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            PAGE_SIZE,
            is_mla=is_mla,
        )
        return o, lse

    o_ref, lse_ref = run(k_ref, v_ref)
    o_xl, lse_xl = run(k_view, v_view)

    # Same data and same compute order; only addressing differs.
    assert torch.equal(o_ref, o_xl)
    assert torch.equal(lse_ref, lse_xl)


@pytest.mark.parametrize(
    "B,H_Q,H_KV,D_QK,D_V,is_mla",
    [
        (256, 32, 1, 576, 512, True),  # grouped kernel
        (1024, 32, 32, 128, 128, False),  # normal kernel
    ],
)
def test_decode_attention_int64_logits_offset(B, H_Q, H_KV, D_QK, D_V, is_mla):
    """attn_logits offsets must not overflow int32 when B * H_Q * num_kv_splits
    * (D_V + 1) exceeds 2**31 (long max_model_len, large decode batch)."""
    PAGE_SIZE = 16
    seq_len = 33
    num_kv_splits = 512
    dtype = torch.bfloat16
    sm_scale = 1.0 / (D_QK**0.5)
    assert B * H_Q * num_kv_splits * (D_V + 1) > 2**31

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.arange(
        B * num_pages_per_batch, device=DEVICE_TYPE, dtype=torch.int32
    ).view(B, num_pages_per_batch)
    k_buffer = torch.randn(
        B * num_pages_per_batch, PAGE_SIZE, H_KV, D_QK, dtype=dtype, device=DEVICE_TYPE
    )
    v_buffer = (
        k_buffer[..., :D_V]
        if is_mla
        else torch.randn(
            B * num_pages_per_batch,
            PAGE_SIZE,
            H_KV,
            D_V,
            dtype=dtype,
            device=DEVICE_TYPE,
        )
    )
    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device=DEVICE_TYPE)
    b_seq_len = torch.full((B,), seq_len, device=DEVICE_TYPE)

    def run(rows):
        n = rows.stop - rows.start
        o = torch.zeros(n, H_Q, D_V, dtype=dtype, device=DEVICE_TYPE)
        lse = torch.zeros(n, H_Q, dtype=dtype, device=DEVICE_TYPE)
        attn_logits = torch.empty(
            (n, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=DEVICE_TYPE
        )
        decode_attention_fwd(
            q[rows],
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_page[rows],
            b_seq_len[rows],
            attn_logits,
            num_kv_splits,
            sm_scale,
            PAGE_SIZE,
            is_mla=is_mla,
        )
        return o, lse

    o, lse = run(slice(0, B))
    o_last, lse_last = run(slice(B - 1, B))

    # Same compute order; the last request only differs in its logits offset.
    assert torch.equal(o[-1:], o_last)
    assert torch.equal(lse[-1:], lse_last)


def _verify_reference(q, cache, block_table, seq_lens, real_seq_lens, qlen, scale):
    """Causal MLA verify in the dtype of ``cache``: row t of a request attends
    its first min(L - qlen + 1 + t, L_real) tokens; a row with none comes back
    zero. A bf16 ``cache`` also rounds P to bf16 before the PV product, as the
    kernel does."""
    kv_lora_rank = 512
    page_size = cache.shape[1]
    lowp = cache.dtype == torch.bfloat16
    dtype = torch.float32 if lowp else cache.dtype
    out = torch.zeros(
        q.shape[0], q.shape[1], kv_lora_rank, dtype=dtype, device=q.device
    )
    for b, seq_len in enumerate(seq_lens.tolist()):
        real = seq_len if real_seq_lens is None else real_seq_lens[b].item()
        lim = (torch.arange(qlen) + seq_len - qlen + 1).clamp(0, real).tolist()
        if max(lim) == 0:
            continue
        num_pages = cdiv(seq_len, page_size)
        kv = cache[block_table[b, :num_pages].long()].reshape(-1, cache.shape[-1])
        kv = kv[: max(lim)].to(dtype)
        rows = slice(b * qlen, (b + 1) * qlen)
        s = torch.einsum("thd,kd->thk", q[rows].to(dtype), kv) * scale
        keys = torch.arange(kv.shape[0], device=q.device)
        hidden = keys[None, :] >= torch.tensor(lim, device=q.device)[:, None]
        s = s.masked_fill(hidden[:, None, :], float("-inf"))
        p = torch.exp(s - s.amax(-1, keepdim=True)).nan_to_num(0.0)
        pv = p.bfloat16().to(dtype) if lowp else p
        o = torch.einsum("thk,kd->thd", pv, kv[:, :kv_lora_rank])
        out[rows] = o / p.sum(-1, keepdim=True).clamp(min=1e-30)
    return out.bfloat16().to(dtype) if lowp else out


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Only the ROCm AITER MLA backend uses it"
)
@pytest.mark.parametrize("num_heads", [8, 16, 128])
@pytest.mark.parametrize("qlen", [5, 8])
# 3 requests at 8 or 16 heads stay on the small-tile config; the rest reach
# the 64-row tiles.
@pytest.mark.parametrize("batch", [3, 24])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
# 1 skips stage 2; 32 combines more splits than one stage-2 step holds.
@pytest.mark.parametrize("num_kv_splits", [1, 4, 32])
def test_decode_attention_fwd_verify(
    num_heads, qlen, batch, page_size, kv_dtype, num_kv_splits
):
    from vllm.v1.attention.ops.triton_decode_attention import (
        decode_attention_fwd_verify,
    )

    set_random_seed(0)
    dim, kv_lora_rank = 576, 512
    scale = 1.0 / (192**0.5)
    # Ragged lengths, one shorter than qlen, one long enough for 32 splits,
    # and a graph padding request.
    lens = torch.randint(qlen, 700, (batch,))
    lens[0] = qlen - 2
    lens[1] = 4500
    lens[-1] = 0
    seq_lens = lens.to(torch.int32).to(DEVICE_TYPE)
    pages_per_req = cdiv(int(lens.max()), page_size) + 1
    num_pages = batch * pages_per_req + 3
    block_table = (
        torch.randperm(num_pages - 1, device=DEVICE_TYPE)[: batch * pages_per_req]
        .view(batch, pages_per_req)
        .to(torch.int32)
    )
    cache = torch.randn(num_pages, page_size, dim, device=DEVICE_TYPE) * 0.5
    # Keys past each request's length, and the pages its table does not use,
    # are NaN, so reading any of them fails instead of passing on valid data.
    poison = num_pages - 1
    cache[poison] = float("nan")
    for b, n in enumerate(lens.tolist()):
        used = cdiv(n, page_size)
        block_table[b, used:] = poison
        if n % page_size:
            cache[block_table[b, used - 1], n % page_size :] = float("nan")
    k_scale = torch.tensor(1.0, device=DEVICE_TYPE)
    if kv_dtype == "fp8":
        cache = cache.to(current_platform.fp8_dtype())
        # Not a power of two, so dequantized keys round when cast to bf16.
        k_scale = torch.tensor(0.37, device=DEVICE_TYPE)
    else:
        cache = cache.to(torch.bfloat16)
    exact_cache = cache.double() * k_scale.item()
    q = torch.randn(batch * qlen, num_heads, dim, device=DEVICE_TYPE).to(torch.bfloat16)
    paged = cache.unsqueeze(2)
    for real_seq_lens in (None, (seq_lens - 2).clamp(min=0)):
        o = torch.full(
            (batch * qlen, num_heads, kv_lora_rank),
            float("nan"),
            dtype=torch.bfloat16,
            device=DEVICE_TYPE,
        )
        lse = torch.empty(batch * qlen, num_heads, device=DEVICE_TYPE)
        logits = torch.full(
            (batch * qlen, num_heads, num_kv_splits, kv_lora_rank + 1),
            float("nan"),
            device=DEVICE_TYPE,
        )
        decode_attention_fwd_verify(
            q,
            paged,
            paged[..., :kv_lora_rank],
            o,
            lse,
            block_table,
            seq_lens,
            real_seq_lens,
            qlen,
            logits,
            num_kv_splits,
            scale,
            page_size,
            k_scale=k_scale,
        )
        args = (block_table, seq_lens, real_seq_lens, qlen, scale)
        ref = _verify_reference(q, exact_cache, *args)
        # The kernel rounds the dequantized keys, P and the output to bf16;
        # a reference with the same roundings bounds its error from fp64.
        # Per request, so short requests do not set the long ones' bound.
        ref_bf16 = _verify_reference(q, exact_cache.bfloat16(), *args).double()
        err = (o.double() - ref).abs().view(batch, -1).amax(-1)
        bf16_err = (ref_bf16 - ref).abs().view(batch, -1).amax(-1)
        assert (err <= 2 * bf16_err).all(), (err, bf16_err)


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Only the ROCm AITER MLA backend uses it"
)
# 16 heads take the small-tile config, 128 the 64-row QSPLIT one.
@pytest.mark.parametrize("num_heads", [16, 128])
def test_decode_attention_fwd_verify_int64_page_offset(num_heads):
    """Page offsets must not overflow int32 once the cache holds more than
    2**31 elements (fp8 past 2 GiB)."""
    from vllm.v1.attention.ops.triton_decode_attention import (
        decode_attention_fwd_verify,
    )

    dim, kv_lora_rank, page_size, qlen = 576, 512, 16, 8
    pages_per_req = cdiv(200, page_size)
    # Every page the table points at starts past element 2**31.
    first_high = 2**31 // (page_size * dim) + 1
    num_pages = first_high + 2 * pages_per_req
    torch.accelerator.empty_cache()
    if torch.accelerator.get_memory_info()[0] < num_pages * page_size * dim + 2**30:
        pytest.skip("needs about 3 GiB of free device memory")
    set_random_seed(0)
    fp8 = current_platform.fp8_dtype()
    big = torch.empty(num_pages, page_size, 1, dim, dtype=fp8, device=DEVICE_TYPE)
    seq_lens = torch.tensor([200, 37], dtype=torch.int32, device=DEVICE_TYPE)
    high = torch.arange(first_high, num_pages, device=DEVICE_TYPE)
    block_table = high.view(2, pages_per_req).to(torch.int32)
    data = torch.randn(2 * pages_per_req, page_size, 1, dim, device=DEVICE_TYPE)
    big[high] = data.to(fp8)
    small = big[high].clone()
    small_table = torch.arange(2 * pages_per_req, device=DEVICE_TYPE)
    small_table = small_table.view(2, pages_per_req).to(torch.int32)
    q = torch.randn(2 * qlen, num_heads, dim, device=DEVICE_TYPE).to(torch.bfloat16)
    k_scale = torch.tensor(0.37, device=DEVICE_TYPE)

    def run(cache, table):
        rows = 2 * qlen
        o = torch.empty(
            rows, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=DEVICE_TYPE
        )
        lse = torch.empty(rows, num_heads, device=DEVICE_TYPE)
        logits = torch.empty(rows, num_heads, 2, kv_lora_rank + 1, device=DEVICE_TYPE)
        decode_attention_fwd_verify(
            q,
            cache,
            cache[..., :kv_lora_rank],
            o,
            lse,
            table,
            seq_lens,
            None,
            qlen,
            logits,
            2,
            dim**-0.5,
            page_size,
            k_scale=k_scale,
        )
        return o, lse

    o_big, lse_big = run(big, block_table)
    o_small, lse_small = run(small, small_table)
    # Same data and compute order; only the page offsets differ.
    assert torch.equal(o_big, o_small)
    assert torch.equal(lse_big, lse_small)
