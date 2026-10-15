# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as sparse_mod
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
    _use_rocm_sparse_triton,
    fit_kpool_indices_to_aiter,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _sparse_kv_row_offset,
    _sparse_query_row_offset,
    _validate_dsv4_sparse_dims,
    _validate_sparse_dims,
)
from vllm.v1.kv_cache_interface import KVCacheLayout


@triton.jit
def _store_sparse_kv_row_offset_kernel(slot_ptr, output_ptr, stride: tl.constexpr):
    slot = tl.load(slot_ptr)
    tl.store(output_ptr, _sparse_kv_row_offset(slot, stride))


def _fit_kpool_indices_reference(
    token_indices: torch.Tensor, topk_tokens: int
) -> torch.Tensor:
    history = token_indices[:, :topk_tokens]
    tail = token_indices[:, topk_tokens:]
    valid_history = (history >= 0).sum(dim=1)
    valid_tail = (tail >= 0).sum(dim=1)
    keep_history = torch.minimum(valid_history, topk_tokens - valid_tail)

    columns = torch.arange(topk_tokens, device=token_indices.device).unsqueeze(0)
    tail_offsets = columns - keep_history.unsqueeze(1)
    tail_values = torch.gather(
        tail, 1, tail_offsets.clamp(min=0, max=tail.shape[1] - 1)
    )
    output = torch.where(columns < keep_history.unsqueeze(1), history, tail_values)
    valid_output = columns < (keep_history + valid_tail).unsqueeze(1)
    return torch.where(valid_output, output, -1)


@triton.jit
def _store_sparse_query_row_offset_kernel(
    query_idx_ptr, output_ptr, stride: tl.constexpr
):
    query_idx = tl.load(query_idx_ptr)
    tl.store(output_ptr, _sparse_query_row_offset(query_idx, stride))


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_fit_kpool_indices_preserves_tail_and_best_history():
    token_indices = torch.tensor(
        [
            [10, 9, 8, 7, 6, 5, 100, 101],
            [10, 9, 8, -1, -1, -1, 100, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    fitted = fit_kpool_indices_to_aiter(token_indices, topk_tokens=6)

    assert fitted.tolist() == [
        [10, 9, 8, 7, 100, 101],
        [10, 9, 8, 100, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ]


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
@pytest.mark.parametrize(
    ("num_tokens", "topk_tokens", "tail_width"),
    [(1, 4, 1), (7, 6, 2), (13, 128, 128), (3, 2048, 128)],
)
def test_fit_kpool_indices_matches_eager_reference(num_tokens, topk_tokens, tail_width):
    """The Triton fit must reproduce the eager packing rule exactly."""
    set_random_seed(topk_tokens + num_tokens)
    width = topk_tokens + tail_width
    token_indices = torch.randint(0, 4096, (num_tokens, width), dtype=torch.int32)
    valid = torch.rand((num_tokens, width)) < 0.5
    valid[0] = False
    valid[-1] = True
    token_indices = torch.where(valid, token_indices, -1).cuda()

    fitted = fit_kpool_indices_to_aiter(token_indices, topk_tokens=topk_tokens)

    torch.testing.assert_close(
        fitted, _fit_kpool_indices_reference(token_indices, topk_tokens)
    )


def test_fit_kpool_indices_exact_width_is_noop():
    token_indices = torch.tensor([[3, 2, 1, -1]], dtype=torch.int32)

    fitted = fit_kpool_indices_to_aiter(token_indices, topk_tokens=4)

    assert fitted.data_ptr() == token_indices.data_ptr()


def test_fit_kpool_indices_rejects_narrow_input():
    with pytest.raises(ValueError, match="at least topk_tokens"):
        fit_kpool_indices_to_aiter(
            torch.zeros((1, 3), dtype=torch.int32), topk_tokens=4
        )


def test_rocm_sparse_mla_supports_glm_packed_layout():
    assert KVCacheLayout.BLHNC in ROCMAiterMLASparseBackend.supported_kv_cache_layouts()


@pytest.mark.parametrize(
    ("head_size", "num_prefills", "num_decodes", "expected"),
    [
        (512, 1, 0, True),
        (512, 1, 2, True),
        (512, 0, 2, True),
        (576, 1, 0, False),
        (512, 0, 0, False),
    ],
)
def test_rocm_sparse_triton_route(head_size, num_prefills, num_decodes, expected):
    """Validate Triton routing for prefill, decode, and MTP verification."""
    assert (
        _use_rocm_sparse_triton(
            head_size=head_size,
            kv_lora_rank=512,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("gfx942", "fp8", "num_prefills", "num_decodes", "prefill_max_seq_len", "route"),
    [
        pytest.param(True, False, 1, 0, 129, "triton", id="above-topk"),
        pytest.param(False, False, 1, 0, 129, "aiter", id="gfx950"),
        pytest.param(True, True, 1, 0, 129, "aiter", id="fp8"),
        pytest.param(True, False, 0, 2, 129, "aiter", id="decode-only"),
        pytest.param(True, False, 1, 1, 129, "aiter", id="mixed"),
        pytest.param(True, False, 1, 0, 128, "aiter", id="equal-topk"),
        pytest.param(True, False, 1, 0, 127, "aiter", id="below-topk"),
        pytest.param(True, False, 0, 0, 129, "aiter", id="empty"),
        pytest.param(
            True, False, 1, 0, None, "missing", id="required-prefill-metadata"
        ),
    ],
)
def test_rocm_long_bf16_prefill_dispatch(
    monkeypatch, gfx942, fp8, num_prefills, num_decodes, prefill_max_seq_len, route
):
    """Check the new dispatch gate using CPU tensors and intercepted kernels."""
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx942", lambda: gfx942)

    class KernelSelected(Exception):
        pass

    def triton_kernel(**kwargs):
        raise KernelSelected("triton")

    def aiter_kernel(*args, **kwargs):
        raise KernelSelected("aiter")

    monkeypatch.setattr(sparse_mod, "rocm_sparse_attn_prefill", triton_kernel)
    monkeypatch.setattr(sparse_mod.rocm_aiter_ops, "mla_decode_fwd", aiter_kernel)
    impl = object.__new__(sparse_mod.ROCMAiterMLASparseImpl)
    impl.num_heads = 64
    impl.kv_lora_rank = 512
    impl.kv_cache_dtype = "fp8" if fp8 else "auto"
    impl.scale = 576**-0.5
    impl.sinks = None

    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    q = torch.empty(2, 64, 576, dtype=dtype)
    kv = torch.empty(4, 1, 576, dtype=dtype)
    metadata = SimpleNamespace(
        attn_out_dtype=torch.bfloat16,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        max_query_len=2,
        prefill_max_seq_len=prefill_max_seq_len,
        topk_tokens=128,
        work_meta_data=None,
        qo_indptr=torch.arange(3, dtype=torch.int32),
        paged_kv_indices=torch.empty(0, dtype=torch.int32),
        paged_kv_indptr=torch.zeros(3, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(2, dtype=torch.int32),
    )
    # Irrelevant routes must not inspect prefill-only metadata.
    if (
        not gfx942
        or fp8
        or num_decodes
        or not num_prefills
        or prefill_max_seq_len is None
    ):
        del metadata.prefill_max_seq_len
        del metadata.topk_tokens
    layer = SimpleNamespace(_q_scale=1.0, _k_scale=1.0)
    error = AttributeError if route == "missing" else KernelSelected
    match = "prefill_max_seq_len" if route == "missing" else f"^{route}$"
    with pytest.raises(error, match=match):
        impl._forward_mla(layer, q, kv, metadata)


@pytest.mark.parametrize("gfx942", [False, True])
@pytest.mark.parametrize("num_heads", [8, 12])
def test_rocm_sparse_triton_route_preserves_padded_sinks(
    monkeypatch, num_heads, gfx942
):
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx942", lambda: gfx942)
    captured = {}

    def fake_rocm_sparse_attn_prefill(**kwargs):
        output = kwargs["output"]
        captured["attn_sink"] = kwargs["attn_sink"]
        output.copy_(
            captured["attn_sink"].to(output.dtype).view(1, -1, 1).expand_as(output)
        )

    monkeypatch.setattr(
        sparse_mod, "rocm_sparse_attn_prefill", fake_rocm_sparse_attn_prefill
    )

    impl = object.__new__(sparse_mod.ROCMAiterMLASparseImpl)
    impl.num_heads = num_heads
    impl.kv_lora_rank = 512
    impl.kv_cache_dtype = "auto"
    impl.scale = 512**-0.5
    impl.sinks = torch.arange(num_heads, dtype=torch.float32)

    q = torch.zeros(2, 16, 512, dtype=torch.bfloat16)
    kv = torch.zeros(4, 1, 512, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        attn_out_dtype=torch.bfloat16,
        num_prefills=1,
        num_decodes=0,
        num_decode_tokens=0,
        max_query_len=2,
        paged_kv_indices=torch.empty(0, dtype=torch.int32),
        paged_kv_indptr=torch.zeros(3, dtype=torch.int32),
    )

    layer = SimpleNamespace(_k_scale_float=1.0)
    output, lse = impl._forward_mla(layer, q, kv, metadata)

    if num_heads == 8:
        expected_sinks = impl.sinks.repeat_interleave(2)
    else:
        expected_sinks = torch.cat((impl.sinks, impl.sinks[:4]))
    torch.testing.assert_close(captured["attn_sink"], expected_sinks)
    assert output.shape == (2, num_heads, 512)
    torch.testing.assert_close(
        output[:, :, 0].float(),
        impl.sinks.expand(2, -1),
    )
    assert lse is None


@pytest.mark.parametrize("q_form", ["tensor", "tuple"])
@pytest.mark.parametrize("workload", ["mixed", "decode"])
@pytest.mark.parametrize(
    ("head_size", "use_aiter_sparse_mla", "route"),
    [
        pytest.param(512, False, "triton", id="nope-triton"),
        pytest.param(576, False, "asm", id="rope-asm"),
        pytest.param(512, True, "aiter-sparse-mla", id="nope-aiter-sparse-mla"),
    ],
)
def test_forward_mqa_fp8_kv_query_contract(
    monkeypatch, q_form, workload, head_size, use_aiter_sparse_mla, route
):
    """FP8 KV keeps NoPE Q in model dtype; every other route still quantizes Q.

    The non-unit KV scale must reach whichever attention wrapper is selected.
    """
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx942", lambda: False)
    fp8_dtype = current_platform.fp8_dtype()
    num_tokens, num_heads, kv_lora_rank, k_scale = 2, 16, 512, 2.5
    calls = []

    def record(name, q, kv_scale):
        calls.append((name, q.dtype, float(kv_scale)))

    def fake_triton_prefill(**kwargs):
        record("triton_prefill", kwargs["q"], kwargs["kv_scale"])

    def fake_triton_decode(**kwargs):
        record("triton_decode", kwargs["q"], kwargs["kv_scale"])

    def fake_asm(q, kv, out, *args, **kwargs):
        record("asm", q, kwargs["kv_scale"])

    def fake_aiter_sparse_mla(q, kv, out, *args, **kwargs):
        record("aiter-sparse-mla", q, kwargs["kv_scale"])

    def fake_scaled_fp8_quant(q, scale):
        calls.append(("scaled_fp8_quant",))
        return q.to(fp8_dtype), None

    def fake_concat_quant(ql_nope, q_pe, scale):
        calls.append(("concat_quant",))
        return torch.cat((ql_nope, q_pe), dim=-1).to(fp8_dtype)

    monkeypatch.setattr(sparse_mod, "rocm_sparse_attn_prefill", fake_triton_prefill)
    monkeypatch.setattr(sparse_mod, "rocm_sparse_attn_decode_bf16", fake_triton_decode)
    monkeypatch.setattr(
        sparse_mod, "rocm_sparse_decode_bf16_num_splits", lambda *args: 4
    )
    monkeypatch.setattr(sparse_mod.rocm_aiter_ops, "mla_decode_fwd", fake_asm)
    monkeypatch.setattr(
        sparse_mod.rocm_aiter_ops, "triton_sparse_mla_fwd", fake_aiter_sparse_mla
    )
    monkeypatch.setattr(sparse_mod.ops, "scaled_fp8_quant", fake_scaled_fp8_quant)
    monkeypatch.setattr(
        sparse_mod, "triton_convert_req_index_to_global_index", lambda *a, **k: None
    )

    impl = object.__new__(sparse_mod.ROCMAiterMLASparseImpl)
    impl.num_heads = num_heads
    impl.kv_lora_rank = kv_lora_rank
    impl.qk_rope_head_dim = head_size - kv_lora_rank
    impl.kv_cache_dtype = "fp8"
    impl.scale = head_size**-0.5
    impl.sinks = None
    impl.use_aiter_sparse_mla = use_aiter_sparse_mla
    impl.topk_indices_buffer = torch.zeros(num_tokens, 4, dtype=torch.int32)
    impl.owns_indexer = True
    impl.index_epoch = 0
    impl.q_concat_buffer = torch.empty(
        num_tokens, num_heads, head_size, dtype=torch.bfloat16
    )

    num_decodes = num_tokens if workload == "decode" else 1
    metadata = SimpleNamespace(
        attn_out_dtype=torch.bfloat16,
        num_actual_tokens=num_tokens,
        num_prefills=num_tokens - num_decodes,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        max_seq_len=4096,
        topk_tokens=4,
        block_size=1,
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32),
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        work_meta_data=None,
        qo_indptr=torch.arange(num_tokens + 1, dtype=torch.int32),
        paged_kv_indices=torch.zeros(4, dtype=torch.int32),
        paged_kv_indptr=torch.arange(num_tokens + 1, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(num_tokens, dtype=torch.int32),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0),
        _k_scale=torch.tensor(k_scale),
        _k_scale_float=k_scale,
        _decode_concat_quant_fp8_op=fake_concat_quant,
    )
    q = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    if q_form == "tuple":
        q = (q[..., :kv_lora_rank], q[..., kv_lora_rank:])
    kv_cache = torch.zeros(4, 1, head_size, dtype=torch.uint8)

    impl.forward_mqa(q, kv_cache, metadata, layer)

    if route == "triton":
        wrapper = "triton_decode" if workload == "decode" else "triton_prefill"
        assert calls == [(wrapper, torch.bfloat16, k_scale)]
    else:
        quant = "concat_quant" if q_form == "tuple" else "scaled_fp8_quant"
        assert calls == [(quant,), (route, fp8_dtype, k_scale)]


def test_rocm_sparse_attention_accepts_glm_nope_dimensions():
    _validate_sparse_dims(512, 512, 0, "test")


def test_rocm_sparse_attention_rejects_inconsistent_dimensions():
    with pytest.raises(AssertionError, match="expected head_dim"):
        _validate_sparse_dims(511, 512, 0, "test")


def test_dsv4_sparse_attention_keeps_layout_constraint():
    _validate_dsv4_sparse_dims(512, 448, 64, "test")
    with pytest.raises(AssertionError, match="expects 448 NoPE dims"):
        _validate_dsv4_sparse_dims(512, 512, 0, "test")


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_sparse_prefill_kv_row_offset_does_not_overflow_int32():
    # GLM's 640-token pages cross the signed-int32 address boundary at block
    # 6554 for a 512-element KV row. The production kernel must promote the
    # slot before multiplying by the row stride.
    slot = torch.tensor([6554 * 640], dtype=torch.int32, device="cuda")
    output = torch.empty(1, dtype=torch.int64, device="cuda")

    _store_sparse_kv_row_offset_kernel[(1,)](slot, output, stride=512)

    assert output.item() == 6554 * 640 * 512


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_sparse_prefill_query_row_offset_does_not_overflow_int32():
    # A padded 128-head, 512-wide output row has a 65,536-element stride.
    # Query 131,071 therefore addresses almost 2**33 elements into the tensor.
    query_idx = torch.tensor([131071], dtype=torch.int32, device="cuda")
    output = torch.empty(1, dtype=torch.int64, device="cuda")

    _store_sparse_query_row_offset_kernel[(1,)](query_idx, output, stride=128 * 512)

    assert output.item() == 131071 * 128 * 512


def _sparse_attn_fp32_reference(q, kv, indices, indptr, scale, attn_sink):
    q = q.float()
    out = torch.zeros_like(q)
    for row in range(q.shape[0]):
        slots = indices[indptr[row] : indptr[row + 1]]
        slots = slots[slots >= 0].long()
        scores = q[row] @ kv[slots].T * scale
        if attn_sink is not None:
            scores = torch.cat((scores, attn_sink[:, None]), dim=1)
        if scores.shape[1] == 0:
            continue
        probs = scores.softmax(dim=-1)[:, : slots.numel()]
        out[row] = probs @ kv[slots]
    return out


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
@pytest.mark.parametrize("kv_scale", [1.5, 0.37])
@pytest.mark.parametrize("with_sink", [False, True])
@pytest.mark.parametrize("num_splits", [None, 4])
def test_sparse_fp8_nope_matches_fp32_reference(num_splits, with_sink, kv_scale):
    """NoPE FP8 KV must dequantize in Triton, not reach 576-wide AITER asm.

    ``num_splits=None`` covers the ragged kernel, otherwise split-K decode.
    Row 1 is empty, and row 2 is shorter than ``num_splits`` so later splits
    are empty.
    """
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_decode_ragged_bf16_triton,
        _rocm_sparse_attn_prefill_ragged_triton,
    )

    set_random_seed(0)
    device = "cuda"
    num_heads, head_dim, num_kv = 16, 512, 1024
    kv_lens = [64, 0, 2, 37, 300]
    q = torch.randn(
        len(kv_lens), num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    kv_source = torch.randn(num_kv, head_dim, dtype=torch.float32, device=device)
    kv_fp8 = (kv_source / kv_scale).to(current_platform.fp8_dtype())
    indices = torch.randint(
        0, num_kv, (sum(kv_lens),), device=device, dtype=torch.int32
    )
    indices[kv_lens[0] + 5] = -1
    indptr = torch.tensor(
        [0, *torch.tensor(kv_lens).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    attn_sink = (
        torch.randn(num_heads, dtype=torch.float32, device=device) + 1.0
        if with_sink
        else None
    )
    common = dict(
        q=q,
        kv=kv_fp8,
        indices=indices,
        indptr=indptr,
        scale=head_dim**-0.5,
        attn_sink=attn_sink,
        nope_head_dim=head_dim,
        rope_head_dim=0,
        kv_scale=kv_scale,
    )
    if num_splits is None:
        out = _rocm_sparse_attn_prefill_ragged_triton(**common)
    else:
        out = _rocm_sparse_attn_decode_ragged_bf16_triton(
            **common, num_splits=num_splits
        )

    expected = _sparse_attn_fp32_reference(
        q, kv_fp8.float() * kv_scale, indices, indptr, head_dim**-0.5, attn_sink
    )
    assert out[1].count_nonzero() == 0
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
