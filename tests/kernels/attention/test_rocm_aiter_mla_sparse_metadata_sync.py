# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "ROCm AITER sparse MLA metadata sync test requires ROCm.",
        allow_module_level=True,
    )

from vllm._aiter_ops import is_aiter_found_and_supported

if not is_aiter_found_and_supported():
    pytest.skip(
        "ROCm AITER sparse MLA metadata sync test requires a supported AITER "
        "installation.",
        allow_module_level=True,
    )

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as sparse_mod


class _FakeAiter(ModuleType):
    get_mla_metadata_v1: Mock


@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("topk_tokens", [128, 256])
def test_rocm_index_conversion_matrix(block_size, topk_tokens):
    """Convert valid, masked, and out-of-range sparse indices exactly."""
    device = torch.device("cuda")
    num_tokens = 4
    max_blocks = 4
    req_ids = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
    block_table = torch.tensor(
        [[3, 5, 7, 9], [2, 4, 6, 8]], dtype=torch.int32, device=device
    )
    token_indices = torch.arange(
        num_tokens * topk_tokens, dtype=torch.int32, device=device
    ).view(num_tokens, topk_tokens)
    token_indices %= max_blocks * block_size
    token_indices[0, 0] = -1
    token_indices[1, 1] = max_blocks * block_size
    cu_seqlens = torch.arange(
        0,
        (num_tokens + 1) * topk_tokens,
        topk_tokens,
        dtype=torch.int32,
        device=device,
    )
    output = torch.full(
        (num_tokens * topk_tokens,), -99, dtype=torch.int32, device=device
    )

    sparse_mod.triton_convert_req_index_to_global_index(
        req_ids,
        block_table,
        token_indices,
        cu_seqlens,
        output,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=topk_tokens,
    )

    block_ids = token_indices // block_size
    valid = (token_indices >= 0) & (block_ids < max_blocks)
    safe_blocks = block_ids.clamp(0, max_blocks - 1)
    bases = block_table[req_ids[:, None].long(), safe_blocks.long()]
    expected = torch.where(valid, bases * block_size + token_indices % block_size, 0)
    torch.testing.assert_close(output.view_as(expected), expected, rtol=0, atol=0)


def test_rocm_index_conversion_rejects_mismatched_token_extents():
    """Reject full-batch request IDs paired with decode-subset indices."""
    with pytest.raises(AssertionError, match="must cover the same tokens"):
        sparse_mod.triton_convert_req_index_to_global_index(
            torch.zeros(3, dtype=torch.int32),
            torch.zeros(1, 4, dtype=torch.int32),
            torch.zeros(2, 128, dtype=torch.int32),
            torch.arange(3, dtype=torch.int32),
            torch.zeros(256, dtype=torch.int32),
            NUM_TOPK_TOKENS=128,
        )


def test_rocm_index_conversion_respects_packed_valid_counts():
    """ROCm's indptr contract packs only each token's valid top-k prefix."""
    device = torch.device("cuda")
    topk_tokens = 128
    valid_counts = torch.tensor([1, 17, 128], dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(4, dtype=torch.int32, device=device)
    torch.cumsum(valid_counts, dim=0, out=cu_seqlens[1:])
    req_ids = torch.tensor([0, 0, 0], dtype=torch.int32, device=device)
    block_table = torch.tensor([[2, 3, 4]], dtype=torch.int32, device=device)
    token_indices = torch.arange(
        3 * topk_tokens, dtype=torch.int32, device=device
    ).view(3, topk_tokens)
    token_indices %= 3 * 64
    output = torch.full((int(cu_seqlens[-1]),), -99, dtype=torch.int32, device=device)

    sparse_mod.triton_convert_req_index_to_global_index(
        req_ids,
        block_table,
        token_indices,
        cu_seqlens,
        output,
        BLOCK_SIZE=64,
        NUM_TOPK_TOKENS=topk_tokens,
    )

    expected_parts = []
    for token_id, count in enumerate(valid_counts.tolist()):
        local = token_indices[token_id, :count]
        expected_parts.append(block_table[0, (local // 64).long()] * 64 + local % 64)
    expected = torch.cat(expected_parts)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _make_builder():
    builder = object.__new__(sparse_mod.ROCMAiterMLASparseMetadataBuilder)
    max_num_batched_tokens = 8
    topk_tokens = 4

    builder.device = torch.device("cpu")
    builder.kv_cache_spec = SimpleNamespace(block_size=1)
    builder.model_dtype = torch.bfloat16
    builder.topk_tokens = topk_tokens
    # SparseMLACommonMetadataBuilder state consumed by its build() method.
    builder.cp_kv_cache_interleave_size = 1
    builder.req_id_per_token_buffer = torch.zeros(
        max_num_batched_tokens, dtype=torch.int32, device="cpu"
    )
    builder.qo_indptr = torch.arange(
        max_num_batched_tokens + 1, dtype=torch.int32, device="cpu"
    )
    builder.paged_kv_last_page_len = torch.ones(
        max_num_batched_tokens, dtype=torch.int32, device="cpu"
    )
    builder.paged_kv_indices = torch.zeros(
        max_num_batched_tokens * topk_tokens, dtype=torch.int32, device="cpu"
    )
    builder.paged_kv_indptr = torch.zeros(
        max_num_batched_tokens + 1, dtype=torch.int32, device="cpu"
    )
    builder._num_attention_heads = 16
    builder._num_compute_units = 1
    builder._mla_q_dtype = torch.bfloat16
    builder._mla_kv_dtype = torch.bfloat16
    builder._mla_work_meta_data = torch.empty(1, dtype=torch.int32, device="cpu")
    builder._mla_work_indptr = torch.empty(1, dtype=torch.int32, device="cpu")
    builder._mla_work_info_set = torch.empty(1, dtype=torch.int32, device="cpu")
    builder._mla_reduce_indptr = torch.empty(1, dtype=torch.int32, device="cpu")
    builder._mla_reduce_final_map = torch.empty(1, dtype=torch.int32, device="cpu")
    builder._mla_reduce_partial_map = torch.empty(1, dtype=torch.int32, device="cpu")
    builder._prev_req_extent = 0
    builder._prev_indices_extent = 0
    builder._prev_metadata_key = None
    return builder


def _make_common_metadata():
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device="cpu")
    seq_lens = torch.tensor([16, 8], dtype=torch.int32, device="cpu")
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens,
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=16,
        block_table_tensor=torch.arange(16, dtype=torch.int32, device="cpu").view(2, 8),
        slot_mapping=torch.arange(2, dtype=torch.int64, device="cpu"),
    )


def test_sparse_persistent_metadata_syncs_only_after_recompute(monkeypatch):
    builder = _make_builder()
    common_metadata = _make_common_metadata()
    events: list[str] = []

    def fake_generate_sparse_seqlen_triton(*args, **kwargs):
        return torch.tensor([1, 2], dtype=torch.int32, device="cpu")

    fake_aiter = _FakeAiter("aiter")

    def fake_get_mla_metadata_v1(*args, **kwargs):
        events.append("metadata")

    fake_get_mla_metadata_v1_mock = Mock(side_effect=fake_get_mla_metadata_v1)
    fake_aiter.get_mla_metadata_v1 = fake_get_mla_metadata_v1_mock
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setattr(
        sparse_mod, "generate_sparse_seqlen_triton", fake_generate_sparse_seqlen_triton
    )
    monkeypatch.setattr(
        sparse_mod.torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(synchronize=lambda: events.append("sync")),
    )

    builder.build(common_prefix_len=0, common_attn_metadata=common_metadata)

    assert events == ["metadata", "sync"]
    assert fake_get_mla_metadata_v1_mock.call_count == 1
    assert fake_get_mla_metadata_v1_mock.call_args.kwargs["dtype_q"] == torch.bfloat16
    assert fake_get_mla_metadata_v1_mock.call_args.kwargs["dtype_kv"] == torch.bfloat16

    events.clear()

    builder.build(common_prefix_len=0, common_attn_metadata=common_metadata)

    assert events == []
    assert fake_get_mla_metadata_v1_mock.call_count == 1


def test_dense_prefill_builds_sparse_metadata_for_decode_subset(monkeypatch):
    """Dense prefill must exclude prefill tokens from AITER sparse metadata.

    A mixed batch is routed as sparse MQA for its leading decode requests and
    dense MHA for its trailing short prefill requests.  The AITER metadata
    buffers therefore must describe only the decode subset; including prefill
    tokens would mismatch the query passed to ``forward_mqa``.
    """
    builder = _make_builder()
    builder.vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(sparse_mla_force_mqa=False)
    )

    query_start_loc = torch.tensor([0, 1, 5], dtype=torch.int32)
    seq_lens = torch.tensor([16, 4], dtype=torch.int32)
    common_metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        max_query_len=4,
        max_seq_len=16,
    )
    metadata = sparse_mod.ROCMAiterMLASparseMetadata(
        num_reqs=2,
        max_query_len=4,
        max_seq_len=16,
        num_actual_tokens=5,
        query_start_loc=query_start_loc,
        slot_mapping=torch.arange(5),
        block_table=torch.arange(16, dtype=torch.int32).view(2, 8),
        req_id_per_token=torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
        topk_tokens=4,
        num_decodes=1,
        num_prefills=1,
        num_decode_tokens=1,
        seq_lens=seq_lens,
        prefill_max_seq_len=4,
        prefill=SimpleNamespace(),
    )

    monkeypatch.setattr(
        sparse_mod.SparseMLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,
    )
    sparse_call: dict[str, torch.Tensor] = {}

    def fake_generate_sparse_seqlen(
        query_lens, sparse_seq_lens, sparse_query_start_loc, topk, num_tokens, max_q
    ):
        sparse_call.update(
            query_lens=query_lens.clone(),
            seq_lens=sparse_seq_lens.clone(),
            query_start_loc=sparse_query_start_loc.clone(),
            num_tokens=num_tokens,
        )
        return torch.ones(num_tokens, dtype=torch.int32)

    monkeypatch.setattr(
        sparse_mod, "generate_sparse_seqlen_triton", fake_generate_sparse_seqlen
    )
    fake_aiter = _FakeAiter("aiter")
    fake_aiter.get_mla_metadata_v1 = Mock()
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setattr(
        sparse_mod.torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(synchronize=lambda: None),
    )

    result = builder.build(0, common_metadata)

    assert result.prefill is metadata.prefill
    assert result.num_prefills == 1
    assert result.req_id_per_token.tolist() == [0]
    assert result.qo_indptr.numel() == 2
    assert result.paged_kv_indptr.numel() == 2
    assert sparse_call["num_tokens"] == 1
    assert sparse_call["query_lens"].tolist() == [1]
    assert sparse_call["seq_lens"].tolist() == [16]
    assert sparse_call["query_start_loc"].tolist() == [0, 1]


def test_long_prefill_shares_full_sparse_metadata_with_decode(monkeypatch):
    """A prefill longer than top-k must remain in shared sparse metadata.

    Dense MHA is only selected when the prefill sequence fits within the
    sparse top-k window.  Once it exceeds that window, both decode and prefill
    are handled by sparse MQA, so every query token must be represented by the
    same sparse paging metadata.
    """
    builder = _make_builder()
    builder.vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(sparse_mla_force_mqa=False)
    )

    # One decode token followed by a five-token prefill.  topk_tokens is four,
    # so the prefill must not be split onto the dense-MHA path.
    query_start_loc = torch.tensor([0, 1, 6], dtype=torch.int32)
    seq_lens = torch.tensor([16, 5], dtype=torch.int32)
    common_metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        max_query_len=5,
        max_seq_len=16,
    )
    metadata = sparse_mod.ROCMAiterMLASparseMetadata(
        num_reqs=2,
        max_query_len=5,
        max_seq_len=16,
        num_actual_tokens=6,
        query_start_loc=query_start_loc,
        slot_mapping=torch.arange(6),
        block_table=torch.arange(16, dtype=torch.int32).view(2, 8),
        req_id_per_token=torch.tensor([0, 1, 1, 1, 1, 1], dtype=torch.int32),
        topk_tokens=4,
        num_decodes=1,
        num_prefills=1,
        num_decode_tokens=1,
        seq_lens=seq_lens,
        prefill_max_seq_len=5,
        prefill=SimpleNamespace(),
    )

    monkeypatch.setattr(
        sparse_mod.SparseMLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,
    )
    sparse_call: dict[str, torch.Tensor] = {}

    def fake_generate_sparse_seqlen(
        query_lens, sparse_seq_lens, sparse_query_start_loc, topk, num_tokens, max_q
    ):
        sparse_call.update(
            query_lens=query_lens.clone(),
            seq_lens=sparse_seq_lens.clone(),
            query_start_loc=sparse_query_start_loc.clone(),
            num_tokens=num_tokens,
        )
        return torch.ones(num_tokens, dtype=torch.int32)

    monkeypatch.setattr(
        sparse_mod, "generate_sparse_seqlen_triton", fake_generate_sparse_seqlen
    )
    fake_aiter = _FakeAiter("aiter")
    get_persistent_metadata = Mock()
    fake_aiter.get_mla_metadata_v1 = get_persistent_metadata
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setattr(
        sparse_mod.torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(synchronize=lambda: None),
    )

    result = builder.build(0, common_metadata)

    assert result.num_decodes == 1
    assert result.num_prefills == 1
    assert result.req_id_per_token.tolist() == [0, 1, 1, 1, 1, 1]
    assert result.qo_indptr.numel() == 7
    assert result.paged_kv_indptr.numel() == 7
    assert result.paged_kv_indices.numel() == 6 * result.topk_tokens
    assert sparse_call["num_tokens"] == 6
    assert sparse_call["query_lens"].tolist() == [1, 5]
    assert sparse_call["seq_lens"].tolist() == [16, 5]
    assert sparse_call["query_start_loc"].tolist() == [0, 1, 6]
    get_persistent_metadata.assert_called_once()
    assert result.work_meta_data is builder._mla_work_meta_data
    assert result.reduce_indptr is builder._mla_reduce_indptr


def test_forward_mqa_slices_full_batch_metadata_to_query_tokens(monkeypatch):
    """The sparse kernel inputs must have the same token extent as its query.

    In a mixed dense-prefill batch, ``forward_mqa`` receives only decode query
    tokens while some metadata buffers may represent the full batch.  This
    guards the slicing that prevents the index conversion from writing beyond
    the decode-sized output.
    """
    impl = object.__new__(sparse_mod.ROCMAiterMLASparseImpl)
    impl.kv_cache_dtype = "auto"
    impl.num_heads = 4
    impl.topk_indices_buffer = torch.arange(48, dtype=torch.int32).view(12, 4)
    expected_output = torch.zeros(2, 4, 512, dtype=torch.bfloat16)
    impl._forward_mla = Mock(return_value=expected_output)

    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0, 1] + [2] * 5 + [3] * 5, dtype=torch.int32),
        block_table=torch.arange(40, dtype=torch.int32).view(4, 10),
        paged_kv_indptr=torch.arange(3, dtype=torch.int32),
        paged_kv_indices=torch.zeros(8, dtype=torch.int32),
        block_size=1,
        topk_tokens=4,
    )
    query = torch.zeros(2, 4, 576, dtype=torch.bfloat16)
    captured = {}

    def fake_convert(req_ids, block_table, token_indices, *args, **kwargs):
        captured["req_ids"] = req_ids.clone()
        captured["token_indices"] = token_indices.clone()

    monkeypatch.setattr(
        sparse_mod, "triton_convert_req_index_to_global_index", fake_convert
    )
    monkeypatch.setattr(
        sparse_mod.AiterMLAHelper, "get_mla_padded_q", lambda num_heads, q: q
    )

    output, lse = impl.forward_mqa(
        query,
        torch.zeros(1, 1, 576, dtype=torch.bfloat16),
        metadata,
        SimpleNamespace(),
    )

    assert output is expected_output
    assert lse is None
    assert captured["req_ids"].tolist() == [0, 1]
    assert captured["token_indices"].shape == (2, 4)
    impl._forward_mla.assert_called_once()


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_forward_mqa_slices_decode_subset_for_bf16_and_fp8(monkeypatch, kv_cache_dtype):
    """Both cache dtypes must slice mixed-batch metadata to decode queries."""
    impl = object.__new__(sparse_mod.ROCMAiterMLASparseImpl)
    impl.kv_cache_dtype = kv_cache_dtype
    impl.num_heads = 4
    impl.topk_indices_buffer = torch.arange(48, dtype=torch.int32).view(12, 4)
    expected_output = torch.zeros(2, 4, 512, dtype=torch.bfloat16)
    impl._forward_mla = Mock(return_value=expected_output)

    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0, 1] + [2] * 5 + [3] * 5, dtype=torch.int32),
        block_table=torch.arange(40, dtype=torch.int32).view(4, 10),
        paged_kv_indptr=torch.arange(3, dtype=torch.int32),
        paged_kv_indices=torch.zeros(8, dtype=torch.int32),
        block_size=1,
        topk_tokens=4,
    )
    query = torch.zeros(2, 4, 576, dtype=torch.bfloat16)
    kv_cache = torch.zeros(1, 1, 576, dtype=torch.uint8)
    captured = {}

    def fake_convert(req_ids, block_table, token_indices, *args, **kwargs):
        captured["req_ids"] = req_ids.clone()
        captured["token_indices"] = token_indices.clone()

    monkeypatch.setattr(
        sparse_mod, "triton_convert_req_index_to_global_index", fake_convert
    )
    monkeypatch.setattr(
        sparse_mod.AiterMLAHelper, "get_mla_padded_q", lambda num_heads, q: q
    )
    if kv_cache_dtype == "fp8":
        fp8_dtype = current_platform.fp8_dtype()
        monkeypatch.setattr(
            sparse_mod.ops,
            "scaled_fp8_quant",
            lambda q, scale: (q.to(fp8_dtype), None),
        )

    output, lse = impl.forward_mqa(
        query,
        kv_cache,
        metadata,
        SimpleNamespace(_q_scale=torch.tensor(1.0)),
    )

    assert output is expected_output
    assert lse is None
    assert captured["req_ids"].tolist() == [0, 1]
    assert captured["token_indices"].shape == (2, 4)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("length_relation", ["short", "long"])
@pytest.mark.parametrize("workload", ["prefill", "mixed", "decode"])
def test_sparse_metadata_workload_length_dtype_matrix(
    monkeypatch, workload, length_relation, kv_cache_dtype
):
    """Cover routing and persistent metadata across the ROCm support matrix.

    The matrix crosses pure prefill, mixed prefill/decode, and decode-only
    workloads with sequence lengths below/above top-k and BF16/FP8 KV cache.
    """
    builder = _make_builder()
    builder.vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(sparse_mla_force_mqa=False)
    )
    if kv_cache_dtype == "fp8":
        builder._mla_q_dtype = current_platform.fp8_dtype()
        builder._mla_kv_dtype = current_platform.fp8_dtype()

    is_long = length_relation == "long"
    prefill_len = 5 if is_long else 3
    decode_seq_len = 8 if is_long else 3
    if workload == "prefill":
        query_lens = [prefill_len]
        seq_lens_list = [prefill_len]
        num_decodes, num_prefills = 0, 1
    elif workload == "mixed":
        query_lens = [1, prefill_len]
        seq_lens_list = [decode_seq_len, prefill_len]
        num_decodes, num_prefills = 1, 1
    else:
        query_lens = [1, 1]
        seq_lens_list = [decode_seq_len, decode_seq_len + 1]
        num_decodes, num_prefills = 2, 0

    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
    num_tokens = sum(query_lens)
    common_metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens_list),
    )
    metadata = sparse_mod.ROCMAiterMLASparseMetadata(
        num_reqs=len(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens_list),
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc,
        slot_mapping=torch.arange(num_tokens),
        block_table=torch.arange(len(query_lens) * 8, dtype=torch.int32).view(
            len(query_lens), 8
        ),
        req_id_per_token=torch.repeat_interleave(
            torch.arange(len(query_lens), dtype=torch.int32),
            torch.tensor(query_lens),
        ),
        topk_tokens=4,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        num_decode_tokens=num_decodes,
        seq_lens=seq_lens,
        prefill_max_seq_len=prefill_len if num_prefills else 0,
        prefill=SimpleNamespace() if num_prefills else None,
    )
    monkeypatch.setattr(
        sparse_mod.SparseMLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,
    )
    monkeypatch.setattr(
        sparse_mod,
        "generate_sparse_seqlen_triton",
        lambda *args, **kwargs: torch.ones(args[4], dtype=torch.int32),
    )
    get_persistent_metadata = Mock()
    fake_aiter = _FakeAiter("aiter")
    fake_aiter.get_mla_metadata_v1 = get_persistent_metadata
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setattr(
        sparse_mod.torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(synchronize=lambda: None),
    )

    result = builder.build(0, common_metadata)

    uses_dense_prefill = workload in ("prefill", "mixed") and not is_long
    expected_sparse_tokens = num_decodes if uses_dense_prefill else num_tokens
    assert result.req_id_per_token.numel() == expected_sparse_tokens
    assert result.qo_indptr.numel() == expected_sparse_tokens + 1
    assert result.paged_kv_indptr.numel() == expected_sparse_tokens + 1

    uses_fallback = workload == "prefill" and is_long and kv_cache_dtype == "auto"
    if expected_sparse_tokens == 0 or uses_fallback:
        get_persistent_metadata.assert_not_called()
    else:
        get_persistent_metadata.assert_called_once()
    if uses_fallback:
        assert result.work_meta_data is None
        assert result.reduce_indptr is None
    else:
        assert result.work_meta_data is builder._mla_work_meta_data
        assert result.reduce_indptr is builder._mla_reduce_indptr
