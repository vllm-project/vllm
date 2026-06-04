# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.attention.test_attention_backends import BATCH_SPECS
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import (
    make_kv_sharing_fast_prefill_common_attn_metadata,
    split_decodes_and_prefills,
    split_decodes_prefills_and_extends,
)
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    _make_metadata_with_slice,
    maybe_create_ubatch_slices,
    slice_query_start_locs,
    split_attn_metadata,
)


@pytest.fixture
def sample_query_start_loc():
    """Sample query_start_loc tensor for testing"""
    return torch.tensor([0, 5, 12, 20, 35, 50])


def test_basic_slice_middle(sample_query_start_loc):
    """Test slicing from middle of tensor"""
    req_slice = slice(1, 3)  # slice from index 1 to 3
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 7, 15])
    assert torch.equal(result, expected)


def test_slice_from_beginning(sample_query_start_loc):
    """Test slicing from the beginning of tensor"""
    req_slice = slice(0, 2)  # slice from index 0 to 2
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 5, 12])
    assert torch.equal(result, expected)


def test_slice_to_end(sample_query_start_loc):
    """Test slicing to the end of tensor"""
    req_slice = slice(3, 5)  # slice from index 3 to 5 (last index)
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 15, 30])
    assert torch.equal(result, expected)


def test_single_element_slice(sample_query_start_loc):
    """Test slice that results in single element"""
    req_slice = slice(2, 3)  # slice from index 2 to 3
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 8])
    assert torch.equal(result, expected)


def test_full_tensor_slice(sample_query_start_loc):
    """Test slicing the entire tensor"""
    req_slice = slice(0, 5)  # slice entire tensor
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 5, 12, 20, 35, 50])
    assert torch.equal(result, expected)


def test_slice_bounds_edge_cases(sample_query_start_loc):
    # Test slice that goes exactly to the last element
    req_slice = slice(4, 5)  # Last index
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 15])
    assert torch.equal(result, expected)


@pytest.fixture
def small_decode_metadata():
    """Create metadata for small decode batch"""
    batch_spec = BATCH_SPECS["small_decode"]
    device = torch.device("cpu")
    return create_common_attn_metadata(batch_spec, block_size=16, device=device)


@pytest.fixture
def large_decode_metadata():
    """Create metadata for small decode batch"""
    batch_spec = BATCH_SPECS["large_decode"]
    device = torch.device("cpu")
    return create_common_attn_metadata(batch_spec, block_size=16, device=device)


@pytest.fixture
def mixed_small_metadata():
    """Create metadata for mixed small batch"""
    batch_spec = BATCH_SPECS["mixed_small"]
    device = torch.device("cpu")
    return create_common_attn_metadata(batch_spec, block_size=16, device=device)


# Tests for _make_metadata_with_slice
def test_make_metadata_with_slice_decode_batch(small_decode_metadata):
    """Test slicing decode batch metadata"""
    # Split first request only
    ubatch_slice = UBatchSlice(slice(0, 1), slice(0, 1))

    result = _make_metadata_with_slice(ubatch_slice, small_decode_metadata)

    # Check sliced results
    assert result.num_reqs == 1  # slice(0, 1) gives 1 requests
    assert result.num_actual_tokens == 1  # slice(0, 1) gives 1 token
    assert result.max_query_len == 1
    assert torch.equal(result.query_start_loc, torch.tensor([0, 1]))
    assert torch.equal(result.seq_lens, torch.tensor([32]))


def test_make_metadata_with_slice_mixed_batch(mixed_small_metadata):
    """Test slicing mixed batch metadata"""
    ubatch_slice = UBatchSlice(slice(1, 3), slice(1, 7))  # Requests 1-3, tokens 1-7

    result = _make_metadata_with_slice(ubatch_slice, mixed_small_metadata)

    assert result.num_reqs == 2  # slice(1, 3) gives 2 requests
    assert result.num_actual_tokens == 6  # slice(1, 7) gives 6 tokens
    assert result.max_query_len == 5
    assert torch.equal(result.query_start_loc, torch.tensor([0, 1, 6]))
    assert torch.equal(result.seq_lens, torch.tensor([40, 48]))


def test_split_attn_metadata_decode_batch(large_decode_metadata):
    """Test splitting decode batch into two equal parts"""
    num_tokens = large_decode_metadata.num_reqs
    mid_point = num_tokens // 2
    ubatch_slices = [
        UBatchSlice(slice(0, mid_point), slice(0, mid_point)),
        UBatchSlice(slice(mid_point, num_tokens), slice(mid_point, num_tokens)),
    ]

    results = split_attn_metadata(ubatch_slices, large_decode_metadata)

    assert len(results) == 2

    # Check first split
    assert results[0].num_reqs == mid_point
    assert results[0].num_actual_tokens == mid_point
    assert torch.equal(results[0].seq_lens, torch.tensor([2048] * mid_point))

    # Check second split
    assert results[1].num_reqs == mid_point
    assert results[1].num_actual_tokens == mid_point
    assert torch.equal(results[1].seq_lens, torch.tensor([2048] * mid_point))


def apply_split_decodes_and_prefills(
    query_lens: list[int],
    decode_threshold: int,
    require_uniform: bool,
    padded_num_tokens: int | None = None,
):
    """Helper function to apply split_decodes_and_prefills and return
    the results."""
    device = torch.device("cpu")
    seq_lens = [10 * (i + 1) for i in range(len(query_lens))]
    common_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=16,
        device=device,
    )

    if padded_num_tokens is not None:
        common_metadata.num_actual_tokens = padded_num_tokens

    return split_decodes_and_prefills(
        common_metadata,
        decode_threshold=decode_threshold,
        require_uniform=require_uniform,
    )


def test_split_decodes_and_prefills_nonuniform_all_ones():
    query_lens = [1, 1, 1]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 1, False)
    )
    assert num_decodes == 3
    assert num_prefills == 0
    assert num_decode_tokens == 3
    assert num_prefill_tokens == 0


def test_split_decodes_and_prefills_nonuniform_all_short_decodes():
    query_lens = [1, 2, 1, 3, 2, 1, 2]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, False)
    )
    assert num_decodes == 7
    assert num_prefills == 0
    assert num_decode_tokens == sum(query_lens)
    assert num_prefill_tokens == 0


def test_split_decodes_and_prefills_nonuniform_all_prefills():
    query_lens = [4, 5, 6, 7]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, False)
    )
    assert num_decodes == 0
    assert num_prefills == 4
    assert num_decode_tokens == 0
    assert num_prefill_tokens == sum(query_lens)


def test_split_decodes_and_prefills_nonuniform_mixed_batch():
    query_lens = [2, 1, 3, 4, 5, 6, 7, 8]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 4, False)
    )
    assert num_decodes == 4  # 2, 1, 3, 4 are all <= 4
    assert num_prefills == 4  # 5, 6, 7, 8 are all > 4
    assert num_decode_tokens == 10  # 2 + 1 + 3 + 4
    assert num_prefill_tokens == 26  # 5 + 6 + 7 + 8


def test_split_decodes_and_prefills_uniform_all_ones():
    query_lens = [1, 1, 1]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 1, True)
    )
    assert num_decodes == 3
    assert num_prefills == 0
    assert num_decode_tokens == 3
    assert num_prefill_tokens == 0


def test_split_decodes_and_prefills_uniform_all_short_decodes():
    query_lens = [2, 2, 1, 3, 2, 1, 2]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, True)
    )
    assert num_decodes == 2
    assert num_prefills == 5
    assert num_decode_tokens == 4
    assert num_prefill_tokens == (1 + 3 + 2 + 1 + 2)


def test_split_decodes_and_prefills_uniform_all_prefills():
    query_lens = [4, 5, 6, 7]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, True)
    )
    assert num_decodes == 0
    assert num_prefills == 4
    assert num_decode_tokens == 0
    assert num_prefill_tokens == sum(query_lens)


def test_split_decodes_and_prefills_uniform_mixed_batch_all_uniform_decodes():
    query_lens = [2, 2, 2, 4, 5, 6, 7, 8]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 4, True)
    )
    assert num_decodes == 3  # 2, 2, 2 are all <= 4 and uniform
    assert num_prefills == 5  # 4, 5, 6, 7, 8 are all > 4
    assert num_decode_tokens == 6  # 2 + 2 + 2
    assert num_prefill_tokens == 30  # 4 + 5 + 6 + 7 + 8


def test_split_decodes_and_prefills_uniform_mixed_batch_non_uniform_decodes():
    query_lens = [2, 1, 2, 4, 5, 6, 7, 8]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 4, True)
    )
    assert num_decodes == 1  # only the first 2 is taken as decode
    assert num_prefills == 7  # 1, 2, 4, 5, 6, 7, 8 are all > 4 or non-uniform
    assert num_decode_tokens == 2  # only the first 2
    assert num_prefill_tokens == (sum(query_lens) - 2)  # rest of the tokens


def test_split_decodes_and_prefills_uniform_padded_batch_all_same():
    """uniform batch where all query lengths are identical with 0 length padded reqs."""
    # All query lengths are 2, with decode_threshold=3 (so 2 <= 3)
    # This triggers the padded uniform path at line 891
    query_lens = [2, 2, 2, 0]
    padded_num_tokens = 8
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, True, padded_num_tokens)
    )
    # With uniform batch, all requests are treated as decodes
    assert num_decodes == 4
    assert num_prefills == 0
    assert num_decode_tokens == padded_num_tokens
    assert num_prefill_tokens == 0


@pytest.mark.parametrize(
    "seq_lens,query_lens,split_point,expected_first_reqs,expected_second_reqs",
    [
        # Split in the middle of request 1
        ([32, 40], [8, 8], 12, 2, 1),
        # Split inside the first request
        ([32, 40], [8, 8], 4, 1, 2),
    ],
)
def test_prefill_split_across_ubatches(
    seq_lens, query_lens, split_point, expected_first_reqs, expected_second_reqs
):
    """Test splitting a prefill across ubatches"""
    import numpy as np

    device = torch.device("cpu")
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(batch_spec, block_size=16, device=device)

    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    qsl_np = common.query_start_loc_cpu.numpy()
    num_tokens = common.num_actual_tokens

    ubatch_slices, _ = maybe_create_ubatch_slices(
        True,
        num_scheduled_tokens,
        num_tokens,
        batch_spec.batch_size,
        split_point=split_point,
        num_ubatches=2,
    )
    assert ubatch_slices is not None and len(ubatch_slices) == 2

    first_meta = _make_metadata_with_slice(ubatch_slices[0], common)
    second_meta = _make_metadata_with_slice(ubatch_slices[1], common)

    # Token counts match the split
    assert first_meta.num_actual_tokens == split_point
    assert second_meta.num_actual_tokens == num_tokens - split_point

    # Number of requests per ubatch
    assert first_meta.num_reqs == expected_first_reqs
    assert second_meta.num_reqs == expected_second_reqs

    # Identify which request is split and how many tokens are in the first chunk
    split_req_idx = int(np.searchsorted(qsl_np, split_point, side="right") - 1)
    tokens_in_first_chunk = split_point - int(qsl_np[split_req_idx])
    orig_q_lens = common.query_start_loc_cpu[1:] - common.query_start_loc_cpu[:-1]

    # Check query length continuity: first-chunk + second-chunk == original qlen
    # First ubatch last request query length
    qlen_first_last = int(
        first_meta.query_start_loc_cpu[-1] - first_meta.query_start_loc_cpu[-2]
    )
    # Second ubatch first request query length
    qlen_second_first = int(
        second_meta.query_start_loc_cpu[1] - second_meta.query_start_loc_cpu[0]
    )
    assert qlen_first_last == tokens_in_first_chunk
    assert qlen_first_last + qlen_second_first == int(orig_q_lens[split_req_idx])

    # Check seq_lens adjustments
    # Context lengths per original request
    context_lens = [s - q for s, q in zip(seq_lens, query_lens)]

    # First ubatch: last request's seq_len should be
    #  context + tokens_in_first_chunk
    expected_seqlen = context_lens[split_req_idx] + tokens_in_first_chunk
    assert int(first_meta.seq_lens[-1]) == expected_seqlen

    # For full preceding requests in first ubatch, seq_lens should match
    #  originals
    for i in range(first_meta.num_reqs - 1):
        assert int(first_meta.seq_lens[i]) == seq_lens[i]

    # Second ubatch: first request (continuation) seq_len should be full
    #  original
    assert int(second_meta.seq_lens[0]) == seq_lens[split_req_idx]
    # Any following full requests in second ubatch should match originals
    for j in range(1, second_meta.num_reqs):
        # Map to original request index
        orig_idx = split_req_idx + j
        assert int(second_meta.seq_lens[j]) == seq_lens[orig_idx]


# ---------------------------------------------------------------------------
# Tests for split_decodes_prefills_and_extends (6-tuple version)
# ---------------------------------------------------------------------------


def apply_split_decodes_prefills_and_extends(
    query_lens: list[int],
    seq_lens: list[int] | None = None,
    decode_threshold: int = 1,
):
    """Helper: build CommonAttentionMetadata and call the 6-tuple split."""
    device = torch.device("cpu")
    if seq_lens is None:
        seq_lens = [10 * (i + 1) for i in range(len(query_lens))]
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=16,
        device=device,
    )
    return split_decodes_prefills_and_extends(common, decode_threshold=decode_threshold)


def test_split_decodes_prefills_and_extends_returns_ints():
    """All 6 return values must be Python ints, not tensors."""
    result = apply_split_decodes_prefills_and_extends([1, 3, 8])
    assert len(result) == 6
    for i, val in enumerate(result):
        assert isinstance(val, int), (
            f"Return value at position {i} is {type(val).__name__}, expected int: {val}"
        )


def test_split_decodes_prefills_and_extends_all_decodes():
    """All single-token requests → all decodes, no extends, no prefills."""
    query_lens = [1, 1, 1]
    seq_lens = [10, 20, 30]  # seq_len > query_len so not prefills
    nd, ne, np_, ndt, net, npt = apply_split_decodes_prefills_and_extends(
        query_lens, seq_lens=seq_lens
    )
    assert nd == 3
    assert ne == 0
    assert np_ == 0
    assert ndt == 3
    assert net == 0
    assert npt == 0
    assert ndt + net + npt == sum(query_lens)


def test_split_decodes_prefills_and_extends_all_prefills():
    """All first-token requests (seq_len == query_len) → all prefills."""
    query_lens = [5, 6, 7]
    seq_lens = [5, 6, 7]  # seq_len == query_len → prefill
    nd, ne, np_, ndt, net, npt = apply_split_decodes_prefills_and_extends(
        query_lens, seq_lens=seq_lens, decode_threshold=1
    )
    assert nd == 0
    assert ne == 0
    assert np_ == 3
    assert ndt == 0
    assert net == 0
    assert npt == sum(query_lens)
    assert ndt + net + npt == sum(query_lens)


def test_split_decodes_prefills_and_extends_all_extends():
    """Multi-token requests where seq_len > query_len → all extends."""
    query_lens = [3, 4, 5]
    seq_lens = [10, 20, 30]  # seq_len > query_len, query_len > 1 → extends
    nd, ne, np_, ndt, net, npt = apply_split_decodes_prefills_and_extends(
        query_lens, seq_lens=seq_lens, decode_threshold=1
    )
    assert nd == 0
    assert ne == 3
    assert np_ == 0
    assert ndt == 0
    assert net == sum(query_lens)
    assert npt == 0
    assert ndt + net + npt == sum(query_lens)


def test_split_decodes_prefills_and_extends_mixed():
    """Mixed batch: decodes first, then extends, then prefills."""
    # decodes: query_len=1, seq_len > query_len
    # extends: query_len > 1, seq_len > query_len
    # prefills: query_len == seq_len (fresh prompts)
    query_lens = [1, 1, 3, 4, 6, 8]
    seq_lens = [20, 30, 10, 15, 6, 8]
    # requests 0,1: decode (query=1, seq>1)
    # requests 2,3: extend (query>1, seq>query)
    # requests 4,5: prefill (query==seq)
    nd, ne, np_, ndt, net, npt = apply_split_decodes_prefills_and_extends(
        query_lens, seq_lens=seq_lens, decode_threshold=1
    )
    assert nd == 2
    assert ne == 2
    assert np_ == 2
    assert ndt == 2  # 1 + 1
    assert net == 7  # 3 + 4
    assert npt == 14  # 6 + 8
    assert ndt + net + npt == sum(query_lens)


def test_split_decodes_prefills_and_extends_token_counts_sum():
    """Token counts across all three groups always sum to total tokens."""
    test_cases = [
        ([1, 1, 1], [10, 20, 30]),
        ([5, 5, 5], [5, 5, 5]),
        ([1, 3, 8], None),
        ([1, 1, 4, 6, 10, 10], [20, 30, 4, 6, 10, 10]),
    ]
    for query_lens, seq_lens in test_cases:
        result = apply_split_decodes_prefills_and_extends(query_lens, seq_lens=seq_lens)
        nd, ne, np_, ndt, net, npt = result
        assert ndt + net + npt == sum(query_lens), (
            f"Token count mismatch for query_lens={query_lens}: "
            f"{ndt}+{net}+{npt} != {sum(query_lens)}"
        )
        assert nd + ne + np_ == len(query_lens), (
            f"Request count mismatch: {nd}+{ne}+{np_} != {len(query_lens)}"
        )


# ---------------------------------------------------------------------------
# Tests for make_kv_sharing_fast_prefill_common_attn_metadata
# ---------------------------------------------------------------------------


def _make_fast_prefill_input(
    query_lens: list[int],
    seq_lens: list[int],
    logits_indices: list[int],
) -> CommonAttentionMetadata:
    """Build a minimal CommonAttentionMetadata for fast-prefill metadata tests.

    All tensors are kept on CPU so no GPU is required.
    """
    n = len(query_lens)
    qsl = [0] + list(__import__("itertools").accumulate(query_lens))
    query_start_loc = torch.tensor(qsl, dtype=torch.int32)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    li = torch.tensor(logits_indices, dtype=torch.int32)
    dummy = torch.zeros(0, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=seq_lens_t,
        num_reqs=n,
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=dummy,
        slot_mapping=dummy,
        seq_lens_cpu_upper_bound=seq_lens_t.clone(),
        logits_indices_padded=li,
        num_logits_indices=len(logits_indices),
    )


def test_make_kv_sharing_fast_prefill_pure_decode_shortcircuit():
    """max_query_len == 1 → function returns the input unchanged."""
    meta = _make_fast_prefill_input(
        query_lens=[1, 1, 1],
        seq_lens=[10, 20, 30],
        logits_indices=[0, 1, 2],
    )
    result = make_kv_sharing_fast_prefill_common_attn_metadata(meta)
    assert result is meta


@pytest.mark.parametrize(
    "query_lens,seq_lens,logits_indices,exp_total,exp_max",
    [
        # 3 requests: 3/2/1 decode tokens each
        # query_start_loc = [0, 5, 10, 15]; valid range for each req:
        #   req0=[0,4], req1=[5,9], req2=[10,14]
        (
            [5, 5, 5],
            [41, 31, 40],
            [0, 1, 2, 5, 8, 12],
            6,
            3,
        ),
        # 2 requests: 1 decode token each (equal distribution)
        # query_start_loc = [0, 3, 6]; req0=[0,2], req1=[3,5]
        (
            [3, 3],
            [20, 30],
            [0, 3],
            2,
            1,
        ),
        # 1 request: all tokens are decode
        # query_start_loc = [0, 4]; req0=[0,3]
        (
            [4],
            [10],
            [0, 1, 2, 3],
            4,
            4,
        ),
    ],
)
def test_make_kv_sharing_fast_prefill_scalar_values(
    query_lens, seq_lens, logits_indices, exp_total, exp_max
):
    """num_actual_tokens and max_query_len are correct Python ints."""
    meta = _make_fast_prefill_input(query_lens, seq_lens, logits_indices)
    result = make_kv_sharing_fast_prefill_common_attn_metadata(meta)

    assert isinstance(result.num_actual_tokens, int), (
        f"num_actual_tokens should be int, got {type(result.num_actual_tokens)}"
    )
    assert isinstance(result.max_query_len, int), (
        f"max_query_len should be int, got {type(result.max_query_len)}"
    )
    assert result.num_actual_tokens == exp_total, (
        f"Expected num_actual_tokens={exp_total}, got {result.num_actual_tokens}"
    )
    assert result.max_query_len == exp_max, (
        f"Expected max_query_len={exp_max}, got {result.max_query_len}"
    )


def test_make_kv_sharing_fast_prefill_query_start_loc_consistent():
    """query_start_loc_cpu last element equals num_actual_tokens."""
    # query_start_loc=[0,5,10,15]; valid idx: req0=[0,4], req1=[5,9], req2=[10,14]
    meta = _make_fast_prefill_input(
        query_lens=[5, 5, 5],
        seq_lens=[41, 31, 40],
        logits_indices=[0, 1, 2, 5, 8, 12],
    )
    result = make_kv_sharing_fast_prefill_common_attn_metadata(meta)

    cpu = result.query_start_loc_cpu
    assert cpu.device.type == "cpu"
    assert int(cpu[-1]) == result.num_actual_tokens
    # Adjacent differences must be non-negative
    diffs = cpu[1:] - cpu[:-1]
    assert (diffs >= 0).all()


def test_build_attention_metadata_zeros_stale_is_prefilling():
    """_build_attention_metadata zeroes is_prefilling for padded rows."""
    from unittest.mock import MagicMock, patch

    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    num_reqs = 3
    num_reqs_padded = 5

    # Real rows [0-2] have known computed/prompt values; padded rows [3-4]
    # carry stale data from a prior prefill (num_computed < num_prompt → True).
    num_computed = torch.tensor([50, 100, 200, 10, 20], dtype=torch.int32)
    num_prompt = torch.tensor([50, 200, 200, 100, 200], dtype=torch.int32)

    runner = MagicMock()
    runner.kv_cache_config.kv_cache_groups = [
        MagicMock()
    ]  # non-empty: skip early return
    runner.attn_groups = [[]]  # empty inner list: inner loop never runs
    runner.input_batch.num_computed_tokens_cpu_tensor = num_computed
    runner.input_batch.num_prompt_tokens_cpu_tensor = num_prompt
    runner.optimistic_seq_lens_cpu = torch.tensor([100, 200, 300, 0, 0])
    runner.query_start_loc.gpu = torch.zeros(num_reqs_padded + 1, dtype=torch.int32)
    runner.query_start_loc.cpu = torch.zeros(num_reqs_padded + 1, dtype=torch.int32)
    runner.seq_lens = torch.zeros(num_reqs_padded, dtype=torch.int32)
    runner.positions = torch.zeros(num_reqs_padded, dtype=torch.int64)
    runner.routed_experts_initialized = False
    runner.use_async_spec_decode = False
    runner.dcp_world_size = 1
    runner.speculative_config = None
    runner.is_mm_prefix_lm = False
    runner._get_encoder_seq_lens.return_value = (None, None)

    # Intercept CommonAttentionMetadata construction to capture is_prefilling.
    # With speculative_config=None the constructor is called exactly once (for
    # cm_base), so captured reflects what the fix produced before storage.
    captured_is_prefilling = None
    original_init = CommonAttentionMetadata.__init__

    def capturing_init(self, *args, **kwargs):
        nonlocal captured_is_prefilling
        if "is_prefilling" in kwargs:
            captured_is_prefilling = kwargs["is_prefilling"]
        original_init(self, *args, **kwargs)

    with patch.object(CommonAttentionMetadata, "__init__", capturing_init):
        GPUModelRunner._build_attention_metadata(
            runner,
            num_tokens=num_reqs,
            num_reqs=num_reqs,
            max_query_len=1,
            num_tokens_padded=num_reqs_padded,
            num_reqs_padded=num_reqs_padded,
            slot_mappings={0: torch.zeros(num_reqs_padded, dtype=torch.int64)},
        )

    assert captured_is_prefilling is not None
    assert not captured_is_prefilling[0]  # decode  (50 >= 50)
    assert captured_is_prefilling[1]  # prefill (100 < 200)
    assert not captured_is_prefilling[2]  # decode  (200 >= 200)
    assert not captured_is_prefilling[3]  # stale data (10 < 100) zeroed
    assert not captured_is_prefilling[4]  # stale data (20 < 200) zeroed
