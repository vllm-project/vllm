# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HiSparse decode/prefill routing in the ROCm AITER sparse MLA impl.

The ROCm analogue of ``test_flashattn_hisparse_decode_uses_index_group``. Under
HiSparse the KV base handed to the AITER kernel is no longer the paged KV
cache: decode tokens attend the hot buffer the residency resolver just filled,
and prefill tokens attend rows staged from host memory. Getting that
substitution wrong reads plausible-looking garbage without crashing, so these
tests assert on the exact tensor that reaches ``_forward_mla`` and on the
ragged indices built for it.

``_forward_mla`` is the only thing stubbed -- the dense->ragged conversion and
the logical->physical remap run for real.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "ROCm AITER sparse MLA index group test requires ROCm.",
        allow_module_level=True,
    )

from vllm._aiter_ops import is_aiter_found_and_supported

if not is_aiter_found_and_supported():
    pytest.skip(
        "ROCm AITER sparse MLA index group test requires a supported AITER "
        "installation.",
        allow_module_level=True,
    )

from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseImpl,
)

DEVICE = current_platform.device_type
NUM_HEADS = 16  # A multiple of AITER's 16-head floor, so q is not head-padded.
KV_LORA_RANK = 8
TOPK = 128  # The remap kernel tiles columns by 128 and asserts divisibility.


def _rows(*values_per_row):
    """Build a ``-1``-padded ``[rows, TOPK]`` index tensor."""
    padded = [list(v) + [-1] * (TOPK - len(v)) for v in values_per_row]
    return torch.tensor(padded, dtype=torch.int32, device=DEVICE)


def _make_impl(index_group):
    impl = object.__new__(ROCMAiterMLASparseImpl)
    impl.kv_cache_dtype = "auto"
    impl.num_heads = NUM_HEADS
    impl.kv_lora_rank = KV_LORA_RANK
    impl.index_group = index_group
    impl.index_group_index = 0
    return impl


def _make_metadata(num_tokens, num_decode_tokens, num_decodes, num_prefills):
    return SimpleNamespace(
        num_actual_tokens=num_tokens,
        num_decode_tokens=num_decode_tokens,
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        topk_tokens=TOPK,
        block_size=1,
        decode_max_query_len=1,
        max_query_len=num_tokens - num_decode_tokens,
        qo_indptr=torch.arange(num_tokens + 1, dtype=torch.int32, device=DEVICE),
        paged_kv_indptr=torch.zeros(num_tokens + 1, dtype=torch.int32, device=DEVICE),
        paged_kv_indices=torch.full(
            (num_tokens * TOPK,), -7, dtype=torch.int32, device=DEVICE
        ),
        paged_kv_last_page_len=torch.ones(num_tokens, dtype=torch.int32, device=DEVICE),
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE),
    )


def _capture_forward_mla(impl):
    """Stub ``_forward_mla``, recording the KV base and indices it was handed."""
    calls = []

    def fake(layer, q, kv_cache, attn_metadata, *, token_slice=None):
        source = token_slice if token_slice is not None else attn_metadata
        span = int(source.paged_kv_indptr[-1])
        calls.append(
            SimpleNamespace(
                kv_cache=kv_cache,
                num_tokens=q.shape[0],
                indptr=source.paged_kv_indptr.clone(),
                indices=source.paged_kv_indices[:span].clone(),
                num_decode_tokens=source.num_decode_tokens,
                num_prefills=source.num_prefills,
            )
        )
        return (
            torch.zeros(
                q.shape[0], NUM_HEADS, KV_LORA_RANK, dtype=q.dtype, device=DEVICE
            ),
            None,
        )

    impl._forward_mla = fake
    return calls


def _make_index_group(*, physical_kv_cache, resident=True, staged=None):
    group = object.__new__(HiSparseMLAIndexGroup)
    group.physical_kv_cache = MagicMock(return_value=physical_kv_cache)
    group.cache = MagicMock(
        return_value=SimpleNamespace(all_context_pages_resident=resident)
    )
    group.stage_prefill_rows = MagicMock(return_value=staged)
    return group


def _kv_buffer(num_rows):
    return torch.zeros(num_rows, 1, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)


def test_hisparse_decode_attends_hot_buffer_with_ragged_indices():
    """Decode reads ``physical_kv_cache``, flattened from the resolver's rows."""
    num_tokens = 3
    hot = _kv_buffer(64)
    # Row 1 is short: its padding must drop out and shrink that row's span.
    physical = _rows([10, 11, 12, 13], [20, 21], [30, 31, 32])
    group = _make_index_group(physical_kv_cache=hot)
    group.convert_decode_logical_to_physical_topk = MagicMock(return_value=physical)

    impl = _make_impl(group)
    impl.topk_indices_buffer = torch.zeros(
        num_tokens, TOPK, dtype=torch.int32, device=DEVICE
    )
    calls = _capture_forward_mla(impl)
    metadata = _make_metadata(num_tokens, num_tokens, num_tokens, 0)
    q = torch.zeros(num_tokens, NUM_HEADS, KV_LORA_RANK, device=DEVICE)

    output, lse = ROCMAiterMLASparseImpl.forward_mqa(
        impl, q, torch.empty(1, device=DEVICE), metadata, SimpleNamespace()
    )

    assert lse is None
    assert output.shape == (num_tokens, NUM_HEADS, KV_LORA_RANK)
    group.convert_decode_logical_to_physical_topk.assert_called_once()
    group.stage_prefill_rows.assert_not_called()

    assert len(calls) == 1
    # The hot buffer, not the paged KV cache passed to forward_mqa.
    assert calls[0].kv_cache.data_ptr() == hot.data_ptr()
    assert calls[0].indptr.tolist() == [0, 4, 6, 9]
    assert calls[0].indices.tolist() == [10, 11, 12, 13, 20, 21, 30, 31, 32]


def test_hisparse_prefill_attends_staged_rows_when_not_resident():
    """A non-resident prefill stages host rows and attends those instead."""
    num_tokens = 2
    hot = _kv_buffer(64)
    staged_cache = _kv_buffer(32)
    # block_size 1, so the remap is a straight block_table lookup.
    block_table = torch.tensor([[5, 6, 7, 8]], dtype=torch.int32, device=DEVICE)
    req_ids = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
    group = _make_index_group(
        physical_kv_cache=hot,
        resident=False,
        staged=(staged_cache, block_table, req_ids),
    )
    group.convert_decode_logical_to_physical_topk = MagicMock()

    impl = _make_impl(group)
    impl.topk_indices_buffer = _rows([0, 1, 2], [3])
    calls = _capture_forward_mla(impl)
    metadata = _make_metadata(num_tokens, 0, 0, 1)
    q = torch.zeros(num_tokens, NUM_HEADS, KV_LORA_RANK, device=DEVICE)

    ROCMAiterMLASparseImpl.forward_mqa(
        impl, q, torch.empty(1, device=DEVICE), metadata, SimpleNamespace()
    )

    group.convert_decode_logical_to_physical_topk.assert_not_called()
    group.stage_prefill_rows.assert_called_once()

    assert len(calls) == 1
    assert calls[0].kv_cache.data_ptr() == staged_cache.data_ptr()
    assert calls[0].indptr.tolist() == [0, 3, 4]
    assert calls[0].indices.tolist() == [5, 6, 7, 8]


def test_hisparse_mixed_batch_splits_kv_bases():
    """A mixed batch attends two bases and concatenates the two outputs."""
    num_decode_tokens, num_prefill_tokens = 1, 2
    num_tokens = num_decode_tokens + num_prefill_tokens
    hot = _kv_buffer(64)
    staged_cache = _kv_buffer(32)
    block_table = torch.tensor([[5, 6, 7, 8]], dtype=torch.int32, device=DEVICE)
    req_ids = torch.zeros(num_prefill_tokens, dtype=torch.int32, device=DEVICE)
    group = _make_index_group(
        physical_kv_cache=hot,
        resident=True,  # Ignored: a batch with decode tokens always stages.
        staged=(staged_cache, block_table, req_ids),
    )
    group.convert_decode_logical_to_physical_topk = MagicMock(
        return_value=_rows([40, 41])
    )

    impl = _make_impl(group)
    impl.topk_indices_buffer = _rows([0, 1, 2, 3], [0, 1], [2])
    calls = _capture_forward_mla(impl)
    metadata = _make_metadata(num_tokens, num_decode_tokens, 1, 1)
    q = torch.zeros(num_tokens, NUM_HEADS, KV_LORA_RANK, device=DEVICE)

    output, _ = ROCMAiterMLASparseImpl.forward_mqa(
        impl, q, torch.empty(1, device=DEVICE), metadata, SimpleNamespace()
    )

    assert output.shape == (num_tokens, NUM_HEADS, KV_LORA_RANK)
    group.stage_prefill_rows.assert_called_once()

    decode_call, prefill_call = calls
    assert decode_call.kv_cache.data_ptr() == hot.data_ptr()
    assert decode_call.num_tokens == num_decode_tokens
    assert decode_call.num_decode_tokens == num_decode_tokens
    assert decode_call.num_prefills == 0
    assert decode_call.indices.tolist() == [40, 41]

    assert prefill_call.kv_cache.data_ptr() == staged_cache.data_ptr()
    assert prefill_call.num_tokens == num_prefill_tokens
    assert prefill_call.num_decode_tokens == 0
    assert prefill_call.num_prefills == 1
    assert prefill_call.indices.tolist() == [5, 6, 7]


def test_non_hisparse_index_group_keeps_paged_kv_cache():
    """The plain sparse path must not be diverted onto a hot buffer."""
    num_tokens = 2
    paged = _kv_buffer(64)
    impl = _make_impl(index_group=None)
    impl.topk_indices_buffer = torch.zeros(
        num_tokens, TOPK, dtype=torch.int32, device=DEVICE
    )
    calls = _capture_forward_mla(impl)
    metadata = _make_metadata(num_tokens, num_tokens, num_tokens, 0)
    metadata.block_table = torch.zeros(1, 4, dtype=torch.int32, device=DEVICE)
    q = torch.zeros(num_tokens, NUM_HEADS, KV_LORA_RANK, device=DEVICE)

    ROCMAiterMLASparseImpl.forward_mqa(impl, q, paged, metadata, SimpleNamespace())

    assert len(calls) == 1
    assert calls[0].kv_cache.data_ptr() == paged.data_ptr()
