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


def _make_builder():
    builder = object.__new__(sparse_mod.ROCMAiterMLASparseMetadataBuilder)
    max_num_batched_tokens = 8
    topk_tokens = 4

    builder.device = torch.device("cpu")
    builder.kv_cache_spec = SimpleNamespace(block_size=1)
    builder.model_dtype = torch.bfloat16
    builder.topk_tokens = topk_tokens
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

    events.clear()

    builder.build(common_prefix_len=0, common_attn_metadata=common_metadata)

    assert events == []
    assert fake_get_mla_metadata_v1_mock.call_count == 1
