# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import sys
from dataclasses import replace
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
    builder.kv_cache_dtype = "fp8"
    builder.mla_dims = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
    builder.topk_tokens = topk_tokens
    builder._seq_lens_ub_slack = 0
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
    builder._use_persistent_metadata = True
    builder._num_attention_heads = 16
    builder._num_compute_units = current_platform.num_compute_units()
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
        seq_lens_cpu_upper_bound=seq_lens,
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=16,
        block_table_tensor=torch.arange(16, dtype=torch.int32, device="cpu").view(2, 8),
        slot_mapping=torch.arange(2, dtype=torch.int64, device="cpu"),
    )


def _make_mixed_common_metadata():
    # req0: query_len 1 (decode), req1: query_len 4 (prefill) -> decode-first
    query_start_loc = torch.tensor([0, 1, 5], dtype=torch.int32, device="cpu")
    seq_lens = torch.tensor([16, 10], dtype=torch.int32, device="cpu")
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens,
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=4,
        max_seq_len=16,
        block_table_tensor=torch.arange(16, dtype=torch.int32, device="cpu").view(2, 8),
        slot_mapping=torch.arange(5, dtype=torch.int64, device="cpu"),
    )


def _patch_build_deps(monkeypatch, events=None):
    """Stub the aiter kernel, triton helper and CUDA sync so ``build()`` runs
    on CPU."""

    def fake_generate_sparse_seqlen_triton(
        query_lens, seq_lens, cu_query_lens, topk_token, num_tokens, max_query_len
    ):
        return torch.zeros(num_tokens, dtype=torch.int32, device="cpu")

    fake_aiter = _FakeAiter("aiter")
    fake_aiter.get_mla_metadata_v1 = Mock(side_effect=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setattr(
        sparse_mod, "generate_sparse_seqlen_triton", fake_generate_sparse_seqlen_triton
    )
    monkeypatch.setattr(
        sparse_mod.torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(
            synchronize=lambda: events.append("sync") if events is not None else None
        ),
    )
    return fake_aiter


def test_build_populates_decode_only_split_fields(monkeypatch):
    """Decode-only batch: all reqs count as decodes, prefill fields default."""
    builder = _make_builder()
    _patch_build_deps(monkeypatch)

    md = builder.build(
        common_prefix_len=0, common_attn_metadata=_make_common_metadata()
    )

    assert md.num_decodes == 2
    assert md.num_prefills == 0
    assert md.num_decode_tokens == 2
    assert md.prefill_max_seq_len == 0
    assert md.prefill is None


def test_build_populates_mixed_split_fields(monkeypatch):
    """Mixed decode+prefill batch: split is reported, prefill fields stay
    default because this impl always runs the MQA path."""
    builder = _make_builder()
    _patch_build_deps(monkeypatch)

    md = builder.build(
        common_prefix_len=0, common_attn_metadata=_make_mixed_common_metadata()
    )

    assert md.num_decodes == 1
    assert md.num_prefills == 1
    assert md.num_decode_tokens == 1
    assert md.prefill_max_seq_len == 0
    assert md.prefill is None


def test_sink_build_skips_persistent_metadata(monkeypatch):
    builder = _make_builder()
    builder._use_persistent_metadata = False
    fake_aiter = _patch_build_deps(monkeypatch)

    md = builder.build(
        common_prefix_len=0, common_attn_metadata=_make_common_metadata()
    )

    fake_aiter.get_mla_metadata_v1.assert_not_called()
    assert md.work_meta_data is None
    assert md.work_indptr is None
    assert md.work_info_set is None
    assert md.reduce_indptr is None
    assert md.reduce_final_map is None
    assert md.reduce_partial_map is None
    assert builder._prev_metadata_key is None


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
    assert fake_get_mla_metadata_v1_mock.call_args.kwargs["max_split_per_batch"] == 1

    events.clear()

    builder.build(common_prefix_len=0, common_attn_metadata=common_metadata)

    assert events == []
    assert fake_get_mla_metadata_v1_mock.call_count == 1


@pytest.mark.parametrize(
    "slack,ub_offset,expect_sync",
    [
        # No spec decode: the upper bound is exact.
        (0, 0, False),
        # Spec decode, context upper bounds (17, 9) minus slack are >= topk (4).
        (2, 2, False),
        # Spec decode, context upper bound 7 minus slack 8 < topk: needs exact.
        (8, 0, True),
    ],
)
def test_persistent_metadata_key_from_upper_bound(
    monkeypatch, slack, ub_offset, expect_sync
):
    """The metadata key skips the seq_lens D2H copy when the host upper bound
    provably gives the same key, and matches the exact key either way."""
    syncs: list[str] = []

    @contextlib.contextmanager
    def fake_gpu_sync_allowed():
        syncs.append("sync")
        yield

    _patch_build_deps(monkeypatch)
    monkeypatch.setattr(sparse_mod, "gpu_sync_allowed", fake_gpu_sync_allowed)
    common_metadata = _make_common_metadata()

    exact = _make_builder()
    exact.build(
        common_prefix_len=0,
        common_attn_metadata=replace(common_metadata, seq_lens_cpu_upper_bound=None),
    )
    assert syncs == ["sync"]
    syncs.clear()

    builder = _make_builder()
    builder._seq_lens_ub_slack = slack
    upper_bound = common_metadata.seq_lens + ub_offset
    builder.build(
        common_prefix_len=0,
        common_attn_metadata=replace(
            common_metadata, seq_lens_cpu_upper_bound=upper_bound
        ),
    )

    assert syncs == (["sync"] if expect_sync else [])
    assert builder._prev_metadata_key == exact._prev_metadata_key
