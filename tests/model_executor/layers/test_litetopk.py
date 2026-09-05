# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers import (
    litetopk_indexer,
    sparse_attn_indexer,
)
from vllm.utils import deep_gemm as deep_gemm_utils
from vllm.v1.attention.backends.mla import indexer as indexer_metadata


@pytest.fixture(autouse=True)
def clear_litetopk_caches():
    caches = (
        litetopk_indexer.production_extension_available,
        deep_gemm_utils._import_deep_gemm,
        deep_gemm_utils._get_fp8_fp4_mqa_logits_out_impl,
        deep_gemm_utils._probe_fp8_fp4_mqa_logits_out,
    )
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


def test_litetopk_envs_are_registered():
    expected = {
        "VLLM_LITETOPK",
        "VLLM_LITETOPK_BUILD",
        "VLLM_LITETOPK_SO",
        "VLLM_LITETOPK_SO_SHA256",
        "VLLM_LITETOPK_PRODUCTION_MIN_S",
        "VLLM_LITETOPK_FP4_PRODUCTION_MIN_S",
        "VLLM_LITETOPK_MERGE_CAP",
        "VLLM_LITETOPK_OVF_WATERMARK",
        "VLLM_LITETOPK_PCP_FRONTIER_CARRY",
        "VLLM_LITETOPK_TP_QUERY_SHARD",
    }
    assert expected <= envs.environment_variables.keys()


def test_deepgemm_mqa_out_keyword_detection():
    def legacy_backend(*args, clean_logits):
        return None

    def output_backend(*args, clean_logits, out=None):
        return out

    assert not deep_gemm_utils._callable_accepts_keyword(legacy_backend, "out")
    assert deep_gemm_utils._callable_accepts_keyword(output_backend, "out")


def test_deepgemm_mqa_out_wrapper_requires_alias(monkeypatch):
    placeholder = torch.empty(1)
    out = torch.empty(16)

    def output_backend(*args, clean_logits, out=None):
        return out[:4].view(2, 2)

    monkeypatch.setattr(deep_gemm_utils, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        deep_gemm_utils,
        "_get_fp8_fp4_mqa_logits_out_impl",
        lambda: output_backend,
    )
    result = deep_gemm_utils.fp8_fp4_mqa_logits(
        (placeholder, None),
        (placeholder, placeholder),
        placeholder,
        placeholder,
        placeholder,
        clean_logits=False,
        out=out,
    )
    assert result.data_ptr() == out.data_ptr()

    def non_aliasing_backend(*args, clean_logits, out=None):
        return out.clone()

    monkeypatch.setattr(
        deep_gemm_utils,
        "_get_fp8_fp4_mqa_logits_out_impl",
        lambda: non_aliasing_backend,
    )
    with pytest.raises(RuntimeError, match="did not return an alias"):
        deep_gemm_utils.fp8_fp4_mqa_logits(
            (placeholder, None),
            (placeholder, placeholder),
            placeholder,
            placeholder,
            placeholder,
            clean_logits=False,
            out=out,
        )


def test_fused_planner_preflights_extension_and_capacity(monkeypatch):
    common_ops = {
        "plan_and_permuted_paged_gather_out": object(),
        "seed_prep_litetopk_": object(),
        "map_topk_vote_stats_litetopk_": object(),
        "cand_count_stats_litetopk_": object(),
    }
    fp8_ext = SimpleNamespace(
        **common_ops,
        mqa_logits_dsa_static_hot_nohist_litetopk_=object(),
        h2048_safe_topk_out_litetopk_=object(),
    )
    monkeypatch.setattr(litetopk_indexer, "ENABLED", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(litetopk_indexer, "_ext", lambda: fp8_ext)
    monkeypatch.setattr(
        deep_gemm_utils, "is_fp8_fp4_mqa_logits_out_supported", lambda: True
    )

    assert indexer_metadata._litetopk_extension_ready_for_planning(
        use_fp4=False,
        topk=2048,
    )
    assert litetopk_indexer.MERGE_CAP == 49152
    assert litetopk_indexer.OVF_WATERMARK == 40960
    assert litetopk_indexer.minimum_merge_cap(2048) == 49152
    assert litetopk_indexer.minimum_merge_cap(1008) == 32256
    assert litetopk_indexer.minimum_merge_cap(512) == 16384

    monkeypatch.setattr(litetopk_indexer, "MERGE_CAP", 49151)
    litetopk_indexer.production_extension_available.cache_clear()
    assert not indexer_metadata._litetopk_extension_ready_for_planning(
        use_fp4=False,
        topk=2048,
    )


def test_tp8_fp4_local_query_lengths_are_opt_in(monkeypatch):
    monkeypatch.setattr(litetopk_indexer, "TP_QUERY_SHARD_ENABLED", False)
    assert litetopk_indexer._supported_fused_query_len(8192)
    assert not litetopk_indexer._supported_fused_query_len(1024)
    assert not litetopk_indexer._supported_fused_query_len(4096)

    monkeypatch.setattr(litetopk_indexer, "TP_QUERY_SHARD_ENABLED", True)
    assert litetopk_indexer._supported_fused_query_len(1024)
    assert litetopk_indexer._supported_fused_query_len(1016)
    assert litetopk_indexer._supported_fused_query_len(4096)
    assert litetopk_indexer._supported_fused_query_len(4088)
    assert litetopk_indexer._supported_fused_query_len(2048)
    assert litetopk_indexer._supported_fused_query_len(2032)
    assert not litetopk_indexer._supported_fused_query_len(512)


@pytest.mark.parametrize("full_q", (8192, 8128, 32768, 32704))
def test_tp8_fp4_shard_uses_compressed_query_offset(monkeypatch, full_q):
    group = SimpleNamespace(world_size=8, rank_in_group=6)
    monkeypatch.setattr(envs, "VLLM_LITETOPK_TP_QUERY_SHARD", True)
    monkeypatch.setattr(sparse_attn_indexer, "get_tp_group", lambda: group)
    monkeypatch.setattr(sparse_attn_indexer, "_LITETOPK_TP_SHARD_BUFS", {})
    monkeypatch.setattr(sparse_attn_indexer, "_LITETOPK_TP_SHARD_STATUS", {})
    monkeypatch.setattr(sparse_attn_indexer, "_LITETOPK_TP_SHARD_LOGGED", False)

    shard = sparse_attn_indexer._litetopk_tp_query_shard(
        full_q,
        512,
        torch.device("cpu"),
        use_fp4_cache=True,
        use_pcp=False,
        pcp_world_size=1,
        compress_ratio=4,
        num_heads=64,
        num_reqs=1,
        dcp_world_size=1,
    )
    assert shard is not None
    lo, hi, out, local_status, all_status = shard
    assert (lo, hi) == (6 * (full_q // 8), 7 * (full_q // 8))
    assert out.shape == (full_q // 8, 512)
    assert local_status.shape == (1,)
    assert all_status.shape == (8,)

    raw_common_numerator = 1_000_003
    global_common_ke = raw_common_numerator // 4
    local_common_ke = global_common_ke + (
        sparse_attn_indexer._litetopk_tp_compressed_row_offset(lo, 4)
    )
    assert local_common_ke == (raw_common_numerator + lo) // 4


def test_tp8_fp4_32k_planning_is_explicitly_opt_in():
    common = {
        "num_reqs": 1,
        "total_seq_len": 262144,
        "query_len": 32768,
        "fused_min_seq_len": 65536,
    }
    assert not indexer_metadata._should_plan_fused_indexer(**common)
    assert indexer_metadata._should_plan_fused_indexer(
        **common, allow_tp8_fp4_query_shard=True
    )
    assert not indexer_metadata._should_plan_fused_indexer(
        **(common | {"query_len": 32760}),
        allow_tp8_fp4_query_shard=True,
    )


@pytest.mark.parametrize("query_len", (32768, 32704))
def test_tp8_fp4_32k_planner_keeps_whole_query(query_len):
    chunks = indexer_metadata.split_indexer_prefill_chunks(
        torch.tensor([262144]),
        torch.tensor([query_len]),
        workspace_size=1 << 22,
        max_logits_bytes=2 * 1024**3,
        fused_min_seq_len=65536,
        allow_tp8_fp4_query_shard=True,
    )
    assert chunks == [(slice(0, 1), slice(0, query_len))]

    budgeted_chunks = indexer_metadata.split_indexer_prefill_chunks(
        torch.tensor([262144]),
        torch.tensor([query_len]),
        workspace_size=1 << 22,
        max_logits_bytes=2 * 1024**3,
        fused_min_seq_len=65536,
    )
    assert len(budgeted_chunks) > 1


def test_tp8_fp4_shard_rejects_unqualified_layouts(monkeypatch):
    group = SimpleNamespace(world_size=8, rank_in_group=0)
    monkeypatch.setattr(envs, "VLLM_LITETOPK_TP_QUERY_SHARD", True)
    monkeypatch.setattr(sparse_attn_indexer, "get_tp_group", lambda: group)
    common = {
        "use_fp4_cache": True,
        "use_pcp": False,
        "pcp_world_size": 1,
        "compress_ratio": 4,
        "num_heads": 64,
        "num_reqs": 1,
        "dcp_world_size": 1,
    }
    for override in (
        {"use_fp4_cache": False},
        {"use_pcp": True},
        {"pcp_world_size": 2},
        {"compress_ratio": 1},
        {"num_heads": 32},
        {"num_reqs": 2},
        {"dcp_world_size": 2},
    ):
        assert (
            sparse_attn_indexer._litetopk_tp_query_shard(
                8192,
                512,
                torch.device("cpu"),
                **(common | override),
            )
            is None
        )


@pytest.mark.parametrize("full_q", (8192, 8128))
def test_tp4_fp8_glm_shard_profile(monkeypatch, full_q):
    group = SimpleNamespace(world_size=4, rank_in_group=3)
    monkeypatch.setattr(envs, "VLLM_LITETOPK_TP_QUERY_SHARD", True)
    monkeypatch.setattr(sparse_attn_indexer, "get_tp_group", lambda: group)
    monkeypatch.setattr(sparse_attn_indexer, "_LITETOPK_TP_SHARD_BUFS", {})
    monkeypatch.setattr(sparse_attn_indexer, "_LITETOPK_TP_SHARD_STATUS", {})
    monkeypatch.setattr(sparse_attn_indexer, "_LITETOPK_TP_SHARD_LOGGED", False)

    shard = sparse_attn_indexer._litetopk_tp_query_shard(
        full_q,
        2048,
        torch.device("cpu"),
        use_fp4_cache=False,
        use_pcp=True,
        pcp_world_size=2,
        compress_ratio=1,
        num_heads=32,
        num_reqs=1,
        dcp_world_size=1,
    )
    assert shard is not None
    lo, hi, out, local_status, all_status = shard
    assert (lo, hi) == (3 * (full_q // 4), full_q)
    assert out.shape == (full_q // 4, 2048)
    assert local_status.shape == (1,)
    assert all_status.shape == (4,)
    assert sparse_attn_indexer._litetopk_tp_compressed_row_offset(lo, 1) == lo


@pytest.mark.parametrize(
    ("pcp_world_size", "seq_len", "should_seed"),
    (
        (1, 163840, True),
        (2, 163840, True),
        (2, 139264, False),
        (4, 139264, True),
        (4, 196608, False),
    ),
)
def test_dense_seed_window_tracks_pcp_scheduler_stride(
    monkeypatch, pcp_world_size, seq_len, should_seed
):
    calls = []
    monkeypatch.setattr(
        litetopk_indexer, "production_extension_available", lambda **_: True
    )
    monkeypatch.setattr(
        litetopk_indexer, "stash_carry", lambda *args, **kwargs: calls.append(args)
    )

    litetopk_indexer.stash_dense_carry(
        torch.empty((1, 2048), dtype=torch.int32),
        seq_len,
        "model.layers.0.indexer",
        pcp_world_size=pcp_world_size,
    )

    assert bool(calls) is should_seed


def _pcp_frontier_chunk(
    *, query_len=8192, extent, common_ke_min, fused=True, skip=False
):
    return SimpleNamespace(
        token_start=0,
        token_end=query_len,
        num_reqs=1,
        fused_indexer_planned=fused,
        skip_kv_gather=skip,
        local_total_seq_lens=extent,
        max_local_total_seq_lens=extent,
        total_seq_lens=extent,
        common_ke_min=common_ke_min,
    )


def test_pcp_frontier_requires_two_stable_fused_chunks():
    chunks = [
        _pcp_frontier_chunk(extent=204800, common_ke_min=196609),
        _pcp_frontier_chunk(extent=262144, common_ke_min=253953),
    ]
    metadata = SimpleNamespace(
        num_decodes=0,
        num_prefills=2,
        prefill=SimpleNamespace(chunks=chunks),
    )
    assert sparse_attn_indexer._litetopk_pcp_frontier_local_extents(metadata) == (
        204800,
        262144,
    )

    chunks[1].fused_indexer_planned = False
    assert sparse_attn_indexer._litetopk_pcp_frontier_local_extents(metadata) is None


def test_pcp_frontier_uses_global_a_and_b_sources():
    descriptors = [
        (1, 204800, 262144),
        (1, 212992, 253952),
        (1, 221184, 245760),
        (1, 229376, 237568),
    ]
    assert sparse_attn_indexer._litetopk_pcp_frontier_sources(descriptors) == (
        (3, 229376),
        (0, 262144),
    )
    descriptors[2] = (0, 0, 0)
    assert sparse_attn_indexer._litetopk_pcp_frontier_sources(descriptors) is None


def test_pcp_frontier_broadcast_is_covered_by_ready_event():
    source = inspect.getsource(litetopk_indexer._pcp_frontier_broadcast_carry)
    broadcast = source.index("pynccl_comm.broadcast")
    assert "stream=side" in source[broadcast:]

    publisher = inspect.getsource(litetopk_indexer._publish_carry)
    assert publisher.index("_pcp_frontier_broadcast_carry") < publisher.index(
        "ready.record(side)"
    )

    stash = inspect.getsource(litetopk_indexer.stash_carry)
    assert stash.index("_pcp_frontier_broadcast_carry") < stash.index("ev.record()")
