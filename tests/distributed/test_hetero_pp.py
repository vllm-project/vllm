# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for heterogeneous per-stage TP sizes in pipeline parallelism."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.distributed.parallel_state import (
    pp_boundary_dst_tp_ranks,
    pp_boundary_src_tp_rank,
)
from vllm.distributed.pp_transport.copy_engine import _header_layout
from vllm.v1.core.kv_cache_planning import (
    get_global_kv_cache_groups,
    get_kv_cache_configs,
    get_pipeline_stages,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig


def test_parallel_config_heterogeneous_pp_normalization_and_mapping():
    cfg = ParallelConfig(
        pipeline_parallel_tp_sizes=[1, 2, 1],
        distributed_executor_backend="mp",
    )
    assert cfg.pipeline_parallel_size == 3
    assert cfg.is_heterogeneous_pp
    assert cfg.world_size == 4
    assert cfg.pp_stage_tp_sizes == (1, 2, 1)
    assert cfg.pp_stage_offsets == (0, 1, 3)
    assert [cfg.get_pp_stage_and_tp_rank(r) for r in range(4)] == [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
    ]
    assert [cfg.get_pp_stage_ranks(s) for s in range(3)] == [[0], [1, 2], [3]]

    cfg_242 = ParallelConfig(
        pipeline_parallel_size=3,
        pipeline_parallel_tp_sizes=[2, 4, 2],
        distributed_executor_backend="mp",
    )
    assert cfg_242.world_size == 8
    assert cfg_242.pp_stage_offsets == (0, 2, 6)
    assert [cfg_242.get_pp_stage_tp_size_for_rank(r) for r in range(8)] == [
        2,
        2,
        4,
        4,
        4,
        4,
        2,
        2,
    ]

    # Uniform list normalizes onto standard uniform PP.
    cfg_uniform = ParallelConfig(
        pipeline_parallel_tp_sizes=[2, 2],
        distributed_executor_backend="mp",
    )
    assert not cfg_uniform.is_heterogeneous_pp
    assert cfg_uniform.pipeline_parallel_size == 2
    assert cfg_uniform.tensor_parallel_size == 2
    assert cfg_uniform.world_size == 4


def test_parallel_config_heterogeneous_pp_validation_errors():
    with pytest.raises(ValueError, match="must not be empty"):
        ParallelConfig(pipeline_parallel_tp_sizes=[])

    with pytest.raises(ValueError, match="positive integers"):
        ParallelConfig(pipeline_parallel_tp_sizes=[1, 0, 1])

    with pytest.raises(ValueError, match="must match pipeline_parallel_size"):
        ParallelConfig(
            pipeline_parallel_size=2,
            pipeline_parallel_tp_sizes=[1, 2, 1],
            distributed_executor_backend="mp",
        )

    with pytest.raises(ValueError, match="Cannot set tensor_parallel_size > 1"):
        ParallelConfig(
            tensor_parallel_size=2,
            pipeline_parallel_tp_sizes=[1, 2, 1],
            distributed_executor_backend="mp",
        )

    with pytest.raises(ValueError, match="data_parallel_size=1"):
        ParallelConfig(
            data_parallel_size=2,
            pipeline_parallel_tp_sizes=[1, 2, 1],
            distributed_executor_backend="mp",
        )

    with pytest.raises(ValueError, match="distributed_executor_backend='mp'"):
        ParallelConfig(
            pipeline_parallel_tp_sizes=[1, 2, 1],
            distributed_executor_backend="uni",
        )


def test_pp_boundary_peer_bijection():
    for src_tp in range(1, 9):
        for dst_tp in range(1, 9):
            fed_receivers: list[int] = []
            for src_r in range(src_tp):
                dsts = pp_boundary_dst_tp_ranks(src_r, src_tp, dst_tp)
                for dst_r in dsts:
                    assert pp_boundary_src_tp_rank(dst_r, src_tp, dst_tp) == src_r
                fed_receivers.extend(dsts)
            assert sorted(fed_receivers) == list(range(dst_tp))


def test_copy_engine_header_layout():
    assert _header_layout(0) == ([8], [64], 256)
    assert _header_layout(1) == ([8], [64], 256)
    for num_dsts in (2, 3, 4, 8):
        empty_offs, eng_offs, data_off = _header_layout(num_dsts)
        assert len(empty_offs) == num_dsts and len(eng_offs) == num_dsts
        # Non-overlapping with full (0..8), stage_empty (16..24), and each other.
        assert empty_offs[0] >= 24
        assert empty_offs[-1] + 8 <= eng_offs[0]
        assert eng_offs[-1] + 192 <= data_off
        assert data_off % 256 == 0


def _make_kv_planning_vllm_config(stage_tp_sizes: list[int]) -> VllmConfig:
    cfg = VllmConfig.__new__(VllmConfig)
    cfg.parallel_config = ParallelConfig(
        pipeline_parallel_tp_sizes=stage_tp_sizes,
        distributed_executor_backend="mp",
    )
    cfg.cache_config = CacheConfig(block_size=16)
    cfg.cache_config.kv_cache_layout = "LBNHC"
    cfg.scheduler_config = SchedulerConfig(
        max_num_batched_tokens=2048,
        max_num_seqs=16,
        max_model_len=2048,
        is_encoder_decoder=False,
    )
    cfg.model_config = SimpleNamespace(
        max_model_len=2048,
        original_max_model_len=2048,
        supports_kv_cache_grouping=False,
        is_hybrid=True,
    )
    cfg.attention_config = SimpleNamespace(hisparse_config=None)
    cfg.speculative_config = None
    return cfg


def _attn_layer(num_kv_heads: int, sw: int | None = None) -> KVCacheLayerConfig:
    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=128,
        dtype=torch.bfloat16,
    )
    if sw is not None:
        return SlidingWindowManager.layer_config(spec, sliding_window=sw)
    return FullAttentionManager.layer_config(spec)


def _inkling_conv_layer(tp_size: int) -> KVCacheLayerConfig:
    # Matches InklingConvState(replicate_hidden=True) for hidden=2304, kv=8.
    num_kv_heads = 8 // tp_size
    hidden_per_head = 2304 // num_kv_heads
    raw_head_size = 2 * 128 + 2 * hidden_per_head
    head_size = 1 << (raw_head_size - 1).bit_length()
    spec = AttentionSpec(
        block_size=4,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    return SlidingWindowManager.layer_config(spec, sliding_window=4)


def test_hetero_pp_kv_cache_planning_hybrid_full_swa():
    vllm_config = _make_kv_planning_vllm_config([1, 2, 1])
    s0 = {
        "l0.attn": _attn_layer(8, sw=1024),
        "l1.attn": _attn_layer(8, sw=None),
    }
    s1 = {
        "l2.attn": _attn_layer(4, sw=1024),
        "l3.attn": _attn_layer(4, sw=None),
    }
    s2 = {
        "l4.attn": _attn_layer(8, sw=1024),
        "l5.attn": _attn_layer(8, sw=None),
    }
    worker_specs = [s0, s1, s1, s2]

    groups = get_global_kv_cache_groups(vllm_config, worker_specs)
    # 2 groups (1 full, 1 SWA), both retaining block_size=16 without splitting!
    assert len(groups) == 2
    assert all(g.block_size == 16 for g in groups)

    kv_configs = get_kv_cache_configs(
        vllm_config, worker_specs, [1024 * 1024 * 1024] * 4, groups
    )
    assert len({cfg.num_blocks for cfg in kv_configs}) == 1
    stages = get_pipeline_stages(vllm_config, kv_configs)
    assert len(stages) == 3
    # Stage 0 & 2 (TP=1): 64 KiB/block; Stage 1 (TP=2): 32 KiB/block.
    assert stages[0].bytes_per_block == 65536
    assert stages[1].bytes_per_block == 32768
    assert stages[2].bytes_per_block == 65536


@pytest.mark.parametrize(
    "stage_tp_sizes,expected_bs", [([1, 2, 1], 32), ([2, 4, 2], 64)]
)
def test_hetero_pp_kv_cache_planning_inkling2_loop(
    stage_tp_sizes: list[int], expected_bs: int
):
    """Verify exact Inkling-2-Mini hybrid specs (Full + SWA + Conv) under
    heterogeneous PP=[1, 2, 1] and [2, 4, 2]."""
    vllm_config = _make_kv_planning_vllm_config(stage_tp_sizes)

    def make_stage(
        prefix: str, num_repeats: int, tp: int
    ) -> dict[str, KVCacheLayerConfig]:
        out: dict[str, KVCacheLayerConfig] = {}
        kv_heads = 8 // tp
        for r in range(num_repeats):
            for i in range(6):
                sw = None if i == 5 else 1024
                out[f"{prefix}.r{r}.l{i}.attn"] = _attn_layer(kv_heads, sw=sw)
                out[f"{prefix}.r{r}.l{i}.conv"] = _inkling_conv_layer(tp)
        return out

    # Prelude: 6 layers (1x), Core: 12 layers (3x = 36), Coda: 6 layers (1x).
    s0 = make_stage("pre", 1, stage_tp_sizes[0])
    s1 = make_stage("core", 6, stage_tp_sizes[1])
    s2 = make_stage("coda", 1, stage_tp_sizes[2])

    worker_specs = (
        [s0] * stage_tp_sizes[0] + [s1] * stage_tp_sizes[1] + [s2] * stage_tp_sizes[2]
    )
    groups = get_global_kv_cache_groups(vllm_config, worker_specs)
    # 1 Full + 5 SWA + 6 Conv = 12 groups, identical to uniform PP!
    assert len(groups) == 12
    full_groups = [g for g in groups if issubclass(g.manager_cls, FullAttentionManager)]
    swa_groups = [
        g
        for g in groups
        if issubclass(g.manager_cls, SlidingWindowManager)
        and g.manager_config.params.sliding_window == 1024
    ]
    conv_groups = [
        g
        for g in groups
        if issubclass(g.manager_cls, SlidingWindowManager)
        and g.manager_config.params.sliding_window == 4
    ]
    assert len(full_groups) == 1 and full_groups[0].block_size == expected_bs
    assert len(swa_groups) == 5 and all(g.block_size == expected_bs for g in swa_groups)
    assert len(conv_groups) == 6 and all(g.block_size == 4 for g in conv_groups)

    kv_configs = get_kv_cache_configs(
        vllm_config,
        worker_specs,
        [2 * 1024 * 1024 * 1024] * sum(stage_tp_sizes),
        groups,
    )
    assert len({cfg.num_blocks for cfg in kv_configs}) == 1
    stages = get_pipeline_stages(vllm_config, kv_configs)
    assert len(stages) == 3
    # Stage 0 & 2: 1 layer/group × 128 KiB = 128 KiB;
    # Stage 1 (core): 6 layers/group × 64 KiB = 384 KiB.
    assert stages[0].bytes_per_block == 131072
    assert stages[1].bytes_per_block == 393216
    assert stages[2].bytes_per_block == 131072
