# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV aliasing round trip for packed DeepSeek-V4.1 groupings.

Groups the layers with the planner, binds every layer's real cache view with
``allocate_kv_cache``, and gives each group its own blocks, as the shared block
pool does. A distinct pattern per layer, block and byte is written through each
view; reading every layer back must return exactly its own pattern. A wrong
stride, offset or alias between layers would overwrite another layer's bytes.

CPU only.
"""

from math import lcm
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_planning import (
    _get_kv_cache_bytes_per_block,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    KVCacheLayout,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.worker.utils import allocate_kv_cache

pytestmark = pytest.mark.cpu_test

BLOCKS_PER_GROUP = 3
NUM_BLOCKS = 64


def _config(layout: str):
    config = MagicMock()
    config.cache_config = CacheConfig()
    config.cache_config.num_gpu_blocks_override = None
    config.cache_config.kv_cache_layout = layout
    config.cache_config.block_size = 16
    config.cache_config.enable_prefix_caching = False
    config.cache_config.prefix_match_unit = None
    config.cache_config.mamba_cache_mode = "none"
    config.attention_config.hisparse_config = None
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.speculative_config = None
    config.model_config.max_model_len = 65536
    config.model_config.get_total_num_hidden_layers.return_value = 64
    config.model_config.get_total_num_kv_heads.return_value = 2
    config.model_config.get_num_kv_heads.return_value = 2
    config.max_in_flight_tokens = 32768
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    return config


def _dsv41_fp8_specs() -> dict[str, KVCacheSpec]:
    """Stock fp8 records: 584 B main and SWA states, 132 B indexer."""
    specs: dict[str, KVCacheSpec] = {}
    for layer, ratio in ((2, 2), (8, 2), (14, 2), (20, 1)):
        specs[f"layers.{layer}.attn"] = MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=584,
            dtype=torch.uint8,
            tokens_per_state=ratio,
            alignment=576,
        )
        specs[f"layers.{layer}.idx"] = MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.uint8,
            tokens_per_state=ratio,
            alignment=576,
        )
    for layer in range(43):
        specs[f"layers.{layer}.swa"] = SlidingWindowMLASpec(
            block_size=64,
            num_kv_heads=1,
            head_size=584,
            dtype=torch.uint8,
            sliding_window=128,
            alignment=576,
        )
    return specs


def _dsv41_mxfp4_specs() -> dict[str, KVCacheSpec]:
    """272 B MXFP4 compressed records beside 528 B SWA records: the planner
    widens the block past the MLA group to fit 4 SWA pages per group."""
    record_rows = lcm(528, 272)
    specs: dict[str, KVCacheSpec] = {}
    for layer in range(40):
        specs[f"layers.{layer}.swa"] = SlidingWindowMLASpec(
            block_size=32,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            sliding_window=128,
            state_content_bytes=528,
            alignment=512,
            block_stride_alignment=record_rows,
        )
    for layer, ratio in ((2, 2), (8, 2), (14, 2), (20, 1)):
        specs[f"layers.{layer}.attn"] = MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            tokens_per_state=ratio,
            state_content_bytes=272,
            alignment=512,
        )
        specs[f"layers.{layer}.idx"] = MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=68,
            dtype=torch.uint8,
            tokens_per_state=ratio,
            alignment=512,
        )
    return specs


def _pattern(view_block: torch.Tensor, layer_index: int, block_id: int) -> torch.Tensor:
    n = view_block.numel()
    seed = layer_index * 7919 + block_id * 104729
    values = (torch.arange(n, dtype=torch.int64) * 131 + seed) % 251
    return values.to(torch.uint8).view(view_block.shape)


def _bind(specs: dict[str, KVCacheSpec], layout: str):
    config = _config(layout)
    groups = get_kv_cache_groups(config, specs)
    bytes_per_block = _get_kv_cache_bytes_per_block(groups, KVCacheLayout[layout])
    kv_config = get_kv_cache_config_from_groups(
        config, groups, NUM_BLOCKS * bytes_per_block
    )
    views = allocate_kv_cache(
        kv_config, torch.device("cpu"), KVCacheLayout[layout], None
    )
    # One sequence covering several blocks in every group, with each group
    # owning distinct, non-contiguous block ids from the shared pool (block 0
    # is the null block).
    generator = torch.Generator().manual_seed(0)
    ids = (torch.randperm(kv_config.num_blocks - 1, generator=generator) + 1).tolist()
    assert len(groups) * BLOCKS_PER_GROUP <= len(ids)
    blocks = {
        gid: ids[gid * BLOCKS_PER_GROUP : (gid + 1) * BLOCKS_PER_GROUP]
        for gid in range(len(groups))
    }
    return groups, views, blocks


def _round_trip(groups, views, blocks) -> list[str]:
    order = {name: i for i, name in enumerate(views)}
    for gid, group in enumerate(groups):
        for name in group.layer_names:
            for b in blocks[gid]:
                views[name][b].copy_(_pattern(views[name][b], order[name], b))
    mismatches = []
    for gid, group in enumerate(groups):
        for name in group.layer_names:
            for b in blocks[gid]:
                if not torch.equal(
                    views[name][b], _pattern(views[name][b], order[name], b)
                ):
                    mismatches.append(f"{name} block {b}")
    return mismatches


@pytest.mark.parametrize("layout", ["BLHNC", "BLNHC"])
@pytest.mark.parametrize("make_specs", [_dsv41_fp8_specs, _dsv41_mxfp4_specs])
def test_every_layer_reads_back_only_its_own_pattern(make_specs, layout):
    specs = make_specs()
    groups, views, blocks = _bind(specs, layout)
    assert sorted(views) == sorted(specs)
    assert sorted(n for g in groups for n in g.layer_names) == sorted(specs)
    assert _round_trip(groups, views, blocks) == []


def test_round_trip_catches_an_aliased_layer():
    """Negative control: bind one SWA layer onto another's bytes."""
    groups, views, blocks = _bind(_dsv41_mxfp4_specs(), "BLHNC")
    group = next(g for g in groups if len(g.layer_names) > 1)
    victim, other = group.layer_names[:2]
    views[victim] = views[other]
    assert _round_trip(groups, views, blocks)
