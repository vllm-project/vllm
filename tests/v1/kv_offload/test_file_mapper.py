# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FileMapper."""

from unittest.mock import MagicMock

from vllm.v1.kv_offload.base import OffloadingSpec, make_offload_key
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.file_mapper import FileMapper

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_mapper_from_offloading_spec(**kwargs) -> FileMapper:
    """Build a FileMapper from a mocked spec carrying a hand-built config."""
    config = OffloadingConfig(
        groups=tuple(
            OffloadingGroupConfig(
                tokens_per_block=tokens_per_block,
                layer_names=(layer_name,),
            )
            for tokens_per_block, layer_name in kwargs.get("groups", ())
        ),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=False,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(
            name=kwargs.get("model_name", "test-model"),
            dtype=kwargs.get("dtype", "float16"),
        ),
        cache=OffloadingCacheConfig(
            tokens_per_hash=kwargs.get("tokens_per_hash", 16),
            blocks_per_chunk=kwargs.get("blocks_per_chunk", 1),
        ),
        parallel=OffloadingParallelConfig(
            rank=kwargs.get("rank", 0),
            world_size=kwargs.get("world_size", 1),
            tp_size=kwargs.get("tp_size", 1),
            pp_size=kwargs.get("pp_size", 1),
            pcp_size=kwargs.get("pcp_size", 1),
            dcp_size=kwargs.get("dcp_size", 1),
            data_parallel_index=0,
            is_parallelism_agnostic=kwargs.get("is_parallelism_agnostic", False),
        ),
    )
    spec = MagicMock(spec=OffloadingSpec)
    spec.config = config
    return FileMapper.from_offloading_spec(
        root_dir=kwargs.get("root_dir", "/tmp/cache"),
        offloading_spec=spec,
        blocks_per_file=config.cache.blocks_per_chunk,
        parallel_agnostic=kwargs.get("parallel_agnostic", False),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_file_name_full_structure():
    """
    Path must match: <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin

    Concretely:
      - The segment immediately after base_path must end with `_r3`
      - The next segment is the first 3 hex chars of the block hash
      - The next segment is <2 hex chars>_g<group_idx>
      - The final segment is <full hash hex>.bin
    """
    rank = 3
    group_idx = 2
    block_hash = bytes(range(8))  # deterministic, non-zero bytes
    fm = make_mapper_from_offloading_spec(rank=rank)
    key = make_offload_key(block_hash, group_idx)
    path = fm.get_file_name(key)

    expected_path = (
        "/tmp/cache/test-model_42b94bdc9933_r3/000/10_g2/0001020304050607.bin"
    )
    assert path == expected_path


def test_get_run_config_fields():
    fm = make_mapper_from_offloading_spec(
        model_name="my-model",
        dtype="bfloat16",
        tp_size=4,
        pp_size=3,
        pcp_size=2,
        dcp_size=2,
        groups=((64, "layer0"),),
        tokens_per_hash=64,
        blocks_per_chunk=3,
    )
    cfg = fm.get_run_config()
    assert cfg == {
        "model_name": "my-model",
        "tokens_per_hash": 64,
        "blocks_per_file": 3,
        "tp_size": 4,
        "pp_size": 3,
        "pcp_size": 2,
        "dcp_size": 2,
        "dtype": "bfloat16",
        "kv_cache_groups": [
            {
                "tokens_per_block": 64,
                "layer_names": ["layer0"],
            }
        ],
        "inference_engine": "vllm",
    }


def test_get_config_file_path():
    fm = make_mapper_from_offloading_spec()
    config_path = fm.get_config_file_path()
    assert config_path == f"{fm.base_path}/config.json"


def test_hybrid_file_identity_uses_resolved_tokens_per_hash():
    # For heterogeneous groups the namespace records the resolved hash
    # granularity (GCD of the group block sizes), which is the actual
    # granularity of the offload block hashes.
    fm = make_mapper_from_offloading_spec(
        groups=((12, "full_layer"), (16, "mla_layer")),
        tokens_per_hash=4,
    )
    assert fm.fields["tokens_per_hash"] == 4
    assert fm.fields["kv_cache_groups"] == [
        {"tokens_per_block": 12, "layer_names": ["full_layer"]},
        {"tokens_per_block": 16, "layer_names": ["mla_layer"]},
    ]


# ---------------------------------------------------------------------------
# parallel_agnostic: opt-in honored only when the config marks the layout
# parallelism-agnostic (predicate computation is covered in test_factory.py)
# ---------------------------------------------------------------------------


def test_parallel_agnostic_collapses_namespace_when_config_allows():
    fm = make_mapper_from_offloading_spec(
        tp_size=4,
        pp_size=3,
        pcp_size=2,
        dcp_size=2,
        rank=1,
        is_parallelism_agnostic=True,
        parallel_agnostic=True,
    )
    assert fm.fields["tp_size"] == 1
    assert fm.fields["pp_size"] == 1
    assert fm.fields["pcp_size"] == 1
    assert fm.fields["dcp_size"] == 1
    assert fm.rank == 0


def test_parallel_agnostic_ignored_when_config_disallows():
    fm = make_mapper_from_offloading_spec(
        tp_size=2,
        rank=1,
        is_parallelism_agnostic=False,
        parallel_agnostic=True,
    )
    assert fm.fields["tp_size"] == 2
    assert fm.rank == 1


def test_namespace_kept_without_parallel_agnostic_opt_in():
    fm = make_mapper_from_offloading_spec(
        tp_size=2,
        rank=1,
        is_parallelism_agnostic=True,
        parallel_agnostic=False,
    )
    assert fm.fields["tp_size"] == 2
    assert fm.rank == 1
