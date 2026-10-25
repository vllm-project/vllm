# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FileMapper."""

from fractions import Fraction
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
                tokens_per_block=g[0],
                layer_names=(g[1],),
                group_id=group_id,
                tokens_per_state=g[2] if len(g) > 2 else (1,),
            )
            for group_id, g in enumerate(kwargs.get("groups", ()))
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
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=kwargs.get("is_parallelism_agnostic", False),
        ),
        replicated_layout=kwargs.get("replicated_layout", False),
        canonical_layout=kwargs.get("canonical_layout", False),
        kv_cache_layout=kwargs.get("kv_cache_layout", "LBNHC"),
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
    """Path must match: <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin.

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
        "/tmp/cache/test-model_de3bba26cf36_r3/000/10_g2/0001020304050607.bin"
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
                "tokens_per_state": ["1"],
            }
        ],
        "inference_engine": "vllm",
        "parallel_agnostic": False,
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
        {
            "tokens_per_block": 12,
            "layer_names": ["full_layer"],
            "tokens_per_state": ["1"],
        },
        {
            "tokens_per_block": 16,
            "layer_names": ["mla_layer"],
            "tokens_per_state": ["1"],
        },
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
    assert "parallel_agnostic" not in fm.fields


def test_parallel_agnostic_ignored_when_config_disallows():
    fm = make_mapper_from_offloading_spec(
        tp_size=2,
        rank=1,
        is_parallelism_agnostic=False,
        parallel_agnostic=True,
    )
    assert fm.fields["tp_size"] == 2
    assert fm.rank == 1
    assert fm.fields["parallel_agnostic"] is False


def test_namespace_kept_without_parallel_agnostic_opt_in():
    fm = make_mapper_from_offloading_spec(
        tp_size=2,
        rank=1,
        is_parallelism_agnostic=True,
        parallel_agnostic=False,
    )
    assert fm.fields["tp_size"] == 2
    assert fm.rank == 1
    assert fm.fields["parallel_agnostic"] is False


def test_parallel_agnostic_separates_persistent_layouts():
    agnostic = make_mapper_from_offloading_spec(
        is_parallelism_agnostic=True,
        parallel_agnostic=True,
    )
    specific = make_mapper_from_offloading_spec(
        is_parallelism_agnostic=False,
        parallel_agnostic=True,
    )

    assert agnostic.base_path != specific.base_path
    assert "parallel_agnostic" not in agnostic.fields
    assert specific.fields["parallel_agnostic"] is False


def test_canonical_layout_changes_storage_namespace():
    # Canonical bytes are not interchangeable with the direct layout, so the
    # format id must fork the storage namespace.
    direct = make_mapper_from_offloading_spec()
    canonical = make_mapper_from_offloading_spec(canonical_layout=True)
    assert "canonical_format" not in direct.fields
    assert canonical.fields["canonical_format"] == "v1-nhd"
    assert direct.base_path != canonical.base_path


# ---------------------------------------------------------------------------
# replicated_layout: OR'd into parallel-agnostic identity for compact rows
# ---------------------------------------------------------------------------


def test_replicated_layout_collapses_parallel_identity():
    shared = dict(
        model_name="mla-model",
        groups=((16, "mla_layer"),),
        replicated_layout=True,
        parallel_agnostic=True,
    )
    tp2 = make_mapper_from_offloading_spec(tp_size=2, world_size=2, rank=1, **shared)
    tp4 = make_mapper_from_offloading_spec(tp_size=4, world_size=4, rank=3, **shared)

    assert tp2.base_path == tp4.base_path
    for fm in (tp2, tp4):
        assert fm.fields["tp_size"] == 1
        assert fm.fields["pp_size"] == 1
        assert fm.fields["pcp_size"] == 1
        assert fm.fields["dcp_size"] == 1
        assert fm.rank == 0
        assert "parallel_agnostic" not in fm.fields
        assert fm.fields["replicated_layout"] is True
        assert fm.get_file_name(make_offload_key(b"\x01" * 8, 0)).startswith(
            f"{fm.base_path}_r0/"
        )


def test_replicated_layout_requires_caller_opt_in():
    fm = make_mapper_from_offloading_spec(
        tp_size=2,
        world_size=2,
        rank=1,
        replicated_layout=True,
        parallel_agnostic=False,
    )
    assert fm.fields["tp_size"] == 2
    assert fm.rank == 1
    assert fm.fields["parallel_agnostic"] is False
    assert "replicated_layout" not in fm.fields
    baseline = make_mapper_from_offloading_spec(
        tp_size=2,
        world_size=2,
        rank=1,
        replicated_layout=False,
        parallel_agnostic=False,
    )
    assert fm.base_path == baseline.base_path


def test_non_replicated_keeps_parallel_identity():
    fm = make_mapper_from_offloading_spec(
        tp_size=4,
        world_size=4,
        rank=2,
        replicated_layout=False,
        is_parallelism_agnostic=False,
        parallel_agnostic=True,
    )
    assert fm.fields["tp_size"] == 4
    assert fm.rank == 2
    assert fm.fields["parallel_agnostic"] is False
    assert fm.get_file_name(make_offload_key(b"\x02" * 8, 0)).startswith(
        f"{fm.base_path}_r2/"
    )


def test_replicated_and_parallelism_agnostic_separate_layouts():
    shared = dict(
        model_name="shared-model",
        groups=((16, "layer0"),),
        tp_size=2,
        world_size=2,
        rank=1,
        parallel_agnostic=True,
    )
    via_agnostic = make_mapper_from_offloading_spec(
        is_parallelism_agnostic=True,
        replicated_layout=False,
        **shared,
    )
    via_replicated = make_mapper_from_offloading_spec(
        is_parallelism_agnostic=False,
        replicated_layout=True,
        **shared,
    )
    assert via_agnostic.base_path != via_replicated.base_path
    assert "replicated_layout" not in via_agnostic.fields
    assert via_replicated.fields["replicated_layout"] is True


def test_replicated_layout_run_config_tp_invariant():
    shared = dict(
        model_name="mla-model",
        groups=((16, "mla_layer"),),
        replicated_layout=True,
        parallel_agnostic=True,
    )
    tp2 = make_mapper_from_offloading_spec(tp_size=2, world_size=2, rank=0, **shared)
    tp4 = make_mapper_from_offloading_spec(tp_size=4, world_size=4, rank=2, **shared)
    assert tp2.get_run_config() == tp4.get_run_config()


def test_tokens_per_state_default_serializes_as_one():
    fm = make_mapper_from_offloading_spec(groups=((16, "layer0"),))
    assert fm.fields["kv_cache_groups"] == [
        {
            "tokens_per_block": 16,
            "tokens_per_state": ["1"],
            "layer_names": ["layer0"],
        }
    ]


def test_tokens_per_state_changes_storage_namespace():
    # Compressed vs. uncompressed MLA must never collide on disk.
    uncompressed = make_mapper_from_offloading_spec(
        model_name="mla-model", groups=((16, "mla_layer", (1,)),)
    )
    compressed = make_mapper_from_offloading_spec(
        model_name="mla-model", groups=((16, "mla_layer", (2,)),)
    )
    assert uncompressed.fields["kv_cache_groups"][0]["tokens_per_state"] == ["1"]
    assert compressed.fields["kv_cache_groups"][0]["tokens_per_state"] == ["2"]
    assert uncompressed.base_path != compressed.base_path


def test_different_compression_ratios_fork_namespace():
    # Two different compress ratios for the same layer must also fork,
    # not just compressed-vs-uncompressed.
    ratio2 = make_mapper_from_offloading_spec(groups=((16, "mla_layer", (2,)),))
    ratio4 = make_mapper_from_offloading_spec(groups=((16, "mla_layer", (4,)),))
    assert ratio2.base_path != ratio4.base_path


def test_fraction_tokens_per_state_serializes_without_error():
    # Whisper-style pooling uses Fraction, which json.dumps cannot handle
    # directly; it must be stringified before hashing.
    fm = make_mapper_from_offloading_spec(
        groups=((16, "whisper_layer", (Fraction(1, 2),)),)
    )
    assert fm.fields["kv_cache_groups"] == [
        {
            "tokens_per_block": 16,
            "tokens_per_state": ["1/2"],
            "layer_names": ["whisper_layer"],
        }
    ]


def test_hybrid_groups_track_independent_tokens_per_state():
    # Extends test_hybrid_file_identity_uses_resolved_tokens_per_hash:
    # heterogeneous groups must each carry their own tokens_per_state.
    fm = make_mapper_from_offloading_spec(
        groups=((12, "full_layer", (1,)), (16, "mla_layer", (2,))),
        tokens_per_hash=4,
    )
    assert fm.fields["kv_cache_groups"] == [
        {
            "tokens_per_block": 12,
            "tokens_per_state": ["1"],
            "layer_names": ["full_layer"],
        },
        {
            "tokens_per_block": 16,
            "tokens_per_state": ["2"],
            "layer_names": ["mla_layer"],
        },
    ]


def test_mixed_tokens_per_state_within_single_group_serializes_all_values():
    # DeepSeek-V4-style case: one UniformTypeKVCacheSpecs group wraps layers
    # with different compress_ratio, so tokens_per_state is a multi-value
    # tuple within a single group entry.
    config = OffloadingConfig(
        groups=(
            OffloadingGroupConfig(
                tokens_per_block=16,
                layer_names=("mla_ratio2", "mla_ratio4"),
                group_id=0,
                tokens_per_state=(2, 4),
            ),
        ),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=False,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float16"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
    )
    spec = MagicMock(spec=OffloadingSpec)
    spec.config = config
    fm = FileMapper.from_offloading_spec(
        root_dir="/tmp/cache",
        offloading_spec=spec,
        blocks_per_file=1,
    )
    assert fm.fields["kv_cache_groups"] == [
        {
            "tokens_per_block": 16,
            "tokens_per_state": ["2", "4"],
            "layer_names": ["mla_ratio2", "mla_ratio4"],
        }
    ]


def test_mixed_tokens_per_state_order_changes_namespace():
    # Order within the tuple must matter too (positional correspondence
    # with layer_names), since swapping which layer has which ratio
    # is a materially different on-disk byte layout.
    def _fm(order):
        config_groups = (
            OffloadingGroupConfig(
                tokens_per_block=16,
                layer_names=("layer_a", "layer_b"),
                group_id=0,
                tokens_per_state=order,
            ),
        )
        config = OffloadingConfig(
            groups=config_groups,
            worker_kv_bytes_per_block=0,
            enable_kv_cache_events=False,
            extra_config={},
            engine_id="test-engine",
            model=OffloadingModelConfig(name="test-model", dtype="float16"),
            cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
            parallel=OffloadingParallelConfig(
                rank=0,
                world_size=1,
                tp_size=1,
                pp_size=1,
                pcp_size=1,
                dcp_size=1,
                data_parallel_index=0,
                data_parallel_size=1,
                data_parallel_rank_local=None,
                is_parallelism_agnostic=False,
            ),
        )
        spec = MagicMock(spec=OffloadingSpec)
        spec.config = config
        return FileMapper.from_offloading_spec(
            root_dir="/tmp/cache", offloading_spec=spec, blocks_per_file=1
        )

    assert _fm((2, 4)).base_path != _fm((4, 2)).base_path
