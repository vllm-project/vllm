# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
    NixlConnectorMetadata,
    PPLayerMap,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NIXL_CONNECTOR_VERSION,
    compute_nixl_compatibility_hash,
)


def _meta(
    pp_rank: int,
    start_layer: int,
    end_layer: int,
    *,
    pp_size: int = 4,
    registered_layer_indices: list[int] | None = None,
    registered_layer_names: list[str] | None = None,
    region_members: list[list[str]] | None = None,
) -> NixlAgentMetadata:
    if registered_layer_indices is None:
        registered_layer_indices = list(range(start_layer, end_layer))
    if registered_layer_names is None:
        registered_layer_names = [
            f"model.layers.{idx}.self_attn" for idx in registered_layer_indices
        ]
    if region_members is None:
        # Default: each advertised region holds exactly its representative
        # layer name (the non-pooled / no-HMA shape).
        region_members = [[name] for name in registered_layer_names]
    return NixlAgentMetadata(
        engine_id="engine",
        agent_metadata=b"agent",
        kv_caches_base_addr=list(range(len(registered_layer_indices))),
        device_id=pp_rank,
        num_blocks=1,
        block_lens=[1024] * len(registered_layer_indices),
        kv_cache_layout="HND",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASH_ATTN",
        physical_blocks_per_logical_kv_block=1,
        pp_rank=pp_rank,
        pp_size=pp_size,
        start_layer=start_layer,
        end_layer=end_layer,
        registered_layer_indices=registered_layer_indices,
        registered_layer_names=registered_layer_names,
        region_members=region_members,
    )


def _metas(boundaries: list[tuple[int, int]]) -> list[NixlAgentMetadata]:
    pp_size = len(boundaries)
    return [
        _meta(rank, start, end, pp_size=pp_size)
        for rank, (start, end) in enumerate(boundaries)
    ]


class _FakeAttentionBackend:
    @staticmethod
    def get_name() -> str:
        return "FAKE_ATTN"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, int, int, int, int]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


def _fake_vllm_config(pipeline_parallel_size: int = 1) -> SimpleNamespace:
    model_config = SimpleNamespace(
        model="fake-model",
        dtype="float16",
        get_total_num_kv_heads=lambda: 8,
        get_head_size=lambda: 16,
        get_total_num_hidden_layers=lambda: 32,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=1,
        ),
    )


def test_nixl_connector_version_is_bumped_to_v6():
    assert NIXL_CONNECTOR_VERSION == 6


def test_pp_layer_map_round_trip_queries():
    layer_map = PPLayerMap.from_metadata_shards(
        _metas([(0, 8), (8, 16), (16, 24), (24, 32)]),
        total_num_hidden_layers=32,
    )

    assert layer_map.pp_size == 4
    assert layer_map.boundaries == ((0, 8), (8, 16), (16, 24), (24, 32))
    assert layer_map.registered_layer_indices == tuple(
        tuple(range(start, end)) for start, end in ((0, 8), (8, 16), (16, 24), (24, 32))
    )
    assert layer_map.total_num_hidden_layers == 32


@pytest.mark.parametrize(
    "boundaries,total",
    [
        ([(0, 8), (10, 16), (16, 24), (24, 32)], 32),
        ([(0, 8), (8, 15), (16, 24), (24, 32)], 32),
        ([(0, 8), (8, 16), (16, 24), (24, 30)], 32),
    ],
)
def test_pp_layer_map_rejects_non_contiguous_or_incomplete_coverage(boundaries, total):
    with pytest.raises(ValueError, match="boundaries must"):
        PPLayerMap.from_metadata_shards(
            _metas(boundaries), total_num_hidden_layers=total
        )


def test_pp_layer_map_collapses_duplicate_tp_records():
    metas = _metas([(0, 8), (8, 16), (16, 24), (24, 32)])
    metas.append(_meta(1, 8, 16, pp_size=4))

    layer_map = PPLayerMap.from_metadata_shards(metas, total_num_hidden_layers=32)

    assert layer_map.registered_layer_indices[1] == tuple(range(8, 16))


def test_pp_layer_map_rejects_conflicting_duplicate_tp_records():
    metas = _metas([(0, 8), (8, 16), (16, 24), (24, 32)])
    metas.append(_meta(1, 8, 16, pp_size=4, registered_layer_indices=[8, 8]))

    with pytest.raises(ValueError, match="conflicting metadata"):
        PPLayerMap.from_metadata_shards(metas, total_num_hidden_layers=32)


def test_compatibility_hash_ignores_pipeline_parallel_size():
    assert compute_nixl_compatibility_hash(
        _fake_vllm_config(pipeline_parallel_size=1), "FLASH_ATTN", False
    ) == compute_nixl_compatibility_hash(
        _fake_vllm_config(pipeline_parallel_size=4), "FLASH_ATTN", False
    )


def test_req_meta_reads_pp_size_and_defaults_to_one():
    metadata = NixlConnectorMetadata()
    params = {
        "remote_block_ids": ([0],),
        "remote_engine_id": "engine",
        "remote_request_id": "remote-req",
        "remote_host": "localhost",
        "remote_port": 1234,
        "tp_size": 2,
        "pp_size": 4,
    }

    metadata.add_new_req_to_recv("req", ([0],), params)
    assert metadata.reqs_to_recv["req"].pp_size == 4

    params.pop("pp_size")
    metadata.add_new_req_to_recv("req-default", ([0],), params)
    assert metadata.reqs_to_recv["req-default"].pp_size == 1
