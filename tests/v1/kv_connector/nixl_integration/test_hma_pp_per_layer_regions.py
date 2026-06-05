# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict

import msgspec

from tests.v1.kv_connector.nixl_integration.test_consumer_shard_refactor import (
    REMOTE_ENGINE_ID,
    _make_worker,
)
from tests.v1.kv_connector.nixl_integration.test_pp_layer_map import _meta
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NIXL_CONNECTOR_VERSION,
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    _make_shard_desc_layout,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


def _attn(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}.attn"


def _swa(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}.attn.swa_cache"


def _compressor(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}.attn.compressor.state_cache"


def _configure_hma_worker(layer_names: list[str], group_ids: list[int]):
    worker = _make_worker(total_layers=128)
    worker.local_seen_layer_names = layer_names
    worker.block_len_per_layer = [1024] * len(layer_names)
    worker.kv_caches_base_addr[worker.engine_id][worker._local_kv_cache_key] = [
        100_000 + i * 10_000 for i in range(len(layer_names))
    ]
    layer_name_to_region_indices: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(layer_names):
        layer_name_to_region_indices[name].append(idx)
    worker._local_layer_name_to_region_indices = layer_name_to_region_indices
    worker._layer_name_to_kv_group_index = dict(zip(layer_names, group_ids))
    worker._is_hma_required = True
    return worker


def test_asymmetric_dsv4_pool_case_resolves_by_layer_name():
    local_layer_names = [
        _attn(15),
        _compressor(13),
        _swa(14),
        _swa(15),
        _attn(16),
        _compressor(16),
        _swa(16),
        _swa(17),
        _attn(18),
        _compressor(18),
    ]
    worker = _configure_hma_worker(local_layer_names, [0] * len(local_layer_names))

    producer_region_names = [
        _compressor(15),
        _swa(15),
        _attn(16),
        _compressor(16),
        _swa(16),
    ]
    # The old pool-subset matcher failed this shape because these names span
    # two decode-side HMA pools. Per-layer regions need only exact names.
    worker.local_seen_layer_names.insert(1, _compressor(15))
    worker.block_len_per_layer.insert(1, 1024)
    worker.kv_caches_base_addr[worker.engine_id][worker._local_kv_cache_key].insert(
        1, 110_000
    )
    worker._layer_name_to_kv_group_index[_compressor(15)] = 0
    # Rebuild the layer-name → region index map so it reflects the insert.
    worker._local_layer_name_to_region_indices = defaultdict(
        list,
        {name: [idx] for idx, name in enumerate(worker.local_seen_layer_names)},
    )

    assert worker._local_region_indices_for_layer_names(producer_region_names) == [
        worker.local_seen_layer_names.index(name) for name in producer_region_names
    ]


def test_pool_member_resolves_to_sharing_region_index():
    # Models the DeepseekV4 + PP failure: the local side pools (e.g.) L14's SWA
    # cache with L16's main attention (HMA shared region), so L14's swa name is
    # dedup'd out of ``local_seen_layer_names`` even though it lives in
    # ``kv_caches``. The producer's PP slice ends right at L14 so it advertises
    # ``model.layers.14.attn.swa_cache`` as a pool representative. The matcher
    # must still route it to the local region that holds L14's SWA data.
    representative_layer_names = [
        _attn(16),  # local representative for the shared (c4a + swa) pool
        _attn(18),  # second shared-pool representative
    ]
    worker = _configure_hma_worker(
        representative_layer_names, [0] * len(representative_layer_names)
    )
    # L14's SWA and L16's main attn share an HMA region; the dedup keeps L16
    # in ``local_seen_layer_names`` but L14's swa is still part of the local
    # kv_caches and must resolve to the same NIXL region.
    worker._local_layer_name_to_region_indices[_swa(14)].append(0)
    worker._local_layer_name_to_region_indices[_swa(16)].append(1)
    worker._layer_name_to_kv_group_index[_swa(14)] = 1
    worker._layer_name_to_kv_group_index[_swa(16)] = 1

    producer_layer_names = [
        _attn(16),
        _swa(14),  # producer's alone-SWA representative
        _attn(18),
        _swa(16),
    ]

    assert worker._local_region_indices_for_layer_names(producer_layer_names) == [
        0,
        0,
        1,
        1,
    ]


def test_descriptor_ids_are_per_layer_and_kv_group_specific():
    layer_names = [_attn(15), _swa(15), _attn(16), _compressor(16)]
    worker = _configure_hma_worker(layer_names, [0, 1, 0, 2])
    worker._xfer_desc_layouts[(REMOTE_ENGINE_ID, 1, "local")] = _make_shard_desc_layout(
        num_blocks=10,
        region_group_ids=(0, 1, 0, 2),
    )

    desc_ids = worker._get_block_descs_ids_for_shard(
        REMOTE_ENGINE_ID,
        1,
        "local",
        ([1, 2], [7], [4, 5]),
    )

    assert desc_ids.tolist() == [1, 2, 17, 21, 22, 34, 35]


def test_expand_remote_members_routes_each_member_to_its_local_region():
    # Member-identity routing keyed by LAYER NAME: every producer member —
    # incl. HMA cross-group pooled swa members — is resolved to the consumer
    # region that physically holds it, independent of the producer's pooling. A
    # producer region pooling attn(16) with swa(14)/swa(15) expands to three
    # transfer units routed to whichever consumer regions hold them.
    worker = _configure_hma_worker([_attn(16), _attn(18)], [0, 0])
    # Consumer layout (swa is kv-group 1): region 0 holds attn(16)+swa(14)+
    # swa(15); region 1 holds attn(18)+swa(16)+swa(17).
    worker._member_to_local_region = {
        _attn(16): 0,
        _swa(14): 0,
        _swa(15): 0,
        _attn(18): 1,
        _swa(16): 1,
        _swa(17): 1,
    }
    worker._layer_name_to_kv_group_index.update(
        {
            _attn(16): 0,
            _attn(18): 0,
            _swa(14): 1,
            _swa(15): 1,
            _swa(16): 1,
            _swa(17): 1,
        }
    )
    meta = _meta(
        0,
        14,
        19,
        pp_size=2,
        registered_layer_indices=[16, 18],
        registered_layer_names=[_attn(16), _attn(18)],
        region_members=[
            [_attn(16), _swa(14), _swa(15)],
            [_attn(18), _swa(16), _swa(17)],
        ],
    )

    member_local_regions, member_groups, member_meta = worker._expand_remote_members(
        meta
    )

    assert member_local_regions == [0, 0, 0, 1, 1, 1]
    assert member_groups == (0, 1, 1, 0, 1, 1)
    # Each producer region's base addr / block len is repeated per member so the
    # region-based builders emit one descriptor group per member.
    r0, r1 = meta.kv_caches_base_addr[0], meta.kv_caches_base_addr[1]
    assert member_meta.kv_caches_base_addr == [r0, r0, r0, r1, r1, r1]
    assert len(member_meta.block_lens) == 6


def test_expand_remote_members_raises_on_unknown_member():
    worker = _configure_hma_worker([_attn(16)], [0])
    worker._member_to_local_region = {_attn(16): 0}
    meta = _meta(
        0,
        16,
        17,
        pp_size=1,
        registered_layer_indices=[16],
        registered_layer_names=[_attn(16)],
        region_members=[[_attn(16), _swa(14)]],  # swa(14) has no local region
    )

    try:
        worker._expand_remote_members(meta)
    except RuntimeError as exc:
        assert "no matching local region" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for unmapped member")


def test_distinct_layer_names_in_same_kv_group_route_to_distinct_regions():
    # Regression: an MLA layer's main latent and its indexer/compressor cache
    # can both land in kv-group 0 (UniformTypeKVCacheSpecs merges their specs),
    # so a (layer_index, kv_group_index) member key is non-unique across
    # regions. Member identity must key on the LAYER NAME so the two distinct
    # caches route to their own consumer regions instead of collapsing onto one
    # — collapsing double-writes one region and leaves the other's slots stale,
    # which corrupts long-context KV under PP+HMA disaggregation.
    worker = _configure_hma_worker([_attn(2), _compressor(2)], [0, 0])
    worker._member_to_local_region = {_attn(2): 0, _compressor(2): 1}
    meta = _meta(
        0,
        2,
        3,
        pp_size=1,
        registered_layer_indices=[2, 2],
        registered_layer_names=[_attn(2), _compressor(2)],
        region_members=[[_attn(2)], [_compressor(2)]],
    )

    member_local_regions, member_groups, _ = worker._expand_remote_members(meta)

    # Distinct regions (0, 1) — NOT collapsed to [0, 0] as a (layer, group) key
    # would. Both belong to kv-group 0.
    assert member_local_regions == [0, 1]
    assert member_groups == (0, 0)


def test_mamba_descriptor_ids_use_mamba_suffix_and_group_filter():
    layer_names = [_attn(15), _compressor(16)]
    worker = _configure_hma_worker(layer_names, [0, 1])
    worker._has_mamba = True
    worker._group_spec_types = (FullAttentionSpec, MambaSpec)
    worker._xfer_desc_layouts[(REMOTE_ENGINE_ID, 1, "local")] = _make_shard_desc_layout(
        num_blocks=10,
        region_group_ids=(0, 1),
        physical_blocks_per_logical=2,
        mamba_region_count=8,
        mamba_region_group_ids=(0, 0, 0, 0, 1, 1, 1, 1),
    )

    desc_ids = worker._get_block_descs_ids_for_shard(
        REMOTE_ENGINE_ID,
        1,
        "local",
        ([1, 2], [3]),
    )

    assert desc_ids.tolist() == [1, 2, 43, 48, 53, 58]


def test_repeated_layer_name_uses_matching_occurrence_for_split_regions():
    layer_name = "model.layers.3.self_attn"
    worker = _configure_hma_worker([layer_name, layer_name], [0, 0])

    assert worker._local_region_indices_for_layer_names([layer_name, layer_name]) == [
        0,
        1,
    ]


def test_nixl_agent_metadata_v6_registered_layer_names_round_trip():
    meta = _meta(
        0,
        0,
        4,
        pp_size=1,
        registered_layer_indices=[0, 2],
        registered_layer_names=[
            "model.layers.0.self_attn",
            "model.layers.2.indexer",
        ],
    )

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(meta), type=NixlAgentMetadata
    )

    assert NIXL_CONNECTOR_VERSION == 6
    assert decoded.registered_layer_names == [
        "model.layers.0.self_attn",
        "model.layers.2.indexer",
    ]


def test_nixl_agent_metadata_v6_region_members_round_trip():
    # region_members advertises, per registered NIXL region, every layer name
    # sharing that region — including HMA cross-group pooled members (e.g. an
    # swa_cache pooled onto a later layer's main-attn region) that are dedup'd
    # out of registered_layer_names.
    meta = _meta(
        0,
        0,
        4,
        pp_size=1,
        registered_layer_indices=[2, 4],
        registered_layer_names=[_attn(2), _attn(4)],
        region_members=[
            [_attn(2), _swa(0), _swa(1)],  # region holds L2 attn + L0/L1 swa
            [_attn(4), _swa(2), _swa(3)],  # region holds L4 attn + L2/L3 swa
        ],
    )

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(meta), type=NixlAgentMetadata
    )

    assert decoded.region_members == [
        [_attn(2), _swa(0), _swa(1)],
        [_attn(4), _swa(2), _swa(3)],
    ]


def test_shard_local_handler_uses_registered_layer_names():
    layer_names = [_attn(15), _swa(15), _attn(16)]
    worker = _configure_hma_worker(layer_names, [0, 1, 0])
    worker.nixl_wrapper._next_handle = 0

    _, blocks_data = worker.register_local_xfer_handler(
        worker.block_size,
        registered_layer_names=[_swa(15), _attn(16)],
    )

    assert worker._region_group_ids_for_layer_names([_swa(15), _attn(16)]) == (
        1,
        1,
        0,
        0,
    )
    assert [blocks_data[i][0] for i in (0, 8)] == [110_000, 120_000]
