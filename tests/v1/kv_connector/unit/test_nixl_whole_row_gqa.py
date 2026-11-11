# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL descriptor geometry for GQA caches under a block-outermost layout.

With `VLLM_KV_CACHE_LAYOUT=BLHNC` every layer's page of a block sits in one
contiguous row. Opting in with `kv_connector_extra_config.whole_row_gqa`
registers that row as one region, so a block moves as one descriptor instead
of one per layer; peers that register rows differently are rejected.
"""

import re
from unittest.mock import patch

import msgspec
import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker as bw
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    create_kv_cache_views,
)

from .test_nixl_desc_geometry import _RecordingNixl
from .utils import create_vllm_config, expand_strided_descs

BLOCK_SIZE = 16
NUM_BLOCKS = 4


# Only the external NIXL agent is replaced (by a recorder of the descriptor
# lists it is given); the workers, their config, the single-rank parallel
# state (`dist_init`) and the attention backend are real.
def record_nixl(test):
    return patch.object(bw, "NixlWrapper", _RecordingNixl)(test)


def _make_gqa_worker(
    layout: KVCacheLayout,
    *,
    whole_row: bool,
    num_layers: int = 3,
    num_kv_heads: int = 2,
    head_size: int = 64,
    worker_cls=NixlPullConnectorWorker,
    engine_id: str = "local",
):
    """Register real views of one GQA KV tensor on a real worker."""
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
    )
    page = spec.page_size_bytes
    names = [f"model.layers.{i}.self_attn.attn" for i in range(num_layers)]
    if layout.is_block_outermost:
        layer_stride, block_stride = page, num_layers * page
    else:
        layer_stride, block_stride = NUM_BLOCKS * page, page
    raw = torch.zeros(num_layers * NUM_BLOCKS * page, dtype=torch.int8)
    tensor = KVCacheTensor(
        size=raw.nbytes,
        layers=names,
        layer_stride=layer_stride,
        block_stride=block_stride,
    )
    caches = dict(
        zip(names, create_kv_cache_views(raw, spec, NUM_BLOCKS, layout, tensor))
    )
    config = create_vllm_config(
        block_size=BLOCK_SIZE,
        kv_connector_extra_config={"whole_row_gqa": True} if whole_row else None,
    )
    config.cache_config.kv_cache_layout = layout.name
    kv_cache_config = KVCacheConfig(
        NUM_BLOCKS, [tensor], [KVCacheGroupSpec(names, spec)]
    )
    with set_current_vllm_config(config):
        worker = worker_cls(config, engine_id, kv_cache_config)
        worker.register_kv_caches(caches)
    return worker, raw, caches, page


def _metadata(worker, engine_id="remote") -> NixlAgentMetadata:
    meta = msgspec.msgpack.decode(
        worker.xfer_handshake_metadata.agent_metadata_bytes, type=NixlAgentMetadata
    )
    meta.engine_id = engine_id
    return meta


@record_nixl
@pytest.mark.parametrize(
    "worker_cls", [NixlPullConnectorWorker, NixlPushConnectorWorker]
)
def test_whole_row_gqa_transfers_one_descriptor_per_block(worker_cls, dist_init):
    # Fails without the opt-in: BLHNC would still register one region per
    # layer, so a block moves as num_layers descriptors.
    worker, raw, caches, page = _make_gqa_worker(
        KVCacheLayout.BLHNC, whole_row=True, worker_cls=worker_cls
    )
    row = len(caches) * page
    assert worker.num_regions == 1
    assert worker.num_descs == NUM_BLOCKS
    assert worker.src_blocks_data.tolist() == [
        [raw.data_ptr(), row, 0, row, NUM_BLOCKS]
    ]
    assert worker.region_members == [list(caches)]
    assert _metadata(worker).packed_member_layouts == {
        name: (i * page, page) for i, name in enumerate(caches)
    }


@record_nixl
def test_whole_row_gqa_descriptor_covers_exactly_each_blocks_pages(dist_init):
    worker, raw, caches, page = _make_gqa_worker(KVCacheLayout.BLHNC, whole_row=True)
    rows = expand_strided_descs(worker.src_blocks_data)
    for b, (addr, length, _dev) in enumerate(rows.tolist()):
        pages = sorted(
            (view[b].data_ptr(), view[b].data_ptr() + page) for view in caches.values()
        )
        # The pages tile the row with no gap, overlap or padding.
        assert pages[0][0] == addr and pages[-1][1] == addr + length
        assert all(a[1] == b_[0] for a, b_ in zip(pages, pages[1:]))


@record_nixl
@pytest.mark.parametrize(
    "layout, whole_row",
    [(KVCacheLayout.BLHNC, False), (KVCacheLayout.LBHNC, True)],
)
def test_whole_row_gqa_needs_opt_in_and_a_block_outermost_layout(
    layout, whole_row, dist_init
):
    worker, _, caches, _ = _make_gqa_worker(layout, whole_row=whole_row)
    assert worker.num_regions == len(caches)
    assert worker.num_descs == len(caches) * NUM_BLOCKS
    assert _metadata(worker).packed_member_layouts == {}


@record_nixl
def test_per_layer_peer_addresses_pages_inside_whole_gqa_rows(dist_init):
    # In pull mode, a BLHNC decoder without the opt-in (one region per layer)
    # views the prefiller's rows as per-layer pages.
    producer, raw, caches, page = _make_gqa_worker(KVCacheLayout.BLHNC, whole_row=True)
    meta = _metadata(producer)
    row = len(caches) * page
    # The prefiller advertises one whole-row region ...
    assert meta.block_lens == [row]
    assert meta.region_members == [list(caches)]
    # ... which a per-layer peer views as one region per layer.
    bw.NixlBaseConnectorWorker._expand_remote_packed_rows(meta)
    assert meta.kv_caches_base_addr == [
        raw.data_ptr() + i * page for i in range(len(caches))
    ]
    assert meta.block_lens == [page] * len(caches)
    assert meta.block_strides == [row] * len(caches)
    assert meta.region_members == [[name] for name in caches]


@record_nixl
def test_per_layer_decoder_reads_layer_pages_from_a_whole_row_prefiller(dist_init):
    # Pull mode: a BLHNC decoder without the opt-in keeps one region per
    # layer and reads each layer's page from inside the prefiller's rows.
    producer, p_raw, caches, page = _make_gqa_worker(
        KVCacheLayout.BLHNC, whole_row=True
    )
    consumer, d_raw, _, _ = _make_gqa_worker(KVCacheLayout.BLHNC, whole_row=False)
    num_layers = len(caches)
    row = num_layers * page
    meta = _metadata(producer)
    # The prefiller advertises whole rows; main registers per-layer regions.
    assert meta.block_lens == [row]
    assert meta.packed_member_layouts == {
        name: (i * page, page) for i, name in enumerate(caches)
    }
    consumer.add_remote_agent(meta)
    assert consumer.num_regions == num_layers
    handle = consumer.dst_xfer_side_handles["remote"][0]
    remote_descs = consumer.nixl_wrapper.dlists[handle]
    assert sorted(map(tuple, remote_descs.tolist())) == sorted(
        (p_raw.data_ptr() + b * row + i * page, page, 0)
        for b in range(NUM_BLOCKS)
        for i in range(num_layers)
    )
    # Block 1 locally is read from block 2 remotely, layer by layer.
    pairs = consumer._dst_pairs["remote"]
    local_ids = consumer._compute_desc_ids(
        [[1]],
        consumer.num_blocks,
        None,
        1,
        region_num_blocks=consumer.dst_region_num_blocks[consumer.engine_id],
        pairs=(pairs.local, pairs.group),
    )
    remote_ids = consumer._compute_desc_ids(
        [[2]],
        meta.num_blocks,
        None,
        1,
        region_num_blocks=consumer.dst_region_num_blocks["remote"],
        pairs=(pairs.remote, pairs.group),
    )
    local_descs = expand_strided_descs(consumer.src_blocks_data)
    assert local_descs[local_ids].tolist() == [
        [d_raw.data_ptr() + row + i * page, page, 0] for i in range(num_layers)
    ]
    assert remote_descs[remote_ids].tolist() == [
        [p_raw.data_ptr() + 2 * row + i * page, page, 0] for i in range(num_layers)
    ]


@record_nixl
def test_whole_row_gqa_rejects_a_per_layer_peer(dist_init):
    producer, *_ = _make_gqa_worker(KVCacheLayout.BLHNC, whole_row=False)
    consumer, *_ = _make_gqa_worker(KVCacheLayout.BLHNC, whole_row=True)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "whole_row_gqa: local=True, remote=False; "
            "KV heads per rank: local=2, remote=2."
        ),
    ):
        consumer.add_remote_agent(_metadata(producer))


@record_nixl
def test_whole_row_gqa_rejects_a_peer_with_other_head_sharding(dist_init):
    # E.g. P at TP1 and D at TP2: D's pages hold half the KV heads, so the
    # rows differ and a whole row cannot move as one unit.
    producer, *_ = _make_gqa_worker(KVCacheLayout.BLHNC, whole_row=True, num_kv_heads=2)
    consumer, *_ = _make_gqa_worker(KVCacheLayout.BLHNC, whole_row=True, num_kv_heads=1)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "whole_row_gqa: local=True, remote=True; "
            "KV heads per rank: local=1, remote=2."
        ),
    ):
        consumer.add_remote_agent(_metadata(producer))
