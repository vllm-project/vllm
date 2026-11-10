# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO on KV caches the allocator splits into smaller kernel blocks.

Backends whose kernels need a smaller block than `--block-size` (e.g. the
ROCm sparse MLA and indexer backends at block size 8) get caches viewed as
kernel blocks. Transfers still move whole manager blocks, so prefill and
decode may split differently.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    RemoteAllocInfo,
    WriteTask,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout import (
    get_layer_transfer_geometry,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from ..utils import make_moriio_writer
from .test_moriio_hetero_tp_write import (
    BLOCK_SIZE,
    DECODE_BLOCKS,
    MODELS,
    NOTIFY_PORT,
    NUM_BLOCKS,
    PREFILL_BLOCKS,
    _build_rank,
    _connect,
    _consumer_wrapper,
    _global_state,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("block_size", "kernel_block_size", "head_size", "content", "tokens_per_state"),
    [(8, 1, 576, None, 1), (8, 1, 128, 132, 1), (640, 128, 128, 132, 4)],
    ids=["mla_latent", "indexer", "pooled_indexer"],
)
def test_kernel_split_view_matches_manager_geometry(
    block_size, kernel_block_size, head_size, content, tokens_per_state
):
    num_blocks = 3
    spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        head_size_v=0,
        dtype=torch.uint8,
        state_content_bytes=content,
        tokens_per_state=tokens_per_state,
    )
    raw = torch.zeros(num_blocks * spec.page_size_bytes, dtype=torch.int8)
    (manager_view,) = dense_kv_cache_views(
        raw, spec, num_blocks, 1, KVCacheLayout.LBNHC
    )
    (kernel_view,) = dense_kv_cache_views(
        raw, spec, num_blocks, 1, KVCacheLayout.LBNHC, kernel_block_size
    )
    layer_to_spec = {"layer": spec}

    geometry = get_layer_transfer_geometry("layer", kernel_view, layer_to_spec)

    assert geometry == get_layer_transfer_geometry("layer", manager_view, layer_to_spec)
    assert geometry.num_blocks == num_blocks
    assert geometry.block_len == geometry.block_stride == spec.page_size_bytes


def test_kernel_split_multi_head_view_is_unsupported():
    # A raw block copy keeps the byte order of a multi-head split only if P
    # and D split alike, which the handshake does not check.
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=4, dtype=torch.bfloat16
    )
    raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
    (kernel_view,) = dense_kv_cache_views(raw, spec, 3, 1, KVCacheLayout.LBNHC, 4)

    with pytest.raises(ValueError, match="Unsupported MoRIIO"):
        get_layer_transfer_geometry("layer", kernel_view, {"layer": spec})


@pytest.mark.parametrize(
    ("prefill_kernel_block_size", "decode_kernel_block_size"),
    [(1, BLOCK_SIZE), (BLOCK_SIZE, 1), (1, 2)],
)
def test_write_kernel_split_lands_byte_exact(
    monkeypatch, prefill_kernel_block_size, decode_kernel_block_size
):
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.CONSUMER)
    heads, layers = MODELS["mla"]
    state = _global_state(layers, heads)
    decode = _build_rank(
        "mla", 0, 1, KVCacheLayout.LBNHC, True, state, False, decode_kernel_block_size
    )
    decode.wrapper = _consumer_wrapper()
    prefill = _build_rank(
        "mla", 0, 1, KVCacheLayout.LBNHC, True, state, True, prefill_kernel_block_size
    )
    _connect(prefill, [decode], "decode")
    mapping = prefill.worker._write_tp_mapping(1)
    info = RemoteAllocInfo(block_ids=[DECODE_BLOCKS["full"]])
    prefill.wrapper.done_remote_allocate_req_dict["t"] = info
    writer = make_moriio_writer(prefill.worker)
    for layer in layers:
        writer._execute_write_task(
            WriteTask(
                request_id="req",
                transfer_id="t",
                dst_engine_id="decode",
                local_block_ids=[PREFILL_BLOCKS["full"]],
                remote_block_ids_hint=None,
                layer_name=layer.name,
                event=SimpleNamespace(synchronize=lambda: None),
                remote_notify_port=NOTIFY_PORT,
                remote_ip="decode",
                remote_tp_size=1,
                tp_mapping=mapping,
            )
        )

    for layer, region in zip(layers, decode.regions, strict=True):
        received = region.view(NUM_BLOCKS, -1)
        expected = torch.zeros_like(received)
        for src, dst in zip(PREFILL_BLOCKS["full"], DECODE_BLOCKS["full"]):
            expected[dst] = state[layer.name][src].flatten()
        assert torch.equal(received, expected), layer.name
