# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL registration and handshake for FlashInfer separate K/V head slots.

With ``AttentionConfig.flashinfer_separate_kv_head_slots``, an unquantized
FlashInfer page under LBHNC is ``[K heads (H, N, D) | V heads (H, N, D)]``.
The NIXL worker registers the K half and the V half of each page as two
regions that step by the full page. A remote that keeps K and V in separate
per-layer slabs (two dense regions per layer) can then exchange blocks with
it, with no permute after the transfer.
"""

from unittest.mock import patch

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
    NixlConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    compute_nixl_compatibility_hash,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
)

from .test_nixl_connector import FakeNixlConnectorWorker, FakeNixlWrapper
from .utils import create_vllm_config, expand_strided_descs

NUM_BLOCKS = 8
NUM_KV_HEADS = 4
HEAD_SIZE = 16
DTYPE = torch.float16
ELEM = 2  # bytes per fp16 element

NIXL_WORKER = "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker"


def _flashinfer_split_spec(block_size: int) -> FullAttentionSpec:
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    return FlashInferBackend.customize_spec(spec)


def _split_config(layout: str = "LBHNC", block_size: int = 16):
    vllm_config = create_vllm_config(
        attention_backend="FLASHINFER", block_size=block_size
    )
    vllm_config.attention_config.flashinfer_separate_kv_head_slots = True
    vllm_config.cache_config.kv_cache_layout = layout
    return vllm_config


def _register(vllm_config, block_size: int, layout: str, num_layers: int = 2):
    """Run register_kv_caches on dense FlashInfer split-slot caches.

    Returns (worker, caches, block_runs, raw).
    """
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    with set_current_vllm_config(vllm_config):
        spec = _flashinfer_split_spec(block_size)
    layers = [f"layer{i}" for i in range(num_layers)]
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layers, spec)],
    )
    with (
        patch(f"{NIXL_WORKER}.NixlWrapper") as mock_wrapper,
        patch(f"{NIXL_WORKER}.threading.Event"),
        patch(f"{NIXL_WORKER}.threading.Thread") as mock_thread,
        patch(f"{NIXL_WORKER}.get_current_attn_backends") as mock_backends,
        set_current_vllm_config(vllm_config),
    ):
        mock_backends.return_value = [FlashInferBackend]
        connector = NixlConnector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
        connector.connector_worker = FakeNixlConnectorWorker(
            vllm_config,
            connector.engine_id,
            hand_shake_latency=0,
            kv_cache_layout=layout,
            kv_cache_config=kv_cache_config,
        )
        wrapper = mock_wrapper.return_value
        connector.connector_worker.nixl_wrapper = wrapper
        wrapper.get_agent_metadata.return_value = b"fake_agent_metadata"
        mock_thread.return_value.is_alive.return_value = False

        raw = torch.zeros(
            spec.page_size_bytes * NUM_BLOCKS * num_layers,
            dtype=torch.int8,
            device=current_platform.device_type,
        )
        caches = dense_kv_cache_views(
            raw, spec, NUM_BLOCKS, num_layers=num_layers, layout=KVCacheLayout[layout]
        )
        connector.register_kv_caches(dict(zip(layers, caches)))
        _agent, block_runs = wrapper.prep_xfer_dlist.call_args[0]
        return connector.connector_worker, caches, block_runs, raw, spec


@pytest.mark.skip_global_cleanup
def test_customize_spec_splits_unquantized_kv_only_when_enabled():
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    base = FullAttentionSpec(
        block_size=16, num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE, dtype=DTYPE
    )
    off = create_vllm_config(attention_backend="FLASHINFER")
    with set_current_vllm_config(off):
        assert FlashInferBackend.customize_spec(base) == base

    on = _split_config()
    with set_current_vllm_config(on):
        spec = FlashInferBackend.customize_spec(base)
    assert spec.num_head_slots == 2 * NUM_KV_HEADS
    assert spec.state_content_bytes == HEAD_SIZE * ELEM
    # Same page size as the packed layout: only the order inside a page moves.
    assert spec.page_size_bytes == base.page_size_bytes


@pytest.mark.skip_global_cleanup
def test_split_kv_head_slots_restrict_layouts_to_block_contiguous():
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    with set_current_vllm_config(_split_config()):
        layouts = FlashInferBackend.supported_kv_cache_layouts()
    assert layouts == (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)
    assert all(layout.is_block_contiguous for layout in layouts)


@pytest.mark.skip_global_cleanup
def test_compatibility_hash_includes_split_kv_head_slots():
    packed = create_vllm_config(attention_backend="FLASHINFER")
    split = _split_config()
    assert compute_nixl_compatibility_hash(
        packed, "FLASHINFER"
    ) != compute_nixl_compatibility_hash(split, "FLASHINFER")
    assert compute_nixl_compatibility_hash(
        split, "FLASHINFER"
    ) == compute_nixl_compatibility_hash(_split_config(), "FLASHINFER")


@pytest.mark.parametrize("layout", ["LBHNC", "BLHNC"])
def test_split_slot_page_registers_k_and_v_regions(
    default_vllm_config, dist_init, layout
):
    block_size = 16
    worker, caches, block_runs, raw, spec = _register(
        _split_config(layout, block_size), block_size, layout
    )
    page = spec.page_size_bytes
    half = page // 2
    num_layers = len(caches)

    # Two regions per layer, each half a page long, stepping by the layer's
    # block stride (a full page for LBHNC; all layers' pages for BLHNC).
    assert worker.block_len_per_layer == [half] * (2 * num_layers)
    bases = worker.kv_caches_base_addr[worker.engine_id][0]
    for layer, cache in enumerate(caches):
        stride = cache.stride(0) * cache.element_size()
        assert bases[2 * layer] == cache.data_ptr()
        assert bases[2 * layer + 1] == cache.data_ptr() + half
        assert worker.block_stride_per_layer[2 * layer : 2 * layer + 2] == [stride] * 2

    # The K region of block b holds exactly the K heads of block b, and the V
    # region exactly its V heads.
    blocks = expand_strided_descs(block_runs)
    assert len(blocks) == 2 * num_layers * NUM_BLOCKS
    for region in range(2 * num_layers):
        layer, kv = divmod(region, 2)
        k_or_v = caches[layer][:, kv * NUM_KV_HEADS : (kv + 1) * NUM_KV_HEADS]
        for b in range(NUM_BLOCKS):
            addr, length, _dev = blocks[region * NUM_BLOCKS + b]
            assert length == half
            assert addr == k_or_v[b].data_ptr()
            assert k_or_v[b].is_contiguous()
            assert k_or_v[b].nbytes == half


@pytest.mark.skip_global_cleanup
def test_split_slot_flag_without_split_spec_keeps_one_region(
    default_vllm_config, dist_init
):
    # The flag alone does not split a page whose spec was not customized,
    # e.g. a non-FlashInfer backend. The page stays one region.
    vllm_config = _split_config()
    with patch(
        "vllm.v1.attention.backends.flashinfer._separate_kv_head_slots_enabled",
        return_value=False,
    ):
        worker, caches, _runs, _raw, spec = _register(vllm_config, 16, "LBHNC")
    assert spec.num_head_slots is None
    assert worker.block_len_per_layer == [spec.page_size_bytes] * len(caches)


def test_handshake_with_separate_kv_slab_peer(default_vllm_config, dist_init):
    """A remote with K and V as two dense slabs per layer pairs region for
    region with a split-slot local cache: same count, same block_len. Remote
    descriptors step by the remote's own (dense) stride."""
    block_size = 16
    worker, caches, _runs, _raw, spec = _register(
        _split_config("LBHNC", block_size), block_size, "LBHNC"
    )
    worker.nixl_wrapper = FakeNixlWrapper("local")
    captured = []
    real_prep = worker.nixl_wrapper.prep_xfer_dlist

    def capture(agent_name, descs, *args, **kwargs):
        captured.append((agent_name, descs))
        return real_prep(agent_name, descs, *args, **kwargs)

    worker.nixl_wrapper.prep_xfer_dlist = capture
    worker.nixl_wrapper.get_xfer_descs = lambda data, mem: data

    half = spec.page_size_bytes // 2
    num_regions = 2 * len(caches)
    slab = NUM_BLOCKS * half
    remote_bases = [0x10000000 + r * slab for r in range(num_regions)]
    meta = NixlAgentMetadata(
        engine_id=FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
        agent_metadata=FakeNixlWrapper.AGENT_METADATA,
        kv_caches_base_addr=remote_bases,
        device_id=0,
        num_blocks=NUM_BLOCKS,
        block_lens=[half] * num_regions,
        block_strides=[half] * num_regions,
        kv_cache_layout="LBHNC",
        block_size=block_size,
        ssm_sizes=(0, 0),
        attn_backend_name=worker.backend_name,
        physical_blocks_per_logical_kv_block=1,
    )
    worker.add_remote_agent(meta, remote_tp_size=1)

    remote = [d for name, d in captured if name == FakeNixlWrapper.REMOTE_AGENT_NAME]
    assert len(remote) == 1
    blocks = expand_strided_descs(remote[0])
    expected = [
        (remote_bases[r] + b * half, half, 0)
        for r in range(num_regions)
        for b in range(NUM_BLOCKS)
    ]
    assert [tuple(int(x) for x in row) for row in blocks] == expected
