# SPDX-License-Identifier: Apache-2.0
"""In-process, real-NIXL integration test for the P2P L2 adapter.

Stands up a peer side (a real ``StorageManager`` with objects in L1, a NIXL
transfer-channel context registered against that L1, and an MQ server hosting a
``P2PController``) and a local side (the global NIXL context over a destination
buffer + a ``P2PL2Adapter``). It then drives the adapter through the full
lookup -> load (loopback RDMA read) -> unlock lifecycle and verifies the pulled
bytes match the peer's.

Requires a working NIXL runtime and CUDA (the L1 pool is pinned DRAM); skipped
otherwise.
"""

# Standard
import itertools
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

nixl = pytest.importorskip("nixl")
# Third Party
import zmq  # noqa: E402

# First Party
from lmcache.v1.distributed.api import (  # noqa: E402
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.config import (  # noqa: E402
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.internal_api import L1MemoryDesc  # noqa: E402
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import (  # noqa: E402
    P2PL2Adapter,
    P2PL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager  # noqa: E402
from lmcache.v1.distributed.transfer_channel import (  # noqa: E402
    delete_transfer_channel_context,
    initialize_transfer_channel_context,
)
from lmcache.v1.distributed.transfer_channel.impl.nixl_impl import (  # noqa: E402
    NixlTransferChannelContext,
)
from lmcache.v1.multiprocess.config import (  # noqa: E402
    CoordinatorConfig,
    P2PConfig,
)
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController  # noqa: E402
from lmcache.v1.multiprocess.mq import MessageQueueServer  # noqa: E402
from lmcache.v1.multiprocess.protocol import get_payload_classes  # noqa: E402

_PAGE = 4096
_NUM_KEYS = 3
_port_counter = itertools.count(18300)


def _next_url() -> str:
    return f"127.0.0.1:{next(_port_counter)}"


def _make_storage_manager(size_bytes: int) -> StorageManager:
    memory_config = L1MemoryManagerConfig(
        size_in_bytes=size_bytes,
        use_lazy=False,
        init_size_in_bytes=size_bytes,
        align_bytes=_PAGE,
    )
    l1_config = L1ManagerConfig(
        memory_config=memory_config,
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    config = StorageManagerConfig(
        l1_manager_config=l1_config,
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    return StorageManager(config)


class _PeerContext:
    """Minimal P2PController context -- only ``storage_manager`` is used."""

    def __init__(self, storage_manager: StorageManager) -> None:
        self.storage_manager = storage_manager


def _key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(i),
        model_name="test_model",
        kv_rank=0,
    )


def _poll(fn, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    result = fn()
    while result is None and time.monotonic() < deadline:
        time.sleep(0.02)
        result = fn()
    return result


def test_p2p_adapter_end_to_end():
    keys = [_key(i) for i in range(_NUM_KEYS)]
    layout = MemoryLayoutDesc(shapes=[torch.Size([_PAGE])], dtypes=[torch.uint8])

    peer_sm = _make_storage_manager(64 * 1024 * 1024)
    peer_tc_ctx = None
    mq_server = None
    adapter = None
    local_buf = torch.zeros((_NUM_KEYS + 1) * _PAGE, dtype=torch.uint8)

    try:
        # --- Peer side: store known objects in L1 ---
        reserved = peer_sm.reserve_write(keys, layout, mode="new")
        assert all(reserved[k] is not None for k in keys)
        expected_values = {}
        for i, key in enumerate(keys):
            value = i + 1
            reserved[key].tensor.fill_(value)
            expected_values[key] = value
        peer_sm.finish_write(keys)

        # --- Peer side: NIXL context over the peer's L1 pool ---
        peer_l1_desc = peer_sm._l1_manager.get_l1_memory_desc()
        peer_tc_url = _next_url()
        peer_tc_ctx = NixlTransferChannelContext(
            peer_l1_desc, listen_url=peer_tc_url, advertise_url=peer_tc_url
        )

        # --- Peer side: MQ server hosting the P2P controller ---
        controller = P2PController(
            _PeerContext(peer_sm),
            P2PConfig(),
            CoordinatorConfig(),
            instance_id="peer",
        )
        peer_mq_url = f"tcp://{_next_url()}"
        mq_server = MessageQueueServer(peer_mq_url, zmq.Context.instance())
        specs = controller.get_handlers()
        for spec in specs:
            mq_server.add_blocking_handler(
                spec.request_type,
                get_payload_classes(spec.request_type),
                spec.handler,
            )
        mq_server.add_normal_thread_pool([s.request_type for s in specs], max_workers=4)
        mq_server.start()

        # --- Local side: global NIXL context over the destination buffer ---
        local_tc_url = _next_url()
        local_l1_desc = L1MemoryDesc(
            ptr=local_buf.data_ptr(),
            size=local_buf.numel(),
            align_bytes=_PAGE,
        )
        initialize_transfer_channel_context(
            "nixl", local_l1_desc, local_tc_url, local_tc_url
        )

        # --- Build the adapter and drive the lifecycle ---
        adapter = P2PL2Adapter(
            P2PL2AdapterConfig(peer_mq_url, peer_tc_url, lookup_timeout_s=10.0)
        )

        # Lookup: every key is resident on the peer.
        lookup_id = adapter.submit_lookup_and_lock_task(keys, layout)
        bitmap = _poll(lambda: adapter.query_lookup_and_lock_result(lookup_id))
        assert bitmap is not None
        for i in range(_NUM_KEYS):
            assert bitmap.test(i) is True
        # Stashed remote addresses match the peer objects' real offsets.
        for key in keys:
            assert adapter._remote_addresses[key].offset == reserved[key].shm_offset

        # Load: pull each key into a distinct page of the local buffer.
        local_objs = [_LocalObj(offset=i * _PAGE, size=_PAGE) for i in range(_NUM_KEYS)]
        load_id = adapter.submit_load_task(keys, local_objs)
        load_bitmap = _poll(lambda: adapter.query_load_result(load_id))
        assert load_bitmap is not None
        for i, key in enumerate(keys):
            assert load_bitmap.test(i) is True
            page = local_buf[i * _PAGE : (i + 1) * _PAGE]
            assert torch.all(page == expected_values[key]), (
                f"page {i} did not receive the peer's bytes"
            )

        # Unlock: releases the peer locks and clears the stashed addresses.
        adapter.submit_unlock(keys)
        for key in keys:
            assert key not in adapter._remote_addresses
    finally:
        if adapter is not None:
            adapter.close()
        if mq_server is not None:
            mq_server.close()
        delete_transfer_channel_context()
        if peer_tc_ctx is not None:
            peer_tc_ctx.close()
        peer_sm.close()


class _LocalObj:
    """A stand-in for an L1 MemoryObj exposing only the offset/size the
    adapter reads when translating local transfer-channel addresses."""

    def __init__(self, offset: int, size: int) -> None:
        self.shm_offset = offset
        self.shm_byte_length = size
