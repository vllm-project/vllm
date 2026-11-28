# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""kv_block_validation: the consumer must reject KV blocks whose producer-side
metadata entry does not name the content it expects.

The producer invalidates a block's entry before rewriting the block and
publishes the block hash after the forward pass that filled it; the consumer
reads the entries after the data, so a stale, reused or wrong block is caught.
"""

import ctypes
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlConnector,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.kv_validation import (
    ENTRY_SIZE,
    KVBlockMetaPool,
    KVBlockValidator,
    make_entries,
)
from vllm.forward_context import ForwardContext

from .test_nixl_connector import FakeNixlConnectorWorker, FakeNixlWrapper
from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
)

BLOCK_SIZE = 16
NUM_BLOCKS = 32
_REMOTE = FakeNixlConnectorWorker.REMOTE_ENGINE_ID
_HASHES = [bytes([i]) * 32 for i in range(8)]


class CopyingFakeNixlWrapper(FakeNixlWrapper):
    """Fake NIXL that really copies metadata entries between host buffers.

    KV data transfers stay no-ops: their descriptors address fake caches.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_dlists: dict[int, np.ndarray] = {}
        self.copies: dict[int, list[tuple[int, int]]] = {}
        self.notifs: list[bytes] = []

    def prep_xfer_dlist(self, agent_name, descs, mem_type=None) -> int:
        handle = super().prep_xfer_dlist(agent_name, descs, mem_type)
        descs = np.asarray(descs)
        if (
            descs.ndim == 2
            and descs.shape[1] == 3
            and (descs[:, 1] == ENTRY_SIZE).all()
        ):
            self.entry_dlists[handle] = descs
        return handle

    def make_prepped_xfer(
        self, xfer_type, local, local_ids, remote, remote_ids, notif_msg=None
    ) -> int:
        handle = super().make_prepped_xfer(
            xfer_type, local, local_ids, remote, remote_ids, notif_msg
        )
        if local in self.entry_dlists and remote in self.entry_dlists:
            self.copies[handle] = [
                (
                    int(self.entry_dlists[local][i, 0]),
                    int(self.entry_dlists[remote][j, 0]),
                )
                for i, j in zip(local_ids, remote_ids)
            ]
        return handle

    def transfer(self, handle: int) -> str:
        for dst, src in self.copies.pop(handle, []):
            ctypes.memmove(dst, src, ENTRY_SIZE)
        return "PROC"

    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        self.notifs.append(notif_msg)


def _producer_meta(scheduler):
    output = scheduler.schedule()
    meta = output.kv_connector_metadata
    assert isinstance(meta, NixlConnectorMetadata)
    return output, meta


def test_producer_invalidates_new_blocks_and_publishes_filled_ones():
    """Fresh blocks are invalidated before the forward pass that writes them and
    published once full, chunk by chunk; prefix hits keep their entries."""
    vllm_config = create_vllm_config(
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=64,
        kv_connector_extra_config={"kv_block_validation": "enforce"},
        kv_role="kv_producer",
    )
    scheduler = create_scheduler(vllm_config)
    first = create_request(num_tokens=100, common_prefix_len=64, max_tokens=4)
    scheduler.add_request(first)

    # Chunk 1 computes tokens [0, 64): allocates and fills blocks 0-3.
    output, meta = _producer_meta(scheduler)
    blocks = scheduler.kv_cache_manager.get_block_ids(first.request_id)[0]
    assert meta.kv_meta_invalidate == blocks[:4]
    assert meta.kv_meta_publish_ids == blocks[:4]
    assert meta.kv_meta_publish_hashes == first.block_hashes[:4]
    scheduler.update_from_output(output, create_model_runner_output([first]))

    # Chunk 2 computes [64, 100): allocates blocks 4-6 and fills 4-5. The
    # partial block 6 has no hash yet.
    output, meta = _producer_meta(scheduler)
    blocks = scheduler.kv_cache_manager.get_block_ids(first.request_id)[0]
    assert meta.kv_meta_invalidate == blocks[4:7]
    assert meta.kv_meta_publish_ids == blocks[4:6]
    assert meta.kv_meta_publish_hashes == first.block_hashes[4:6]
    scheduler.update_from_output(output, create_model_runner_output([first]))

    # A request sharing the first 64 tokens hits blocks 0-3 in the prefix
    # cache: their entries are kept, only its own blocks are invalidated.
    second = create_request(num_tokens=80, common_prefix_len=64, max_tokens=4)
    scheduler.add_request(second)
    _, meta = _producer_meta(scheduler)
    shared = scheduler.kv_cache_manager.get_block_ids(second.request_id)[0]
    assert shared[:4] == blocks[:4]
    assert set(shared[4:]) <= set(meta.kv_meta_invalidate)
    assert not set(shared[:4]) & set(meta.kv_meta_invalidate)
    assert shared[4] in meta.kv_meta_publish_ids


@pytest.fixture
def entries_pair():
    """A producer pool and a consumer validator connected by a copying fake."""
    nixl = CopyingFakeNixlWrapper("agent")
    producer = KVBlockMetaPool(NUM_BLOCKS, torch.device("cpu"))
    consumer = KVBlockMetaPool(NUM_BLOCKS, torch.device("cpu"))
    validator = KVBlockValidator(
        nixl, [], "enforce", "id", NUM_BLOCKS, "DRAM", pool=consumer
    )
    validator.add_remote(
        _REMOTE, 0, "remote", producer.entries.data_ptr(), NUM_BLOCKS, 0, "id"
    )
    return nixl, producer, consumer, validator


def _validate(validator, remote_ids, expected_hashes, local_ids=(1, 2)):
    validator.add_read(
        "req", _REMOTE, 0, 0, list(local_ids), remote_ids, expected_hashes
    )
    for handle in validator.post("req"):
        validator.nixl_wrapper.transfer(handle)
    return validator.finish("req")


def test_validator_accepts_blocks_holding_the_expected_content(entries_pair):
    _, producer, consumer, validator = entries_pair
    producer.write([5, 6], make_entries(_HASHES[:2]))

    assert _validate(validator, [5, 6], _HASHES[:2]) == 0
    # The received blocks are now valid sources themselves.
    assert torch.equal(consumer.entries[[1, 2]], producer.entries[[5, 6]])


@pytest.mark.parametrize("reused", [False, True])
def test_validator_rejects_stale_or_reused_blocks(entries_pair, reused):
    """Block 6 was reallocated after the producer published it: its entry is
    INVALID mid-rewrite, or names the new content afterwards."""
    _, producer, consumer, validator = entries_pair
    producer.write([5, 6], make_entries(_HASHES[:2]))
    producer.invalidate([6])
    if reused:
        producer.write([6], make_entries([_HASHES[7]]))

    assert _validate(validator, [5, 6], _HASHES[:2]) == 1
    assert not consumer.entries[[1, 2]].any()


def test_validator_skips_peers_hashing_differently():
    nixl = CopyingFakeNixlWrapper("agent")
    validator = KVBlockValidator(nixl, [], "enforce", "id", NUM_BLOCKS, "DRAM")
    validator.add_remote(_REMOTE, 0, "remote", 1234, NUM_BLOCKS, 0, "other-id")
    validator.add_remote(_REMOTE, 1, "remote", None, 0, 0, None)
    assert not validator.supports(_REMOTE, 0)
    assert not validator.supports(_REMOTE, 1)


@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer", "kv_both"])
def test_only_producers_keep_a_pool(dist_init, kv_role):
    """A consumer's blocks are never read, so it validates but keeps no pool;
    a producer keeps the pool but has nothing to validate."""
    vllm_config = create_vllm_config(
        block_size=BLOCK_SIZE,
        kv_connector_extra_config={"kv_block_validation": "enforce"},
        kv_role=kv_role,
    )
    kv_cache_config = make_kv_cache_config(block_size=BLOCK_SIZE, num_blocks=NUM_BLOCKS)
    producer = kv_role != "kv_consumer"
    consumer = kv_role != "kv_producer"

    scheduler = NixlConnector(vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config)
    assert (scheduler.connector_scheduler._kv_meta_tracker is not None) is producer
    scheduler.shutdown()

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.NixlWrapper",
        CopyingFakeNixlWrapper,
    ):
        worker = FakeNixlConnectorWorker(
            vllm_config,
            "engine",
            hand_shake_latency=0,
            kv_cache_config=kv_cache_config,
        )
    pool_addr = worker._setup_kv_validation(torch.device("cpu"))
    assert (pool_addr is not None) is producer
    assert (worker._kv_meta_pool is not None) is producer
    assert (worker._kv_validator is not None) is consumer


def _make_connector(mode: str):
    vllm_config = create_vllm_config(
        block_size=BLOCK_SIZE, kv_connector_extra_config={"kv_block_validation": mode}
    )
    kv_cache_config = make_kv_cache_config(block_size=BLOCK_SIZE, num_blocks=NUM_BLOCKS)
    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
    worker = FakeNixlConnectorWorker(
        vllm_config,
        connector.engine_id,
        hand_shake_latency=0,
        kv_cache_config=kv_cache_config,
    )
    connector.connector_worker = worker
    remote_agents, _ = worker._nixl_handshake(
        host="localhost", port=1234, remote_tp_size=1, expected_engine_id=_REMOTE
    )
    worker._remote_agents[_REMOTE] = remote_agents
    worker._setup_kv_validation(torch.device("cpu"))
    # The fake handshake carries no metadata pool: attach the producer's.
    producer = KVBlockMetaPool(NUM_BLOCKS, torch.device("cpu"))
    assert worker._kv_validator is not None
    worker._kv_validator.add_remote(
        _REMOTE,
        0,
        remote_agents[(0, 0)],
        producer.entries.data_ptr(),
        NUM_BLOCKS,
        0,
        worker._kv_validator.identity,
    )
    return connector, worker, producer


def _recv(connector):
    meta = NixlConnectorMetadata()
    meta.add_new_req_to_recv(
        request_id="req",
        local_block_ids=([10, 11],),
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_block_ids": ([20, 21],),
            "remote_engine_id": _REMOTE,
            "remote_request_id": "prefill-req",
            "remote_host": "localhost",
            "remote_port": 1234,
            "tp_size": 1,
        },
        awaiting_kvs=True,
    )
    req = meta.reqs_to_recv["req"]
    req.kv_expected_hashes = _HASHES[:2]
    meta.kv_meta_invalidate = [10, 11]
    connector.bind_connector_metadata(meta)
    ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    connector.start_load_kv(ctx)
    return connector.connector_worker.get_transfer_results()


@pytest.mark.parametrize(
    "mode,stale,failed",
    [("enforce", False, False), ("enforce", True, True), ("log", True, False)],
)
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.NixlWrapper",
    CopyingFakeNixlWrapper,
)
def test_pull_validates_before_reporting_the_recv(dist_init, mode, stale, failed):
    """The producer is notified only after the metadata read; an "enforce"
    mismatch becomes a failed recv, following kv_load_failure_policy."""
    connector, worker, producer = _make_connector(mode)
    producer.write([20, 21], make_entries(_HASHES[:2]))
    if stale:
        producer.invalidate([21])

    result = _recv(connector)

    assert "req" in result.finished_recving
    assert ("req" in result.failed_recving) is failed
    assert worker.nixl_wrapper.notifs == [b"prefill-req:1"]
    assert worker.xfer_stats.data["num_kv_block_validation_failures"] == (
        [1] if stale else []
    )
    # Metadata reads are not KV transfers.
    assert worker.xfer_stats.num_successful_transfers == 1
