# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real HF3FS save queue with faults only at storage services."""

import atexit
import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from unittest.mock import patch

import pytest
import torch

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CacheConfig, KVTransferConfig, ModelConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import hf3fs
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    DEFAULT_MAX_IO_ENTRIES,
    HF3FSKVConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.hf3fs_mock_client import (
    Hf3fsClient,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

BLOCKS = DEFAULT_MAX_IO_ENTRIES + 1
PAGE_BYTES = 16 * 4 * 4
DEFAULT_REPLY = object()


class StorageFaults:
    """Fault controls for the two external HF3FS services."""

    def __init__(self):
        self.allocation_reply = DEFAULT_REPLY
        self.writer_reply = DEFAULT_REPLY
        self.hold_tail = False
        self.tail_started = threading.Event()
        self.release_tail = threading.Event()
        self.confirmed = []
        self.writes = []
        self.allocations = []

    def initialize(self, *args, **kwargs):
        """Accept the connector's metadata-service registration."""

    def allocate_pages_for_keys(self, rank, keys):
        """Allocate distinct pages or return the injected service response."""
        self.allocations.append((rank, keys))
        if self.allocation_reply is not DEFAULT_REPLY:
            return self.allocation_reply
        return [(key, i) for i, (key, _) in enumerate(keys)]

    def confirm_write_for_keys(self, rank, keys, failed_keys):
        """Record publication only after the connector validates its writers."""
        self.confirmed.extend(keys)

    def release_pages(self, *args):
        """Accept old-code cleanup during negative-control runs."""

    def batch_key_exists(self, keys):
        """Keep serving requests cold in the controlled storage service."""
        return [False] * len(keys)


@pytest.fixture
def storage_faults(monkeypatch):
    """Replace only the external metadata and file storage services."""
    faults = StorageFaults()

    class FaultClient(Hf3fsClient):
        """A file-backed storage service with bounded, explicit write faults."""

        def batch_write(self, offsets, tensors, event):
            event.synchronize()
            faults.writes.append(list(offsets))
            if faults.hold_tail:
                if offsets[0] == 0:
                    if not faults.tail_started.wait(10):
                        raise TimeoutError("second storage writer never started")
                else:
                    faults.tail_started.set()
                    if not faults.release_tail.wait(15):
                        raise TimeoutError("held storage writer was not released")
                    return super().batch_write(offsets, tensors, event)
            result = faults.writer_reply
            if result is not DEFAULT_REPLY:
                if isinstance(result, Exception):
                    raise result
                return result
            return super().batch_write(offsets, tensors, event)

    module = hf3fs.hf3fs_connector
    monkeypatch.setattr(module, "Hf3fsMetadataClient", lambda: faults)
    monkeypatch.setattr(module, "Hf3fsClient", FaultClient)
    try:
        yield faults
    finally:
        faults.release_tail.set()


@pytest.fixture
def save_connector(tmp_path, storage_faults, dist_init):
    """Start real worker queues, CUDA streams, buffers and file-backed I/O."""
    faults = storage_faults
    config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", skip_tokenizer_init=True),
        cache_config=CacheConfig(block_size=16),
        kv_transfer_config=KVTransferConfig(
            kv_connector="HF3FSKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "hf3fs_storage_path": str(tmp_path),
                "hf3fs_file_size": PAGE_BYTES * 64,
                "hf3fs_client_numjobs": 2,
            },
        ),
    )
    with patch("signal.signal"):
        connector = HF3FSKVConnector(
            config,
            KVConnectorRole.WORKER,
            KVCacheConfig(num_blocks=BLOCKS, kv_cache_tensors=[], kv_cache_groups=[]),
        )
    cache = torch.arange(BLOCKS * 16 * 4, device="cuda", dtype=torch.float32)
    try:
        connector.register_kv_caches({"layer": cache.reshape(BLOCKS, 1, 16, 4)})
        yield connector, faults, cache
    finally:
        faults.release_tail.set()
        connector.close()
        atexit.unregister(connector.close)


@pytest.mark.parametrize(
    "reply",
    [[], [PAGE_BYTES] * 2, None, [True], [PAGE_BYTES - 1], RuntimeError("I/O failed")],
    ids=["empty", "too-many", "none", "bool", "short-page", "exception"],
)
def test_invalid_writer_result_is_not_published(save_connector, reply):
    """An accepted write must report exactly one full integer result per page."""
    connector, faults, _ = save_connector
    faults.writer_reply = reply
    operation = connector._async_manager.submit_save_operation("r", [0], ["hash-0"])
    assert operation.result(timeout=10) is False
    assert faults.writes == [[0]]
    assert faults.confirmed == []


def test_missing_writer_result_is_not_published(save_connector):
    """A partial result list cannot authorize both allocated pages."""
    connector, faults, _ = save_connector
    faults.writer_reply = [PAGE_BYTES]
    operation = connector._async_manager.submit_save_operation(
        "r", [0, 1], ["hash-0", "hash-1"]
    )
    assert operation.result(timeout=10) is False
    assert faults.confirmed == []


@pytest.mark.parametrize(
    "reply",
    [
        {},
        [],
        [("wrong-hash", 0)],
        [("hash-0", True)],
        [("hash-0", "0")],
        [("hash-0", -1)],
        [("hash-0", 0), ("extra", 1)],
    ],
)
def test_invalid_allocation_reply_rejects_without_io(save_connector, reply):
    """Malformed metadata cannot start a storage write or publish a key."""
    connector, faults, _ = save_connector
    faults.allocation_reply = reply
    operation = connector._async_manager.submit_save_operation("r", [0], ["hash-0"])
    assert operation.result(timeout=10) is False
    assert faults.writes == [] and faults.confirmed == []


def test_duplicate_pages_reject_without_io(save_connector):
    """Two cache keys cannot concurrently write the same returned page."""
    connector, faults, _ = save_connector
    faults.allocation_reply = [("hash-0", 7), ("hash-1", 7)]
    operation = connector._async_manager.submit_save_operation(
        "r", [0, 1], ["hash-0", "hash-1"]
    )
    assert operation.result(timeout=10) is False
    assert faults.writes == [] and faults.confirmed == []


def test_block_key_count_mismatch_rejects_before_allocation(save_connector):
    """A missing key cannot silently drop a source block from a save."""
    connector, faults, _ = save_connector
    operation = connector._async_manager.submit_save_operation("r", [0, 1], ["hash-0"])
    assert operation.result(timeout=10) is False
    assert faults.allocations == [] and faults.writes == []


@pytest.mark.parametrize("raise_error", [False, True])
def test_failed_first_writer_waits_for_live_tail(save_connector, raise_error):
    """A failure cannot release the request while a later writer owns buffers."""
    connector, faults, _ = save_connector
    faults.hold_tail = True
    faults.writer_reply = (
        RuntimeError("first writer failed")
        if raise_error
        else [PAGE_BYTES - 1] * DEFAULT_MAX_IO_ENTRIES
    )
    manager = connector._async_manager
    operation = manager.submit_save_operation(
        "r", list(range(BLOCKS)), [f"hash-{i}" for i in range(BLOCKS)]
    )
    try:
        assert faults.tail_started.wait(10)
        with pytest.raises(TimeoutError):
            operation.result(timeout=0.2)
        completed, _ = manager.get_finished_operations({"r"})
        assert "r" not in completed
        assert faults.confirmed == []
    finally:
        faults.release_tail.set()
    assert operation.result(timeout=10) is False
    completed, _ = manager.get_finished_operations(set())
    assert "r" in completed
    assert faults.confirmed == []


def test_successful_save_preserves_bytes(save_connector):
    """The success control writes the gathered GPU data before publication."""
    connector, faults, cache = save_connector
    operation = connector._async_manager.submit_save_operation("r", [0], ["hash-0"])
    assert operation.result(timeout=10) is True
    assert faults.confirmed == [("hash-0", 0)]
    restored = torch.empty((1, 1, 16, 4), dtype=torch.float32)
    assert connector._clients[0].batch_read([0], [restored]) == [PAGE_BYTES]
    assert torch.equal(restored.flatten(), cache[:64].cpu())


def test_cancelled_writer_is_drained(save_connector):
    """An executor-cancelled queued writer is a failure, not a deadlock."""
    connector, _, _ = save_connector
    writer: Future[list[int]] = Future()
    assert writer.cancel()
    assert connector._async_manager._drain_writers([(writer, 1)]) is False


def test_shutdown_waits_for_active_save(save_connector):
    """The engine lifecycle hook must finish active writes before returning."""
    connector, faults, _ = save_connector
    faults.hold_tail = True
    operation = connector._async_manager.submit_save_operation(
        "r", list(range(BLOCKS)), [f"hash-{i}" for i in range(BLOCKS)]
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            assert faults.tail_started.wait(10)
            shutdown = executor.submit(connector.shutdown)
            with pytest.raises(TimeoutError):
                shutdown.result(timeout=0.2)
        finally:
            faults.release_tail.set()
        shutdown.result(timeout=10)
    assert operation.result(timeout=10) is True


@single_gpu_only
def test_llm_does_not_publish_empty_save_results(
    tmp_path, monkeypatch, vllm_runner, storage_faults
):
    """Real generation cannot publish KV after the storage service wrote nothing."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    storage_faults.writer_reply = []
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.4,
        block_size=16,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="HF3FSKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "hf3fs_storage_path": str(tmp_path),
                "hf3fs_file_size": 64 * 1024 * 1024,
                "hf3fs_client_numjobs": 2,
            },
        ),
    ) as runner:
        outputs = runner.llm.generate(
            {"prompt_token_ids": [1000 + i % 100 for i in range(160)]},
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
        )
        assert len(outputs) == 1 and len(outputs[0].outputs[0].token_ids) == 1
    assert storage_faults.writes, "generation must exercise the real save path"
    assert storage_faults.confirmed == []
