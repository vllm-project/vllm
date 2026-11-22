# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HF3FS KV Connector high-level components:
- TestHf3fsMockClient      : file-backed mock client I/O correctness
- TestHF3FSKVConnectorStats: metric collection, aggregation, serialisation
"""

import contextlib
import os
import signal
import socket
import threading
import time
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    AsyncOperationManager,
    HF3FSKVConnector,
    HF3FSKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.common import (
    AtomicCounter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.hf3fs_mock_client import (
    Hf3fsClient as MockHf3fsClient,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def hf3fs_stats():
    """Fresh HF3FSKVConnectorStats instance."""
    return HF3FSKVConnectorStats()


def _make_cuda_event():
    """Return a real CUDA event when available, otherwise a MagicMock."""
    if torch.cuda.is_available():
        return torch.cuda.Event()
    return MagicMock()


# ===========================================================================
# TestHf3fsMockClient
# ===========================================================================


class TestHf3fsMockClient:
    """Tests for hf3fs_mock_client.Hf3fsClient (file-backend mock)."""

    def test_init_creates_file(self, tmp_path):
        """Initializing the client should create the backing file."""
        path = str(tmp_path / "test_file")
        client = MockHf3fsClient(path=path, size=4096, bytes_per_page=512, entries=4)
        assert os.path.exists(path), "Backing file should be created on init"
        assert os.path.getsize(path) == 4096
        client.close()

    @pytest.mark.parametrize(
        "dtype, bytes_per_page",
        [
            (torch.float32, 512),
            (torch.float16, 256),
            (torch.bfloat16, 256),
        ],
        ids=["float32", "float16", "bfloat16"],
    )
    def test_batch_write_and_read_dtype(self, tmp_path, dtype, bytes_per_page):
        """Write a tensor of the given dtype and verify round-trip correctness."""
        path = str(tmp_path / f"rw_{dtype}")
        client = MockHf3fsClient(
            path=path, size=bytes_per_page * 8, bytes_per_page=bytes_per_page, entries=4
        )
        elem_size = torch.tensor([], dtype=dtype).element_size()
        numel = bytes_per_page // elem_size
        tensor_write = torch.arange(numel, dtype=dtype)
        event = _make_cuda_event()

        results = client.batch_write([0], [tensor_write], event)
        assert results == [bytes_per_page], f"Write should succeed, got {results}"

        tensor_read = torch.zeros(numel, dtype=dtype)
        results = client.batch_read([0], [tensor_read])
        assert results == [bytes_per_page], f"Read should succeed, got {results}"
        assert torch.equal(tensor_write, tensor_read), (
            "Read tensor should match written tensor"
        )
        client.close()

    def test_batch_read_empty_file_returns_error(self, tmp_path):
        """Reading out-of-bounds offset should return -1."""
        bytes_per_page = 128
        size = bytes_per_page * 4
        path = str(tmp_path / "empty_read")
        client = MockHf3fsClient(
            path=path, size=size, bytes_per_page=bytes_per_page, entries=4
        )
        numel = bytes_per_page // 4
        tensor_read = torch.zeros(numel, dtype=torch.float32)
        results = client.batch_read([size], [tensor_read])  # offset == size => OOB
        assert results[0] == -1, "Out-of-bounds read should return -1"
        client.close()

    def test_batch_write_out_of_bounds_returns_error(self, tmp_path):
        """Writing at an offset beyond file size should return -1."""
        bytes_per_page = 128
        size = bytes_per_page * 4
        path = str(tmp_path / "oob_write")
        client = MockHf3fsClient(
            path=path, size=size, bytes_per_page=bytes_per_page, entries=4
        )
        numel = bytes_per_page // 4
        tensor = torch.ones(numel, dtype=torch.float32)
        event = _make_cuda_event()
        results = client.batch_write([size], [tensor], event)  # OOB offset
        assert results[0] == -1, "Out-of-bounds write should return -1"
        client.close()

    def test_multiple_tensors_rw(self, tmp_path):
        """Write multiple tensors at different offsets, then read all back."""
        bytes_per_page = 128
        n = 4
        path = str(tmp_path / "multi_rw")
        client = MockHf3fsClient(
            path=path,
            size=bytes_per_page * n * 2,
            bytes_per_page=bytes_per_page,
            entries=8,
        )
        tensors_write = [
            torch.full((bytes_per_page // 4,), float(i), dtype=torch.float32)
            for i in range(n)
        ]
        offsets = [i * bytes_per_page for i in range(n)]
        event = _make_cuda_event()

        results = client.batch_write(offsets, tensors_write, event)
        assert all(r == bytes_per_page for r in results)

        tensors_read = [
            torch.zeros(bytes_per_page // 4, dtype=torch.float32) for _ in range(n)
        ]
        results = client.batch_read(offsets, tensors_read)
        assert all(r == bytes_per_page for r in results)

        for i, (tw, tr) in enumerate(zip(tensors_write, tensors_read)):
            assert torch.allclose(tw, tr), f"Tensor {i} mismatch after round-trip"
        client.close()

    def test_flush_and_close_no_error(self, tmp_path):
        """flush() and close() should not raise exceptions."""
        path = str(tmp_path / "flush_close")
        client = MockHf3fsClient(path=path, size=1024, bytes_per_page=128, entries=4)
        client.flush()
        client.close()


# ===========================================================================
# TestHF3FSKVConnectorStats
# ===========================================================================


class TestHF3FSKVConnectorStats:
    """Tests for HF3FSKVConnectorStats metric collection and aggregation."""

    def test_initial_is_empty(self, hf3fs_stats):
        """Fresh stats object should report is_empty() == True."""
        assert hf3fs_stats.is_empty() is True

    @pytest.mark.parametrize(
        "task_type, duration_key",
        [
            ("Saved", "save_duration"),
            ("Loaded", "load_duration"),
        ],
        ids=["save", "load"],
    )
    def test_record_success_duration(self, hf3fs_stats, task_type, duration_key):
        """Recording a successful task should update duration list and total count."""
        hf3fs_stats.record_success_task_duration(task_type, 0.5)
        assert not hf3fs_stats.is_empty()
        assert len(hf3fs_stats.data[duration_key]) == 1
        assert hf3fs_stats.data[duration_key][0] == pytest.approx(0.5)
        assert hf3fs_stats.data["num_transfer_task"] == 1

    @pytest.mark.parametrize(
        "task_type, failed_key",
        [
            ("Saved", "num_failed_save"),
            ("Loaded", "num_failed_load"),
        ],
        ids=["save", "load"],
    )
    def test_record_failed_task(self, hf3fs_stats, task_type, failed_key):
        """Recording a failed task should increment the corresponding counter."""
        hf3fs_stats.record_failed_task_count(task_type)
        assert hf3fs_stats.data[failed_key] == 1
        assert hf3fs_stats.data["num_transfer_task"] == 1

    def test_aggregate_two_stats(self):
        """aggregate() should merge save/load duration lists and sum counters."""
        stats1 = HF3FSKVConnectorStats()
        stats1.record_success_task_duration("Saved", 0.1)
        stats1.record_success_task_duration("Loaded", 0.2)

        stats2 = HF3FSKVConnectorStats()
        stats2.record_success_task_duration("Saved", 0.3)
        stats2.record_failed_task_count("Loaded")

        stats1.aggregate(stats2)
        assert stats1.data["save_duration"] == pytest.approx([0.1, 0.3])
        assert stats1.data["load_duration"] == pytest.approx([0.2])
        assert stats1.data["num_failed_load"] == 1
        assert stats1.data["num_transfer_task"] == 4

    def test_reduce_with_data(self):
        """reduce() computes correct averages when data is present."""
        stats = HF3FSKVConnectorStats()
        stats.record_success_task_duration("Saved", 1.0)
        stats.record_success_task_duration("Saved", 3.0)
        result = stats.reduce()
        assert result["Num save task success"] == pytest.approx(2.0, rel=0.01)
        assert result["Num save task failed"] == pytest.approx(0.0, rel=0.01)
        assert result["Avg save duration (ms)"] == pytest.approx(2000.0, rel=0.01)
        # Reduced values must be plain Python scalars so CLI logging renders
        # them without numpy reprs (eg np.float64(...)).
        assert all(not isinstance(v, np.generic) for v in result.values())

    def test_clone_and_reset(self, hf3fs_stats):
        """clone_and_reset() returns a copy with data and resets the original."""
        hf3fs_stats.record_success_task_duration("Saved", 0.7)
        hf3fs_stats.record_success_task_duration("Loaded", 0.4)

        clone = hf3fs_stats.clone_and_reset()
        assert clone.data["num_transfer_task"] == 2
        assert hf3fs_stats.is_empty()


# ===========================================================================
# Failed-load block ids (issue #69058)
# ===========================================================================

_PAGE_BYTES = 64


class _CpuCudaStream:
    def synchronize(self) -> None:
        return None

    def wait_event(self, event: object) -> None:
        return None


class _CpuCudaEvent:
    def record(self, stream: object = None) -> None:
        return None


def _patch_cpu_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace CUDA stream setup. The load worker calls it even on CPU."""
    monkeypatch.setattr(torch.cuda, "Stream", _CpuCudaStream)
    monkeypatch.setattr(torch.cuda, "Event", _CpuCudaEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: contextlib.nullcontext())
    monkeypatch.setattr(
        torch.accelerator, "set_device_index", lambda *_args, **_kwargs: None
    )


class _MetadataClient:
    def __init__(self, locations: Callable[[list], list]):
        self._locations = locations

    def get_key_locations(self, rank: int, hashes: list) -> list:
        return self._locations(hashes)


class _FileClient:
    def __init__(self, read_result: int | Callable):
        self._read_result = read_result

    def batch_read(self, offsets: list[int], buffers: list) -> list[int]:
        result = self._read_result
        if callable(result):
            return result(offsets, buffers)
        return [result] * len(offsets)


class _LoadServices:
    """External metadata and file clients used by one load manager."""

    def __init__(self, locations: Callable[[list], list], read_result: int | Callable):
        self._device = torch.device("cpu")
        self._dtype = torch.uint8
        self._shape_per_page = [_PAGE_BYTES]
        self._bytes_per_page = _PAGE_BYTES
        self._rank = 0
        self._numjobs = 1
        self._max_device_buffer_count = 4
        self._metadata_client = _MetadataClient(locations)
        self._clients = [_FileClient(read_result)]
        self._ac = AtomicCounter(1)
        self.scattered: list[list[int]] = []

    def _gather_or_scatter_kv_caches(self, block_ids, buffers, operation: str) -> None:
        self.scattered.append(list(block_ids))


class _ReportingConnector(HF3FSKVConnector):
    """Real get_finished / get_block_ids_with_load_errors without storage setup.

    `HF3FSKVConnector.__init__` installs process signal handlers and
    `register_kv_caches` opens the metadata server. Those are outside the
    load-completion path this regression drives.
    """

    def __init__(self, manager: AsyncOperationManager) -> None:
        self._async_manager = manager


def _poll_load(connector: _ReportingConnector) -> tuple[set[str], set[int]]:
    """Worker order: transfer results first, then load-error block ids."""
    results = connector.get_transfer_results(set())
    block_ids = connector.get_block_ids_with_load_errors()
    return results.finished_recving, block_ids


@pytest.mark.cpu_test
class TestHF3FSLoadErrorBlockIds:
    """Failed loads stay in finished_recving and return their block ids."""

    @pytest.fixture
    def open_loader(self, monkeypatch: pytest.MonkeyPatch):
        _patch_cpu_cuda(monkeypatch)
        managers: list[AsyncOperationManager] = []

        def _open(
            locations: Callable[[list], list],
            read_result: int | Callable = _PAGE_BYTES,
        ):
            services = _LoadServices(locations, read_result)
            manager = AsyncOperationManager(services)
            managers.append(manager)
            return _ReportingConnector(manager), services

        yield _open
        for manager in managers:
            manager.shutdown()

    def test_missing_key_reports_block_after_finished(self, open_loader):
        """A missing page is finished receiving and returns block 5."""
        connector, services = open_loader(lambda _hashes: [None])
        block_ids = [5]
        future = connector._async_manager.submit_load_operation("B", block_ids, ["h"])
        block_ids.append(99)
        assert future.result(timeout=10) is False

        finished, errors = _poll_load(connector)
        assert finished == {"B"}
        assert errors == {5}
        assert connector.get_block_ids_with_load_errors() == set()
        assert services.scattered == []

    def test_metadata_exception_reports_block_after_finished(self, open_loader):
        """A raised lookup, as after metadata /clear, still returns block 5."""

        def _raise(_hashes: list) -> list:
            raise ValueError("Rank 0 not initialized")

        connector, services = open_loader(_raise)
        future = connector._async_manager.submit_load_operation("B", [5], ["h"])
        assert future.result(timeout=10) is False

        finished, errors = _poll_load(connector)
        assert finished == {"B"}
        assert errors == {5}
        assert services.scattered == []

    @pytest.mark.parametrize("nbytes", [-1, _PAGE_BYTES - 1])
    def test_short_read_reports_every_block(self, open_loader, nbytes: int):
        """A short or failed read reports each block in that load."""
        connector, services = open_loader(lambda _hashes: [0, 1], nbytes)
        future = connector._async_manager.submit_load_operation(
            "B", [5, 6], ["h0", "h1"]
        )
        assert future.result(timeout=10) is False

        finished, errors = _poll_load(connector)
        assert finished == {"B"}
        assert errors == {5, 6}
        assert services.scattered == []

    def test_successful_load_reports_no_block_errors(self, open_loader):
        """A full-page read is finished receiving and has no load errors."""
        connector, services = open_loader(lambda _hashes: [0])
        future = connector._async_manager.submit_load_operation("B", [7], ["h"])
        assert future.result(timeout=10) is True

        finished, errors = _poll_load(connector)
        assert finished == {"B"}
        assert errors == set()
        assert services.scattered == [[7]]

    def test_only_failed_blocks_in_a_mixed_pass(self, open_loader):
        """One pass can finish a success and a failure; only the failure's ids."""

        def _locations(hashes: list) -> list:
            if hashes == ["missing"]:
                return [None]
            return [0]

        connector, _services = open_loader(_locations)
        manager = connector._async_manager
        failed = manager.submit_load_operation("bad", [5], ["missing"])
        loaded = manager.submit_load_operation("ok", [7], ["present"])
        assert failed.result(timeout=10) is False
        assert loaded.result(timeout=10) is True

        finished, errors = _poll_load(connector)
        assert finished == {"bad", "ok"}
        assert errors == {5}

    def test_errors_appear_on_the_pass_that_finishes_the_load(self, open_loader):
        """An in-flight load contributes neither finished_recving nor errors."""
        entered = threading.Event()
        release = threading.Event()

        def _locations(_hashes: list) -> list:
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("load gate was not released")
            return [None]

        connector, _services = open_loader(_locations)
        future = connector._async_manager.submit_load_operation("B", [5], ["h"])
        try:
            assert entered.wait(timeout=5)
            finished, errors = _poll_load(connector)
            assert finished == set()
            assert errors == set()
            assert not future.done()
        finally:
            release.set()

        assert future.result(timeout=10) is False
        finished, errors = _poll_load(connector)
        assert finished == {"B"}
        assert errors == {5}

    def test_earlier_failed_batch_is_kept_until_later_batch_finishes(self, open_loader):
        """A later success must not drop an earlier batch's failed block ids.

        `start_load_kv` submits one future per device-buffer batch under the
        same request id. The request stays unfinished while any batch is
        running, and the failed ids appear in the pass that finishes it.
        """
        entered = threading.Event()
        release = threading.Event()

        def _locations(hashes: list) -> list:
            if hashes == ["late"]:
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("load gate was not released")
                return [0]
            return [None, None]

        connector, services = open_loader(_locations)
        manager = connector._async_manager
        early = manager.submit_load_operation("B", [5, 6], ["early", "early2"])
        late = manager.submit_load_operation("B", [7], ["late"])
        try:
            assert entered.wait(timeout=5)
            assert early.result(timeout=10) is False
            assert not late.done()
            finished, errors = _poll_load(connector)
            assert finished == set()
            assert errors == set()
        finally:
            release.set()

        assert late.result(timeout=10) is True
        finished, errors = _poll_load(connector)
        assert finished == {"B"}
        assert errors == {5, 6}
        assert connector.get_block_ids_with_load_errors() == set()
        assert services.scattered == [[7]]

    def test_every_failed_batch_is_reported(self, open_loader):
        """Missing key and a later short read both contribute block ids."""

        def _locations(hashes: list) -> list:
            if hashes[0].startswith("late"):
                return [0] * len(hashes)
            return [None]

        connector, services = open_loader(_locations, -1)
        manager = connector._async_manager
        early = manager.submit_load_operation("B", [5], ["early"])
        late = manager.submit_load_operation("B", [8, 9], ["late0", "late1"])
        assert early.result(timeout=10) is False
        assert late.result(timeout=10) is False

        finished, errors = _poll_load(connector)
        assert finished == {"B"}
        assert errors == {5, 8, 9}
        assert services.scattered == []

    def test_successful_batches_finish_together_without_errors(self, open_loader):
        """Every successful batch scatters, and the request finishes once."""
        entered = threading.Event()
        release = threading.Event()

        def _locations(hashes: list) -> list:
            if hashes == ["late"]:
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("load gate was not released")
                return [0]
            return [0, 0]

        connector, services = open_loader(_locations)
        manager = connector._async_manager
        early = manager.submit_load_operation("B", [1, 2], ["early", "early2"])
        late = manager.submit_load_operation("B", [3], ["late"])
        try:
            assert entered.wait(timeout=5)
            assert early.result(timeout=10) is True
            finished, errors = _poll_load(connector)
            assert finished == set()
            assert errors == set()
            assert services.scattered == [[1, 2]]
        finally:
            release.set()

        assert late.result(timeout=10) is True
        finished, errors = _poll_load(connector)
        assert finished == {"B"}
        assert errors == set()
        assert services.scattered == [[1, 2], [3]]

    def test_connector_without_manager_returns_no_errors(self):
        """Scheduler-side construction has no async manager."""

        class _Unstarted(HF3FSKVConnector):
            def __init__(self) -> None:
                pass

        assert _Unstarted().get_block_ids_with_load_errors() == set()


def _publish_blocks(client, hashes: list[str]) -> None:
    """Confirm one page per hash on rank 0 of a live metadata server."""
    client.initialize(0, num_pages=max(8, len(hashes)), role="worker")
    allocated = client.allocate_pages_for_keys(
        0, [(block_hash, "") for block_hash in hashes]
    )
    pairs = [(item[0], item[1]) for item in allocated]
    client.confirm_write_for_keys(0, pairs, [])


def _wait_for_metadata(url: str) -> None:
    import requests

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            requests.post(
                f"{url}/rank/0/initialize",
                json={"role": "scheduler", "num_pages": 0},
                timeout=0.2,
            ).raise_for_status()
            return
        except requests.RequestException:
            time.sleep(0.05)
    raise TimeoutError(f"metadata server at {url} did not accept requests")


@pytest.mark.cpu_test
class TestHF3FSRecomputeDoesNotReload:
    """A failed load is not offered again while the same keys still exist."""

    @pytest.fixture
    def metadata_url(self):
        import uvicorn

        from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_metadata_server import (  # noqa: E501
            Hf3fsMetadataServer,
        )

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(
            uvicorn.Config(
                Hf3fsMetadataServer().app,
                host="127.0.0.1",
                port=port,
                log_level="error",
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            _wait_for_metadata(url)
            yield url
        finally:
            server.should_exit = True
            thread.join(timeout=5)

    @pytest.mark.parametrize("policy", ["fail", "recompute"])
    def test_scheduler_policy_uses_one_failed_worker_load(
        self, metadata_url: str, policy: str, monkeypatch: pytest.MonkeyPatch
    ):
        """Fail ends the request; recompute runs the prompt locally once.

        The worker load goes through AsyncOperationManager against the same
        metadata server the scheduler connector used. Keys stay published, so
        a second offer would be another remote load. This environment cannot
        construct Scheduler via ModelConfig (the model stack needs vllm._C),
        so the test calls Scheduler's real failure methods on that connector.
        """
        from dataclasses import dataclass

        from vllm.config import CacheConfig, KVTransferConfig
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks
        from vllm.v1.core.kv_cache_utils import KVCacheBlock
        from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.outputs import KVConnectorOutput
        from vllm.v1.request import Request, RequestStatus

        @dataclass
        class _ModelConfig:
            use_mla: bool = False

        @dataclass
        class _VllmConfig:
            cache_config: CacheConfig
            model_config: _ModelConfig
            kv_transfer_config: KVTransferConfig

        class _AllocatedBlocks:
            def __init__(self, block_ids: list[int]):
                self._block_ids = block_ids
                self.freed = False

            def get_block_ids(self, request_id: str) -> tuple[list[int]]:
                return (self._block_ids,)

            def free(self, request: Request) -> None:
                self.freed = True

            def cache_blocks(self, request: Request, num_tokens: int) -> None:
                raise AssertionError(num_tokens)

        _patch_cpu_cuda(monkeypatch)
        saved_signals = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)
        }
        vllm_config = _VllmConfig(
            cache_config=CacheConfig(block_size=16),
            model_config=_ModelConfig(),
            kv_transfer_config=KVTransferConfig(
                kv_connector="HF3FSKVConnector",
                kv_role="kv_both",
                kv_load_failure_policy=policy,  # type: ignore[arg-type]
                kv_connector_extra_config={
                    "hf3fs_metadata_server_url": metadata_url,
                },
            ),
        )
        connector = HF3FSKVConnector(
            vllm_config, KVConnectorRole.SCHEDULER, _ModelConfig()
        )
        managers: list[AsyncOperationManager] = []
        try:
            prompt = list(range(65))
            request = Request(
                request_id="B",
                prompt_token_ids=prompt,
                sampling_params=SamplingParams(max_tokens=1),
                pooling_params=None,
            )
            checked = connector._align_to_block_size(len(prompt) - 1)
            hashes = connector._generate_block_hashes(prompt[:checked], 0)
            _publish_blocks(connector._metadata_client, hashes)
            hit, async_load = connector.get_num_new_matched_tokens(request, 0)
            assert (hit, async_load) == (checked, True)

            block_ids = [3, 5, 7, 9]
            connector.update_state_after_alloc(
                request,
                KVCacheBlocks((tuple(KVCacheBlock(block_id=i) for i in block_ids),)),
                hit,
            )
            request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            request.num_computed_tokens = checked
            # The scheduler step sends the load to the worker here.
            connector.build_connector_meta(SchedulerOutput.make_empty())

            services = _LoadServices(
                lambda found: connector._metadata_client.get_key_locations(0, found),
                -1,
            )
            manager = AsyncOperationManager(services)
            managers.append(manager)
            worker = _ReportingConnector(manager)
            future = manager.submit_load_operation(
                request.request_id, block_ids, hashes
            )
            assert future.result(timeout=10) is False
            finished, errors = _poll_load(worker)
            assert finished == {request.request_id}
            assert errors == set(block_ids)

            blocks = _AllocatedBlocks(block_ids)
            scheduler = type(
                "_FailureScheduler",
                (),
                {
                    "block_size": 16,
                    "recompute_kv_load_failures": policy == "recompute",
                    "kv_cache_config": type(
                        "_Groups", (), {"kv_cache_groups": (object(),)}
                    )(),
                    "kv_holding_waiting": [request],
                    "running": [],
                    "kv_cache_manager": blocks,
                    "failed_recving_kv_req_ids": set(),
                    "finished_recving_kv_req_ids": set(),
                    "connector": connector,
                    "requests": {request.request_id: request},
                    "_kv_fetch_stages": None,
                    "prefix_replay_tokens": 0,
                },
            )()
            for name in (
                "_set_kv_fetch_stage",
                "_update_requests_with_invalid_blocks",
                "_update_from_kv_xfer_finished",
                "_update_waiting_for_remote_kv",
                "_mark_prefix_replay",
            ):
                setattr(scheduler, name, getattr(Scheduler, name).__get__(scheduler))
            output = KVConnectorOutput(
                finished_recving={request.request_id},
                invalid_block_ids=errors,
            )
            failed_ids = Scheduler._handle_invalid_blocks(scheduler, errors, {})
            Scheduler._update_from_kv_xfer_finished(scheduler, output)
            assert connector._scheduling_states[request.request_id].load_failed
            assert connector._metadata_client.batch_key_exists(hashes) == [True] * len(
                hashes
            )

            again, again_async = connector.get_num_new_matched_tokens(request, 0)
            assert (again, again_async) == (0, False)

            if policy == "fail":
                assert failed_ids == {request.request_id}
                assert request.num_computed_tokens == 0
                return

            assert failed_ids == set()
            assert request.num_computed_tokens == 0
            assert request.request_id in scheduler.failed_recving_kv_req_ids
            Scheduler._update_waiting_for_remote_kv(scheduler, request)
            assert blocks.freed
            assert request.num_computed_tokens == 0
            request.status = RequestStatus.WAITING
            resumed, resumed_async = connector.get_num_new_matched_tokens(request, 0)
            assert (resumed, resumed_async) == (0, False)
            assert request.status != RequestStatus.WAITING_FOR_REMOTE_KVS
        finally:
            for manager in managers:
                manager.shutdown()
            connector.close()
            for signum, handler in saved_signals.items():
                signal.signal(signum, handler)
