# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HF3FS KV Connector high-level components:
- TestHf3fsMockClient      : file-backed mock client I/O correctness
- TestHF3FSKVConnectorStats: metric collection, aggregation, serialisation
- TestHF3FSExternalCacheKeys: keys separate what the engine's block hashes separate
"""

import os
import socket
import subprocess
import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config, make_kv_cache_config
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs import hf3fs_connector
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    HF3FSKVConnector,
    HF3FSKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.common import (
    HF3FSRequestMetadata,
    RequestSchedulingState,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.hf3fs_mock_client import (
    Hf3fsClient as MockHf3fsClient,
)
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

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
# TestHF3FSExternalCacheKeys
# ===========================================================================

BLOCK_SIZE = 16
# The lookup covers the full blocks of len(prompt) - 1, so two of these three.
TOKENS = list(range(1000, 1000 + BLOCK_SIZE * 3))


def _make_request(
    request_id: str,
    token_ids: list[int] = TOKENS,
    mm_hash: str | None = None,
    cache_salt: str | None = None,
    lora_request: LoRARequest | None = None,
) -> Request:
    mm_features = mm_hash and [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            mm_position=PlaceholderRange(offset=0, length=len(token_ids)),
            identifier=mm_hash,
            modality="image",
        )
    ]
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(token_ids),
        mm_features=mm_features or None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=lora_request,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _worker_keys(request: Request, block_size: int) -> list[str]:
    state = RequestSchedulingState(request_id=request.request_id, request=request)
    state.token_ids = list(request.prompt_token_ids)
    state.allocated_block_ids = list(range(len(state.token_ids) // block_size))
    metadata = HF3FSRequestMetadata.from_scheduling_state(state, block_size, BLOCK_SIZE)
    assert metadata is not None
    return metadata.block_keys


@pytest.fixture
def lookup_keys(monkeypatch):
    """Keys a scheduler-side connector asks the external metadata service about."""
    metadata_service = MagicMock()
    metadata_service.batch_key_exists.side_effect = lambda keys: [False] * len(keys)
    monkeypatch.setattr(
        hf3fs_connector, "Hf3fsMetadataClient", lambda: metadata_service
    )
    monkeypatch.setattr(hf3fs_connector.atexit, "register", lambda *_: None)
    monkeypatch.setattr(hf3fs_connector.signal, "signal", lambda *_: None)
    connector = HF3FSKVConnector(
        create_vllm_config(block_size=BLOCK_SIZE, kv_connector="HF3FSKVConnector"),
        KVConnectorRole.SCHEDULER,
        make_kv_cache_config(block_size=BLOCK_SIZE),
    )

    def lookup(request: Request) -> list[str]:
        connector.get_num_new_matched_tokens(request, 0)
        return metadata_service.batch_key_exists.call_args.args[0]

    return lookup


class TestHF3FSExternalCacheKeys:
    @pytest.fixture(autouse=True)
    def _none_hash(self):
        init_none_hash(sha256)

    @pytest.mark.parametrize(
        "kwargs_a, kwargs_b",
        [
            ({"mm_hash": "image-a"}, {"mm_hash": "image-b"}),
            (
                {"lora_request": LoRARequest("adapter-a", 1, "/tmp/a")},
                {"lora_request": LoRARequest("adapter-b", 2, "/tmp/b")},
            ),
            ({"cache_salt": "tenant-1"}, {"cache_salt": "tenant-2"}),
        ],
        ids=["multimodal", "lora", "cache_salt"],
    )
    def test_lookup_keys_partition_requests(self, lookup_keys, kwargs_a, kwargs_b):
        keys_a = lookup_keys(_make_request("a", **kwargs_a))
        keys_b = lookup_keys(_make_request("b", **kwargs_b))
        assert len(keys_a) == len(keys_b) == 2
        assert keys_a != keys_b

    def test_lookup_keys_chain_on_the_prefix(self, lookup_keys):
        other_tail = TOKENS[:BLOCK_SIZE] + [t + 1 for t in TOKENS[BLOCK_SIZE:]]
        keys = lookup_keys(_make_request("a"))
        assert keys == lookup_keys(_make_request("b"))
        other_keys = lookup_keys(_make_request("c", token_ids=other_tail))
        assert other_keys[0] == keys[0]
        assert other_keys[1] != keys[1]

    def test_worker_keys_extend_lookup_keys(self, lookup_keys):
        request = _make_request("a", cache_salt="tenant-1")
        looked_up = lookup_keys(request)
        assert _worker_keys(request, BLOCK_SIZE)[: len(looked_up)] == looked_up

    def test_worker_keys_use_the_scheduler_block_size(self):
        request = _make_request("a", token_ids=list(range(BLOCK_SIZE * 4)))
        fine = _worker_keys(request, BLOCK_SIZE)
        assert _worker_keys(request, BLOCK_SIZE * 2) == [fine[1], fine[3]]


@pytest.fixture
def hf3fs_metadata_server():
    server_code = (
        "from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs."
        "hf3fs_metadata_server import Hf3fsMetadataServer; "
        "Hf3fsMetadataServer().run(host='127.0.0.1', port=18000)"
    )
    process = subprocess.Popen([sys.executable, "-c", server_code])
    deadline = time.monotonic() + 60
    while True:
        try:
            socket.create_connection(("127.0.0.1", 18000), timeout=1).close()
            break
        except OSError:
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.2)
    yield
    process.terminate()
    process.wait(timeout=30)


def test_external_cache_is_not_shared_across_cache_salts(
    vllm_runner, tmp_path, hf3fs_metadata_server
):
    prompt_token_ids = list(range(1000, 1000 + BLOCK_SIZE * 8 + 1))
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    with vllm_runner(
        "facebook/opt-125m",
        block_size=BLOCK_SIZE,
        enable_prefix_caching=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="HF3FSKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "hf3fs_storage_path": str(tmp_path),
                "hf3fs_client_numjobs": 2,
            },
        ),
    ) as runner:

        def cached_tokens(cache_salt: str) -> int:
            output = runner.llm.generate(
                [{"prompt_token_ids": prompt_token_ids, "cache_salt": cache_salt}],
                sampling_params,
                use_tqdm=False,
            )[0]
            return output.num_cached_tokens

        assert cached_tokens("tenant-1") == 0
        expected = BLOCK_SIZE * 8
        deadline = time.monotonic() + 60
        while (cached := cached_tokens("tenant-1")) != expected:
            assert time.monotonic() < deadline, f"cached {cached} of {expected} tokens"
            time.sleep(0.5)
        assert cached_tokens("tenant-2") == 0
