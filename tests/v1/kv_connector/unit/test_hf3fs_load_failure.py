# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF3FS load failures through the scheduler, and through `LLM` on a GPU."""

import atexit
import gc
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs import hf3fs_connector
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    DEFAULT_MAX_IO_ENTRIES,
    HF3FSKVConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_metadata_server import (
    GlobalMetadataState,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.hf3fs_mock_client import (
    Hf3fsClient as MockHf3fsClient,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from .utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

BLOCK_SIZE = 16
NUM_HEADS = 1
HEAD_DIM = 4
NUM_GPU_BLOCKS = DEFAULT_MAX_IO_ENTRIES + 2
# One float32 layer per page.
PAGE_BYTES = NUM_HEADS * BLOCK_SIZE * HEAD_DIM * 4
NUM_FILE_PAGES = 16


class FakeMetadataService:
    """Stands in for the external HF3FS metadata server."""

    def __init__(self):
        self.get_key_locations_replies: list = []
        # Used once the queued replies run out.
        self.default_reply = None
        self.num_location_lookups = 0

    def initialize(self, *args, **kwargs):
        """Accept the scheduler and worker registrations."""

    def batch_key_exists(self, keys):
        """Report every key as stored."""
        return [True] * len(keys)

    def get_key_locations(self, rank, keys):
        """Serve the next queued reply, or the default reply."""
        self.num_location_lookups += 1
        if self.get_key_locations_replies:
            return self.get_key_locations_replies.pop(0)(keys)
        return self.default_reply(keys)


def _hf3fs_extra_config(tmp_path) -> dict:
    return {
        "hf3fs_storage_path": str(tmp_path),
        "hf3fs_file_size": PAGE_BYTES * NUM_FILE_PAGES,
        "hf3fs_client_numjobs": 1,
        "hf3fs_max_device_buffer_count": DEFAULT_MAX_IO_ENTRIES,
    }


@pytest.fixture
def storage_services(monkeypatch):
    """Replace only the external metadata service and the 3FS file system."""
    metadata = FakeMetadataService()
    monkeypatch.setattr(
        hf3fs_connector, "Hf3fsMetadataClient", lambda *_a, **_k: metadata
    )
    monkeypatch.setattr(hf3fs_connector, "Hf3fsClient", MockHf3fsClient)
    if torch.cuda.is_available():
        return metadata, "cuda"
    # The load worker needs CUDA streams and an accelerator device.
    # On CPU they are no-ops, and a load fails before it scatters.
    monkeypatch.setattr(torch.cuda, "Stream", MagicMock)
    monkeypatch.setattr(torch.cuda, "Event", MagicMock)
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda _: None)
    return metadata, "cpu"


@pytest.fixture
def make_worker_connector(storage_services, dist_init):
    """Build worker-side connectors with real queues, buffers and file I/O."""
    _, device = storage_services
    connectors = []

    def make(config, kv_cache_config, num_blocks):
        with patch("signal.signal"):
            connector = HF3FSKVConnector(
                config, KVConnectorRole.WORKER, kv_cache_config
            )
        connectors.append(connector)
        kv_cache = torch.zeros(
            num_blocks, NUM_HEADS, BLOCK_SIZE, HEAD_DIM, device=device
        )
        connector.register_kv_caches({"layer.0": kv_cache})
        return connector

    yield make
    for connector in connectors:
        connector.close()
        atexit.unregister(connector.close)


def _wait_for_finished_recving(connector, request_id, timeout=10.0):
    """Poll get_finished until it reports the request as received."""
    deadline = time.monotonic() + timeout
    while True:
        _, finished_recving = connector.get_finished(set())
        if request_id in finished_recving:
            return
        assert time.monotonic() < deadline, f"load of {request_id} never finished"
        time.sleep(0.01)


NUM_SCHEDULER_BLOCKS = 64


def _hf3fs_scheduler(tmp_path, policy):
    config = create_vllm_config(
        kv_connector="HF3FSKVConnector",
        kv_role="kv_both",
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=256,
        kv_load_failure_policy=policy,
        kv_connector_extra_config=_hf3fs_extra_config(tmp_path),
    )
    with patch("signal.signal"):
        scheduler = create_scheduler(config, num_blocks=NUM_SCHEDULER_BLOCKS)
    atexit.unregister(scheduler.connector.close)
    return config, scheduler


def _run_worker(worker, scheduler_output, wait_for_recving=None):
    """Run one worker pass and return its output for the scheduler."""
    worker.bind_connector_metadata(scheduler_output.kv_connector_metadata)
    worker.start_load_kv(None)
    if wait_for_recving is not None:
        _wait_for_finished_recving(worker, wait_for_recving)
    finished_sending, finished_recving = worker.get_finished(
        scheduler_output.finished_req_ids
    )
    if wait_for_recving is not None:
        finished_recving.add(wait_for_recving)
    invalid_block_ids = worker.get_block_ids_with_load_errors()
    worker.clear_connector_metadata()
    return create_model_runner_output(
        [],
        finished_sending=finished_sending,
        finished_recving=finished_recving,
        invalid_block_ids=invalid_block_ids,
    )


class TestHF3FSSchedulerLoadFailure:
    """The scheduler sees a failed HF3FS load through the real connectors."""

    @pytest.mark.parametrize("policy", ["fail", "recompute"])
    def test_failed_load_honors_failure_policy(
        self, tmp_path, storage_services, make_worker_connector, policy
    ):
        metadata, _ = storage_services
        # Every key is reported as stored, but every read comes back short.
        metadata.default_reply = lambda keys: [NUM_FILE_PAGES] * len(keys)
        config, scheduler = _hf3fs_scheduler(tmp_path, policy)
        worker = make_worker_connector(
            config, scheduler.kv_cache_config, NUM_SCHEDULER_BLOCKS
        )

        num_prompt_tokens = 4 * BLOCK_SIZE + 1
        request = create_request(num_tokens=num_prompt_tokens, block_size=BLOCK_SIZE)
        scheduler.add_request(request)
        scheduler_output = scheduler.schedule()
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        model_runner_output = _run_worker(
            worker, scheduler_output, wait_for_recving=request.request_id
        )
        scheduler.update_from_output(scheduler_output, model_runner_output)

        if policy == "fail":
            assert request.status == RequestStatus.FINISHED_ERROR
            # The next pass must not release the request a second time.
            scheduler_output = scheduler.schedule()
            model_runner_output = _run_worker(worker, scheduler_output)
            scheduler.update_from_output(scheduler_output, model_runner_output)
            assert_scheduler_empty(scheduler)
            return

        # The rerun recomputes the whole prompt instead of loading it again.
        scheduler_output = scheduler.schedule()
        assert request.status == RequestStatus.RUNNING
        assert scheduler_output.num_scheduled_tokens == {
            request.request_id: num_prompt_tokens
        }
        assert metadata.num_location_lookups == 1

    def test_failed_later_batch_keeps_prefix_and_saves_the_rest(
        self, tmp_path, storage_services
    ):
        # A successful batch needs a GPU, so the worker result is written here.
        _, scheduler = _hf3fs_scheduler(tmp_path, "recompute")
        num_loaded_blocks = DEFAULT_MAX_IO_ENTRIES + 2
        num_prompt_tokens = num_loaded_blocks * BLOCK_SIZE + 1
        request = create_request(num_tokens=num_prompt_tokens, block_size=BLOCK_SIZE)
        scheduler.add_request(request)
        scheduler_output = scheduler.schedule()
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        (load,) = scheduler_output.kv_connector_metadata.requests
        block_ids = load.block_ids[:num_loaded_blocks]

        # The first batch loaded, and the second batch failed.
        model_runner_output = create_model_runner_output(
            [],
            finished_recving={request.request_id},
            invalid_block_ids=set(block_ids[DEFAULT_MAX_IO_ENTRIES:]),
        )
        scheduler.update_from_output(scheduler_output, model_runner_output)

        scheduler_output = scheduler.schedule()
        assert request.status == RequestStatus.RUNNING
        num_valid_tokens = DEFAULT_MAX_IO_ENTRIES * BLOCK_SIZE
        assert scheduler_output.num_scheduled_tokens == {
            request.request_id: num_prompt_tokens - num_valid_tokens
        }
        # Saves start after the kept prefix, so the failed entries are replaced.
        (save,) = scheduler_output.kv_connector_metadata.requests
        assert save.save_block_op is not None
        assert save.save_block_op.skip_leading_blocks == DEFAULT_MAX_IO_ENTRIES

    def test_reused_blocks_do_not_mark_a_finished_load(
        self, tmp_path, storage_services
    ):
        _, scheduler = _hf3fs_scheduler(tmp_path, "recompute")
        connector = scheduler.connector
        num_hit_tokens = 4 * BLOCK_SIZE
        block_ids = [1, 2, 3, 4]

        def start_load(request):
            assert connector.get_num_new_matched_tokens(request, 0) == (
                num_hit_tokens,
                True,
            )
            blocks = KVCacheBlocks(blocks=([KVCacheBlock(i) for i in block_ids],))
            connector.update_state_after_alloc(request, blocks, num_hit_tokens)
            connector.build_connector_meta(SchedulerOutput.make_empty())

        request_a = create_request(num_tokens=num_hit_tokens + 1)
        request_b = create_request(num_tokens=num_hit_tokens + 1)
        start_load(request_a)
        connector.update_connector_output(
            KVConnectorOutput(finished_recving={request_a.request_id})
        )
        # A was preempted, and B now loads into the blocks A released.
        start_load(request_b)
        connector.update_connector_output(
            KVConnectorOutput(
                finished_recving={request_b.request_id},
                invalid_block_ids=set(block_ids),
            )
        )

        assert connector.get_num_new_matched_tokens(request_b, 0) == (0, False)
        assert connector.get_num_new_matched_tokens(request_a, 0) == (
            num_hit_tokens,
            True,
        )


# ===========================================================================
# TestHF3FSLoadFailureEndToEnd
# ===========================================================================

E2E_MODEL = "facebook/opt-125m"
E2E_MAX_TOKENS = 16
# The longest prompt is about 320 tokens.
E2E_MAX_MODEL_LEN = 512
E2E_NUM_GPU_BLOCKS = 64


class InProcessMetadataService:
    """The in-tree metadata server state, served in process instead of over HTTP."""

    def __init__(self):
        self.state = GlobalMetadataState()

    def initialize(self, rank, num_pages=0, role="worker"):
        if role == "worker":
            self.state.initialize_rank(rank, num_pages)

    def allocate_pages_for_keys(self, rank, keys):
        return list(self.state.allocate_pages_for_keys(rank, keys).items())

    def confirm_write_for_keys(self, rank, key_confirmations, pages_to_release=None):
        self.state.confirm_write_for_keys(rank, key_confirmations, pages_to_release)

    def batch_key_exists(self, keys):
        return self.state.batch_key_exists(keys)

    def get_key_locations(self, rank, keys):
        return self.state.get_key_locations(rank, keys)

    def num_stored_keys(self) -> int:
        with self.state.global_lock:
            return sum(m.is_complete() for m in self.state.key_metadata.values())


class StorageFaults:
    """Switches the file-backed 3FS client to failing every read."""

    def __init__(self):
        self.fail_reads = False
        self.num_failed_reads = 0


@pytest.fixture
def in_process_storage(monkeypatch):
    """Replace only the external metadata server and the 3FS file system."""
    # The engine and its worker must share this process to share the fakes.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    metadata = InProcessMetadataService()
    faults = StorageFaults()

    class FaultyFileClient(MockHf3fsClient):
        def batch_read(self, offsets, tensors):
            if faults.fail_reads:
                faults.num_failed_reads += 1
                return [-1] * len(offsets)
            return super().batch_read(offsets, tensors)

    monkeypatch.setattr(
        hf3fs_connector, "Hf3fsMetadataClient", lambda *_a, **_k: metadata
    )
    monkeypatch.setattr(hf3fs_connector, "Hf3fsClient", FaultyFileClient)
    return metadata, faults


def _e2e_prompts() -> tuple[str, list[str]]:
    target = " ".join(f"Item {i} costs {i * 7} dollars." for i in range(40))
    # Together the fillers need more blocks than the GPU has.
    # They evict the target from the local prefix cache and overwrite its blocks.
    fillers = [
        " ".join(f"Filler {j} line {i} says hello." for i in range(40))
        for j in range(6)
    ]
    return target, fillers


def _greedy_output(llm, prompt):
    params = SamplingParams(temperature=0, max_tokens=E2E_MAX_TOKENS, logprobs=5)
    return llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]


def _wait_until(condition, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met before the timeout"
        time.sleep(0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
class TestHF3FSLoadFailureEndToEnd:
    """A failed HF3FS read must not turn into a cache hit through `LLM`."""

    @pytest.mark.parametrize("policy", ["fail", "recompute"])
    def test_failed_read_honors_policy(self, tmp_path, in_process_storage, policy):
        metadata, faults = in_process_storage
        target, fillers = _e2e_prompts()
        transfer_config = KVTransferConfig(
            kv_connector="HF3FSKVConnector",
            kv_role="kv_both",
            kv_load_failure_policy=policy,
            kv_connector_extra_config={
                "hf3fs_storage_path": str(tmp_path),
                "hf3fs_file_size": 256 * 1024 * 1024,
                "hf3fs_client_numjobs": 2,
            },
        )
        # One in-process engine per test, torn down like the other in-process tests.
        gc.collect()
        torch.accelerator.empty_cache()
        with patch("signal.signal"):
            llm = LLM(
                model=E2E_MODEL,
                enforce_eager=True,
                gpu_memory_utilization=0.3,
                max_model_len=E2E_MAX_MODEL_LEN,
                block_size=BLOCK_SIZE,
                num_gpu_blocks_override=E2E_NUM_GPU_BLOCKS,
                enable_prefix_caching=True,
                kv_transfer_config=transfer_config,
            )
        connectors = [
            llm.llm_engine.engine_core.engine_core.scheduler.connector,
            get_kv_transfer_group(),
        ]
        try:
            # No key is stored yet, so this run computes the whole prompt.
            # It is the reference the recomputed run must match.
            cold = _greedy_output(llm, target)
            num_target_blocks = len(llm.get_tokenizer().encode(target)) // BLOCK_SIZE
            _wait_until(lambda: metadata.num_stored_keys() >= num_target_blocks)
            llm.generate(
                fillers,
                SamplingParams(temperature=0, max_tokens=1),
                use_tqdm=False,
            )

            faults.fail_reads = True
            retry = _greedy_output(llm, target)
        finally:
            # The connectors' exit hooks would keep their KV cache references alive.
            for connector in connectors:
                atexit.unregister(connector.close)
            llm.llm_engine.engine_core.shutdown()
            for connector in connectors:
                connector.close()
            del llm, connectors
            gc.collect()
            torch.accelerator.empty_cache()

        # The target came back from HF3FS, and that read failed.
        assert faults.num_failed_reads > 0
        if policy == "fail":
            assert retry.finish_reason == "error"
            return
        assert retry.finish_reason == "length"
        # Same engine, same single-request prefill, so the tokens must be equal.
        assert retry.token_ids == cold.token_ids
