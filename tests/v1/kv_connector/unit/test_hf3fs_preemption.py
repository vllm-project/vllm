# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF3FS saves of a preempted request must not read blocks another request reuses."""

import atexit
import functools
import gc
import os
import threading
import time
from unittest.mock import patch

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs import hf3fs_connector
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    AsyncOperationManager,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_metadata_server import (
    GlobalMetadataState,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.hf3fs_mock_client import (
    Hf3fsClient as MockHf3fsClient,
)

# ===========================================================================
# GPU: a held save of a preempted request, then a request that reuses its blocks
# ===========================================================================

E2E_MODEL = "facebook/opt-125m"
BLOCK_SIZE = 16
# Each prompt is about 16 blocks, and each request grows to about 46 of the 48 blocks.
# So decoding preempts one of them, and the other then takes most of its blocks.
NUM_GPU_BLOCKS = 48
# The startup check needs one request of this length to fit in the pool.
MAX_MODEL_LEN = 736
DECODE_TOKENS = 480
# A watchdog only: the test releases the hold itself, and fails if this expires.
HOLD_TIMEOUT_S = 120.0
# After release, the held save waits this long for the fixed wait to return first.
# A wait that returns before the gather is then caught, since nothing has gathered.
GATHER_DELAY_S = 2.0


class InProcessMetadataService:
    """The in-tree metadata server state, served in process instead of over HTTP."""

    def __init__(self):
        self.state = GlobalMetadataState()
        self.num_pages: dict[int, int] = {}

    def initialize(self, rank, num_pages=0, role="worker"):
        if role == "worker":
            self.num_pages[rank] = num_pages
            self.state.initialize_rank(rank, num_pages)

    def reset(self):
        """Forget every stored key, as if the storage were empty."""
        self.state = GlobalMetadataState()
        for rank, num_pages in self.num_pages.items():
            self.state.initialize_rank(rank, num_pages)

    def allocate_pages_for_keys(self, rank, keys):
        return list(self.state.allocate_pages_for_keys(rank, keys).items())

    def confirm_write_for_keys(self, rank, key_confirmations, pages_to_release=None):
        self.state.confirm_write_for_keys(rank, key_confirmations, pages_to_release)

    def batch_key_exists(self, keys):
        return self.state.batch_key_exists(keys)

    def get_key_locations(self, rank, keys):
        return self.state.get_key_locations(rank, keys)


@pytest.fixture
def in_process_metadata(monkeypatch):
    # The engine and its worker must share this process to share the fake.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    metadata = InProcessMetadataService()
    monkeypatch.setattr(
        hf3fs_connector, "Hf3fsMetadataClient", lambda *_a, **_k: metadata
    )
    monkeypatch.setattr(hf3fs_connector, "Hf3fsClient", MockHf3fsClient)
    return metadata


class SaveHold:
    """Holds the first save that covers the target's blocks until released."""

    def __init__(self, target_hashes: set[str]):
        self.target_hashes = target_hashes
        self.armed = False
        self.held_block_ids: set[int] = set()
        self.release = threading.Event()
        self.timed_out = False
        # True while the held save is waiting, before it reads the blocks.
        self.holding = False
        self.entered = threading.Event()
        self.wait_returned = threading.Event()
        # Read right after the held save, before a later save can rewrite the pages.
        self.after_held = None
        self.snapshot = None
        # Save tasks taken off the queue and not finished yet.
        self.in_flight = 0
        self.lock = threading.Lock()

    def wrap(self, handle_save_task):
        hold = self

        def held_save(manager, task):
            _, block_ids, block_hashes, *_ = task
            with hold.lock:
                hold.in_flight += 1
            try:
                if hold.armed and hold.target_hashes & set(block_hashes):
                    hold.armed = False
                    hold.held_block_ids = set(block_ids)
                    hold.holding = True
                    hold.entered.set()
                    hold.timed_out = not hold.release.wait(HOLD_TIMEOUT_S)
                    hold.holding = False
                    hold.wait_returned.wait(GATHER_DELAY_S)
                    result = handle_save_task(manager, task)
                    hold.snapshot = hold.after_held()
                    return result
                return handle_save_task(manager, task)
            finally:
                with hold.lock:
                    hold.in_flight -= 1

        return held_save


def _prompt(tag: str) -> str:
    # About 253 tokens with the OPT tokenizer.
    return " ".join(f"{tag} item {i} costs {i * 7} dollars." for i in range(36))


def _wait_for_saves(
    manager: AsyncOperationManager, hold: "SaveHold", timeout: float = 120.0
):
    """Poll until no save is queued or running."""
    deadline = time.monotonic() + timeout

    def idle() -> bool:
        pending = [f for fs in manager._save_futures.values() for f in fs]
        return (
            manager._save_queue.empty()
            and hold.in_flight == 0
            and all(f.done() for f in pending)
        )

    while not idle():
        assert time.monotonic() < deadline, "saves did not finish"
        time.sleep(0.05)


def _reset_prefix_cache(llm, max_steps: int = 50) -> None:
    """Step the engine until the finished request's blocks are freed, then reset.

    HF3FS frees a finished request's blocks only once its saves are reported done.
    That report comes back on a later engine step.
    The scheduler still steps while it has finished request IDs to send.
    """
    for _ in range(max_steps):
        if llm.reset_prefix_cache():
            return
        llm.llm_engine.step()
        time.sleep(0.05)
    pytest.fail("the local prefix cache could not be reset")


def _read_pages(worker, metadata, hashes) -> list[torch.Tensor]:
    pages = metadata.get_key_locations(0, hashes)
    assert all(page is not None for page in pages), "a target block was not stored"
    path = os.path.join(worker._storage_path, "hf3fs_vllm_data_file_0")
    out = []
    with open(path, "rb") as f:
        for page in pages:
            f.seek(page * worker._bytes_per_page)
            raw = bytearray(f.read(worker._bytes_per_page))
            out.append(torch.frombuffer(raw, dtype=worker._dtype).float())
    return out


def _hf3fs_llm(tmp_path) -> LLM:
    transfer_config = KVTransferConfig(
        kv_connector="HF3FSKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "hf3fs_storage_path": str(tmp_path),
            "hf3fs_file_size": 256 * 1024 * 1024,
            "hf3fs_client_numjobs": 2,
        },
    )
    gc.collect()
    torch.accelerator.empty_cache()
    with patch("signal.signal"):
        return LLM(
            model=E2E_MODEL,
            enforce_eager=True,
            gpu_memory_utilization=0.3,
            max_model_len=MAX_MODEL_LEN,
            block_size=BLOCK_SIZE,
            num_gpu_blocks_override=NUM_GPU_BLOCKS,
            enable_prefix_caching=True,
            kv_transfer_config=transfer_config,
        )


def _shutdown(llm: LLM, connectors: list) -> None:
    for connector in connectors:
        atexit.unregister(connector.close)
    llm.llm_engine.engine_core.shutdown()
    for connector in connectors:
        connector.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
class TestHF3FSPreemptionEndToEnd:
    def test_preempted_request_saves_its_own_kv(self, tmp_path, in_process_metadata):
        metadata = in_process_metadata
        llm = _hf3fs_llm(tmp_path)
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        worker = get_kv_transfer_group()
        manager = worker._async_manager
        connectors = [scheduler.connector, worker]
        target, other = _prompt("Target"), _prompt("Other")
        target_ids = llm.get_tokenizer().encode(target)
        num_blocks = len(target_ids) // BLOCK_SIZE
        target_hashes = scheduler.connector._generate_block_hashes(
            target_ids[: num_blocks * BLOCK_SIZE], 0
        )
        hold = SaveHold(set(target_hashes))
        hold.after_held = functools.partial(
            _read_pages, worker, metadata, target_hashes
        )
        seen = {
            "preempted": False,
            "target_id": None,
            "reused": set(),
            "gathered_when_wait_returned": None,
        }
        # Every save goes through the hold, which only holds once it is armed.
        save_patch = patch.object(
            AsyncOperationManager,
            "_handle_save_task",
            hold.wrap(AsyncOperationManager._handle_save_task),
        )
        save_patch.start()
        try:
            # Reference pages: the target alone, nothing else in the batch.
            llm.generate(
                [target], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False
            )
            _wait_for_saves(manager, hold)
            reference = _read_pages(worker, metadata, target_hashes)

            # Start again from empty storage and an empty local prefix cache.
            metadata.reset()
            _reset_prefix_cache(llm)

            preempt = scheduler._preempt_request
            update = scheduler.update_from_output

            def record_preempt(request, *args, **kwargs):
                # Only a preemption while the save is held exercises the race.
                if list(request.prompt_token_ids) == target_ids:
                    # Preempt only once the save is held, so the race is set up.
                    hold.entered.wait(HOLD_TIMEOUT_S)
                    if hold.holding:
                        seen["preempted"] = True
                        seen["target_id"] = request.request_id
                return preempt(request, *args, **kwargs)

            def release_after_reuse(*args, **kwargs):
                result = update(*args, **kwargs)
                for request in scheduler.running:
                    if request is None or request.request_id == seen["target_id"]:
                        continue
                    ids = scheduler.kv_cache_manager.get_block_ids(request.request_id)
                    if seen["preempted"]:
                        seen["reused"] |= set(ids[0]) & hold.held_block_ids
                # Without the fix, another request's forward has now written
                # into the held blocks, so let the save gather them.
                if seen["reused"]:
                    hold.release.set()
                return result

            scheduler._preempt_request = record_preempt
            scheduler.update_from_output = release_after_reuse
            wait_for_gathers = getattr(manager, "wait_for_gathers", None)
            save_gathers = getattr(manager, "_save_gathers", None)
            if wait_for_gathers is not None:

                def release_on_wait(request_ids):
                    # With the fix, the preempting step waits here for the gather.
                    if seen["target_id"] not in request_ids:
                        return wait_for_gathers(request_ids)
                    hold.release.set()
                    wait_for_gathers(request_ids)
                    # The held save waits for this flag before it gathers.
                    # So only a wait that covers the gather sees it done here.
                    seen["gathered_when_wait_returned"] = all(
                        gather.enqueued.is_set()
                        and (gather.cuda_event is None or gather.cuda_event.query())
                        for gather in save_gathers[seen["target_id"]]
                    )
                    hold.wait_returned.set()

                manager.wait_for_gathers = release_on_wait
            hold.armed = True
            # The target arrives second, so it is the one preempted.
            llm.generate(
                [other, target],
                SamplingParams(
                    temperature=0, max_tokens=DECODE_TOKENS, ignore_eos=True
                ),
                use_tqdm=False,
            )
            hold.release.set()
            _wait_for_saves(manager, hold)
            stored = _read_pages(worker, metadata, target_hashes)
        finally:
            save_patch.stop()
            _shutdown(llm, connectors)
            del llm, connectors, worker, manager
            gc.collect()
            torch.accelerator.empty_cache()

        # The scenario happened: the target was preempted with a save held,
        # and the other request was given blocks that save still had to read.
        assert hold.held_block_ids, "no save of the target was held"
        assert seen["preempted"], "the target was not preempted while its save was held"
        assert seen["reused"], "no held block was reused after the preemption"
        assert not hold.timed_out, "the hold was not released by the scenario"
        if wait_for_gathers is not None:
            assert seen["gathered_when_wait_returned"], "the wait ended too early"
        print(
            f"held={sorted(hold.held_block_ids)} reused={sorted(seen['reused'])} "
            f"hold_timed_out={hold.timed_out}"
        )

        def relative_errors(pages):
            return [
                ((p - r).norm() / r.norm().clamp_min(1e-6)).item()
                for p, r in zip(pages, reference)
            ]

        # The pages at the end are informational: a later save may rewrite them.
        print(
            f"end-of-run page errors: {[round(e, 4) for e in relative_errors(stored)]}"
        )
        assert hold.snapshot is not None, "the held save did not finish"
        # Batching can move the values slightly, another request's KV cannot.
        errors = relative_errors(hold.snapshot)
        print(f"held-save page errors: {[round(e, 4) for e in errors]}")
        assert max(errors) < 0.05, errors

    def test_failed_gather_still_fences_its_reads(self, tmp_path, in_process_metadata):
        llm = _hf3fs_llm(tmp_path)
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        worker = get_kv_transfer_group()
        manager = worker._async_manager
        connectors = [scheduler.connector, worker]
        gather_or_scatter = worker._gather_or_scatter_kv_caches
        stream_synchronize = torch.cuda.Stream.synchronize
        save_stream = manager._save_stream
        injected = threading.Event()
        save_stream_synced = threading.Event()

        def gather_one_block_then_fail(block_ids, buffers, direction):
            if direction != "gather":
                return gather_or_scatter(block_ids, buffers, direction)
            gather_or_scatter(block_ids[:1], buffers[:1], direction)
            injected.set()
            raise RuntimeError("injected gather failure")

        def record_save_stream_sync(stream):
            stream_synchronize(stream)
            if stream is save_stream:
                save_stream_synced.set()

        def sync_fails(stream):
            raise RuntimeError("injected sync failure")

        fail_gather = patch.object(
            worker, "_gather_or_scatter_kv_caches", gather_one_block_then_fail
        )
        try:
            watch_sync = patch.object(
                torch.cuda.Stream, "synchronize", record_save_stream_sync
            )
            with fail_gather, watch_sync:
                failed = manager.submit_save_operation("failed", [0, 1], ["a", "b"])
                manager.wait_for_gathers({"failed"})
                assert injected.is_set(), "the gather failure was not injected"
                # The partial gather was fenced before the wait returned.
                assert save_stream_synced.is_set(), "the wait returned before the fence"
            assert failed.result(timeout=60) is False
            # The save thread survives the failure and serves the next save.
            assert manager.submit_save_operation("next", [2], ["c"]).result(timeout=60)

            # Without a working sync, the reads can't be fenced, so reuse must stop.
            injected.clear()
            fail_sync = patch.object(torch.cuda.Stream, "synchronize", sync_fails)
            with fail_gather, fail_sync:
                unfenced = manager.submit_save_operation("unfenced", [3, 4], ["d", "e"])
                assert unfenced.result(timeout=60) is False
                assert injected.is_set(), "the gather failure was not injected"
                with pytest.raises(RuntimeError, match="could not fence"):
                    manager.wait_for_gathers({"unfenced"})
            with pytest.raises(RuntimeError, match="could not fence"):
                manager.get_finished_operations({"unfenced"})
            # Its buffers stay out of the pool.
            pool = manager._save_buffer_allocator._free_buffers
            assert len(manager._unfenced_buffers) == 2
            assert not any(
                buffer is retained
                for buffer in pool
                for retained in manager._unfenced_buffers
            )
        finally:
            _shutdown(llm, connectors)
            del llm, connectors, worker, manager
            gc.collect()
            torch.accelerator.empty_cache()
