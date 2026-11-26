# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import regex as re
import torch
from safetensors.numpy import save

from tests.distributed.aux_output_connector.utils import fake_store, wait_for_writer
from tests.v1.core.test_async_scheduler import PipelinedEngine
from tests.v1.core.utils import (
    EOS_TOKEN_ID,
    create_scheduler,
    mock_kv,
    new_kv_cache_group,
)
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata,
    AuxOutputSchedulerConnector,
    retained_objects,
)
from vllm.distributed.aux_output_connector.object_store import (
    AuxObjectWriter,
    aux_namespace_cache_salt,
    read_aux_rows,
)
from vllm.distributed.aux_output_connector.routed_experts import RoutedExpertsBuffer
from vllm.distributed.aux_output_connector.store import BlockObjectStore
from vllm.distributed.aux_output_connector.worker import (
    AuxOutputWorkerConnector,
    _ObjectRun,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.engine import CheckpointState, FinishReason
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.worker_base import WorkerBase

pytestmark = pytest.mark.cpu_test

_BLOCK = 16
_PREFIX = "job"
_NAMESPACE = "g/1"


def _requests(
    request_ids,
    num_tokens,
    max_tokens,
    *,
    namespace=_NAMESPACE,
    prompt=None,
    prompt_start=0,
    lookahead=0,
    stop_token_ids=None,
    token=0,
    cache_salt=None,
):
    init_none_hash(sha256)
    prompt = prompt or [token] * num_tokens
    requests = []
    for request_id in request_ids:
        params = SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=True,
            stop_token_ids=stop_token_ids,
            aux_namespace=namespace,
            aux_prompt_start=prompt_start,
        )
        params.update_from_generation_config({}, EOS_TOKEN_ID)
        requests.append(
            Request(
                request_id=request_id,
                prompt_token_ids=prompt,
                sampling_params=params,
                pooling_params=None,
                cache_salt=cache_salt
                if namespace is None
                else aux_namespace_cache_salt(_PREFIX, namespace, cache_salt),
                block_hasher=get_request_block_hasher(_BLOCK, sha256, lookahead),
            )
        )
    return requests


def _worker(store):
    worker = object.__new__(AuxOutputWorkerConnector)
    worker._capturer = Mock()
    worker._store = BlockObjectStore(max_bytes=1 << 20, object_nbytes=_BLOCK * 6)
    worker._buffer = RoutedExpertsBuffer(
        np.dtype(np.uint8), (3, 2), _BLOCK, 16, 1024, 2
    )
    worker._writer = AuxObjectWriter(store, timeout_s=600.0)
    worker._requests = {}
    worker._generation = 0
    worker._step_metadata = None
    worker._pending_outputs = []
    worker._lock = threading.Lock()
    worker._max_concurrent_batches = 2
    return worker


def _input_batch(request_ids, starts, counts):
    return SimpleNamespace(
        req_ids=request_ids,
        num_computed_tokens_np=np.array(starts),
        query_start_loc_np=np.cumsum([0, *counts]),
    )


def _tag(row):
    return (
        int(row[0, 0]),
        int(row[0, 1]),
        int(row[1, 0]) * 256 + int(row[1, 1]),
        int(row[2, 0]),
        int(row[2, 1]),
    )


def _model_output(
    scheduler_output: SchedulerOutput, sampled: dict[str, list[int]]
) -> ModelRunnerOutput:
    request_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=request_ids,
        sampled_token_ids=[sampled.get(r, []) for r in request_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


class _Engine:
    def __init__(
        self,
        scheduler: Scheduler | None = None,
        kv_load_failure_policy: str = "recompute",
        drafts: int = 0,
        **kwargs: Any,
    ) -> None:
        if drafts:
            kwargs.update(num_speculative_tokens=drafts, speculative_method="ngram_gpu")
        if scheduler is None:
            scheduler = create_scheduler(enable_prefix_caching=True, **kwargs)
        self.scheduler = scheduler
        manager = scheduler.kv_cache_manager
        self.connector = AuxOutputSchedulerConnector(
            manager,
            retained_objects(scheduler.vllm_config, manager.kv_cache_config)[0],
            _PREFIX,
        )
        scheduler.aux_output_connector = self.connector
        self.client, self.store = fake_store()
        self.worker = _worker(self.store)
        begin_step = self.worker.begin_step

        def record_writes(metadata):
            # Every path that hands writes to the worker, pauses included.
            if metadata is not None:
                self._queue_writes(metadata)
            begin_step(metadata)

        self.worker.begin_step = record_writes
        self.kv_block = self.connector._block_size
        self.version = 0
        self.num_steps = 0
        self.num_hits = 0
        self.kv: dict[Any, Any] = {}
        self.tier: dict[Any, Any] = {}
        self.fail_loads_from = None
        # Loads the connector reports asynchronously, the whole prompt
        # included.
        self.async_loads = False
        self._recving: set[Any] = set()
        self.requests: dict[Any, Any] = {}
        self.owners: dict[Any, Any] = {}
        self.by_owner: dict[Any, Any] = {}
        self.used: dict[Any, Any] = {}
        self.computed: list[Any] = []
        self.blocks: dict[Any, Any] = {}
        self.hits: dict[Any, Any] = {}
        self.sampled: dict[Any, Any] = {}
        # Rows computed over a draft token the request did not keep, or over
        # a position whose token was unknown when the output was dropped.
        self.off_token: set[Any] = set()
        # Steps that computed a request's rows over a failed load.
        self.poisoned: set[Any] = set()
        # By owner: the end of the positions it computed, and, for those a
        # later run of it started behind, the computation whose KV it used
        # from then on: the KV it hit, or its own recomputation.
        self.ends: dict[Any, Any] = {}
        self.rerun: dict[Any, Any] = {}
        # By owner: the step a later run of it started at, for the positions
        # it computes again; and the step before which each object's write
        # was sent.
        self.resumed: dict[Any, Any] = {}
        self.queued: dict[Any, Any] = {}
        self.finals: list[Any] = []
        self.listings: dict[Any, Any] = {}
        self._tags: dict[Any, Any] = {}
        self._in_flight: dict[Any, Any] = {}
        self._loads: list[Any] = []
        self._load_from: dict[Any, Any] = {}
        self._schedule = scheduler.schedule
        self._update = scheduler.update_from_output
        scheduler.schedule = self.schedule  # type: ignore[method-assign]
        scheduler.update_from_output = self.update_from_output  # type: ignore[method-assign]
        if scheduler.connector is not None:
            scheduler.connector.get_num_new_matched_tokens = self._matched  # type: ignore[method-assign]
            scheduler.connector.update_state_after_alloc = self._allocated  # type: ignore[method-assign]
            scheduler.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

    def _matched(self, request, num_computed_tokens):
        first = end = num_computed_tokens // _BLOCK
        limit = request.num_tokens - (not self.async_loads)
        while (
            end < min(limit // _BLOCK, len(request.block_hashes))
            and request.block_hashes[end] in self.tier
        ):
            end += 1
        self._load_from[request.request_id] = first
        return (end - first) * _BLOCK, self.async_loads and end > first

    def _allocated(self, request, blocks, num_external_tokens):
        first = self._load_from.pop(request.request_id, 0)
        (block_ids,) = blocks.get_block_ids()
        if self.async_loads and num_external_tokens:
            self._recving.add(request.request_id)
        for index in range(first, first + num_external_tokens // _BLOCK):
            self._loads.append((block_ids[index], request.block_hashes[index], index))

    def _load(self):
        invalid = set()
        for block_id, block_hash, index in self._loads:
            slots = list(self.tier[block_hash])
            failed = self.fail_loads_from is not None
            if failed and index * _BLOCK >= self.fail_loads_from:
                slots = [tag[:4] + (1,) for tag in slots]
                invalid.add(block_id)
            self.kv[block_id] = slots
        self._loads = []
        return invalid

    def _slot(self, block_ids, position):
        slots = self.kv.get(block_ids[position // self.kv_block])
        return None if slots is None else slots[position % self.kv_block]

    def schedule(self):
        scheduler_output = self._schedule()
        metadata = scheduler_output.aux_output_connector_metadata
        self.worker.begin_step(metadata)
        for copy in scheduler_output.kv_cache_block_copies or ():
            if copy.src_block_id in self.kv:
                self.kv[copy.dst_block_id] = list(self.kv[copy.src_block_id])
        self.num_steps += 1
        step, version = self.num_steps, self.version % 256
        invalid = self._load()
        scheduled = {}
        ids, starts, counts, rows = [], [], [], []
        for request_id, count in scheduler_output.num_scheduled_tokens.items():
            request = self.scheduler.requests[request_id]
            self.requests[request_id] = request
            owner = self.owners.setdefault(id(request), len(self.owners) + 1)
            start = request.num_computed_tokens - count
            block_ids = self.scheduler.kv_cache_manager.get_block_ids(request_id)[
                self.connector._group_id
            ]
            assert metadata is not None
            if request_id in (metadata.namespace_runs or {}):
                self.hits[request_id] = start
                self.num_hits += start > 0
                rerun = self.rerun.setdefault(owner, {})
                resumed = self.resumed.setdefault(owner, {})
                for position in range(self.ends.get(owner, 0)):
                    if position >= start:
                        rerun[position] = None
                        resumed[position] = step
                    elif (tag := self._slot(block_ids, position)) is not None:
                        rerun[position] = tag[2:4]
            self.blocks[id(request)] = list(block_ids)
            self.by_owner[owner] = request
            used = self.used.setdefault(id(request), {})
            poisoned = 0
            for position in range(start):
                if (tag := self._slot(block_ids, position)) is not None:
                    used.setdefault(position, set()).add(tag[2:4])
                    poisoned |= tag[4]
            if poisoned:
                self.poisoned.add((step, owner))
            step_rows = np.zeros((count, 3, 2), dtype=np.uint8)
            for offset, position in enumerate(range(start, start + count)):
                tag = (position, version, step, owner, poisoned)
                used.setdefault(position, set()).add((step, owner))
                if position in self.rerun.get(owner, ()):
                    self.rerun[owner][position] = (step, owner)
                self.computed.append((step, owner, position))
                slots = self.kv.setdefault(
                    block_ids[position // self.kv_block], [None] * self.kv_block
                )
                slots[position % self.kv_block] = tag
                step_rows[offset] = [
                    [position, version],
                    [step >> 8, step & 255],
                    [owner, poisoned],
                ]
            self.ends[owner] = max(self.ends.get(owner, 0), start + count)
            scheduled[request_id] = (request, start, count)
            ids.append(request_id)
            starts.append(start)
            counts.append(count)
            rows.append(step_rows)
        pending = None
        if ids:
            routing = torch.from_numpy(np.concatenate(rows))
            self.worker._capturer.snapshot_routing_data.return_value = routing
            pending = self.worker.prepare_output(_input_batch(ids, starts, counts))
        self._in_flight[id(scheduler_output)] = (
            step,
            version,
            pending,
            invalid,
            scheduled,
        )
        return scheduler_output

    def update_from_output(self, scheduler_output, model_runner_output):
        step, version, pending, invalid, scheduled = self._in_flight.pop(
            id(scheduler_output)
        )
        recving, self._recving = self._recving, set()
        if invalid or recving:
            model_runner_output.kv_connector_output = KVConnectorOutput(
                finished_recving=recving or None, invalid_block_ids=invalid
            )
        sampled = dict(
            zip(model_runner_output.req_ids, model_runner_output.sampled_token_ids)
        )
        if pending is not None:
            drafts = scheduler_output.scheduled_spec_decode_tokens
            tokens = [sampled.get(r, []) for r in pending.request_ids]
            pending.enqueue_cpu_copy(
                np.array([len(t) for t in tokens]),
                np.array(
                    [
                        max(len(drafts.get(r, ())) + 1 - len(t), 0) if t else 0
                        for r, t in zip(pending.request_ids, tokens)
                    ]
                ),
            )
            model_runner_output.aux_output_connector_output = pending.process_output()
        before = {
            request_id: request.num_tokens
            for request_id, (request, _, _) in scheduled.items()
        }
        outputs = self._update(scheduler_output, model_runner_output)
        for request_id, (request, start, count) in scheduled.items():
            num_tokens = before[request_id]
            owner = self.owners[id(request)]
            appended = request.num_tokens - num_tokens
            versions = self.sampled.setdefault(id(request), {})
            for index in range(num_tokens, request.num_tokens):
                versions[index - 1] = (version, step)
            kept = num_tokens + max(appended - 1, 0)
            for position in range(max(start, kept), start + count):
                self.off_token.add((step, owner, position))
            if (
                self.scheduler.connector is not None
                and request.is_finished()
                and request.status != RequestStatus.FINISHED_ERROR
            ):
                self._offload(request)
        for client_outputs in outputs.values():
            for output in client_outputs.outputs:
                entry = scheduled.get(output.request_id)
                request = entry[0] if entry else self.requests[output.request_id]
                if output.finish_reason in (FinishReason.STOP, FinishReason.LENGTH):
                    self._final(request, output.aux_objects, False)
        return outputs

    def _final(self, request, objects, aborted):
        self.finals.append((request, objects, request.num_tokens, aborted))
        self.listings[request.request_id] = objects

    def _offload(self, request):
        block_ids = self.blocks.get(id(request), [])
        num_blocks = min(
            request.num_computed_tokens // _BLOCK,
            len(request.block_hashes),
            len(block_ids),
        )
        for index in range(num_blocks):
            slots = self.kv.get(block_ids[index])
            if slots and all(tag is not None and not tag[4] for tag in slots):
                self.tier.setdefault(request.block_hashes[index], list(slots))

    def step(self, scheduler_output, sampled=None):
        outputs = self.update_from_output(
            scheduler_output, _model_output(scheduler_output, sampled or {})
        )
        return {
            output.request_id: output
            for client_outputs in outputs.values()
            for output in client_outputs.outputs
        }

    def complete(self, scheduler_output):
        requests = self.scheduler.requests
        return {
            request_id: [0]
            for request_id in scheduler_output.num_scheduled_tokens
            if request_id in requests
            and requests[request_id].num_computed_tokens
            >= requests[request_id].num_tokens
        }

    def run(self, after_step=None):
        outputs: dict[str, Any] = {}
        for _ in range(1000):
            if not self.scheduler.has_requests():
                return outputs
            scheduler_output = self.schedule()
            outputs.update(self.step(scheduler_output, self.complete(scheduler_output)))
            if after_step is not None:
                after_step()
        raise AssertionError("the engine did not converge")

    def _queue_writes(self, metadata):
        for writes in (metadata.object_writes or {}).values():
            for key, _, _ in writes:
                self.queued.setdefault(key, self.num_steps)

    def pause(self):
        self.worker.begin_step(self.connector.flush_metadata())
        self.wait_for_writes()

    def wait_for_writes(self):
        # Without a write barrier, wait for the writes queued so far.
        wait_for_writer(self.worker._writer, 0.005)

    def tags(self, key):
        if key not in self._tags:
            rows = read_aux_rows(self.store, [key], "routed_experts", timeout_s=0)
            self._tags[key] = [_tag(row) for row in rows]
        return self._tags[key]

    def close(self):
        assert not self._in_flight
        self.run()
        self.pause()
        self.worker.close()
        _check(self)


def _splice(engine, objects):
    held = {}
    end = None
    assert len({key for key, _, _ in objects}) == len(objects), objects
    for key, start, stop in objects:
        tags = engine.tags(key)
        assert [tag[0] for tag in tags[: stop - start]] == list(range(start, stop))
        # Entries abut: no overlap, no gap.
        assert end is None or start == end, objects
        end = stop
        for position in range(start, stop):
            held[position] = (key, tags[position - start])
    return held


def _check(engine):
    tokens = {}

    def prefix(request, position):
        if id(request) not in tokens:
            tokens[id(request)] = list(request.all_token_ids)
        prompt = tuple(tokens[id(request)][: position + 1])
        return request.cache_salt, position, prompt

    computations = Counter(
        prefix(engine.by_owner[owner], position)
        for _, owner, position in engine.computed
    )
    for key in engine.client.objects:
        for tag in engine.tags(key):
            assert not tag[4], key
            assert (tag[2], tag[3], tag[0]) not in engine.off_token, key
    for request, objects, num_tokens, aborted in engine.finals:
        namespace = request.sampling_params.aux_namespace
        if namespace is None:
            assert objects is None
            continue
        if aborted and objects is None:
            continue
        assert objects is not None, request.request_id
        assert objects or not aborted, request.request_id
        start = request.sampling_params.aux_prompt_start
        first, end = start - start % _BLOCK, num_tokens - 1
        held = _splice(engine, objects)
        positions = sorted(held)
        if aborted:
            assert positions == list(range(first, first + len(positions)))
            assert first + len(positions) <= end
        else:
            assert positions == (list(range(first, end)) if start < end else [])
        owner = engine.owners.get(id(request))
        tail = re.compile(rf"{_PREFIX}/{re.escape(namespace)}/tail-[0-9a-f]{{32}}")
        for key, object_start, object_end in objects:
            if tail.fullmatch(key):
                assert {tag[3] for tag in engine.tags(key)} == {owner}
                continue
            size = engine.connector._object_size
            block_hash = request.block_hashes[(object_start + size) // _BLOCK - 1]
            assert key == f"{_PREFIX}/{namespace}/{block_hash.hex()}"
            assert object_start % size == 0
            assert object_start < object_end <= object_start + size
        sampled = engine.sampled.get(id(request), {})
        used = engine.used.get(id(request), {})
        for position, (key, tag) in held.items():
            where = (request.request_id, position, key)
            expected = prefix(request, position)
            assert prefix(engine.by_owner[tag[3]], position) == expected, where
            if computations[expected] == 1:
                assert tag[2:4] in used.get(position, ()), where
            rerun = engine.rerun.get(owner, {}).get(position)
            if rerun is not None and tag[2:4] == rerun:
                # Resumed after a preemption or a reset: the rows of the KV
                # it used from then on replace the earlier ones, as inline
                # rows do. Anything else is checked
                # against the sampling computation.
                continue
            resumed = engine.resumed.get(owner, {}).get(position)
            if resumed is not None and tag[3] == owner:
                # Its own rows from before it resumed: only the first copy of
                # an object sent before then.
                assert engine.queued.get(key, resumed) < resumed, where
            if position not in sampled:
                continue
            version, step = sampled[position]
            if (*tag[1:3], tag[3]) != (version, step, owner) and (
                (step, owner) not in engine.poisoned
            ):
                # A full block another computation stored first. A token
                # sampled over a failed load keeps no rows.
                assert not tail.fullmatch(key), where
                assert tag[2] <= step, where


def _spans(objects):
    return [(start, end) for _, start, end in objects]


def _keys(engine, request_id):
    return [key for key, _, _ in engine.listings[request_id]]


def _block_keys(request: Request, num_blocks: int | None = None) -> list[str]:
    return [
        f"{_PREFIX}/{_NAMESPACE}/{h.hex()}" for h in request.block_hashes[:num_blocks]
    ]


def _tails(keys: Iterable[str]) -> set[str]:
    return {key for key in keys if "/tail-" in key}


def _versions(engine, objects):
    held = _splice(engine, objects)
    return [held[position][1][1] for position in sorted(held)]


def _abort(engine, request_id):
    core = object.__new__(EngineCoreProc)
    core.scheduler = engine.scheduler
    core.output_queue = queue.Queue()
    (request,) = engine.scheduler.finish_requests(
        request_id, RequestStatus.FINISHED_ABORTED
    )
    core._send_abort_outputs([request])
    ((_, outputs),) = core.output_queue.queue
    (output,) = outputs.outputs
    engine._final(request, output.aux_objects, True)
    return output


def _engine_core(engine):
    worker = object.__new__(WorkerBase)
    worker.model_runner = SimpleNamespace(aux_output_connector=engine.worker)
    core = object.__new__(EngineCore)
    core.scheduler = engine.scheduler
    core.model_executor = Mock()
    core.mm_receiver_cache = None
    core._kv_cache_reclaimable = False
    core._checkpoint_state = CheckpointState.RUNNING
    core.deferred_step = None
    core.collective_rpc = lambda method, args=(): [getattr(worker, method)(*args)]  # type: ignore[method-assign, misc]
    return core


def _connector_scheduler(**kwargs):
    return create_scheduler(
        enable_prefix_caching=True, use_kv_connector=mock_kv(0, False), **kwargs
    )


def _add(
    engine: _Engine, request_id: str, num_tokens: int, max_tokens: int, **kwargs: Any
) -> Request:
    (request,) = _requests([request_id], num_tokens, max_tokens, **kwargs)
    engine.scheduler.add_request(request)
    return request


def _turn(engine, request_id, num_tokens, max_tokens=1, *, version=None, **kwargs):
    if version is not None:
        engine.version = version
    _add(engine, request_id, num_tokens, max_tokens, **kwargs)
    return engine.run()


def _preempt(engine, request, drop=False):
    engine.scheduler.running.remove(request)
    engine.scheduler._preempt_request(request, 0.0, drop_stale_output=drop)


def _slow_puts(engine):
    put_batch = engine.client.put_batch

    def slow_put_batch(*args):
        time.sleep(0.05)
        return put_batch(*args)

    engine.client.put_batch = slow_put_batch


@pytest.mark.parametrize(
    "async_scheduling,queue_size,lookahead",
    [(False, 1, 0), (True, 2, 0), (True, 2, 1)],
    ids=["sync", "async", "async-lookahead"],
)
def test_listings_resolve_under_preemption_and_weight_updates(
    async_scheduling, queue_size, lookahead
):
    engine = _Engine(
        async_scheduling=async_scheduling, num_blocks=20, max_num_batched_tokens=64
    )
    scheduler = engine.scheduler
    pending = [
        *_requests([f"0-{i}" for i in range(5)], 70, 27, lookahead=lookahead),
        *_requests([f"1-{i}" for i in range(3)], 102, 27, lookahead=lookahead),
    ]
    request_ids = {request.request_id for request in pending}
    preempted = []
    preempt_request = scheduler._preempt_request

    def preempt(request, timestamp, **kwargs):
        preempted.append(request.request_id)
        return preempt_request(request, timestamp, **kwargs)

    scheduler._preempt_request = preempt  # type: ignore[method-assign]

    def before_step(step, _):
        engine.version = step // 4
        if "0-3" in engine.requests and "0-3" in scheduler.requests:
            scheduler.finish_requests("0-3", RequestStatus.FINISHED_ABORTED)
        if step % 2 == 0 and pending:
            scheduler.add_request(pending.pop(0))

    scheduler.add_request(pending.pop(0))
    PipelinedEngine(scheduler, queue_size=queue_size).run(before_step=before_step)
    engine.close()
    finished = {request.request_id for request, *_ in engine.finals}
    assert finished == request_ids - {"0-3"}
    assert preempted and engine.num_hits


@pytest.mark.parametrize(
    "num_tokens,budget,version",
    [(40, 64, 2), (47, 64, 1), (64, 32, 1)],
    ids=["decode-inside-a-block", "decode-completes-a-block", "chunk-ends-the-prompt"],
)
@pytest.mark.parametrize("drop", [False, True], ids=["keep", "drop"])
def test_request_preempted_with_a_step_in_flight_keeps_its_first_copies(
    drop, num_tokens, budget, version
):
    engine = _Engine(async_scheduling=True, max_num_batched_tokens=budget)
    engine.version = 1
    request = _add(engine, "r", num_tokens, 8)
    first, in_flight = engine.schedule(), engine.schedule()
    engine.step(first, {"r": [0]} if num_tokens <= budget else {})
    _preempt(engine, request, drop)
    engine.version = 2
    resumed = engine.schedule()
    engine.step(in_flight, {"r": [0]})
    engine.step(resumed, engine.complete(resumed))
    engine.close()
    position = num_tokens if num_tokens <= budget else num_tokens - 1
    held = _splice(engine, engine.listings["r"])
    # A block the step in flight completed keeps its first copy, whether
    # or not the prefix cache holds it; a position inside a block is
    # computed again.
    assert held[position][1][1] == version


def test_siblings_admitted_in_one_step_share_the_prompt():
    engine = _Engine()
    for request in _requests(["first", "second"], 40, 1):
        engine.scheduler.add_request(request)
    scheduler_output = engine.schedule()
    assert scheduler_output.num_scheduled_tokens == {"first": 40, "second": 8}
    engine.step(scheduler_output, {"first": [1], "second": [2]})
    engine.close()
    first, second = _keys(engine, "first"), _keys(engine, "second")
    assert len(first) == len(second) == 3
    assert first[:2] == second[:2] and first[2] != second[2]


def _preempted_with_its_last_chunk_in_flight(
    drop: bool,
) -> tuple[_Engine, Request, SchedulerOutput]:
    engine = _Engine(async_scheduling=True, max_num_batched_tokens=32)
    a = _add(engine, "a", 48, 4)
    first, last = engine.schedule(), engine.schedule()
    engine.step(first)
    _preempt(engine, a, drop)
    return engine, a, last


@pytest.mark.parametrize("order", ["keep", "drop", "resume", "output", "pause"])
def test_request_hits_the_block_a_preempted_then_aborted_request_completed(order):
    engine, a, last = _preempted_with_its_last_chunk_in_flight(drop=order != "keep")
    assert last.num_scheduled_tokens == {"a": 16}
    scheduler = engine.scheduler
    if order == "resume":
        # The resumed run computes part of the block the dropped step did;
        # aborted with that step in flight, `a` lists its full blocks.
        scheduler.max_num_scheduled_tokens = 8
        resume = engine.schedule()
        assert resume.num_scheduled_tokens == {"a": 8}
        scheduler.max_num_scheduled_tokens = 32
        output = _abort(engine, "a")
        engine.step(last, {"a": [0]})
        engine.step(resume)
    elif order in ("output", "pause"):
        engine.step(last, {"a": [0]})
        if order == "pause":
            engine.pause()
        output = _abort(engine, "a")
    else:
        output = _abort(engine, "a")
    _add(engine, "b", 64, 1)
    hit = engine.schedule()
    assert hit.num_scheduled_tokens == {"b": 16}
    if order in ("keep", "drop"):
        engine.step(last, {"a": [0]})
    engine.step(hit, engine.complete(hit))
    engine.close()
    listed = 32 if order == "resume" else 47
    assert sorted(_splice(engine, output.aux_objects)) == list(range(listed))
    assert engine.hits["b"] == 48
    assert _keys(engine, "b")[:3] == _block_keys(a, 3)


def test_preempted_request_aborted_after_a_rejected_draft_completed_a_block():
    engine = _Engine(async_scheduling=True, drafts=3)
    request = _add(engine, "r", 44, 30, token=1)
    prefill, drafts = engine.schedule(), engine.schedule()
    assert drafts.num_scheduled_tokens == {"r": 4}
    engine.step(prefill, {"r": [1]})
    _preempt(engine, request)
    # Two drafts are accepted: position 47 was computed over the third.
    engine.step(drafts, {"r": [1, 1, 1]})
    assert request.num_tokens == 48
    output = _abort(engine, "r")
    engine.close()
    assert _spans(output.aux_objects) == [(0, 16), (16, 32), (32, 47)]


@pytest.mark.parametrize(
    "num_tokens,block_written",
    [(40, False), (46, False), (47, True)],
    ids=["tail-of-settled-positions", "block-without-a-hash", "block-in-flight"],
)
def test_aborted_request_lists_its_settled_positions(num_tokens, block_written):
    engine = _Engine(async_scheduling=True)
    request = _add(engine, "r", num_tokens, 8)
    prefill, decode = engine.schedule(), engine.schedule()
    engine.step(prefill, {"r": [0]})
    in_flight = engine.schedule()
    output = _abort(engine, "r")
    engine.step(decode, {"r": [0]})
    engine.step(in_flight, {"r": [0]})
    engine.pause()
    assert sorted(_splice(engine, output.aux_objects)) == list(range(num_tokens))
    written = len(request.block_hashes) > 2 and (
        _block_keys(request)[2] in engine.client.objects
    )
    assert written == block_written
    _turn(engine, "next", 64, prompt_start=num_tokens)
    engine.close()


def test_request_resubmitted_after_an_abort_lists_the_aborted_objects():
    engine = _Engine(async_scheduling=True)
    request = _add(engine, "first", 40, 8)
    prefill, decode = engine.schedule(), engine.schedule()
    engine.step(prefill, {"first": [0]})
    output = _abort(engine, "first")
    engine.step(decode, {"first": [0]})
    engine.pause()
    assert _spans(output.aux_objects) == [(0, 16), (16, 32), (32, 40)]
    aborted = _splice(engine, output.aux_objects)
    owner = engine.owners[id(request)]
    assert {tag[1:4] for _, tag in aborted.values()} == {(0, 1, owner)}
    engine.version = 1
    retry = _add(engine, "retry", 41, 1)
    scheduler_output = engine.schedule()
    assert scheduler_output.num_scheduled_tokens == {"retry": 9}
    engine.step(scheduler_output, {"retry": [0]})
    engine.close()
    listing = engine.listings["retry"]
    assert listing[:2] == output.aux_objects[:2]
    held = _splice(engine, listing)
    assert [held[p][1] for p in range(32)] == [aborted[p][1] for p in range(32)]
    assert {held[p][1][1] for p in range(32, 40)} == {1}
    assert {held[p][1][3] for p in range(32, 40)} == {engine.owners[id(retry)]}


@pytest.mark.parametrize(
    "abort",
    [
        lambda core: core.abort_requests(["r"]),
        lambda core: core.pause_scheduler(mode="abort", clear_cache=False),
    ],
    ids=["client-abort", "in-process-pause-abort"],
)
def test_abort_without_an_output_keeps_no_listing(abort):
    engine = _Engine()
    request = _add(engine, "r", 40, 8)
    engine.step(engine.schedule())
    abort(_engine_core(engine))
    engine.close()
    assert engine.listings == {}
    # Its spans and a tail nobody lists are still written.
    objects = set(engine.client.objects)
    assert objects - _tails(objects) == set(_block_keys(request, 2))
    assert len(_tails(objects)) == 1


@pytest.mark.parametrize(
    "num_steps_after_preemption", [0, 1], ids=["waiting", "scheduled-again"]
)
def test_request_aborted_after_a_preemption_lists_its_returned_positions(
    num_steps_after_preemption,
):
    engine = _Engine()
    request = _add(engine, "r", 40, 8)
    for _ in range(3):
        engine.step(engine.schedule(), {"r": [0]})
    _preempt(engine, request)
    engine.pause()
    for _ in range(num_steps_after_preemption):
        engine.step(engine.schedule(), {"r": [0]})
    output = _abort(engine, "r")
    engine.pause()
    positions = sorted(_splice(engine, output.aux_objects))
    assert positions == list(range(request.num_tokens - 1))
    engine.close()


def test_request_preempted_with_output_in_flight_lists_it_when_aborted():
    engine = _Engine(async_scheduling=True)
    request = _add(engine, "r", 40, 8)
    prefill, decode = engine.schedule(), engine.schedule()
    engine.step(prefill, {"r": [0]})
    _preempt(engine, request)
    idle = engine.schedule()
    assert "r" not in idle.num_scheduled_tokens
    engine.step(decode, {"r": [0]})
    engine.step(idle)
    output = _abort(engine, "r")
    engine.pause()
    assert sorted(_splice(engine, output.aux_objects)) == list(range(41))
    engine.close()


def test_preempted_request_that_stops_on_its_output_in_flight():
    engine = _Engine(async_scheduling=True, num_blocks=3)
    _add(engine, "first", 15, 5, token=1)
    engine.step(engine.schedule(), {"first": [1]})
    _add(engine, "second", 16, 1, lookahead=1)
    step = engine.schedule()
    assert step.num_scheduled_tokens == {"first": 1, "second": 16}
    preempting = engine.schedule()
    assert preempting.preempted_req_ids == {"second"}
    outputs = engine.step(step, {"first": [2], "second": [3]})
    assert outputs["second"].finish_reason is not None
    engine.step(preempting, {"first": [4]})
    engine.close()
    assert outputs["second"].aux_objects


def test_preempted_request_whose_accepted_drafts_complete_an_object():
    engine = _Engine(async_scheduling=True, drafts=2, num_blocks=12)
    _add(engine, "first", 26, 30, token=1)
    engine.step(engine.schedule(), {"first": [1]})
    second = _add(engine, "second", 46, 4)
    prefill = engine.schedule()
    drafts = engine.schedule()
    assert drafts.num_scheduled_tokens == {"first": 3, "second": 3}
    engine.step(prefill, {"first": [2, 3, 4], "second": [5]})
    block_pool = engine.scheduler.kv_cache_manager.block_pool
    held = block_pool.get_new_blocks(block_pool.get_num_free_blocks())
    preempting = engine.schedule()
    assert preempting.preempted_req_ids == {"second"}
    outputs = engine.step(drafts, {"first": [6, 7, 8], "second": [9, 10, 11]})
    assert outputs["second"].finish_reason is not None
    engine.step(preempting, {"first": [12]})
    block_pool.free_blocks(held)
    engine.close()
    # Block 2 is hashed by the accepted drafts, after the preemption: it is
    # written and listed.
    block = _block_keys(second)[2]
    assert block in engine.client.objects
    assert [key for key, _, _ in outputs["second"].aux_objects][2] == block


@pytest.mark.parametrize(
    "abort,expected", [(False, [(0, 13)]), (True, [(0, 12)])], ids=["stop", "abort"]
)
def test_request_ending_inside_a_block_its_drafts_computed_lists_a_tail(
    abort, expected
):
    engine = _Engine(async_scheduling=True, drafts=3)
    _add(engine, "r", 12, 30, stop_token_ids=[7])
    prefill, drafts = engine.schedule(), engine.schedule()
    assert drafts.num_scheduled_tokens == {"r": 4}
    engine.step(prefill, {"r": [1]})
    if abort:
        _abort(engine, "r")
    outputs = engine.step(drafts, {"r": [7, 1, 1, 1]})
    assert abort or outputs["r"].finish_reason is not None
    engine.close()
    assert _spans(engine.listings["r"]) == expected
    assert all("/tail-" in key for key in engine.client.objects)


@pytest.mark.parametrize(
    "namespace,cache_salt,expected",
    [(_NAMESPACE, None, 8), ("g/2", None, 40), (None, None, 40), (_NAMESPACE, "b", 40)],
    ids=["same-namespace", "other-namespace", "no-namespace", "other-cache-salt"],
)
def test_only_requests_of_a_namespace_hit_its_blocks(namespace, cache_salt, expected):
    engine = _Engine()
    _turn(engine, "first", 40)
    _add(engine, "second", 40, 1, namespace=namespace, cache_salt=cache_salt)
    scheduler_output = engine.schedule()
    assert scheduler_output.num_scheduled_tokens == {"second": expected}
    engine.step(scheduler_output, {"second": [0]})
    engine.close()


@pytest.mark.parametrize(
    "prompt,max_tokens,prompt_start,hit,expected",
    [
        ([0] * 72, 1, 0, 32, [(0, 16), (16, 32), (32, 48), (48, 64), (64, 72)]),
        ([0] * 72, 1, 40, 32, [(32, 48), (48, 64), (64, 72)]),
        ([0] * 72, 1, 70, 32, [(64, 72)]),
        ([0] * 72, 1, 72, 32, []),
        ([0] * 72, 9, 72, 32, [(64, 80)]),
        ([0] * 20 + [1] * 52, 1, 20, 16, [(16, 32), (32, 48), (48, 64), (64, 72)]),
    ],
    ids=[
        "hit-blocks",
        "mid-block",
        "inside-the-tail",
        "prompt-length",
        "prompt-length-then-decode",
        "rewritten-after-the-common-prefix",
    ],
)
def test_listing_covers_the_positions_from_aux_prompt_start(
    prompt, max_tokens, prompt_start, hit, expected
):
    engine = _Engine()
    _turn(engine, "first", 40, version=1)
    _turn(
        engine,
        "second",
        len(prompt),
        max_tokens,
        version=2,
        prompt=prompt,
        prompt_start=prompt_start,
    )
    engine.close()
    assert engine.hits["second"] == hit
    objects = engine.listings["second"]
    assert objects is not None and _spans(objects) == expected
    first = {start: key for key, start, _ in engine.listings["first"]}
    for key, start, _ in objects:
        assert (key == first.get(start)) == (start < hit)


@pytest.mark.parametrize(
    "prompt_start,expected",
    [(40, [(32, 48), (48, 56)]), (50, [(48, 56)]), (72, None)],
    ids=["before-the-settled-block", "inside-the-tail", "past-the-settled-end"],
)
def test_aborted_request_lists_its_positions_from_aux_prompt_start(
    prompt_start, expected
):
    engine = _Engine(max_num_batched_tokens=24)
    _turn(engine, "first", 40, version=1)
    _add(engine, "second", 72, 8, prompt_start=prompt_start)
    prefill = engine.schedule()
    assert prefill.num_scheduled_tokens == {"second": 24}
    engine.step(prefill)
    output = _abort(engine, "second")
    engine.pause()
    listed = _keys(engine, "first") + [key for key, _, _ in output.aux_objects or []]
    assert _tails(engine.client.objects) == _tails(listed)
    if expected is None:
        assert output.aux_objects is None
    else:
        assert _spans(output.aux_objects) == expected
        assert sorted(_splice(engine, output.aux_objects)) == list(
            range(expected[0][0], 56)
        )
    engine.close()


@pytest.mark.parametrize("tier,hit", [(False, 0), (True, 64)], ids=["no-tier", "tier"])
def test_turn_running_at_a_prefix_cache_reset_resumes(tier, hit):
    engine = _Engine(_connector_scheduler() if tier else None)
    _turn(engine, "first", 72, version=1)
    _add(engine, "r", 72, 6)
    for _ in range(2):
        engine.step(engine.schedule(), {"r": [0]})
    assert engine.hits["r"] == 64
    assert engine.scheduler.reset_prefix_cache(reset_running_requests=True)
    engine.run()
    engine.close()
    assert engine.hits["r"] == hit
    assert engine.listings["r"][:4] == engine.listings["first"][:4]


@pytest.mark.parametrize("reset", ["prefix-cache-reset", "failed-pause-reset"])
def test_request_preempted_by_a_reset_computes_its_partial_block_again(reset):
    engine = _Engine()
    engine.version = 1
    request = _add(engine, "r", 40, 6)
    for _ in range(2):
        engine.step(engine.schedule(), {"r": [0]})
    if reset == "prefix-cache-reset":
        assert engine.scheduler.reset_prefix_cache(reset_running_requests=True)
    else:
        block_pool = engine.scheduler.kv_cache_manager.block_pool
        held = block_pool.get_new_blocks(1)
        core = _engine_core(engine)
        with pytest.raises(RuntimeError, match="Failed to reset KV cache"):
            core.pause_scheduler(mode="keep")
        assert request.status == RequestStatus.PREEMPTED
        block_pool.free_blocks(held)
        core.resume_scheduler()
    engine.version = 2
    engine.close()
    # The partial block is computed again after the reset, under one tail.
    assert _spans(engine.listings["r"]) == [(0, 16), (16, 32), (32, 45)]
    assert _versions(engine, engine.listings["r"]) == [1] * 32 + [2] * 13


@pytest.mark.parametrize("tier", [False, True], ids=["no-tier", "tier"])
def test_turn_lists_the_block_hashed_at_the_previous_stop(tier):
    # `first` stops on the token that hashes block 2, which it computed: the
    # prefix cache never holds it, but `first` writes it. `second` computes
    # it again, or hits it in the KV tier that kept it, and lists it.
    engine = _Engine(_connector_scheduler() if tier else None)
    _turn(engine, "first", 48, 1, version=1, lookahead=1)
    _turn(engine, "second", 72, version=2, lookahead=1)
    engine.close()
    first, second = _keys(engine, "first"), _keys(engine, "second")
    assert engine.hits["second"] == (48 if tier else 32)
    block = _block_keys(engine.requests["first"])[2]
    assert first == second[:3] and first[2] == block


def test_block_computed_again_after_a_reset_keeps_its_first_copy():
    engine = _Engine()
    _turn(engine, "first", 40, version=1)
    assert engine.scheduler.reset_prefix_cache()
    _turn(engine, "second", 40, version=2)
    engine.close()
    assert engine.hits["second"] == 0
    assert _keys(engine, "second")[:2] == _keys(engine, "first")[:2]
    assert _versions(engine, engine.listings["second"]) == [1] * 32 + [2] * 8


def test_objects_listed_before_a_prefix_cache_reset_are_written():
    engine = _Engine()
    engine.version = 1
    _add(engine, "first", 40, 1)
    engine.run(
        lambda: "first" in engine.listings and engine.scheduler.reset_prefix_cache()
    )
    engine.close()
    assert len(engine.listings["first"]) == 3


def test_reset_writes_the_blocks_of_the_requests_it_preempts():
    engine = _Engine()
    engine.version = 1
    span = (32, 48)
    _add(engine, "r", 40, 12, lookahead=1)
    # Block 2 is hashed by the output of the 9th step: only the preemption
    # writes it.
    for _ in range(9):
        engine.step(engine.schedule(), {"r": [0]})
    assert engine.scheduler.reset_prefix_cache(reset_running_requests=True)
    resumed = engine.schedule()
    engine.pause()
    (key,) = [key for key in engine.client.objects if engine.tags(key)[0][0] == 32]
    assert [tag[0] for tag in engine.tags(key)] == list(range(*span))
    assert {tag[1] for tag in engine.tags(key)} == {1}
    engine.step(resumed, engine.complete(resumed))
    engine.close()
    assert (key, *span) in engine.listings["r"]


def test_pause_writes_every_object_of_the_stopped_and_aborted_requests():
    engine = _Engine()
    _slow_puts(engine)
    _turn(engine, "first", 40, 9, version=1)
    _add(engine, "repeat", 40, 4)
    engine.step(engine.schedule(), {"repeat": [0]})
    output = _abort(engine, "repeat")
    engine.pause()
    listed = _keys(engine, "first") + [key for key, _, _ in output.aux_objects]
    assert set(listed) <= set(engine.client.objects)
    engine.close()


def test_engine_core_pause_writes_the_objects_of_the_requests_it_preempts():
    engine = _Engine(async_scheduling=True)
    _slow_puts(engine)
    request = _add(engine, "r", 47, 8)
    prompt, decode = engine.schedule(), engine.schedule()
    engine.step(prompt, {"r": [0]})
    engine.step(decode, {"r": [0]})
    core = _engine_core(engine)
    core.pause_scheduler(mode="keep")
    engine.wait_for_writes()
    # Block 2 was hashed after its step was scheduled; the prefix cache
    # holds it once that step's output settled, so the pause writes it.
    assert set(engine.client.objects) == set(_block_keys(request, 3))
    core.resume_scheduler()
    engine.close()


def _done(value):
    future: Future[Any] = Future()
    future.set_result(value)
    return future


def _stepping_engine_core(engine, batch_queue):
    """An engine core whose own step methods drive the test engine."""
    core = _engine_core(engine)
    core.prefill_schedule_interval = 1
    core.num_executed_steps = 0
    core.is_mm_encoder_only = core.is_pooling_model = False
    core._weight_version = None
    core._untimed_iteration_details = lambda scheduler_output: None
    core._process_aborts_queue = lambda: None
    core.batch_queue = deque() if batch_queue else None
    core.batch_queue_size = 2
    engine.scheduler.schedule = lambda throttle_prefills=False: engine.schedule()
    scheduled: deque[Any] = deque()

    def output(scheduler_output):
        return _model_output(scheduler_output, engine.complete(scheduler_output))

    def execute_model(scheduler_output, non_block=False):
        # A step without scheduled tokens returns its output without sampling.
        if not scheduler_output.total_num_scheduled_tokens:
            return _done(output(scheduler_output))
        scheduled.append(scheduler_output)
        return _done(None)

    def sample_tokens(grammar_output, non_block=False):
        model_output = output(scheduled.popleft())
        return _done(model_output) if non_block else model_output

    core.model_executor = SimpleNamespace(
        execute_model=execute_model, sample_tokens=sample_tokens
    )
    return core


def test_engine_core_abort_leaves_the_hand_over_to_the_next_step():
    # An engine core process handles aborts between a step's forward pass and
    # its sampling, before the worker counts the step's outputs: the abort
    # hands nothing to the worker, and the core process steps again.
    engine = _Engine()
    core = _stepping_engine_core(engine, batch_queue=False)
    _add(engine, "r", 40, 1)
    _add(engine, "other", 8, 100, namespace=None, token=1)
    while "r" in engine.scheduler.requests:
        core.step()
    core.abort_requests(["other"])
    engine.wait_for_writes()
    assert engine.connector._writes and not engine.client.objects
    engine.close()


@pytest.mark.parametrize("second_id", ["b", "a"], ids=["siblings", "same-request-id"])
def test_requests_stopping_on_the_same_tokens_write_their_own_tails(second_id):
    engine = _Engine()
    _turn(engine, "a", 40, version=1)
    first = engine.listings["a"][-1]
    _turn(engine, second_id, 40, version=2)
    second = engine.listings[second_id][-1]
    engine.close()
    assert first[1:] == second[1:] == (32, 40) and first[0] != second[0]
    assert [tag[1] for tag in engine.tags(second[0])] == [2] * 8


@pytest.mark.parametrize(
    "reset", [False, True], ids=["preemption", "prefix-cache-reset"]
)
def test_preempted_request_lists_one_object_per_block(reset):
    engine = _Engine()
    engine.version = 1
    request = _add(engine, "r", 40, 20)
    for _ in range(10):
        engine.step(engine.schedule(), {"r": [0]})
    if reset:
        assert engine.scheduler.reset_prefix_cache(reset_running_requests=True)
    else:
        _preempt(engine, request)
    engine.pause()
    # The preemption writes the full blocks it computed, no tail.
    assert not _tails(engine.client.objects)
    engine.version = 2
    engine.close()
    # The partial block is computed again; the listing has one key per block.
    listing = engine.listings["r"]
    assert _spans(listing) == [(0, 16), (16, 32), (32, 48), (48, 59)]
    assert _versions(engine, listing) == [1] * 48 + [2] * 11
    tails = _tails(engine.client.objects)
    assert set(engine.client.objects) - tails == set(_block_keys(request, 3))
    assert len(tails) == 1


def _recompute_after_a_preemption(engine):
    engine.version = 1
    request = _add(engine, "r", 40, 20)
    for _ in range(10):
        engine.step(engine.schedule(), {"r": [0]})
    _preempt(engine, request)
    engine.pause()
    assert engine.scheduler.reset_prefix_cache()
    engine.version = 2
    recompute = engine.schedule()
    assert recompute.num_scheduled_tokens == {"r": 40}
    engine.step(recompute)
    return request


@pytest.mark.parametrize("abort", [False, True], ids=["preempted-again", "aborted"])
def test_request_interrupted_while_it_recomputes_keeps_its_first_copies(abort):
    engine = _Engine(max_num_batched_tokens=40)
    request = _recompute_after_a_preemption(engine)
    if abort:
        output = _abort(engine, "r")
        engine.pause()
        assert _versions(engine, output.aux_objects) == [1] * 32 + [2] * 8
    else:
        _preempt(engine, request)
        engine.pause()
        engine.version = 3
    engine.close()
    if not abort:
        assert _versions(engine, engine.listings["r"]) == [1] * 48 + [3] * 11


def _recompute_inside_blocks_a_dropped_step_wrote():
    engine = _Engine(async_scheduling=True, max_num_batched_tokens=40)
    scheduler = engine.scheduler
    scheduler.max_num_scheduled_tokens = 32
    engine.version = 1
    request = _add(engine, "r", 96, 4)
    first, second = engine.schedule(), engine.schedule()
    assert second.num_scheduled_tokens == {"r": 32}
    engine.step(first)
    _preempt(engine, request, drop=True)
    engine.step(second)
    engine.pause()
    assert scheduler.reset_prefix_cache()
    scheduler.max_num_scheduled_tokens = 40
    engine.version = 2
    recompute = engine.schedule()
    assert recompute.num_scheduled_tokens == {"r": 40}
    engine.step(recompute)
    return engine, request


@pytest.mark.parametrize(
    "ending", ["preempted", "preempted-then-aborted", "aborted-while-running"]
)
def test_request_interrupted_inside_blocks_a_dropped_step_wrote(ending):
    engine, request = _recompute_inside_blocks_a_dropped_step_wrote()
    if ending != "aborted-while-running":
        _preempt(engine, request)
        engine.pause()
    if ending == "preempted":
        engine.version = 3
        engine.close()
        assert _versions(engine, engine.listings["r"]) == [1] * 64 + [3] * 35
        return
    output = _abort(engine, "r")
    engine.pause()
    # It lists what its last run computed: blocks keep their first copies.
    assert _versions(engine, output.aux_objects) == [1] * 32 + [2] * 8
    engine.close()


def test_request_preempted_with_rejected_drafts_in_flight_computes_its_block_again():
    engine = _Engine(async_scheduling=True, drafts=2)
    scheduler = engine.scheduler
    engine.version = 1
    request = _add(engine, "r", 40, 20, token=1)
    prefill, drafts = engine.schedule(), engine.schedule()
    assert drafts.num_scheduled_tokens == {"r": 3}
    engine.step(prefill, {"r": [1]})
    _preempt(engine, request)
    engine.step(drafts, {"r": [1, 1]})
    assert request.num_tokens == 43
    engine.pause()
    assert scheduler.reset_prefix_cache()
    engine.version = 2
    resumed = engine.schedule()
    assert resumed.num_scheduled_tokens == {"r": 43}
    engine.step(resumed, {"r": [1]})
    _preempt(engine, request)
    engine.pause()
    engine.version = 3
    engine.close()
    assert _versions(engine, engine.listings["r"]) == [1] * 32 + [3] * 27


@pytest.mark.parametrize("lookahead", [0, 1], ids=["block-hashed", "block-unhashed"])
def test_listing_ends_before_the_position_a_step_in_flight_computes(lookahead):
    # Async scheduling: block 2 is completed, and hashed, by a step scheduled
    # before its hash was known, and `r` stops on the step in flight, whose
    # output is never processed: block 2 is not cached, and it is written
    # once hashed.
    engine = _Engine(async_scheduling=True)
    request = _add(engine, "r", 46, 10, stop_token_ids=[7], lookahead=lookahead)
    prefill, first_decode = engine.schedule(), engine.schedule()
    engine.step(prefill, {"r": [1]})
    second_decode = engine.schedule()
    assert second_decode.num_scheduled_tokens == {"r": 1}
    engine.step(first_decode, {"r": [7]})
    assert request.num_tokens == request.num_computed_tokens == 48
    engine.step(second_decode, {"r": [1]})
    engine.close()
    assert _spans(engine.listings["r"]) == [(0, 16), (16, 32), (32, 47)]
    hashes = request.block_hashes
    assert len(hashes) == 3 - lookahead
    # Every hashed block it computed is written, whatever the prefix cache
    # holds: block 2's hash came after the step computing it was scheduled.
    pool = engine.scheduler.kv_cache_manager.block_pool
    assert lookahead or not pool.get_cached_block(hashes[2], [0])
    blocks = set(engine.client.objects) - _tails(engine.client.objects)
    assert blocks == set(_block_keys(request))


@pytest.mark.parametrize("still_running", [False, True], ids=["finished", "running"])
def test_pause_hands_the_writes_due_to_the_writer(still_running):
    # Async scheduling: `r` finishes in the last step before a pause, and
    # `s` keeps running with block 1 completed by that step. The pause
    # queues their writes without waiting for them; nothing is scheduled
    # until the resume.
    engine = _Engine(async_scheduling=True)
    request = _add(engine, "r", 46, 2)
    if still_running:
        other = _add(engine, "s", 31, 100, token=1)
    for _ in range(100):
        if request.is_finished():
            break
        engine.step(engine.schedule(), {"r": [0], "s": [1]})
    assert request.is_finished()
    release = threading.Event()
    put_batch = engine.client.put_batch
    engine.client.put_batch = lambda *args: release.wait(10) and put_batch(*args)
    start = time.monotonic()
    _engine_core(engine).pause_scheduler(mode="keep", clear_cache=False)
    # The pause queued the writes without waiting for the store.
    assert time.monotonic() - start < 1.0
    release.set()
    engine.wait_for_writes()
    keys = {key for key, _, _ in engine.listings["r"]}
    assert len(keys) == 3 and keys <= set(engine.client.objects)
    if still_running:
        settled = other.num_computed_tokens - other.num_in_flight_tokens
        blocks = _block_keys(other, settled // _BLOCK)
        assert len(blocks) == 2
        assert set(blocks) <= set(engine.client.objects)
    engine.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    engine.worker.close()


def test_external_hit_inside_a_kv_block_is_refused():
    engine = _Engine()
    (request,) = _requests(["r"], 40, 1, prompt_start=40)
    assert engine.connector.cap_external_hit(request, 0, 8) == 0
    assert engine.connector.cap_external_hit(request, 8, 8) == 8
    assert engine.connector.cap_external_hit(request, 0, 16) == 16
    # Past the hashed positions.
    assert engine.connector.cap_external_hit(request, 16, 32) == 0
    engine.worker.close()


def test_requests_failing_on_a_failed_load_list_nothing():
    # `a` computes over a block whose load fails, `b` over `a`'s blocks; both
    # have a step in flight when the failure fails them. `_check` reads every
    # stored object: none was computed over the load.
    engine = _Engine(
        _connector_scheduler(async_scheduling=True), kv_load_failure_policy="fail"
    )
    _turn(engine, "first", 40, version=1)
    assert engine.scheduler.reset_prefix_cache()
    engine.fail_loads_from = 16
    engine.version = 9
    a = _add(engine, "a", 64, 4)
    load = engine.schedule()
    b = _add(engine, "b", 96, 4)
    shared = engine.schedule()
    assert shared.num_scheduled_tokens["b"] == 32
    outputs = engine.step(load, {"a": [0]})
    assert a.status == b.status == RequestStatus.FINISHED_ERROR
    assert [(outputs[r].finish_reason, outputs[r].aux_objects) for r in ("a", "b")] == [
        (FinishReason.ERROR, None)
    ] * 2
    engine.step(shared)
    engine.fail_loads_from = None
    _turn(engine, "c", 96, version=2)
    # Their runs ended in the worker.
    assert not engine.worker._requests.keys() & {"a", "b"}
    engine.close()
    assert 9 not in _versions(engine, engine.listings["c"])


@pytest.mark.parametrize("client", [False, True], ids=["engine-abort", "client-abort"])
def test_request_aborted_after_a_recompute_of_its_failed_load_lists_no_tail(client):
    # Recompute policy, async scheduling: `r` hits blocks 0 and 1 from the
    # connector, computes [32, 64) and has a decode step in flight when the
    # load of block 1 is reported failed, which takes it back to 16. Aborted
    # before it computes again, its run holds no rows below 32: it lists its
    # blocks and no tail, and nothing is written from rows the run never had.
    engine = _Engine(_connector_scheduler(async_scheduling=True))
    _turn(engine, "first", 40, version=1)
    assert engine.scheduler.reset_prefix_cache()
    engine.fail_loads_from = 16
    request = _add(engine, "r", 64, 4)
    load, decode = engine.schedule(), engine.schedule()
    assert load.num_scheduled_tokens["r"] == 32
    engine.step(load, {"r": [0]})
    assert request.num_computed_tokens < 32
    engine.fail_loads_from = None
    if client:
        _engine_core(engine).abort_requests(["r"])
    else:
        assert _spans(_abort(engine, "r").aux_objects) == [(0, 16), (16, 32)]
    engine.step(decode)
    engine.close()
    assert _tails(engine.client.objects) == _tails(_keys(engine, "first"))


def test_request_aborted_after_a_reset_dropped_its_run_lists_its_blocks():
    # `a` writes its three prompt blocks when it is preempted with its last
    # chunk in flight, whose output is dropped. A reset drops its run; aborted
    # before it runs again, it lists the blocks that end before its last
    # position and queues no tail, which no run could write.
    engine, _, last = _preempted_with_its_last_chunk_in_flight(drop=True)
    engine.step(last, {"a": [0]})
    core = _engine_core(engine)
    core.pause_scheduler(mode="keep")
    output = _abort(engine, "a")
    core.resume_scheduler()
    engine.close()
    assert _spans(output.aux_objects) == [(0, 16), (16, 32)]


def test_request_ending_with_an_error_writes_its_settled_blocks():
    # Async scheduling: `r` fails with a decode step in flight, right after
    # its prefill settled blocks 0 and 1. The prefix cache keeps them, so
    # `next` hits them and lists them: they are written.
    engine = _Engine(async_scheduling=True)
    request = _add(engine, "r", 40, 8)
    prefill, decode = engine.schedule(), engine.schedule()
    engine.step(prefill, {"r": [0]})
    engine.scheduler.finish_requests("r", RequestStatus.FINISHED_ERROR)
    engine.step(decode, {"r": [0]})
    _turn(engine, "next", 40)
    assert engine.hits["next"] == 32
    assert "r" not in engine.worker._requests
    engine.close()
    assert _keys(engine, "next")[:2] == _block_keys(request, 2)


@pytest.mark.parametrize(
    "status,max_tokens,sampled,sibling",
    [
        (RequestStatus.FINISHED_ABORTED, 1, {"r": [1]}, False),
        (RequestStatus.FINISHED_ERROR, 8, {}, False),
        (RequestStatus.FINISHED_ERROR, 8, {"next": [0]}, True),
    ],
    ids=["aborted", "later", "same-step"],
)
def test_request_ending_before_its_output_writes_its_blocks(
    status, max_tokens, sampled, sibling
):
    # The prefix cache holds `r`'s two full prompt blocks from the step that
    # computes them; `r` fails, or is aborted, before that step's output is
    # processed, and still writes them. `next` hits them: in the same step as
    # `r`, or later.
    engine = _Engine()
    requests = _requests(["r"], 40, max_tokens) + _requests(["next"], 40, 1)
    for request in requests[: 1 + sibling]:
        engine.scheduler.add_request(request)
    prefill = engine.schedule()
    engine.scheduler.finish_requests("r", status)
    engine.step(prefill, sampled)
    if not sibling:
        engine.scheduler.add_request(requests[1])
        hit = engine.schedule()
        assert hit.num_scheduled_tokens == {"next": 8}
        engine.step(hit, engine.complete(hit))
    engine.close()
    assert engine.hits["next"] == 32 and "r" not in engine.listings
    assert len(engine.listings["next"]) == 3
    assert _keys(engine, "next")[:2] == _block_keys(requests[0], 2)


def test_namespace_keeps_its_writes_after_a_failed_batch():
    # The store fails the first batch: its keys are missing, and later
    # objects of the namespace are still written and readable.
    engine = _Engine()
    put_batch = engine.client.put_batch
    failed: list[list[str]] = []

    def put(keys, values, config):
        if not failed:
            failed.append(list(keys))
            raise RuntimeError("put failed")
        return put_batch(keys, values, config)

    engine.client.put_batch = put
    _turn(engine, "first", 40)
    engine.pause()
    _turn(engine, "second", 40, token=1)
    engine.pause()
    assert failed and not set(failed[0]) & set(engine.client.objects)
    keys, end = _keys(engine, "second"), engine.listings["second"][-1][2]
    assert len(read_aux_rows(engine.store, keys, "routed_experts", timeout_s=0)) == end
    engine.worker.close()


def test_object_run_writes_the_bytes_of_the_rows_in_position_order():
    # A prefill computes two whole blocks and opens block 2, which decode
    # steps fill; its tail is written before the block.
    rows = np.arange(48 * 6, dtype=np.uint8).reshape(48, 3, 2)
    run = _ObjectRun(_BLOCK)
    run.add(0, rows[:40])
    for position in range(40, 47):
        run.add(position, rows[position : position + 1])
    run.writes = [("b0", 0, 16), ("b1", 16, 32), ("t", 32, 47), ("t0", 0, 40)]
    objects = run.take_ready()
    run.add(47, rows[47:])
    run.writes = [("b2", 32, 48)]
    objects += run.take_ready()
    spans = [
        ("b0", 0, 16),
        ("b1", 16, 32),
        ("t", 32, 47),
        ("t0", 0, 40),
        ("b2", 32, 48),
    ]
    assert [(key, save(fields)) for key, fields in objects] == [
        (key, save({"routed_experts": rows[start:end]})) for key, start, end in spans
    ]
    assert list(run.blocks) == [2]


def test_object_run_recompute_leaves_taken_objects_unchanged():
    # A failed load computes again positions of a block already taken.
    run = _ObjectRun(_BLOCK)
    run.add(32, np.ones((16, 3, 2), dtype=np.uint8))
    run.writes = [("b2", 32, 48)]
    [(_, fields)] = run.take_ready()
    run.add(16, np.full((24, 3, 2), 2, dtype=np.uint8))
    assert (fields["routed_experts"] == 1).all()
    assert (run.blocks[2][:8] == 2).all() and (run.blocks[2][8:] == 1).all()


@pytest.mark.parametrize("rows_first", [False, True], ids=["on-output", "on-step"])
def test_worker_puts_the_objects_of_a_step_at_once(rows_first):
    # Two namespace requests complete a block each in one step: the worker
    # puts them in one batch, so the step waits for the writer at most once,
    # whether their rows or their writes arrive last.
    client, store = fake_store()
    worker = _worker(store)
    puts = []
    put = worker._writer.put

    def record_put(objects):
        puts.append([key for key, _ in objects])
        return put(objects)

    worker._writer.put = record_put
    worker.begin_step(
        AuxOutputConnectorMetadata(
            0,
            {},
            {},
            (),
            namespace_runs=["r", "s"],
            object_block_size=_BLOCK,
        )
    )
    rows = torch.zeros((32, 3, 2), dtype=torch.uint8)
    worker._capturer.snapshot_routing_data.return_value = rows
    pending = worker.prepare_output(_input_batch(["r", "s"], [0, 0], [16, 16]))
    writes = {"r": [("job/g/1/a", 0, 16)], "s": [("job/g/2/b", 0, 16)]}
    metadata = AuxOutputConnectorMetadata(0, {}, {}, (), object_writes=writes)
    pending.enqueue_cpu_copy(np.array([1, 1]), np.array([0, 0]))
    if rows_first:
        pending.process_output()
        worker.begin_step(metadata)
    else:
        worker.begin_step(metadata)
        pending.process_output()
    worker.close()
    assert puts == [["job/g/1/a", "job/g/2/b"]]
    assert sorted(client.objects) == ["job/g/1/a", "job/g/2/b"]


def test_step_scheduled_behind_rejected_drafts_keeps_rows_in_position_order():
    # Async scheduling with drafts: the step at 40 verifies two drafts and
    # rejects one, while the next step, scheduled at 43, computes from 42.
    worker = _worker(fake_store()[1])
    metadata = AuxOutputConnectorMetadata(
        0, {}, {}, (), namespace_runs=["r"], object_block_size=_BLOCK
    )
    steps = []
    for token_start, positions, rejected in [
        (0, range(40), 0),
        (40, range(40, 43), 1),
        (43, range(42, 45), 0),
    ]:
        worker.begin_step(metadata)
        metadata = AuxOutputConnectorMetadata(0, {}, {}, ())
        rows = np.zeros((len(positions), 3, 2), dtype=np.uint8)
        rows[:, 0, 0] = positions
        worker._capturer.snapshot_routing_data.return_value = torch.from_numpy(rows)
        batch = _input_batch(["r"], [token_start], [len(positions)])
        steps.append((worker.prepare_output(batch), rejected))
        if token_start == 0:
            # The next two steps are in flight together.
            pending, rejected = steps.pop()
            pending.enqueue_cpu_copy(np.array([1]), np.array([rejected]))
            pending.process_output()
    for pending, rejected in steps:
        pending.enqueue_cpu_copy(np.array([3 - rejected]), np.array([rejected]))
        pending.process_output()
    run = worker._requests["r"].run
    run.writes = [("t", 32, 45)]
    ((_, fields),) = run.take_ready()
    assert fields["routed_experts"][:, 0, 0].tolist() == list(range(32, 45))
    worker.close()


def _kv_cache_manager(
    manager_types=(FullAttentionManager,),
    full_attention_group_id=None,
    partial_hash_hits=False,
    prefix_replay_tokens=0,
    caching=True,
    block_sizes=None,
):
    managers = [object.__new__(manager_type) for manager_type in manager_types]
    for manager, block_size in zip(
        managers, block_sizes or [2 * _BLOCK] * len(managers), strict=True
    ):
        manager.block_size = block_size
    return SimpleNamespace(
        enable_caching=caching,
        coordinator=SimpleNamespace(
            single_type_managers=managers,
            full_attention_group_id=full_attention_group_id,
            enable_partial_hash_hits=partial_hash_hits,
        ),
        block_pool=SimpleNamespace(hash_block_size=_BLOCK),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(prefix_replay_tokens=prefix_replay_tokens)
                for _ in managers
            ]
        ),
    )


@pytest.mark.parametrize(
    "key_prefix,kwargs,error",
    [
        (
            "job",
            {
                "manager_types": (SlidingWindowManager, FullAttentionManager),
                "block_sizes": (_BLOCK, 2 * _BLOCK),
                "full_attention_group_id": 1,
            },
            (1, 32),
        ),
        ("job", {"partial_hash_hits": True}, (0, 16)),
        (None, {"prefix_replay_tokens": 64, "caching": False}, (0, 32)),
        ("job", {"prefix_replay_tokens": 64}, "--no-swa-bounded-replay"),
        ("job", {"caching": False}, "needs prefix caching"),
    ],
    ids=[
        "sliding-window-alongside",
        "partial-hash-hits",
        "no-key-prefix",
        "prefix-replay",
        "no-prefix-caching",
    ],
)
def test_scheduler_connector_with_a_key_prefix_needs_prefix_caching_only(
    key_prefix, kwargs, error
):
    kv_cache_manager = _kv_cache_manager(**kwargs)
    if isinstance(error, tuple):
        # Accepted: the anchor group and the positions per object.
        connector = AuxOutputSchedulerConnector(kv_cache_manager, 4, key_prefix)
        assert (connector._group_id, connector._object_size) == error
        return
    with pytest.raises(ValueError, match=re.escape(error)):
        AuxOutputSchedulerConnector(kv_cache_manager, 4, key_prefix)


def test_hashes_looking_ahead_three_tokens_key_the_blocks_they_cover():
    # Block 1's hash covers three tokens past it: `r` decodes them, writes
    # the block, and `next` hits it.
    scheduler = create_scheduler(enable_prefix_caching=True)
    scheduler.kv_cache_manager.block_pool.num_hash_lookahead_tokens = 3
    engine = _Engine(scheduler)
    _turn(engine, "r", 30, 6, version=1, lookahead=3)
    _turn(engine, "next", 64, version=2, lookahead=3)
    engine.close()
    assert _spans(engine.listings["r"]) == [(0, 16), (16, 32), (32, 35)]
    assert engine.hits["next"] == 32
    assert _keys(engine, "next")[:2] == _keys(engine, "r")[:2]


def _hashed_scheduler(block_size, mamba=True):
    # A full-attention group and a Mamba group, or a sliding-window
    # group, of `block_size` tokens, hashed every `_BLOCK` tokens. With the
    # Mamba group, hits may end inside a KV block.
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=8192,
            max_model_len=8192,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=ModelConfig(
            model="facebook/opt-125m",
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            skip_tokenizer_init=True,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            enable_prefix_caching=True,
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = 100
    spec = AttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    mamba_spec = MambaSpec(
        block_size=block_size, shapes=((1, 1),), dtypes=(torch.float32,)
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            new_kv_cache_group(["fa"], spec),
            new_kv_cache_group(
                ["mamba"], mamba_spec, MambaManager.config(enable_prefix_caching=True)
            )
            if mamba
            else new_kv_cache_group(
                ["swa"], spec, SlidingWindowManager.config(sliding_window=1024)
            ),
        ],
    )
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=_BLOCK,
        collect_stats=True,
    )


def test_mamba_hybrid_with_partial_hash_hits_writes_an_object_per_hash_block():
    engine = _Engine(_hashed_scheduler(4 * _BLOCK))
    assert engine.scheduler.kv_cache_manager.coordinator.enable_partial_hash_hits
    _turn(engine, "first", 100, version=1)
    _turn(engine, "second", 120, version=2)
    engine.close()
    spans = [(start, start + _BLOCK) for start in range(0, 96, _BLOCK)]
    assert _spans(engine.listings["first"]) == [*spans, (96, 100)]
    hit = engine.hits["second"]
    # The hit ends inside a KV block.
    assert hit > 0 and hit % _BLOCK == 0 and hit % (4 * _BLOCK)
    assert (
        _keys(engine, "second")[: hit // _BLOCK]
        == (_keys(engine, "first")[: hit // _BLOCK])
    )
    # The positions after the hit hold its own rows.
    assert set(_versions(engine, engine.listings["second"])[hit:]) == {2}


def test_sliding_window_hybrid_with_drafts_lists_first_copies():
    # A sliding-window MoE layout (full attention alongside a 32-token
    # window) with drafts: steps accept 0 to 2 of them, and a later turn
    # hits the blocks the first one wrote.
    spec = AttentionSpec(
        block_size=_BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    scheduler = create_scheduler(
        async_scheduling=True,
        enable_prefix_caching=True,
        num_speculative_tokens=2,
        speculative_method="ngram_gpu",
        mamba_layer_config=SlidingWindowManager.layer_config(
            spec, sliding_window=2 * _BLOCK
        ),
    )
    engine = _Engine(scheduler)

    def turn(request_id, num_tokens, version):
        engine.version = version
        _add(engine, request_id, num_tokens, 20)
        for i in range(1000):
            if not engine.scheduler.has_requests():
                return
            step = engine.schedule()
            drafts = step.scheduled_spec_decode_tokens
            engine.step(
                step,
                {
                    r: [1] * (1 + i % (len(drafts.get(r, ())) + 1))
                    for r in engine.complete(step)
                },
            )
        raise AssertionError("the engine did not converge")

    turn("first", 70, 1)
    turn("second", 100, 2)
    engine.close()
    hit = engine.hits["second"]
    assert hit >= 4 * _BLOCK
    assert (
        _keys(engine, "second")[: hit // _BLOCK]
        == (_keys(engine, "first")[: hit // _BLOCK])
    )
    assert set(_versions(engine, engine.listings["second"])[:hit]) == {1}


def test_object_of_a_block_hashed_twice_is_keyed_by_its_last_hash():
    engine = _Engine(_hashed_scheduler(2 * _BLOCK, mamba=False))
    _turn(engine, "first", 80, version=1)
    _turn(engine, "second", 100, version=2)
    engine.close()
    assert _spans(engine.listings["first"]) == [(0, 32), (32, 64), (64, 80)]
    assert _keys(engine, "first")[1] == _block_keys(engine.requests["first"])[3]
    assert engine.hits["second"] == 64
    assert _keys(engine, "second")[:2] == _keys(engine, "first")[:2]


def test_request_loading_its_whole_prompt_lists_the_objects_it_hit():
    # The connector loads the whole prompt asynchronously, and the base
    # computes its last position again: the run starts at the end of the
    # hit, and the row it computes again is neither written nor listed.
    engine = _Engine(_connector_scheduler())
    _turn(engine, "first", 40, version=1)
    assert engine.scheduler.reset_prefix_cache()
    engine.async_loads = True
    _turn(engine, "second", 32, 20, version=2)
    engine.close()
    assert engine.hits["second"] == 31
    assert _keys(engine, "second")[:2] == _keys(engine, "first")[:2]
    assert _versions(engine, engine.listings["second"])[:32] == [1] * 32


def test_request_preempted_and_aborted_after_loading_its_whole_prompt():
    # Its run starts at the end of the hit and holds one row, of the last
    # prompt position: it lists the blocks before it, and no tail.
    engine = _Engine(_connector_scheduler())
    _turn(engine, "first", 40, version=1)
    assert engine.scheduler.reset_prefix_cache()
    engine.async_loads = True
    engine.version = 2
    request = _add(engine, "second", 32, 20)
    while "second" not in (step := engine.schedule()).num_scheduled_tokens:
        engine.step(step)
    # Aborted before the output of the step that computed position 31.
    _preempt(engine, request)
    output = _abort(engine, "second")
    engine.step(step, {"second": [0]})
    engine.pause()
    engine.close()
    assert engine.hits["second"] == 31
    assert _spans(output.aux_objects) == [(0, 16)]
    assert _versions(engine, output.aux_objects) == [1] * 16
