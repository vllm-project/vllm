# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent state end to end, with the dummy-model plugin's stub.

Every engine here is built with its real constructors from the plugin's
`StubPersistentStateForCausalLM` (facebook/opt-125m config and tokenizer,
dummy weights), whose output depends only on the state slot it is given and
its inputs. The in-process engine core exposes the scheduler and the worker;
`LLM` and `AsyncLLM` are the public entries. See
docs/design/persistent_state.md for the design.

Each in-process engine test starts its own engine in a new process; behaviours
that need the same stream state share one test where a second startup would
only repeat the setup.
"""

import asyncio
import gc
import logging
import os
import re
import threading
import time
from contextlib import contextmanager, nullcontext

import pytest
import torch

from tests.utils import call_in_spawned_process, create_new_process_for_each_test
from tests.v1.core.utils import mentions_number
from vllm import LLM, SamplingParams
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.platforms import current_platform
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.request import RequestStatus
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

if not current_platform.is_cuda():
    pytest.skip(reason="runs engines on a CUDA device", allow_module_level=True)

MiB = 1 << 20
STATE_LAYER = "state"
FIRST_INPUT = [10, 20, 30]
SECOND_INPUT = [40, 50]
MAX_STEPS = 64
TIMEOUT = 60
RESET_REFUSAL = (
    "persistent state: {n} state slot(s) are held, and resetting running "
    "requests would end their streams. Abort the streams and wait until they "
    "have retired, then reset again."
)
PAUSE_REFUSAL = (
    "persistent state: {n} state slot(s) are held, and clearing the cache would "
    "end their streams. Pause with clear_cache=False; to clear the cache, abort "
    "the streams, wait until they have retired, then reset the prefix cache."
)
RETRY_REFUSAL = (
    "persistent state: {n} state slot(s) are still retiring; retry the reset "
    "once they have retired."
)
RELEASE_REFUSAL = (
    "persistent state: the KV cache holds per-stream state, so it cannot be "
    "released while the engine runs."
)


@pytest.fixture(autouse=True)
def _dummy_model_plugin(monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "register_dummy_model")
    # In-process engines: the engine core, the executor and the worker share
    # this process, so the scheduler and the model are reachable.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


# What the persistent-state defaults resolve to, set explicitly where a test
# is about something else.
RESOLVED_DEFAULTS = dict(
    async_scheduling=False, enable_extensible_kv_cache=False, enforce_eager=True
)


def _engine_kwargs(path, **overrides) -> dict:
    """Arguments for LLM, EngineArgs and AsyncEngineArgs alike."""
    return dict(
        model=path,
        load_format="dummy",
        enable_prefix_caching=False,
        max_model_len=64,
        max_num_batched_tokens=512,
        **{"max_num_seqs": 2, **overrides},
    )


def _engine_core(path, **overrides) -> EngineCore:
    args = EngineArgs(**_engine_kwargs(path, **{**RESOLVED_DEFAULTS, **overrides}))
    return EngineCore(
        vllm_config=args.create_engine_config(),
        executor_class=UniProcExecutor,
        log_stats=False,
    )


def _model(core: EngineCore):
    return core.model_executor.driver_worker.worker.model_runner.model


def _stub_events() -> list:
    from vllm_add_dummy_model.my_persistent_state import EVENTS

    return EVENTS


def _expected_turns(vocab_size: int, eos_token_id: int) -> list[list[int]]:
    """The tokens the stub emits for FIRST_INPUT then SECOND_INPUT on one slot."""
    from vllm_add_dummy_model.my_persistent_state import expected_tokens

    first = expected_tokens(0, FIRST_INPUT[-1], vocab_size, eos_token_id)
    second = expected_tokens(len(first), SECOND_INPUT[-1], vocab_size, eos_token_id)
    return [first, second]


# The LLM regression: slots reused across prompts, with the default
# configuration and with the values it resolves to passed explicitly.


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize(
    "configuration",
    [
        # The persistent-state defaults left unset.
        "defaults",
        # Generation and slot reuse only, with nothing to resolve.
        "explicit",
    ],
)
def test_llm_generates_the_stub_tokens_on_reused_slots(
    dummy_state_model_path, configuration
):
    """With max_num_seqs=2, six greedy prompts each produce the stub's tokens
    from a fresh counter: a slot reused without its initializer, or held by
    two requests at once, would change them. With the persistent-state
    defaults left unset, they resolve to synchronous scheduling, no extensible
    KV cache and no runner CUDA graphs, and releasing the KV cache is refused.
    With those values passed explicitly (and a pool of three pages), only the
    generation and the slot reuse are checked."""
    from vllm_add_dummy_model.my_persistent_state import expected_tokens

    explicit = configuration == "explicit"
    # What the defaults resolve to on the non-eager path, with the pool that
    # planning sizes for max_num_seqs=2 (max_num_seqs + 1 pages).
    overrides: dict = (
        dict(
            async_scheduling=False,
            enable_extensible_kv_cache=False,
            compilation_config={"cudagraph_mode": "NONE"},
            enforce_eager=False,
            num_gpu_blocks_override=3,
        )
        if explicit
        else {}
    )
    llm = LLM(**_engine_kwargs(dummy_state_model_path, **overrides))
    config = llm.llm_engine.vllm_config
    assert config.scheduler_config.async_scheduling is False
    assert config.cache_config.enable_extensible_kv_cache is False
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert config.model_config.enforce_eager is False

    # Without multiprocessing, LLM runs the engine core in this process.
    client = llm.llm_engine.engine_core
    assert isinstance(client, InprocClient)
    model = _model(client.engine_core)
    prompts = [[11 * i + 1, 11 * i + 2, 11 * i + 3] for i in range(6)]
    params = SamplingParams(temperature=0, max_tokens=8)
    outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    assert [list(output.outputs[0].token_ids) for output in outputs] == [
        expected_tokens(0, prompt[-1], model.vocab_size, model.eos_token_id)
        for prompt in prompts
    ]
    assert {output.outputs[0].finish_reason for output in outputs} == {"stop"}

    if not explicit:
        with pytest.raises(RuntimeError) as refused:
            llm.release_kv_cache_memory()
        assert str(refused.value) == RELEASE_REFUSAL


# Startup.


@create_new_process_for_each_test("spawn")
def test_engine_refuses_more_slots_than_the_fixed_budget_holds(
    dummy_state_model_path,
):
    """kv_cache_memory_bytes of three 1 MiB pages under max_num_seqs=4 (five
    pages) fails startup; the message states the planned and fitting pages,
    the page size and the largest max_num_seqs that fits."""
    with pytest.raises(ValueError) as rejected:
        _engine_core(
            dummy_state_model_path, max_num_seqs=4, kv_cache_memory_bytes=3 * MiB
        )
    message = str(rejected.value)
    for figure in (5, 3, 2):
        assert mentions_number(message, figure), (figure, message)
    assert "max_num_seqs" in message, message


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("override", [0, 1])
def test_engine_refuses_an_override_below_two_pages_before_profiling(
    dummy_state_model_path, override
):
    """An override below two pages holds no slot: startup fails with a
    ValueError saying so, before memory profiling allocates any temporary
    pool, so no state pool, temporary or serving, was ever bound."""
    events = _stub_events()
    events.clear()
    with pytest.raises(ValueError) as rejected:
        _engine_core(dummy_state_model_path, num_gpu_blocks_override=override)
    lower = str(rejected.value).lower()
    assert "slot" in lower and re.search(r"\b(no|zero|0|none)\b", lower), lower
    assert [e for e in events if e[0] == "initialize_state_slots"] == []


@create_new_process_for_each_test("spawn")
def test_engine_refuses_an_undeclared_state_layer(dummy_undeclared_state_model_path):
    """A model that builds a persistent-state layer without declaring
    `supports_persistent_state` fails startup with a ValueError naming the
    class attribute, before memory profiling runs the model."""
    events = _stub_events()
    events.clear()
    with pytest.raises(ValueError, match="supports_persistent_state"):
        _engine_core(dummy_undeclared_state_model_path)
    assert [e for e in events if e[0] == "forward"] == []


@create_new_process_for_each_test("spawn")
def test_declared_model_gets_prepare_step_before_its_slots_are_bound(
    dummy_state_model_path,
):
    """The profiling and dummy runs before the serving bindings exist call
    `prepare_step` on a declared model, with no slots bound."""
    events = _stub_events()
    events.clear()
    core = _engine_core(dummy_state_model_path)
    try:
        serving = events.index(("initialize_state_slots",))
        early = [e for e in events[:serving] if e[0] == "prepare_step"]
        assert early, events
        assert all(e[1] or e[2] for e in early), early
        assert all(e[3] is False for e in early), early
    finally:
        core.shutdown()


# A resumable stream through the in-process engine core: cache clears never
# end a live stream.


def _params(core: EngineCore, **kwargs) -> SamplingParams:
    params = SamplingParams(temperature=0, **{"max_tokens": 8, **kwargs})
    params.update_from_generation_config({}, _model(core).eos_token_id)
    return params


def _feed(core, request_id, tokens, resumable=True, mm_features=None, **params):
    request = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(tokens),
        mm_features=mm_features,
        sampling_params=_params(core, **params),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
    )
    core.add_request(*core.preprocess_add_request(request))


def _run_turn(core, request_id) -> list[int]:
    """Step until the stream's turn ends (an output with a finish reason)."""
    tokens: list[int] = []
    for _ in range(MAX_STEPS):
        outputs, _ = core.step_fn()
        for batch in outputs.values():
            for output in batch.outputs:
                if output.request_id != request_id:
                    continue
                tokens.extend(output.new_token_ids)
                if output.finish_reason is not None:
                    return tokens
    raise AssertionError(f"{request_id} did not end its turn in {MAX_STEPS} steps")


def _retiring(core) -> list[str]:
    return [r.request_id for r in core.scheduler.state_retirements]


def _manager(core):
    return core.scheduler.kv_cache_manager.coordinator.state_managers[STATE_LAYER]


def _snapshot(core):
    """Everything a refused clear must leave as it is."""
    scheduler = core.scheduler
    manager = _manager(core)
    return (
        {rid: manager.get(rid) for rid in scheduler.requests},
        manager.num_free_slots,
        {rid: request.status for rid, request in scheduler.requests.items()},
        [request.request_id for request in scheduler.running],
        _retiring(core),
        scheduler.pause_state,
    )


def _park(core, request_id="stream"):
    _feed(core, request_id, FIRST_INPUT)
    turn = _run_turn(core, request_id)
    assert core.scheduler.requests[request_id].status == (
        RequestStatus.WAITING_FOR_STREAMING_REQ
    )
    return turn


def _bind(core, request_id="stream"):
    """One step of a long request: it is running and its slot is bound."""
    _feed(
        core, request_id, FIRST_INPUT, resumable=False, max_tokens=50, ignore_eos=True
    )
    core.step_fn()
    assert core.scheduler.requests[request_id].status == RequestStatus.RUNNING
    assert _manager(core).get(request_id) is not None


def _retire(core, request_id="ended"):
    """End a parked stream; its slot is retiring until its fence is acked."""
    _park(core, request_id)
    core.abort_requests([request_id])
    assert _retiring(core) == [request_id]


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("held", ["parked", "bound"])
def test_in_process_keep_pause_clearing_the_cache_refused(dummy_state_model_path, held):
    """A keep pause that clears the cache is refused with the exact message
    while a slot is parked or bound, before anything changes."""
    core = _engine_core(dummy_state_model_path)
    try:
        if held == "parked":
            _park(core)
        else:
            _bind(core)
        before = _snapshot(core)
        with pytest.raises(RuntimeError) as refused:
            core.pause_scheduler(mode="keep", clear_cache=True)
        assert str(refused.value) == PAUSE_REFUSAL.format(n=1)
        assert _snapshot(core) == before
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("state", ["parked", "paused-without-slots"])
def test_in_process_release_refused(dummy_state_model_path, state):
    """Releasing the KV cache is refused before its pause check, while a stream
    is parked and also once a completed pause has left no slot held."""
    core = _engine_core(dummy_state_model_path)
    try:
        free = _manager(core).num_free_slots
        _park(core)
        if state == "paused-without-slots":
            assert core.pause_scheduler() is None
            assert _manager(core).num_free_slots == free
        before = _snapshot(core)
        with pytest.raises(RuntimeError) as refused:
            core.release_kv_cache_memory()
        assert str(refused.value) == RELEASE_REFUSAL
        assert _snapshot(core) == before
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_in_process_keep_pause_without_clear_preserves_the_stream(
    dummy_state_model_path,
):
    """A stream parked across a keep pause without a cache clear resumes under
    the same handle with the tokens of an unpaused stream."""
    core = _engine_core(dummy_state_model_path)
    try:
        model = _model(core)
        first = _park(core)
        handle = _manager(core).get("stream")
        assert core.pause_scheduler(mode="keep", clear_cache=False) is None
        core.resume_scheduler()
        assert _manager(core).get("stream") == handle
        _feed(core, "stream", SECOND_INPUT)
        second = _run_turn(core, "stream")
        assert [first, second] == _expected_turns(model.vocab_size, model.eos_token_id)
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("reset_connector", [True, False])
def test_in_process_abort_pause_drains_and_clears(
    dummy_state_model_path, reset_connector
):
    """The default abort pause, whatever reset_connector is, ends a parked
    stream, steps the engine until its slot has retired and clears the cache;
    a new stream then starts from the initializer."""
    core = _engine_core(dummy_state_model_path)
    try:
        model = _model(core)
        free = _manager(core).num_free_slots
        _park(core, "aborted")
        assert core.pause_scheduler(reset_connector=reset_connector) is None
        assert isinstance(core.scheduler, Scheduler)
        assert "aborted" not in core.scheduler.requests
        assert _retiring(core) == []
        assert _manager(core).num_free_slots == free
        core.resume_scheduler()
        first = _park(core, "next")
        _feed(core, "next", SECOND_INPUT)
        second = _run_turn(core, "next")
        assert [first, second] == _expected_turns(model.vocab_size, model.eos_token_id)
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("reset_running_requests", [True, False])
def test_in_process_reset_drains_retiring_slots(
    dummy_state_model_path, reset_running_requests
):
    """With no unfinished request and only retiring slots, the in-process
    reset steps the engine until no slot is retiring and then resets."""
    core = _engine_core(dummy_state_model_path)
    try:
        free = _manager(core).num_free_slots
        _retire(core)
        assert core.reset_prefix_cache(reset_running_requests=reset_running_requests)
        assert _retiring(core) == []
        assert _manager(core).num_free_slots == free
    finally:
        core.shutdown()


# A pause without a cache clear leaves the slot retiring; the queued request
# then waits in the scheduler, which a paused scheduler does not report as
# unfinished.
_ALREADY_PAUSED = pytest.mark.parametrize("paused", [None, "keep", "abort"])


def _pause_without_clear(core, paused):
    if paused is not None:
        assert core.pause_scheduler(mode=paused, clear_cache=False) is None


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("reset_running_requests", [True, False])
@pytest.mark.parametrize("paused", [None, "keep"])
def test_in_process_reset_with_a_queued_request_does_not_step(
    dummy_state_model_path, reset_running_requests, paused
):
    """With a slot retiring and a request queued, the in-process reset of
    running requests raises the retry message, and the reset without them
    returns False; neither steps the engine or changes anything, also when
    the engine is already paused and so reports no unfinished request."""
    core = _engine_core(dummy_state_model_path)
    try:
        _retire(core)
        _pause_without_clear(core, paused)
        _feed(core, "queued", FIRST_INPUT, resumable=False)
        before = _snapshot(core)
        if reset_running_requests:
            with pytest.raises(RuntimeError) as refused:
                core.reset_prefix_cache(reset_running_requests=True)
            assert str(refused.value) == RETRY_REFUSAL.format(n=1)
        else:
            assert core.reset_prefix_cache(reset_running_requests=False) is False
        assert _snapshot(core) == before
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
@_ALREADY_PAUSED
def test_in_process_clearing_keep_pause_with_a_queued_request_asks_for_a_retry(
    dummy_state_model_path, paused
):
    """With a slot retiring and a request queued, a keep pause that clears the
    cache raises the retry message before any change, whether or not the
    engine is already paused."""
    core = _engine_core(dummy_state_model_path)
    try:
        _retire(core)
        _pause_without_clear(core, paused)
        _feed(core, "queued", FIRST_INPUT, resumable=False)
        before = _snapshot(core)
        with pytest.raises(RuntimeError) as refused:
            core.pause_scheduler(mode="keep", clear_cache=True)
        assert str(refused.value) == RETRY_REFUSAL.format(n=1)
        assert _snapshot(core) == before
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_in_process_clearing_keep_pause_with_only_retiring_slots_drains(
    dummy_state_model_path,
):
    """With only a retiring slot, a keep pause that clears the cache steps the
    engine until the slot has retired, then completes."""
    core = _engine_core(dummy_state_model_path)
    try:
        free = _manager(core).num_free_slots
        _retire(core)
        assert core.pause_scheduler(mode="keep", clear_cache=True) is None
        assert _retiring(core) == []
        assert _manager(core).num_free_slots == free
        assert core.scheduler.pause_state == PauseState.PAUSED_ALL
    finally:
        core.shutdown()


def _end_with_misplaced_input(core, request_id="stream"):
    """End a parked stream with an input whose image lies past its prompt: its
    slot retires and its error output is queued for the next step."""
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        MultiModalKwargsItem,
        PlaceholderRange,
    )

    image = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        mm_position=PlaceholderRange(offset=len(SECOND_INPUT), length=1),
        identifier="image",
        modality="image",
    )
    _feed(core, request_id, SECOND_INPUT, mm_features=[image])
    assert request_id not in core.scheduler.requests
    assert _retiring(core) == [request_id]


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize(
    ("mode", "reset_connector"),
    [(None, True), ("keep", True), ("keep", False), ("abort", False)],
    ids=["reset", "keep-pause", "keep-pause-keeping-connector", "abort-pause"],
)
def test_in_process_cache_clear_delivers_a_queued_error_once(
    dummy_state_model_path, mode, reset_connector
):
    """A stream that ended with an error just before a cache clear gets its
    error output exactly once, after the clear: the steps that retire its slot
    during the reset or the clearing pause deliver nothing and drop nothing."""
    from vllm.v1.engine import FinishReason

    core = _engine_core(dummy_state_model_path)
    try:
        _park(core)
        _end_with_misplaced_input(core)
        if mode is None:
            assert core.reset_prefix_cache(reset_running_requests=True)
        else:
            assert (
                core.pause_scheduler(
                    mode=mode, clear_cache=True, reset_connector=reset_connector
                )
                is None
            )
            core.resume_scheduler()
        assert _retiring(core) == []
        ends = []
        for _ in range(3):
            outputs, _ = core.step_fn()
            ends += [
                output.finish_reason
                for batch in outputs.values()
                for output in batch.outputs
                if output.request_id == "stream" and output.finish_reason is not None
            ]
        assert ends == [FinishReason.ERROR]
    finally:
        core.shutdown()


# The multiprocess engine through AsyncLLM. A running request's slot
# stays bound from step to step, but a multiprocess client cannot order its
# pause or reset before the request finishes, so bound-slot refusals are
# tested in-process above.


def _async_llm(path, monkeypatch):
    from vllm.v1.engine.async_llm import AsyncLLM

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    return AsyncLLM.from_engine_args(
        AsyncEngineArgs(**_engine_kwargs(path, **RESOLVED_DEFAULTS))
    )


class _Stream:
    """One streaming-input request: FIRST_INPUT, then SECOND_INPUT once
    `release()` is called; `parked` is set when the first turn ends."""

    def __init__(self, llm, request_id):
        from vllm.sampling_params import RequestOutputKind

        self.llm = llm
        self.request_id = request_id
        self.gate = asyncio.Event()
        self.parked = asyncio.Event()
        self.params = SamplingParams(
            max_tokens=8, temperature=0, output_kind=RequestOutputKind.DELTA
        )
        self.task = asyncio.create_task(self._run())

    async def _inputs(self):
        # Copies: the output processor extends a request's prompt list in place.
        from vllm.engine.protocol import StreamingInput
        from vllm.inputs import TokensPrompt

        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=list(FIRST_INPUT)))
        await self.gate.wait()
        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=list(SECOND_INPUT)))

    async def _run(self):
        turns: list[list[int]] = [[]]
        final = None
        async for output in self.llm.generate(
            self._inputs(), self.params, self.request_id
        ):
            completion = output.outputs[0]
            turns[-1].extend(completion.token_ids)
            if completion.finish_reason is not None and not output.finished:
                turns.append([])
                self.parked.set()
            if output.finished:
                final = completion.finish_reason
        return [turn for turn in turns if turn], final

    async def wait_parked(self):
        await asyncio.wait_for(self.parked.wait(), TIMEOUT)

    async def finish(self):
        self.gate.set()
        return await asyncio.wait_for(self.task, TIMEOUT)


def _llm_expected_turns(llm) -> list[list[int]]:
    hf = llm.model_config.hf_config
    return _expected_turns(llm.model_config.get_vocab_size(), int(hf.eos_token_id))


@pytest.mark.asyncio
async def test_async_llm_clearing_pauses_and_reset_refused_while_parked(
    dummy_state_model_path, monkeypatch
):
    """While a stream is parked, keep and wait pauses that clear the cache and
    a reset of running requests are refused with the exact messages; the
    stream then continues with the tokens of an unrefused one."""
    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        stream = _Stream(llm, "parked")
        await stream.wait_parked()
        for mode in ("keep", "wait"):
            with pytest.raises(Exception, match=re.escape(PAUSE_REFUSAL.format(n=1))):
                await asyncio.wait_for(
                    llm.pause_generation(mode=mode, clear_cache=True), TIMEOUT
                )
        with pytest.raises(Exception, match=re.escape(RESET_REFUSAL.format(n=1))):
            await asyncio.wait_for(
                llm.reset_prefix_cache(reset_running_requests=True), TIMEOUT
            )
        turns, _ = await stream.finish()
        assert turns == _llm_expected_turns(llm)
    finally:
        llm.shutdown()


@pytest.mark.asyncio
async def test_async_llm_keep_pause_without_clear_preserves_the_stream(
    dummy_state_model_path, monkeypatch
):
    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        stream = _Stream(llm, "paused")
        await stream.wait_parked()
        await asyncio.wait_for(
            llm.pause_generation(mode="keep", clear_cache=False), TIMEOUT
        )
        await asyncio.wait_for(llm.resume_generation(), TIMEOUT)
        turns, _ = await stream.finish()
        assert turns == _llm_expected_turns(llm)
    finally:
        llm.shutdown()


@pytest.mark.asyncio
async def test_async_llm_abort_pause_drains_a_parked_stream(
    dummy_state_model_path, monkeypatch
):
    """The default abort pause ends a parked stream with an abort, completes
    once its slot has retired, and a new stream then starts from the
    initializer."""
    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        stream = _Stream(llm, "aborted")
        await stream.wait_parked()
        await asyncio.wait_for(llm.pause_generation(), TIMEOUT)
        turns, final = await asyncio.wait_for(stream.task, TIMEOUT)
        assert final == "abort"
        await asyncio.wait_for(llm.resume_generation(), TIMEOUT)
        after = _Stream(llm, "after")
        await after.wait_parked()
        assert (await after.finish())[0] == _llm_expected_turns(llm)
    finally:
        llm.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["keep", "wait"])
async def test_async_llm_clearing_pause_with_only_retiring_slots_completes(
    dummy_state_model_path, monkeypatch, mode
):
    """After a parked stream is aborted, a pause that clears the cache
    completes (once its slot has retired) and a new stream starts from the
    initializer. Whether the pause arrives before or after the retirement is
    not controlled, so the pause's repeated check at its deferred finish is
    not claimed here."""
    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        stream = _Stream(llm, "ended")
        await stream.wait_parked()
        await asyncio.wait_for(llm.abort("ended"), TIMEOUT)
        await asyncio.wait_for(
            llm.pause_generation(mode=mode, clear_cache=True), TIMEOUT
        )
        await asyncio.wait_for(llm.resume_generation(), TIMEOUT)
        # The abort ended the stream.
        assert (await asyncio.wait_for(stream.task, TIMEOUT))[1] == "abort"
        after = _Stream(llm, "after")
        await after.wait_parked()
        assert (await after.finish())[0] == _llm_expected_turns(llm)
    finally:
        llm.shutdown()


def _record_engine_inputs(llm) -> list[tuple[list[int], bool]]:
    """The prompt and resumable flag of each request AsyncLLM sends to the
    engine core, in order, copied when it is sent."""
    sent: list[tuple[list[int], bool]] = []
    add = llm.engine_core.add_request_async

    async def record(request):
        sent.append((list(request.prompt_token_ids or ()), request.resumable))
        return await add(request)

    llm.engine_core.add_request_async = record
    return sent


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["length", "stop_token"])
@pytest.mark.parametrize("later", ["after_the_end", "queued"])
async def test_async_llm_stream_ends_with_a_turn_that_stops_off_eos(
    dummy_state_model_path, monkeypatch, stop, later
):
    """A turn that stops on the length limit or a stop token finishes the
    client's stream with that finish reason, so `generate` returns. An input
    offered after the client has seen the end never reaches the engine, and
    no end-of-input request follows it; an input queued before the end is
    dropped with the request. Both slots are then free for two new streams.
    (Repetition cannot end a stub turn: the stub never emits one token twice
    in a row, and a turn has two tokens before its EOS.)"""
    from vllm_add_dummy_model.my_persistent_state import expected_tokens

    from vllm.engine.protocol import StreamingInput
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        hf = llm.model_config.hf_config
        turn = expected_tokens(
            0, FIRST_INPUT[-1], llm.model_config.get_vocab_size(), int(hf.eos_token_id)
        )
        if stop == "length":
            params: dict = {"max_tokens": 1}
            expected = (turn[:1], "length", None)
        else:
            params = {"max_tokens": 8, "stop_token_ids": [turn[1]]}
            expected = (turn[:2], "stop", turn[1])
        sent = _record_engine_inputs(llm)
        ended = asyncio.Event()

        async def inputs():
            yield StreamingInput(
                prompt=TokensPrompt(prompt_token_ids=list(FIRST_INPUT))
            )
            if later == "after_the_end":
                await ended.wait()
            yield StreamingInput(
                prompt=TokensPrompt(prompt_token_ids=list(SECOND_INPUT))
            )

        outputs = []

        async def run():
            sampling = SamplingParams(
                temperature=0, output_kind=RequestOutputKind.DELTA, **params
            )
            async for output in llm.generate(inputs(), sampling, "ended"):
                outputs.append(output)
                if output.outputs[0].finish_reason is not None:
                    ended.set()

        await asyncio.wait_for(run(), TIMEOUT)
        final = outputs[-1].outputs[0]
        assert [t for o in outputs for t in o.outputs[0].token_ids] == expected[0]
        assert outputs[-1].finished
        assert (final.finish_reason, final.stop_reason) == expected[1:]
        assert not any(o.finished for o in outputs[:-1])
        if later == "after_the_end":
            assert sent == [(FIRST_INPUT, True)]
        streams = [_Stream(llm, f"after-{i}") for i in range(2)]
        for after in streams:
            await after.wait_parked()
        for after in streams:
            assert (await after.finish())[0] == _llm_expected_turns(llm)
    finally:
        llm.shutdown()


# Stream admission: the engine reserves a stream's slot before any input.


def _admission(core, request_id="stream", **fields) -> EngineCoreRequest:
    return EngineCoreRequest(
        **{
            "request_id": request_id,
            "prompt_token_ids": None,
            "mm_features": None,
            "sampling_params": _params(core),
            "pooling_params": None,
            "arrival_time": time.time(),
            "lora_request": None,
            "cache_salt": None,
            "data_parallel_rank": None,
            "resumable": True,
            **fields,
        }
    )


def _feed_admitted(core, update_seq, tokens, *, resumable=True, request_id="stream"):
    """Send input `update_seq` of an admitted stream, or with
    `resumable=False` the close that carries no input."""
    from vllm.v1.engine import StreamAdmissionContext

    request = _admission(
        core,
        request_id,
        prompt_token_ids=list(tokens),
        resumable=resumable,
        stream_admission=StreamAdmissionContext(
            input_closed=not resumable, update_seq=update_seq
        ),
    )
    core.add_request(*core.preprocess_add_request(request))


def _steps_until_retired(core, request_id="stream"):
    """Step until a batch carries the stream's retirement notice; returns
    every output of the stream seen on the way."""
    seen = []
    for _ in range(MAX_STEPS):
        outputs, _ = core.step_fn()
        for batch in outputs.values():
            seen += [o for o in batch.outputs if o.request_id == request_id]
            if request_id in (batch.retired_streams or ()):
                return seen
    raise AssertionError(f"{request_id} did not retire in {MAX_STEPS} steps")


@create_new_process_for_each_test("spawn")
def test_engine_core_admitted_stream_runs_on_the_slot_from_admission(
    dummy_state_model_path,
):
    """An admission reserves a slot and leaves the engine with no work; the
    stream's inputs then run on that slot with the tokens of an ordinary
    stream, and after the close the engine's steps deliver the retirement
    notice to the caller of admit_stream."""
    from vllm.v1.engine import StreamAdmissionResult

    core = _engine_core(dummy_state_model_path)
    try:
        model = _model(core)
        result = core.admit_stream(_admission(core))
        assert result == StreamAdmissionResult(True, "stream")
        handle = _manager(core).get("stream")
        assert handle is not None
        assert core.step_fn() == ({}, False)
        _feed_admitted(core, 1, FIRST_INPUT)
        first = _run_turn(core, "stream")
        _feed_admitted(core, 2, SECOND_INPUT)
        second = _run_turn(core, "stream")
        assert [first, second] == _expected_turns(model.vocab_size, model.eos_token_id)
        assert _manager(core).get("stream") == handle
        _feed_admitted(core, 3, [0], resumable=False)
        (end,) = _steps_until_retired(core)
        assert end.finish_reason is not None and end.ends_stream is True
        assert _manager(core).num_free_slots == 2
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_in_process_abort_pause_that_clears_the_cache_keeps_one_retirement_notice(
    dummy_state_model_path,
):
    """An abort pause ends an admitted stream as it ends other requests: the
    in-process engine sends no output for it and steps until its scheduled
    slot has retired, dropping those steps' outputs; the stream's retirement
    notice is not dropped with them, and the steps after the pause deliver
    it exactly once."""
    core = _engine_core(dummy_state_model_path)
    try:
        assert core.admit_stream(_admission(core)).accepted
        _feed_admitted(core, 1, FIRST_INPUT)
        _run_turn(core, "stream")
        assert core.pause_scheduler() is None
        assert "stream" not in core.scheduler.requests
        assert _retiring(core) == []
        core.resume_scheduler()
        notices = []
        outputs_of_stream = []
        for _ in range(4):
            outputs, _ = core.step_fn()
            for batch in outputs.values():
                notices += batch.retired_streams or []
                outputs_of_stream += [
                    o for o in batch.outputs if o.request_id == "stream"
                ]
        assert notices == ["stream"]
        assert outputs_of_stream == []
        assert _manager(core).num_free_slots == 2
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("multiprocessing", ["0", "1"], ids=["in_process", "sync_mp"])
def test_synchronous_clients_do_not_implement_stream_admission(
    dummy_state_model_path, monkeypatch, multiprocessing
):
    """Only the async multiprocess client implements admission and the wait
    for retirement: the in-process client and the synchronous multiprocess
    client, which LLM uses, raise NotImplementedError."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", multiprocessing)
    llm = LLM(**_engine_kwargs(dummy_state_model_path, **RESOLVED_DEFAULTS))
    client = llm.llm_engine.engine_core
    assert isinstance(client, InprocClient) == (multiprocessing == "0")
    request = _raw_admission(llm.llm_engine.vllm_config)
    with pytest.raises(NotImplementedError):
        asyncio.run(client.admit_stream_async(request))
    with pytest.raises(NotImplementedError):
        asyncio.run(client.wait_stream_retired_async("raw"))


class _AdmittedStream:
    """One admitted streaming-input request on AsyncLLM: each `feed()` sends
    the next input, `close()` ends the generator; `task` collects the turns
    and the finish reason of the output that ends the stream. `admitted` is
    set when the frontend first reads the generator, which it does only
    after the admission has succeeded."""

    def __init__(self, llm, request_id):
        from vllm.sampling_params import RequestOutputKind

        self.llm = llm
        self.request_id = request_id
        self.inputs: asyncio.Queue = asyncio.Queue()
        self.turn_ended = asyncio.Event()
        self.admitted = asyncio.Event()
        self.params = SamplingParams(
            max_tokens=8, temperature=0, output_kind=RequestOutputKind.DELTA
        )
        self.task = asyncio.create_task(self._run())

    async def _inputs(self):
        from vllm.engine.protocol import StreamingInput
        from vllm.inputs import TokensPrompt

        self.admitted.set()
        while (tokens := await self.inputs.get()) is not None:
            yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=list(tokens)))

    async def _run(self):
        turns: list[list[int]] = [[]]
        final = None
        async for output in self.llm.generate(
            self._inputs(), self.params, self.request_id, stream_admission=True
        ):
            completion = output.outputs[0]
            turns[-1].extend(completion.token_ids)
            if completion.finish_reason is not None and not output.finished:
                turns.append([])
                self.turn_ended.set()
            if output.finished:
                final = completion.finish_reason
        return [turn for turn in turns if turn], final

    async def feed(self, tokens):
        self.turn_ended.clear()
        self.inputs.put_nowait(tokens)
        await asyncio.wait_for(self.turn_ended.wait(), TIMEOUT)

    async def close(self):
        self.inputs.put_nowait(None)
        return await asyncio.wait_for(self.task, TIMEOUT)


@pytest.mark.asyncio
async def test_async_llm_admitted_stream_runs_closes_and_retires(
    dummy_state_model_path, monkeypatch
):
    """An admitted stream produces the tokens of an ordinary stream for each
    input, ends with a stop output once its generator ends, and
    wait_stream_retired returns once its slot is free; the caller's request
    ID can then be admitted again."""
    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        stream = _AdmittedStream(llm, "stream")
        await stream.feed(FIRST_INPUT)
        await stream.feed(SECOND_INPUT)
        turns, final = await stream.close()
        assert turns == _llm_expected_turns(llm)
        assert final == "stop"
        await asyncio.wait_for(llm.wait_stream_retired("stream"), TIMEOUT)
        again = _AdmittedStream(llm, "stream")
        await again.feed(FIRST_INPUT)
        assert (await again.close())[0] == _llm_expected_turns(llm)[:1]
    finally:
        llm.shutdown()


@pytest.mark.asyncio
async def test_async_llm_refuses_a_stream_when_every_slot_is_held(
    dummy_state_model_path, monkeypatch
):
    """With both slots held by streams that have no input yet, a third
    admission raises StreamAdmissionError with reason capacity and HTTP
    status 503; ending a stream frees its slot for the next admission."""
    from http import HTTPStatus

    from vllm.v1.engine import StreamAdmissionError

    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        held = [_AdmittedStream(llm, f"held-{i}") for i in range(2)]
        for stream in held:
            await asyncio.wait_for(stream.admitted.wait(), TIMEOUT)
        refused = _AdmittedStream(llm, "refused")
        with pytest.raises(StreamAdmissionError) as error:
            await asyncio.wait_for(refused.task, TIMEOUT)
        assert error.value.result.reason == "capacity"
        assert error.value.http_status == HTTPStatus.SERVICE_UNAVAILABLE
        await held[0].close()
        await asyncio.wait_for(llm.wait_stream_retired("held-0"), TIMEOUT)
        admitted = _AdmittedStream(llm, "admitted")
        await admitted.feed(FIRST_INPUT)
        await admitted.close()
        await held[1].close()
    finally:
        llm.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["keep", "wait"])
async def test_async_llm_never_fed_stream_leaves_the_engine_idle_for_a_pause(
    dummy_state_model_path, monkeypatch, mode
):
    """A never-fed stream is not unfinished work: a pause that keeps or waits
    for requests completes while it holds its slot, and an admission during
    the pause is refused as engine_paused."""
    from vllm.v1.engine import StreamAdmissionError

    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        idle = _AdmittedStream(llm, "idle")
        await asyncio.wait_for(idle.admitted.wait(), TIMEOUT)
        await asyncio.wait_for(
            llm.pause_generation(mode=mode, clear_cache=False), TIMEOUT
        )
        refused = _AdmittedStream(llm, "refused")
        with pytest.raises(StreamAdmissionError) as error:
            await asyncio.wait_for(refused.task, TIMEOUT)
        assert error.value.result.reason == "engine_paused"
        await asyncio.wait_for(llm.resume_generation(), TIMEOUT)
        await idle.feed(FIRST_INPUT)
        assert (await idle.close())[0] == _llm_expected_turns(llm)[:1]
    finally:
        llm.shutdown()


@pytest.mark.asyncio
async def test_async_llm_never_fed_stream_expires_while_the_engine_is_idle(
    dummy_state_model_path, monkeypatch
):
    """The idle engine wakes at the earliest stream deadline: a stream that
    gets no input within the configured never-fed timeout ends with one abort
    output, and its slot retires."""
    from vllm.v1.engine.async_llm import AsyncLLM

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            **_engine_kwargs(
                dummy_state_model_path,
                **RESOLVED_DEFAULTS,
                stream_admission_timeouts={"never_fed_timeout": 0.5},
            )
        )
    )
    try:
        stream = _AdmittedStream(llm, "idle")
        turns, final = await asyncio.wait_for(stream.task, TIMEOUT)
        assert (turns, final) == ([], "abort")
        await asyncio.wait_for(llm.wait_stream_retired("idle"), TIMEOUT)
    finally:
        llm.shutdown()


async def _next_output(client, request_id):
    """The next engine output for `request_id`, with its batch."""
    while True:
        batch = await asyncio.wait_for(client.get_output_async(), TIMEOUT)
        for output in batch.outputs:
            if output.request_id == request_id:
                return output, batch


# The two tests below read the engine's outputs from the multiprocess client
# itself, so they build the client without AsyncLLM, whose output handler
# would consume them.


def _async_mp_client(path, monkeypatch, **overrides):
    from vllm.v1.engine.core_client import EngineCoreClient
    from vllm.v1.executor import Executor

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    config = AsyncEngineArgs(
        **_engine_kwargs(path, **RESOLVED_DEFAULTS, **overrides)
    ).create_engine_config()
    return config, EngineCoreClient.make_async_mp_client(
        config, Executor.get_class(config), log_stats=False
    )


def _raw_admission(config, request_id="raw", **fields) -> EngineCoreRequest:
    params = SamplingParams(temperature=0, max_tokens=8)
    params.update_from_generation_config(
        {}, int(config.model_config.hf_config.eos_token_id)
    )
    return EngineCoreRequest(
        **{
            "request_id": request_id,
            "prompt_token_ids": None,
            "mm_features": None,
            "sampling_params": params,
            "pooling_params": None,
            "arrival_time": time.time(),
            "lora_request": None,
            "cache_salt": None,
            "data_parallel_rank": None,
            "resumable": True,
            "external_req_id": request_id,
            **fields,
        }
    )


@pytest.mark.asyncio
async def test_multiprocess_abort_pause_output_carries_no_stream_marking(
    dummy_state_model_path, monkeypatch
):
    """The abort output that the multiprocess engine core sends for an
    admitted stream an abort pause ends has neither ends_stream nor
    stream_output; the stream's retirement notice follows."""
    from vllm.v1.engine import FinishReason

    config, client = _async_mp_client(dummy_state_model_path, monkeypatch)
    try:
        result = await asyncio.wait_for(
            client.admit_stream_async(_raw_admission(config)), TIMEOUT
        )
        assert result.accepted
        await asyncio.wait_for(
            client.pause_scheduler_async(mode="abort", clear_cache=False), TIMEOUT
        )
        output, _ = await _next_output(client, "raw")
        assert output.finish_reason == FinishReason.ABORT
        assert output.ends_stream is False
        assert output.stream_output is None
        await asyncio.wait_for(client.wait_stream_retired_async("raw"), TIMEOUT)
    finally:
        client.shutdown()


@pytest.mark.asyncio
async def test_multiprocess_engine_rejects_an_input_it_cannot_validate(
    dummy_state_model_path, monkeypatch
):
    """An input of an admitted stream that fails the engine's validation
    before preprocessing gets one rejection output with its own context, a
    validation_error RequestError and no finished request; the session then
    runs a valid input."""
    from vllm.v1.engine import FinishReason, StreamAdmissionContext, StreamOutput

    config, client = _async_mp_client(dummy_state_model_path, monkeypatch)
    try:
        await asyncio.wait_for(
            client.admit_stream_async(_raw_admission(config)), TIMEOUT
        )
        context = StreamAdmissionContext(update_seq=1)
        invalid = _raw_admission(
            config,
            prompt_token_ids=list(FIRST_INPUT),
            stream_admission=context,
            sampling_params=SamplingParams(n=2),
        )
        await client.add_request_async(invalid)
        rejection, batch = await _next_output(client, "raw")
        assert rejection.new_token_ids == []
        assert rejection.finish_reason == FinishReason.ERROR
        assert rejection.ends_stream is False
        assert rejection.stream_output == StreamOutput(context, "validation_error")
        assert rejection.error.code == "validation_error"
        assert not batch.finished_requests

        valid = _raw_admission(
            config,
            prompt_token_ids=list(FIRST_INPUT),
            stream_admission=StreamAdmissionContext(update_seq=2),
        )
        await client.add_request_async(valid)
        output, _ = await _next_output(client, "raw")
        assert output.stream_output == StreamOutput(
            StreamAdmissionContext(update_seq=2)
        )
        await client.abort_requests_async(["raw"])
        await asyncio.wait_for(client.wait_stream_retired_async("raw"), TIMEOUT)
    finally:
        client.shutdown()


def _raw_input(config, update_seq, tokens=FIRST_INPUT, **fields):
    from vllm.v1.engine import StreamAdmissionContext

    return _raw_admission(
        config,
        prompt_token_ids=list(tokens),
        stream_admission=StreamAdmissionContext(update_seq=update_seq),
        **fields,
    )


@pytest.mark.asyncio
async def test_multiprocess_rejected_input_with_abort_immediately_keeps_its_session(
    dummy_state_model_path, monkeypatch
):
    """An input that the scheduler rejects leaves its session unchanged also
    when it has abort_immediately set: the session then runs a valid input."""
    from vllm.v1.engine import StreamAdmissionContext, StreamOutput

    config, client = _async_mp_client(dummy_state_model_path, monkeypatch)
    try:
        await asyncio.wait_for(
            client.admit_stream_async(_raw_admission(config)), TIMEOUT
        )
        empty = _raw_input(config, 1, tokens=(), abort_immediately=True)
        await client.add_request_async(empty)
        rejection, batch = await _next_output(client, "raw")
        assert rejection.stream_output == StreamOutput(
            empty.stream_admission, "empty_update"
        )
        assert not batch.finished_requests
        await client.add_request_async(_raw_input(config, 2))
        output, _ = await _next_output(client, "raw")
        assert output.stream_output == StreamOutput(
            StreamAdmissionContext(update_seq=2)
        )
        await client.abort_requests_async(["raw"])
        await asyncio.wait_for(client.wait_stream_retired_async("raw"), TIMEOUT)
    finally:
        client.shutdown()


@pytest.mark.asyncio
async def test_multiprocess_paused_engine_rejections_and_their_precedence(
    dummy_state_model_path, monkeypatch
):
    """While the engine rejects new requests, an input whose session has
    ended is rejected as engine_paused with a retryable RequestError, not as
    unknown_admission, and an input that also fails validation is rejected
    as validation_error, which preprocessing reports first. A new request
    without a context keeps the ordinary paused-engine error output."""
    from vllm.v1.engine import FinishReason, StreamOutput

    config, client = _async_mp_client(dummy_state_model_path, monkeypatch)
    try:
        await asyncio.wait_for(
            client.admit_stream_async(_raw_admission(config)), TIMEOUT
        )
        await client.abort_requests_async(["raw"])
        await asyncio.wait_for(client.wait_stream_retired_async("raw"), TIMEOUT)
        await asyncio.wait_for(
            client.pause_generation_async(mode="wait", clear_cache=False), TIMEOUT
        )

        late = _raw_input(config, 1)
        await client.add_request_async(late)
        paused, batch = await _next_output(client, "raw")
        assert paused.finish_reason == FinishReason.ERROR
        assert paused.ends_stream is False
        assert paused.stream_output == StreamOutput(
            late.stream_admission, "engine_paused"
        )
        assert paused.error.code == "engine_paused" and paused.error.retryable
        assert not batch.finished_requests

        invalid = _raw_input(config, 2, sampling_params=SamplingParams(n=2))
        await client.add_request_async(invalid)
        refused, _ = await _next_output(client, "raw")
        assert refused.stream_output == StreamOutput(
            invalid.stream_admission, "validation_error"
        )
        assert refused.error.code == "validation_error"

        plain = _raw_admission(
            config, "plain", prompt_token_ids=list(FIRST_INPUT), resumable=False
        )
        await client.add_request_async(plain)
        ordinary, batch = await _next_output(client, "plain")
        assert ordinary.finish_reason == FinishReason.ERROR
        assert ordinary.stream_output is None
        assert ordinary.error.code == "engine_paused"
        assert "plain" in batch.finished_requests
    finally:
        client.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("context", [True, False], ids=["input", "context_free"])
async def test_multiprocess_preprocessing_failure_leaves_the_session_unchanged(
    dummy_state_model_path, monkeypatch, context
):
    """A request under an admitted ID whose preprocessing raises an
    unexpected error changes nothing in the session, whose next valid input
    then runs its whole turn with the tokens of a first input and parks.
    With a context the request is rejected as preprocessing_error; without
    one it gets the ordinary error output, with no stream_output and its ID
    in finished_requests.

    The trigger is sampling parameters without max_tokens: they pass the
    stream-input validation, and `Request.__init__` then fails its assert
    that max_tokens is set. If that assert goes, this test needs another
    unexpected preprocessing error."""
    from vllm.v1.engine import FinishReason, StreamAdmissionContext, StreamOutput

    config, client = _async_mp_client(dummy_state_model_path, monkeypatch)
    try:
        await asyncio.wait_for(
            client.admit_stream_async(_raw_admission(config)), TIMEOUT
        )
        failing = _raw_input(config, 1, sampling_params=SamplingParams(max_tokens=None))
        if not context:
            failing.stream_admission = None
        await client.add_request_async(failing)
        refused, batch = await _next_output(client, "raw")
        assert refused.new_token_ids == []
        assert refused.finish_reason == FinishReason.ERROR
        assert refused.ends_stream is False
        if context:
            assert refused.stream_output == StreamOutput(
                failing.stream_admission, "preprocessing_error"
            )
            assert not batch.finished_requests
        else:
            assert refused.stream_output is None
            assert "raw" in batch.finished_requests

        await client.add_request_async(_raw_input(config, 2))
        tokens: list[int] = []
        for _ in range(MAX_STEPS):
            output, _ = await _next_output(client, "raw")
            assert output.error is None
            assert output.stream_output == StreamOutput(
                StreamAdmissionContext(update_seq=2)
            )
            tokens.extend(output.new_token_ids)
            if output.finish_reason is not None:
                break
        hf_config = config.model_config.hf_config
        assert (
            tokens
            == _expected_turns(
                config.model_config.get_vocab_size(), int(hf_config.eos_token_id)
            )[0]
        )
        assert output.finish_reason == FinishReason.STOP
        assert output.ends_stream is False
        await client.abort_requests_async(["raw"])
        await asyncio.wait_for(client.wait_stream_retired_async("raw"), TIMEOUT)
    finally:
        client.shutdown()


@pytest.mark.asyncio
async def test_multiprocess_draining_shutdown_does_not_wait_for_a_never_fed_stream(
    dummy_state_model_path, monkeypatch
):
    """A never-fed stream is not unfinished work, so a shutdown that drains
    does not wait for it: on SIGTERM the engine process exits by itself with
    code 0, well inside the shutdown window and long before the stream's
    never-fed deadline; an engine that waited for the session would still be
    running when the window closes. The client gets no output for the
    session, only the engine's end.

    The never-fed deadline is set through stream_admission_timeouts so that
    it cannot end the session during the test."""
    from vllm.v1.engine.exceptions import EngineDeadError

    window = 60
    config, client = _async_mp_client(
        dummy_state_model_path,
        monkeypatch,
        shutdown_timeout=window,
        stream_admission_timeouts={"never_fed_timeout": 10 * window},
    )
    try:
        result = await asyncio.wait_for(
            client.admit_stream_async(_raw_admission(config)), TIMEOUT
        )
        assert result.accepted
        (process,) = client.resources.engine_manager.processes
        batches = []

        async def read_outputs():
            while True:
                batches.append(await client.get_output_async())

        reader = asyncio.create_task(read_outputs())
        process.terminate()
        await asyncio.to_thread(process.join, window)
        assert process.exitcode == 0
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(reader, TIMEOUT)
        assert [o for batch in batches for o in batch.outputs] == []
    finally:
        client.shutdown()


# Model-state hooks: execution memory, the model state's own CUDA graphs,
# post-warmup and shutdown.
#
# A scenario is one stub configuration. `_observe` starts it once, in a new
# process, with the real engine constructors, and returns plain data: what the
# stub's hooks recorded, what the worker and the runner show after startup,
# and how startup failed if it did. The tests below each assert one behaviour
# on that data, so tests of one scenario share its single start. An engine
# here is not eager unless the scenario says so; the persistent-state defaults
# then resolve `cudagraph_mode` to NONE, so the only graph is the one the stub
# captures when asked.

GiB = 1 << 30
CONTRIBUTORS = ["stub_capture_mib", "stub_execution_mib"]
# The whole note the runner adds for those contributors.
CONTRIBUTOR_NOTE = (
    "The model declares these options as using device memory: "
    "stub_capture_mib, stub_execution_mib."
)
CAPTURING = {"capture_graphs": True, "capture_mib": 64}
NAMED = {"memory_contributors": CONTRIBUTORS}
FIXED_BUDGET = dict(kv_cache_memory_bytes=8 * MiB)
START_TIMEOUT = 600

# (profile_only, bytes) for each `capture_model` call of a recording runner.
CAPTURE_RESULTS: list[tuple[bool, int]] = []


# The exceptions that left the KV cache allocation of a recording runner, as
# they were when they left it.
ALLOCATION_FAILURES: list[BaseException] = []

# Set when a recording runner enters `shutdown`.
SHUTDOWN_ENTERED = threading.Event()


@contextmanager
def _recording_allocation_failures(allocation_context):
    """The worker's allocation context, with a record of the exception that
    leaves the allocation inside it."""
    with allocation_context or nullcontext():
        try:
            yield
        except BaseException as error:
            ALLOCATION_FAILURES.append(error)
            raise


class RecordingRunner(GPUModelRunner):
    """The standard runner. It records what each `capture_model` returns,
    the exception of a failed KV cache allocation, and when `shutdown` is
    entered; every method does the runner's own work."""

    def capture_model(self, *, profile_only: bool = False) -> int:
        captured = super().capture_model(profile_only=profile_only)
        CAPTURE_RESULTS.append((profile_only, int(captured)))
        return captured

    def initialize_kv_cache(
        self,
        kv_cache_config,
        is_profiling: bool = False,
        kv_cache_allocation_context=None,
        extensible: bool = False,
    ) -> None:
        super().initialize_kv_cache(
            kv_cache_config,
            is_profiling,
            _recording_allocation_failures(kv_cache_allocation_context),
            extensible,
        )

    def shutdown(self) -> None:
        SHUTDOWN_ENTERED.set()
        super().shutdown()


@contextmanager
def _recording_worker_gc_freezes():
    """Inside the block, each heap freeze the worker module makes is recorded
    in the stub's `EVENTS` as ("worker_gc_freeze", {}) before it happens. The
    worker module's own binding of the name is the one replaced, so the
    freezes of other modules are not recorded."""
    from vllm_add_dummy_model.my_persistent_state import EVENTS

    freeze = gpu_worker.freeze_gc_heap

    def recorded_freeze() -> None:
        EVENTS.append(("worker_gc_freeze", {}))
        freeze()

    gpu_worker.freeze_gc_heap = recorded_freeze
    try:
        yield
    finally:
        gpu_worker.freeze_gc_heap = freeze


class RecordingWorker(Worker):
    """The standard worker with a recording runner. It records, in the
    stub's `EVENTS`, each heap freeze of its warmup and each time it reaches
    the JIT monitor's activation; the worker's own work is unchanged."""

    def _make_model_runner(self) -> GPUModelRunner:
        return RecordingRunner(self.vllm_config, self.device)

    def compile_or_warm_up_model(self):
        with _recording_worker_gc_freezes():
            return super().compile_or_warm_up_model()

    def _maybe_activate_jit_monitor(self) -> None:
        from vllm_add_dummy_model.my_persistent_state import EVENTS

        from vllm.utils import jit_monitor

        super()._maybe_activate_jit_monitor()
        EVENTS.append(("worker_jit_monitor", {"active": jit_monitor.is_active()}))


def _scenario(
    options=None, *, mode="engine", env=None, probes=(), encoder=False, **overrides
):
    return dict(
        options=options or {},
        mode=mode,
        env=env or {},
        probes=probes,
        # Whether the model is the stub with a vision encoder.
        encoder=encoder,
        overrides=overrides,
    )


# The stub with a vision encoder: one encoder CUDA graph, for four tokens,
# under `cudagraph_mode` NONE. With four slots and no page-count override the
# pool is planned at five pages, above the temporary pool that the estimate
# allocates by its own rule. The stub's processor processes nothing, so
# multimodal profiling is skipped.
ENCODER = dict(
    encoder=True,
    max_num_seqs=4,
    limit_mm_per_prompt={"image": 1},
    skip_mm_profiling=True,
    compilation_config={
        "cudagraph_mode": "NONE",
        "cudagraph_mm_encoder": True,
        "encoder_cudagraph_token_budgets": [4],
        "encoder_cudagraph_max_vision_items_per_batch": 1,
    },
)


SCENARIOS = {
    "plain": _scenario(),
    "plain-held-shutdown": _scenario(probes=("held_shutdown",), **FIXED_BUDGET),
    "execution": _scenario({"execution_mib": 4096}),
    "execution-fails": _scenario({"fail": ["prepare_execution_memory"]}),
    "profile-run-fails": _scenario({"fail": ["profile_run"]}),
    "capturing": _scenario(CAPTURING),
    "capturing-sync-check": _scenario(CAPTURING, env={"VLLM_GPU_SYNC_CHECK": "warn"}),
    # Compilation pinned to the mode in which the worker's warmup triggers
    # Inductor's lazy initialization; the stub itself compiles nothing.
    "capturing-inductor": _scenario(
        CAPTURING,
        compilation_config={
            "mode": int(CompilationMode.VLLM_COMPILE),
            "backend": "inductor",
            "cudagraph_mode": "NONE",
        },
    ),
    "capturing-eager": _scenario(CAPTURING, enforce_eager=True),
    "capturing-fixed": _scenario(CAPTURING, **FIXED_BUDGET),
    "capturing-estimate-off": _scenario(
        CAPTURING, env={"VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0"}
    ),
    "capturing-no-jit-warmup": _scenario(
        CAPTURING, kernel_config={"enable_jit_warmup": False}
    ),
    "capturing-five-slots": _scenario(CAPTURING, max_num_seqs=4),
    "capturing-override": _scenario(CAPTURING, num_gpu_blocks_override=7),
    "profiling-capture-fails": _scenario({**CAPTURING, "fail": ["capture_profile"]}),
    "serving-capture-fails": _scenario({**CAPTURING, "fail": ["capture_serving"]}),
    "noted-capture-fails": _scenario(
        {**CAPTURING, **NAMED, "fail": ["capture_serving"]}, **FIXED_BUDGET
    ),
    "noted-capture-oom": _scenario(
        {**CAPTURING, **NAMED, "fail": ["capture_serving"], "fail_oom": True},
        **FIXED_BUDGET,
    ),
    "release-fails": _scenario({**CAPTURING, "fail": ["clear_cudagraphs"]}),
    "capture-and-release-fail": _scenario(
        {**CAPTURING, **NAMED, "fail": ["capture_profile", "clear_cudagraphs"]}
    ),
    "post-warmup-fails": _scenario({"fail": ["post_warmup"]}),
    "shutdown-fails": _scenario({"fail": ["shutdown"]}),
    # The same failures raised as `StubInterrupt`, which is not an
    # `Exception`.
    "noted-capture-interrupt": _scenario(
        {
            **CAPTURING,
            **NAMED,
            "fail": ["capture_serving"],
            "interrupt": ["capture_serving"],
        },
        **FIXED_BUDGET,
    ),
    "capture-fails-release-interrupt": _scenario(
        {
            **CAPTURING,
            "fail": ["capture_profile", "clear_cudagraphs"],
            "interrupt": ["clear_cudagraphs"],
        }
    ),
    "shutdown-interrupt": _scenario({"fail": ["shutdown"], "interrupt": ["shutdown"]}),
    # A context of one token, in which no prefill request can be sized.
    "plain-one-token": _scenario(max_model_len=1),
    # State pools larger than the device; see `_oversized_pool`.
    "oversized-temporary-named": _scenario(
        {**CAPTURING, **NAMED}, oversized="temporary"
    ),
    "oversized-temporary": _scenario(CAPTURING, oversized="temporary"),
    "oversized-serving-named": _scenario(NAMED, oversized="serving"),
    "oversized-serving": _scenario(oversized="serving"),
    # The graph-memory estimate alone, on a worker that is kept afterwards.
    # Three 256 MiB pages, so a temporary pool that stayed would show.
    "estimate": _scenario({**CAPTURING, "page_mib": 256}, mode="estimate"),
    "estimate-release-fails": _scenario(
        {**CAPTURING, "page_mib": 256, "fail": ["clear_cudagraphs"]}, mode="estimate"
    ),
    "estimate-release-interrupt": _scenario(
        {
            **CAPTURING,
            "page_mib": 256,
            "fail": ["clear_cudagraphs"],
            "interrupt": ["clear_cudagraphs"],
        },
        mode="estimate",
    ),
    # Encoder graphs, without and with graphs of the model state: the two
    # differ only in the model state's capture predicate.
    "encoder": _scenario({"capture_graphs": False, "capture_mib": 64}, **ENCODER),
    "encoder-capturing": _scenario(CAPTURING, **ENCODER),
}


def _stub_module():
    """The stub's module, with the records of earlier engines of this process
    cleared."""
    import vllm_add_dummy_model.my_persistent_state as stub

    for record in (stub.EVENTS, stub.BINDINGS, stub.RAISED, stub.WATCHED_STREAMS):
        record.clear()
    CAPTURE_RESULTS.clear()
    ALLOCATION_FAILURES.clear()
    SHUTDOWN_ENTERED.clear()
    return stub


def _worker(core: EngineCore):
    return core.model_executor.driver_worker.worker


def _oversized_pool(overrides: dict) -> int | None:
    """Replace the `oversized` marker by a pool of 1 MiB pages that totals
    twice the device's memory, rounded up to whole GiB; return that total in
    GiB. The page count is an override, so only a pool of the planned size
    cannot be allocated: a pool of a few pages fits any device. For the
    serving pool the budget is fixed at the pool's size, so planning passes."""
    which = overrides.pop("oversized", None)
    if which is None:
        return None
    total_bytes = torch.cuda.mem_get_info()[1]
    gib = 2 * -(-total_bytes // GiB)
    overrides["num_gpu_blocks_override"] = gib * 1024
    if which == "serving":
        overrides["kv_cache_memory_bytes"] = gib * GiB
    return gib


def _hook_config(path, scenario: dict):
    fields = {f"stub_{name}": value for name, value in scenario["options"].items()}
    overrides = {
        "enforce_eager": False,
        "hf_overrides": fields,
        "worker_cls": f"{__name__}.{RecordingWorker.__qualname__}",
        **scenario["overrides"],
    }
    oversized_gib = _oversized_pool(overrides)
    kwargs = _engine_kwargs(path, **RESOLVED_DEFAULTS)
    args = EngineArgs(**{**kwargs, **overrides})
    return args.create_engine_config(), oversized_gib


def _error(stub, error: BaseException) -> dict:
    """A failure as plain data. `raised` is the position of the exception
    object among those the stub's hooks raised, if it is one of them;
    `left_allocation` is whether it is the very object that left a KV cache
    allocation, out of `allocation_failures` such objects."""
    raised = [i for i, e in enumerate(stub.RAISED) if e is error]
    context = None if error.__suppress_context__ else error.__context__
    return dict(
        type=type(error).__name__,
        message=str(error),
        notes=list(getattr(error, "__notes__", [])),
        raised=raised[0] if raised else None,
        chained=error.__cause__ is not None or context is not None,
        left_allocation=any(e is error for e in ALLOCATION_FAILURES),
        allocation_failures=len(ALLOCATION_FAILURES),
    )


def _logged_raised(stub, records) -> list[int]:
    """Which of the exceptions the stub raised were logged with a traceback."""
    return sorted(
        {
            index
            for record in records
            if record.exc_info and record.exc_info[2] is not None
            for index, error in enumerate(stub.RAISED)
            if record.exc_info[1] is error
        }
    )


def _device_state(stub) -> dict:
    return dict(
        allocator_settings=stub.allocator_settings(),
        global_graph_pool=current_platform.get_global_graph_pool(),
        on_main_stream=torch.cuda.current_stream() == torch.cuda.default_stream(),
    )


def _shutdown_behind_held_stream(stub, core: EngineCore) -> dict:
    """Shut the engine down while a stream other than the main one has
    outstanding work: a host function holds it until this function lets it
    go (bounded, so a failure cannot hang the device). The absence window
    starts only once the runner has entered its shutdown."""
    import ctypes

    release = threading.Event()

    def hold(_):
        release.wait(TIMEOUT)

    host_fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(hold)
    libcuda = ctypes.CDLL("libcuda.so.1")
    stream = torch.cuda.Stream()
    stub.WATCHED_STREAMS.append(stream)
    launched = libcuda.cuLaunchHostFunc(
        ctypes.c_void_p(stream.cuda_stream), host_fn, None
    )
    assert launched == 0
    assert not stream.query()

    failures: list[str] = []

    def shut_down() -> None:
        try:
            core.shutdown()
        except BaseException as error:
            failures.append(repr(error))

    shutdown = threading.Thread(target=shut_down)
    shutdown.start()
    try:
        runner_entered = SHUTDOWN_ENTERED.wait(TIMEOUT)
        # Bounded wait for something that must not happen: the runner is in
        # its shutdown, and one that waits for the device is still waiting
        # when the second has passed.
        shutdown.join(1)
        still_waiting = shutdown.is_alive()
        hook_ran_while_held = any(e[0] == "shutdown" for e in stub.EVENTS)
    finally:
        release.set()
        shutdown.join(TIMEOUT)
        # The host function has returned before its callback object goes.
        stream.synchronize()
    return dict(
        runner_entered=runner_entered,
        still_waiting=still_waiting,
        hook_ran_while_held=hook_ran_while_held,
        finished=not shutdown.is_alive(),
        failures=failures,
    )


def _observe_engine(stub, seen: dict, config, scenario: dict) -> None:
    from vllm.model_executor.warmup.attention_warmup import (
        mixed_batch_attention_warmup,
    )
    from vllm.utils import gpu_sync_debug, jit_monitor
    from vllm.v1.worker.gpu.warmup import (
        run_mixed_prefill_decode_warmup,
        warmup_kernels,
    )
    from vllm.v1.worker.workspace import current_workspace_manager

    try:
        core = EngineCore(
            vllm_config=config, executor_class=UniProcExecutor, log_stats=False
        )
    except (Exception, stub.StubInterrupt) as error:
        seen["error"] = _error(stub, error)
        seen["after_failure"] = _device_state(stub)
        return
    seen["startup_events"] = len(stub.EVENTS)
    worker = _worker(core)
    runner = worker.model_runner
    manager = current_workspace_manager()
    try:
        manager.get_persistent_resource(("stub", "growth"), list)
        growth_refused = False
    except AssertionError:
        growth_refused = True
    seen["after_startup"] = dict(
        _device_state(stub),
        jit_monitor_active=jit_monitor.is_active(),
        sync_check_enabled=gpu_sync_debug._sync_check_enabled,
        inductor_initialized=stub.inductor_initialized(),
        initial_seed=torch.initial_seed(),
        workspace_growth_refused=growth_refused,
        runner_needs_capture=runner.needs_cudagraph_capture(),
        serving_bindings_live=stub.BINDINGS[-1]["ref"]() is runner.state_slots,
        **{
            name: getattr(worker, name, None)
            for name in (
                "requested_memory",
                "available_kv_cache_memory_bytes",
                "total_consumed",
                "peak_activation_memory",
                "cudagraph_memory_estimate",
            )
        },
    )

    # The warmup entry points again, on the started runner. Guarded, so that
    # a failure here shows in the tests about warmup only.
    steps = seen["warmup_again"] = dict(error=None)
    try:
        warmup_kernels(runner, worker.execute_model, worker.sample_tokens)
        steps["after_warmup_kernels"] = len(_scheduled_steps(stub.EVENTS))
        steps["mixed_warmup_ran"] = run_mixed_prefill_decode_warmup(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            num_tokens=runner.max_num_tokens,
        )
        steps["after_mixed_warmup"] = len(_scheduled_steps(stub.EVENTS))
        steps["mixed_fallback_warmup_ran"] = run_mixed_prefill_decode_warmup(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            num_tokens=runner.max_num_tokens,
            prefill_only_fallback=True,
        )
        steps["after_mixed_fallback_warmup"] = len(_scheduled_steps(stub.EVENTS))
        # A registered warmup that is built on that function.
        mixed_batch_attention_warmup(worker)
        steps["after_attention_warmup"] = len(_scheduled_steps(stub.EVENTS))
    except Exception as error:
        steps["error"] = repr(error)

    seen["shutdown_error"] = None
    if "held_shutdown" in scenario["probes"]:
        seen["held_shutdown"] = _shutdown_behind_held_stream(stub, core)
    else:
        try:
            core.shutdown()
        except stub.StubInterrupt as error:
            seen["shutdown_error"] = _error(stub, error)
    seen["after_shutdown"] = dict(
        runner_bindings_dropped=runner.state_slots is None,
        model_state_dropped=not hasattr(runner, "model_state"),
    )
    runner.shutdown()


def _observe_estimate(stub, seen: dict, config, records: list) -> None:
    """Profile the KV cache budget, which runs the graph-memory estimate, on
    an executor that is kept, and look at the runner before any shutdown."""
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups

    executor = UniProcExecutor(config)
    try:
        # What the engine core does before it profiles memory.
        layer_configs = executor.get_kv_cache_layer_configs()
        layout = resolve_kv_cache_layout(
            config,
            executor.get_supported_kv_cache_layouts(),
            [lc for layers in layer_configs for lc in layers.values()],
        )
        groups = get_global_kv_cache_groups(config, layer_configs)
        config.adjust_dcp_kv_cache_interleave_size(groups)
        executor.set_kv_cache_layout(
            layout.name, config.parallel_config.cp_kv_cache_interleave_size
        )

        worker = executor.driver_worker.worker
        runner = worker.model_runner
        serving_pool = current_platform.get_global_graph_pool()
        torch.cuda.synchronize()
        reserved = torch.cuda.memory_reserved()
        try:
            executor.determine_available_memory()
        except (Exception, stub.StubInterrupt) as error:
            seen["error"] = _error(stub, error)
        # Nothing of the failure stays referenced here: neither the exception
        # and its frames nor a log record of it.
        seen["logged_raised"] = _logged_raised(stub, records)
        records.clear()
        stub.RAISED.clear()
        ALLOCATION_FAILURES.clear()
        gc.collect()
        torch.cuda.empty_cache()
        seen["after_estimate"] = dict(
            runner_bindings_dropped=runner.state_slots is None,
            model_state_bindings_cleared=runner.model_state.state_slots is None,
            kv_caches=len(getattr(runner, "kv_caches", [])),
            kv_cache_config_dropped=not hasattr(runner, "kv_cache_config"),
            graph_manager_dropped=runner.cudagraph_manager is None,
            serving_pool_restored=(
                current_platform.get_global_graph_pool() == serving_pool
            ),
            bindings_alive=[b["ref"]() is not None for b in stub.BINDINGS],
            reserved_growth=torch.cuda.memory_reserved() - reserved,
        )
    finally:
        executor.shutdown()


def _observe(path: str, name: str) -> dict:
    """Start scenario `name` in this process and return what was observed."""
    scenario = SCENARIOS[name]
    stub = _stub_module()
    config, oversized_gib = _hook_config(path, scenario)
    compilation = config.compilation_config
    seen: dict = dict(
        error=None,
        oversized_gib=oversized_gib,
        cudagraph_mode=compilation.cudagraph_mode.name,
        enable_jit_warmup=config.kernel_config.enable_jit_warmup,
        # Whether the worker's warmup triggers Inductor's lazy initialization.
        inductor_expected=(
            compilation.mode != CompilationMode.NONE
            and compilation.backend == "inductor"
        ),
        seed=config.model_config.seed,
        # The estimate's own rule for its temporary KV cache, which applies
        # when the model state captures no graph of its own: one block per
        # sequence a graph can hold, at least one, replaced by
        # `num_gpu_blocks_override` when that is set.
        estimate_pool_pages=(
            config.cache_config.num_gpu_blocks_override
            or min(
                config.scheduler_config.max_num_seqs,
                compilation.max_cudagraph_capture_size or 0,
            )
            or 1
        ),
    )
    with _vllm_log_records() as records:
        if scenario["mode"] == "estimate":
            _observe_estimate(stub, seen, config, records)
        else:
            _observe_engine(stub, seen, config, scenario)
            seen["logged_raised"] = _logged_raised(stub, records)
        seen["attention_warmup_skip_logged"] = any(
            "Skipping mixed prefill+decode attention warmup" in record.getMessage()
            for record in records
        )
    seen["events"] = list(stub.EVENTS)
    seen["bindings"] = [
        {key: value for key, value in binding.items() if key != "ref"}
        | {"alive": binding["ref"]() is not None}
        for binding in stub.BINDINGS
    ]
    seen["capture_results"] = list(CAPTURE_RESULTS)
    return seen


@pytest.fixture(scope="module")
def _observed_starts() -> dict:
    return {}


@pytest.fixture
def started(_observed_starts, dummy_state_model_path, dummy_encoder_state_model_path):
    """Returns what `_observe` saw of a scenario. Each scenario starts once
    per module, in a new process; a start that failed or timed out is not
    repeated, and every test of its scenario fails with its error."""

    def start(name: str) -> dict:
        if name not in _observed_starts:
            env = SCENARIOS[name]["env"]
            saved = {variable: os.environ.get(variable) for variable in env}
            os.environ.update(env)
            try:
                path = (
                    dummy_encoder_state_model_path
                    if SCENARIOS[name]["encoder"]
                    else dummy_state_model_path
                )
                _observed_starts[name] = call_in_spawned_process(
                    _observe, path, name, timeout=START_TIMEOUT
                )
            except RuntimeError as error:
                _observed_starts[name] = error
            finally:
                for variable, value in saved.items():
                    if value is None:
                        del os.environ[variable]
                    else:
                        os.environ[variable] = value
        seen = _observed_starts[name]
        if isinstance(seen, RuntimeError):
            raise RuntimeError(f"scenario {name!r} did not start: {seen}")
        return seen

    return start


def _calls(seen, hook: str) -> list[dict]:
    """What each call of `hook` recorded, in call order. `seen` is an
    observation or the stub's module."""
    events = seen["events"] if isinstance(seen, dict) else seen.EVENTS
    return [event[1] for event in events if event[0] == hook]


def _positions(seen, kind: str) -> list[int]:
    events = seen["events"] if isinstance(seen, dict) else seen.EVENTS
    return [index for index, event in enumerate(events) if event[0] == kind]


def _scheduled_steps(events) -> list:
    """The steps the model ran with `dummy_run=False`."""
    return [e for e in events if e[0] == "prepare_step" and not e[1]]


def _startup_events_only(seen: dict) -> dict:
    """The observation, with only the events recorded until startup ended."""
    return dict(seen, events=seen["events"][: seen["startup_events"]])


def _startup_events(seen: dict) -> list:
    return _startup_events_only(seen)["events"]


@contextmanager
def _vllm_log_records():
    """The records vLLM's loggers emit inside the block."""
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = logging.getLogger("vllm")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


# Execution memory.


@pytest.mark.parametrize("scenario", ["capturing", "capturing-fixed"])
def test_execution_memory_is_prepared_once_before_the_profiling_pass(started, scenario):
    """The model state's `prepare_execution_memory` runs once, before the
    profiling forward pass and before any state pool exists, whether the KV
    cache budget is profiled or fixed."""
    seen = started(scenario)
    (prepared,) = _calls(seen, "prepare_execution_memory")
    assert prepared["slots_bound"] is False
    (position,) = _positions(seen, "prepare_execution_memory")
    assert position < _positions(seen, "profile_run")[0]
    assert position < _positions(seen, "forward")[0]
    assert position < _positions(seen, "initialize_state_slots")[0]


def test_execution_memory_is_prepared_before_the_split_limit(started):
    """With a profiled budget the hook runs with the current vLLM
    configuration set and at the allocator settings of the process; the
    worker limits block splitting afterwards, for the profiling pass."""
    seen = started("execution")
    (prepared,) = _calls(seen, "prepare_execution_memory")
    assert prepared["config_set"] is True
    profiling = _calls(seen, "profile_run")[0]
    assert "max_split_size_mb" in profiling["allocator_settings"]
    assert "max_split_size_mb" not in prepared["allocator_settings"]


def test_execution_memory_is_counted_once_in_the_profiled_budget(started):
    """The 4 GiB that the hook takes from the device are inside the
    memory-profiling window: the consumption the worker measured holds them
    once, not twice and not at all, and the KV cache budget is what was
    requested less that consumption and the peak activation memory."""
    seen = started("execution")
    (prepared,) = _calls(seen, "prepare_execution_memory")
    taken = prepared["device_bytes_taken"]
    assert taken >= 4 * GiB
    after = seen["after_startup"]
    assert taken <= after["total_consumed"] < 2 * taken
    budget = (
        after["requested_memory"]
        - after["total_consumed"]
        - after["peak_activation_memory"]
    )
    assert abs(after["available_kv_cache_memory_bytes"] - budget) < 1


def test_execution_memory_failure_fails_startup(started):
    """The exception that `prepare_execution_memory` raised is the one that
    fails startup, before the model runs."""
    seen = started("execution-fails")
    assert seen["error"]["raised"] == 0
    assert "prepare_execution_memory" in seen["error"]["message"]
    assert _positions(seen, "forward") == []


@pytest.mark.parametrize(
    ("scenario", "state"),
    [("plain", "after_startup"), ("profile-run-fails", "after_failure")],
    ids=["ends", "raises"],
)
def test_allocator_settings_are_restored_after_the_profiling_pass(
    started, scenario, state
):
    """Once the profiling pass has ended or raised, the allocator settings
    are again those in effect when `prepare_execution_memory` ran."""
    seen = started(scenario)
    assert (seen["error"] is None) == (state == "after_startup")
    (prepared,) = _calls(seen, "prepare_execution_memory")
    assert seen[state]["allocator_settings"] == prepared["allocator_settings"]


# The model state's own graphs: whether the runner captures.


def test_capture_predicate_is_first_asked_before_any_state_pool(started):
    """Under NONE with the graph-memory estimate, the runner first asks the
    model state whether it captures before any state binding exists. The
    stub's predicate takes no argument, so the runner passes none."""
    seen = started("capturing")
    assert seen["cudagraph_mode"] == "NONE"
    first = _positions(seen, "needs_cudagraph_capture")[0]
    assert first < _positions(seen, "initialize_state_slots")[0]
    assert _calls(seen, "needs_cudagraph_capture")[0]["slots_bound"] is False


@pytest.mark.parametrize(
    ("scenario", "captures"), [("capturing", True), ("plain", False)]
)
def test_runner_needs_capture_when_the_model_state_does(started, scenario, captures):
    """Without eager mode, and with no encoder or decoder graph to capture,
    the runner has graphs to capture exactly when the model state says it
    captures, although `cudagraph_mode` is NONE."""
    seen = started(scenario)
    assert seen["cudagraph_mode"] == "NONE"
    assert seen["after_startup"]["runner_needs_capture"] is captures


def test_eager_mode_never_asks_the_capture_predicate(started):
    """With `enforce_eager` the runner has nothing to capture and does not
    ask the model state, at startup or when asked itself afterwards."""
    seen = started("capturing-eager")
    assert seen["after_startup"]["runner_needs_capture"] is False
    assert _calls(seen, "needs_cudagraph_capture") == []


# The capture calls.


def test_estimate_and_warmup_each_capture_exactly_once(started):
    """With no encoder or decoder graph, each `capture_model` call still
    invokes `capture_cudagraphs` once: one call from the graph-memory
    estimate, then one from the worker's warmup."""
    seen = started("capturing")
    captures = _calls(seen, "capture_cudagraphs")
    assert [capture["profile_only"] for capture in captures] == [True, False]
    assert [profile_only for profile_only, _ in seen["capture_results"]] == [
        True,
        False,
    ]


def test_estimate_captures_into_its_throwaway_pool(started):
    """The estimate's capture call has `profile_only=True` and runs while the
    platform's global graph pool is the estimate's throwaway pool, which is
    not the pool that serves afterwards."""
    seen = started("capturing")
    profiling = _calls(seen, "capture_cudagraphs")[0]
    assert profiling["profile_only"] is True
    assert profiling["graph_pool"] == profiling["global_graph_pool"]
    assert profiling["graph_pool"] != seen["after_startup"]["global_graph_pool"]


def test_serving_capture_gets_the_global_pool_and_a_capture_stream(started):
    """The serving call receives `profile_only=False`, the platform's global
    graph pool, and a stream that is current during the call and is not the
    main stream; the runner has left that stream when the next hook runs."""
    seen = started("capturing")
    serving = _calls(seen, "capture_cudagraphs")[-1]
    assert serving["profile_only"] is False
    assert serving["graph_pool"] == seen["after_startup"]["global_graph_pool"]
    assert serving["stream_is_current"] is True
    assert serving["stream_is_main"] is False
    (after,) = _calls(seen, "post_warmup")
    assert after["on_main_stream"] is True


@pytest.mark.parametrize("scenario", ["capturing", "capturing-fixed"])
def test_warmup_captures_once_over_the_serving_bindings(started, scenario):
    """Whether or not the estimate runs, the worker's warmup makes exactly one
    `profile_only=False` call, while the model state holds the serving
    bindings: after the registered JIT warmups, and before the final dummy
    run and `post_warmup`."""
    seen = started(scenario)
    captures = _calls(seen, "capture_cudagraphs")
    (serving,) = [c for c in captures if not c["profile_only"]]
    assert serving["bindings"] == len(seen["bindings"]) - 1
    position = _positions(seen, "capture_cudagraphs")[-1]
    assert position > _positions(seen, "initialize_state_slots")[-1]
    assert position > _positions(seen, "jit_warmup")[-1]
    assert position < _positions(_startup_events_only(seen), "prepare_step")[-1]
    assert position < _positions(seen, "post_warmup")[0]


def test_model_state_graphs_are_in_the_estimate_without_extrapolation(started):
    """The worker's graph-memory estimate is exactly what the estimate's
    `capture_model` call returned, and that figure holds the 64 MiB the
    profiling `capture_cudagraphs` call kept."""
    seen = started("capturing")
    profiling = _calls(seen, "capture_cudagraphs")[0]
    assert profiling["scratch_bytes"] == 64 * MiB
    assert profiling["device_bytes_taken"] >= profiling["scratch_bytes"]
    profile_only, captured = seen["capture_results"][0]
    assert profile_only is True
    assert captured >= profiling["scratch_bytes"]
    assert seen["after_startup"]["cudagraph_memory_estimate"] == captured


def test_graph_memory_estimate_is_subtracted_from_the_budget(started):
    """The KV cache budget is what was requested, less the measured
    consumption, the transient peak and the graph-memory estimate, each
    once: the worker's peak activation memory is the last two together."""
    after = started("capturing")["after_startup"]
    assert after["cudagraph_memory_estimate"] >= 64 * MiB
    assert after["peak_activation_memory"] >= after["cudagraph_memory_estimate"]
    budget = (
        after["requested_memory"]
        - after["total_consumed"]
        - after["peak_activation_memory"]
    )
    assert abs(after["available_kv_cache_memory_bytes"] - budget) < 1


@pytest.mark.parametrize("scenario", ["capturing", "serving-capture-fails"])
def test_stub_capture_reports_its_device_work_complete(started, scenario):
    """A check of the stub, not of the runner: the stub synchronizes the
    runner's stream and its own before a capture call returns or raises, and
    then finds both idle. It is how the stub keeps the capture contract that
    the runner relies on."""
    captures = _calls(started(scenario), "capture_cudagraphs")
    assert [capture["profile_only"] for capture in captures] == [True, False]
    assert [capture["work_completed"] for capture in captures] == [True, True]


@pytest.mark.parametrize(
    ("scenario", "call"),
    [
        ("profiling-capture-fails", "capture_profile"),
        ("serving-capture-fails", "capture_serving"),
    ],
)
def test_capture_failure_fails_startup(started, scenario, call):
    """The exception that `capture_cudagraphs` raised, in the estimate or in
    the warmup, is the one that fails startup."""
    error = started(scenario)["error"]
    assert error["raised"] == 0
    assert error["type"] == "StubHookError"
    assert call in error["message"]


@pytest.mark.parametrize(
    ("scenario", "captures"), [("capturing", True), ("plain", False)]
)
def test_workspace_is_locked_by_a_serving_capture_only(started, scenario, captures):
    """After the serving capture the workspace is locked, so `post_warmup`
    sees it locked and a new workspace resource is refused; the profiling
    capture leaves it unlocked. When nothing is captured it stays unlocked."""
    seen = started(scenario)
    assert [c["workspace_locked"] for c in _calls(seen, "clear_cudagraphs")] == (
        [False] if captures else []
    )
    (after,) = _calls(seen, "post_warmup")
    assert after["workspace_locked"] is captures
    assert seen["after_startup"]["workspace_growth_refused"] is captures


# The profiling capture: its pool, its release and its failures.


@pytest.mark.parametrize(
    ("scenario", "pages"),
    [("capturing-five-slots", 5), ("capturing-override", 7)],
    ids=["max_num_seqs", "override"],
)
def test_profiling_pool_has_the_planned_page_count(started, scenario, pages):
    """When the model state captures, the estimate's temporary state pool has
    the planned page count, `max_num_seqs + 1` or the override, so the
    profiling capture sees the pool shape of the serving capture."""
    seen = started(scenario)
    profiling, serving = _calls(seen, "capture_cudagraphs")
    assert profiling["profile_only"] is True
    assert profiling["pool_pages"] == pages
    assert serving["pool_pages"] == pages
    assert [binding["pages"] for binding in seen["bindings"]] == [pages, pages]


def _assert_pool_allocation_failed(seen: dict) -> None:
    """Startup failed in the allocation of the oversized pool itself: an
    out-of-memory error for the pool's whole size, and the exception that
    fails startup is the object that left that allocation, not a copy."""
    error = seen["error"]
    assert error["type"] == "OutOfMemoryError"
    assert f"Tried to allocate {seen['oversized_gib']}.00 GiB" in error["message"]
    assert error["allocation_failures"] == 1
    assert error["left_allocation"] is True
    assert error["chained"] is False
    assert _positions(seen, "initialize_state_slots") == []


def test_profiling_pool_allocation_failure_fails_startup_as_it_is(started):
    """When the device cannot hold the temporary pool of the planned size,
    the allocation's own exception fails startup: it is not the
    planned-capacity `ValueError`, it states no `max_num_seqs`, and neither
    the capture nor `clear_cudagraphs` runs."""
    seen = started("oversized-temporary")
    _assert_pool_allocation_failed(seen)
    assert "max_num_seqs" not in seen["error"]["message"]
    assert _calls(seen, "capture_cudagraphs") == []
    assert _calls(seen, "clear_cudagraphs") == []


@pytest.mark.parametrize(
    ("scenario", "started_up"),
    [("capturing", True), ("profiling-capture-fails", False)],
    ids=["returns", "raises"],
)
def test_estimate_teardown_clears_the_profiling_capture_once(
    started, scenario, started_up
):
    """The estimate's teardown calls `clear_cudagraphs` once, after the
    profiling capture returned or raised, while the model state still holds
    the profiling bindings and before the serving pool is bound."""
    seen = started(scenario)
    assert (seen["error"] is None) is started_up
    (cleared,) = _calls(seen, "clear_cudagraphs")
    assert cleared["held_graph"] is True
    assert cleared["bindings"] == 0
    (position,) = _positions(seen, "clear_cudagraphs")
    assert position > _positions(seen, "capture_cudagraphs")[0]
    later_bindings = _positions(seen, "initialize_state_slots")[1:]
    assert all(position < later for later in later_bindings)
    assert len(later_bindings) == (1 if started_up else 0)


def test_failed_release_fails_startup(started):
    """With no other exception propagating, the exception that
    `clear_cudagraphs` raised is the one that fails startup."""
    error = started("release-fails")["error"]
    assert error["raised"] == 0
    assert "clear_cudagraphs" in error["message"]


@pytest.mark.parametrize(
    "scenario", ["estimate", "estimate-release-fails", "estimate-release-interrupt"]
)
def test_estimate_releases_everything_it_built(started, scenario):
    """When the estimate has ended, with `clear_cudagraphs` returning or
    raising (an interrupt included), and before any shutdown: the runner
    holds no bindings, pool,
    KV cache configuration or graph manager, the model state's bindings are
    cleared, the profiling bindings are gone, the platform's global graph
    pool is the serving pool again, and the device has the 768 MiB temporary
    pool and the capture's memory back."""
    seen = started(scenario)
    failed = scenario != "estimate"
    assert (seen["error"] is not None) is failed
    assert not failed or seen["error"]["raised"] == 0
    assert len(_calls(seen, "capture_cudagraphs")) == 1
    assert len(_calls(seen, "clear_cudagraphs")) == 1
    assert seen["after_estimate"] == dict(
        seen["after_estimate"],
        runner_bindings_dropped=True,
        model_state_bindings_cleared=True,
        kv_caches=0,
        kv_cache_config_dropped=True,
        graph_manager_dropped=True,
        serving_pool_restored=True,
        bindings_alive=[False],
    )
    assert seen["after_estimate"]["reserved_growth"] < 256 * MiB


def test_release_interrupt_alone_propagates_from_the_estimate(started):
    """With no other exception propagating, a `clear_cudagraphs` failure that
    is not an `Exception` leaves the estimate as the object the hook raised,
    and is not logged in its place."""
    seen = started("estimate-release-interrupt")
    assert seen["error"]["type"] == "StubInterrupt"
    assert seen["error"]["raised"] == 0
    assert seen["logged_raised"] == []


def test_earlier_failure_outranks_a_release_interrupt(started):
    """When the profiling capture raised and `clear_cudagraphs` then raises
    something that is not an `Exception`, the capture's exception still fails
    startup, and the later one is logged with its traceback and not raised."""
    seen = started("capture-fails-release-interrupt")
    assert [e[0] for e in seen["events"] if e[0].endswith("cudagraphs")] == [
        "capture_cudagraphs",
        "clear_cudagraphs",
    ]
    assert seen["error"]["type"] == "StubHookError"
    assert seen["error"]["raised"] == 0
    assert "capture_profile" in seen["error"]["message"]
    assert 1 in seen["logged_raised"]
    assert 0 not in seen["logged_raised"]


def test_estimate_alone_runs_the_profiling_capture(started):
    """The retained worker's estimate is the one a start runs: one
    `profile_only=True` capture over a pool of the planned three pages."""
    seen = started("estimate")
    (profiling,) = _calls(seen, "capture_cudagraphs")
    assert profiling["profile_only"] is True
    assert profiling["pool_pages"] == 3


def test_earlier_failure_outranks_a_failed_release(started):
    """When the profiling capture raised and `clear_cudagraphs` then raises
    too, the capture's exception fails startup, with its contributor note,
    and the exception from `clear_cudagraphs` is logged with its traceback
    and not raised."""
    seen = started("capture-and-release-fail")
    assert len(seen["bindings"]) == 1
    assert [e[0] for e in seen["events"] if e[0].endswith("cudagraphs")] == [
        "capture_cudagraphs",
        "clear_cudagraphs",
    ]
    assert seen["error"]["raised"] == 0
    assert "capture_profile" in seen["error"]["message"]
    assert seen["error"]["notes"] == [CONTRIBUTOR_NOTE]
    assert 1 in seen["logged_raised"]
    assert 0 not in seen["logged_raised"]


@pytest.mark.parametrize("scenario", ["capturing-fixed", "capturing-estimate-off"])
def test_without_the_estimate_only_the_serving_capture_runs(started, scenario):
    """With a fixed budget, or with the graph-memory estimate switched off,
    there is no `profile_only=True` call and no `clear_cudagraphs` call; the
    serving capture still runs."""
    seen = started(scenario)
    captures = _calls(seen, "capture_cudagraphs")
    assert [capture["profile_only"] for capture in captures] == [False]
    assert _calls(seen, "clear_cudagraphs") == []
    assert len(seen["bindings"]) == 1


def test_eager_mode_runs_the_other_hooks_but_no_capture(started):
    """With `enforce_eager`, a model state that would capture gets neither
    `capture_cudagraphs` nor `clear_cudagraphs`; it still gets
    `prepare_execution_memory`, `post_warmup` and `shutdown`, once each."""
    seen = started("capturing-eager")
    assert _calls(seen, "capture_cudagraphs") == []
    assert _calls(seen, "clear_cudagraphs") == []
    for hook in ("prepare_execution_memory", "post_warmup", "shutdown"):
        assert len(_calls(seen, hook)) == 1, hook


def test_serving_bindings_replace_the_profiling_bindings(started):
    """When the runner has graphs to capture, `initialize_state_slots` gets
    bindings over the temporary pool and then, once, new bindings over the
    serving pool. Before that second call the runner has dropped the
    profiling bindings and cleared the model state's; the second bindings
    are the ones the runner serves with."""
    seen = started("capturing")
    profiling, serving = seen["bindings"]
    assert serving["previous_alive"] is False
    assert serving["slots_cleared"] is True
    assert profiling["alive"] is False
    assert seen["after_startup"]["serving_bindings_live"] is True


def test_without_graphs_the_serving_bindings_are_the_only_ones(started):
    """When the runner has nothing to capture, the estimate allocates no
    temporary pool and `initialize_state_slots` runs once."""
    seen = started("plain")
    assert len(seen["bindings"]) == 1
    assert seen["after_startup"]["serving_bindings_live"] is True


# Encoder graphs under a persistent-state layout: the runner has graphs to
# capture whatever the model state's predicate returns, so the estimate
# allocates a temporary pool either way. Its size follows the predicate.


def test_encoder_graphs_alone_keep_the_minimal_temporary_pool(started):
    """With encoder graphs and a model state that captures none of its own,
    the estimate's temporary state pool keeps the size the estimate gives
    any temporary KV cache, which here is not the planned five pages; the
    serving pool has the planned pages."""
    seen = started("encoder")
    assert seen["cudagraph_mode"] == "NONE"
    assert seen["estimate_pool_pages"] != 5
    temporary, serving = seen["bindings"]
    assert temporary["pages"] == seen["estimate_pool_pages"]
    assert serving["pages"] == 5


def test_encoder_graphs_with_model_state_graphs_get_the_planned_pool(started):
    """The same model with a model state that captures: the temporary state
    pool has the planned five pages, as the serving pool has, and not the
    size of the estimate's own rule."""
    seen = started("encoder-capturing")
    assert seen["cudagraph_mode"] == "NONE"
    assert seen["estimate_pool_pages"] != 5
    assert [binding["pages"] for binding in seen["bindings"]] == [5, 5]
    profiling = _calls(seen, "capture_cudagraphs")[0]
    assert profiling["profile_only"] is True
    assert profiling["pool_pages"] == 5


@pytest.mark.parametrize("scenario", ["encoder", "encoder-capturing"])
def test_encoder_graphs_are_captured_in_the_estimate_and_in_warmup(started, scenario):
    """The stub's encoder graph is captured once over the temporary pool, in
    the estimate, and once over the serving pool, in the warmup, whether or
    not the model state captures graphs of its own."""
    seen = _startup_events_only(started(scenario))
    estimate, warmup = _positions(seen, "encoder_capture")
    temporary, serving = _positions(seen, "initialize_state_slots")
    assert temporary < estimate < serving < warmup
    assert started(scenario)["after_startup"]["runner_needs_capture"] is True


def test_encoder_graphs_alone_call_no_model_state_capture_hook(started):
    """With the predicate returning `False`, the estimate and the warmup
    capture the encoder graph and call neither `capture_cudagraphs` nor
    `clear_cudagraphs`."""
    seen = started("encoder")
    assert len(_positions(seen, "encoder_capture")) == 2
    assert _calls(seen, "capture_cudagraphs") == []
    assert _calls(seen, "clear_cudagraphs") == []


def test_model_state_capture_follows_the_encoder_graphs(started):
    """Under a persistent-state layout too, each `capture_model` call
    captures the encoder graph before it calls `capture_cudagraphs`."""
    seen = _startup_events_only(started("encoder-capturing"))
    encoder = _positions(seen, "encoder_capture")
    model_state = _positions(seen, "capture_cudagraphs")
    assert len(encoder) == len(model_state) == 2
    assert encoder[0] < model_state[0] < encoder[1] < model_state[1]


# Startup pool access during the capture calls.


@pytest.mark.parametrize(
    ("scenario", "next_hook"),
    [("capturing", "post_warmup"), ("serving-capture-fails", "shutdown")],
    ids=["returns", "raises"],
)
def test_startup_access_is_open_during_each_capture_call_only(
    started, scenario, next_hook
):
    """Each capture call can take startup access to the pools of its
    bindings, and that access is refused once the call has returned or
    raised: at `clear_cudagraphs` after the profiling call, and at
    `post_warmup`, or at `shutdown` after a failure, for the serving call."""
    seen = started(scenario)
    captures = _calls(seen, "capture_cudagraphs")
    assert [capture["pool_pages"] for capture in captures] == [3, 3]
    (cleared,) = _calls(seen, "clear_cudagraphs")
    assert cleared["stale_access_refused"] is True
    assert _calls(seen, next_hook)[0]["stale_access_refused"] is True


# Memory contributors on allocation and capture failures. The stub's errors
# reach the test in the process that raised them, so the notes on them are
# exactly those the runner added.


@pytest.mark.parametrize(
    ("scenario", "error_type"),
    [
        ("noted-capture-fails", "StubHookError"),
        ("noted-capture-oom", "OutOfMemoryError"),
    ],
)
def test_capture_failure_notes_the_memory_contributors(started, scenario, error_type):
    """When `capture_cudagraphs` raises, whatever the exception's type, the
    same exception object fails startup with one note: the statement that
    the model declares the named options as using device memory, and nothing
    else."""
    error = started(scenario)["error"]
    assert error["type"] == error_type
    assert error["raised"] == 0
    assert error["notes"] == [CONTRIBUTOR_NOTE]


def test_capture_interrupt_notes_the_memory_contributors(started):
    """A `capture_cudagraphs` failure that is not an `Exception` gets the same
    one note, on the object that fails startup."""
    error = started("noted-capture-interrupt")["error"]
    assert error["type"] == "StubInterrupt"
    assert error["raised"] == 0
    assert error["notes"] == [CONTRIBUTOR_NOTE]


def test_capture_failure_without_contributors_gets_no_note(started):
    """When the spec declares no memory contributors, the runner adds no note
    to the exception from `capture_cudagraphs`."""
    error = started("serving-capture-fails")["error"]
    assert error["raised"] == 0
    assert error["notes"] == []


@pytest.mark.parametrize("pool", ["temporary", "serving"])
def test_state_pool_allocation_failure_notes_the_memory_contributors(started, pool):
    """When the device cannot hold a state pool, the estimate's temporary
    pool or the serving pool, the allocation's exception propagates as it
    was raised, with the one contributor note added."""
    seen = started(f"oversized-{pool}-named")
    _assert_pool_allocation_failed(seen)
    assert seen["error"]["notes"] == [CONTRIBUTOR_NOTE]


@pytest.mark.parametrize("pool", ["temporary", "serving"])
def test_state_pool_allocation_failure_without_contributors_gets_no_note(started, pool):
    """Without declared contributors the allocation's exception carries no
    note."""
    seen = started(f"oversized-{pool}")
    _assert_pool_allocation_failed(seen)
    assert seen["error"]["notes"] == []


# Warmup under a persistent-state layout.


@pytest.mark.parametrize("scenario", ["capturing", "plain"])
def test_warmup_executes_no_scheduled_step(started, scenario):
    """Under NONE without `enforce_eager`, JIT warmup is enabled, yet the
    model runs no step with `dummy_run=False` during startup, whatever the
    model state's capture predicate returns."""
    seen = started(scenario)
    assert seen["enable_jit_warmup"] is True
    assert _scheduled_steps(_startup_events(seen)) == []


@pytest.mark.parametrize("scenario", ["capturing", "plain", "plain-one-token"])
def test_warmup_entry_points_execute_no_step_on_a_started_runner(started, scenario):
    """Called with the started runner and the worker's own callbacks,
    `warmup_kernels` executes no step, and `run_mixed_prefill_decode_warmup`
    executes none and returns `False`, with and without its prefill-only
    fallback. That holds for a context of one token too, in which the
    warmup could size no prefill request."""
    seen = started(scenario)
    assert seen["error"] is None
    again = seen["warmup_again"]
    assert again["error"] is None
    assert again["after_warmup_kernels"] == 0
    assert again["mixed_warmup_ran"] is False
    assert again["after_mixed_warmup"] == 0
    assert again["mixed_fallback_warmup_ran"] is False
    assert again["after_mixed_fallback_warmup"] == 0


@pytest.mark.parametrize("scenario", ["capturing", "plain"])
def test_registered_attention_warmup_skips_under_a_state_layout(started, scenario):
    """`mixed_batch_attention_warmup`, a registered warmup built on
    `run_mixed_prefill_decode_warmup`, runs on the started worker and
    executes no step. Its skip message, which is logged once per process, is
    among the records of the start."""
    seen = started(scenario)
    again = seen["warmup_again"]
    assert again["error"] is None
    assert again["after_attention_warmup"] == 0
    assert seen["attention_warmup_skip_logged"] is True


@pytest.mark.parametrize(
    ("scenario", "jit_warmup"),
    [("capturing", True), ("capturing-no-jit-warmup", False)],
)
def test_registered_jit_warmups_run_under_a_state_layout(started, scenario, jit_warmup):
    """The JIT warmup that the stub registers runs once when JIT warmup is
    enabled, before the serving capture, and JIT monitoring is on afterwards
    under the same condition; with JIT warmup disabled neither happens."""
    seen = started(scenario)
    assert seen["enable_jit_warmup"] is jit_warmup
    assert len(_positions(seen, "jit_warmup")) == (1 if jit_warmup else 0)
    assert seen["after_startup"]["jit_monitor_active"] is jit_warmup


@create_new_process_for_each_test("spawn")
def test_capturing_model_starts_and_serves_without_eager_mode(dummy_state_model_path):
    """With NONE and without `enforce_eager`, a model state that captures its
    own graph starts through `LLM` and generates the stub's tokens: the
    scheduled steps run after a warmup that captured, ran the final dummy run
    and warmed the sampler up."""
    from vllm_add_dummy_model.my_persistent_state import expected_tokens

    stub = _stub_module()
    llm = LLM(
        **_engine_kwargs(
            dummy_state_model_path,
            enforce_eager=False,
            compilation_config={"cudagraph_mode": "NONE"},
            hf_overrides={"stub_capture_graphs": True},
        )
    )
    capture = _positions(stub, "capture_cudagraphs")[-1]
    assert capture < _positions(stub, "dummy_sampler_run")[-1]
    assert _scheduled_steps(stub.EVENTS) == []

    model = _model(llm.llm_engine.engine_core.engine_core)
    prompts = [[11 * i + 1, 11 * i + 2, 11 * i + 3] for i in range(4)]
    params = SamplingParams(temperature=0, max_tokens=8)
    outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    assert [list(output.outputs[0].token_ids) for output in outputs] == [
        expected_tokens(0, prompt[-1], model.vocab_size, model.eos_token_id)
        for prompt in prompts
    ]


# Post-warmup.


@pytest.mark.parametrize(
    "scenario", ["capturing-eager", "capturing", "capturing-no-jit-warmup", "plain"]
)
def test_warmup_calls_post_warmup_once(started, scenario):
    """A successful startup calls `post_warmup` exactly once, with or without
    eager mode, JIT warmup and graphs of the model state."""
    assert len(_calls(_startup_events_only(started(scenario)), "post_warmup")) == 1


def test_post_warmup_follows_every_stage_that_runs_the_model(started):
    """`post_warmup` runs after the registered JIT warmup, the serving
    capture, the final dummy run and the sampler warmup, in that order, and
    nothing that runs the model follows it at startup."""
    seen = _startup_events_only(started("capturing"))
    (position,) = _positions(seen, "post_warmup")
    stages = [
        _positions(seen, "jit_warmup")[-1],
        _positions(seen, "capture_cudagraphs")[-1],
        _positions(seen, "forward")[-1],
        _positions(seen, "dummy_sampler_run")[-1],
        position,
    ]
    assert stages == sorted(stages)
    later = {event[0] for event in seen["events"][position + 1 :]}
    assert not later & {
        "prepare_step",
        "forward",
        "dummy_sampler_run",
        "jit_warmup",
        "capture_cudagraphs",
    }


def test_post_warmup_precedes_the_seed_reset(started):
    """The stub's `post_warmup` seeds torch's generator with a value of its
    own; after startup the generator's seed is the configured one, so the
    worker reset the seed after the hook."""
    seen = started("capturing")
    assert len(_calls(seen, "post_warmup")) == 1
    assert seen["seed"] is not None
    assert seen["after_startup"]["initial_seed"] == seen["seed"]


def test_post_warmup_precedes_jit_monitoring_and_gc_freezing(started):
    """The worker's warmup activates the JIT monitor once and freezes the
    heap once, in that order, both after the sampler warmup and after
    `post_warmup`; JIT monitoring is off when the hook runs.

    The recording worker records its own activation step and its own heap
    freeze in the stub's event log, so the engine core's later freeze is not
    among them."""
    seen = _startup_events_only(started("capturing"))
    (hook,) = _positions(seen, "post_warmup")
    (monitor,) = _positions(seen, "worker_jit_monitor")
    (freeze,) = _positions(seen, "worker_gc_freeze")
    assert _positions(seen, "dummy_sampler_run")[-1] < hook < monitor < freeze
    (at_hook,) = _calls(seen, "post_warmup")
    (at_monitor,) = _calls(seen, "worker_jit_monitor")
    assert at_hook["jit_monitor_active"] is False
    assert at_monitor["active"] is True


def test_post_warmup_precedes_the_gpu_sync_check(started):
    """With `VLLM_GPU_SYNC_CHECK` set, the sync-check gate is off when
    `post_warmup` runs and on after startup."""
    seen = started("capturing-sync-check")
    (hook,) = _calls(seen, "post_warmup")
    assert hook["sync_check_enabled"] is False
    assert seen["after_startup"]["sync_check_enabled"] is True


def test_post_warmup_precedes_inductor_lazy_initialization(started):
    """With compilation pinned to vLLM's Inductor mode, the worker triggers
    Inductor's lazy initialization after `post_warmup`: the stub compiles
    nothing, so it has not run when the hook does, and it has run after
    startup."""
    seen = started("capturing-inductor")
    assert seen["inductor_expected"] is True
    (hook,) = _calls(seen, "post_warmup")
    after = seen["after_startup"]
    oracle = (hook["inductor_initialized"], after["inductor_initialized"])
    assert None not in oracle, (
        "unsupported test infrastructure: this torch does not expose "
        "torch._inductor.fx_passes.joint_graph.lazy_init.cache_info, the "
        "oracle of this test"
    )
    assert hook["inductor_initialized"] is False
    assert after["inductor_initialized"] is True


def test_post_warmup_failure_fails_startup(started):
    """The exception that `post_warmup` raised is the one that fails
    startup."""
    error = started("post-warmup-fails")["error"]
    assert error["raised"] == 0
    assert "post_warmup" in error["message"]


def test_post_warmup_sees_the_pools_where_the_capture_saw_them(started):
    """The layout snapshot the stub takes in `post_warmup`, over the serving
    bindings, equals the one it took in the serving capture: the pages did
    not move during warmup."""
    seen = started("capturing")
    (hook,) = _calls(seen, "post_warmup")
    assert hook["layout_unchanged"] is True
    assert hook["bindings"] == len(seen["bindings"]) - 1


# Shutdown.


def test_runner_shutdown_calls_the_hook_before_it_releases_the_state(started):
    """`GPUModelRunner.shutdown` calls the model state's `shutdown` while
    the state bindings and their pool still exist, and releases them
    afterwards."""
    seen = started("plain")
    assert _calls(_startup_events_only(seen), "shutdown") == []
    hook = _calls(seen, "shutdown")[0]
    assert hook["slots_bound"] is True
    assert hook["pool_pages"] == 3
    assert seen["after_shutdown"]["runner_bindings_dropped"] is True


def test_runner_shutdown_waits_for_the_device_before_the_hook(started):
    """While a stream other than the main one has outstanding work, a
    shutdown that the runner has entered has not reached the hook a second
    later; once that work is let go the shutdown finishes, and the hook found
    the stream idle."""
    seen = started("plain-held-shutdown")
    held = seen["held_shutdown"]
    assert held["failures"] == []
    assert held["runner_entered"] is True
    assert held["still_waiting"] is True
    assert held["hook_ran_while_held"] is False
    assert held["finished"] is True
    assert _calls(seen, "shutdown")[0]["watched_streams_idle"] is True


def test_second_runner_shutdown_does_not_call_the_hook_again(started):
    """Once a shutdown has released the model state, shutting the runner down
    again calls no hook."""
    seen = started("plain")
    assert seen["after_shutdown"]["model_state_dropped"] is True
    assert len(_calls(seen, "shutdown")) == 1


def test_failed_startup_shuts_the_model_state_down(started):
    """After a startup that fails once the model state exists, the runner's
    shutdown calls the hook on that model state, once, after everything else
    it did."""
    seen = started("post-warmup-fails")
    assert seen["error"] is not None
    assert _positions(seen, "shutdown") == [len(seen["events"]) - 1]


def test_shutdown_hook_failure_is_logged_and_cleanup_continues(started):
    """An exception from the model state's `shutdown` does not leave
    `GPUModelRunner.shutdown`: it is logged with its traceback, and the runner
    still releases the state bindings and the model state."""
    seen = started("shutdown-fails")
    assert seen["error"] is None
    assert len(_calls(seen, "shutdown")) == 1
    assert seen["logged_raised"] == [0]
    assert seen["after_shutdown"] == dict(
        runner_bindings_dropped=True, model_state_dropped=True
    )


def test_shutdown_hook_interrupt_propagates_after_the_cleanup(started):
    """A `shutdown` failure that is not an `Exception` leaves the engine's
    shutdown as the object the hook raised, without a note, once the runner
    has released the state bindings and the model state."""
    seen = started("shutdown-interrupt")
    assert seen["error"] is None
    assert len(_calls(seen, "shutdown")) == 1
    assert seen["shutdown_error"]["type"] == "StubInterrupt"
    assert seen["shutdown_error"]["raised"] == 0
    assert seen["shutdown_error"]["notes"] == []
    assert seen["after_shutdown"] == dict(
        runner_bindings_dropped=True, model_state_dropped=True
    )
