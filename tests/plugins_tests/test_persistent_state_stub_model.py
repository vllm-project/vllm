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
import re
import time

import pytest

from tests.utils import create_new_process_for_each_test
from tests.v1.core.utils import mentions_number
from vllm import LLM, SamplingParams
from vllm.config import CUDAGraphMode
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.platforms import current_platform
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.request import RequestStatus

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
