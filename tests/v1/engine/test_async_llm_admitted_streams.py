# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admitted streams in AsyncLLM and the client's retirement signal.

AsyncLLM and AsyncMPClient are built with their real constructors from the
config of the dummy-model plugin's persistent-state stub (facebook/opt-125m
config and tokenizer, no weights). Only the ZMQ sockets are replaced: a fake
engine on the other side decodes what the client sends and pushes
msgpack-encoded EngineCoreOutputs back, so the frontend runs end to end on the
CPU. See docs/design/persistent_state.md for the design.
"""

import asyncio
import threading

import numpy as np
import pytest
import pytest_asyncio
import zmq

import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.exceptions import QueueOverflowError, VLLMValidationError
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestError, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    StreamAdmissionContext,
    StreamAdmissionError,
    StreamAdmissionResult,
    StreamOutput,
    UtilityOutput,
)
from vllm.v1.engine import core_client as core_client_mod
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import ADMIT_STREAM_CALL, EngineCoreProc
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.metrics.loggers import AggregateStatLoggerBase
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult
from vllm.v1.token_buffer import TokenBuffer

pytestmark = pytest.mark.cpu_test

# Deadlock guard for every wait; nothing here sleeps for a fixed time.
TIMEOUT = 5.0
_ADDRESSES = {"input_address": "inproc://in", "output_address": "inproc://out"}


def _context(update_seq, input_closed=False):
    return StreamAdmissionContext(update_seq=update_seq, input_closed=input_closed)


async def _until(predicate):
    """Poll an explicit condition, yielding to the loop, with a deadline."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TIMEOUT
    while not predicate():
        if loop.time() > deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0)


# Tasks this module creates or can identify (test tasks and the tasks that
# iterate input generators); fixtures cancel and await only these.
_OWNED: list[asyncio.Task] = []


def _spawn(coro):
    task = asyncio.ensure_future(coro)
    _OWNED.append(task)
    return task


async def _finished(task):
    _, pending = await asyncio.wait({task}, timeout=TIMEOUT)
    assert not pending


async def _cancel_owned(*tasks):
    owned = [t for t in (*_OWNED, *tasks) if t is not None]
    _OWNED.clear()
    for task in owned:
        task.cancel()
    if owned:
        await asyncio.wait_for(asyncio.gather(*owned, return_exceptions=True), TIMEOUT)


def _as_engine_request(request: EngineCoreRequest) -> EngineCoreRequest:
    # The client sends the prompt as an int32 ndarray viewing the received
    # frame; copy it as the engine's Request does.
    if request.prompt_token_ids is not None:
        request.prompt_token_ids = TokenBuffer(request.prompt_token_ids)
    return request


class _FakeEngine:
    """The engine side of the client's sockets.

    Supported-task queries and barriers are answered as they are sent; every
    other message waits in `inbox` for the test. With `hold` set to a utility
    method name or "abort", the next such message is only delivered once
    `release()` completes its send, and never if the send is cancelled first.
    """

    def __init__(self):
        self.to_client: asyncio.Queue = asyncio.Queue()
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.encoder = MsgpackEncoder()
        self.hold: str | None = None
        self.held: list[asyncio.Future] = []
        # Refuse admissions with this reason as soon as they are delivered.
        self.auto_refuse: str | None = None
        # The frames of each admit_stream utility call, as the client sent them.
        self.admission_frames: list[list] = []
        self.barriers = 0
        # Closing a socket blocks until `may_close` is set.
        self.closing = threading.Event()
        self.may_close = threading.Event()
        self.may_close.set()

    def send_multipart(self, message, copy=False):
        _, kind, *frames = message
        kind = EngineCoreRequestType(bytes(kind))
        if kind == EngineCoreRequestType.ADD:
            payload = _as_engine_request(
                MsgpackDecoder(EngineCoreRequest).decode(frames)
            )
        else:
            payload = MsgpackDecoder().decode(frames)
            if kind == EngineCoreRequestType.UTILITY and payload[2] == "admit_stream":
                # Decoded again as EngineCoreProc does, with the request
                # typed; a request that cannot be decoded fails its own call.
                self.admission_frames.append(frames)
                payload, failure = EngineCoreProc._decode_admission_call(
                    MsgpackDecoder(ADMIT_STREAM_CALL), frames, payload
                )
                if failure is not None:
                    self.push(EngineCoreOutputs(utility_output=failure))
                    sent = asyncio.get_running_loop().create_future()
                    sent.set_result(None)
                    return sent
                payload = list(payload)
                payload[3] = (_as_engine_request(payload[3][0]),)
        sent = asyncio.get_running_loop().create_future()
        if self.hold is not None and (
            (kind == EngineCoreRequestType.ABORT and self.hold == "abort")
            or (kind == EngineCoreRequestType.UTILITY and payload[2] == self.hold)
        ):
            self.hold = None
            sent.add_done_callback(
                lambda f: None if f.cancelled() else self._deliver(kind, payload)
            )
            self.held.append(sent)
        else:
            sent.set_result(None)
            self._deliver(kind, payload)
        return sent

    async def recv_multipart(self, copy=False):
        return await self.to_client.get()

    def release(self):
        self.held = [sent for sent in self.held if not sent.done()]
        self.held.pop(0).set_result(None)

    def holding(self):
        return any(not sent.done() for sent in self.held)

    def _deliver(self, kind, payload):
        if kind == EngineCoreRequestType.UTILITY:
            _, call_id, method, _ = payload
            if method == "get_supported_tasks":
                return self.reply(call_id, ["generate"])
            if method == "barrier":
                return self.reply(call_id, None)
            if method == "admit_stream" and self.auto_refuse is not None:
                return self.refuse(call_id, payload[3][0], self.auto_refuse)
        elif kind == EngineCoreRequestType.ADD and (
            payload.external_req_id or ""
        ).startswith("barrier-"):
            return self.push(
                EngineCoreOutputs(
                    outputs=[
                        EngineCoreOutput(
                            payload.request_id, [1], finish_reason=FinishReason.STOP
                        )
                    ]
                )
            )
        self.inbox.put_nowait((kind, payload))

    # Messages from the client.
    async def recv(self):
        return await asyncio.wait_for(self.inbox.get(), TIMEOUT)

    def pending(self):
        messages = []
        while not self.inbox.empty():
            messages.append(self.inbox.get_nowait())
        return messages

    async def admission(self):
        kind, payload = await self.recv()
        assert kind == EngineCoreRequestType.UTILITY
        _, call_id, method, args = payload
        assert method == "admit_stream"
        return call_id, args[0]

    async def add(self):
        kind, request = await self.recv()
        assert kind == EngineCoreRequestType.ADD
        return request

    async def abort(self):
        kind, request_ids = await self.recv()
        assert kind == EngineCoreRequestType.ABORT
        return list(request_ids)

    # Messages to the client.
    def push(self, outputs: EngineCoreOutputs):
        frames = [zmq.Frame(bytes(buf)) for buf in self.encoder.encode(outputs)]
        self.to_client.put_nowait(frames)

    def reply(self, call_id, result):
        self.push(
            EngineCoreOutputs(
                utility_output=UtilityOutput(call_id, result=UtilityResult(result))
            )
        )

    def fail(self, call_id, message):
        """The utility call raised in the engine."""
        self.push(
            EngineCoreOutputs(
                utility_output=UtilityOutput(call_id, failure_message=message)
            )
        )

    def accept(self, call_id, request):
        self.reply(call_id, StreamAdmissionResult(True, request.request_id))

    def refuse(self, call_id, request, reason="capacity"):
        self.reply(call_id, StreamAdmissionResult(False, request.request_id, reason))

    def output(
        self,
        request_id,
        update_seq,
        tokens=(),
        finish=None,
        ends_stream=False,
        stop_reason=None,
    ):
        """An output on input `update_seq`; `ends_stream` as the scheduler
        marks the output that ends the session."""
        self.push(
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id,
                        list(tokens),
                        finish_reason=finish,
                        stop_reason=stop_reason,
                        ends_stream=ends_stream,
                        stream_output=StreamOutput(_context(update_seq)),
                    )
                ]
            )
        )

    def rejection(self, request_id, update_seq, reason, closed=False, error=None):
        # The engine's rejection keeps an ERROR finish; the frontend converts it.
        self.push(
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id,
                        [],
                        finish_reason=FinishReason.ERROR,
                        error=error,
                        stream_output=StreamOutput(
                            _context(update_seq, closed), reason
                        ),
                    )
                ]
            )
        )

    def retire(self, *request_ids):
        self.push(EngineCoreOutputs(retired_streams=list(request_ids)))

    def kill(self):
        self.to_client.put_nowait([zmq.Frame(EngineCoreProc.ENGINE_CORE_DEAD)])

    async def client_barrier(self, client):
        """Return once the client's socket reader has handled every frame
        pushed so far, and the tasks those frames woke have run: the second
        round trip lets the wake-ups queued during the first one go first."""
        for _ in range(2):
            await asyncio.wait_for(client.call_utility_async("barrier"), TIMEOUT)

    async def llm_barrier(self, llm):
        """Return once AsyncLLM's output handler has processed every batch
        pushed so far: an ordinary request finishes behind them."""
        self.barriers += 1

        async def run():
            async for _ in llm.generate(
                TokensPrompt(prompt_token_ids=[1]),
                SamplingParams(max_tokens=1),
                f"barrier-{self.barriers}",
            ):
                pass

        await asyncio.wait_for(run(), TIMEOUT)


class _SyncShadow:
    def poll(self, timeout):
        return 1

    def recv_multipart(self):
        return (b"\x00\x00", b"")


class _Socket:
    def __init__(self, engine):
        self.engine = engine

    def send_multipart(self, message, copy=False):
        return self.engine.send_multipart(message, copy)

    async def recv_multipart(self, copy=False):
        return await self.engine.recv_multipart(copy)

    def getsockopt_string(self, option):
        return "inproc://fake"

    def close(self, linger=0):
        self.engine.closing.set()
        assert self.engine.may_close.wait(TIMEOUT)


def _engine_config(model, **engine_args):
    return AsyncEngineArgs(
        model=model, enforce_eager=True, max_model_len=128, **engine_args
    ).create_engine_config()


@pytest.fixture
def state_engine_args(state_stub_models, dummy_state_model_path):
    """Engine arguments for the stub model, which declares persistent state."""
    return dict(
        model=dummy_state_model_path,
        load_format="dummy",
        enable_prefix_caching=False,
        async_scheduling=False,
        max_num_seqs=2,
    )


@pytest.fixture
def vllm_config(state_engine_args):
    return _engine_config(**state_engine_args)


@pytest.fixture
def engine(monkeypatch):
    engine = _FakeEngine()
    monkeypatch.setattr(
        core_client_mod, "make_zmq_socket", lambda *args, **kwargs: _Socket(engine)
    )
    monkeypatch.setattr(core_client_mod.zmq.Socket, "shadow", lambda *_: _SyncShadow())
    return engine


@pytest_asyncio.fixture
async def client(engine, vllm_config):
    _OWNED.clear()
    client = core_client_mod.AsyncMPClient(
        vllm_config, UniProcExecutor, False, client_addresses=_ADDRESSES
    )
    yield client
    reader = client.resources.output_queue_task
    client.shutdown()
    await _cancel_owned(reader)


@pytest_asyncio.fixture
async def make_llm(engine):
    """Builds AsyncLLMs on the fake engine and shuts them down afterwards."""
    _OWNED.clear()
    llms = []

    def make(vllm_config, **kwargs):
        llm = AsyncLLM(
            vllm_config,
            UniProcExecutor,
            **{"log_stats": False, "log_requests": False, **kwargs},
            client_addresses=_ADDRESSES,
        )
        llms.append(llm)
        return llm

    yield make
    # Shut down first, so the engine's own background work is handled by
    # shutdown(); then cancel and await only the tasks this module owns.
    tasks = []
    for llm in llms:
        tasks += [llm.output_handler, llm.engine_core.resources.output_queue_task]
        llm.shutdown()
    await _cancel_owned(*tasks)


@pytest_asyncio.fixture
async def llm(make_llm, vllm_config):
    return make_llm(vllm_config)


def _params(output_kind=RequestOutputKind.DELTA, **kwargs):
    return SamplingParams(max_tokens=4, output_kind=output_kind, **kwargs)


class _Inputs:
    """An input generator; after its items it optionally waits for `release()`
    and then either ends or raises."""

    def __init__(self, *prompts, then=None):
        self.prompts = prompts
        self.then = then
        self.gate = asyncio.Event()
        if then is None:
            self.gate.set()
        self.reads = 0
        self.task: asyncio.Task | None = None

    def release(self):
        self.gate.set()

    async def __call__(self):
        # The task that iterates the generator: the frontend's input task.
        self.task = asyncio.current_task()
        _OWNED.append(self.task)
        for prompt in self.prompts:
            self.reads += 1
            if not isinstance(prompt, StreamingInput):
                prompt = StreamingInput(prompt=TokensPrompt(prompt_token_ids=prompt))
            yield prompt
        await self.gate.wait()
        if self.then == "raise":
            raise RuntimeError("generator failed")


async def _admit(llm, engine, caller="caller", inputs=None):
    """Admit a stream through AsyncLLM.add_request; returns (queue, internal)."""
    inputs = inputs or _Inputs()
    task = _spawn(llm.add_request(caller, inputs(), _params(), stream_admission=True))
    call_id, request = await engine.admission()
    engine.accept(call_id, request)
    queue = await asyncio.wait_for(task, TIMEOUT)
    return queue, request.request_id


async def _start_generate(llm, engine, inputs, caller="caller"):
    """Start generate() and accept its admission; returns (stream, first, id)."""
    stream = llm.generate(inputs(), _params(), caller, stream_admission=True)
    first = _spawn(stream.__anext__())
    call_id, admission = await engine.admission()
    engine.accept(call_id, admission)
    return stream, first, admission.request_id


async def _admit_on_client(
    client, engine, request_id, external_req_id=None, prompt_token_ids=()
):
    request = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(prompt_token_ids),
        mm_features=None,
        sampling_params=_params(),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=True,
        external_req_id=external_req_id or request_id,
    )
    return _spawn(client.admit_stream_async(request))


async def _admitted_on_client(client, engine, request_id, external_req_id=None):
    task = await _admit_on_client(client, engine, request_id, external_req_id)
    call_id, sent = await engine.admission()
    engine.accept(call_id, sent)
    assert (await asyncio.wait_for(task, TIMEOUT)).accepted


async def _settled(llm, engine, task):
    """The result of `task` once the client has handled everything sent so
    far; a call that is still waiting, for example on an admission reply,
    fails here."""
    await engine.client_barrier(llm.engine_core)
    assert task.done(), "the call is still pending"
    return task.result()


class _RejectionLogger(AggregateStatLoggerBase):
    """A stat logger that keeps the admission rejections it is told of."""

    rejections: list[str] = []

    def __init__(self, vllm_config, engine_indexes):
        pass

    def record(self, *args, **kwargs):
        pass

    def log_engine_initialized(self):
        pass

    def record_admission_rejection(self, reason):
        self.rejections.append(reason)


@pytest.fixture
def rejections():
    _RejectionLogger.rejections = []
    return _RejectionLogger.rejections


def _messages_of(messages, kind):
    return [payload for message_kind, payload in messages if message_kind == kind]


# AsyncLLM: validation and admission.


@pytest.mark.asyncio
async def test_admitted_stream_admits_before_reading_generator(llm, engine):
    inputs = _Inputs([1, 2, 3])
    task = _spawn(llm.add_request("caller", inputs(), _params(), stream_admission=True))
    call_id, admission = await engine.admission()
    # An admission carries no prompt.
    assert admission.prompt_token_ids is None
    assert admission.resumable
    assert admission.stream_admission is None
    assert admission.external_req_id == "caller"
    assert llm.output_processor.has_request(admission.request_id)
    assert inputs.reads == 0
    assert not task.done()

    engine.accept(call_id, admission)
    await asyncio.wait_for(task, TIMEOUT)
    first = await engine.add()
    assert inputs.reads == 1
    assert first.request_id == admission.request_id
    assert first.prompt_token_ids == [1, 2, 3]
    assert first.resumable
    assert first.stream_admission == _context(1)


@pytest.mark.asyncio
async def test_admitted_stream_empty_generator_closes_with_one_end(llm, engine):
    stream, first, internal = await _start_generate(llm, engine, _Inputs())
    close = await engine.add()
    assert close.request_id == internal
    assert not close.resumable
    assert close.stream_admission == _context(1, True)
    assert not first.done()
    engine.output(internal, 1, (), FinishReason.STOP, ends_stream=True)
    final = await asyncio.wait_for(first, TIMEOUT)
    assert final.finished is True
    assert final.outputs[0].token_ids == []
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(stream.__anext__(), TIMEOUT)
    assert not llm.output_processor.has_request(internal)


@pytest.mark.asyncio
async def test_admitted_stream_duplicate_call_while_admission_pending(llm, engine):
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    call_id, admission = await engine.admission()
    with pytest.raises(StreamAdmissionError) as refused:
        await asyncio.wait_for(
            llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True),
            TIMEOUT,
        )
    assert refused.value.result.reason == "request_id_in_use"
    assert engine.pending() == []
    engine.accept(call_id, admission)
    await asyncio.wait_for(task, TIMEOUT)


@pytest.mark.asyncio
async def test_admitted_stream_caller_id_reserved_until_retired(llm, engine):
    _, internal = await _admit(llm, engine, inputs=_Inputs([1]))
    await engine.add()
    error_cls = StreamAdmissionError

    with pytest.raises(error_cls) as refused:
        await llm.add_request(
            "caller", _Inputs([1])(), _params(), stream_admission=True
        )
    assert refused.value.result.accepted is False
    assert refused.value.result.reason == "request_id_in_use"
    close = await engine.add()
    assert not close.resumable
    assert engine.pending() == []

    engine.output(internal, 2, finish=FinishReason.STOP, ends_stream=True)
    await engine.llm_barrier(llm)
    with pytest.raises(error_cls):
        await llm.add_request(
            "caller", _Inputs([1])(), _params(), stream_admission=True
        )

    waiter = _spawn(llm.wait_stream_retired("caller"))
    await engine.llm_barrier(llm)
    assert not waiter.done()
    engine.retire(internal)
    await asyncio.wait_for(waiter, TIMEOUT)

    _, reused = await _admit(llm, engine)
    assert reused != internal


@pytest.mark.asyncio
async def test_admitted_stream_reservation_release_keeps_caller_retirement(llm, engine):
    _, internal = await _admit(llm, engine, inputs=_Inputs([1]))
    await engine.add()
    await engine.add()
    engine.output(internal, 2, finish=FinishReason.STOP, ends_stream=True)
    engine.retire(internal)
    await engine.llm_barrier(llm)
    # Releasing the reservation must not consume the caller's retirement.
    await asyncio.wait_for(llm.wait_stream_retired("caller"), TIMEOUT)
    _, reused = await _admit(llm, engine)
    assert reused != internal


@pytest.mark.asyncio
async def test_admitted_stream_pending_wait_survives_overlapping_reuse(llm, engine):
    """A re-admission that becomes runnable as soon as stream 1's
    retirement is read runs before the wait it wakes, so it can rebind the
    caller ID first; the earlier wait still reports stream 1, and a wait started
    after the re-admission is issued refers to stream 2."""
    _, first = await _admit(llm, engine, inputs=_Inputs([1]))
    await engine.add()
    await engine.add()
    engine.output(first, 2, finish=FinishReason.STOP, ends_stream=True)
    await engine.llm_barrier(llm)
    earlier = _spawn(llm.wait_stream_retired("caller"))
    await engine.llm_barrier(llm)
    assert not earlier.done()

    engine.hold = "admit_stream"
    # No barrier between these: the socket reader is woken by the retirement,
    # then the re-admission is queued behind it, so it runs after the
    # retirement is read but before the woken wait resumes.
    engine.retire(first)
    readmission = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    await asyncio.wait_for(earlier, TIMEOUT)
    await _until(lambda: engine.held)
    assert not readmission.done()

    later = _spawn(llm.wait_stream_retired("caller"))
    await engine.llm_barrier(llm)
    assert not later.done()
    engine.release()
    call_id, admission = await engine.admission()
    second = admission.request_id
    assert second != first
    engine.accept(call_id, admission)
    await asyncio.wait_for(readmission, TIMEOUT)
    # A stray retirement of the earlier stream does not resolve the new wait.
    engine.retire(first)
    await engine.llm_barrier(llm)
    assert not later.done()
    engine.retire(second)
    await asyncio.wait_for(later, TIMEOUT)


@pytest.mark.asyncio
@pytest.mark.parametrize("disable_randomization", [False, True])
async def test_admitted_stream_internal_id_follows_randomization(
    llm, engine, monkeypatch, disable_randomization
):
    monkeypatch.setattr(
        envs, "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", disable_randomization
    )
    _, internal = await _admit(llm, engine)
    assert (internal == "caller") == disable_randomization
    assert internal.startswith("caller")


@pytest.mark.asyncio
async def test_admitted_stream_refusal_raises_and_frees_everything(llm, engine):
    inputs = _Inputs([1])
    task = _spawn(llm.add_request("caller", inputs(), _params(), stream_admission=True))
    call_id, admission = await engine.admission()
    engine.refuse(call_id, admission, "capacity")
    with pytest.raises(StreamAdmissionError) as refused:
        await asyncio.wait_for(task, TIMEOUT)
    result = refused.value.result
    assert isinstance(result, StreamAdmissionResult)
    assert (result.accepted, result.reason) == (False, "capacity")
    assert not llm.output_processor.has_request(admission.request_id)
    assert inputs.reads == 0
    with pytest.raises(ValueError):
        await asyncio.wait_for(llm.wait_stream_retired("caller"), TIMEOUT)
    # The caller ID is free again: the next call reaches the engine.
    _, internal = await _admit(llm, engine)
    assert internal != admission.request_id


@pytest.mark.asyncio
async def test_admitted_stream_cancelled_before_sent_frees_caller(llm, engine):
    engine.hold = "admit_stream"
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    await _until(lambda: engine.held)
    waiter = _spawn(llm.wait_stream_retired("caller"))
    await engine.llm_barrier(llm)
    assert not waiter.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ValueError):
        await asyncio.wait_for(waiter, TIMEOUT)
    await engine.llm_barrier(llm)
    assert engine.pending() == []
    _, internal = await _admit(llm, engine)
    assert internal.startswith("caller")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", ["awaiting_the_reply", "as_the_send_completes"])
async def test_admitted_stream_cancel_after_send_accepted_late_reply(
    llm, engine, cancelled
):
    """A call cancelled once its admission is sent, also when the send has
    completed but the call has not resumed, still has its reply collected: a
    late acceptance is aborted, and the caller ID stays reserved until that
    stream retires."""
    if cancelled == "as_the_send_completes":
        engine.hold = "admit_stream"
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    if cancelled == "as_the_send_completes":
        await _until(engine.holding)
        waiter = _spawn(llm.wait_stream_retired("caller"))
        await engine.client_barrier(llm.engine_core)
        engine.release()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        call_id, admission = await engine.admission()
    else:
        call_id, admission = await engine.admission()
        waiter = _spawn(llm.wait_stream_retired("caller"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    internal = admission.request_id
    engine.accept(call_id, admission)
    assert await engine.abort() == [internal]
    await engine.llm_barrier(llm)
    assert not waiter.done()

    with pytest.raises(StreamAdmissionError) as refused:
        await asyncio.wait_for(
            llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True),
            TIMEOUT,
        )
    assert refused.value.result.reason == "request_id_in_use"
    assert engine.pending() == []

    engine.output(internal, 0, (), FinishReason.ABORT, ends_stream=True)
    engine.retire(internal)
    await asyncio.wait_for(waiter, TIMEOUT)
    _, reused = await _admit(llm, engine)
    assert reused != internal


@pytest.mark.asyncio
async def test_admitted_stream_cancel_after_send_refused_late_reply(llm, engine):
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    call_id, admission = await engine.admission()
    waiter = _spawn(llm.wait_stream_retired("caller"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    engine.refuse(call_id, admission)
    with pytest.raises(StreamAdmissionError):
        await asyncio.wait_for(waiter, TIMEOUT)
    await engine.llm_barrier(llm)
    assert _messages_of(engine.pending(), EngineCoreRequestType.ABORT) == []
    assert not llm.output_processor.has_request(admission.request_id)
    _, internal = await _admit(llm, engine)
    assert internal != admission.request_id


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["accepted", "refused"])
async def test_admitted_stream_wait_during_call_setup_follows_new_admission(
    llm, engine, outcome
):
    """A wait started after the call began, before the admission is sent,
    follows the stream that call admits, or raises its refusal even when the
    refusal is read before the wait resumes."""
    engine.hold = "admit_stream"
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    await _until(engine.holding)
    waiter = _spawn(llm.wait_stream_retired("caller"))
    await engine.client_barrier(llm.engine_core)
    assert not waiter.done()
    if outcome == "refused":
        engine.auto_refuse = "capacity"
        engine.release()
        with pytest.raises(StreamAdmissionError):
            await asyncio.wait_for(task, TIMEOUT)
        with pytest.raises(StreamAdmissionError) as refused:
            await asyncio.wait_for(waiter, TIMEOUT)
        assert refused.value.result.reason == "capacity"
        return
    engine.release()
    call_id, admission = await engine.admission()
    internal = admission.request_id
    engine.accept(call_id, admission)
    await asyncio.wait_for(task, TIMEOUT)
    await engine.add()
    await engine.add()
    engine.output(internal, 2, (), FinishReason.STOP, ends_stream=True)
    await engine.llm_barrier(llm)
    assert not waiter.done()
    engine.retire(internal)
    await asyncio.wait_for(waiter, TIMEOUT)


# AsyncLLM: inputs, close, rejection and session end.


@pytest.mark.asyncio
async def test_admitted_stream_inputs_numbered_then_close(llm, engine):
    _, internal = await _admit(llm, engine, inputs=_Inputs([1], [2, 2]))
    first, second, close = [await engine.add() for _ in range(3)]
    assert [r.request_id for r in (first, second, close)] == [internal] * 3
    assert (first.prompt_token_ids, first.stream_admission) == ([1], _context(1))
    assert (second.prompt_token_ids, second.stream_admission) == (
        [2, 2],
        _context(2),
    )
    assert first.resumable and second.resumable
    assert not close.resumable
    assert close.stream_admission == _context(3, True)


@pytest.mark.asyncio
async def test_admitted_stream_invalid_item_aborts_unnumbered(llm, engine):
    invalid = StreamingInput(
        prompt=TokensPrompt(prompt_token_ids=[2]),
        sampling_params=SamplingParams(max_tokens=4, stop=["end"]),
    )
    stream, first, internal = await _start_generate(llm, engine, _Inputs([1], invalid))
    assert (await engine.add()).stream_admission == _context(1)
    with pytest.raises(VLLMValidationError):
        await asyncio.wait_for(first, TIMEOUT)
    await engine.llm_barrier(llm)
    messages = engine.pending()
    assert _messages_of(messages, EngineCoreRequestType.ADD) == []
    assert [
        list(ids) for ids in _messages_of(messages, EngineCoreRequestType.ABORT)
    ] == [[internal]]
    assert not llm.output_processor.has_request(internal)


@pytest.mark.asyncio
async def test_admitted_stream_input_task_cleans_up_for_add_request(llm, engine):
    inputs = _Inputs([1], then="raise")
    inputs.release()
    queue, internal = await _admit(llm, engine, inputs=inputs)
    assert (await engine.add()).stream_admission == _context(1)
    assert await engine.abort() == [internal]
    with pytest.raises(Exception, match="generator failed"):
        await asyncio.wait_for(queue.get(), TIMEOUT)
    await engine.llm_barrier(llm)
    assert engine.pending() == []
    assert not llm.output_processor.has_request(internal)


@pytest.mark.asyncio
async def test_admitted_stream_caller_stream_ends_on_input_closed_output(llm, engine):
    stream, first, internal = await _start_generate(llm, engine, _Inputs([1]))
    await engine.add()
    close = await engine.add()
    assert not close.resumable
    assert not first.done()

    # The EOS that ends input 1 is not the end of the stream.
    engine.output(internal, 1, [7], FinishReason.STOP)
    boundary = await asyncio.wait_for(first, TIMEOUT)
    assert isinstance(boundary, RequestOutput)
    assert boundary.finished is False
    assert boundary.outputs[0].finish_reason == "stop"

    engine.output(internal, 2, (), FinishReason.STOP, ends_stream=True)
    final = await asyncio.wait_for(stream.__anext__(), TIMEOUT)
    assert final.finished is True
    assert final.outputs[0].token_ids == []
    engine.output(internal, 2, [5])
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(stream.__anext__(), TIMEOUT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "error"),
    [
        ("empty_update", None),
        (
            "engine_paused",
            RequestError(code="engine_paused", message="paused", retryable=True),
        ),
    ],
)
async def test_admitted_stream_rejection_reaches_caller_and_stream_continues(
    llm, engine, reason, error
):
    """A rejection, with or without a RequestError, is one unfinished output
    attributed to its input; the stream goes on to its end, and nothing is
    aborted."""
    stream, first, internal = await _start_generate(llm, engine, _Inputs([1], [2]))
    for _ in range(3):
        await engine.add()
    engine.rejection(internal, 2, reason, error=error)
    rejected = await asyncio.wait_for(first, TIMEOUT)
    assert rejected.finished is False
    assert rejected.outputs[0].finish_reason is None
    assert rejected.stream_rejection.input_index == 1
    assert rejected.stream_rejection.reason == reason
    assert rejected.stream_rejection.error == error
    engine.output(internal, 1, [7], FinishReason.STOP, ends_stream=True)
    final = await asyncio.wait_for(stream.__anext__(), TIMEOUT)
    assert final.finished is True
    assert final.stream_rejection is None
    await engine.llm_barrier(llm)
    assert _messages_of(engine.pending(), EngineCoreRequestType.ABORT) == []


@pytest.mark.asyncio
async def test_admitted_stream_rejected_close_aborts_without_rejection(llm, engine):
    queue, internal = await _admit(llm, engine, inputs=_Inputs([1]))
    await engine.add()
    close = await engine.add()
    assert not close.resumable
    engine.rejection(internal, 2, "engine_paused", closed=True)
    assert await engine.abort() == [internal]
    # AsyncLLM.abort ends the stream locally; the rejection never surfaces.
    end = await asyncio.wait_for(queue.get(), TIMEOUT)
    assert end.finished is True
    assert getattr(end, "stream_rejection", None) is None
    engine.output(internal, 1, (), FinishReason.ABORT, ends_stream=True)
    await engine.llm_barrier(llm)
    assert queue.get_nowait() is None
    assert not llm.output_processor.has_request(internal)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish,stop_reason",
    [
        (FinishReason.ABORT, None),  # client abort or deadline expiry
        (FinishReason.ERROR, None),  # expiry, preemption or a failed state step
        (FinishReason.LENGTH, None),
        (FinishReason.STOP, 6),  # a stop token other than EOS
    ],
    ids=["abort", "error", "length", "stop_token"],
)
async def test_admitted_stream_engine_terminal_output_is_the_one_end(
    llm, engine, finish, stop_reason
):
    inputs = _Inputs([1], then="end")
    stream, first, internal = await _start_generate(llm, engine, inputs)
    await engine.add()
    engine.output(internal, 1, [6], finish, ends_stream=True, stop_reason=stop_reason)
    final = await asyncio.wait_for(first, TIMEOUT)
    assert final.finished is True
    assert final.outputs[0].finish_reason == str(finish)
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(stream.__anext__(), TIMEOUT)
    assert not llm.output_processor.has_request(internal)


@pytest.mark.asyncio
async def test_admitted_stream_cancelled_generate_aborts_once(llm, engine):
    inputs = _Inputs([1], then="end")
    stream, first, internal = await _start_generate(llm, engine, inputs)
    await engine.add()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await engine.abort() == [internal]
    engine.output(internal, 1, (), FinishReason.ABORT, ends_stream=True)
    await engine.llm_barrier(llm)
    assert _messages_of(engine.pending(), EngineCoreRequestType.ABORT) == []
    assert not llm.output_processor.has_request(internal)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("output", "generator_error"),
        ("generator_error", "output"),
        ("output", "engine_death"),
        ("engine_death", "generator_error"),
    ],
)
async def test_admitted_stream_first_end_to_reach_the_collector_wins(
    llm, engine, first, second
):
    """When two ends reach the collector before it is read, the caller sees
    the first and nothing after it."""
    inputs = _Inputs([1], then="raise")
    queue, internal = await _admit(llm, engine, inputs=inputs)
    await engine.add()
    for end in (first, second):
        if end == "output":
            engine.output(internal, 1, (), FinishReason.ABORT, ends_stream=True)
            await engine.llm_barrier(llm)
        elif end == "generator_error":
            inputs.release()
            await _finished(inputs.task)
        else:
            engine.kill()
            await _until(lambda: not llm.is_running)
    if first == "output":
        assert queue.get_nowait().finished is True
    elif first == "generator_error":
        with pytest.raises(Exception, match="generator failed"):
            queue.get_nowait()
    else:
        with pytest.raises(EngineDeadError):
            queue.get_nowait()
    assert queue.get_nowait() is None


@pytest.mark.asyncio
async def test_admitted_stream_engine_death_raises_once(llm, engine):
    stream, first, internal = await _start_generate(
        llm, engine, _Inputs([1], then="end")
    )
    await engine.add()
    waiter = _spawn(llm.wait_stream_retired("caller"))
    engine.kill()
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(first, TIMEOUT)
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(waiter, TIMEOUT)
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(stream.__anext__(), TIMEOUT)


@pytest.mark.asyncio
async def test_admitted_stream_generator_error_abort_survives_generate_exit(
    llm, engine
):
    """The input task's abort completes although generate() ends on the
    published error and closes the collector."""
    inputs = _Inputs([1], then="raise")
    stream, first, internal = await _start_generate(llm, engine, inputs)
    await engine.add()
    engine.hold = "abort"
    inputs.release()
    with pytest.raises(RuntimeError, match="generator failed"):
        await asyncio.wait_for(first, TIMEOUT)
    await _until(engine.holding)
    engine.release()
    assert await engine.abort() == [internal]


@pytest.mark.asyncio
async def test_unadmitted_stream_refused_input_ends_its_session(llm, engine):
    """A stream without stream_admission sends no context, and the engine
    refuses its input with the ordinary error output (no stream_output, the
    ID finished): the caller's stream ends, the session is aborted, and no
    retirement can be waited for."""
    stream = llm.generate(_Inputs([1], then="end")(), _params(), "caller")
    first = _spawn(stream.__anext__())
    request = await engine.add()
    assert request.stream_admission is None
    engine.push(
        EngineCoreOutputs(
            finished_requests={request.request_id},
            outputs=[
                EngineCoreOutput(
                    request.request_id, [], finish_reason=FinishReason.ERROR
                )
            ],
        )
    )
    output = await asyncio.wait_for(first, TIMEOUT)
    assert output.finished is True
    assert output.stream_rejection is None
    await engine.llm_barrier(llm)
    aborts = _messages_of(engine.pending(), EngineCoreRequestType.ABORT)
    assert [list(ids) for ids in aborts] == [[request.request_id]]
    with pytest.raises(ValueError):
        await asyncio.wait_for(llm.wait_stream_retired("caller"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_wait_by_caller_id_unknown_raises(llm, engine):
    with pytest.raises(ValueError):
        await asyncio.wait_for(llm.wait_stream_retired("nobody"), TIMEOUT)


# EngineCoreClient: retirement tracking.


@pytest.mark.asyncio
async def test_admit_stream_async_routes_stream_to_calling_client(engine, vllm_config):
    """An admitted stream's outputs return to the client that admitted it."""
    _OWNED.clear()
    client = core_client_mod.AsyncMPClient(
        vllm_config,
        UniProcExecutor,
        False,
        client_addresses=_ADDRESSES,
        client_count=4,
        client_index=3,
    )
    try:
        task = await _admit_on_client(client, engine, "stream")
        kind, (client_index, call_id, method, args) = await engine.recv()
        assert kind == EngineCoreRequestType.UTILITY
        assert (client_index, method) == (3, "admit_stream")
        (request,) = args
        assert (request.request_id, request.client_index) == ("stream", 3)
        engine.accept(call_id, request)
        result = await asyncio.wait_for(task, TIMEOUT)
        assert result == StreamAdmissionResult(True, "stream")
    finally:
        reader = client.resources.output_queue_task
        client.shutdown()
        await _cancel_owned(reader)


@pytest.mark.asyncio
async def test_retirement_retirement_only_batch_is_read(client, engine):
    await _admitted_on_client(client, engine, "s-1")
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)
    assert not waiter.done()
    engine.retire("other")
    await engine.client_barrier(client)
    assert not waiter.done()
    engine.retire("s-1")
    await asyncio.wait_for(waiter, TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_wait_during_admission_follows_reply(client, engine):
    task = await _admit_on_client(client, engine, "s-1")
    call_id, sent = await engine.admission()
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)
    assert not waiter.done()
    engine.accept(call_id, sent)
    await asyncio.wait_for(task, TIMEOUT)
    await engine.client_barrier(client)
    assert not waiter.done()
    engine.retire("s-1")
    await asyncio.wait_for(waiter, TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_retirement_before_admission_reply_is_kept(client, engine):
    task = await _admit_on_client(client, engine, "s-1")
    call_id, sent = await engine.admission()
    engine.retire("s-1")
    engine.accept(call_id, sent)
    assert (await asyncio.wait_for(task, TIMEOUT)).accepted
    await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_early_retirement_waits_for_admission_reply(client, engine):
    """A wait pending while the admission reply is outstanding keeps
    waiting until the reply arrives, even if the retirement arrives first."""
    task = await _admit_on_client(client, engine, "s-1")
    call_id, sent = await engine.admission()
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)
    engine.retire("s-1")
    await engine.client_barrier(client)
    assert not waiter.done()
    engine.accept(call_id, sent)
    assert (await asyncio.wait_for(task, TIMEOUT)).accepted
    await asyncio.wait_for(waiter, TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_woken_wait_cancelled_before_return_keeps_retirement(
    client, engine
):
    """A retirement is consumed only by a wait that returns it."""
    await _admitted_on_client(client, engine, "s-1")
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)
    engine.retire("s-1")
    # The socket reader wakes the wait; cancel it before it can resume.
    asyncio.get_running_loop().call_soon(waiter.cancel)
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_refusal_fails_pending_waits(client, engine):
    task = await _admit_on_client(client, engine, "s-1")
    call_id, sent = await engine.admission()
    waiters = [_spawn(client.wait_stream_retired_async("s-1")) for _ in range(2)]
    await engine.client_barrier(client)
    engine.refuse(call_id, sent, "active_limit")
    assert not (await asyncio.wait_for(task, TIMEOUT)).accepted
    for waiter in waiters:
        with pytest.raises(StreamAdmissionError) as refused:
            await asyncio.wait_for(waiter, TIMEOUT)
        assert refused.value.result.reason == "active_limit"
    with pytest.raises(ValueError):
        await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_cancel_before_send_fails_pending_waits(client, engine):
    """The admission is never delivered to the engine."""
    engine.hold = "admit_stream"
    task = await _admit_on_client(client, engine, "s-1")
    await _until(lambda: engine.held)
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)
    assert not waiter.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ValueError):
        await asyncio.wait_for(waiter, TIMEOUT)
    with pytest.raises(ValueError):
        await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)
    await engine.client_barrier(client)
    assert engine.pending() == []


@pytest.mark.asyncio
async def test_retirement_early_retirement_kept_until_first_wait(client, engine):
    await _admitted_on_client(client, engine, "s-1")
    engine.retire("s-1")
    await engine.client_barrier(client)
    await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)
    with pytest.raises(ValueError):
        await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_unwaited_cap_drops_oldest(client, engine):
    """Eviction follows retirement age, not admission age."""
    ids = [f"s-{i}" for i in range(1025)]
    for request_id in ids:
        await _admitted_on_client(client, engine, request_id)
    # Retire newest-admitted first, so the oldest retirement is ids[-1].
    for request_id in reversed(ids):
        engine.retire(request_id)
    await engine.client_barrier(client)
    with pytest.raises(ValueError):
        await asyncio.wait_for(client.wait_stream_retired_async(ids[-1]), TIMEOUT)
    await asyncio.wait_for(client.wait_stream_retired_async(ids[0]), TIMEOUT)
    await asyncio.wait_for(client.wait_stream_retired_async(ids[-2]), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_new_admission_drops_same_caller_retirement(client, engine):
    await _admitted_on_client(client, engine, "a-1", external_req_id="a")
    await _admitted_on_client(client, engine, "b-1", external_req_id="b")
    engine.retire("a-1", "b-1")
    await engine.client_barrier(client)
    await _admitted_on_client(client, engine, "a-2", external_req_id="a")
    with pytest.raises(ValueError):
        await asyncio.wait_for(client.wait_stream_retired_async("a-1"), TIMEOUT)
    await asyncio.wait_for(client.wait_stream_retired_async("b-1"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_unknown_stream_raises_at_once(client, engine):
    with pytest.raises(ValueError):
        await asyncio.wait_for(client.wait_stream_retired_async("never"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_concurrent_waits_and_cancellation(client, engine):
    await _admitted_on_client(client, engine, "s-1")
    waiters = [_spawn(client.wait_stream_retired_async("s-1")) for _ in range(3)]
    await engine.client_barrier(client)
    waiters[0].cancel()
    engine.retire("s-1")
    await asyncio.wait_for(asyncio.gather(*waiters[1:]), TIMEOUT)
    assert waiters[0].cancelled()


@pytest.mark.asyncio
async def test_retirement_cancelled_wait_keeps_tracking(client, engine):
    await _admitted_on_client(client, engine, "s-1")
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    engine.retire("s-1")
    await engine.client_barrier(client)
    await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)


# AsyncLLM: call validation, queue limits and admission failures.


def _invalid_calls():
    """Calls that break a condition of the admission their caller controls:
    add_request() keyword arguments and the parameters."""

    def sampling(**kwargs):
        return SamplingParams(max_tokens=4, **kwargs)

    structured = StructuredOutputsParams(regex="[0-9]+")
    return {
        "data_parallel_rank": ({"data_parallel_rank": 1}, sampling()),
        "pooling": ({}, PoolingParams(task="embed")),
        "n": ({}, sampling(n=2)),
        "stop": ({}, sampling(stop=["end"])),
        "final_only": ({}, sampling(output_kind=RequestOutputKind.FINAL_ONLY)),
        "structured": ({}, sampling(structured_outputs=structured)),
        "kv_transfer": ({}, sampling(extra_args={"kv_transfer_params": {"a": 1}})),
        "ec_transfer": ({}, sampling(extra_args={"ec_transfer_params": {"a": 1}})),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", list(_invalid_calls()))
async def test_admitted_stream_call_conditions_raise_before_anything_is_reserved(
    llm, engine, invalid
):
    """A call that breaks a condition its caller controls raises
    VLLMValidationError before the caller's request ID is reserved or
    anything is sent."""
    call, params = _invalid_calls()[invalid]
    inputs = _Inputs([1])
    task = _spawn(
        llm.add_request("caller", inputs(), params, stream_admission=True, **call)
    )
    with pytest.raises(VLLMValidationError):
        await _settled(llm, engine, task)
    assert engine.pending() == []
    assert inputs.reads == 0
    with pytest.raises(ValueError):
        await asyncio.wait_for(llm.wait_stream_retired("caller"), TIMEOUT)


@pytest.mark.asyncio
async def test_admitted_stream_overlong_id_is_never_visible_as_a_reservation(
    llm, engine
):
    """An ID too long for an internal request ID is refused before the call
    awaits anything: with setup held, the ID is not reserved, so another
    call under it is not refused as request_id_in_use and no wait attaches."""
    caller = "x" * 300
    engine.hold = "get_supported_tasks"
    calls = [
        _spawn(
            llm.add_request(caller, _Inputs([1])(), _params(), stream_admission=True)
        )
        for _ in range(2)
    ]
    for call in calls:
        with pytest.raises(VLLMValidationError):
            await asyncio.wait_for(call, TIMEOUT)
    assert not engine.holding()
    with pytest.raises(ValueError):
        await asyncio.wait_for(llm.wait_stream_retired(caller), TIMEOUT)
    assert engine.pending() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("disable_randomization", "longest"), [(False, 247), (True, 256)]
)
async def test_admitted_stream_internal_id_length_boundary(
    llm, engine, monkeypatch, disable_randomization, longest
):
    """The longest caller ID is the one whose internal ID has 256
    characters: 247 with vLLM's random suffix and 256 without it; one more
    character is refused before anything is sent."""
    monkeypatch.setattr(
        envs, "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", disable_randomization
    )
    with pytest.raises(VLLMValidationError):
        await asyncio.wait_for(
            llm.add_request(
                "x" * (longest + 1), _Inputs()(), _params(), stream_admission=True
            ),
            TIMEOUT,
        )
    assert engine.pending() == []
    _, internal = await _admit(llm, engine, "x" * longest)
    assert len(internal) == 256


@pytest.mark.asyncio
async def test_admitted_stream_engine_death_during_setup_raises(llm, engine):
    """A call that loses the engine while it asks for the supported tasks
    raises EngineDeadError; it has reserved nothing, so no wait hangs."""
    engine.hold = "get_supported_tasks"
    inputs = _Inputs([1])
    task = _spawn(llm.add_request("caller", inputs(), _params(), stream_admission=True))
    await _until(engine.holding)
    with pytest.raises(ValueError):
        await asyncio.wait_for(llm.wait_stream_retired("caller"), TIMEOUT)
    engine.kill()
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(task, TIMEOUT)
    assert inputs.reads == 0


@pytest.mark.asyncio
async def test_admitted_stream_under_a_live_unadmitted_request_id_is_refused(
    llm, engine, monkeypatch
):
    """With request ID randomization disabled the internal ID is the
    caller's: a call whose ID a live request that was not admitted holds is
    refused as request_id_in_use without reaching the engine, and that
    request's output state is untouched."""
    monkeypatch.setattr(envs, "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", True)
    await asyncio.wait_for(
        llm.add_request("caller", TokensPrompt(prompt_token_ids=[1, 2]), _params()),
        TIMEOUT,
    )
    assert (await engine.add()).request_id == "caller"
    with pytest.raises(StreamAdmissionError) as refused:
        await asyncio.wait_for(
            llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True),
            TIMEOUT,
        )
    assert refused.value.result.reason == "request_id_in_use"
    assert engine.pending() == []
    assert llm.output_processor.has_request("caller")
    assert llm.output_processor.get_num_unfinished_requests() == 1
    with pytest.raises(ValueError):
        await asyncio.wait_for(llm.wait_stream_retired("caller"), TIMEOUT)


@pytest.mark.asyncio
async def test_admitted_stream_requires_an_async_generator_prompt(llm, engine):
    with pytest.raises(VLLMValidationError):
        await asyncio.wait_for(
            llm.add_request(
                "caller",
                TokensPrompt(prompt_token_ids=[1]),
                _params(),
                stream_admission=True,
            ),
            TIMEOUT,
        )
    assert engine.pending() == []


@pytest.mark.asyncio
async def test_admitted_stream_requires_a_model_with_persistent_state(make_llm, engine):
    """On a model without persistent state the call raises
    VLLMValidationError itself, before it reserves or sends anything."""
    llm = make_llm(_engine_config("facebook/opt-125m"))
    inputs = _Inputs([1])
    task = _spawn(llm.add_request("caller", inputs(), _params(), stream_admission=True))
    with pytest.raises(VLLMValidationError):
        await _settled(llm, engine, task)
    assert engine.pending() == []
    assert inputs.reads == 0


@pytest.mark.asyncio
async def test_admitted_stream_queue_limit_rejects_before_sending(
    make_llm, engine, state_engine_args
):
    """check_admission runs once before the admission is sent: with the
    request queue full the call raises its rejection, sends nothing, reads
    nothing and frees the caller's request ID."""
    llm = make_llm(_engine_config(**state_engine_args, max_num_queued_reqs=1))
    await _admit(llm, engine, "first", _Inputs(then="end"))
    inputs = _Inputs([1])
    task = _spawn(llm.add_request("caller", inputs(), _params(), stream_admission=True))
    with pytest.raises(QueueOverflowError):
        await _settled(llm, engine, task)
    assert engine.pending() == []
    assert inputs.reads == 0
    with pytest.raises(ValueError):
        await asyncio.wait_for(llm.wait_stream_retired("caller"), TIMEOUT)


@pytest.mark.asyncio
async def test_admitted_stream_inputs_and_close_skip_the_queue_limit(
    make_llm, engine, state_engine_args
):
    """Only the admission is checked: with the queue at its limit because of
    the stream itself, its inputs and its close still reach the engine."""
    llm = make_llm(_engine_config(**state_engine_args, max_num_queued_reqs=1))
    _, internal = await _admit(llm, engine, inputs=_Inputs([1], [2]))
    sent = [await engine.add() for _ in range(3)]
    assert [request.request_id for request in sent] == [internal] * 3
    assert [request.resumable for request in sent] == [True, True, False]


@pytest.mark.asyncio
async def test_admitted_stream_counts_toward_the_queue_limit_without_input(
    make_llm, engine, state_engine_args
):
    """From registration an admitted stream is one unfinished request for the
    queue limit of later requests, although it has no input yet."""
    llm = make_llm(_engine_config(**state_engine_args, max_num_queued_reqs=1))
    await _admit(llm, engine, inputs=_Inputs(then="end"))
    with pytest.raises(QueueOverflowError):
        await asyncio.wait_for(
            llm.add_request("other", TokensPrompt(prompt_token_ids=[1]), _params()),
            TIMEOUT,
        )


@pytest.mark.asyncio
async def test_refused_admission_stops_counting_toward_the_queue_limit(
    make_llm, engine, state_engine_args
):
    """A refused admission removes its output state, so the next request is
    not counted against it."""
    llm = make_llm(_engine_config(**state_engine_args, max_num_queued_reqs=1))
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    call_id, admission = await engine.admission()
    engine.refuse(call_id, admission, "capacity")
    with pytest.raises(StreamAdmissionError):
        await asyncio.wait_for(task, TIMEOUT)
    await asyncio.wait_for(
        llm.add_request("other", TokensPrompt(prompt_token_ids=[1]), _params()),
        TIMEOUT,
    )
    assert (await engine.add()).external_req_id == "other"


@pytest.mark.asyncio
async def test_stream_refusal_is_not_an_admission_rejection(
    make_llm, engine, state_engine_args, rejections
):
    """A refusal that admit_stream returns is not counted among the frontend's
    admission rejections; a check_admission rejection of an admitted stream
    is, as for any request."""
    llm = make_llm(
        _engine_config(**state_engine_args, max_num_queued_reqs=1),
        log_stats=True,
        stat_loggers=[_RejectionLogger],
    )
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    call_id, admission = await engine.admission()
    engine.refuse(call_id, admission, "capacity")
    with pytest.raises(StreamAdmissionError):
        await asyncio.wait_for(task, TIMEOUT)
    assert rejections == []

    await _admit(llm, engine, "first", _Inputs(then="end"))
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    with pytest.raises(QueueOverflowError):
        await _settled(llm, engine, task)
    assert rejections == ["max_num_queued_reqs"]


_UTILITY_FAILURE = "Call to admit_stream method failed: Admission requires an empty"


@pytest.mark.asyncio
async def test_admission_failure_reaches_add_request_unchanged(llm, engine):
    """An admission that fails with an exception raises from add_request() as
    the client raised it, a plain Exception carrying the engine's message; the
    output state is removed, the generator is unread, a pending wait raises
    the same failure and the caller's request ID is free again."""
    inputs = _Inputs([1])
    task = _spawn(llm.add_request("caller", inputs(), _params(), stream_admission=True))
    call_id, admission = await engine.admission()
    waiter = _spawn(llm.wait_stream_retired("caller"))
    await engine.client_barrier(llm.engine_core)
    engine.fail(call_id, _UTILITY_FAILURE)
    with pytest.raises(Exception, match=_UTILITY_FAILURE) as failure:
        await asyncio.wait_for(task, TIMEOUT)
    assert type(failure.value) is Exception
    with pytest.raises(Exception, match=_UTILITY_FAILURE) as waited:
        await asyncio.wait_for(waiter, TIMEOUT)
    assert type(waited.value) is Exception
    assert not llm.output_processor.has_request(admission.request_id)
    assert inputs.reads == 0
    _, internal = await _admit(llm, engine)
    assert internal != admission.request_id


@pytest.mark.asyncio
async def test_generate_wraps_an_admission_failure_in_engine_generate_error(
    llm, engine
):
    stream = llm.generate(_Inputs([1])(), _params(), "caller", stream_admission=True)
    first = _spawn(stream.__anext__())
    call_id, _ = await engine.admission()
    engine.fail(call_id, _UTILITY_FAILURE)
    with pytest.raises(EngineGenerateError):
        await asyncio.wait_for(first, TIMEOUT)


@pytest.mark.asyncio
async def test_generate_raises_a_refusal_as_stream_admission_error(llm, engine):
    stream = llm.generate(_Inputs([1])(), _params(), "caller", stream_admission=True)
    first = _spawn(stream.__anext__())
    call_id, admission = await engine.admission()
    engine.refuse(call_id, admission, "active_limit")
    with pytest.raises(StreamAdmissionError) as refused:
        await asyncio.wait_for(first, TIMEOUT)
    assert refused.value.result == StreamAdmissionResult(
        False, admission.request_id, "active_limit"
    )


@pytest.mark.asyncio
async def test_admitted_stream_ended_by_an_unmarked_abort_output_aborts_the_session(
    llm, engine
):
    """An abort output the engine core sends outside the scheduler has
    neither ends_stream nor stream_output: it ends the caller's stream, and
    the frontend aborts the session the engine may still hold."""
    stream, first, internal = await _start_generate(
        llm, engine, _Inputs([1], then="end")
    )
    await engine.add()
    engine.push(
        EngineCoreOutputs(
            outputs=[EngineCoreOutput(internal, [], finish_reason=FinishReason.ABORT)]
        )
    )
    final = await asyncio.wait_for(first, TIMEOUT)
    assert final.finished is True
    assert await engine.abort() == [internal]


# EngineCoreClient: admission failures, duplicates and shutdown.


@pytest.mark.asyncio
async def test_retirement_failed_admission_fails_pending_waits(client, engine):
    """A direct caller's admission is sent as given, here with a prompt for
    the engine to refuse. When the engine's utility call fails, the call
    raises the failure as a plain Exception; pending waits raise what the
    call raised and the stream is no longer tracked."""
    task = await _admit_on_client(client, engine, "s-1", prompt_token_ids=[1, 2])
    await engine.client_barrier(client)
    if task.done():
        task.result()  # Raises what the client decided without the engine.
    call_id, sent = await engine.admission()
    assert list(sent.prompt_token_ids) == [1, 2]
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)
    engine.fail(call_id, _UTILITY_FAILURE)
    for pending in (task, waiter):
        with pytest.raises(Exception, match=_UTILITY_FAILURE) as failure:
            await asyncio.wait_for(pending, TIMEOUT)
        assert type(failure.value) is Exception
    with pytest.raises(ValueError):
        await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", ["pending", "accepted"])
async def test_retirement_admission_under_a_tracked_id_is_refused_by_the_client(
    client, engine, existing
):
    """While an admission under a request ID is pending or its stream has not
    retired, another admission under that ID raises request_id_in_use from
    its own call without being sent, whatever it holds, here a prompt that
    cannot be serialized: the existing admission, its waits and its tracking
    stay, and its retirement reaches its wait. Under an ID that is not
    tracked the encoder's error is the call's, and the ID stays untracked."""
    first = await _admit_on_client(client, engine, "s-1")
    first_call, first_sent = await engine.admission()
    if existing == "accepted":
        engine.accept(first_call, first_sent)
        assert (await asyncio.wait_for(first, TIMEOUT)).accepted
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)

    duplicate = await _admit_on_client(client, engine, "s-1", prompt_token_ids=[2**100])
    with pytest.raises(StreamAdmissionError) as refused:
        await asyncio.wait_for(duplicate, TIMEOUT)
    assert refused.value.result == StreamAdmissionResult(
        False, "s-1", "request_id_in_use"
    )
    unencodable = await _admit_on_client(
        client, engine, "s-2", prompt_token_ids=[2**100]
    )
    with pytest.raises(OverflowError):
        await asyncio.wait_for(unencodable, TIMEOUT)
    with pytest.raises(ValueError):
        await asyncio.wait_for(client.wait_stream_retired_async("s-2"), TIMEOUT)
    await engine.client_barrier(client)
    assert engine.pending() == []
    assert len(engine.admission_frames) == 1
    assert not waiter.done()
    # The failed encode left nothing behind, so its ID is admitted.
    await _admitted_on_client(client, engine, "s-2")

    if existing == "pending":
        assert not first.done()
        engine.accept(first_call, first_sent)
        assert (await asyncio.wait_for(first, TIMEOUT)).accepted
        await engine.client_barrier(client)
        assert not waiter.done()
    engine.retire("s-1")
    await asyncio.wait_for(waiter, TIMEOUT)
    # Once the stream has retired, its ID is admitted again.
    await _admitted_on_client(client, engine, "s-1")


@pytest.mark.asyncio
async def test_retirement_cancelled_admission_settles_from_its_reply_without_resend(
    client, engine
):
    """Cancelling a call after its admission is sent keeps the reply: an
    accepted stream stays tracked, and the admission is sent only once."""
    task = await _admit_on_client(client, engine, "s-1")
    call_id, sent = await engine.admission()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    engine.accept(call_id, sent)
    await engine.client_barrier(client)
    assert not waiter.done()
    assert len(engine.admission_frames) == 1
    # The caller aborts the stream; the client never does.
    assert _messages_of(engine.pending(), EngineCoreRequestType.ABORT) == []
    engine.retire("s-1")
    await asyncio.wait_for(waiter, TIMEOUT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt",
    [np.empty(0, dtype=np.int32), np.array([1, 2], dtype=np.int32), [1, 2], []],
    ids=["empty_array", "array", "list", "empty_list"],
)
async def test_admission_prompt_reaches_the_engine_as_an_add_prompt_does(
    client, engine, prompt
):
    """The admission's request has the wire form of an ADD, so a direct
    caller's prompt, NumPy or list, empty or not, arrives as the same
    tokens."""
    request = EngineCoreRequest(
        request_id="s-1",
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=_params(),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=True,
    )
    task = _spawn(client.admit_stream_async(request))
    call_id, sent = await engine.admission()
    assert sent.prompt_token_ids is not None
    assert list(sent.prompt_token_ids) == list(prompt)
    engine.accept(call_id, sent)
    assert (await asyncio.wait_for(task, TIMEOUT)).accepted


@pytest.mark.asyncio
async def test_admission_that_cannot_be_decoded_fails_only_its_own_call(client, engine):
    """A direct caller's admission with a field of the wrong type fails its
    own call as a utility failure, a plain Exception, and stops tracking;
    the engine's decoding goes on, and the next admission succeeds."""
    request = EngineCoreRequest(
        request_id="s-1",
        prompt_token_ids=None,
        mm_features=None,
        sampling_params=_params(),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank="bad",
        resumable=True,
    )
    with pytest.raises(Exception, match="Call to admit_stream method failed") as e:
        await asyncio.wait_for(client.admit_stream_async(request), TIMEOUT)
    assert type(e.value) is Exception
    assert engine.pending() == []
    with pytest.raises(ValueError):
        await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)
    await _admitted_on_client(client, engine, "s-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["accepted", "refused"])
async def test_admission_accepted_after_cancellation_is_reported_once(
    client, engine, reply
):
    """on_late_accept is called once when a call cancelled after its
    admission was sent is then accepted, and not for a refusal."""
    reports = []
    request = EngineCoreRequest(
        request_id="s-1",
        prompt_token_ids=None,
        mm_features=None,
        sampling_params=_params(),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=True,
    )
    task = _spawn(
        client.admit_stream_async(request, on_late_accept=lambda: reports.append("s-1"))
    )
    call_id, sent = await engine.admission()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reports == []
    if reply == "accepted":
        engine.accept(call_id, sent)
    else:
        engine.refuse(call_id, sent)
    await engine.client_barrier(client)
    assert reports == (["s-1"] if reply == "accepted" else [])


@pytest.mark.asyncio
async def test_retirement_is_not_reported_once_a_shutdown_off_the_loop_has_begun(
    client, engine
):
    """A shutdown called from another thread marks the tracking dead before
    its blocking part, held here where it closes the sockets to the engine:
    a retirement that arrives meanwhile is not reported, and the pending
    wait raises EngineDeadError."""
    await _admitted_on_client(client, engine, "s-1")
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)
    # The shutdown blocks where it closes the sockets to the engine.
    engine.may_close.clear()
    thread = threading.Thread(target=client.shutdown)
    thread.start()
    try:
        await _until(engine.closing.is_set)
        engine.retire("s-1")
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(waiter, TIMEOUT)
    finally:
        engine.may_close.set()
        await asyncio.to_thread(thread.join, TIMEOUT)
    assert not thread.is_alive()
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)


def _end_client(client, engine, cause):
    if cause == "engine_death":
        engine.kill()
    else:
        client.shutdown()


_CLIENT_ENDS = pytest.mark.parametrize("cause", ["engine_death", "client_shutdown"])


@pytest.mark.asyncio
@_CLIENT_ENDS
async def test_retirement_pending_waits_and_admissions_fail_when_the_client_ends(
    client, engine, cause
):
    """When the engine dies or the client shuts down, every pending wait
    raises EngineDeadError: on a tracked stream, on a stream whose admission
    reply is outstanding although its retirement arrived, and one that a
    retirement woke but that has not resumed. The pending admission raises
    it too."""
    await _admitted_on_client(client, engine, "tracked")
    await _admitted_on_client(client, engine, "woken")
    pending = await _admit_on_client(client, engine, "pending")
    await engine.admission()
    waiters = [
        _spawn(client.wait_stream_retired_async(request_id))
        for request_id in ("tracked", "tracked", "pending", "woken")
    ]
    engine.retire("pending")
    await engine.client_barrier(client)
    assert not any(waiter.done() for waiter in waiters)
    # Read in one step with the end, before the woken wait resumes.
    engine.retire("woken")
    _end_client(client, engine, cause)
    results = await asyncio.wait_for(
        asyncio.gather(*waiters, pending, return_exceptions=True), TIMEOUT
    )
    assert [type(result) for result in results] == [EngineDeadError] * 5


@pytest.mark.asyncio
@_CLIENT_ENDS
async def test_retirement_no_later_wait_reports_a_retirement_once_the_client_ended(
    client, engine, cause
):
    """After the engine died or the client shut down, a wait raises
    EngineDeadError, also one for a retirement that arrived before."""
    await _admitted_on_client(client, engine, "tracked")
    await _admitted_on_client(client, engine, "retired")
    engine.retire("retired")
    await engine.client_barrier(client)
    _end_client(client, engine, cause)
    if cause == "engine_death":
        await _until(lambda: client.resources.engine_dead)
    for request_id in ("tracked", "retired"):
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(
                client.wait_stream_retired_async(request_id), TIMEOUT
            )
