# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admitted streams in AsyncLLM and the client's retirement signal.

AsyncLLM and AsyncMPClient are built with their real constructors from the
facebook/opt-125m config and tokenizer (no weights). Only the ZMQ sockets are
replaced: a fake engine on the other side decodes what the client sends and
pushes msgpack-encoded EngineCoreOutputs back, so the frontend runs end to end
on the CPU.
"""

import asyncio

import msgspec
import numpy as np
import pytest
import pytest_asyncio
import regex as re
import zmq

import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.exceptions import VLLMValidationError
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
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
    StreamAdmissionResult,
    StreamOutput,
    UtilityOutput,
)
from vllm.v1.engine import core_client as core_client_mod
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult
from vllm.v1.token_buffer import TokenBuffer

pytestmark = pytest.mark.cpu_test

# Deadlock guard for every wait; nothing here sleeps for a fixed time.
TIMEOUT = 5.0
_ADDRESSES = {"input_address": "inproc://in", "output_address": "inproc://out"}
# Bound so its signature is the utility's, as EngineCoreProc looks it up.
_ADMIT_STREAM = EngineCore.admit_stream.__get__(object())


def _context(update_seq, input_closed=False):
    from vllm.v1.engine import StreamAdmissionContext

    return StreamAdmissionContext(update_seq=update_seq, input_closed=input_closed)


def _stream_admission_error():
    # Assumed location: next to StreamAdmissionResult.
    from vllm.v1.engine import StreamAdmissionError

    return StreamAdmissionError


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
                # Decoded as EngineCoreProc does: as builtins, then converted to
                # the utility's argument types.
                self.admission_frames.append(frames)
                (request,) = EngineCoreProc._convert_msgspec_args(
                    _ADMIT_STREAM, payload[3]
                )
                payload[3] = (_as_engine_request(request),)
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
        self.reply(
            call_id, StreamAdmissionResult(True, request.request_id, "never_fed")
        )

    def refuse(self, call_id, request, reason="capacity"):
        self.reply(
            call_id,
            StreamAdmissionResult(False, request.request_id, "refused", reason),
        )

    def output(
        self,
        request_id,
        update_seq,
        tokens=(),
        finish=None,
        closed=False,
        stop_reason=None,
    ):
        self.push(
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id,
                        list(tokens),
                        finish_reason=finish,
                        stop_reason=stop_reason,
                        stream_output=StreamOutput(_context(update_seq, closed)),
                    )
                ]
            )
        )

    def rejection(self, request_id, update_seq, reason, closed=False):
        # The engine's rejection keeps an ERROR finish; the frontend converts it.
        self.push(
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id,
                        [],
                        finish_reason=FinishReason.ERROR,
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
        pass


@pytest.fixture(scope="module")
def vllm_config():
    return AsyncEngineArgs(
        model="facebook/opt-125m", enforce_eager=True, max_model_len=128
    ).create_engine_config()


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
async def llm(engine, vllm_config):
    _OWNED.clear()
    llm = AsyncLLM(
        vllm_config,
        UniProcExecutor,
        log_stats=False,
        log_requests=False,
        client_addresses=_ADDRESSES,
    )
    yield llm
    # Shut down first, so the engine's own background work is handled by
    # shutdown(); then cancel and await only the tasks this module owns.
    handler = llm.output_handler
    reader = llm.engine_core.resources.output_queue_task
    llm.shutdown()
    await _cancel_owned(handler, reader)


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
        prompt_token_ids=prompt_token_ids
        if isinstance(prompt_token_ids, np.ndarray)
        else list(prompt_token_ids),
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


def _messages_of(messages, kind):
    return [payload for message_kind, payload in messages if message_kind == kind]


# AsyncLLM: validation and admission.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {"n": 2},
        {"stop": ["end"]},
        {"output_kind": RequestOutputKind.FINAL_ONLY},
        {"structured_outputs": StructuredOutputsParams(regex="[0-9]+")},
    ],
    ids=["n", "stop", "final_only", "structured"],
)
async def test_admitted_stream_restricted_params_raise_validation_error(
    llm, engine, params
):
    """Admitted streams require the streaming-input and admission restrictions."""
    inputs = _Inputs([1])
    with pytest.raises(VLLMValidationError):
        await asyncio.wait_for(
            llm.add_request(
                "caller",
                inputs(),
                SamplingParams(**{"max_tokens": 4, **params}),
                stream_admission=True,
            ),
            TIMEOUT,
        )
    assert engine.pending() == []
    assert inputs.reads == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Stream admission requires persistent state",
        "Admission requires an empty resumable sampling request",
    ],
)
async def test_admitted_stream_admission_failure_raises_validation_error(
    llm, engine, message
):
    """Any failure of the admission utility is a VLLMValidationError carrying
    the engine's message (a non-state engine is one instance)."""
    inputs = _Inputs([1])
    task = _spawn(llm.add_request("caller", inputs(), _params(), stream_admission=True))
    call_id, admission = await engine.admission()
    engine.fail(call_id, message)
    with pytest.raises(VLLMValidationError, match=re.escape(message)):
        await asyncio.wait_for(task, TIMEOUT)
    assert not llm.output_processor.has_request(admission.request_id)
    assert inputs.reads == 0


@pytest.mark.asyncio
async def test_admitted_stream_a1_admits_before_reading_generator(llm, engine):
    """A1."""
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
async def test_admitted_stream_a1_a5_empty_generator_closes_with_one_end(llm, engine):
    """A1, A5."""
    stream, first, internal = await _start_generate(llm, engine, _Inputs())
    close = await engine.add()
    assert close.request_id == internal
    assert not close.resumable
    assert close.stream_admission == _context(1, True)
    assert not first.done()
    engine.output(internal, 1, (), FinishReason.STOP, closed=True)
    final = await asyncio.wait_for(first, TIMEOUT)
    assert final.finished is True
    assert final.outputs[0].token_ids == []
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(stream.__anext__(), TIMEOUT)
    assert not llm.output_processor.has_request(internal)


@pytest.mark.asyncio
async def test_admitted_stream_a2_duplicate_call_while_admission_pending(llm, engine):
    """A2."""
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    call_id, admission = await engine.admission()
    with pytest.raises(_stream_admission_error()) as refused:
        await asyncio.wait_for(
            llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True),
            TIMEOUT,
        )
    assert refused.value.result.reason == "request_id_in_use"
    assert engine.pending() == []
    engine.accept(call_id, admission)
    await asyncio.wait_for(task, TIMEOUT)


@pytest.mark.asyncio
async def test_admitted_stream_a2_caller_id_reserved_until_retired(llm, engine):
    """A2, R1."""
    _, internal = await _admit(llm, engine, inputs=_Inputs([1]))
    await engine.add()
    error_cls = _stream_admission_error()

    with pytest.raises(error_cls) as refused:
        await llm.add_request(
            "caller", _Inputs([1])(), _params(), stream_admission=True
        )
    assert refused.value.result.accepted is False
    assert refused.value.result.reason == "request_id_in_use"
    close = await engine.add()
    assert not close.resumable
    assert engine.pending() == []

    engine.output(internal, 2, finish=FinishReason.STOP, closed=True)
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
async def test_admitted_stream_a2_reservation_release_keeps_caller_retirement(
    llm, engine
):
    """A2, R1, R3."""
    _, internal = await _admit(llm, engine, inputs=_Inputs([1]))
    await engine.add()
    await engine.add()
    engine.output(internal, 2, finish=FinishReason.STOP, closed=True)
    engine.retire(internal)
    await engine.llm_barrier(llm)
    # Releasing the reservation must not consume the caller's retirement.
    await asyncio.wait_for(llm.wait_stream_retired("caller"), TIMEOUT)
    _, reused = await _admit(llm, engine)
    assert reused != internal


@pytest.mark.asyncio
async def test_admitted_stream_a2_pending_wait_stays_bound_to_earlier_stream(
    llm, engine
):
    """A2, R1: a wait pending on an ended but unretired stream resolves on that
    stream's retirement batch; a wait after the ID is reused resolves only on
    the new stream's retirement."""
    _, first = await _admit(llm, engine, inputs=_Inputs([1]))
    await engine.add()
    await engine.add()
    engine.output(first, 2, finish=FinishReason.STOP, closed=True)
    await engine.llm_barrier(llm)
    earlier = _spawn(llm.wait_stream_retired("caller"))
    await engine.llm_barrier(llm)
    assert not earlier.done()
    engine.retire(first)
    await engine.llm_barrier(llm)
    # Resolved by stream 1's retirement batch, before the ID is reused.
    assert earlier.done() and earlier.exception() is None
    _, second = await _admit(llm, engine, inputs=_Inputs([1]))
    assert second != first
    later = _spawn(llm.wait_stream_retired("caller"))
    await engine.llm_barrier(llm)
    assert not later.done()
    # A stray retirement of the earlier stream does not resolve the new wait.
    engine.retire(first)
    await engine.llm_barrier(llm)
    assert not later.done()
    engine.retire(second)
    await asyncio.wait_for(later, TIMEOUT)


@pytest.mark.asyncio
async def test_admitted_stream_a2_pending_wait_survives_overlapping_reuse(llm, engine):
    """A2, R1, R2: a re-admission that becomes runnable as soon as stream 1's
    retirement is read runs before the wait it wakes, so it can rebind the
    caller ID first; the earlier wait still reports stream 1, and a wait started
    after the re-admission is issued refers to stream 2."""
    _, first = await _admit(llm, engine, inputs=_Inputs([1]))
    await engine.add()
    await engine.add()
    engine.output(first, 2, finish=FinishReason.STOP, closed=True)
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
    await engine.llm_barrier(llm)
    assert not later.done()
    engine.retire(second)
    await asyncio.wait_for(later, TIMEOUT)


@pytest.mark.asyncio
@pytest.mark.parametrize("disable_randomization", [False, True])
async def test_admitted_stream_a2_internal_id_follows_randomization(
    llm, engine, monkeypatch, disable_randomization
):
    """A2."""
    monkeypatch.setattr(
        envs, "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", disable_randomization
    )
    _, internal = await _admit(llm, engine)
    assert (internal == "caller") == disable_randomization
    assert internal.startswith("caller")


@pytest.mark.asyncio
async def test_admitted_stream_a3_refusal_raises_and_frees_everything(llm, engine):
    """A3, R2."""
    inputs = _Inputs([1])
    task = _spawn(llm.add_request("caller", inputs(), _params(), stream_admission=True))
    call_id, admission = await engine.admission()
    engine.refuse(call_id, admission, "capacity")
    with pytest.raises(_stream_admission_error()) as refused:
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
async def test_admitted_stream_a3_cancelled_before_sent_frees_caller(llm, engine):
    """A2, A3, R2."""
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
async def test_admitted_stream_a3_cancel_after_send_accepted_late_reply(llm, engine):
    """A2, A3, R1, R2: the late acceptance is collected, the stream aborted,
    and the caller ID stays reserved until that stream retires."""
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    call_id, admission = await engine.admission()
    internal = admission.request_id
    waiter = _spawn(llm.wait_stream_retired("caller"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    engine.accept(call_id, admission)
    assert await engine.abort() == [internal]
    assert not waiter.done()

    with pytest.raises(_stream_admission_error()) as refused:
        await asyncio.wait_for(
            llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True),
            TIMEOUT,
        )
    assert refused.value.result.reason == "request_id_in_use"
    assert engine.pending() == []

    engine.output(internal, 0, (), FinishReason.ABORT, closed=True)
    engine.retire(internal)
    await asyncio.wait_for(waiter, TIMEOUT)
    _, reused = await _admit(llm, engine)
    assert reused != internal


@pytest.mark.asyncio
async def test_admitted_stream_a3_cancel_after_send_refused_late_reply(llm, engine):
    """A2, A3, R2."""
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    call_id, admission = await engine.admission()
    waiter = _spawn(llm.wait_stream_retired("caller"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    engine.refuse(call_id, admission)
    with pytest.raises(_stream_admission_error()):
        await asyncio.wait_for(waiter, TIMEOUT)
    await engine.llm_barrier(llm)
    assert _messages_of(engine.pending(), EngineCoreRequestType.ABORT) == []
    assert not llm.output_processor.has_request(admission.request_id)
    _, internal = await _admit(llm, engine)
    assert internal != admission.request_id


@pytest.mark.asyncio
async def test_admitted_stream_a3_cancel_after_send_completes_is_after_send(
    llm, engine
):
    """A2, A3, R2: cancelled once the admission send has completed but before
    the call resumes, the admission counts as sent: the reply is collected, a
    late acceptance is aborted, and the caller ID stays reserved until the
    stream retires."""
    engine.hold = "admit_stream"
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    await _until(engine.holding)
    waiter = _spawn(llm.wait_stream_retired("caller"))
    await engine.client_barrier(llm.engine_core)
    # Complete the send, then cancel before the call can resume.
    engine.release()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    call_id, admission = await engine.admission()
    internal = admission.request_id
    engine.accept(call_id, admission)
    assert await engine.abort() == [internal]
    await engine.llm_barrier(llm)
    assert not waiter.done()
    with pytest.raises(_stream_admission_error()) as refused:
        await asyncio.wait_for(
            llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True),
            TIMEOUT,
        )
    assert refused.value.result.reason == "request_id_in_use"
    engine.output(internal, 0, (), FinishReason.ABORT, closed=True)
    engine.retire(internal)
    await asyncio.wait_for(waiter, TIMEOUT)
    _, reused = await _admit(llm, engine)
    assert reused != internal


@pytest.mark.asyncio
async def test_admitted_stream_a2_wait_during_call_setup_follows_new_admission(
    llm, engine
):
    """A2, R2: a wait started after the call began, before the admission is
    sent, waits for the stream that call admits."""
    engine.hold = "get_supported_tasks"
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    await _until(engine.holding)
    waiter = _spawn(llm.wait_stream_retired("caller"))
    await engine.client_barrier(llm.engine_core)
    assert not waiter.done()
    engine.release()
    call_id, admission = await engine.admission()
    internal = admission.request_id
    engine.accept(call_id, admission)
    await asyncio.wait_for(task, TIMEOUT)
    await engine.add()
    await engine.add()
    engine.output(internal, 2, (), FinishReason.STOP, closed=True)
    await engine.llm_barrier(llm)
    assert not waiter.done()
    engine.retire(internal)
    await asyncio.wait_for(waiter, TIMEOUT)


@pytest.mark.asyncio
async def test_admitted_stream_a2_r2_wait_during_call_setup_sees_immediate_refusal(
    llm, engine
):
    """A2, R2: a wait started before the admission is sent raises the refusal,
    even when the refusal is read before the wait resumes."""
    engine.hold = "get_supported_tasks"
    task = _spawn(
        llm.add_request("caller", _Inputs([1])(), _params(), stream_admission=True)
    )
    await _until(engine.holding)
    waiter = _spawn(llm.wait_stream_retired("caller"))
    await engine.client_barrier(llm.engine_core)
    assert not waiter.done()
    engine.auto_refuse = "capacity"
    engine.release()
    with pytest.raises(_stream_admission_error()):
        await asyncio.wait_for(task, TIMEOUT)
    with pytest.raises(_stream_admission_error()) as refused:
        await asyncio.wait_for(waiter, TIMEOUT)
    assert refused.value.result.reason == "capacity"


# AsyncLLM: inputs, close, rejection and session end.


@pytest.mark.asyncio
async def test_admitted_stream_a4_a5_inputs_numbered_then_close(llm, engine):
    """A4, A5."""
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
async def test_admitted_stream_a4_invalid_item_aborts_unnumbered(llm, engine):
    """A4, A9."""
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
async def test_admitted_stream_a9_input_task_cleans_up_for_add_request(llm, engine):
    """A9."""
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
async def test_admitted_stream_a5_caller_stream_ends_on_input_closed_output(
    llm, engine
):
    """A5, A9."""
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

    engine.output(internal, 2, (), FinishReason.STOP, closed=True)
    final = await asyncio.wait_for(stream.__anext__(), TIMEOUT)
    assert final.finished is True
    assert final.outputs[0].token_ids == []
    engine.output(internal, 2, [5])
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(stream.__anext__(), TIMEOUT)


@pytest.mark.asyncio
async def test_admitted_stream_a6_rejection_reaches_caller_and_stream_continues(
    llm, engine
):
    """A6, A7."""
    stream, first, internal = await _start_generate(llm, engine, _Inputs([1], [2]))
    for _ in range(3):
        await engine.add()
    engine.rejection(internal, 2, "empty_update")
    rejected = await asyncio.wait_for(first, TIMEOUT)
    assert rejected.finished is False
    assert rejected.stream_rejection.input_index == 1
    assert rejected.stream_rejection.reason == "empty_update"
    assert rejected.stream_rejection.error is None
    engine.output(internal, 1, [7], FinishReason.STOP, closed=True)
    final = await asyncio.wait_for(stream.__anext__(), TIMEOUT)
    assert final.finished is True
    assert final.stream_rejection is None
    await engine.llm_barrier(llm)
    assert _messages_of(engine.pending(), EngineCoreRequestType.ABORT) == []


@pytest.mark.asyncio
async def test_admitted_stream_a8_rejected_close_aborts_without_rejection(llm, engine):
    """A8, A9."""
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
    engine.output(internal, 1, (), FinishReason.ABORT, closed=True)
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
async def test_admitted_stream_a9_engine_terminal_output_is_the_one_end(
    llm, engine, finish, stop_reason
):
    """A9."""
    inputs = _Inputs([1], then="end")
    stream, first, internal = await _start_generate(llm, engine, inputs)
    await engine.add()
    engine.output(internal, 1, [6], finish, closed=True, stop_reason=stop_reason)
    final = await asyncio.wait_for(first, TIMEOUT)
    assert final.finished is True
    assert final.outputs[0].finish_reason == str(finish)
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(stream.__anext__(), TIMEOUT)
    assert not llm.output_processor.has_request(internal)


@pytest.mark.asyncio
async def test_admitted_stream_a9_cancelled_generate_aborts_once(llm, engine):
    """A9."""
    inputs = _Inputs([1], then="end")
    stream, first, internal = await _start_generate(llm, engine, inputs)
    await engine.add()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await engine.abort() == [internal]
    engine.output(internal, 1, (), FinishReason.ABORT, closed=True)
    await engine.llm_barrier(llm)
    assert _messages_of(engine.pending(), EngineCoreRequestType.ABORT) == []
    assert not llm.output_processor.has_request(internal)


@pytest.mark.asyncio
async def test_admitted_stream_a9_terminal_output_beats_later_generator_error(
    llm, engine
):
    """A9: both reach the collector before it is read; the first wins."""
    inputs = _Inputs([1], then="raise")
    queue, internal = await _admit(llm, engine, inputs=inputs)
    await engine.add()
    engine.output(internal, 1, (), FinishReason.ERROR, closed=True)
    await engine.llm_barrier(llm)
    inputs.release()
    await _finished(inputs.task)
    end = queue.get_nowait()
    assert end.finished is True
    assert queue.get_nowait() is None


@pytest.mark.asyncio
async def test_admitted_stream_a9_generator_error_beats_later_terminal_output(
    llm, engine
):
    """A9: both reach the collector before it is read; the first wins."""
    inputs = _Inputs([1], then="raise")
    queue, internal = await _admit(llm, engine, inputs=inputs)
    await engine.add()
    inputs.release()
    assert await engine.abort() == [internal]
    await _finished(inputs.task)
    engine.output(internal, 1, (), FinishReason.ABORT, closed=True)
    await engine.llm_barrier(llm)
    with pytest.raises(Exception, match="generator failed"):
        queue.get_nowait()
    assert queue.get_nowait() is None


@pytest.mark.asyncio
async def test_admitted_stream_a9_terminal_output_beats_later_engine_death(llm, engine):
    """A9: both reach the collector before it is read; the first wins."""
    queue, internal = await _admit(llm, engine, inputs=_Inputs([1], then="end"))
    await engine.add()
    engine.output(internal, 1, (), FinishReason.ABORT, closed=True)
    await engine.llm_barrier(llm)
    engine.kill()
    await _until(lambda: not llm.is_running)
    end = queue.get_nowait()
    assert end.finished is True
    assert queue.get_nowait() is None


@pytest.mark.asyncio
async def test_admitted_stream_a9_engine_death_raises_once(llm, engine):
    """A9, R5."""
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
async def test_admitted_stream_a9_generator_error_abort_survives_generate_exit(
    llm, engine
):
    """A9: the input task's abort completes although generate() ends on the
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
async def test_admitted_stream_a9_engine_death_beats_later_generator_error(llm, engine):
    """A9: both reach the collector before it is read; the first wins."""
    inputs = _Inputs([1], then="raise")
    queue, internal = await _admit(llm, engine, inputs=inputs)
    await engine.add()
    engine.kill()
    await _until(lambda: not llm.is_running)
    inputs.release()
    await _finished(inputs.task)
    with pytest.raises(EngineDeadError):
        queue.get_nowait()
    assert queue.get_nowait() is None


@pytest.mark.asyncio
async def test_admitted_stream_a10_unadmitted_rejection_ends_session(llm, engine):
    """A10."""
    stream = llm.generate(_Inputs([1])(), _params(), "caller")
    first = _spawn(stream.__anext__())
    request = await engine.add()
    assert request.stream_admission is None
    engine.push(
        EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request.request_id,
                    [],
                    finish_reason=FinishReason.ERROR,
                    stream_output=StreamOutput(None, "validation_error"),
                )
            ]
        )
    )
    output = await asyncio.wait_for(first, TIMEOUT)
    assert output.finished is True
    assert not llm.output_processor.has_request(request.request_id)


@pytest.mark.asyncio
async def test_retirement_r1_wait_by_caller_id_unknown_raises(llm, engine):
    """R1, R3."""
    with pytest.raises(ValueError):
        await asyncio.wait_for(llm.wait_stream_retired("nobody"), TIMEOUT)


# EngineCoreClient: retirement tracking.


@pytest.mark.asyncio
async def test_admit_stream_utility_decodes_on_engine_path_out_of_band(
    engine, vllm_config, monkeypatch
):
    """The client's admit_stream call, with every buffer sent out of band,
    passes EngineCoreProc's utility decoding into EngineCore.admit_stream."""
    import queue

    from tests.v1.core.test_scheduler import _pausable_stream_core

    monkeypatch.setattr(envs, "VLLM_MSGPACK_ZERO_COPY_THRESHOLD", 0)
    _OWNED.clear()
    client = core_client_mod.AsyncMPClient(
        vllm_config, UniProcExecutor, False, client_addresses=_ADDRESSES
    )
    try:
        task = await _admit_on_client(client, engine, "stream")
        delivered = _spawn(engine.admission())
        await asyncio.wait(
            {task, delivered}, timeout=TIMEOUT, return_when=asyncio.FIRST_COMPLETED
        )
        if task.done():
            task.result()  # Raises why the admission was not delivered.
        call_id, request = await delivered
        (frames,) = engine.admission_frames
        proc = _pausable_stream_core()
        proc.output_queue = queue.Queue()
        # As EngineCoreProc's input thread decodes a utility call.
        proc._handle_client_request(
            EngineCoreRequestType.UTILITY, MsgpackDecoder().decode(frames)
        )
        _, outputs = proc.output_queue.get_nowait()
        reply = outputs.utility_output
        assert reply.failure_message is None
        result = msgspec.convert(reply.result.result, StreamAdmissionResult)
        assert result == StreamAdmissionResult(True, "stream", "never_fed")
        assert proc.scheduler.requests["stream"].prompt_token_ids == []
        engine.accept(call_id, request)
        assert (await asyncio.wait_for(task, TIMEOUT)).accepted
    finally:
        reader = client.resources.output_queue_task
        client.shutdown()
        await _cancel_owned(reader)


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
        assert result == StreamAdmissionResult(True, "stream", "never_fed")
    finally:
        reader = client.resources.output_queue_task
        client.shutdown()
        await _cancel_owned(reader)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt", [[1], np.array([1], dtype=np.int32)], ids=["list", "ndarray"]
)
async def test_admit_stream_async_refuses_a_prompt_before_tracking(
    client, engine, prompt
):
    """An admission with a prompt is refused as the engine would refuse it,
    before anything is tracked or sent."""
    task = await _admit_on_client(client, engine, "s-1", prompt_token_ids=prompt)
    with pytest.raises(ValueError, match="Admission requires an empty"):
        await asyncio.wait_for(task, TIMEOUT)
    await engine.client_barrier(client)
    assert engine.pending() == []
    assert engine.admission_frames == []
    with pytest.raises(ValueError, match="not tracked"):
        await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_r1_retirement_only_batch_is_read(client, engine):
    """R1."""
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
async def test_retirement_r2_wait_during_admission_follows_reply(client, engine):
    """R2."""
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
async def test_retirement_r2_retirement_before_admission_reply_is_kept(client, engine):
    """R2."""
    task = await _admit_on_client(client, engine, "s-1")
    call_id, sent = await engine.admission()
    engine.retire("s-1")
    engine.accept(call_id, sent)
    assert (await asyncio.wait_for(task, TIMEOUT)).accepted
    await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_r2_early_retirement_waits_for_admission_reply(client, engine):
    """R2: a wait pending while the admission reply is outstanding keeps
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
async def test_retirement_r2_r5_engine_death_after_early_retirement_fails_admission(
    client, engine
):
    """R2, R5: with the admission reply outstanding, engine death fails both
    the admission and the waits, even after an early retirement."""
    task = await _admit_on_client(client, engine, "s-1")
    await engine.admission()
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    engine.retire("s-1")
    await engine.client_barrier(client)
    engine.kill()
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(waiter, TIMEOUT)
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(task, TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_r2_identical_retry_joins_existing_tracking(client, engine):
    """R2: an identical admission retry keeps existing waits attached."""
    await _admitted_on_client(client, engine, "s-1")
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)
    await _admitted_on_client(client, engine, "s-1")
    await engine.client_barrier(client)
    assert not waiter.done()
    engine.retire("s-1")
    await asyncio.wait_for(waiter, TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_r2_conflicting_retry_refusal_keeps_live_tracking(
    client, engine
):
    """R2: refusing a conflicting retry does not end the live stream's tracking."""
    await _admitted_on_client(client, engine, "s-1")
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)
    task = await _admit_on_client(client, engine, "s-1", external_req_id="other")
    call_id, sent = await engine.admission()
    engine.refuse(call_id, sent, "request_id_in_use")
    assert not (await asyncio.wait_for(task, TIMEOUT)).accepted
    await engine.client_barrier(client)
    assert not waiter.done()
    engine.retire("s-1")
    await asyncio.wait_for(waiter, TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_r3_r4_woken_wait_cancelled_before_return_keeps_retirement(
    client, engine
):
    """R3, R4: a retirement is consumed only by a wait that returns it."""
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
async def test_retirement_r2_refusal_fails_pending_waits(client, engine):
    """R2."""
    task = await _admit_on_client(client, engine, "s-1")
    call_id, sent = await engine.admission()
    waiters = [_spawn(client.wait_stream_retired_async("s-1")) for _ in range(2)]
    await engine.client_barrier(client)
    engine.refuse(call_id, sent, "active_limit")
    assert not (await asyncio.wait_for(task, TIMEOUT)).accepted
    for waiter in waiters:
        with pytest.raises(_stream_admission_error()) as refused:
            await asyncio.wait_for(waiter, TIMEOUT)
        assert refused.value.result.reason == "active_limit"
    with pytest.raises(ValueError):
        await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_r2_cancel_before_send_fails_pending_waits(client, engine):
    """R2: the admission is never delivered to the engine."""
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
async def test_retirement_r2_r5_engine_death_during_admission_fails_waits(
    client, engine
):
    """R2, R5."""
    await _admit_on_client(client, engine, "s-1")
    await engine.admission()
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)
    assert not waiter.done()
    engine.kill()
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(waiter, TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_r2_r3_early_retirement_kept_until_first_wait(client, engine):
    """R2, R3."""
    await _admitted_on_client(client, engine, "s-1")
    engine.retire("s-1")
    await engine.client_barrier(client)
    await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)
    with pytest.raises(ValueError):
        await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_r3_unwaited_cap_drops_oldest(client, engine):
    """R3: eviction follows retirement age, not admission age."""
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
async def test_retirement_r3_new_admission_drops_same_caller_retirement(client, engine):
    """R3."""
    await _admitted_on_client(client, engine, "a-1", external_req_id="a")
    await _admitted_on_client(client, engine, "b-1", external_req_id="b")
    engine.retire("a-1", "b-1")
    await engine.client_barrier(client)
    await _admitted_on_client(client, engine, "a-2", external_req_id="a")
    with pytest.raises(ValueError):
        await asyncio.wait_for(client.wait_stream_retired_async("a-1"), TIMEOUT)
    await asyncio.wait_for(client.wait_stream_retired_async("b-1"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_r3_unknown_stream_raises_at_once(client, engine):
    """R3."""
    with pytest.raises(ValueError):
        await asyncio.wait_for(client.wait_stream_retired_async("never"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_r4_concurrent_waits_and_cancellation(client, engine):
    """R4."""
    await _admitted_on_client(client, engine, "s-1")
    waiters = [_spawn(client.wait_stream_retired_async("s-1")) for _ in range(3)]
    await engine.client_barrier(client)
    waiters[0].cancel()
    engine.retire("s-1")
    await asyncio.wait_for(asyncio.gather(*waiters[1:]), TIMEOUT)
    assert waiters[0].cancelled()


@pytest.mark.asyncio
async def test_retirement_r4_cancelled_wait_keeps_tracking(client, engine):
    """R4."""
    await _admitted_on_client(client, engine, "s-1")
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    engine.retire("s-1")
    await engine.client_barrier(client)
    await asyncio.wait_for(client.wait_stream_retired_async("s-1"), TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_r5_death_before_woken_wait_resumes_fails_it(client, engine):
    """R5: a wait that has not returned when the engine dies raises
    EngineDeadError, even if a retirement woke it first."""
    await _admitted_on_client(client, engine, "s-1")
    waiter = _spawn(client.wait_stream_retired_async("s-1"))
    await engine.client_barrier(client)
    # Both frames are read in one step, before the woken wait resumes.
    engine.retire("s-1")
    engine.kill()
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(waiter, TIMEOUT)


@pytest.mark.asyncio
async def test_retirement_r5_engine_death_fails_pending_waits(client, engine):
    """R5."""
    await _admitted_on_client(client, engine, "s-1")
    await _admitted_on_client(client, engine, "s-2")
    waiters = [
        _spawn(client.wait_stream_retired_async(request_id))
        for request_id in ("s-1", "s-1", "s-2")
    ]
    await engine.client_barrier(client)
    engine.kill()
    for waiter in waiters:
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(waiter, TIMEOUT)
