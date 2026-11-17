# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for AsyncLLM.shutdown() teardown ordering and shutdown rejection."""

import asyncio
import json
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.embed.protocol import EmbeddingCompletionRequest
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.exceptions import RetryableRequestError
from vllm.outputs import RequestError
from vllm.renderers.chat_utils import ChatTemplateConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
)
from vllm.v1.engine import async_llm as async_llm_module
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import (
    DPEngineCoreProc,
    EngineCore,
    EngineCoreProc,
    EngineShutdownState,
)
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.executor import Executor

pytestmark = pytest.mark.cpu_test

MODEL_NAME = "vllm-project/tiny-random-LlamaForCausalLM"
TIMEOUT_SEC = 10.0


class _BlockingDrainEngineCoreClient:
    """Minimal EngineCoreClient whose shutdown() blocks until released."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        shutdown_error: Exception | None = None,
        supported_tasks: tuple[str, ...] = ("embed",),
    ) -> None:
        self._loop = loop
        self._shutdown_error = shutdown_error
        self._supported_tasks = supported_tasks
        self.engine_ranks_managed = [0]
        self.resources = SimpleNamespace(stop_requested=False, engine_dead=False)
        self.requests: list[EngineCoreRequest] = []
        self.request_added = asyncio.Event()
        self.shutdown_entered = threading.Event()
        self.allow_exit = threading.Event()
        self.shutdown_timeouts: list[float | None] = []
        self._shutdown_called = False
        # Production MPClient does not push EngineDeadError into the output
        # handler on the graceful path; it cancels that task. Tests that need
        # the handler to observe the death set this True.
        self.signal_engine_dead = True
        self._output_queue: asyncio.Queue[EngineCoreOutputs | BaseException] = (
            asyncio.Queue()
        )

    async def get_supported_tasks_async(self) -> tuple[str, ...]:
        return self._supported_tasks

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self.requests.append(request)
        self.request_added.set()

    async def get_output_async(self) -> EngineCoreOutputs:
        item = await self._output_queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        return None

    def push_output(self, outputs: EngineCoreOutputs) -> None:
        self._output_queue.put_nowait(outputs)

    def shutdown(self, timeout: float | None = None) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self.shutdown_timeouts.append(timeout)
        self.shutdown_entered.set()
        if self._shutdown_error is not None:
            self.resources.stop_requested = True
            if not self._loop.is_closed():
                self._loop.call_soon_threadsafe(
                    self._output_queue.put_nowait, EngineDeadError()
                )
            raise self._shutdown_error
        assert self.allow_exit.wait(timeout=TIMEOUT_SEC)
        # BackgroundResources.__call__ sets engine_dead before MPClient.shutdown
        # returns, which is before AsyncLLM tears down the renderer.
        self.resources.stop_requested = True
        self.resources.engine_dead = True
        if self.signal_engine_dead and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(
                self._output_queue.put_nowait, EngineDeadError()
            )


def _make_async_llm(
    monkeypatch: pytest.MonkeyPatch,
    loop: asyncio.AbstractEventLoop,
    *,
    shutdown_error: Exception | None = None,
) -> tuple[AsyncLLM, _BlockingDrainEngineCoreClient]:
    vllm_config = AsyncEngineArgs(
        model=MODEL_NAME,
        runner="pooling",
        shutdown_timeout=5,
    ).create_engine_config()
    core_client = _BlockingDrainEngineCoreClient(loop, shutdown_error=shutdown_error)
    monkeypatch.setattr(
        async_llm_module.EngineCoreClient,
        "make_async_mp_client",
        lambda *args, **kwargs: core_client,
    )
    engine = AsyncLLM(vllm_config, executor_class=Executor, log_stats=False)
    return engine, core_client


@pytest.mark.asyncio
async def test_pooling_request_completes_during_engine_core_drain(
    monkeypatch: pytest.MonkeyPatch,
):
    loop = asyncio.get_running_loop()
    prometheus_shutdown_events: list[bool] = []
    real_shutdown_prometheus = async_llm_module.shutdown_prometheus

    def spy_shutdown_prometheus() -> None:
        prometheus_shutdown_events.append(core_client.allow_exit.is_set())
        real_shutdown_prometheus()

    monkeypatch.setattr(
        async_llm_module, "shutdown_prometheus", spy_shutdown_prometheus
    )

    engine, core_client = _make_async_llm(monkeypatch, loop)
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)],
    )
    serving = ServingEmbedding(
        engine,
        models,
        request_logger=None,
        chat_template_config=ChatTemplateConfig(chat_template=None),
    )

    request_task = asyncio.create_task(
        serving(EmbeddingCompletionRequest(model=MODEL_NAME, input="hello"))
    )
    shutdown_future: asyncio.Future[None] | None = None
    try:
        added_waiter = asyncio.create_task(core_client.request_added.wait())
        done, _ = await asyncio.wait(
            {added_waiter, request_task},
            timeout=TIMEOUT_SEC,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if request_task in done:
            added_waiter.cancel()
            await request_task
        assert added_waiter in done

        shutdown_future = loop.run_in_executor(None, engine.shutdown, 5.0)
        entered = await loop.run_in_executor(
            None, core_client.shutdown_entered.wait, TIMEOUT_SEC
        )
        assert entered

        req = core_client.requests[0]
        core_client.push_output(
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id=req.request_id,
                        new_token_ids=[],
                        pooling_output=torch.tensor([0.6, 0.8]),
                        finish_reason=FinishReason.STOP,
                    )
                ]
            )
        )

        # Renderer executors and Prometheus must stay alive while
        # engine_core.shutdown(timeout=...) is draining in-flight work.
        response = await asyncio.wait_for(request_task, timeout=TIMEOUT_SEC)
        assert response.status_code == 200
        payload = json.loads(response.body)
        assert len(payload["data"]) == 1
        assert len(payload["data"][0]["embedding"]) == 2
        assert prometheus_shutdown_events == []

        core_client.allow_exit.set()
        await asyncio.wait_for(shutdown_future, timeout=TIMEOUT_SEC)

        assert core_client.shutdown_timeouts == [5.0]
        assert prometheus_shutdown_events == [True]
        assert engine.output_handler is not None
        await asyncio.wait_for(
            asyncio.gather(engine.output_handler, return_exceptions=True),
            timeout=TIMEOUT_SEC,
        )
        assert engine.output_handler.done()
        with pytest.raises(RuntimeError):
            engine.renderer._executor.submit(lambda: None)
    finally:
        core_client.allow_exit.set()
        if not request_task.done():
            request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)
        if shutdown_future is not None:
            await asyncio.gather(shutdown_future, return_exceptions=True)
        engine.shutdown()


@pytest.mark.asyncio
async def test_shutdown_cleans_up_resources_when_engine_core_raises(
    monkeypatch: pytest.MonkeyPatch,
):
    loop = asyncio.get_running_loop()
    prometheus_calls = 0
    real_shutdown_prometheus = async_llm_module.shutdown_prometheus

    def spy_shutdown_prometheus() -> None:
        nonlocal prometheus_calls
        prometheus_calls += 1
        real_shutdown_prometheus()

    monkeypatch.setattr(
        async_llm_module, "shutdown_prometheus", spy_shutdown_prometheus
    )

    engine, _ = _make_async_llm(
        monkeypatch, loop, shutdown_error=RuntimeError("core shutdown error")
    )
    with pytest.raises(RuntimeError, match="core shutdown error"):
        await loop.run_in_executor(None, engine.shutdown, 5.0)

    assert prometheus_calls == 1
    assert engine.output_handler is not None
    await asyncio.wait_for(
        asyncio.gather(engine.output_handler, return_exceptions=True),
        timeout=TIMEOUT_SEC,
    )
    assert engine.output_handler.done()
    with pytest.raises(RuntimeError):
        engine.renderer._executor.submit(lambda: None)


@pytest.mark.parametrize(
    "shutdown_state",
    [
        EngineShutdownState.RUNNING,
        EngineShutdownState.REQUESTED,
        EngineShutdownState.SHUTTING_DOWN,
    ],
)
@pytest.mark.parametrize(
    "kind", ["new", "pd", "cleanup", "session", "session_finish", "duplicate"]
)
def test_engine_core_rejects_new_requests_during_shutdown(
    shutdown_state: EngineShutdownState,
    kind: str,
):
    """New requests arriving during EngineCore shutdown fail with a retryable
    error (mapped to HTTP 503), rather than FinishReason.ABORT (HTTP 200).
    Cleanup markers and open streaming-session inputs still pass through, and
    rejected P/D requests still trigger KV cleanup.

    A duplicate of a normal in-flight id is not an open session.
    `Scheduler.add_request` asserts "duplicate request id" for that reuse, which
    kills the busy loop, so it must get the same 503 and must not be added.
    The finish sentinel of a resumable session is not resumable itself, and
    still must be forwarded or the live session cannot end.
    """
    core = object.__new__(EngineCoreProc)
    core.shutdown_state = shutdown_state
    core.reject_new_requests = False
    core.output_queue = queue.Queue()
    core.scheduler = MagicMock()
    core.scheduler.requests = {}
    if kind in ("session", "session_finish"):
        core.scheduler.requests["r"] = SimpleNamespace(resumable=True)
    elif kind == "duplicate":
        # Live, but not a resumable session: `has_request` is true and
        # `streaming_queue` is unset, so `add_request` would assert.
        core.scheduler.requests["r"] = SimpleNamespace(resumable=False)
    core.scheduler.has_request.side_effect = lambda request_id: (
        request_id in core.scheduler.requests
    )
    core.add_request = MagicMock()

    request = MagicMock(
        request_id="r",
        client_index=1,
        abort_immediately=kind == "cleanup",
        resumable=kind == "session",
        kv_transfer_params={"do_remote_prefill": True} if kind == "pd" else None,
    )

    core._handle_client_request(EngineCoreRequestType.ADD, (request, 0))

    rejected = shutdown_state != EngineShutdownState.RUNNING and kind in (
        "new",
        "pd",
        "duplicate",
    )
    if rejected:
        client_index, eco = core.output_queue.get_nowait()
        assert client_index == 1
        assert core.output_queue.empty()
        assert len(eco.outputs) == 1
        out = eco.outputs[0]
        assert out.request_id == "r"
        assert out.finish_reason == FinishReason.ERROR
        assert isinstance(out.error, RequestError)
        assert out.error.code == "engine_shutting_down"
        assert out.error.retryable is True
        exc = out.error.to_exception()
        assert isinstance(exc, RetryableRequestError)
        error_response = create_error_response(exc)
        assert error_response.error.code == 503
        assert error_response.error.type == "ServiceUnavailableError"
    else:
        assert core.output_queue.empty()

    if rejected and kind != "pd":
        core.add_request.assert_not_called()
    elif rejected and kind == "pd":
        assert request.abort_immediately is True
        core.add_request.assert_called_once_with(request)
    else:
        assert request.abort_immediately == (kind == "cleanup")
        core.add_request.assert_called_once_with(request, 0)


@pytest.mark.asyncio
async def test_new_request_during_drain_surfaces_503_while_inflight_completes(
    monkeypatch: pytest.MonkeyPatch,
):
    loop = asyncio.get_running_loop()
    vllm_config = AsyncEngineArgs(
        model=MODEL_NAME,
        shutdown_timeout=5,
    ).create_engine_config()
    core_client = _BlockingDrainEngineCoreClient(loop, supported_tasks=("generate",))
    monkeypatch.setattr(
        async_llm_module.EngineCoreClient,
        "make_async_mp_client",
        lambda *args, **kwargs: core_client,
    )
    engine = AsyncLLM(vllm_config, executor_class=Executor, log_stats=False)

    core = object.__new__(EngineCoreProc)
    core.shutdown_state = EngineShutdownState.SHUTTING_DOWN
    core.reject_new_requests = False
    core.output_queue = queue.Queue()
    core.scheduler = MagicMock()
    core.scheduler.has_request.return_value = False
    core.add_request = MagicMock()

    async def collect_first_or_error(req_id: str, prompt_tokens: list[int]):
        outputs = []
        async for out in engine.generate(
            {"prompt_token_ids": prompt_tokens},
            SamplingParams(max_tokens=4),
            req_id,
        ):
            outputs.append(out)
        return outputs[-1]

    inflight_task = asyncio.create_task(collect_first_or_error("inflight-req", [1, 2]))
    rejected_task: asyncio.Task | None = None
    shutdown_future: asyncio.Future[None] | None = None
    try:
        await asyncio.wait_for(core_client.request_added.wait(), timeout=TIMEOUT_SEC)

        shutdown_future = loop.run_in_executor(None, engine.shutdown, 5.0)
        entered = await loop.run_in_executor(
            None, core_client.shutdown_entered.wait, TIMEOUT_SEC
        )
        assert entered

        core_client.request_added.clear()
        rejected_task = asyncio.create_task(collect_first_or_error("new-req", [3, 4]))
        await asyncio.wait_for(core_client.request_added.wait(), timeout=TIMEOUT_SEC)

        new_req = core_client.requests[1]
        core._handle_client_request(
            EngineCoreRequestType.ADD,
            (
                SimpleNamespace(
                    request_id=new_req.request_id,
                    client_index=0,
                    abort_immediately=False,
                    kv_transfer_params=None,
                ),
                0,
            ),
        )
        _, rejected_eco = core.output_queue.get_nowait()
        core_client.push_output(rejected_eco)

        with pytest.raises(RetryableRequestError) as exc_info:
            await asyncio.wait_for(rejected_task, timeout=TIMEOUT_SEC)
        error_response = create_error_response(exc_info.value)
        assert error_response.error.code == 503
        assert error_response.error.type == "ServiceUnavailableError"

        inflight_req = core_client.requests[0]
        core_client.push_output(
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id=inflight_req.request_id,
                        new_token_ids=[5],
                        finish_reason=FinishReason.STOP,
                    )
                ]
            )
        )
        inflight_out = await asyncio.wait_for(inflight_task, timeout=TIMEOUT_SEC)
        assert inflight_out.finished
        assert inflight_out.outputs[0].finish_reason == "stop"

        core_client.allow_exit.set()
        await asyncio.wait_for(shutdown_future, timeout=TIMEOUT_SEC)
    finally:
        core_client.allow_exit.set()
        for task in (inflight_task, rejected_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if shutdown_future is not None:
            await asyncio.gather(shutdown_future, return_exceptions=True)
        engine.shutdown()


def _generate_engine(
    monkeypatch: pytest.MonkeyPatch,
    loop: asyncio.AbstractEventLoop,
) -> tuple[AsyncLLM, _BlockingDrainEngineCoreClient]:
    vllm_config = AsyncEngineArgs(
        model=MODEL_NAME,
        shutdown_timeout=5,
    ).create_engine_config()
    core_client = _BlockingDrainEngineCoreClient(loop, supported_tasks=("generate",))
    monkeypatch.setattr(
        async_llm_module.EngineCoreClient,
        "make_async_mp_client",
        lambda *args, **kwargs: core_client,
    )
    engine = AsyncLLM(vllm_config, executor_class=Executor, log_stats=False)
    return engine, core_client


async def _collect_generate(engine: AsyncLLM, request_id: str):
    outputs = []
    async for out in engine.generate(
        {"prompt_token_ids": [1, 2]},
        SamplingParams(max_tokens=4),
        request_id,
    ):
        outputs.append(out)
    return outputs[-1]


@pytest.mark.asyncio
async def test_new_request_after_engine_core_shutdown_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
):
    """A request that arrives after engine_core.shutdown() returns, while the
    renderer is still up, raises EngineDeadError instead of hanging.

    On main the renderer executor was already closed here, so the same request
    failed immediately with HTTP 500. Shutdown now keeps the renderer until
    the engine process is gone; add_request() must still fail before it waits
    on that process.
    """
    loop = asyncio.get_running_loop()
    engine, core_client = _generate_engine(monkeypatch, loop)
    renderer_shutdown_entered = threading.Event()
    release_renderer_shutdown = threading.Event()
    real_renderer_shutdown = engine.renderer.shutdown

    def blocking_renderer_shutdown() -> None:
        renderer_shutdown_entered.set()
        assert release_renderer_shutdown.wait(timeout=TIMEOUT_SEC)
        real_renderer_shutdown()

    engine.renderer.shutdown = blocking_renderer_shutdown  # type: ignore[method-assign]
    core_client.allow_exit.set()
    shutdown_future = loop.run_in_executor(None, engine.shutdown, 5.0)
    late_task: asyncio.Task | None = None
    try:
        entered = await loop.run_in_executor(
            None, renderer_shutdown_entered.wait, TIMEOUT_SEC
        )
        assert entered
        assert engine._shut_down
        # The pool is still the one serving requests. submit() raises only
        # after renderer.shutdown() closes it.
        assert (
            engine.renderer._executor.submit(lambda: None).result(timeout=TIMEOUT_SEC)
            is None
        )

        late_task = asyncio.create_task(_collect_generate(engine, "after-shutdown"))
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(late_task, timeout=TIMEOUT_SEC)
    finally:
        release_renderer_shutdown.set()
        if late_task is not None and not late_task.done():
            late_task.cancel()
            await asyncio.gather(late_task, return_exceptions=True)
        await asyncio.gather(shutdown_future, return_exceptions=True)
        engine.shutdown()


@pytest.mark.asyncio
async def test_request_waiting_when_engine_core_exits_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
):
    """A request already waiting on the output handler when EngineCore exits
    must not block until the HTTP server closes.

    Graceful MPClient shutdown cancels the output handler instead of
    delivering EngineDeadError through the socket task. CancelledError is not
    an Exception, so the handler used to skip propagate_error.
    """
    loop = asyncio.get_running_loop()
    engine, core_client = _generate_engine(monkeypatch, loop)
    # Match production: the socket task is cancelled along with the handler,
    # so the handler never observes a queued EngineDeadError.
    core_client.signal_engine_dead = False

    inflight_task = asyncio.create_task(_collect_generate(engine, "inflight-req"))
    late_task: asyncio.Task | None = None
    shutdown_future: asyncio.Future[None] | None = None
    try:
        await asyncio.wait_for(core_client.request_added.wait(), timeout=TIMEOUT_SEC)
        shutdown_future = loop.run_in_executor(None, engine.shutdown, 5.0)
        entered = await loop.run_in_executor(
            None, core_client.shutdown_entered.wait, TIMEOUT_SEC
        )
        assert entered

        inflight_req = core_client.requests[0]
        core_client.push_output(
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id=inflight_req.request_id,
                        new_token_ids=[5],
                        finish_reason=FinishReason.STOP,
                    )
                ]
            )
        )
        inflight_out = await asyncio.wait_for(inflight_task, timeout=TIMEOUT_SEC)
        assert inflight_out.finished

        core_client.request_added.clear()
        late_task = asyncio.create_task(_collect_generate(engine, "after-exit"))
        await asyncio.wait_for(core_client.request_added.wait(), timeout=TIMEOUT_SEC)
        # The late request is registered and blocked on its collector.
        # Release engine_core.shutdown(); the handler is cancelled with no
        # output ever pushed for this request.
        core_client.allow_exit.set()
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(late_task, timeout=TIMEOUT_SEC)
        await asyncio.wait_for(shutdown_future, timeout=TIMEOUT_SEC)
    finally:
        core_client.allow_exit.set()
        for task in (inflight_task, late_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if shutdown_future is not None:
            await asyncio.gather(shutdown_future, return_exceptions=True)
        engine.shutdown()


@pytest.mark.parametrize(
    "shutdown_state",
    [EngineShutdownState.REQUESTED, EngineShutdownState.SHUTTING_DOWN],
)
def test_dp_rank_does_not_start_wave_for_adds_during_shutdown(
    shutdown_state: EngineShutdownState,
):
    """An add during shutdown must not publish `start_wave`.

    `DPEngineCoreProc.add_request` wakes idle peers when the scheduler is
    unpaused. A rejected prefill/decode request is added only to free remote
    KV, with `abort_immediately` set. During shutdown an idle peer may already
    have destroyed `dp_group`, so the wave's all-reduce hangs until SIGKILL.
    """
    core = object.__new__(DPEngineCoreProc)
    core.has_coordinator = True
    core.current_wave = 0
    core.engines_running = False
    core.shutdown_state = shutdown_state
    core.scheduler = MagicMock(pause_state=PauseState.UNPAUSED)
    core.output_queue = MagicMock()
    request = SimpleNamespace(request_id="r", abort_immediately=True)

    with patch.object(EngineCore, "add_request"):
        core.add_request(request, 0)

    assert core.engines_running is False
    core.output_queue.put_nowait.assert_not_called()
