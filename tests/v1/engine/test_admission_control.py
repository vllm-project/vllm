# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for admission control (max_num_queued_reqs / max_num_queued_tokens).

These tests cover:
- OutputProcessor.get_num_queued_tokens() token counting
- AsyncLLM.check_admission() admission control logic
- Admission rejection Prometheus samples
- Exception classes (GracefulHTTPError, QueueOverflowError, MaxQueuedTokensError)
- create_error_response() mapping GracefulHTTPError to HTTP 503
- SchedulerConfig field defaults and validation
- human_readable_int CLI notation for max_num_queued_tokens
"""

import argparse
import asyncio
import logging
import multiprocessing
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client import REGISTRY
from pydantic import ValidationError

from vllm.config import VllmConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.exceptions import (
    GracefulHTTPError,
    MaxQueuedTokensError,
    QueueOverflowError,
    VLLMError,
)
from vllm.outputs import STREAM_FINISHED
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import human_readable_int
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.admission_control import SharedAdmissionStats
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineGenerateError
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.metrics.loggers import StatLoggerManager

_ADMISSION_METRIC_MODEL = "admission-metric-test"
_ADMISSION_LOGGER = "vllm.v1.engine.async_llm"

pytestmark = pytest.mark.cpu_test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_req_state(prompt_len: int, is_prefilling: bool = True):
    return SimpleNamespace(
        prompt_len=prompt_len,
        is_prefilling=is_prefilling,
    )


def _make_async_llm(
    max_num_queued_reqs: int | None = None,
    max_num_queued_tokens: int | None = None,
    num_unfinished: int = 0,
    num_queued_tokens: int = 0,
) -> AsyncLLM:
    """Create a bare AsyncLLM with just the attributes needed for scheduling."""
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.vllm_config = VllmConfig()
    llm.scheduler_config = SimpleNamespace(
        max_num_queued_reqs=max_num_queued_reqs,
        max_num_queued_tokens=max_num_queued_tokens,
    )
    llm.output_processor = MagicMock()
    llm.output_processor.get_num_unfinished_requests.return_value = num_unfinished
    llm.output_processor.get_num_queued_tokens.return_value = num_queued_tokens
    llm.logger_manager = None
    llm.admission_stats = None
    return llm


def _attach_prometheus_logger(
    llm: AsyncLLM, engine_indexes: list[int] | None = None
) -> None:
    """Attach a real Prometheus logger manager to a lightweight AsyncLLM."""
    vllm_config = MagicMock()
    vllm_config.observability_config.show_hidden_metrics = False
    vllm_config.observability_config.kv_cache_metrics = False
    vllm_config.observability_config.custom_histogram_buckets = None
    vllm_config.model_config.served_model_name = _ADMISSION_METRIC_MODEL
    vllm_config.model_config.max_model_len = 2048
    vllm_config.model_config.is_diffusion = False
    vllm_config.speculative_config = None
    vllm_config.kv_transfer_config = None
    vllm_config.ec_transfer_config = None
    vllm_config.lora_config = None
    llm.logger_manager = StatLoggerManager(
        vllm_config=vllm_config,
        engine_idxs=engine_indexes,
        enable_default_loggers=False,
    )


def _admission_rejection_count(reason: str) -> float:
    """Read the Prometheus sample for one admission rejection reason."""
    value = REGISTRY.get_sample_value(
        "vllm:admission_rejections_total",
        {"model_name": _ADMISSION_METRIC_MODEL, "reason": reason},
    )
    assert value is not None
    return value


def _make_output_processor(**request_states) -> OutputProcessor:
    op = OutputProcessor.__new__(OutputProcessor)
    op.request_states = request_states
    return op


def _make_engine_request(request_id: str, n: int) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(n=n),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _make_request_test_llm(
    max_num_queued_reqs: int,
    add_request_async: Callable[[EngineCoreRequest], Awaitable[None]],
) -> AsyncLLM:
    llm = _make_async_llm(max_num_queued_reqs=max_num_queued_reqs)
    llm.output_processor = OutputProcessor(None, collect_stats=False)
    llm.engine_core = SimpleNamespace(
        resources=SimpleNamespace(engine_dead=False),
        add_request_async=AsyncMock(side_effect=add_request_async),
        abort_requests_async=AsyncMock(),
        shutdown=MagicMock(),
    )
    llm.output_handler = None
    llm.log_requests = False
    llm._run_output_handler = MagicMock()  # type: ignore[method-assign]
    llm.input_processor = MagicMock()
    llm.input_processor.assign_request_id.side_effect = lambda request: setattr(
        request, "external_req_id", request.request_id
    )
    return llm


def _make_scheduler_config(**kwargs) -> SchedulerConfig:
    return SchedulerConfig(
        runner_type="generate",
        max_model_len=4096,
        is_encoder_decoder=False,
        **kwargs,
    )


def _make_shared_stats(client_count: int = 2) -> list[SharedAdmissionStats]:
    counters = multiprocessing.RawArray(
        "q", SharedAdmissionStats.num_counters(client_count)
    )
    return [
        SharedAdmissionStats({"mp_admission_counters": counters}, client_count, index)
        for index in range(client_count)
    ]


@pytest.mark.skip_global_cleanup
def test_shared_admission_stats_aggregate_api_servers():
    server_0, server_1 = _make_shared_stats()

    server_0.set_num_requests(2)
    assert server_0.get_num_requests() == 2
    server_1.set_num_requests(3)
    assert server_0.get_num_requests() == server_1.get_num_requests() == 5

    server_0.set_num_requests(1)
    assert server_0.get_num_requests() == 4


@pytest.mark.skip_global_cleanup
def test_shared_request_snapshot_enforces_global_limit():
    server_0, server_1 = _make_shared_stats()
    server_0.set_num_requests(2)
    llm = _make_async_llm(max_num_queued_reqs=2)
    llm.admission_stats = server_1
    _attach_prometheus_logger(llm)

    with pytest.raises(QueueOverflowError):
        llm.check_admission()

    assert server_0.get_num_requests() == 2
    assert _admission_rejection_count("max_num_queued_reqs") == 1


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


def test_graceful_http_error_carries_status_and_message():
    err = GracefulHTTPError("custom message", HTTPStatus.SERVICE_UNAVAILABLE)
    assert err.message == "custom message"
    assert err.http_status == HTTPStatus.SERVICE_UNAVAILABLE
    assert str(err) == "custom message"


def test_graceful_http_error_is_vllm_error():
    err = GracefulHTTPError("msg", HTTPStatus.TOO_MANY_REQUESTS)
    assert isinstance(err, VLLMError)


def test_queue_overflow_error():
    err = QueueOverflowError()
    assert err.http_status == HTTPStatus.SERVICE_UNAVAILABLE
    assert isinstance(err, GracefulHTTPError)
    assert "busy" in err.message.lower() or "try again" in err.message.lower()


def test_max_queued_tokens_error():
    err = MaxQueuedTokensError()
    assert err.http_status == HTTPStatus.SERVICE_UNAVAILABLE
    assert isinstance(err, GracefulHTTPError)
    assert "backlog" in err.message.lower() or "try again" in err.message.lower()


@pytest.mark.parametrize("exc_cls", [QueueOverflowError, MaxQueuedTokensError])
def test_admission_exceptions_are_vllm_errors(exc_cls):
    """Admission rejections must reach the VLLMError HTTP handler."""
    assert issubclass(exc_cls, VLLMError)


# ---------------------------------------------------------------------------
# OutputProcessor.get_num_queued_tokens
# ---------------------------------------------------------------------------


def test_queued_tokens_empty():
    assert _make_output_processor().get_num_queued_tokens() == 0


def test_queued_tokens_sums_prefilling_requests():
    op = _make_output_processor(r1=_make_req_state(100), r2=_make_req_state(200))
    assert op.get_num_queued_tokens() == 300


def test_queued_tokens_excludes_non_prefilling():
    op = _make_output_processor(
        r1=_make_req_state(100, is_prefilling=True),
        r2=_make_req_state(200, is_prefilling=False),
        r3=_make_req_state(50, is_prefilling=True),
    )
    assert op.get_num_queued_tokens() == 150


def test_queued_tokens_all_non_prefilling():
    op = _make_output_processor(
        r1=_make_req_state(100, is_prefilling=False),
        r2=_make_req_state(200, is_prefilling=False),
    )
    assert op.get_num_queued_tokens() == 0


# ---------------------------------------------------------------------------
# AsyncLLM.check_admission
# ---------------------------------------------------------------------------


def test_admission_no_limits_allows_everything():
    llm = _make_async_llm(num_unfinished=999, num_queued_tokens=999)
    llm.check_admission()


# -- max_num_queued_reqs ----------------------------------------------------


def test_admission_reqs_allows_when_under_limit():
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=5)
    _attach_prometheus_logger(llm)
    llm.check_admission()
    llm.check_admission()
    assert _admission_rejection_count("max_num_queued_reqs") == 0
    assert _admission_rejection_count("max_num_queued_tokens") == 0


def test_admission_reqs_rejects_at_limit(caplog):
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=10)
    _attach_prometheus_logger(llm, engine_indexes=[0, 1, 2])
    with (
        caplog.at_level(logging.INFO, logger=_ADMISSION_LOGGER),
        pytest.raises(QueueOverflowError),
    ):
        llm.check_admission()
    assert _admission_rejection_count("max_num_queued_reqs") == 1
    assert _admission_rejection_count("max_num_queued_tokens") == 0
    assert not any(
        "Request queue full" in record.getMessage() for record in caplog.records
    )


def test_admission_reqs_rejects_with_n():
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=8)
    _attach_prometheus_logger(llm)
    with pytest.raises(QueueOverflowError):
        llm.check_admission(3)
    assert _admission_rejection_count("max_num_queued_reqs") == 1
    assert _admission_rejection_count("max_num_queued_tokens") == 0


def test_admission_reqs_allows_n_at_boundary():
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=7)
    llm.check_admission(3)


def test_admission_reqs_rejects_when_zero_limit():
    llm = _make_async_llm(max_num_queued_reqs=0, num_unfinished=0)
    with pytest.raises(QueueOverflowError):
        llm.check_admission()


# -- max_num_queued_tokens --------------------------------------------------


def test_admission_tokens_allows_when_under_limit():
    llm = _make_async_llm(max_num_queued_tokens=1000, num_queued_tokens=500)
    llm.check_admission()


def test_admission_tokens_rejects_at_limit():
    llm = _make_async_llm(max_num_queued_tokens=1000, num_queued_tokens=1000)
    _attach_prometheus_logger(llm)
    with pytest.raises(MaxQueuedTokensError):
        llm.check_admission()
    assert _admission_rejection_count("max_num_queued_reqs") == 0
    assert _admission_rejection_count("max_num_queued_tokens") == 1


def test_admission_tokens_rejects_over_limit():
    llm = _make_async_llm(max_num_queued_tokens=1000, num_queued_tokens=1500)
    with pytest.raises(MaxQueuedTokensError):
        llm.check_admission()


def test_admission_tokens_rejects_when_zero_limit():
    llm = _make_async_llm(max_num_queued_tokens=0, num_queued_tokens=0)
    with pytest.raises(MaxQueuedTokensError):
        llm.check_admission()


# -- interaction between both limits ----------------------------------------


def test_admission_both_limits_checked_independently():
    llm = _make_async_llm(
        max_num_queued_reqs=100,
        max_num_queued_tokens=1000,
        num_unfinished=5,
        num_queued_tokens=1000,
    )
    with pytest.raises(MaxQueuedTokensError):
        llm.check_admission()


def test_admission_req_limit_checked_before_token_limit():
    llm = _make_async_llm(
        max_num_queued_reqs=10,
        max_num_queued_tokens=1000,
        num_unfinished=10,
        num_queued_tokens=1000,
    )
    _attach_prometheus_logger(llm)
    with pytest.raises(QueueOverflowError):
        llm.check_admission()
    assert _admission_rejection_count("max_num_queued_reqs") == 1
    assert _admission_rejection_count("max_num_queued_tokens") == 0


# -- n derived from params at the add_request call site ---------------------


@pytest.mark.parametrize("params", [SamplingParams(), PoolingParams()])
def test_admission_params_without_explicit_n_count_as_one_slot(params):
    """PoolingParams has no `.n`; add_request must fall back to 1."""
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=9)
    llm.check_admission(getattr(params, "n", 1) or 1)

    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=10)
    with pytest.raises(QueueOverflowError):
        llm.check_admission(getattr(params, "n", 1) or 1)


@pytest.mark.asyncio
async def test_concurrent_single_request_admission_respects_limit():
    """Concurrent n=1 admissions cannot consume the same slot."""
    llm = _make_async_llm(max_num_queued_reqs=1)
    output_processor: MagicMock = llm.output_processor
    request_states: dict[str, SimpleNamespace] = {}
    output_processor.get_num_unfinished_requests.side_effect = lambda: len(
        request_states
    )
    output_processor.has_request.side_effect = request_states.__contains__

    def add_request(request, *_args):
        request_states[request.request_id] = _make_req_state(
            len(request.prompt_token_ids)
        )

    output_processor.add_request.side_effect = add_request

    async def add_request_async(_request):
        await asyncio.sleep(0)

    llm.engine_core = SimpleNamespace(
        add_request_async=AsyncMock(side_effect=add_request_async),
        shutdown=MagicMock(),
    )
    llm.log_requests = False

    requests = [
        SimpleNamespace(request_id=f"request-{idx}", prompt_token_ids=[idx])
        for idx in range(2)
    ]
    results = await asyncio.gather(
        *(
            llm._add_request(request, None, None, 0, MagicMock())
            for request in requests
        ),
        return_exceptions=True,
    )

    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, QueueOverflowError) for result in results) == 1
    assert len(request_states) == 1


def test_admission_preflight_pass_then_reject_records_once():
    llm = _make_async_llm(max_num_queued_reqs=1, num_unfinished=0)
    _attach_prometheus_logger(llm)
    llm.check_admission()
    output_processor: MagicMock = llm.output_processor
    output_processor.get_num_unfinished_requests.return_value = 1
    with pytest.raises(QueueOverflowError):
        llm.check_admission()
    assert _admission_rejection_count("max_num_queued_reqs") == 1
    assert _admission_rejection_count("max_num_queued_tokens") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("first_n", [1, 3])
async def test_parallel_admission_is_all_or_nothing(first_n: int):
    """An n>1 request reserves either all its capacity or none of it."""

    async def add_request_async(_request):
        await asyncio.sleep(0)

    llm = _make_request_test_llm(3, add_request_async)
    second_n = 3 if first_n == 1 else 1
    first = _make_engine_request("first", first_n)
    second = _make_engine_request("second", second_n)
    results = await asyncio.gather(
        llm.add_request("first", first, first.params),
        llm.add_request("second", second, second.params),
        return_exceptions=True,
    )

    assert not isinstance(results[0], Exception)
    assert isinstance(results[1], QueueOverflowError)
    assert llm.output_processor.get_num_unfinished_requests() == first_n
    assert llm.engine_core.add_request_async.await_count == first_n


@pytest.mark.asyncio
async def test_parallel_admission_cancellation_cleans_up_all_children():
    """Cancellation during core submission must release every reserved slot."""
    first_submission_started = asyncio.Event()

    async def add_request_async(_request):
        first_submission_started.set()
        await asyncio.Event().wait()

    llm = _make_request_test_llm(3, add_request_async)
    request = _make_engine_request("parallel", 3)
    output = llm.generate(request, request.params, request.request_id)
    generate_task = asyncio.create_task(anext(output))

    await first_submission_started.wait()
    generate_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await generate_task

    assert llm.engine_core.add_request_async.await_count == 1
    assert llm.output_processor.get_num_unfinished_requests() == 0
    aborted_ids = llm.engine_core.abort_requests_async.await_args.args[0]
    assert set(aborted_ids) == {"0_parallel", "1_parallel", "2_parallel"}


# ---------------------------------------------------------------------------
# create_error_response integration
# ---------------------------------------------------------------------------


def test_queue_overflow_maps_to_503():
    resp = create_error_response(QueueOverflowError())
    assert resp.error.code == HTTPStatus.SERVICE_UNAVAILABLE.value
    assert resp.error.type == HTTPStatus.SERVICE_UNAVAILABLE.phrase
    assert resp.error.param is None
    msg = resp.error.message.lower()
    assert "busy" in msg or "try again" in msg


def test_max_queued_tokens_maps_to_503():
    resp = create_error_response(MaxQueuedTokensError())
    assert resp.error.code == HTTPStatus.SERVICE_UNAVAILABLE.value
    assert resp.error.type == HTTPStatus.SERVICE_UNAVAILABLE.phrase
    msg = resp.error.message.lower()
    assert "backlog" in msg or "try again" in msg


def test_custom_graceful_error_maps_to_its_status():
    err = GracefulHTTPError("custom", HTTPStatus.SERVICE_UNAVAILABLE)
    resp = create_error_response(err)
    assert resp.error.code == HTTPStatus.SERVICE_UNAVAILABLE.value
    assert resp.error.type == HTTPStatus.SERVICE_UNAVAILABLE.phrase


# ---------------------------------------------------------------------------
# SchedulerConfig field defaults
# ---------------------------------------------------------------------------


def test_scheduler_config_defaults_are_none():
    config = _make_scheduler_config()
    assert config.max_num_queued_reqs is None
    assert config.max_num_queued_tokens is None


def test_scheduler_config_accepts_explicit_values():
    config = _make_scheduler_config(
        max_num_queued_reqs=100,
        max_num_queued_tokens=32000,
    )
    assert config.max_num_queued_reqs == 100
    assert config.max_num_queued_tokens == 32000


@pytest.mark.parametrize("field", ["max_num_queued_reqs", "max_num_queued_tokens"])
def test_scheduler_config_rejects_negative(field):
    with pytest.raises(ValidationError):
        _make_scheduler_config(**{field: -1})


# ---------------------------------------------------------------------------
# human_readable_int for CLI notation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("32k", 32_000),
        ("1k", 1_000),
        ("1K", 1_024),
        ("1m", 1_000_000),
        ("1M", 1_048_576),
        ("100", 100),
        ("2.5k", 2_500),
        ("0", 0),
    ],
)
def test_human_readable_int_parses_notation(input_str: str, expected: int):
    assert human_readable_int(input_str) == expected


@pytest.mark.parametrize("invalid", ["abc", "1x", "", "k", "1.5K"])
def test_human_readable_int_rejects_invalid(invalid: str):
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        human_readable_int(invalid)


# ---------------------------------------------------------------------------
# Eager admission for streaming responses (generate_admitted)
# ---------------------------------------------------------------------------


async def _noop_add_request_async(_request):
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_generate_admitted_cancellation_releases_admission_slot():
    """Cancelling eager submission must roll back local request registration."""
    submission_started = asyncio.Event()

    async def add_request_async(_request):
        submission_started.set()
        await asyncio.Event().wait()

    llm = _make_request_test_llm(1, add_request_async)
    request = _make_engine_request("single", 1)
    submission = asyncio.create_task(
        llm.generate_admitted(request, request.params, request.request_id)
    )

    await submission_started.wait()
    submission.cancel()
    with pytest.raises(asyncio.CancelledError):
        await submission

    assert llm.output_processor.get_num_unfinished_requests() == 0
    llm.engine_core.abort_requests_async.assert_awaited_once_with(["single"])
    llm.check_admission(request_id="next")


@pytest.mark.asyncio
async def test_generate_admitted_rollback_failure_preserves_submission_error():
    """Rollback cleanup must not replace the original submission failure."""

    async def add_request_async(_request):
        raise RuntimeError("submission failed")

    llm = _make_request_test_llm(1, add_request_async)
    llm.engine_core.abort_requests_async = AsyncMock(
        side_effect=RuntimeError("abort failed")
    )
    request = _make_engine_request("single", 1)

    with pytest.raises(EngineGenerateError) as exc_info:
        await llm.generate_admitted(request, request.params, request.request_id)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "submission failed"
    assert llm.output_processor.get_num_unfinished_requests() == 0
    llm.engine_core.abort_requests_async.assert_awaited_once_with(["single"])


@pytest.mark.asyncio
async def test_generate_admitted_parallel_rollback_preserves_submission_error():
    """Parallel-sampling rollback must also preserve the original failure."""

    async def add_request_async(_request):
        raise RuntimeError("parallel submission failed")

    llm = _make_request_test_llm(2, add_request_async)
    llm.engine_core.abort_requests_async = AsyncMock(
        side_effect=RuntimeError("parallel abort failed")
    )
    request = _make_engine_request("parallel", 2)

    with pytest.raises(EngineGenerateError) as exc_info:
        await llm.generate_admitted(request, request.params, request.request_id)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "parallel submission failed"
    assert llm.output_processor.get_num_unfinished_requests() == 0
    aborted_ids = llm.engine_core.abort_requests_async.await_args.args[0]
    assert set(aborted_ids) == {"0_parallel", "1_parallel"}


@pytest.mark.asyncio
async def test_generate_admitted_wraps_unexpected_submission_error():
    """Unexpected submission failures keep generate()'s server-error class."""

    async def add_request_async(_request):
        raise TypeError("request serialization failed")

    llm = _make_request_test_llm(1, add_request_async)
    request = _make_engine_request("single", 1)

    with pytest.raises(EngineGenerateError) as exc_info:
        await llm.generate_admitted(request, request.params, request.request_id)
    assert isinstance(exc_info.value.__cause__, TypeError)


@pytest.mark.asyncio
async def test_admitted_stream_close_cancellation_preserves_abort():
    """Response teardown must not cancel the engine-side abort send."""
    llm = _make_request_test_llm(1, _noop_add_request_async)
    request = _make_engine_request("interrupted-close", 1)
    stream = await llm.generate_admitted(request, request.params, request.request_id)
    abort_started = asyncio.Event()
    abort_release = asyncio.Event()

    async def abort_requests(_request_ids):
        abort_started.set()
        await abort_release.wait()

    llm.engine_core.abort_requests_async = AsyncMock(side_effect=abort_requests)
    close_task = asyncio.create_task(stream.aclose())
    await abort_started.wait()
    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert not stream._finished
    second_close = asyncio.create_task(stream.aclose())
    await asyncio.sleep(0)
    llm.engine_core.abort_requests_async.assert_awaited_once()

    abort_release.set()
    await second_close
    assert stream._finished
    assert llm.output_processor.get_num_unfinished_requests() == 0
    llm.engine_core.abort_requests_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_admitted_rejects_before_stream_unlike_lazy_generate():
    """Overload must surface when awaiting admission, not on first iteration."""
    llm = _make_request_test_llm(1, _noop_add_request_async)
    first = _make_engine_request("first", 1)
    stream = await llm.generate_admitted(first, first.params, first.request_id)
    assert llm.output_processor.get_num_unfinished_requests() == 1

    # Lazy generate() defers admission until iteration (after HTTP 200).
    second = _make_engine_request("second", 1)
    lazy = llm.generate(second, second.params, second.request_id)
    with pytest.raises(QueueOverflowError):
        await anext(lazy)

    # generate_admitted() raises before any stream object is returned.
    third = _make_engine_request("third", 1)
    with pytest.raises(QueueOverflowError):
        await llm.generate_admitted(third, third.params, third.request_id)
    assert llm.output_processor.get_num_unfinished_requests() == 1

    await stream.aclose()
    assert llm.output_processor.get_num_unfinished_requests() == 0
    aborted_ids = llm.engine_core.abort_requests_async.await_args.args[0]
    assert list(aborted_ids) == ["first"]


@pytest.mark.asyncio
async def test_generate_admitted_unconsumed_stream_is_aborted_on_gc():
    """An admitted stream dropped before iteration must not leak the request."""
    import gc

    llm = _make_request_test_llm(1, _noop_add_request_async)
    request = _make_engine_request("dropped", 1)
    stream = await llm.generate_admitted(request, request.params, request.request_id)
    assert llm.output_processor.get_num_unfinished_requests() == 1

    del stream
    gc.collect()
    for _ in range(3):
        await asyncio.sleep(0)

    assert llm.output_processor.get_num_unfinished_requests() == 0
    aborted_ids = llm.engine_core.abort_requests_async.await_args.args[0]
    assert list(aborted_ids) == ["dropped"]


@pytest.mark.asyncio
async def test_generate_admitted_stream_finishes_without_abort():
    llm = _make_request_test_llm(1, _noop_add_request_async)
    request = _make_engine_request("done", 1)
    stream = await llm.generate_admitted(request, request.params, request.request_id)
    stream._queue.put(STREAM_FINISHED)
    assert [out async for out in stream] == []
    # Natural completion detaches the GC finalizer and does not abort.
    await stream.aclose()
    llm.engine_core.abort_requests_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_admitted_reclaims_request_rejected_after_admission():
    """An engine rejection delivered via the queue aborts like generate()."""
    llm = _make_request_test_llm(1, _noop_add_request_async)
    request = _make_engine_request("rejected", 1)
    stream = await llm.generate_admitted(request, request.params, request.request_id)

    # E.g. a paused engine rejects the request on arrival; nothing was run.
    stream._queue.put(QueueOverflowError())
    with pytest.raises(QueueOverflowError):
        await anext(stream)

    assert llm.output_processor.get_num_unfinished_requests() == 0
    aborted_ids = llm.engine_core.abort_requests_async.await_args.args[0]
    assert list(aborted_ids) == ["rejected"]
    # The abort already released the request; a later close is a no-op.
    await stream.aclose()
    llm.engine_core.abort_requests_async.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_serving_start_generate_admits_only_streaming_requests(stream: bool):
    from vllm.entrypoints.generate.base.serving import GenerateBaseServing

    engine = MagicMock()
    engine.generate_admitted = AsyncMock(return_value="admitted-stream")
    engine.generate = MagicMock(return_value="lazy-generator")
    serving = SimpleNamespace(engine_client=engine)

    result = await GenerateBaseServing._start_generate(
        serving, stream, "prompt", SamplingParams(), "req-1", priority=3
    )

    if stream:
        assert result == "admitted-stream"
        engine.generate_admitted.assert_awaited_once()
        engine.generate.assert_not_called()
    else:
        assert result == "lazy-generator"
        engine.generate.assert_called_once()
        engine.generate_admitted.assert_not_awaited()


@pytest.mark.asyncio
async def test_serving_close_generators_releases_admitted_streams():
    from vllm.entrypoints.generate.base.serving import GenerateBaseServing

    closed: list[str] = []

    class Closable:
        def __init__(self, name):
            self.name = name

        async def aclose(self):
            closed.append(self.name)

    class Broken:
        async def aclose(self):
            raise RuntimeError("boom")

    await GenerateBaseServing._close_generators(
        [Closable("a"), Broken(), object(), Closable("b")]
    )
    assert closed == ["a", "b"]
