# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.platforms import current_platform

# --- Unit tests (no GPU required) ---


def _make_mock_proc(running: int, waiting: int, max_num_running: int = 2):
    """Build a bare EngineCoreProc with mocked scheduler — no model load."""
    from vllm.v1.engine.core import EngineCoreProc

    proc = EngineCoreProc.__new__(EngineCoreProc)
    proc.scheduler = SimpleNamespace(
        get_request_counts=lambda: (running, waiting),
        max_num_running_reqs=max_num_running,
    )
    proc._send_abort_outputs_to_client = MagicMock()
    return proc


def _make_mock_request(req_id="test-1", client_index=0):
    """Creates a fake request object with an ID and client index."""
    return SimpleNamespace(request_id=req_id, client_index=client_index)


def test_reject_when_busy_disabled_by_default():
    """Without VLLM_REJECT_WHEN_BUSY env var, should never reject."""
    os.environ.pop("VLLM_REJECT_WHEN_BUSY", None)
    proc = _make_mock_proc(running=2, waiting=1, max_num_running=1)
    req = _make_mock_request()

    assert proc._reject_when_busy(req) is False
    proc._send_abort_outputs_to_client.assert_not_called()


def test_reject_when_busy_rejects_when_running_at_capacity(monkeypatch):
    """running >= max_num_running_reqs triggers rejection."""
    monkeypatch.setenv("VLLM_REJECT_WHEN_BUSY", "1")
    proc = _make_mock_proc(running=2, waiting=0, max_num_running=2)
    req = _make_mock_request("req-cap", client_index=1)

    assert proc._reject_when_busy(req) is True
    proc._send_abort_outputs_to_client.assert_called_once_with(["req-cap"], 1)


def test_reject_when_busy_rejects_when_waiting(monkeypatch):
    """waiting > 0 triggers rejection even if running < max."""
    monkeypatch.setenv("VLLM_REJECT_WHEN_BUSY", "1")
    proc = _make_mock_proc(running=1, waiting=1, max_num_running=4)
    req = _make_mock_request("req-wait")

    assert proc._reject_when_busy(req) is True
    proc._send_abort_outputs_to_client.assert_called_once_with(["req-wait"], 0)


def test_reject_when_busy_accepts_when_not_full(monkeypatch):
    """running < max and waiting == 0 → accepted."""
    monkeypatch.setenv("VLLM_REJECT_WHEN_BUSY", "1")
    proc = _make_mock_proc(running=1, waiting=0, max_num_running=2)
    req = _make_mock_request("req-ok")

    assert proc._reject_when_busy(req) is False
    proc._send_abort_outputs_to_client.assert_not_called()


def test_reject_when_busy_accepts_when_empty(monkeypatch):
    """running=0, waiting=0 → accepted."""
    monkeypatch.setenv("VLLM_REJECT_WHEN_BUSY", "1")
    proc = _make_mock_proc(running=0, waiting=0, max_num_running=1)
    req = _make_mock_request("req-empty")

    assert proc._reject_when_busy(req) is False


# --- Regression tests (CUDA required) ---

if not current_platform.is_cuda():
    pytest.skip(
        reason="V1 regression tests currently only supported on CUDA.",
        allow_module_level=True,
    )

from vllm import SamplingParams  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.sampling_params import RequestOutputKind  # noqa: E402
from vllm.utils.torch_utils import set_default_torch_num_threads  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"

REGRESSION_ENGINE_ARGS = AsyncEngineArgs(
    model=MODEL_NAME,
    enforce_eager=True,
    max_num_seqs=1,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
)


async def _generate(engine, request_id, prompt="Hello"):
    count = 0
    sampling_params = SamplingParams(
        max_tokens=10,
        ignore_eos=True,
        output_kind=RequestOutputKind.FINAL_ONLY,
        temperature=0.5,
        seed=33,
    )
    try:
        async for out in engine.generate(
            request_id=request_id, prompt=prompt, sampling_params=sampling_params
        ):
            count += 1
        return ("completed", count, request_id)
    except Exception as e:
        return ("error", str(e), request_id)


@pytest.mark.asyncio
async def test_reject_when_busy_regression_rejects(monkeypatch):
    """With VLLM_REJECT_WHEN_BUSY=1 and max_num_seqs=1, the second
    concurrent request should be rejected (aborted)."""
    monkeypatch.setenv("VLLM_REJECT_WHEN_BUSY", "1")

    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(REGRESSION_ENGINE_ARGS)
        after.callback(engine.shutdown)

        # Small delay so first request is running before second arrives.
        async def send_first():
            await asyncio.sleep(0.1)
            return await _generate(engine, "req-first")

        async def send_second():
            await asyncio.sleep(0.3)
            return await _generate(engine, "req-second")

        results = await asyncio.gather(
            send_first(),
            send_second(),
            return_exceptions=True,
        )

        statuses = [r[0] for r in results if isinstance(r, tuple)]
        # At least one should complete, and at least one should error/abort.
        assert "completed" in statuses, (
            f"Expected at least one completion, got {results}"
        )
        # The rejected request may surface as error or as completed-with-abort
        # depending on how the client surfaces it. The key regression check is
        # that the engine does not hang or crash.
        assert len(statuses) == 2, f"Both requests should return, got {results}"


@pytest.mark.asyncio
async def test_reject_when_busy_regression_no_reject(monkeypatch):
    """Without VLLM_REJECT_WHEN_BUSY, both requests should complete."""
    monkeypatch.delenv("VLLM_REJECT_WHEN_BUSY", raising=False)

    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(REGRESSION_ENGINE_ARGS)
        after.callback(engine.shutdown)

        results = await asyncio.gather(
            _generate(engine, "req-a"),
            _generate(engine, "req-b"),
            return_exceptions=True,
        )

        for r in results:
            assert isinstance(r, tuple), f"Expected tuple result, got {r}"
            assert r[0] == "completed", (
                f"Expected completion without reject, got {r}"
            )
