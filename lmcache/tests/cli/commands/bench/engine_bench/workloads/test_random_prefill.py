# SPDX-License-Identifier: Apache-2.0
"""Tests for random-prefill workload config and workload."""

# Standard
from unittest.mock import AsyncMock, MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.cli.commands.bench.engine_bench.workloads.random_prefill import (
    RandomPrefillConfig,
    RandomPrefillWorkload,
)

# ---------------------------------------------------------------------------
# RandomPrefillConfig — direct construction
# ---------------------------------------------------------------------------


class TestRandomPrefillConfig:
    def test_defaults(self) -> None:
        cfg = RandomPrefillConfig()
        assert cfg.request_length == 10000
        assert cfg.num_requests == 50

    def test_custom_values(self) -> None:
        cfg = RandomPrefillConfig(request_length=5000, num_requests=100)
        assert cfg.request_length == 5000
        assert cfg.num_requests == 100

    def test_invalid_request_length(self) -> None:
        with pytest.raises(ValueError, match="request_length must be positive"):
            RandomPrefillConfig(request_length=0)

    def test_invalid_request_length_negative(self) -> None:
        with pytest.raises(ValueError, match="request_length must be positive"):
            RandomPrefillConfig(request_length=-1)

    def test_invalid_num_requests(self) -> None:
        with pytest.raises(ValueError, match="num_requests must be >= 1"):
            RandomPrefillConfig(num_requests=0)


# ---------------------------------------------------------------------------
# RandomPrefillConfig.resolve
# ---------------------------------------------------------------------------


class TestRandomPrefillConfigResolve:
    def test_resolve_defaults(self) -> None:
        cfg = RandomPrefillConfig.resolve()
        assert cfg.request_length == 10000
        assert cfg.num_requests == 50

    def test_resolve_custom(self) -> None:
        cfg = RandomPrefillConfig.resolve(
            request_length=5000,
            num_requests=20,
        )
        assert cfg.request_length == 5000
        assert cfg.num_requests == 20


# ---------------------------------------------------------------------------
# RandomPrefillWorkload — helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> RandomPrefillConfig:
    defaults = dict(
        request_length=100,
        num_requests=5,
    )
    defaults.update(overrides)
    return RandomPrefillConfig(**defaults)  # type: ignore[arg-type]


def _make_mock_result(request_id: str = "req_0") -> RequestResult:
    now = time.time()
    return RequestResult(
        request_id=request_id,
        successful=True,
        ttft=0.1,
        request_latency=0.5,
        num_input_tokens=100,
        num_output_tokens=1,
        decode_speed=10.0,
        submit_time=now,
        first_token_time=now + 0.1,
        finish_time=now + 0.5,
        error="",
    )


def _make_mock_sender() -> MagicMock:
    sender = MagicMock()
    sender.send_request = AsyncMock(return_value=_make_mock_result())
    sender.send_warmup_request = AsyncMock(return_value=_make_mock_result())
    sender.close = AsyncMock(return_value=None)
    return sender


def _make_workload(
    config: RandomPrefillConfig | None = None,
    seed: int = 42,
) -> tuple[RandomPrefillWorkload, MagicMock, MagicMock, MagicMock]:
    if config is None:
        config = _make_config()
    sender = _make_mock_sender()
    collector = MagicMock()
    monitor = MagicMock()
    workload = RandomPrefillWorkload(
        config,
        sender,
        collector,
        monitor,
        seed=seed,
    )
    return workload, sender, collector, monitor


# ---------------------------------------------------------------------------
# RandomPrefillWorkload — prompt generation
# ---------------------------------------------------------------------------


class TestRandomPrefillPromptGeneration:
    def test_correct_count(self) -> None:
        w, *_ = _make_workload(_make_config(num_requests=10))
        assert len(w._prompts) == 10

    def test_prompt_format(self) -> None:
        w, *_ = _make_workload(_make_config(num_requests=3))
        assert w._prompts[0].startswith("Request 0: ")
        assert w._prompts[1].startswith("Request 1: ")
        assert w._prompts[2].startswith("Request 2: ")

    def test_prompt_fill_tokens(self) -> None:
        w, *_ = _make_workload(_make_config(request_length=100))
        hi_count = w._prompts[0].count("hi")
        assert hi_count == 90  # 100 - 10


# ---------------------------------------------------------------------------
# RandomPrefillWorkload — warmup
# ---------------------------------------------------------------------------


class TestRandomPrefillWarmup:
    @pytest.mark.asyncio
    async def test_warmup_is_noop(self) -> None:
        w, sender, _, _ = _make_workload()
        await w.warmup()
        assert sender.send_warmup_request.call_count == 0
        assert sender.send_request.call_count == 0


# ---------------------------------------------------------------------------
# RandomPrefillWorkload — step
# ---------------------------------------------------------------------------


class TestRandomPrefillStep:
    @pytest.mark.asyncio
    async def test_first_step_dispatches_all(self) -> None:
        cfg = _make_config(num_requests=5)
        w, sender, _, monitor = _make_workload(cfg)

        result = await w.step(0.0)
        assert result == 0.0
        assert w._dispatched is True
        assert len(w._pending_tasks) == 5
        assert monitor.on_request_sent.call_count == 5

        # Wait for tasks
        # Standard
        import asyncio

        await asyncio.gather(*w._pending_tasks)
        assert sender.send_request.call_count == 5

        # Verify max_tokens=1
        for call in sender.send_request.call_args_list:
            assert call[1]["max_tokens"] == 1 or call[0][2] == 1

    @pytest.mark.asyncio
    async def test_second_step_does_not_redispatch(self) -> None:
        cfg = _make_config(num_requests=3)
        w, sender, _, _ = _make_workload(cfg)

        await w.step(0.0)
        # Standard
        import asyncio

        await asyncio.gather(*w._pending_tasks)

        # Second step — no pending, should return -1
        result = await w.step(0.1)
        assert result == -1.0
        # Still only 3 calls
        assert sender.send_request.call_count == 3

    @pytest.mark.asyncio
    async def test_returns_negative_when_all_done(self) -> None:
        cfg = _make_config(num_requests=2)
        w, _, _, _ = _make_workload(cfg)

        await w.step(0.0)
        # Standard
        import asyncio

        await asyncio.gather(*w._pending_tasks)

        result = await w.step(0.5)
        assert result == -1.0

    @pytest.mark.asyncio
    async def test_max_tokens_is_one(self) -> None:
        cfg = _make_config(num_requests=1)
        w, sender, _, _ = _make_workload(cfg)

        await w.step(0.0)
        # Standard
        import asyncio

        await asyncio.gather(*w._pending_tasks)

        call_kwargs = sender.send_request.call_args[1]
        assert call_kwargs["max_tokens"] == 1

    def test_on_request_finished_noop(self) -> None:
        w, *_ = _make_workload()
        w.on_request_finished("prefill_0", "text")  # should not raise


# ---------------------------------------------------------------------------
# RandomPrefillWorkload — full run
# ---------------------------------------------------------------------------


class TestRandomPrefillFullRun:
    def test_full_run(self) -> None:
        cfg = _make_config(num_requests=5)
        w, sender, collector, _ = _make_workload(cfg)

        w.run()

        # No warmup requests
        assert sender.send_warmup_request.call_count == 0
        # 5 benchmark requests
        assert sender.send_request.call_count == 5
        # Stats reset called
        collector.reset.assert_called_once()
