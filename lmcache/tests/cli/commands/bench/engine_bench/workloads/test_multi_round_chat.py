# SPDX-License-Identifier: Apache-2.0
"""Tests for multi-round-chat workload config and workload."""

# Standard
from unittest.mock import AsyncMock, MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.cli.commands.bench.engine_bench.workloads.multi_round_chat import (
    MultiRoundChatConfig,
    MultiRoundChatWorkload,
    Session,
)

# ---------------------------------------------------------------------------
# MultiRoundChatConfig — direct construction
# ---------------------------------------------------------------------------


class TestMultiRoundChatConfig:
    def test_defaults(self) -> None:
        cfg = MultiRoundChatConfig()
        assert cfg.shared_prompt_length == 2000
        assert cfg.chat_history_length == 10000
        assert cfg.user_input_length == 50
        assert cfg.output_length == 200
        assert cfg.qps == 1.0
        assert cfg.duration == 60.0
        assert cfg.num_concurrent_users == 1

    def test_custom_values(self) -> None:
        cfg = MultiRoundChatConfig(
            shared_prompt_length=500,
            chat_history_length=5000,
            user_input_length=100,
            output_length=300,
            qps=2.0,
            duration=120.0,
            num_concurrent_users=10,
        )
        assert cfg.shared_prompt_length == 500
        assert cfg.chat_history_length == 5000
        assert cfg.user_input_length == 100
        assert cfg.output_length == 300
        assert cfg.qps == 2.0
        assert cfg.duration == 120.0
        assert cfg.num_concurrent_users == 10

    def test_invalid_shared_prompt_length(self) -> None:
        with pytest.raises(ValueError, match="shared_prompt_length must be positive"):
            MultiRoundChatConfig(shared_prompt_length=0)

    def test_invalid_chat_history_length(self) -> None:
        with pytest.raises(ValueError, match="chat_history_length must be positive"):
            MultiRoundChatConfig(chat_history_length=-1)

    def test_invalid_user_input_length(self) -> None:
        with pytest.raises(ValueError, match="user_input_length must be >= 1"):
            MultiRoundChatConfig(user_input_length=0)

    def test_invalid_output_length(self) -> None:
        with pytest.raises(ValueError, match="output_length must be >= 1"):
            MultiRoundChatConfig(output_length=0)

    def test_invalid_qps(self) -> None:
        with pytest.raises(ValueError, match="qps must be positive"):
            MultiRoundChatConfig(qps=0)

    def test_invalid_duration(self) -> None:
        with pytest.raises(ValueError, match="duration must be positive"):
            MultiRoundChatConfig(duration=-1)

    def test_invalid_num_concurrent_users(self) -> None:
        with pytest.raises(ValueError, match="num_concurrent_users must be >= 1"):
            MultiRoundChatConfig(num_concurrent_users=0)


# ---------------------------------------------------------------------------
# MultiRoundChatConfig.resolve
# ---------------------------------------------------------------------------


class TestMultiRoundChatConfigResolve:
    def test_resolve_basic(self) -> None:
        cfg = MultiRoundChatConfig.resolve(
            kv_cache_volume_gb=100.0,
            tokens_per_gb_kvcache=6000,
            shared_prompt_length=2000,
            chat_history_length=10000,
        )
        # 100 * 6000 / (2000 + 10000) = 600000 / 12000 = 50
        assert cfg.num_concurrent_users == 50

    def test_resolve_fractional_floors(self) -> None:
        cfg = MultiRoundChatConfig.resolve(
            kv_cache_volume_gb=1.0,
            tokens_per_gb_kvcache=10000,
            shared_prompt_length=2000,
            chat_history_length=5000,
        )
        # 10000 / 7000 = 1.428 → int(1.428) = 1
        assert cfg.num_concurrent_users == 1

    def test_resolve_minimum_one(self) -> None:
        cfg = MultiRoundChatConfig.resolve(
            kv_cache_volume_gb=0.001,
            tokens_per_gb_kvcache=1,
            shared_prompt_length=2000,
            chat_history_length=10000,
        )
        assert cfg.num_concurrent_users == 1

    def test_resolve_passes_all_fields(self) -> None:
        cfg = MultiRoundChatConfig.resolve(
            kv_cache_volume_gb=10.0,
            tokens_per_gb_kvcache=10000,
            shared_prompt_length=500,
            chat_history_length=5000,
            user_input_length=100,
            output_length=300,
            qps=5.0,
            duration=120.0,
        )
        assert cfg.shared_prompt_length == 500
        assert cfg.chat_history_length == 5000
        assert cfg.user_input_length == 100
        assert cfg.output_length == 300
        assert cfg.qps == 5.0
        assert cfg.duration == 120.0
        # 10 * 10000 / (500 + 5000) = 100000 / 5500 = 18
        assert cfg.num_concurrent_users == 18


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class TestSession:
    def test_construction(self) -> None:
        s = Session(
            session_id=0,
            system_prompt="You are helpful.",
            history_text="Some history.",
        )
        assert s.session_id == 0
        assert s.exchanges == []
        assert s.in_flight is False

    def test_build_messages_no_history(self) -> None:
        s = Session(
            session_id=0,
            system_prompt="System.",
            history_text="",
        )
        msgs = s.build_messages("Hello")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "System."}
        assert msgs[1] == {"role": "user", "content": "Hello"}

    def test_build_messages_with_history(self) -> None:
        s = Session(
            session_id=0,
            system_prompt="System.",
            history_text="Context text.",
        )
        msgs = s.build_messages("Hello")
        assert len(msgs) == 4
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Context text."
        assert msgs[2]["role"] == "assistant"
        assert "Understood" in msgs[2]["content"]
        assert msgs[3] == {"role": "user", "content": "Hello"}

    def test_build_messages_with_exchanges(self) -> None:
        s = Session(
            session_id=0,
            system_prompt="System.",
            history_text="",
            exchanges=[("q1", "a1"), ("q2", "a2")],
        )
        msgs = s.build_messages("q3")
        # system + 2 exchanges (4 msgs) + new query = 6 (no history pair)
        assert len(msgs) == 6
        assert msgs[1] == {"role": "user", "content": "q1"}
        assert msgs[2] == {"role": "assistant", "content": "a1"}
        assert msgs[3] == {"role": "user", "content": "q2"}
        assert msgs[4] == {"role": "assistant", "content": "a2"}
        assert msgs[5] == {"role": "user", "content": "q3"}

    def test_record_answer(self) -> None:
        s = Session(
            session_id=0,
            system_prompt="System.",
            history_text="",
        )
        s.in_flight = True
        s.record_answer("q1", "a1")
        assert s.exchanges == [("q1", "a1")]
        assert s.in_flight is False

    def test_record_multiple_answers(self) -> None:
        s = Session(
            session_id=0,
            system_prompt="System.",
            history_text="",
        )
        s.record_answer("q1", "a1")
        s.record_answer("q2", "a2")
        assert len(s.exchanges) == 2
        assert s.exchanges[1] == ("q2", "a2")


# ---------------------------------------------------------------------------
# MultiRoundChatWorkload — helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> MultiRoundChatConfig:
    defaults = dict(
        shared_prompt_length=100,
        chat_history_length=200,
        user_input_length=10,
        output_length=20,
        qps=10.0,
        duration=5.0,
        num_concurrent_users=3,
    )
    defaults.update(overrides)
    return MultiRoundChatConfig(**defaults)  # type: ignore[arg-type]


def _make_mock_result(request_id: str = "req_0") -> RequestResult:
    now = time.time()
    return RequestResult(
        request_id=request_id,
        successful=True,
        ttft=0.1,
        request_latency=0.5,
        num_input_tokens=100,
        num_output_tokens=10,
        decode_speed=25.0,
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
    config: MultiRoundChatConfig | None = None,
    seed: int = 42,
) -> tuple[MultiRoundChatWorkload, MagicMock, MagicMock, MagicMock]:
    if config is None:
        config = _make_config()
    sender = _make_mock_sender()
    collector = MagicMock()
    monitor = MagicMock()
    workload = MultiRoundChatWorkload(
        config,
        sender,
        collector,
        monitor,
        seed=seed,
    )
    return workload, sender, collector, monitor


# ---------------------------------------------------------------------------
# MultiRoundChatWorkload — session creation
# ---------------------------------------------------------------------------


class TestMultiRoundChatSessionCreation:
    def test_correct_count(self) -> None:
        w, *_ = _make_workload(_make_config(num_concurrent_users=5))
        assert len(w._sessions) == 5

    def test_session_prompts(self) -> None:
        w, *_ = _make_workload(_make_config(num_concurrent_users=2))
        assert "Session 0" in w._sessions[0].system_prompt
        assert "Session 1" in w._sessions[1].system_prompt

    def test_session_history(self) -> None:
        w, *_ = _make_workload(_make_config(num_concurrent_users=2))
        assert "[Session 0 history]" in w._sessions[0].history_text
        assert "[Session 1 history]" in w._sessions[1].history_text

    def test_sessions_start_ready(self) -> None:
        w, *_ = _make_workload()
        for s in w._sessions:
            assert s.in_flight is False
            assert s.exchanges == []


# ---------------------------------------------------------------------------
# MultiRoundChatWorkload — query generation
# ---------------------------------------------------------------------------


class TestMultiRoundChatQueryGeneration:
    def test_query_length(self) -> None:
        w, *_ = _make_workload(_make_config(user_input_length=10))
        query = w._generate_query()
        assert query.count("tell") == 10

    def test_query_single_word(self) -> None:
        w, *_ = _make_workload(_make_config(user_input_length=1))
        query = w._generate_query()
        assert query == "tell"


# ---------------------------------------------------------------------------
# MultiRoundChatWorkload — warmup (async)
# ---------------------------------------------------------------------------


class TestMultiRoundChatWarmup:
    @pytest.mark.asyncio
    async def test_warmup_sends_all_sessions(self) -> None:
        cfg = _make_config(num_concurrent_users=3)
        w, sender, _, monitor = _make_workload(cfg)

        await w.warmup()

        assert sender.send_warmup_request.call_count == 3
        for i, call in enumerate(sender.send_warmup_request.call_args_list):
            request_id = call[0][0]
            messages = call[0][1]
            assert request_id == f"warmup_s{i}"
            assert messages[0]["role"] == "system"
            assert messages[-1]["role"] == "user"
            assert messages[-1]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_warmup_logs_progress(self) -> None:
        cfg = _make_config(num_concurrent_users=2)
        w, _, _, monitor = _make_workload(cfg)
        await w.warmup()
        # 2 per-session messages + 1 completion message
        assert monitor.log_message.call_count == 3
        assert monitor.on_request_sent.call_count == 2


# ---------------------------------------------------------------------------
# MultiRoundChatWorkload — step (async)
# ---------------------------------------------------------------------------


class TestMultiRoundChatStep:
    @pytest.mark.asyncio
    async def test_step_dispatches_and_returns_next_time(self) -> None:
        cfg = _make_config(
            num_concurrent_users=2,
            qps=10.0,
            duration=5.0,
        )
        w, sender, _, _ = _make_workload(cfg)

        result = await w.step(0.0)
        # Next wakeup at 1 * 0.1 = 0.1
        assert abs(result - 0.1) < 1e-9
        assert w._global_index == 1
        # Wait for task
        if w._pending_tasks:
            # Standard
            import asyncio

            await asyncio.gather(*w._pending_tasks)
        assert sender.send_request.call_count == 1

    @pytest.mark.asyncio
    async def test_step_busy_session_returns_small_sleep(self) -> None:
        cfg = _make_config(
            num_concurrent_users=1,
            qps=10.0,
            duration=5.0,
        )
        w, _, _, _ = _make_workload(cfg)

        # Mark session as in-flight
        w._sessions[0].in_flight = True

        result = await w.step(0.5)
        # Should return current time + 0.01
        assert abs(result - 0.51) < 1e-9
        # global_index should NOT have incremented
        assert w._global_index == 0

    @pytest.mark.asyncio
    async def test_step_returns_negative_when_done(self) -> None:
        cfg = _make_config(duration=1.0)
        w, _, _, _ = _make_workload(cfg)

        # Time exceeds duration, no pending tasks
        result = await w.step(2.0)
        assert result == -1.0

    @pytest.mark.asyncio
    async def test_step_waits_for_pending_tasks_after_duration(
        self,
    ) -> None:
        cfg = _make_config(
            num_concurrent_users=1,
            qps=100.0,
            duration=0.01,
        )
        w, _, _, _ = _make_workload(cfg)

        # Dispatch one request at time 0
        await w.step(0.0)
        assert len(w._pending_tasks) > 0

        # Standard
        import asyncio

        # Let task finish
        await asyncio.gather(*w._pending_tasks)

        # Now past duration with no pending → done
        result = await w.step(1.0)
        assert result == -1.0


# ---------------------------------------------------------------------------
# MultiRoundChatWorkload — on_request_finished (stateful)
# ---------------------------------------------------------------------------


class TestMultiRoundChatOnRequestFinished:
    def test_records_answer_in_session(self) -> None:
        w, *_ = _make_workload()
        # Simulate a pending request
        w._pending_info["s0_r0"] = (0, "What is 2+2?")
        w._sessions[0].in_flight = True

        w.on_request_finished("s0_r0", "4")

        assert w._sessions[0].exchanges == [("What is 2+2?", "4")]
        assert w._sessions[0].in_flight is False

    def test_ignores_unknown_request(self) -> None:
        w, *_ = _make_workload()
        # Should not raise
        w.on_request_finished("unknown_id", "text")

    def test_warmup_request_ignored(self) -> None:
        w, *_ = _make_workload()
        # Warmup requests are not in _pending_info
        w.on_request_finished("warmup_s0", "text")
        assert w._sessions[0].exchanges == []


# ---------------------------------------------------------------------------
# MultiRoundChatWorkload — full run
# ---------------------------------------------------------------------------


class TestMultiRoundChatFullRun:
    def test_full_run(self) -> None:
        cfg = _make_config(
            num_concurrent_users=2,
            qps=100.0,  # fast dispatch
            duration=0.1,  # short duration
        )
        w, sender, collector, monitor = _make_workload(cfg)

        w.run()

        # Warmup: 2 sessions
        assert sender.send_warmup_request.call_count == 2
        # Benchmark: at least some requests dispatched
        assert sender.send_request.call_count > 0
        # Stats reset called between warmup and benchmark
        collector.reset.assert_called_once()
