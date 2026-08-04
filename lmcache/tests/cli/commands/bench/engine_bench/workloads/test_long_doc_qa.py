# SPDX-License-Identifier: Apache-2.0
"""Tests for long-doc-qa workload config and workload generator."""

# Standard
from unittest.mock import AsyncMock, MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.cli.commands.bench.engine_bench.workloads.long_doc_qa import (
    LongDocQAConfig,
    LongDocQAWorkload,
)

# ---------------------------------------------------------------------------
# LongDocQAConfig — direct construction
# ---------------------------------------------------------------------------


class TestLongDocQAConfig:
    def test_defaults(self) -> None:
        cfg = LongDocQAConfig()
        assert cfg.document_length == 10000
        assert cfg.query_per_document == 2
        assert cfg.num_documents == 1
        assert cfg.shuffle_policy == "random"
        assert cfg.num_inflight_requests == 3

    def test_custom_values(self) -> None:
        cfg = LongDocQAConfig(
            document_length=5000,
            query_per_document=4,
            num_documents=10,
            shuffle_policy="tile",
            num_inflight_requests=5,
        )
        assert cfg.document_length == 5000
        assert cfg.query_per_document == 4
        assert cfg.num_documents == 10
        assert cfg.shuffle_policy == "tile"
        assert cfg.num_inflight_requests == 5

    def test_invalid_document_length_zero(self) -> None:
        with pytest.raises(ValueError, match="document_length must be positive"):
            LongDocQAConfig(document_length=0)

    def test_invalid_document_length_negative(self) -> None:
        with pytest.raises(ValueError, match="document_length must be positive"):
            LongDocQAConfig(document_length=-1)

    def test_invalid_query_per_document(self) -> None:
        with pytest.raises(ValueError, match="query_per_document must be >= 1"):
            LongDocQAConfig(query_per_document=0)

    def test_invalid_num_inflight_requests(self) -> None:
        with pytest.raises(ValueError, match="num_inflight_requests must be >= 1"):
            LongDocQAConfig(num_inflight_requests=0)

    def test_invalid_shuffle_policy(self) -> None:
        with pytest.raises(
            ValueError, match="shuffle_policy must be 'random' or 'tile'"
        ):
            LongDocQAConfig(shuffle_policy="unknown")

    def test_shuffle_policy_tile(self) -> None:
        cfg = LongDocQAConfig(shuffle_policy="tile")
        assert cfg.shuffle_policy == "tile"


# ---------------------------------------------------------------------------
# LongDocQAConfig.resolve
# ---------------------------------------------------------------------------


class TestLongDocQAConfigResolve:
    def test_resolve_basic(self) -> None:
        cfg = LongDocQAConfig.resolve(
            kv_cache_volume_gb=100.0,
            tokens_per_gb_kvcache=50000,
            document_length=10000,
        )
        # 100 * 50000 / 10000 = 500
        assert cfg.num_documents == 500
        assert cfg.document_length == 10000

    def test_resolve_fractional_floors(self) -> None:
        cfg = LongDocQAConfig.resolve(
            kv_cache_volume_gb=1.0,
            tokens_per_gb_kvcache=10000,
            document_length=3000,
        )
        # int(10000 / 3000) = int(3.333...) = 3
        assert cfg.num_documents == 3

    def test_resolve_minimum_one(self) -> None:
        cfg = LongDocQAConfig.resolve(
            kv_cache_volume_gb=0.001,
            tokens_per_gb_kvcache=1,
            document_length=10000,
        )
        # int(0.001 * 1 / 10000) = 0, clamped to 1
        assert cfg.num_documents == 1

    def test_resolve_passes_all_fields(self) -> None:
        cfg = LongDocQAConfig.resolve(
            kv_cache_volume_gb=10.0,
            tokens_per_gb_kvcache=10000,
            document_length=5000,
            query_per_document=4,
            shuffle_policy="tile",
            num_inflight_requests=5,
        )
        assert cfg.document_length == 5000
        assert cfg.query_per_document == 4
        assert cfg.shuffle_policy == "tile"
        assert cfg.num_inflight_requests == 5
        assert cfg.num_documents == 20  # 10 * 10000 / 5000


# ---------------------------------------------------------------------------
# LongDocQAWorkload — helpers
# ---------------------------------------------------------------------------


def _make_workload_config(**overrides) -> LongDocQAConfig:
    defaults = dict(
        document_length=100,
        query_per_document=2,
        num_documents=3,
        shuffle_policy="tile",
        num_inflight_requests=2,
    )
    defaults.update(overrides)
    return LongDocQAConfig(**defaults)  # type: ignore[arg-type]


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
    config: LongDocQAConfig | None = None,
    seed: int = 42,
) -> tuple[LongDocQAWorkload, MagicMock, MagicMock, MagicMock]:
    if config is None:
        config = _make_workload_config()
    sender = _make_mock_sender()
    collector = MagicMock()
    monitor = MagicMock()
    workload = LongDocQAWorkload(
        config,
        sender,
        collector,
        monitor,
        seed=seed,
    )
    return workload, sender, collector, monitor


# ---------------------------------------------------------------------------
# LongDocQAWorkload — document generation
# ---------------------------------------------------------------------------


class TestLongDocQADocumentGeneration:
    def test_correct_count(self) -> None:
        w, *_ = _make_workload(_make_workload_config(num_documents=5))
        assert len(w._documents) == 5

    def test_document_format(self) -> None:
        w, *_ = _make_workload(_make_workload_config(num_documents=2))
        assert w._documents[0].startswith("Document 0: ")
        assert w._documents[1].startswith("Document 1: ")

    def test_document_fill_tokens(self) -> None:
        w, *_ = _make_workload(_make_workload_config(document_length=100))
        # 100 - 10 = 90 "hi" words
        hi_count = w._documents[0].count("hi")
        assert hi_count == 90


# ---------------------------------------------------------------------------
# LongDocQAWorkload — schedule building
# ---------------------------------------------------------------------------


class TestLongDocQAScheduleBuilding:
    def test_tile_schedule(self) -> None:
        w, *_ = _make_workload(
            _make_workload_config(
                num_documents=2,
                query_per_document=2,
                shuffle_policy="tile",
            )
        )
        # Query-major: all docs for q0, then all docs for q1
        assert w._schedule == [(0, 0), (1, 0), (0, 1), (1, 1)]

    def test_random_schedule_differs_from_tile(self) -> None:
        cfg_tile = _make_workload_config(
            num_documents=5,
            query_per_document=3,
            shuffle_policy="tile",
        )
        cfg_random = _make_workload_config(
            num_documents=5,
            query_per_document=3,
            shuffle_policy="random",
        )
        w_tile, *_ = _make_workload(cfg_tile)
        w_rand, *_ = _make_workload(cfg_random)
        assert set(w_tile._schedule) == set(w_rand._schedule)
        assert w_tile._schedule != w_rand._schedule

    def test_random_schedule_reproducible(self) -> None:
        cfg = _make_workload_config(
            num_documents=5,
            query_per_document=3,
            shuffle_policy="random",
        )
        w1, *_ = _make_workload(cfg, seed=42)
        w2, *_ = _make_workload(cfg, seed=42)
        assert w1._schedule == w2._schedule

    def test_schedule_length(self) -> None:
        w, *_ = _make_workload(
            _make_workload_config(
                num_documents=5,
                query_per_document=3,
            )
        )
        assert len(w._schedule) == 15


# ---------------------------------------------------------------------------
# LongDocQAWorkload — message building
# ---------------------------------------------------------------------------


class TestLongDocQAMessageBuilding:
    def test_message_format(self) -> None:
        w, *_ = _make_workload()
        msgs = w._build_messages(0, 0)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"].startswith("Document 0: ")
        assert "Question 0: What is this document about?" in msgs[0]["content"]

    def test_question_cycling_with_index(self) -> None:
        w, *_ = _make_workload()
        msgs = w._build_messages(0, 4)
        content = msgs[0]["content"]
        # query_index=4 cycles to _QUESTIONS[0] but with unique prefix
        assert "Question 4: What is this document about?" in content

    def test_on_request_finished_noop(self) -> None:
        w, *_ = _make_workload()
        w.on_request_finished("some_id", "some_text")  # should not raise


# ---------------------------------------------------------------------------
# LongDocQAWorkload — warmup (async)
# ---------------------------------------------------------------------------


class TestLongDocQAWarmup:
    @pytest.mark.asyncio
    async def test_warmup_sends_all_documents(self) -> None:
        cfg = _make_workload_config(num_documents=3)
        w, sender, _, monitor = _make_workload(cfg)

        await w.warmup()

        assert sender.send_warmup_request.call_count == 3
        for i, call in enumerate(sender.send_warmup_request.call_args_list):
            request_id = call[0][0]
            messages = call[0][1]
            assert request_id == f"warmup_doc{i}"
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            # Warmup sends just the document — no question appended
            assert "Question" not in messages[0]["content"]
            assert messages[0]["content"].startswith(f"Document {i}: ")

    @pytest.mark.asyncio
    async def test_warmup_logs_progress(self) -> None:
        cfg = _make_workload_config(num_documents=2)
        w, _, _, monitor = _make_workload(cfg)
        await w.warmup()
        # 2 per-doc messages + 1 completion message
        assert monitor.log_message.call_count == 3
        assert monitor.on_request_sent.call_count == 2


# ---------------------------------------------------------------------------
# LongDocQAWorkload — step (async)
# ---------------------------------------------------------------------------


class TestLongDocQAStep:
    @pytest.mark.asyncio
    async def test_step_dispatches_and_returns_zero(self) -> None:
        cfg = _make_workload_config(
            num_documents=1,
            query_per_document=1,
        )
        w, sender, _, _ = _make_workload(cfg)

        result = await w.step(0.0)
        assert result == 0.0
        # Wait for the dispatched task to complete
        if w._pending_tasks:
            # Standard
            import asyncio

            await asyncio.gather(*w._pending_tasks)
        assert sender.send_request.call_count == 1

    @pytest.mark.asyncio
    async def test_step_returns_negative_when_done(self) -> None:
        cfg = _make_workload_config(
            num_documents=1,
            query_per_document=1,
        )
        w, _, _, _ = _make_workload(cfg)

        # Dispatch the single request
        await w.step(0.0)
        # Wait for task
        # Standard
        import asyncio

        if w._pending_tasks:
            await asyncio.gather(*w._pending_tasks)
        # Now schedule is exhausted and no pending tasks
        result = await w.step(0.1)
        assert result == -1.0


# ---------------------------------------------------------------------------
# LongDocQAWorkload — full run (async)
# ---------------------------------------------------------------------------


class TestLongDocQAFullRun:
    def test_full_run(self) -> None:
        cfg = _make_workload_config(
            num_documents=2,
            query_per_document=2,
            shuffle_policy="tile",
        )
        w, sender, collector, monitor = _make_workload(cfg)

        w.run()

        # 2 warmup requests (one per document)
        assert sender.send_warmup_request.call_count == 2
        # 4 benchmark requests (2 docs × 2 queries)
        assert sender.send_request.call_count == 4
        # Stats reset called between warmup and benchmark
        collector.reset.assert_called_once()
