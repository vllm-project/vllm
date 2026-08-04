# SPDX-License-Identifier: Apache-2.0
"""Tests for prefix-suffix-tuner workload config and workload generator."""

# Standard
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.cli.commands.bench.engine_bench.workloads.prefix_suffix_tuner import (
    PrefixSuffixTunerConfig,
    PrefixSuffixTunerWorkload,
)

# ---------------------------------------------------------------------------
# PrefixSuffixTunerConfig — direct construction
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerConfig:
    def test_defaults(self) -> None:
        cfg = PrefixSuffixTunerConfig()
        assert cfg.context_length == 8000
        assert cfg.prefix_ratio == 0.8
        assert cfg.thrash == 20.0
        assert cfg.num_prefixes == 1
        assert cfg.prefix_tokens == 1
        assert cfg.suffix_tokens == 1
        assert cfg.breaker_tokens == 32

    def test_custom_values(self) -> None:
        cfg = PrefixSuffixTunerConfig(
            context_length=4000,
            prefix_ratio=0.5,
            thrash=50.0,
            num_prefixes=10,
            prefix_tokens=2000,
            suffix_tokens=1968,
            breaker_tokens=32,
        )
        assert cfg.context_length == 4000
        assert cfg.prefix_ratio == 0.5
        assert cfg.thrash == 50.0
        assert cfg.num_prefixes == 10

    def test_invalid_context_length(self) -> None:
        with pytest.raises(ValueError, match="context_length must be positive"):
            PrefixSuffixTunerConfig(context_length=0)

    def test_invalid_prefix_ratio_zero(self) -> None:
        with pytest.raises(ValueError, match=r"prefix_ratio must be in \(0.0, 1.0\)"):
            PrefixSuffixTunerConfig(prefix_ratio=0.0)

    def test_invalid_prefix_ratio_one(self) -> None:
        with pytest.raises(ValueError, match=r"prefix_ratio must be in \(0.0, 1.0\)"):
            PrefixSuffixTunerConfig(prefix_ratio=1.0)

    def test_invalid_prefix_ratio_negative(self) -> None:
        with pytest.raises(ValueError, match=r"prefix_ratio must be in \(0.0, 1.0\)"):
            PrefixSuffixTunerConfig(prefix_ratio=-0.5)

    def test_invalid_thrash_zero(self) -> None:
        with pytest.raises(ValueError, match=r"thrash \(GB\) must be positive"):
            PrefixSuffixTunerConfig(thrash=0.0)

    def test_invalid_thrash_negative(self) -> None:
        with pytest.raises(ValueError, match=r"thrash \(GB\) must be positive"):
            PrefixSuffixTunerConfig(thrash=-1.0)

    def test_thrash_small_positive_is_valid(self) -> None:
        # Sub-1-GB target tier is valid for tiny test runs.
        cfg = PrefixSuffixTunerConfig(thrash=0.5)
        assert cfg.thrash == 0.5

    def test_invalid_num_prefixes(self) -> None:
        with pytest.raises(ValueError, match="num_prefixes must be >= 1"):
            PrefixSuffixTunerConfig(num_prefixes=0)

    def test_invalid_breaker_tokens(self) -> None:
        with pytest.raises(ValueError, match="breaker_tokens must be >= 1"):
            PrefixSuffixTunerConfig(breaker_tokens=0)


# ---------------------------------------------------------------------------
# PrefixSuffixTunerConfig.resolve
#
# ``thrash`` is the target tier size in GB; the resolve() helper applies an
# internal _OVERFLOW_FACTOR (1.05) to size the prefix pool slightly larger
# than the targeted tier, so pass-2 misses everything in that tier.
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerConfigResolve:
    def test_resolve_basic(self) -> None:
        cfg = PrefixSuffixTunerConfig.resolve(
            tokens_per_gb_kvcache=10000,
            context_length=4000,
            prefix_ratio=0.5,
            thrash=10.0,
        )
        # prefix_tokens = round(4000 * 0.5) = 2000
        assert cfg.prefix_tokens == 2000
        # suffix_tokens = 4000 - 2000 - 32 = 1968
        assert cfg.suffix_tokens == 1968
        # pool_gb = 10 * 1.05 = 10.5 GB; 10.5 * 10000 / 4000 = 26.25 → 26
        assert cfg.num_prefixes == 26

    def test_resolve_default_thrash_scaling(self) -> None:
        cfg = PrefixSuffixTunerConfig.resolve(
            tokens_per_gb_kvcache=50000,
            context_length=8000,
            prefix_ratio=0.8,
        )
        # prefix_tokens = round(8000 * 0.8) = 6400
        assert cfg.prefix_tokens == 6400
        # default thrash = 20.0 GB; pool = 20 * 1.05 = 21 GB
        # 21 * 50000 / 8000 = 131.25 → 131  (sized by context_length)
        assert cfg.num_prefixes == 131

    def test_resolve_minimum_one_prefix(self) -> None:
        cfg = PrefixSuffixTunerConfig.resolve(
            tokens_per_gb_kvcache=1,
            context_length=4000,
            prefix_ratio=0.5,
            thrash=0.0001,
        )
        assert cfg.num_prefixes == 1

    def test_resolve_suffix_too_small(self) -> None:
        # context=200, prefix_ratio=0.95 → prefix=190, breaker=32, suffix=-22
        with pytest.raises(ValueError, match="suffix_tokens=.* below minimum 100"):
            PrefixSuffixTunerConfig.resolve(
                tokens_per_gb_kvcache=10000,
                context_length=200,
                prefix_ratio=0.95,
                thrash=10.0,
            )

    def test_resolve_suffix_at_minimum(self) -> None:
        # context = 200, prefix=68, breaker=32, suffix=100 (exactly minimum)
        cfg = PrefixSuffixTunerConfig.resolve(
            tokens_per_gb_kvcache=1000,
            context_length=200,
            prefix_ratio=0.34,
            thrash=1.0,
        )
        assert cfg.suffix_tokens == 100

    def test_resolve_thrash_scales_pool_linearly(self) -> None:
        cfg_small = PrefixSuffixTunerConfig.resolve(
            tokens_per_gb_kvcache=10000,
            context_length=4000,
            prefix_ratio=0.5,
            thrash=10.0,
        )
        cfg_big = PrefixSuffixTunerConfig.resolve(
            tokens_per_gb_kvcache=10000,
            context_length=4000,
            prefix_ratio=0.5,
            thrash=20.0,
        )
        # Doubling thrash (target tier GB) should roughly double num_prefixes
        # — within a single prefix of 2x because of integer flooring.
        assert abs(cfg_big.num_prefixes - 2 * cfg_small.num_prefixes) <= 1

    def test_resolve_applies_internal_overflow_factor(self) -> None:
        """``thrash`` is the target tier size, not the pool size — pool is
        ``thrash * _OVERFLOW_FACTOR``.  Verify that with a 100 GB target tier,
        the pool genuinely overshoots."""
        cfg = PrefixSuffixTunerConfig.resolve(
            tokens_per_gb_kvcache=10000,
            context_length=2000,
            prefix_ratio=0.5,
            thrash=100.0,
        )
        # prefix_tokens = 1000
        # pool_gb = 100 * 1.05 = 105
        # Sized by context_length=2000 (not prefix_tokens), so the L1
        # footprint of the pool == thrash * _OVERFLOW_FACTOR GB:
        # num_prefixes = 105 * 10000 / 2000 = 525
        assert cfg.num_prefixes == 525
        # Pool footprint (in tokens of full request) > target tier tokens:
        pool_tokens = cfg.num_prefixes * cfg.context_length
        target_tier_tokens = int(100.0 * 10000)
        assert pool_tokens > target_tier_tokens


# ---------------------------------------------------------------------------
# PrefixSuffixTunerWorkload — helpers
# ---------------------------------------------------------------------------


def _make_workload_config(**overrides) -> PrefixSuffixTunerConfig:
    defaults = dict(
        context_length=200,
        prefix_ratio=0.5,
        thrash=10.0,  # target tier size in GB; resolve() not invoked here
        num_prefixes=4,
        prefix_tokens=100,
        suffix_tokens=68,
        breaker_tokens=32,
    )
    defaults.update(overrides)
    return PrefixSuffixTunerConfig(**defaults)  # type: ignore[arg-type]


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
    config: PrefixSuffixTunerConfig | None = None,
    seed: int = 42,
) -> tuple[PrefixSuffixTunerWorkload, MagicMock, MagicMock, MagicMock]:
    if config is None:
        config = _make_workload_config()
    sender = _make_mock_sender()
    collector = MagicMock()
    monitor = MagicMock()
    workload = PrefixSuffixTunerWorkload(
        config,
        sender,
        collector,
        monitor,
        seed=seed,
    )
    return workload, sender, collector, monitor


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerData:
    def test_correct_prefix_count(self) -> None:
        w, *_ = _make_workload(_make_workload_config(num_prefixes=7))
        assert len(w._prefixes) == 7

    def test_prefixes_have_unique_id_at_start(self) -> None:
        w, *_ = _make_workload(_make_workload_config(num_prefixes=3))
        assert w._prefixes[0].startswith("PREFIX_00000000")
        assert w._prefixes[1].startswith("PREFIX_00000001")
        assert w._prefixes[2].startswith("PREFIX_00000002")

    def test_prefixes_are_distinct(self) -> None:
        w, *_ = _make_workload(_make_workload_config(num_prefixes=5))
        assert len(set(w._prefixes)) == 5

    def test_single_shared_suffix(self) -> None:
        w, *_ = _make_workload()
        assert isinstance(w._suffix, str)
        assert w._suffix.startswith("SUFFIX")

    def test_data_is_deterministic_with_seed(self) -> None:
        cfg = _make_workload_config(num_prefixes=5)
        w1, *_ = _make_workload(cfg, seed=42)
        w2, *_ = _make_workload(cfg, seed=42)
        assert w1._prefixes == w2._prefixes
        assert w1._suffix == w2._suffix

    def test_data_differs_with_different_seed(self) -> None:
        # Only the breaker RNG depends on seed; prefixes/suffix are seed-
        # independent now (deterministic "hi"-filler bodies).  Verify the
        # breaker stream differs across seeds.
        cfg = _make_workload_config(num_prefixes=2)
        w1, *_ = _make_workload(cfg, seed=42)
        w2, *_ = _make_workload(cfg, seed=99)
        b1 = w1._build_messages(0)[0]["content"]
        b2 = w2._build_messages(0)[0]["content"]
        assert b1 != b2  # different breakers produce different prompts

    def test_prefix_bodies_use_hi_fallback_when_no_tokenizer(self) -> None:
        """Fallback path: when no tokenizer is loadable (unit tests pass
        ``model_name=None``), bodies use deterministic ``"hi"`` filler.

        Production runs with the real model name get **content-unique**
        random bodies generated from the tokenizer's vocab; that path is
        verified E2E (loading a real tokenizer in unit tests would be
        slow).  See ``test_prefix_bodies_are_unique_with_mock_tokenizer``
        for the tokenizer path under a mocked tokenizer.
        """
        w, *_ = _make_workload(_make_workload_config(num_prefixes=4))
        # Without a tokenizer, fallback "hi" filler kicks in.
        assert w._tokenizer is None
        bodies = [p.split(" ", 1)[1] for p in w._prefixes]
        assert len(set(bodies)) == 1
        assert all(set(b.split(" ")) == {"hi"} for b in bodies)

    def test_prefix_bodies_are_unique_with_mock_tokenizer(self) -> None:
        """Tokenizer path: each prefix samples a different per-prefix RNG,
        so the random token-ID sequences differ → decoded bodies differ.
        Required for non-blend cache hit-rate metrics to be meaningful
        (otherwise content-hash collisions across identical bodies would
        inflate hit rates regardless of LRU eviction).

        Uses a mocked tokenizer to avoid loading transformers in unit
        tests; the real-tokenizer path is exercised in E2E runs.
        """
        # First Party
        from lmcache.cli.commands.bench.engine_bench.workloads import (
            prefix_suffix_tuner as psf,
        )

        fake_tok = MagicMock()
        fake_tok.decode = lambda ids, **kw: " ".join(f"id{i}" for i in ids)
        original = psf._try_load_tokenizer
        psf._try_load_tokenizer = lambda model_name: fake_tok
        try:
            cfg = _make_workload_config(num_prefixes=8)
            sender = _make_mock_sender()
            collector = MagicMock()
            monitor = MagicMock()
            w = psf.PrefixSuffixTunerWorkload(
                cfg, sender, collector, monitor, seed=42, model_name="mock"
            )
        finally:
            psf._try_load_tokenizer = original

        bodies = [p.split(" ", 1)[1] for p in w._prefixes]
        # All 8 prefix bodies should be distinct token-id sequences.
        assert len(set(bodies)) == 8


# ---------------------------------------------------------------------------
# Message construction — request structure
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerMessages:
    def test_message_has_user_role(self) -> None:
        w, *_ = _make_workload()
        msgs = w._build_messages(0)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_message_contains_prefix(self) -> None:
        w, *_ = _make_workload()
        msgs = w._build_messages(2)
        assert "PREFIX_00000002" in msgs[0]["content"]

    def test_message_contains_shared_suffix(self) -> None:
        w, *_ = _make_workload()
        msgs0 = w._build_messages(0)
        msgs1 = w._build_messages(1)
        # Same suffix appears in both
        assert w._suffix in msgs0[0]["content"]
        assert w._suffix in msgs1[0]["content"]

    def test_breaker_differs_per_request(self) -> None:
        w, *_ = _make_workload()
        msgs_a = w._build_messages(0)
        msgs_b = w._build_messages(0)  # same prefix, different breaker
        # The two prompts must differ even for the same prefix index
        assert msgs_a[0]["content"] != msgs_b[0]["content"]

    def test_request_layout_prefix_breaker_suffix(self) -> None:
        w, *_ = _make_workload(_make_workload_config(num_prefixes=1))
        content = w._build_messages(0)[0]["content"]
        # Prefix appears before suffix
        prefix_pos = content.find("PREFIX_")
        suffix_pos = content.find("SUFFIX")
        assert prefix_pos == 0
        assert suffix_pos > prefix_pos

    def test_on_request_finished_noop(self) -> None:
        w, *_ = _make_workload()
        w.on_request_finished("some_id", "some_text")  # should not raise


# ---------------------------------------------------------------------------
# Pass 1 — warmup (async)
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerWarmup:
    @pytest.mark.asyncio
    async def test_warmup_sends_engine_warmup_then_each_prefix_once(self) -> None:
        cfg = _make_workload_config(num_prefixes=4)
        w, sender, _, _ = _make_workload(cfg)

        await w.warmup()

        # 1 engine warmup + 4 pass-1 prefixes = 5 total
        assert sender.send_warmup_request.call_count == 5
        # First call is the engine warmup (throwaway, before pass 1).
        assert sender.send_warmup_request.call_args_list[0][0][0] == "engine_warmup"
        # Subsequent calls are pass-1 prefixes in index order.
        for i, call in enumerate(sender.send_warmup_request.call_args_list[1:]):
            assert call[0][0] == f"pass1_p{i}"

    @pytest.mark.asyncio
    async def test_warmup_sends_pass1_prefixes_in_order(self) -> None:
        cfg = _make_workload_config(num_prefixes=3)
        w, sender, _, _ = _make_workload(cfg)

        await w.warmup()

        ids = [call[0][0] for call in sender.send_warmup_request.call_args_list]
        assert ids == ["engine_warmup", "pass1_p0", "pass1_p1", "pass1_p2"]

    @pytest.mark.asyncio
    async def test_warmup_uses_real_request_structure(self) -> None:
        cfg = _make_workload_config(num_prefixes=2)
        w, sender, _, _ = _make_workload(cfg)
        await w.warmup()

        # The engine-warmup request (call 0) sends a tiny throwaway prompt;
        # the pass-1 requests (calls 1..N) send full prefix+breaker+suffix.
        warmup_messages = sender.send_warmup_request.call_args_list[0][0][1]
        warmup_content = warmup_messages[0]["content"]
        assert "PREFIX_" not in warmup_content  # throwaway, no real prefix
        first_pass1_messages = sender.send_warmup_request.call_args_list[1][0][1]
        first_pass1_content = first_pass1_messages[0]["content"]
        assert "PREFIX_00000000" in first_pass1_content
        assert w._suffix in first_pass1_content


# ---------------------------------------------------------------------------
# Pass 2 — step (async)
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerStep:
    @pytest.mark.asyncio
    async def test_step_sends_one_request_per_call(self) -> None:
        cfg = _make_workload_config(num_prefixes=2)
        w, sender, _, _ = _make_workload(cfg)

        result = await w.step(0.0)
        assert result == 0.0
        assert sender.send_request.call_count == 1

    @pytest.mark.asyncio
    async def test_step_terminates_after_pool_exhausted(self) -> None:
        cfg = _make_workload_config(num_prefixes=2)
        w, _, _, _ = _make_workload(cfg)

        await w.step(0.0)
        await w.step(0.0)
        assert (await w.step(0.0)) == -1.0

    @pytest.mark.asyncio
    async def test_step_dispatches_in_pool_order(self) -> None:
        cfg = _make_workload_config(num_prefixes=3)
        w, sender, _, _ = _make_workload(cfg)

        await w.step(0.0)
        await w.step(0.0)
        await w.step(0.0)

        ids = [call[0][0] for call in sender.send_request.call_args_list]
        assert ids == ["pass2_p0", "pass2_p1", "pass2_p2"]

    @pytest.mark.asyncio
    async def test_step_uses_max_tokens_one(self) -> None:
        cfg = _make_workload_config(num_prefixes=1)
        w, sender, _, _ = _make_workload(cfg)
        await w.step(0.0)
        # send_request called with max_tokens=1
        assert sender.send_request.call_args.kwargs["max_tokens"] == 1


# ---------------------------------------------------------------------------
# Pass ordering — pass 1 and pass 2 use same prefix order, different breakers
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerTwoPassOrdering:
    @pytest.mark.asyncio
    async def test_pass1_and_pass2_share_prefix_order(self) -> None:
        cfg = _make_workload_config(num_prefixes=4)
        w, sender, _, _ = _make_workload(cfg)

        await w.warmup()
        # Run pass 2 to exhaustion
        while True:
            r = await w.step(0.0)
            if r < 0:
                break

        # Skip the engine-warmup throwaway call (call 0); pass-1 prefixes
        # start at call 1.
        warmup_prefixes = [
            c[0][1][0]["content"].split()[0]
            for c in sender.send_warmup_request.call_args_list[1:]
        ]
        bench_prefixes = [
            c[0][1][0]["content"].split()[0] for c in sender.send_request.call_args_list
        ]
        # Same prefix sequence in both passes
        assert warmup_prefixes == bench_prefixes
        assert warmup_prefixes == [f"PREFIX_{i:08x}" for i in range(4)]

    @pytest.mark.asyncio
    async def test_pass1_and_pass2_use_different_breakers(self) -> None:
        cfg = _make_workload_config(num_prefixes=2)
        w, sender, _, _ = _make_workload(cfg)

        await w.warmup()
        while True:
            r = await w.step(0.0)
            if r < 0:
                break

        # For prefix 0, the pass-1 and pass-2 prompts must differ
        # (same prefix and suffix, but different random breaker).
        # Note: call 0 of send_warmup_request is the engine warmup; pass-1
        # prefix 0 is at call index 1.
        pass1_prefix0 = sender.send_warmup_request.call_args_list[1][0][1][0]["content"]
        pass2_prefix0 = sender.send_request.call_args_list[0][0][1][0]["content"]
        assert pass1_prefix0 != pass2_prefix0


# ---------------------------------------------------------------------------
# Full run end-to-end
# ---------------------------------------------------------------------------


class TestPrefixSuffixTunerFullRun:
    def test_full_run(self) -> None:
        cfg = _make_workload_config(num_prefixes=3)
        w, sender, collector, _ = _make_workload(cfg)

        w.run()

        # 1 engine warmup + 3 pass-1 prefixes = 4 warmup requests total
        assert sender.send_warmup_request.call_count == 4
        # Pass 2: 3 measured requests
        assert sender.send_request.call_count == 3
        # Stats reset between passes (warmup discarded)
        collector.reset.assert_called_once()


# ---------------------------------------------------------------------------
# LRU simulation — verifies the central design invariant
#
# The workload's value rests on a single algorithmic claim: with a sequential
# pass-1 / pass-2 dispatch in identical order and a pool that just barely
# overflows the targeted tier, every pass-2 access misses that tier.  These
# tests prove that claim against an in-memory LRU model, using the access
# order extracted from the actual workload.
# ---------------------------------------------------------------------------


def _simulate_lru(access_seq: list[int], capacity: int) -> tuple[int, int]:
    """Replay *access_seq* against an LRU of *capacity*; return (hits, misses).

    Args:
        access_seq: Ordered list of cache keys to access.
        capacity: Maximum number of entries the cache holds.

    Returns:
        Tuple of (hit_count, miss_count) over the full sequence.
    """
    cache: OrderedDict[int, bool] = OrderedDict()
    hits = 0
    misses = 0
    for key in access_seq:
        if key in cache:
            cache.move_to_end(key)
            hits += 1
        else:
            cache[key] = True
            if len(cache) > capacity:
                cache.popitem(last=False)
            misses += 1
    return hits, misses


async def _capture_workload_access_order(
    workload: PrefixSuffixTunerWorkload,
    sender: MagicMock,
) -> tuple[list[int], list[int]]:
    """Run pass 1 + pass 2 and return the prefix-index access order of each.

    The mocked *sender* records which prefix index it was asked to send.
    The returned tuple lets tests assert on the order of pass 1 and pass
    2 independently.
    """
    pass1: list[int] = []
    pass2: list[int] = []

    async def capture_warmup(req_id, _msgs, **_kw):
        # request_id format: "pass1_p<index>" for cache-population requests;
        # "engine_warmup" for the throwaway request before pass 1, which is
        # not part of the access order under test.
        if req_id == "engine_warmup":
            return _make_mock_result(req_id)
        pass1.append(int(req_id.split("_p")[1]))
        return _make_mock_result(req_id)

    async def capture_request(req_id, _msgs, **_kw):
        pass2.append(int(req_id.split("_p")[1]))
        return _make_mock_result(req_id)

    sender.send_warmup_request = AsyncMock(side_effect=capture_warmup)
    sender.send_request = AsyncMock(side_effect=capture_request)

    await workload.warmup()
    while True:
        next_wake = await workload.step(0.0)
        if next_wake < 0:
            break
    return pass1, pass2


class TestLRUSimulator:
    """Sanity checks on the LRU helper itself."""

    def test_all_misses_when_no_repeats(self) -> None:
        h, m = _simulate_lru([0, 1, 2, 3], capacity=10)
        assert (h, m) == (0, 4)

    def test_all_hits_after_repeat_within_capacity(self) -> None:
        h, m = _simulate_lru([0, 1, 0, 1], capacity=10)
        assert (h, m) == (2, 2)

    def test_eviction_fifo_under_strict_lru(self) -> None:
        # capacity 2; access 0, 1, 2 → evicts 0; access 0 → miss.
        h, m = _simulate_lru([0, 1, 2, 0], capacity=2)
        assert (h, m) == (0, 4)

    def test_recent_access_promotes_to_mru(self) -> None:
        # capacity 2; access 0, 1, 0, 2 → evicts 1 (LRU), not 0.
        h, m = _simulate_lru([0, 1, 0, 2, 0], capacity=2)
        # 0 (miss), 1 (miss), 0 (hit), 2 (miss; evict 1), 0 (hit)
        assert (h, m) == (2, 3)


class TestPrefixSuffixTunerWorkloadAccessOrder:
    """Confirm the workload sends prefixes in identical sequential order
    across both passes — the precondition for the LRU invariant below."""

    @pytest.mark.asyncio
    async def test_pass1_is_sequential(self) -> None:
        cfg = _make_workload_config(num_prefixes=8)
        w, sender, _, _ = _make_workload(cfg)
        pass1, _ = await _capture_workload_access_order(w, sender)
        assert pass1 == list(range(8))

    @pytest.mark.asyncio
    async def test_pass2_matches_pass1(self) -> None:
        cfg = _make_workload_config(num_prefixes=8)
        w, sender, _, _ = _make_workload(cfg)
        pass1, pass2 = await _capture_workload_access_order(w, sender)
        assert pass2 == pass1


class TestPrefixSuffixTunerLRUInvariant:
    """The central design claim: every pass-2 request misses the targeted
    tier when the prefix pool is sized just larger than tier capacity."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "num_prefixes,capacity",
        [
            (21, 20),  # 1.05× overflow — the documented default
            (40, 38),  # ~1.05× at a different scale
            (100, 50),  # 2× overflow
            (1000, 950),  # large pool, light overflow
        ],
    )
    async def test_pass2_zero_hits_when_pool_overflows(
        self, num_prefixes: int, capacity: int
    ) -> None:
        cfg = _make_workload_config(num_prefixes=num_prefixes)
        w, sender, _, _ = _make_workload(cfg)
        pass1, pass2 = await _capture_workload_access_order(w, sender)

        # Pass 1 fills the cache; pass 2 must miss every entry in the
        # targeted tier (capacity).  We measure pass-2 hits independently
        # by replaying pass 1 first to populate the LRU, then pass 2.
        cache: OrderedDict[int, bool] = OrderedDict()
        for key in pass1:
            cache[key] = True
            if len(cache) > capacity:
                cache.popitem(last=False)

        pass2_hits = 0
        pass2_misses = 0
        for key in pass2:
            if key in cache:
                cache.move_to_end(key)
                pass2_hits += 1
            else:
                cache[key] = True
                if len(cache) > capacity:
                    cache.popitem(last=False)
                pass2_misses += 1

        assert pass2_hits == 0, (
            f"thrash invariant broken: {pass2_hits}/{num_prefixes} pass-2 "
            f"requests hit a tier of capacity {capacity}"
        )
        assert pass2_misses == num_prefixes

    @pytest.mark.asyncio
    async def test_pass2_all_hits_when_pool_fits_in_tier(self) -> None:
        # Counter-example: with pool ≤ capacity, pass 1 fills the cache
        # without any eviction, so pass 2 is 100% hits.  This sanity-checks
        # that the LRU sim correctly distinguishes "thrashing" from
        # "cache-friendly" workloads.
        cfg = _make_workload_config(num_prefixes=10)
        w, sender, _, _ = _make_workload(cfg)
        pass1, pass2 = await _capture_workload_access_order(w, sender)

        cache: OrderedDict[int, bool] = OrderedDict()
        capacity = 20  # tier larger than pool
        for key in pass1:
            cache[key] = True
            if len(cache) > capacity:
                cache.popitem(last=False)

        pass2_hits = 0
        for key in pass2:
            if key in cache:
                cache.move_to_end(key)
                pass2_hits += 1
            else:
                cache[key] = True
        assert pass2_hits == 10

    @pytest.mark.asyncio
    async def test_pass2_all_hits_when_pool_equals_tier(self) -> None:
        # Boundary: pool == capacity (thrash == 1.0 exactly).  Pass 1 fills
        # the cache to the brim, pass 2 is 100% hits.  This is why the
        # default thrash is strictly > 1.0.
        cfg = _make_workload_config(num_prefixes=15)
        w, sender, _, _ = _make_workload(cfg)
        pass1, pass2 = await _capture_workload_access_order(w, sender)

        cache: OrderedDict[int, bool] = OrderedDict()
        capacity = 15
        for key in pass1:
            cache[key] = True
            if len(cache) > capacity:
                cache.popitem(last=False)

        pass2_hits = sum(1 for key in pass2 if key in cache)
        assert pass2_hits == 15
