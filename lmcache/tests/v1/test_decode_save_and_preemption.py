# SPDX-License-Identifier: Apache-2.0
"""
Tests for:
1. Fix: decode save cache token slicing uses min(num_computed_tokens, tracker_len)
2. Fix: preemption assertion allows off-by-one from full-hit adjustment

These tests exercise the token slicing and assertion logic in
build_connector_meta (vllm_v1_adapter.py) without requiring GPU or vLLM server.
"""

# Standard
from dataclasses import dataclass, field

# Third Party
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs — just enough to test the slicing + assertion logic
# ---------------------------------------------------------------------------


@dataclass
class StubRequest:
    """Minimal vLLM Request stub."""

    request_id: str
    num_computed_tokens: int
    _all_token_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0

    @property
    def all_token_ids(self):
        return self._all_token_ids


@dataclass
class StubLoadSpec:
    vllm_cached_tokens: int
    lmcache_cached_tokens: int
    can_load: bool = False


# ---------------------------------------------------------------------------
# Extracted logic under test
# ---------------------------------------------------------------------------


def compute_new_token_ids(
    request: StubRequest,
    tracker_token_ids: list[int],
    num_new_tokens: int,
) -> list[int]:
    """
    Reproduces the fixed token slicing logic from build_connector_meta.
    Uses min(num_computed_tokens, tracker_len) as slice base.
    """
    num_current_tokens = request.num_computed_tokens
    tracker_len = len(tracker_token_ids)
    slice_base = min(num_current_tokens, tracker_len)
    return list(request.all_token_ids[slice_base : slice_base + num_new_tokens])


def check_preemption_assertion(
    request: StubRequest,
    load_spec: StubLoadSpec,
):
    """
    Reproduces the fixed assertion from build_connector_meta for preempted
    requests. On full cache hit where lmcache is dominant, expected is -1.
    """
    expected = max(load_spec.lmcache_cached_tokens, load_spec.vllm_cached_tokens)
    full_hit_adj = (
        load_spec.lmcache_cached_tokens == request.num_tokens
        and load_spec.lmcache_cached_tokens > load_spec.vllm_cached_tokens
    )
    if full_hit_adj:
        expected -= 1
    if request.num_computed_tokens != expected:
        raise AssertionError(
            f"Preempted request {request.request_id} has "
            f"num_computed_tokens {request.num_computed_tokens} "
            f"but expected {expected} (full_hit_adj={full_hit_adj})"
        )


# ===========================================================================
# Fix 1: token slicing with min(num_computed_tokens, tracker_len)
# ===========================================================================


class TestTokenSlicing:
    """Tests for the min()-based slice base in build_connector_meta."""

    def test_sync_decode_normal(self):
        """Normal sync decode: num_computed == tracker_len. Both in sync."""
        prompt = list(range(100))
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=100,
            _all_token_ids=prompt + [1001],  # T1 appended by update_from_output
        )
        tracker_ids = prompt.copy()
        result = compute_new_token_ids(request, tracker_ids, num_new_tokens=1)
        assert result == [1001], f"Should get decode token T1, got {result}"

    def test_num_computed_ahead_of_tracker(self):
        """
        The PR #2821 scenario: num_computed_tokens was incremented (e.g. by
        _update_after_schedule) but tracker hasn't been updated yet.
        min() picks tracker_len, giving the correct slice.
        """
        prompt = list(range(100))
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=101,  # already incremented
            _all_token_ids=prompt + [1001],
        )
        tracker_ids = prompt.copy()  # not yet updated
        result = compute_new_token_ids(request, tracker_ids, num_new_tokens=1)
        assert result == [1001], (
            "min() should pick tracker_len=100, slicing [100:101] = [T1]"
        )

    def test_num_computed_ahead_old_code_fails(self):
        """Verify the OLD code (using num_computed_tokens directly) would fail."""
        prompt = list(range(100))
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=101,
            _all_token_ids=prompt + [1001],
        )
        # Old code: slice base = num_computed_tokens = 101
        old_result = request.all_token_ids[101:102]
        assert old_result == [], "Old code produces empty slice (the bug)"

    def test_preemption_tracker_stale(self):
        """
        Preemption: tracker has stale tokens from before preemption,
        num_computed_tokens was reset. min() picks num_computed_tokens.
        """
        prompt = list(range(100))
        decode_tokens = list(range(1001, 1051))  # 50 decode tokens
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=50,  # reset to cached prefix after preemption
            _all_token_ids=prompt + decode_tokens,
        )
        tracker_ids = prompt + decode_tokens  # stale: 150 tokens
        num_new_tokens = 50  # recompute prompt[50:100]

        result = compute_new_token_ids(request, tracker_ids, num_new_tokens)
        # min(50, 150) = 50 → slice [50:100] = prompt tokens 50-99
        assert result == list(range(50, 100)), (
            "min() should pick num_computed=50, getting prompt suffix"
        )

    def test_preemption_old_fix_fails(self):
        """Verify the PR #2821 fix (tracker_len only) would fail on preemption."""
        prompt = list(range(100))
        decode_tokens = list(range(1001, 1051))
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=50,
            _all_token_ids=prompt + decode_tokens,
        )
        tracker_ids = prompt + decode_tokens  # 150 tokens (stale)
        tracker_len = len(tracker_ids)
        # PR #2821 fix: slice base = tracker_len = 150
        pr_result = request.all_token_ids[tracker_len : tracker_len + 50]
        assert pr_result == [], "PR #2821 fix gives empty (regression)"

    def test_chunked_prefill(self):
        """Chunked prefill: tracker has first chunk, processing second."""
        prompt = list(range(500))
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=256,  # first chunk done
            _all_token_ids=prompt,
        )
        tracker_ids = prompt[:256]  # first chunk
        num_new_tokens = 244  # second chunk

        result = compute_new_token_ids(request, tracker_ids, num_new_tokens)
        assert result == list(range(256, 500)), "Should get second chunk tokens"

    def test_both_equal_zero(self):
        """Edge case: brand new request, both are 0."""
        prompt = list(range(100))
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=0,
            _all_token_ids=prompt,
        )
        tracker_ids = []
        result = compute_new_token_ids(request, tracker_ids, num_new_tokens=100)
        assert result == prompt

    def test_multiple_decode_steps(self):
        """Trace through multiple decode steps to verify consistency."""
        prompt = list(range(100))
        tracker_ids = prompt.copy()

        for step in range(10):
            token = 1001 + step
            all_token_ids = prompt + list(range(1001, 1001 + step + 1))
            request = StubRequest(
                request_id="req1",
                num_computed_tokens=100 + step,
                _all_token_ids=all_token_ids,
            )
            result = compute_new_token_ids(request, tracker_ids, num_new_tokens=1)
            assert result == [token], f"Step {step}: expected [{token}], got {result}"
            tracker_ids.append(token)


# ===========================================================================
# Fix 2: preemption assertion off-by-one
# ===========================================================================


class TestPreemptionAssertion:
    """Tests for the deterministic assertion in build_connector_meta."""

    def test_no_full_hit_exact_match(self):
        """No full hit (lmcache < num_tokens): num_computed == expected."""
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=3000,
            num_tokens=4032,
        )
        load_spec = StubLoadSpec(
            vllm_cached_tokens=100,
            lmcache_cached_tokens=3000,
        )
        check_preemption_assertion(request, load_spec)

    def test_full_hit_lmcache_dominant(self):
        """
        Full hit + lmcache > vllm: get_num_new_matched_tokens subtracts 1,
        so num_computed = expected - 1.
        """
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=4031,  # T - 1
            num_tokens=4032,
        )
        load_spec = StubLoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=4032,
        )
        check_preemption_assertion(request, load_spec)

    def test_full_hit_both_equal(self):
        """
        Full hit but vllm == lmcache: need_to_allocate goes negative,
        returns 0, so num_computed = vllm_cached = num_tokens. No adjustment.
        """
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=4032,
            num_tokens=4032,
        )
        load_spec = StubLoadSpec(
            vllm_cached_tokens=4032,
            lmcache_cached_tokens=4032,
        )
        check_preemption_assertion(request, load_spec)

    def test_old_assertion_would_fail(self):
        """Verify the OLD strict assertion would crash on the full-hit case."""
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=4031,
            num_tokens=4032,
        )
        load_spec = StubLoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=4032,
        )
        # Old assertion: strict equality against unadjusted max
        unadjusted = max(load_spec.lmcache_cached_tokens, load_spec.vllm_cached_tokens)
        assert request.num_computed_tokens != unadjusted, (
            "Confirm the old strict == would fail"
        )

    def test_off_by_two_still_fails(self):
        """Off-by-two should still be caught."""
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=4030,  # T - 2
            num_tokens=4032,
        )
        load_spec = StubLoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=4032,
        )
        with pytest.raises(AssertionError):
            check_preemption_assertion(request, load_spec)

    def test_vllm_cached_dominates(self):
        """When vllm_cached > lmcache_cached, no adjustment."""
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=500,
            num_tokens=1000,
        )
        load_spec = StubLoadSpec(
            vllm_cached_tokens=500,
            lmcache_cached_tokens=200,
        )
        check_preemption_assertion(request, load_spec)

    def test_vllm_dominant_wrong_value_fails(self):
        """When vllm dominates, num_computed must equal vllm_cached exactly."""
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=499,  # wrong
            num_tokens=1000,
        )
        load_spec = StubLoadSpec(
            vllm_cached_tokens=500,
            lmcache_cached_tokens=200,
        )
        with pytest.raises(AssertionError):
            check_preemption_assertion(request, load_spec)

    def test_num_computed_above_expected_fails(self):
        """num_computed > expected should fail."""
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=4032,  # should be 4031 with full hit adj
            num_tokens=4032,
        )
        load_spec = StubLoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=4032,
        )
        with pytest.raises(AssertionError):
            check_preemption_assertion(request, load_spec)

    def test_realistic_batch_preemption_scenario(self):
        """
        Realistic scenario from the bug report: batch_size=32, preemption
        during decode with save_decode_cache=True.
        """
        request = StubRequest(
            request_id="req_batch32_7",
            num_computed_tokens=4031,  # T - 1 due to full hit adj
            num_tokens=4032,
        )
        load_spec = StubLoadSpec(
            vllm_cached_tokens=100,  # prefix cache hit (prompt)
            lmcache_cached_tokens=4032,  # full LMCache hit
        )
        check_preemption_assertion(request, load_spec)


# ===========================================================================
# Combined: both fixes working together
# ===========================================================================


class TestCombinedFixes:
    """Test that both fixes work correctly together in a preemption scenario."""

    def test_preempted_full_hit_decode_flow(self):
        """
        Full scenario: preempted request with full LMCache hit.
        Verifies both the slice and assertion work.
        """
        prompt = list(range(100))
        decode_tokens = list(range(1001, 1101))  # 100 decode tokens
        all_tokens = prompt + decode_tokens  # 200 total

        # After preemption + reschedule with full hit (-1 adjustment):
        request = StubRequest(
            request_id="req1",
            num_computed_tokens=199,  # 200 - 1 (full hit adjustment)
            _all_token_ids=all_tokens,
            num_tokens=200,
        )
        load_spec = StubLoadSpec(
            vllm_cached_tokens=100,  # prefix cache
            lmcache_cached_tokens=200,  # full LMCache hit
        )

        # 1. Assertion should pass (off-by-one allowed)
        check_preemption_assertion(request, load_spec)

        # 2. Token slicing: tracker is stale from before preemption
        stale_tracker = all_tokens.copy()  # 200 tokens
        num_new_tokens = 1  # recompute last token

        result = compute_new_token_ids(request, stale_tracker, num_new_tokens)
        # min(199, 200) = 199 → all_tokens[199:200] = [last decode token]
        assert result == [decode_tokens[-1]], (
            f"Should get the last decode token, got {result}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
