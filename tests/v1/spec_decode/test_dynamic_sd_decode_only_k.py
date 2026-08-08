"""Tests for dynamic SD scheduler K-lookup using sampling-request count.

Regression test for #49548: dynamic speculative decoding K-lookup should
count only requests that will sample this step (decode or final-prefill
that reaches sampling), not mid-prefill requests that temporarily inflate
the batch size and cause K to thrash.
"""


class FakeRequest:
    """Minimal stand-in for Request to test the sampling predicate."""

    def __init__(self, num_computed_tokens, num_tokens,
                 num_output_placeholders=0):
        self.num_computed_tokens = num_computed_tokens
        self.num_tokens = num_tokens
        self.num_output_placeholders = num_output_placeholders


def _is_sampling_request(req, num_scheduled):
    """Replicates the predicate in scheduler.schedule()."""
    return (req.num_computed_tokens + num_scheduled
            >= req.num_tokens + req.num_output_placeholders)


def test_dynamic_sd_sampling_only_k_lookup():
    """Verify that only sampling requests are counted for DSD K lookup."""

    # Build a lookup table for schedule [[1, 2, 2], [3, 16, 0]]
    # batch 1-2 -> K=2, batch 3-16 -> K=0
    dynamic_sd_lookup = [0] * 17  # index 0 unused (1-indexed)
    for i in range(1, 17):
        dynamic_sd_lookup[i] = 2 if i <= 2 else 0

    # Case 1: 4 pure decode requests -> decode_count=4 -> K=0
    reqs = {"r0": FakeRequest(100, 100), "r1": FakeRequest(50, 50),
            "r2": FakeRequest(80, 80), "r3": FakeRequest(60, 60)}
    tokens = {"r0": 1, "r1": 1, "r2": 1, "r3": 1}
    dc = sum(1 for rid, n in tokens.items() if _is_sampling_request(reqs[rid], n))
    assert dc == 4, f"Expected 4 sampling, got {dc}"
    k = dynamic_sd_lookup[dc]
    assert k == 0, f"Expected K=0 for 4 decode, got K={k}"

    # Case 2: 2 decode + 1 mid-prefill -> decode_count=2 -> K=2
    # OLD behavior (total count=3 -> K=0) would thrash!
    reqs = {"r0": FakeRequest(100, 100), "r1": FakeRequest(50, 50),
            "r2": FakeRequest(10, 1024)}
    tokens = {"r0": 1, "r1": 1, "r2": 512}
    dc = sum(1 for rid, n in tokens.items() if _is_sampling_request(reqs[rid], n))
    assert dc == 2, f"Expected 2 sampling, got {dc}"
    k = dynamic_sd_lookup[dc]
    assert k == 2, f"Expected K=2 for 2 decode + 1 prefill, got K={k}"

    # Case 3: 1 decode + 1 full prefill (all tokens fit in one chunk)
    # Both reach sampling in the same step -> decode_count=2 -> K=2
    reqs = {"r0": FakeRequest(100, 100), "r1": FakeRequest(0, 1024)}
    tokens = {"r0": 1, "r1": 1024}
    dc = sum(1 for rid, n in tokens.items() if _is_sampling_request(reqs[rid], n))
    assert dc == 2, f"Expected 2 sampling (decode + full prefill), got {dc}"
    k = dynamic_sd_lookup[dc]
    assert k == 2, f"Expected K=2 for decode + full prefill, got K={k}"

    # Case 4: all full-prefill (each request's tokens fit in one chunk)
    # Both reach sampling -> decode_count=2 -> K=2
    reqs = {"r0": FakeRequest(0, 512), "r1": FakeRequest(0, 1024)}
    tokens = {"r0": 512, "r1": 1024}
    dc = sum(1 for rid, n in tokens.items() if _is_sampling_request(reqs[rid], n))
    assert dc == 2, f"Expected 2 sampling (both full prefills), got {dc}"
    lookup_count = dc if dc > 0 else len(tokens)
    k = dynamic_sd_lookup[lookup_count]
    assert k == 2, f"Expected K=2 for 2 full prefills, got K={k}"

    # Case 5: single decode (batch=1) -> K=2
    reqs = {"r0": FakeRequest(100, 100)}
    tokens = {"r0": 1}
    dc = sum(1 for rid, n in tokens.items() if _is_sampling_request(reqs[rid], n))
    assert dc == 1, f"Expected 1 sampling, got {dc}"
    k = dynamic_sd_lookup[dc]
    assert k == 2, f"Expected K=2 for single decode, got K={k}"

    # Case 6: final prefill chunk that reaches sampling (1024 computed + 1 token = 1025 total)
    # This request WILL sample, so should be counted
    reqs = {"r0": FakeRequest(1024, 1025), "r1": FakeRequest(100, 100)}
    tokens = {"r0": 1, "r1": 1}
    dc = sum(1 for rid, n in tokens.items() if _is_sampling_request(reqs[rid], n))
    assert dc == 2, f"Expected 2 sampling (final chunk + decode), got {dc}"
    k = dynamic_sd_lookup[dc]
    assert k == 2, f"Expected K=2 for final-chunk + decode, got K={k}"

    # Case 7: 1-token mid-prefill chunk (Codex's edge case)
    # Request has 1024 tokens, only 1 scheduled, but computed + 1 < total
    reqs = {"r0": FakeRequest(0, 1024), "r1": FakeRequest(100, 100)}
    tokens = {"r0": 1, "r1": 1}
    dc = sum(1 for rid, n in tokens.items() if _is_sampling_request(reqs[rid], n))
    assert dc == 1, f"Expected 1 sampling (1-token prefill chunk excluded), got {dc}"
    k = dynamic_sd_lookup[dc]
    assert k == 2, f"Expected K=2 for 1-token-prefill + decode, got K={k}"

    # Case 8: all mid-prefill chunks (nobody reaches sampling) -> fallback
    # This is the only case where fallback to len() is used
    reqs = {"r0": FakeRequest(0, 4096), "r1": FakeRequest(0, 8192)}
    tokens = {"r0": 1024, "r1": 1024}
    dc = sum(1 for rid, n in tokens.items() if _is_sampling_request(reqs[rid], n))
    assert dc == 0, f"Expected 0 sampling (both mid-prefill), got {dc}"
    lookup_count = dc if dc > 0 else len(tokens)
    k = dynamic_sd_lookup[lookup_count]
    assert k == 2, f"Expected K=2 for mid-prefill fallback (count=2), got K={k}"
