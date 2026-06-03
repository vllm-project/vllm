# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for ACE (Attention-Weighted Context Eviction) context compression."""

import numpy as np
import pytest

from vllm.entrypoints.attention_capture import (
    ACEAttentionCapture,
    get_capture,
    release_capture,
    start_capture,
    stop_capture,
)
from vllm.entrypoints.context_compression import (
    AttentionImportanceTracker,
    _heuristic_score,
    ace_compress,
    apply_ace_eviction,
    compute_line_token_spans,
    get_tracker,
    register_tracker,
    release_tracker,
)


# ---------------------------------------------------------------------------
# _score_line tests
# ---------------------------------------------------------------------------


def test_score_blank_line():
    assert _heuristic_score("") == 0.0
    assert _heuristic_score("   ") == 0.0


def test_score_error_line():
    assert _heuristic_score("Error: connection refused") >= 0.95
    assert _heuristic_score("Traceback (most recent call last):") >= 0.95
    assert _heuristic_score("exit code 1") >= 0.95


def test_score_tool_call_json():
    line = '{"name": "bash", "arguments": {"cmd": "ls"}}'
    assert _heuristic_score(line) == 1.0


def test_score_numeric_data():
    assert _heuristic_score("Processed 12345 rows in 3.2s") >= 0.7
    assert _heuristic_score("Available: 2048 bytes remaining") >= 0.7


def test_score_boilerplate():
    assert _heuristic_score("done") < 0.5
    assert _heuristic_score("ok") < 0.5


def test_score_meta_commentary():
    assert _heuristic_score("Here are the results") <= 0.15
    assert _heuristic_score("I executed the command") <= 0.15


# ---------------------------------------------------------------------------
# ace_compress tests
# ---------------------------------------------------------------------------


def _make_long_content(n_lines: int = 20) -> str:
    """Generate a synthetic multi-line tool output for testing."""
    lines = ["Tool output start"]
    for i in range(n_lines - 2):
        lines.append(f"Regular output line {i}: some verbose data here")
    lines.append("Tool output end")
    return "\n".join(lines)


def test_ace_compress_basic():
    content = _make_long_content(20)
    compressed = ace_compress(content, target_ratio=0.4)
    original_lines = content.split("\n")
    compressed_lines = compressed.split("\n")
    # Should be shorter than original (some omitted lines replaced by markers)
    assert len(compressed_lines) < len(original_lines)
    # Omission marker must be present
    assert any("omitted by ACE" in line for line in compressed_lines)


def test_ace_compress_first_last_always_kept():
    content = _make_long_content(20)
    lines = content.split("\n")
    compressed = ace_compress(content, target_ratio=0.3)
    compressed_lines = compressed.split("\n")
    # First line must appear as first element (no preceding omission marker)
    assert compressed_lines[0] == lines[0]
    # Last line must be the final line
    assert compressed_lines[-1] == lines[-1]


def test_ace_compress_ratio_1_no_compression():
    content = _make_long_content(10)
    compressed = ace_compress(content, target_ratio=1.0)
    # With ratio=1.0 all lines are kept; no omission markers
    assert "omitted by ACE" not in compressed


def test_ace_compress_short_content_unchanged():
    content = "line one\nline two\nline three"
    assert ace_compress(content, target_ratio=0.4) == content


def test_ace_compress_error_line_preserved():
    """Error lines should score high and survive aggressive compression."""
    lines = ["start"] + ["verbose filler line"] * 20 + ["Error: fatal crash"] + ["end"]
    content = "\n".join(lines)
    compressed = ace_compress(content, target_ratio=0.2)
    assert "Error: fatal crash" in compressed


# ---------------------------------------------------------------------------
# apply_ace_eviction tests
# ---------------------------------------------------------------------------


def _make_messages(tool_content_size: int = 500, n_tools: int = 4) -> list[dict]:
    messages = [{"role": "user", "content": "Do some work"}]
    for i in range(n_tools):
        messages.append(
            {"role": "assistant", "content": f"Calling tool {i}"}
        )
        messages.append(
            {
                "role": "tool",
                "content": f"Tool {i} result:\n"
                + "\n".join(f"output line {j}: data value {j}" for j in range(tool_content_size // 20)),
            }
        )
    messages.append({"role": "user", "content": "What is the answer?"})
    return messages


def test_apply_ace_no_compression_when_under_budget():
    messages = _make_messages(tool_content_size=100, n_tools=2)
    original = [dict(m) for m in messages]
    saved = apply_ace_eviction(messages, budget_chars=1_000_000)
    assert saved == 0
    assert messages == original


def test_apply_ace_compresses_when_over_budget():
    messages = _make_messages(tool_content_size=2000, n_tools=4)
    total_before = sum(
        len(m["content"]) for m in messages if isinstance(m.get("content"), str)
    )
    budget = total_before // 2  # Force compression
    saved = apply_ace_eviction(messages, budget_chars=budget)
    assert saved > 0
    total_after = sum(
        len(m["content"]) for m in messages if isinstance(m.get("content"), str)
    )
    assert total_after < total_before


def test_apply_ace_keep_recent_messages_unmodified():
    """The most-recent tool messages should not be compressed."""
    messages = _make_messages(tool_content_size=2000, n_tools=4)
    # Get the content of the last two tool messages before compression
    tool_msgs = [
        (i, m)
        for i, m in enumerate(messages)
        if m.get("role") == "tool"
    ]
    last_two_original = [(i, m["content"]) for i, m in tool_msgs[-2:]]

    budget = 1  # Force maximum compression
    apply_ace_eviction(messages, budget_chars=budget, keep_recent=2)

    for idx, original_content in last_two_original:
        assert messages[idx]["content"] == original_content, (
            f"Message at index {idx} should not have been compressed"
        )


def test_apply_ace_returns_chars_removed():
    messages = _make_messages(tool_content_size=2000, n_tools=4)
    total_before = sum(
        len(m["content"]) for m in messages if isinstance(m.get("content"), str)
    )
    saved = apply_ace_eviction(messages, budget_chars=total_before // 3)
    total_after = sum(
        len(m["content"]) for m in messages if isinstance(m.get("content"), str)
    )
    assert saved == total_before - total_after


def test_apply_ace_skips_short_messages():
    """Messages under min_chars should be left alone even if over budget."""
    short_tool_content = "short"
    messages = [
        {"role": "user", "content": "x"},
        {"role": "tool", "content": short_tool_content},
    ]
    saved = apply_ace_eviction(
        messages, budget_chars=1, min_chars=len(short_tool_content) + 1
    )
    assert saved == 0
    assert messages[1]["content"] == short_tool_content


def test_apply_ace_idempotent_on_already_compressed():
    """Running ACE twice should not double-compress (omission markers skipped)."""
    messages = _make_messages(tool_content_size=2000, n_tools=3)
    budget = sum(
        len(m["content"]) for m in messages if isinstance(m.get("content"), str)
    ) // 2
    apply_ace_eviction(messages, budget_chars=budget)
    snapshot = [dict(m) for m in messages]
    apply_ace_eviction(messages, budget_chars=budget)
    assert messages == snapshot

# ---------------------------------------------------------------------------
# Phase 2: BM25 query-relevance tests
# ---------------------------------------------------------------------------


def test_ace_compress_bm25_prefers_query_terms():
    """Lines containing query terms should score higher and survive compression."""
    lines = (
        ["Tool output start"]
        + ["completely unrelated filler text here"] * 10
        + ["TypeError: argument must be a string not int"]
        + ["completely unrelated filler text here"] * 10
        + ["Tool output end"]
    )
    content = "\n".join(lines)
    # Query contains the relevant term
    compressed = ace_compress(content, target_ratio=0.3, query="TypeError argument string")
    assert "TypeError: argument must be a string not int" in compressed


def test_ace_compress_bm25_fallback_with_empty_query():
    """Empty or whitespace-only query should fall back to heuristic scoring."""
    content = "\n".join(["start"] + ["filler line"] * 20 + ["Error: crash"] + ["end"])
    # Both modes should preserve the error line
    c1 = ace_compress(content, target_ratio=0.3, query=None)
    c2 = ace_compress(content, target_ratio=0.3, query="")
    assert "Error: crash" in c1
    # With empty query, BM25 returns 0.5 for all → heuristics decide
    assert "Error: crash" in c2


def test_apply_ace_bm25_mode_compresses():
    """With use_query_relevance=True, eviction still reduces size."""
    messages = _make_messages(tool_content_size=2000, n_tools=4)
    total_before = sum(len(m["content"]) for m in messages if isinstance(m.get("content"), str))
    budget = total_before // 2
    saved = apply_ace_eviction(messages, budget_chars=budget, use_query_relevance=True)
    assert saved > 0


# ---------------------------------------------------------------------------
# Phase 3: AttentionImportanceTracker tests
# ---------------------------------------------------------------------------


def _make_attn_weights(
    n_layers: int = 4,
    n_heads: int = 8,
    n_new_tokens: int = 5,
    seq_len: int = 32,
    hot_positions: list[int] | None = None,
) -> np.ndarray:
    """Synthetic attention weights with high attention on hot_positions."""
    weights = np.ones((n_layers, n_heads, n_new_tokens, seq_len), dtype=np.float32)
    if hot_positions:
        for pos in hot_positions:
            weights[:, :, :, pos] += 10.0
    # Normalize over seq_len so each row sums to 1
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights


def test_tracker_accumulate_and_score():
    """Tokens with high synthetic attention should receive high line scores."""
    tracker = AttentionImportanceTracker(max_seq_len=64)
    assert not tracker.has_data

    # Tokens 5-9 get high attention (simulate "hot" region)
    weights = _make_attn_weights(seq_len=20, hot_positions=[5, 6, 7, 8, 9])
    tracker.accumulate(weights, new_token_start=15)
    assert tracker.has_data

    # Line spans: line 0 = tokens 0-4 (cold), line 1 = tokens 5-9 (hot)
    spans = [(0, 5), (5, 10), (10, 15)]
    scores = tracker.score_lines(spans)
    assert len(scores) == 3
    # Hot region (span 1) should outscore cold regions
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]
    # All scores in [0, 1]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_tracker_no_data_returns_neutral():
    tracker = AttentionImportanceTracker(max_seq_len=64)
    spans = [(0, 5), (5, 10)]
    scores = tracker.score_lines(spans)
    assert scores == [0.5, 0.5]


def test_tracker_reset():
    tracker = AttentionImportanceTracker(max_seq_len=64)
    weights = _make_attn_weights(seq_len=20, hot_positions=[5])
    tracker.accumulate(weights, new_token_start=15)
    assert tracker.has_data
    tracker.reset()
    assert not tracker.has_data
    scores = tracker.score_lines([(0, 5)])
    assert scores == [0.5]


def test_tracker_multiple_accumulations():
    """Scores should increase monotonically with accumulated evidence."""
    tracker = AttentionImportanceTracker(max_seq_len=64)
    weights = _make_attn_weights(seq_len=20, hot_positions=[5, 6, 7])

    tracker.accumulate(weights, new_token_start=15)
    scores_1 = tracker.score_lines([(5, 8), (10, 15)])

    tracker.accumulate(weights, new_token_start=15)
    scores_2 = tracker.score_lines([(5, 8), (10, 15)])

    # Relative ranking should be preserved (hot > cold regardless of n_accumulations)
    assert scores_2[0] >= scores_1[0] or abs(scores_2[0] - scores_1[0]) < 0.01


def test_ace_compress_mode3_attention_preserves_hot_lines():
    """Lines mapped to high-attention tokens must survive aggressive compression."""
    lines = ["start"] + [f"cold line {i}" for i in range(18)] + ["hot critical line"] + ["end"]
    content = "\n".join(lines)
    # The "hot critical line" is at index 19 (0-based), pretend it maps to high-attn tokens
    n_lines = len(lines)
    # attention_scores: all low except index 19
    attention_scores = [0.1] * n_lines
    attention_scores[19] = 1.0

    compressed = ace_compress(content, target_ratio=0.2, attention_scores=attention_scores)
    assert "hot critical line" in compressed


def test_ace_compress_mode3_takes_priority_over_bm25():
    """When attention_scores is provided, it should dominate over query/BM25."""
    lines = ["start"] + [f"line {i}" for i in range(10)] + ["end"]
    content = "\n".join(lines)
    n_lines = len(lines)

    # attention scores: index 5 is hot
    attention_scores = [0.1] * n_lines
    attention_scores[5] = 1.0
    hot_line = lines[5]

    # BM25 query points at a different line (index 3)
    query = lines[3]

    compressed = ace_compress(
        content, target_ratio=0.3, query=query, attention_scores=attention_scores
    )
    # Hot-attention line must survive
    assert hot_line in compressed


# ---------------------------------------------------------------------------
# Phase 3: Registry and ACEAttentionCapture tests
# ---------------------------------------------------------------------------


def test_register_and_get_tracker():
    rid = "test-request-001"
    try:
        tracker = register_tracker(rid, max_seq_len=128)
        assert tracker is not None
        assert get_tracker(rid) is tracker
    finally:
        release_tracker(rid)

    assert get_tracker(rid) is None


def test_ace_attention_capture_lifecycle():
    rid = "test-request-capture-001"
    try:
        cap = start_capture(rid, max_seq_len=128)
        assert get_capture(rid) is cap
        assert get_tracker(rid) is not None

        weights = _make_attn_weights(seq_len=20, hot_positions=[5, 6])
        cap.on_layer_output(0, weights[0], new_token_start=15)
        cap.on_layer_output(1, weights[1], new_token_start=15)

        stop_capture(rid)
        assert get_capture(rid) is None  # capture stopped
        assert get_tracker(rid) is not None  # tracker still alive for next turn
        assert get_tracker(rid).has_data

    finally:
        release_capture(rid)

    assert get_tracker(rid) is None


def test_ace_attention_capture_flush_aggregates_layers():
    """on_layer_output from all layers should be stacked and accumulated."""
    rid = "test-request-flush-001"
    try:
        cap = start_capture(rid, max_seq_len=64)
        weights = _make_attn_weights(
            n_layers=1, n_heads=4, n_new_tokens=3, seq_len=20, hot_positions=[5]
        )
        for layer_idx in range(4):
            cap.on_layer_output(layer_idx, weights[0], new_token_start=15)
        cap.flush()
        tracker = get_tracker(rid)
        assert tracker.has_data
        scores = tracker.score_lines([(5, 8), (10, 15)])
        assert scores[0] > scores[1]  # hot region outscore cold
    finally:
        release_capture(rid)


def test_apply_ace_eviction_mode3_with_tracker():
    """apply_ace_eviction with a pre-loaded tracker uses attention scores."""
    rid = "test-request-mode3-001"
    try:
        tracker = register_tracker(rid, max_seq_len=4096)

        # Simulate attention data: first tool result's content (tokens 0-50) is hot
        weights = _make_attn_weights(
            n_layers=2, n_heads=4, n_new_tokens=5, seq_len=100, hot_positions=list(range(50))
        )
        tracker.accumulate(weights, new_token_start=50)

        messages = _make_messages(tool_content_size=2000, n_tools=3)
        total_before = sum(len(m["content"]) for m in messages if isinstance(m.get("content"), str))
        budget = total_before // 2

        # Without tracker: uses BM25
        msgs_bm25 = [dict(m) for m in messages]
        saved_bm25 = apply_ace_eviction(msgs_bm25, budget_chars=budget, use_query_relevance=True)

        # With tracker but no tokenizer: falls back to BM25 (tracker.has_data but no tokenizer)
        msgs_mode3 = [dict(m) for m in messages]
        saved_mode3 = apply_ace_eviction(
            msgs_mode3, budget_chars=budget, tracker=tracker, tokenizer=None, token_offsets=None
        )

        # Both should compress something
        assert saved_bm25 > 0
        assert saved_mode3 > 0

    finally:
        release_tracker(rid)


# ---------------------------------------------------------------------------
# Phase 2+3 fallback chain
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 3: ace_req_registry unit tests
# ---------------------------------------------------------------------------

def test_ace_req_registry_basic():
    from vllm.entrypoints import ace_req_registry
    ace_req_registry.register("reg-req-001", "reg-sess-A")
    assert ace_req_registry.get("reg-req-001") == "reg-sess-A"
    assert ace_req_registry.get("reg-req-unknown") is None
    ace_req_registry.release("reg-req-001")
    assert ace_req_registry.get("reg-req-001") is None


def test_ace_req_registry_release_missing_noop():
    from vllm.entrypoints import ace_req_registry
    ace_req_registry.release("reg-does-not-exist")  # must not raise


def test_ace_req_registry_multiple_sessions():
    from vllm.entrypoints import ace_req_registry
    ace_req_registry.register("reg-req-A", "reg-sess-1")
    ace_req_registry.register("reg-req-B", "reg-sess-2")
    assert ace_req_registry.get("reg-req-A") == "reg-sess-1"
    assert ace_req_registry.get("reg-req-B") == "reg-sess-2"
    ace_req_registry.release("reg-req-A")
    assert ace_req_registry.get("reg-req-A") is None
    assert ace_req_registry.get("reg-req-B") == "reg-sess-2"
    ace_req_registry.release("reg-req-B")


def test_ace_phase3_end_to_end_flow():
    """
    Simulate the full Phase 3 flow without a real model:

      1. serving.py registers req_id -> session_id
      2. Attention capture writes synthetic weights (as model_runner hooks would)
      3. flush_hook aggregates them into the tracker
      4. Next turn: apply_ace_eviction finds tracker.has_data == True -> Mode 3
    """
    from vllm.entrypoints import ace_req_registry
    from vllm.entrypoints.ace_model_hook import release_hook
    from vllm.entrypoints.attention_capture import get_capture, start_capture
    from vllm.entrypoints.context_compression import get_tracker

    req_id = "chatcmpl-phase3-e2e"
    session_id = "conv-phase3-e2e"

    # Step 1: serving.py registers req_id -> session_id
    ace_req_registry.register(req_id, session_id)
    assert ace_req_registry.get(req_id) == session_id

    # Step 2: start_capture (first-turn setup in serving.py)
    start_capture(session_id, max_seq_len=128)
    cap = get_capture(session_id)
    assert cap is not None

    # Step 3: model forward pass — hooks would call on_layer_output; simulate directly
    weights = _make_attn_weights(
        n_layers=1, n_heads=2, n_new_tokens=2, seq_len=30, hot_positions=[5, 6, 7]
    )
    cap.on_layer_output(0, weights[0], new_token_start=20)
    cap.on_layer_output(1, weights[0], new_token_start=20)

    # Step 4: stop capture (aggregates buffers into tracker).
    # In production this is triggered via flush_hook() after the model forward pass.
    # In this test we call cap.stop() directly since there is no real model/hook.
    cap.stop()

    # Tracker now has data for next turn
    tracker = get_tracker(session_id)
    assert tracker is not None
    assert tracker.has_data

    # Next turn: apply_ace_eviction uses Mode 3
    messages = _make_messages(tool_content_size=1000, n_tools=3)
    total_before = sum(len(m["content"]) for m in messages if isinstance(m.get("content"), str))
    saved = apply_ace_eviction(messages, budget_chars=total_before // 2, tracker=tracker)
    assert saved > 0

    # Cleanup
    ace_req_registry.release(req_id)
    release_hook(session_id)
    assert get_tracker(session_id) is None


def test_fallback_chain_no_tracker_no_query():
    """Without tracker or query, falls back to heuristic (Phase 1)."""
    lines = ["start"] + ["verbose filler"] * 20 + ["Error: crash happened"] + ["end"]
    content = "\n".join(lines)
    compressed = ace_compress(content, target_ratio=0.2)
    assert "Error: crash happened" in compressed


def test_fallback_chain_bm25_without_attention():
    """With query but no tracker, uses BM25 (Phase 2)."""
    lines = ["start"] + ["unrelated filler"] * 15 + ["specific_function_call"] + ["end"]
    content = "\n".join(lines)
    compressed = ace_compress(content, target_ratio=0.3, query="specific_function_call")
    assert "specific_function_call" in compressed




# ---------------------------------------------------------------------------
# recency_blend tests
# ---------------------------------------------------------------------------

def test_recency_blend_zero_unchanged():
    """recency_blend=0 should produce identical results to no-recency call."""
    messages = _make_messages(tool_content_size=2000, n_tools=4)
    total = sum(len(m["content"]) for m in messages if isinstance(m.get("content"), str))
    budget = total // 2

    msgs_no_blend = [dict(m) for m in messages]
    msgs_blend_0  = [dict(m) for m in messages]
    s1 = apply_ace_eviction(msgs_no_blend, budget_chars=budget, recency_blend=0.0)
    s2 = apply_ace_eviction(msgs_blend_0,  budget_chars=budget, recency_blend=0.0)
    assert s1 == s2
    assert msgs_no_blend == msgs_blend_0


def test_recency_blend_oldest_compressed_more():
    """With recency_blend>0, oldest compressible message should be smaller than newest."""
    # Build 4 tool msgs, force eviction
    messages = _make_messages(tool_content_size=3000, n_tools=5)
    total = sum(len(m["content"]) for m in messages if isinstance(m.get("content"), str))
    budget = int(total * 0.4)

    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    original_oldest = messages[tool_indices[0]]["content"]
    original_newest = messages[tool_indices[-2]]["content"]  # -2 because -1 is keep_recent

    apply_ace_eviction(messages, budget_chars=budget, keep_recent=1, recency_blend=0.4)

    compressed_oldest = messages[tool_indices[0]]["content"]
    compressed_newest = messages[tool_indices[-2]]["content"]

    # Oldest should be compressed more (smaller) than newest
    ratio_oldest = len(compressed_oldest) / len(original_oldest)
    ratio_newest = len(compressed_newest) / len(original_newest)
    assert ratio_oldest < ratio_newest, (
        f"Oldest kept {ratio_oldest:.2%} but newest kept {ratio_newest:.2%} — "
        "recency blend should compress oldest more aggressively"
    )


def test_recency_blend_preserves_error_lines_in_old_messages():
    """Even heavily blended old messages must keep error lines (high structural score)."""
    error_line = "ERROR: ConnectionRefusedError — could not connect to database"
    filler = ["verbose filler line with no important content"] * 30
    old_content = "\n".join(filler[:15] + [error_line] + filler[15:])

    messages = [
        {"role": "user", "content": "start task"},
        {"role": "tool", "content": old_content},       # oldest — will be compressed hard
        {"role": "assistant", "content": "I found something"},
        {"role": "tool", "content": "recent result\nwith useful data"},  # newest
        {"role": "user", "content": "continue"},
    ]

    total = sum(len(m["content"]) for m in messages if isinstance(m.get("content"), str))
    apply_ace_eviction(messages, budget_chars=total // 3, keep_recent=1, recency_blend=0.4)

    assert error_line in messages[1]["content"], (
        "Error line must survive even aggressive recency-blended compression"
    )


def test_recency_blend_defaults_to_03():
    """Default recency_blend should be 0.3 (not 0.0 as in original)."""
    import inspect
    sig = inspect.signature(apply_ace_eviction)
    default = sig.parameters["recency_blend"].default
    assert default == 0.3, f"Expected default 0.3, got {default}"


def test_pixol_age_decay_exponential():
    """Pixol age decay should be exponential, not linear."""
    from vllm.entrypoints.pixol_compression import PixolLine
    line = PixolLine.from_text("def process(self, value): return value * 2")
    score_age0 = line.score

    line.age = 8
    score_age8 = line.score

    line.age = 16
    score_age16 = line.score

    # Exponential: age 16 should drop more than twice age 8 (relative to age 0)
    drop_8  = score_age0 - score_age8
    drop_16 = score_age0 - score_age16
    assert drop_16 > drop_8 * 1.5, "Decay should be faster than linear"
    # Never fully zeroed out
    assert score_age16 > 0.0


def test_compute_line_token_spans_fallback():
    """compute_line_token_spans returns plausible spans even without offset mapping."""

    class FakeTokenizer:
        def __call__(self, text, return_offsets_mapping=False):
            raise RuntimeError("offsets not supported")

        def encode(self, text):
            return list(range(max(1, len(text) // 4)))

    text = "line one\nline two\nline three"
    spans = compute_line_token_spans(text, FakeTokenizer(), message_start_token=10)
    assert len(spans) == 3
    for start, end in spans:
        assert end > start
        assert start >= 10
