# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.tool_parsers.utils import compute_streamed_args_delta


class TestComputeStreamedArgsDelta:
    """Tests for compute_streamed_args_delta."""

    def test_is_complete_flushes_remaining(self):
        """When JSON is complete, flush everything not yet streamed."""
        result = compute_streamed_args_delta(
            cur_args_json='{"city": "Dallas"}',
            prev_args_json='{"city": "Dallas"}',
            already_streamed='{"city": "Dal',
            is_complete=True,
        )
        assert result == 'las"}'

    def test_is_complete_nothing_remaining(self):
        """When JSON is complete and everything already streamed, return None."""
        full = '{"city": "Dallas"}'
        result = compute_streamed_args_delta(
            cur_args_json=full,
            prev_args_json=full,
            already_streamed=full,
            is_complete=True,
        )
        assert result is None

    def test_is_complete_from_scratch(self):
        """When JSON is complete but nothing streamed yet, flush everything."""
        full = '{"city": "Dallas"}'
        result = compute_streamed_args_delta(
            cur_args_json=full,
            prev_args_json=None,
            already_streamed="",
            is_complete=True,
        )
        assert result == full

    def test_prev_none_holdback(self):
        """First time seeing arguments (prev is None) and JSON incomplete —
        hold back to avoid streaming auto-completed structure."""
        result = compute_streamed_args_delta(
            cur_args_json='{"city": "Dallas"}',
            prev_args_json=None,
            already_streamed="",
            is_complete=False,
        )
        assert result is None

    def test_no_change(self):
        """When current and previous are identical, return None."""
        same = '{"city": "Dal"}'
        result = compute_streamed_args_delta(
            cur_args_json=same,
            prev_args_json=same,
            already_streamed='{"city": "Dal',
            is_complete=False,
        )
        assert result is None

    def test_incremental_diff(self):
        """Stream only the stable prefix diff between prev and current."""
        result = compute_streamed_args_delta(
            cur_args_json='{"city": "Dallas"}',
            prev_args_json='{"city": "Dal"}',
            already_streamed='{"city": "Dal',
            is_complete=False,
        )
        # Common prefix of prev and cur is '{"city": "Dal'
        # already_streamed is '{"city": "Dal' — so diff is empty
        assert result is None

    def test_incremental_diff_with_new_content(self):
        """New content appears that extends past what was already streamed."""
        result = compute_streamed_args_delta(
            cur_args_json='{"city": "Dallas"}',
            prev_args_json='{"city": "Dalla"}',
            already_streamed='{"city": "Dal',
            is_complete=False,
        )
        # Common prefix: '{"city": "Dalla' (5 chars of "Dalla" shared)
        # already_streamed length is 14, prefix length is 16 → diff = "la"
        assert result == "la"

    def test_incremental_avoids_auto_completed_closing(self):
        """The partial parser auto-completes closing chars — don't stream
        those until JSON is truly finalized."""
        # prev parse auto-completed to: {"city": "ap"}
        # cur  parse auto-completed to: {"city": "apple"}
        # The common prefix is '{"city": "ap' — the closing '"}' in prev
        # is auto-completed and should not be streamed.
        result = compute_streamed_args_delta(
            cur_args_json='{"city": "apple"}',
            prev_args_json='{"city": "ap"}',
            already_streamed='{"city": "',
            is_complete=False,
        )
        assert result == "ap"

    @pytest.mark.parametrize(
        "cur_args_json,prev_args_json,already,expected",
        [
            # Round 1: first real diff after holdback
            ('{"key": "v"}', '{"ke"}', "", '{"ke'),
            # Round 2: prefix includes the stable closing quote of "key"
            ('{"key": "va"}', '{"key"}', '{"ke', 'y"'),
            # Round 3: colon, space, opening quote, and start of value
            ('{"key": "valu"}', '{"key": "v"}', '{"key"', ': "v'),
            # Round 4: more of the value streams through
            ('{"key": "value"}', '{"key": "valu"}', '{"key": "v', "alu"),
        ],
        ids=["round1", "round2", "round3", "round4"],
    )
    def test_progressive_streaming(
        self, cur_args_json, prev_args_json, already, expected
    ):
        """Simulate multiple rounds of progressive streaming, where each
        round's already_streamed is the accumulation of prior diffs."""
        result = compute_streamed_args_delta(
            cur_args_json=cur_args_json,
            prev_args_json=prev_args_json,
            already_streamed=already,
            is_complete=False,
        )
        assert result == expected
