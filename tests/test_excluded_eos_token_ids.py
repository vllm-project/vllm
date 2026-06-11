# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for SamplingParams.excluded_eos_token_ids.

When tool calling is active on Harmony models, the <|call|> token (200012)
is a delimiter between parallel tool calls, not an end-of-turn signal.
It must be excluded from stop_token_ids so the model can generate past it
and emit multiple tool calls in a single turn.
"""

from vllm.sampling_params import SamplingParams

CALL_TOKEN = 200012
RETURN_TOKEN = 200002
PAD_TOKEN = 199999
PRIMARY_EOS = 100257


def _make_generation_config(*eos_ids: int) -> dict:
    return {"eos_token_id": list(eos_ids)}


class TestExcludedEosTokenIds:
    def test_no_exclusion_adds_all_eos_ids(self):
        """Without exclusion, all eos_token_ids from generation_config
        are added to stop_token_ids."""
        sp = SamplingParams.from_optional()
        sp.update_from_generation_config(
            _make_generation_config(CALL_TOKEN, RETURN_TOKEN, PAD_TOKEN),
            eos_token_id=PRIMARY_EOS,
        )
        assert CALL_TOKEN in sp.stop_token_ids
        assert RETURN_TOKEN in sp.stop_token_ids
        assert PAD_TOKEN in sp.stop_token_ids

    def test_exclusion_removes_call_token(self):
        """With excluded_eos_token_ids={CALL_TOKEN}, the <|call|> token
        should NOT appear in stop_token_ids after update."""
        sp = SamplingParams.from_optional()
        sp.excluded_eos_token_ids = {CALL_TOKEN}
        sp.update_from_generation_config(
            _make_generation_config(CALL_TOKEN, RETURN_TOKEN, PAD_TOKEN),
            eos_token_id=PRIMARY_EOS,
        )
        assert CALL_TOKEN not in sp.stop_token_ids
        assert RETURN_TOKEN in sp.stop_token_ids
        assert PAD_TOKEN in sp.stop_token_ids

    def test_exclusion_does_not_remove_primary_eos(self):
        """The primary eos_token_id is handled separately and should not
        be affected by excluded_eos_token_ids."""
        sp = SamplingParams.from_optional()
        sp.excluded_eos_token_ids = {PRIMARY_EOS}
        sp.update_from_generation_config(
            _make_generation_config(PRIMARY_EOS, RETURN_TOKEN),
            eos_token_id=PRIMARY_EOS,
        )
        assert sp.eos_token_id == PRIMARY_EOS

    def test_exclusion_with_preexisting_stop_tokens(self):
        """Per-request stop_token_ids should be preserved; only the
        excluded eos tokens from generation_config are dropped."""
        user_stop_token = 42
        sp = SamplingParams.from_optional(stop_token_ids=[user_stop_token])
        sp.excluded_eos_token_ids = {CALL_TOKEN}
        sp.update_from_generation_config(
            _make_generation_config(CALL_TOKEN, RETURN_TOKEN),
            eos_token_id=PRIMARY_EOS,
        )
        assert user_stop_token in sp.stop_token_ids
        assert CALL_TOKEN not in sp.stop_token_ids
        assert RETURN_TOKEN in sp.stop_token_ids

    def test_no_exclusion_field_is_backward_compatible(self):
        """When excluded_eos_token_ids is None (default), behavior is
        unchanged from before this feature was added."""
        sp = SamplingParams.from_optional()
        assert sp.excluded_eos_token_ids is None
        sp.update_from_generation_config(
            _make_generation_config(CALL_TOKEN, RETURN_TOKEN),
            eos_token_id=PRIMARY_EOS,
        )
        assert CALL_TOKEN in sp.stop_token_ids
        assert RETURN_TOKEN in sp.stop_token_ids

    def test_empty_exclusion_set_is_noop(self):
        """An empty exclusion set should behave the same as None."""
        sp = SamplingParams.from_optional()
        sp.excluded_eos_token_ids = set()
        sp.update_from_generation_config(
            _make_generation_config(CALL_TOKEN, RETURN_TOKEN),
            eos_token_id=PRIMARY_EOS,
        )
        assert CALL_TOKEN in sp.stop_token_ids
        assert RETURN_TOKEN in sp.stop_token_ids
