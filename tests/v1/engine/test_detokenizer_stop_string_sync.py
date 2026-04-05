# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression test for issue #36830: delta_text and delta_token_ids
desync with stop sequences.

This test verifies that when stop strings are used, the delta_text and
delta_token_ids outputs remain synchronized - i.e., the number of characters
in delta_text corresponds to the tokens in delta_token_ids.

The fix ensures that when stop buffers hold back text to check for stop strings,
the corresponding token IDs are ALSO held back, maintaining synchronization.
"""

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer


def _make_request(
    stop: list[str] | None = None,
    include_stop_str_in_output: bool = False,
) -> EngineCoreRequest:
    """Create a minimal EngineCoreRequest for testing."""
    return EngineCoreRequest(
        request_id="test",
        external_req_id="test-ext",
        prompt_token_ids=[],
        mm_features=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(
            skip_special_tokens=False,
            spaces_between_special_tokens=True,
            stop=stop,
            include_stop_str_in_output=include_stop_str_in_output,
            detokenize=True,
        ),
        pooling_params=None,
    )


class SimpleTestDetokenizer(BaseIncrementalDetokenizer):
    """Simple test detokenizer that maps token IDs to text without a real tokenizer."""

    def __init__(self, request: EngineCoreRequest, text_map: dict[int, str]):
        super().__init__(request)
        self.text_map = text_map

    def decode_next(self, next_token_id: int) -> str:
        return self.text_map.get(next_token_id, f"<unk{next_token_id}>")


class SimpleTestDetokenizerWithPrompt(SimpleTestDetokenizer):
    """Simulates SlowIncrementalDetokenizer by prepending prompt tokens."""

    def __init__(
        self,
        request: EngineCoreRequest,
        text_map: dict[int, str],
        prompt_token_ids: list[int],
    ):
        super().__init__(request, text_map)
        self._prompt_len = len(prompt_token_ids)
        # Prepend prompt tokens like SlowIncrementalDetokenizer does.
        self.token_ids: list[int] = list(prompt_token_ids) + self.token_ids

    @property
    def _output_token_id_start(self) -> int:
        return self._prompt_len

    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self._prompt_len


def test_released_token_count_without_stop_buffer():
    """Test _released_token_count when there's no stop buffer."""
    request = _make_request()
    text_map = {1: "Hello", 2: " ", 3: "world"}
    detokenizer = SimpleTestDetokenizer(request, text_map)

    detokenizer.update([1], stop_terminated=False)
    detokenizer.update([2], stop_terminated=False)
    detokenizer.update([3], stop_terminated=False)

    released = detokenizer._released_token_count(finished=True)
    assert released == 3


def test_released_token_count_with_stop_buffer():
    """Test _released_token_count when stop buffer holds back tokens."""
    request = _make_request(stop=["world"])
    text_map = {1: "Hello", 2: " ", 3: "world", 4: "!"}
    detokenizer = SimpleTestDetokenizer(request, text_map)

    assert detokenizer.stop_buffer_length == 4

    # Add tokens without triggering stop string
    detokenizer.update([1], stop_terminated=False)  # "Hello"
    detokenizer.update([2], stop_terminated=False)  # "Hello "
    detokenizer.update([4], stop_terminated=False)  # "Hello !"

    # Token offsets: [5, 6, 7]
    # When finished, buffer_length = 0, so all tokens released
    assert detokenizer._released_token_count(finished=True) == 3

    # When not finished, buffer_length = 4
    # text_length = 7 - 4 = 3
    # No token has offset <= 3, so 0 released
    assert detokenizer._released_token_count(finished=False) == 0


def test_get_next_output_token_ids_uses_released_count():
    """Test that get_next_output_token_ids returns only released tokens."""
    request = _make_request(stop=["xyz"])
    text_map = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e"}
    detokenizer = SimpleTestDetokenizer(request, text_map)

    for token_id in [1, 2, 3, 4, 5]:
        detokenizer.update([token_id], stop_terminated=False)

    assert len(detokenizer.token_ids) == 5

    released = detokenizer._released_token_count(finished=False)
    assert released < 5

    token_ids = detokenizer.get_next_output_token_ids(finished=False, delta=False)
    assert len(token_ids) == released

    token_ids_finished = detokenizer.get_next_output_token_ids(
        finished=True, delta=False
    )
    assert len(token_ids_finished) == 5


def test_token_offset_tracking():
    """Test that _token_text_offsets tracks cumulative text length."""
    request = _make_request()
    text_map = {1: "abc", 2: "def", 3: "g"}
    detokenizer = SimpleTestDetokenizer(request, text_map)

    detokenizer.update([1], stop_terminated=False)
    assert detokenizer._token_text_offsets == [3]

    detokenizer.update([2], stop_terminated=False)
    assert detokenizer._token_text_offsets == [3, 6]

    detokenizer.update([3], stop_terminated=False)
    assert detokenizer._token_text_offsets == [3, 6, 7]

    assert detokenizer._released_token_count(finished=True) == 3


def test_delta_text_and_token_ids_stay_in_sync():
    """End-to-end test: delta_text and delta_token_ids stay synchronized.

    This is the core regression test for issue #36830. Text is released at
    character granularity while tokens are atomic, so during streaming the
    token-decoded text is always a prefix of the cumulative delta text
    (tokens never outpace text). On finish, both must be equal.

    Before the fix, token IDs were NOT held back by the stop buffer at all,
    so delta_token_ids would race far ahead of delta_text.
    """
    request = _make_request(stop=["stop"])
    # stop_buffer_length = len("stop") - 1 = 3
    text_map = {10: "He", 20: "ll", 30: "o ", 40: "wor", 50: "ld"}
    detokenizer = SimpleTestDetokenizer(request, text_map)

    all_delta_text = ""
    all_delta_token_ids: list[int] = []

    for token_id in [10, 20, 30, 40, 50]:
        detokenizer.update([token_id], stop_terminated=False)
        dt = detokenizer.get_next_output_text(finished=False, delta=True)
        dtids = detokenizer.get_next_output_token_ids(finished=False, delta=True)
        all_delta_text += dt
        all_delta_token_ids.extend(dtids)

        # Token-decoded text must be a prefix of cumulative delta text.
        # Tokens are held back until their full text clears the buffer,
        # so they never outpace the released text.
        token_text = "".join(text_map[tid] for tid in all_delta_token_ids)
        assert all_delta_text.startswith(token_text), (
            f"Desync after token {token_id}: "
            f"delta_text={all_delta_text!r} does not start with "
            f"token_text={token_text!r}"
        )

    # On finish, remaining buffered text/tokens are flushed.
    dt = detokenizer.get_next_output_text(finished=True, delta=True)
    dtids = detokenizer.get_next_output_token_ids(finished=True, delta=True)
    all_delta_text += dt
    all_delta_token_ids.extend(dtids)

    # After finish, both must be exactly equal.
    expected_text = "".join(text_map[tid] for tid in all_delta_token_ids)
    assert all_delta_text == expected_text
    assert all_delta_token_ids == [10, 20, 30, 40, 50]


def test_prompt_prefix_not_included_in_output_token_ids():
    """Test that prompt tokens are excluded from get_next_output_token_ids.

    SlowIncrementalDetokenizer prepends prompt tokens to token_ids.
    The output must only contain generated tokens, not the prompt.
    """
    request = _make_request(stop=["zz"])
    text_map = {101: "x", 102: "y", 103: "z"}
    prompt_ids = [900, 901, 902]  # 3 prompt tokens
    detokenizer = SimpleTestDetokenizerWithPrompt(request, text_map, prompt_ids)

    for token_id in [101, 102, 103]:
        detokenizer.update([token_id], stop_terminated=False)

    # Non-delta, finished: should return only output tokens, not prompt
    token_ids = detokenizer.get_next_output_token_ids(finished=True, delta=False)
    assert token_ids == [101, 102, 103]
    assert 900 not in token_ids

    # Non-delta, not finished: should still exclude prompt, respect buffer
    token_ids_buffered = detokenizer.get_next_output_token_ids(
        finished=False, delta=False
    )
    assert all(tid not in token_ids_buffered for tid in prompt_ids)


def test_prompt_prefix_excluded_in_delta_mode():
    """Test delta mode also excludes prompt tokens."""
    request = _make_request()  # no stop string
    text_map = {10: "a", 20: "b"}
    prompt_ids = [900, 901]
    detokenizer = SimpleTestDetokenizerWithPrompt(request, text_map, prompt_ids)

    detokenizer.update([10], stop_terminated=False)
    dtids = detokenizer.get_next_output_token_ids(finished=False, delta=True)
    assert dtids == [10]

    detokenizer.update([20], stop_terminated=False)
    dtids = detokenizer.get_next_output_token_ids(finished=False, delta=True)
    assert dtids == [20]
