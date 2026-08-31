# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.reasoning.hunyuan_a13b_reasoning_parser import HunyuanA13BReasoningParser

pytestmark = pytest.mark.skip_global_cleanup

# The parser matches hard-coded token id sequences, so the ids below are the ones it
# looks for rather than anything a tokenizer produces. `extract_reasoning_streaming`
# never touches the tokenizer.
THINK_START_IDS = [14023, 771, 397]  # "<think>\n"
RESPONSE_START_IDS = [198, 524, 27963, 397, 27, 9399, 397]  # "\n</think>\n<answer>\n"
ORDINARY_TOKEN = 5000


class StubTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}


def _feed(parser: HunyuanA13BReasoningParser, token_ids: list[int]) -> list:
    """Stream one token at a time, which is what the parser asserts it receives."""
    messages = []
    previous_ids: list[int] = []
    previous_text = ""
    for token_id in token_ids:
        delta_text = f"<{token_id}>"
        current_ids = previous_ids + [token_id]
        messages.append(
            parser.extract_reasoning_streaming(
                previous_text,
                previous_text + delta_text,
                delta_text,
                previous_ids,
                current_ids,
                [token_id],
            )
        )
        previous_ids = current_ids
        previous_text += delta_text
    return messages


def test_streaming_reaches_the_response_state():
    parser = HunyuanA13BReasoningParser(StubTokenizer())
    _feed(parser, THINK_START_IDS + [ORDINARY_TOKEN] + RESPONSE_START_IDS)
    assert parser.current_state == "response"


@pytest.mark.parametrize("matched", range(1, len(RESPONSE_START_IDS)))
def test_partial_response_start_falls_back_to_reasoning(matched: int):
    """A prefix of the response-start sequence followed by ordinary text is not a
    transition, so the buffered tokens have to come back as reasoning.

    `expected_sequence_side` is `response_start_ids_fast`, one token shorter than
    `response_start_ids`, and it used to be indexed without a length check. Diverging
    after six of the seven tokens therefore raised `IndexError` instead of taking this
    fallback, while every earlier divergence took it.
    """
    parser = HunyuanA13BReasoningParser(StubTokenizer())
    prefix = RESPONSE_START_IDS[:matched] + [ORDINARY_TOKEN]
    token_ids = THINK_START_IDS + [ORDINARY_TOKEN] + prefix
    messages = [message for message in _feed(parser, token_ids) if message is not None]

    assert parser.current_state == "think"
    buffered = messages[-1]
    assert buffered.content is None
    assert buffered.reasoning == "".join(f"<{tid}>" for tid in prefix)
