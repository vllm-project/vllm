from typing import Sequence
import pytest

from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup

# Mock the parser so we don't need a real model or tokenizer
class MockParser(BaseThinkingReasoningParser):
    def __init__(self):
        self.start_token_id = 1
        self.end_token_id = 2
        self.start_token = "<think>"
        self.end_token = "</think>"

@pytest.mark.skip_global_cleanup
def test_stop_buffer_desync():
    """
    Tests that the streaming reasoning parser does not leak reasoning tags
    into the content stream when a stop buffer holds back text.
    Reproduces bug #51641.
    """
    parser = MockParser()

    # Round 1: Model generates "</think> Hello" but stop buffer holds back text
    previous_text = "<think>I have a plan"
    delta_text = "</"
    current_text = previous_text + delta_text
    previous_token_ids = [1, 10, 11] # <think>, I, have a plan
    delta_token_ids = [12, 2, 13] # </, </think>, Hello
    current_token_ids = previous_token_ids + delta_token_ids

    res1 = parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text=delta_text,
        previous_token_ids=previous_token_ids,
        current_token_ids=current_token_ids,
        delta_token_ids=delta_token_ids
    )

    # In Round 1, text hasn't caught up to token IDs, so it should still be in reasoning.
    assert res1 is not None
    assert res1.reasoning == "</"
    assert res1.content is None

    # Round 2: Stop buffer releases the rest of the text
    previous_text = current_text
    delta_text = "think> Hello"
    current_text = previous_text + delta_text
    previous_token_ids = current_token_ids
    delta_token_ids = []
    current_token_ids = previous_token_ids + delta_token_ids

    res2 = parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text=delta_text,
        previous_token_ids=previous_token_ids,
        current_token_ids=current_token_ids,
        delta_token_ids=delta_token_ids
    )

    # In Round 2, the end token string is finally complete!
    assert res2 is not None
    assert res2.reasoning is None, f"Expected None reasoning, got {res2.reasoning}"
    # If the bug was present, this would be 'think> Hello' and fail the assertion
    assert res2.content == " Hello", f"Expected ' Hello', got {res2.content}"
