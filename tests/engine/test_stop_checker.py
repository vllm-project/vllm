# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.reasoning import ReasoningParser
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence, SequenceStatus

REASONING_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


class MockReasoningParser(ReasoningParser):
    """Mock reasoning parser for testing purposes."""

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 reasoning_active: bool = False):
        super().__init__(tokenizer)
        self.reasoning_active = reasoning_active

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return not self.reasoning_active

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids


class MockSequence(Sequence):
    """Mock sequence for testing purposes."""

    def __init__(self, token_ids, output_text="test_output", eos_token_id=0):
        self.token_ids = token_ids
        self.output_text = output_text
        self.eos_token_id = eos_token_id
        self.status = SequenceStatus.RUNNING
        self.stop_reason = None

    def get_token_ids(self):
        return self.token_ids

    def get_last_token_id(self):
        return self.token_ids[-1] if self.token_ids else None

    def get_len(self):
        return len(self.token_ids)

    def get_output_len(self):
        return len(self.token_ids) - 1  # Simulating prompt + outputs


@pytest.fixture
def deepseek_r1_qwen_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


@pytest.fixture
def stop_checker():
    return StopChecker(max_model_len=10,
                       get_tokenizer_for_seq=deepseek_r1_qwen_tokenizer)


@pytest.fixture
def stop_checker_with_reasoner():
    reasoner = MockReasoningParser(deepseek_r1_qwen_tokenizer)
    return StopChecker(max_model_len=10,
                       get_tokenizer_for_seq=deepseek_r1_qwen_tokenizer,
                       reasoner=reasoner)


def test_eos_token_stopping(stop_checker):
    """Test sequence stopping when EOS token is encountered."""
    seq = MockSequence(token_ids=[1, 2, 0], eos_token_id=0)
    sampling_params = SamplingParams()

    stop_checker.maybe_stop_sequence(seq,
                                     new_char_count=1,
                                     sampling_params=sampling_params)

    assert seq.status == SequenceStatus.FINISHED_STOPPED


def test_ignore_eos(stop_checker):
    """Test sequence continuing when EOS token is ignored."""
    seq = MockSequence(token_ids=[1, 2, 0], eos_token_id=0)
    sampling_params = SamplingParams(ignore_eos=True)

    stop_checker.maybe_stop_sequence(seq,
                                     new_char_count=1,
                                     sampling_params=sampling_params)

    assert seq.status == SequenceStatus.RUNNING


def test_min_tokens(stop_checker):
    """Test min_tokens prevents early stopping."""
    seq = MockSequence(token_ids=[1, 2, 0], eos_token_id=0)
    sampling_params = SamplingParams(min_tokens=3)

    stop_checker.maybe_stop_sequence(seq,
                                     new_char_count=1,
                                     sampling_params=sampling_params)

    assert seq.status == SequenceStatus.RUNNING


def test_stop_token_ids(stop_checker):
    """Test sequence stopping with custom stop token IDs."""
    seq = MockSequence(token_ids=[1, 2, 3], eos_token_id=0)
    sampling_params = SamplingParams(stop_token_ids=[3])

    stop_checker.maybe_stop_sequence(seq,
                                     new_char_count=1,
                                     sampling_params=sampling_params)

    assert seq.status == SequenceStatus.FINISHED_STOPPED
    assert seq.stop_reason == 3


def test_stop_strings(stop_checker):
    """Test sequence stopping with stop strings."""
    seq = MockSequence(token_ids=[1, 2, 3],
                       output_text="test output with STOP",
                       eos_token_id=0)
    sampling_params = SamplingParams(stop=["STOP"])

    stop_checker.maybe_stop_sequence(seq,
                                     new_char_count=1,
                                     sampling_params=sampling_params)

    assert seq.status == SequenceStatus.FINISHED_STOPPED
    assert seq.stop_reason == "STOP"
    assert "STOP" not in seq.output_text  # Default behavior removes stop string


def test_include_stop_str_in_output(stop_checker):
    """Test keeping stop strings in output."""
    seq = MockSequence(token_ids=[1, 2, 3],
                       output_text="test output with STOP",
                       eos_token_id=0)
    sampling_params = SamplingParams(stop=["STOP"],
                                     include_stop_str_in_output=True)

    stop_checker.maybe_stop_sequence(seq,
                                     new_char_count=5,
                                     sampling_params=sampling_params)

    assert seq.status == SequenceStatus.FINISHED_STOPPED
    assert "STOP" in seq.output_text


def test_max_tokens(stop_checker):
    """Test sequence stopping at max_tokens."""
    seq = MockSequence(token_ids=[1, 2, 3], eos_token_id=0)
    sampling_params = SamplingParams(max_tokens=2)

    stop_checker.maybe_stop_sequence(seq,
                                     new_char_count=1,
                                     sampling_params=sampling_params)

    assert seq.status == SequenceStatus.FINISHED_LENGTH_CAPPED


def test_max_model_len(stop_checker):
    """Test sequence stopping at max_model_len."""
    seq = MockSequence(token_ids=list(range(11)),
                       eos_token_id=0)  # 11 tokens, max is 10
    sampling_params = SamplingParams()

    stop_checker.maybe_stop_sequence(seq,
                                     new_char_count=1,
                                     sampling_params=sampling_params)

    assert seq.status == SequenceStatus.FINISHED_LENGTH_CAPPED


def test_reasoning_skip_stops(stop_checker_with_reasoner):
    """Test that stop tokens and strings are ignored during reasoning."""
    # Set reasoning_active to True to simulate being in reasoning mode
    stop_checker_with_reasoner.reasoner.reasoning_active = True

    # Test with stop token
    seq = MockSequence(token_ids=[1, 2, 3], eos_token_id=0)
    sampling_params = SamplingParams(stop_token_ids=[3])

    stop_checker_with_reasoner.maybe_stop_sequence(
        seq, new_char_count=1, sampling_params=sampling_params)
    assert seq.status == SequenceStatus.RUNNING

    # Test with stop string
    seq = MockSequence(token_ids=[1, 2, 3], output_text="test STOP")
    sampling_params = SamplingParams(stop=["STOP"])

    stop_checker_with_reasoner.maybe_stop_sequence(
        seq, new_char_count=4, sampling_params=sampling_params)
    assert seq.status == SequenceStatus.RUNNING

    # But EOS token still stops the sequence
    seq = MockSequence(token_ids=[1, 2, 0], eos_token_id=0)
    sampling_params = SamplingParams()

    stop_checker_with_reasoner.maybe_stop_sequence(
        seq, new_char_count=1, sampling_params=sampling_params)
    assert seq.status == SequenceStatus.FINISHED_STOPPED


def test_reasoning_end_enables_stops(stop_checker_with_reasoner):
    """Test that stop tokens work after reasoning ends."""
    # Set reasoning_active to False to simulate being out of reasoning mode
    stop_checker_with_reasoner.reasoner.reasoning_active = False

    # Test with stop token
    seq = MockSequence(token_ids=[1, 2, 3], eos_token_id=0)
    sampling_params = SamplingParams(stop_token_ids=[3])

    stop_checker_with_reasoner.maybe_stop_sequence(
        seq, new_char_count=1, sampling_params=sampling_params)
    assert seq.status == SequenceStatus.FINISHED_STOPPED

    # Test with stop string
    seq = MockSequence(token_ids=[1, 2, 3], output_text="test STOP")
    sampling_params = SamplingParams(stop=["STOP"])

    stop_checker_with_reasoner.maybe_stop_sequence(
        seq, new_char_count=4, sampling_params=sampling_params)
    assert seq.status == SequenceStatus.FINISHED_STOPPED
