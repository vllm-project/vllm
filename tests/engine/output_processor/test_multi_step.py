# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import MagicMock

import pytest
from transformers import PreTrainedTokenizer

from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.multi_step import MultiStepOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.sampling_params import SamplingParams
from vllm.sequence import (CompletionSequenceGroupOutput, Logprob,
                           SequenceOutput, SequenceStatus)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter

from ...core.utils import create_seq_group


@pytest.mark.parametrize("seq_output_len", [128])
@pytest.mark.parametrize("num_new_tokens", [1, 12])
@pytest.mark.skip_global_cleanup
def test_appends_token_ids(num_new_tokens: int, seq_output_len: int):
    """Verify multi-step decoding appends token ids correctly.

    We append token ids and verify all the token ids were appended correctly.
    Note that ignore_eos=True.
    """
    detokenizer = MagicMock(spec=Detokenizer)
    scheduler = MagicMock(spec=Scheduler)
    stop_checker = MagicMock(spec=StopChecker)
    seq_counter = Counter()

    output_processor = MultiStepOutputProcessor(
        detokenizer=detokenizer,
        scheduler=[scheduler],
        seq_counter=seq_counter,
        get_tokenizer_for_seq=lambda _: mock_tokenizer(),
        stop_checker=stop_checker,
    )

    seq_group = create_seq_group(
        seq_prompt_len=1024,
        seq_output_lens=[seq_output_len],
        sampling_params=SamplingParams(max_tokens=seq_output_len +
                                       num_new_tokens,
                                       ignore_eos=True),
    )

    seq = seq_group.get_seqs()[0]
    seq.status = SequenceStatus.RUNNING

    new_token_ids = list(range(num_new_tokens))

    outputs = [
        CompletionSequenceGroupOutput(
            samples=[
                SequenceOutput(
                    parent_seq_id=seq.seq_id,
                    output_token=output_token,
                    logprobs={output_token: Logprob(0.0)},
                )
            ],
            prompt_logprobs=None,
        ) for output_token in new_token_ids
    ]

    assert seq.get_token_ids()[-len(new_token_ids):] != new_token_ids
    output_processor.process_outputs(seq_group, outputs)
    assert seq.get_token_ids()[-len(new_token_ids):] == new_token_ids


@pytest.mark.parametrize("seq_prompt_len", [1024])
@pytest.mark.parametrize("seq_output_len", [128])
@pytest.mark.parametrize("num_new_tokens", [5, 6, 7, 8])
@pytest.mark.parametrize("max_tokens", [128 + 3])
@pytest.mark.skip_global_cleanup
def test_respects_max_tokens(num_new_tokens: int, seq_prompt_len: int,
                             seq_output_len: int, max_tokens: int):
    """Verify tokens after max_tokens are dropped and not appended to the
    sequence.
    """
    detokenizer = MagicMock(spec=Detokenizer)
    scheduler = MagicMock(spec=Scheduler)
    stop_checker = MagicMock(spec=StopChecker)
    seq_counter = Counter()

    output_processor = MultiStepOutputProcessor(
        detokenizer=detokenizer,
        scheduler=[scheduler],
        seq_counter=seq_counter,
        get_tokenizer_for_seq=lambda _: mock_tokenizer(),
        stop_checker=stop_checker,
    )

    seq_group = create_seq_group(
        seq_prompt_len=seq_prompt_len,
        seq_output_lens=[seq_output_len],
        sampling_params=SamplingParams(max_tokens=max_tokens, ),
    )

    seq = seq_group.get_seqs()[0]
    seq.status = SequenceStatus.RUNNING

    new_token_ids = list(range(num_new_tokens))

    outputs = [
        CompletionSequenceGroupOutput(
            samples=[
                SequenceOutput(
                    parent_seq_id=seq.seq_id,
                    output_token=output_token,
                    logprobs={output_token: Logprob(0.0)},
                )
            ],
            prompt_logprobs=None,
        ) for output_token in new_token_ids
    ]

    assert seq.get_len() == seq_prompt_len + seq_output_len
    output_processor.process_outputs(seq_group, outputs)

    # Expect the processed sequence to not go over max tokens in len.
    assert seq.get_len() == seq_prompt_len + max_tokens

    # Expect the correct tokens were appended.
    expected_appended_tokens = new_token_ids[:max_tokens - seq_output_len]
    assert seq.get_token_ids(
    )[-len(expected_appended_tokens):] == expected_appended_tokens


@pytest.mark.parametrize("seq_prompt_len", [1024])
@pytest.mark.parametrize("seq_output_len", [128])
@pytest.mark.parametrize("num_new_tokens", [12])
@pytest.mark.parametrize("seed", list(range(6)))
@pytest.mark.skip_global_cleanup
def test_respects_eos_token_id(num_new_tokens: int, seq_prompt_len: int,
                               seq_output_len: int, seed: int):
    """Verify the eos token id is included in the sequence, but subsequent
    tokens are dropped (not appended to sequence).
    """
    random.seed(seed)
    detokenizer = MagicMock(spec=Detokenizer)
    scheduler = MagicMock(spec=Scheduler)
    stop_checker = MagicMock(spec=StopChecker)
    seq_counter = Counter()

    eos_token_id = 100

    output_processor = MultiStepOutputProcessor(
        detokenizer=detokenizer,
        scheduler=[scheduler],
        seq_counter=seq_counter,
        get_tokenizer_for_seq=lambda _: mock_tokenizer(eos_token_id),
        stop_checker=stop_checker,
    )

    seq_group = create_seq_group(
        seq_prompt_len=seq_prompt_len,
        seq_output_lens=[seq_output_len],
        sampling_params=SamplingParams(
            # Ensure enough space.
            max_tokens=seq_output_len + num_new_tokens, ),
    )

    seq = seq_group.get_seqs()[0]
    seq.status = SequenceStatus.RUNNING

    new_token_ids = list(range(num_new_tokens))
    assert eos_token_id not in new_token_ids
    eos_index = random.randint(0, len(new_token_ids) - 1)
    new_token_ids[eos_index] = eos_token_id

    outputs = [
        CompletionSequenceGroupOutput(
            samples=[
                SequenceOutput(
                    parent_seq_id=seq.seq_id,
                    output_token=output_token,
                    logprobs={output_token: Logprob(0.0)},
                )
            ],
            prompt_logprobs=None,
        ) for output_token in new_token_ids
    ]

    assert seq.get_len() == seq_prompt_len + seq_output_len
    output_processor.process_outputs(seq_group, outputs)

    # Expect the processed sequence to not go beyond provided eos.
    assert seq.get_len() == seq_prompt_len + seq_output_len + (eos_index + 1)

    # Expect the correct tokens were appended.
    expected_appended_tokens = new_token_ids[:eos_index + 1]
    assert seq.get_token_ids(
    )[-len(expected_appended_tokens):] == expected_appended_tokens


@pytest.mark.parametrize("seq_prompt_len", [1024])
@pytest.mark.parametrize("seq_output_len", [128])
@pytest.mark.parametrize("num_new_tokens", [12])
@pytest.mark.parametrize("seed", list(range(6)))
@pytest.mark.skip_global_cleanup
def test_ignores_eos_token_id(num_new_tokens: int, seq_prompt_len: int,
                              seq_output_len: int, seed: int):
    """When sampling parameters dictate that we should ignore the eos token id,
    ensure all token ids are appended even if the eos token id is emitted.
    """
    random.seed(seed)
    detokenizer = MagicMock(spec=Detokenizer)
    scheduler = MagicMock(spec=Scheduler)
    stop_checker = MagicMock(spec=StopChecker)
    seq_counter = Counter()

    eos_token_id = 100

    output_processor = MultiStepOutputProcessor(
        detokenizer=detokenizer,
        scheduler=[scheduler],
        seq_counter=seq_counter,
        get_tokenizer_for_seq=lambda _: mock_tokenizer(eos_token_id),
        stop_checker=stop_checker,
    )

    seq_group = create_seq_group(
        seq_prompt_len=seq_prompt_len,
        seq_output_lens=[seq_output_len],
        sampling_params=SamplingParams(
            # Ensure enough space.
            max_tokens=seq_output_len + num_new_tokens,
            ignore_eos=True,
        ),
    )

    seq = seq_group.get_seqs()[0]
    seq.status = SequenceStatus.RUNNING

    new_token_ids = list(range(num_new_tokens))
    assert eos_token_id not in new_token_ids
    eos_index = random.randint(0, len(new_token_ids) - 1)
    new_token_ids[eos_index] = eos_token_id

    outputs = [
        CompletionSequenceGroupOutput(
            samples=[
                SequenceOutput(
                    parent_seq_id=seq.seq_id,
                    output_token=output_token,
                    logprobs={output_token: Logprob(0.0)},
                )
            ],
            prompt_logprobs=None,
        ) for output_token in new_token_ids
    ]

    assert seq.get_len() == seq_prompt_len + seq_output_len
    output_processor.process_outputs(seq_group, outputs)

    # Expect the processed sequence to go beyond eos.
    assert seq.get_len() == seq_prompt_len + seq_output_len + num_new_tokens

    # Expect the correct tokens were appended.
    expected_appended_tokens = new_token_ids[:seq_output_len + num_new_tokens -
                                             seq_output_len]
    assert seq.get_token_ids(
    )[-len(expected_appended_tokens):] == expected_appended_tokens


def mock_tokenizer(eos_token_id=1000):
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    tokenizer.eos_token_id = eos_token_id
    return tokenizer
