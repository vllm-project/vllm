# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality
from vllm.v1.sample.logits_processor.whisper import (
    WHISPER_TIMESTAMP_RULES_KEY,
    WhisperTimestampLogitsProcessor,
)

EOS_TOKEN_ID = 2
NO_TIMESTAMPS_TOKEN_ID = 5
TIMESTAMP_BEGIN = 6
VOCAB_SIZE = 10


def _sampling_params(enabled: bool = True) -> SamplingParams:
    extra_args = None
    if enabled:
        extra_args = {
            WHISPER_TIMESTAMP_RULES_KEY: {
                "eos_token_id": EOS_TOKEN_ID,
                "no_timestamps_token_id": NO_TIMESTAMPS_TOKEN_ID,
                "max_initial_timestamp_index": 1,
                "begin_index": 0,
            }
        }
    return SamplingParams(extra_args=extra_args)


def _processor(output_token_ids: list[int]) -> WhisperTimestampLogitsProcessor:
    processor = WhisperTimestampLogitsProcessor(None, torch.device("cpu"), False)
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, _sampling_params(), [], output_token_ids)],
            moved=[],
        )
    )
    return processor


def test_initial_token_is_a_bounded_timestamp():
    logits = _processor([]).apply(torch.zeros(1, VOCAB_SIZE))

    assert torch.isneginf(logits[0, :TIMESTAMP_BEGIN]).all()
    assert torch.isfinite(logits[0, TIMESTAMP_BEGIN : TIMESTAMP_BEGIN + 2]).all()
    assert torch.isneginf(logits[0, TIMESTAMP_BEGIN + 2 :]).all()


def test_timestamp_pairs_alternate_with_text():
    after_opening_timestamp = _processor([TIMESTAMP_BEGIN]).apply(
        torch.zeros(1, VOCAB_SIZE)
    )
    assert torch.isfinite(after_opening_timestamp[0, :NO_TIMESTAMPS_TOKEN_ID]).all()
    assert torch.isneginf(after_opening_timestamp[0, NO_TIMESTAMPS_TOKEN_ID])
    assert torch.isneginf(after_opening_timestamp[0, TIMESTAMP_BEGIN:]).all()

    after_closing_timestamp = _processor([TIMESTAMP_BEGIN, 3, TIMESTAMP_BEGIN + 1])
    logits = torch.zeros(1, VOCAB_SIZE)
    logits[0, EOS_TOKEN_ID] = 10
    logits = after_closing_timestamp.apply(logits)
    assert torch.isneginf(logits[0, :EOS_TOKEN_ID]).all()
    assert torch.isfinite(logits[0, EOS_TOKEN_ID])
    assert torch.isfinite(logits[0, TIMESTAMP_BEGIN + 1])


def test_timestamps_are_monotonic():
    logits = _processor([TIMESTAMP_BEGIN, 3]).apply(torch.zeros(1, VOCAB_SIZE))

    assert torch.isneginf(logits[0, TIMESTAMP_BEGIN])
    assert torch.isfinite(logits[0, TIMESTAMP_BEGIN + 1])


def test_timestamp_probability_mass_can_force_a_timestamp():
    logits = torch.zeros(1, VOCAB_SIZE)
    logits[0, TIMESTAMP_BEGIN + 1 :] = 1

    processed = _processor([TIMESTAMP_BEGIN, 3]).apply(logits)

    assert torch.isneginf(processed[0, :TIMESTAMP_BEGIN]).all()
    assert torch.isfinite(processed[0, TIMESTAMP_BEGIN + 1 :]).all()


def test_rules_are_request_scoped_and_follow_batch_moves():
    output_token_ids = [TIMESTAMP_BEGIN]
    processor = WhisperTimestampLogitsProcessor(None, torch.device("cpu"), False)
    processor.update_state(
        BatchUpdate(
            batch_size=2,
            removed=[],
            added=[
                (0, _sampling_params(), [], output_token_ids),
                (1, _sampling_params(enabled=False), [], []),
            ],
            moved=[],
        )
    )

    original = torch.zeros(2, VOCAB_SIZE)
    processed = processor.apply(original.clone())
    assert torch.isneginf(processed[0, TIMESTAMP_BEGIN:]).all()
    assert torch.equal(processed[1], original[1])

    processor.update_state(
        BatchUpdate(
            batch_size=2,
            removed=[],
            added=[],
            moved=[(0, 1, MoveDirectionality.UNIDIRECTIONAL)],
        )
    )
    moved = processor.apply(original.clone())
    assert torch.equal(moved[0], original[0])
    assert torch.isneginf(moved[1, TIMESTAMP_BEGIN:]).all()


def test_rules_survive_batch_slot_replacement():
    processor = _processor([TIMESTAMP_BEGIN])
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[0],
            added=[(0, _sampling_params(), [], [])],
            moved=[],
        )
    )

    logits = processor.apply(torch.zeros(1, VOCAB_SIZE))

    assert torch.isneginf(logits[0, :TIMESTAMP_BEGIN]).all()
    assert torch.isfinite(logits[0, TIMESTAMP_BEGIN : TIMESTAMP_BEGIN + 2]).all()


def test_rules_include_generated_tokens_appended_to_beam_prompt():
    sampling_params = _sampling_params()
    sampling_params.extra_args[WHISPER_TIMESTAMP_RULES_KEY]["begin_index"] = 2
    processor = WhisperTimestampLogitsProcessor(None, torch.device("cpu"), False)
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, sampling_params, [3, 4, TIMESTAMP_BEGIN], [])],
            moved=[],
        )
    )

    logits = processor.apply(torch.zeros(1, VOCAB_SIZE))

    assert torch.isfinite(logits[0, :NO_TIMESTAMPS_TOKEN_ID]).all()
    assert torch.isneginf(logits[0, TIMESTAMP_BEGIN:]).all()
