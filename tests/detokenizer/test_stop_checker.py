# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from transformers import PreTrainedTokenizer

from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.inputs import token_inputs
from vllm.sampling_params import SamplingParams
from vllm.sequence import Logprob, Sequence, SequenceStatus


def sequence_with_eos(text: str, eos_token: str,
                      eos_token_id: int) -> Sequence:
    """
    Create a Sequence that ends with an EOS token.
    """
    seq = Sequence(
        seq_id=0,
        inputs=token_inputs([]),
        block_size=16,
        eos_token_id=eos_token_id,
    )
    seq.output_text = text + eos_token

    offset = eos_token_id + 1
    for i in range(offset, len(text) + offset):
        seq.append_token_id(token_id=i, logprobs={i: Logprob(0.0)})
    seq.append_token_id(token_id=eos_token_id,
                        logprobs={eos_token_id: Logprob(0.0)})

    seq.status = SequenceStatus.RUNNING

    return seq


@pytest.mark.parametrize(["text_wo_eos", "eos_token", "eos_token_id"], [
    ("This text ends with EOS token", "</s>", 2),
])
@pytest.mark.parametrize("ignore_eos", [True, False])
@pytest.mark.parametrize("include_stop_str_in_output", [True, False])
@pytest.mark.skip_global_cleanup
def test_stop_on_eos_token(text_wo_eos: str, eos_token: str, eos_token_id: int,
                           ignore_eos: bool, include_stop_str_in_output: bool):
    """
    Test the behavior of the StopChecker's maybe_stop_sequence method
    when an EOS token is encountered.

    This test covers:
    - When the EOS token should stop the sequence and be removed from the output
    - When the EOS token should stop the sequence and be included in the output
    - When the EOS token should be ignored, and the sequence continues
    """

    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    get_tokenizer_for_seq = MagicMock(return_value=tokenizer)
    stop_checker = StopChecker(max_model_len=1024,
                               get_tokenizer_for_seq=get_tokenizer_for_seq)

    seq = sequence_with_eos(
        text=text_wo_eos,
        eos_token=eos_token,
        eos_token_id=eos_token_id,
    )
    new_char_count = len(eos_token)

    # Note that `stop` and `stop_token_ids` are not specified
    sampling_params = SamplingParams(
        min_tokens=1,
        ignore_eos=ignore_eos,
        include_stop_str_in_output=include_stop_str_in_output)

    stop_checker.maybe_stop_sequence(
        seq=seq,
        new_char_count=new_char_count,
        sampling_params=sampling_params,
    )

    if ignore_eos:
        assert seq.status == SequenceStatus.RUNNING
        assert seq.output_text == text_wo_eos + eos_token
    elif include_stop_str_in_output:
        assert seq.status == SequenceStatus.FINISHED_STOPPED
        assert seq.output_text == text_wo_eos + eos_token
    else:
        assert seq.status == SequenceStatus.FINISHED_STOPPED
        assert seq.output_text == text_wo_eos
