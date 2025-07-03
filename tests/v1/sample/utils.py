# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from enum import Enum
from typing import NamedTuple, Optional

import regex as re
import torch

from vllm import CompletionOutput
from vllm.utils import make_tensor_with_pad
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata


class BatchLogprobsComposition(Enum):
    """Types of logprobs configs to include in test batch"""
    NONE = 0
    SAMPLE = 1
    PROMPT = 2
    SAMPLE_PROMPT = 3


BatchLogprobsSpecType = list[tuple[Optional[int], Optional[int]]]


def get_test_batch(
    batch_logprobs_composition: BatchLogprobsComposition
) -> BatchLogprobsSpecType:
    """Generate logprobs configs for a batch of requests
    
    A given request's logprobs configuration is (1) num_sample_logprobs and (2)
    num_prompt_logprobs. The batch logprobs configuration is the list of request
    logprobs configs.

    batch_logprobs_composition == NONE yields a batch with no sample or prompt
    logprobs

    batch_logprobs_composition == SAMPLE yields a batch with some requests
    configured for sample logprobs only, and others configured for no logprobs

    batch_logprobs_composition == PROMPT yields a batch with some requests
    configured for prompt logprobs only, and others configured for no logprobs

    batch_logprobs_composition == SAMPLE_PROMPT yields a batch with some
    requests configured for sample logprobs and prompt logprobs, some configured
    for only sample logprobs or only prompt logprobs, and some configured for
    no logprobs

    Args:
      batch_logprobs_composition: types of logprobs configs to include in batch

    Returns:

      list of (Optional[num_sample_logprobs], Optional[num_prompt_logprobs])
      tuples
    """
    if batch_logprobs_composition == BatchLogprobsComposition.NONE:
        # No requests with sample or prompt logprobs
        return [(None, None)]
    elif batch_logprobs_composition == BatchLogprobsComposition.SAMPLE:
        # Requests requiring sample logprobs or no logprobs
        return [
            (None, None),
            (0, None),
            (5, None),
            (3, None),
        ]
    elif batch_logprobs_composition == BatchLogprobsComposition.PROMPT:
        # Requests requiring prompt logprobs or no logprobs
        return [
            (None, None),
            (None, 0),
            (None, 6),
            (None, 5),
        ]
    elif batch_logprobs_composition == BatchLogprobsComposition.SAMPLE_PROMPT:
        # Requests requiring either no logprobs, just
        # sample logprobs, just prompt logprobs, or
        # both sample and prompt logprobs
        return [
            (None, None),
            (0, None),
            (5, None),
            (3, None),
            (0, 3),
            (6, 0),
            (6, 3),
            (None, 6),
            (None, 5),
            (None, 0),
        ]
    else:
        raise ValueError("Invalid logprobs batch configuration for test.")


def assert_incr_detok_str_matches_non_incr_detok_str(
    incremental_detokenization_str: str,
    non_incremental_detokenization_str: str,
    msg: str,
) -> None:
    """Compare incrementally detok. text to non-incrementally detok. text
    
    Fail if the strings mismatch after non-alphanumeric characters are stripped
    out.

    Rationale: incremental detokenization in the text generation process allows
    the tokenizer to adjust the next token text output based on the token's
    context in the string. However, logprobs detokenization detokenizes each
    token individually, and the resultant strings may include some
    non-alphanumeric placeholder characters where there could be i.e.
    whitespace. So, this function compares only the alphanumeric text
    between two strings and fails if there is a mismatch, which helps
    with validating logprobs detokenization.

    Args:
      incremental_detokenization_str: incrementally-detokenized generated text
      non_incremental_detokenization_str: non-incrementally-detokenized logprob
                                          tokens
      msg: error message if `assert` fails
    """
    rgx = r'[^a-zA-Z0-9]+'
    assert (re.sub(rgx, '', incremental_detokenization_str) == re.sub(
        rgx, '', non_incremental_detokenization_str)), (msg)


def compute_correct_cumulative_logprob(
        completion_output: CompletionOutput) -> float:
    """Compute known-good value for evaluating cumulative logprob
    
    Args:
      completion_output: completion output from engine

    Returns:
      Known-good cumulative logprob value
    """
    token_ids = completion_output.token_ids
    logprobs = completion_output.logprobs
    assert logprobs is not None
    return sum([lp[tok_id].logprob for tok_id, lp in zip(token_ids, logprobs)])


def create_fake_logits(batch_size: int, vocab_size: int) -> torch.Tensor:
    fake_logits = torch.full((batch_size, vocab_size), 1e-2, dtype=torch.float)
    return fake_logits


def create_penalty_tensor(batch_size: int, penalty_value: float,
                          device: torch.device) -> torch.Tensor:
    return torch.full((batch_size, ),
                      fill_value=penalty_value,
                      dtype=torch.float,
                      device=device)


def create_prompt_tokens_tensor(
    prompt_token_ids: list[list[int]],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    return make_tensor_with_pad(
        prompt_token_ids,
        pad=vocab_size,
        device=device,
        dtype=torch.int64,
        pin_memory=False,
    )


class LogitsprocsTestFakes(NamedTuple):
    """Wraps fake data structures to support testing"""
    logits: torch.Tensor
    sampling_metadata: SamplingMetadata

    def get_logitsprocs_by_cls(
        self,
        cls: type[LogitsProcessor],
    ) -> Iterator[LogitsProcessor]:
        """Yield logits processors of a specific class.
        
        Args:
          cls: :class:`LogitsProcessor` subclass

        Returns:
          Iterator over logits processors
        """
        return (lp for lp in self.sampling_metadata.logitsprocs.all
                if isinstance(lp, cls))

    def get_logitsprocs(self) -> Iterator[LogitsProcessor]:
        """Iterator over all logits processors."""
        return self.sampling_metadata.logitsprocs.all


def fake_update_logitsprocs_state(
    test_fakes: LogitsprocsTestFakes,
    batch_update: BatchUpdate,
) -> None:
    """Imitate logits processors persistent batch state update
    in engine core"""
    for logitproc in test_fakes.get_logitsprocs():
        logitproc.update_state(batch_update)


def fake_apply_logitsprocs(
    test_fakes: LogitsprocsTestFakes,
    slice_indices: list[int],
) -> torch.Tensor:
    """Imitate application of logits processors in engine core"""
    logits = test_fakes.logits[torch.tensor(slice_indices,
                                            dtype=torch.long)].clone()
    for processor in test_fakes.get_logitsprocs():
        logits = processor.apply(logits)
    return logits
