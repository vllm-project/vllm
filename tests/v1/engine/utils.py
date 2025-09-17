# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

GeneralTokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# Number of sample logprobs to request when testing sample logprobs
NUM_SAMPLE_LOGPROBS_UNDER_TEST = 5
# Number of prompt logprobs to request when testing prompt logprobs
NUM_PROMPT_LOGPROBS_UNDER_TEST = 7

TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"

FULL_STRINGS = [
    "My name is Robert from Neural Magic and I love working on vLLM so much!",
    "Red Hat is the best open source company by far across Linux, K8s, and AI.",
    "Nick is the name of my brother in addition to my colleague from Red Hat.",
]
STOP_STRINGS = ["I love working on", "company by far", "brother in"]
PROMPT_LEN = 5

random.seed(42)


def _create_random_top_logprob_test_vector(
    num_logprobs: int,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Create a random vector of top logprob float values.
    
    Use to create fake sample logprobs for testing.

    Note that a real production scenario would require
    logprobs to be sorted in descending order, something
    which is omitted in this function.

    Args:
      num_logprobs: number of top logprobs
      lower: lower range of logprob float values
      upper: upper range of logprob float values

    Returns:
      1D length-`num_logprobs` torch Tensor of float logprob values
    """
    return torch.rand(num_logprobs) * (upper - lower) + lower


def _create_random_top_logprob_test_matrix(
    shape: tuple,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Create a random matrix of top logprob float values.
    
    Use to create fake prompt logprobs for testing.

    Note that a real production scenario would require
    logprobs to be sorted in descending order along rows,
    something which is omitted in this function.

    Args:
      shape: (num_tokens,num_logprobs) tuple representing
             matrix shape
      lower: lower range of logprob float values
      upper: upper range of logprob float values

    Returns:
      2D num_tokens x num_logprobs torch Tensor of float logprob values
    """
    return torch.rand(*shape) * (upper - lower) + lower


def _create_random_top_token_test_vector(
        num_logprobs: int,
        lower: int,
        upper: int,
        sampled_token_id: int,
        adjust_num_logprobs: bool = True) -> tuple[torch.Tensor, int]:
    """Create a random vector of top logprob token indices

    Use to create fake sample logprobs for testing. The sampled token
    ID must always be one of the top logprobs, which this dummy test
    vector generator enforces. OpenAI API
    compatible engines must be able to return an additional sample
    logprob for the sampled token if the sampled token was not
    among the top sample logprobs; `adjust_num_logprobs` emulates
    this behavior by increasing the vector length by 1 if
    `adjust_num_logprobs` is set.

    Args:
      num_logprobs: number of top logprobs
      lower: lower range of token ids
      upper: upper range of token ids
      sampled_token_id: the token actually sampled
      adjust_num_logprobs: if True, emulate situation where sampled
                           token logprob must be injected into top
                           logprobs

    Returns:
      1D length-x torch Tensor of token ids where x is
      `num_logprobs+1` if `adjust_num_logprobs` and
      `num_logprobs` otherwise
      sampled_token_rank: the rank of sampled_token_id in the vocab
                          vector when sorted in descending order by
                          logprob
    """

    # Calculate the final number of logprobs required
    total_logprobs = num_logprobs + 1 if adjust_num_logprobs else num_logprobs

    # Generate random indices using torch
    choice_tensor = torch.randperm(upper - lower)[:total_logprobs] + lower

    # Ensure the sampled token ID is included in the tensor
    choice_tensor[0] = sampled_token_id

    # Check if the sampled_token_id occurs in choice_tensor[1:]
    if sampled_token_id in choice_tensor[1:]:
        sampled_token_rank = (choice_tensor[1:] == sampled_token_id).nonzero(
            as_tuple=True)[0].item()
    else:
        # If not found, assign a random int between num_logprobs and 50700
        sampled_token_rank = random.randint(num_logprobs, 50700)

    return choice_tensor, sampled_token_rank


def _create_random_top_token_test_matrix(
    shape: tuple[int, int],
    lower: int,
    upper: int,
    tokens_list: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a random matrix of top logprob token indices

    Use to create fake prompt logprobs for testing.

    Token ids are generated randomly and sampled without
    replacement.

    Args:
      shape: (num_tokens, num_logprobs) tuple representing
             matrix shape
      lower: lower range of token ids
      upper: upper range of token ids

    Returns:
      tuple containing:
      - 2D num_tokens x num_logprobs+1 torch Tensor of token ids
      - 1D tensor of ranks of prompt tokens in their respective
        rows, or random values
    """
    num_elements = shape[0] * shape[1]
    choice_tensor = torch.randperm(upper - lower)[:num_elements] + lower
    matrix = torch.cat(
        (torch.tensor(tokens_list, dtype=torch.int).unsqueeze(-1),
         choice_tensor.view(shape)),
        dim=1)

    # Initialize the tensor for storing the ranks
    prompt_token_ranks = torch.empty(shape[0], dtype=torch.int)

    # Iterate over each row to check presence of
    # tokens_list[rdx] and determine its index
    for rdx in range(shape[0]):
        row = matrix[rdx,
                     1:]  # Skip the first column as it contains the token list
        token_index = (row == tokens_list[rdx]).nonzero(as_tuple=True)[0]
        if token_index.numel() > 0:
            prompt_token_ranks[rdx] = token_index.item()
        else:
            prompt_token_ranks[rdx] = random.randint(shape[1], 50700)

    return matrix, prompt_token_ranks


def decode_token(
    tok_id: int,
    tokenizer: PreTrainedTokenizer,
) -> str:
    """Reproduce the process of detokenizing a token for testing purposes.

    Args:
      tok_id: token id to detokenize
      tokenizer: tokenizer to use for detokenization

    Returns:
      string representation of token
    """
    return tokenizer.convert_ids_to_tokens(tok_id)


def generate_dummy_sample_logprobs(
    sampled_tokens_list: list,
    num_logprobs: int,
    tokenizer: PreTrainedTokenizer,
) -> list[tuple[list[int], list[float], int]]:
    """Generate dummy sample logprobs

    Generate a test data structure which imitates the list of sample logprobs
    which would be assembled in the engine core during decode phase.

    Args:
      sampled_tokens_list: list of sampled tokens
      num_logprobs: return `num_logprobs` or `num_logprobs+1` logprobs per token
      tokenizer: model tokenizer to use for detokenization

    Returns
      list of (top token ids vector, logprobs vector, sampled token rank)
      Python lists tuples; in each tuple the logprobs and top token ids
      vectors have the same length which is either `num_logprobs` or
      `num_logprobs+1`. Sampled token rank is the rank (index+1) of the
      sampled token within the vocab vector when sorted by logprob in
      descending order.
    """
    res = []
    for sampled_token_id in sampled_tokens_list:
        (
            token_vector,
            sampled_token_rank,
        ) = _create_random_top_token_test_vector(num_logprobs, 0,
                                                 len(tokenizer.vocab) - 1,
                                                 sampled_token_id)

        res.append(
            (token_vector,
             _create_random_top_logprob_test_vector(num_logprobs + 1, -100,
                                                    0), sampled_token_rank))

    # Convert tensors in the list tuples to Python lists
    res_list_format = [
        (log_probs_tensor.tolist(), token_ids_tensor.tolist(),
         sampled_token_rank)
        for log_probs_tensor, token_ids_tensor, sampled_token_rank in res
    ]

    return res_list_format


def generate_dummy_prompt_logprobs_tensors(
    prompt_tokens_list: list,
    num_logprobs: int,
    tokenizer: PreTrainedTokenizer,
) -> LogprobsTensors:
    """Generate dummy prompt logprobs tensors

    Generate a test data structure which imitates the torch Tensors of prompt
    logprobs which would be assembled in the engine core during chunked
    prefill.

    Args:
      prompt_tokens_list: list of prompt tokens
      num_logprobs: return `num_logprobs` logprobs per token
      tokenizer: model tokenizer to use for detokenization

    Returns
      Single tuple of (logprobs matrix, top token ids matrix) torch Tensor,
      where both matrices have dimensions
      num_prompt_tokens x num_logprobs
    """
    # For now, assume the whole prompt is processed in one chunk; thus,
    # the number of non-`None` prompt logprobs is `len(prompt_tokens_list)-1`.
    # Prior to injecting `None` at the beginning of prompt logprobs (which
    # happens later in the detokenizer, not here), the prompt logprobs in
    # the ith position are predicting the probability distribution of the
    # prompt token in (i+1)st position. Thus, we concat
    # `prompt_tokens_list[1:]` to the dummy token ids, just as the engine
    # would.
    num_prompt_logprobs = len(prompt_tokens_list) - 1
    (
        token_vector,
        prompt_token_ranks,
    ) = _create_random_top_token_test_matrix(
        (num_prompt_logprobs, num_logprobs), 0,
        len(tokenizer.vocab) - 1, prompt_tokens_list[1:])
    return LogprobsTensors(
        token_vector,
        _create_random_top_logprob_test_matrix(
            (num_prompt_logprobs, num_logprobs + 1), -100, 0),
        prompt_token_ranks)


@dataclass
class DummyOutputProcessorTestVectors:
    """Dummy test vectors for output processor tests"""
    tokenizer: GeneralTokenizerType
    tokenizer_group: TokenizerGroup
    vllm_config: EngineArgs
    full_tokens: list[list[int]]  # Prompt + generated tokens
    prompt_tokens: list[list[int]]
    generation_tokens: list[list[int]]
    # Each request is associated with a tuple of
    # (top tokens, top logprobs, ranks) prompt logprobs tensors
    prompt_logprobs: list[LogprobsTensors]
    # Each request is associated with a sample logprobs; a request's
    # sample logprobs are a list of (top tokens, top logprobs, ranks)
    # sample logprobs tensors at each sequence position
    generation_logprobs: list[list[tuple[list[int], list[float], int]]]
    prompt_strings: list[str]
    prompt_strings_len: list[int]
    generation_strings: list[str]


class MockEngineCore:
    """Mock engine core outputs form premade tokens lists."""

    def __init__(
        self,
        tokens_list: list[list[int]],
        # For each request, for each sampled token offset,
        # a tuple of
        # (list of topk token ids, list of sample logprob vals, rank)
        generated_logprobs_raw: Optional[list[list[tuple[list[int],
                                                         list[float],
                                                         int]]]] = None,
        # For each request, a tuple of
        # (prompt logprob val matrix, prompt logprob tok id matrix);
        # each matrix has dimensions
        # (num prompt toks) x (num prompt logprobs+1)
        prompt_logprobs_raw: Optional[list[LogprobsTensors]] = None,
        eos_token_id: Optional[int] = None,
        stop_token_ids: Optional[list[int]] = None,
        ignore_eos: bool = False,
    ) -> None:
        self.num_requests = len(tokens_list)
        self.tokens_list = tokens_list
        self.current_idx = 0
        self.generated_logprobs_raw = generated_logprobs_raw
        self.do_logprobs = generated_logprobs_raw is not None
        self.prompt_logprobs_raw = prompt_logprobs_raw
        self.do_prompt_logprobs = prompt_logprobs_raw is not None
        self.request_finished = [False for _ in range(self.num_requests)]
        self.eos_token_id = eos_token_id
        self.stop_token_ids = stop_token_ids
        self.ignore_eos = ignore_eos

    def get_outputs(self) -> list[EngineCoreOutput]:
        do_logprobs = self.do_logprobs
        do_prompt_logprobs = self.do_prompt_logprobs
        token_idx = self.current_idx

        outputs = []
        for req_idx, token_ids in enumerate(self.tokens_list):
            if not self.request_finished[req_idx]:
                if do_logprobs:
                    assert self.generated_logprobs_raw is not None
                    (logprobs_token_ids_, logprobs_, sampled_token_ranks_) = (
                        self.generated_logprobs_raw[req_idx][token_idx])
                    logprobs = LogprobsLists(
                        [logprobs_token_ids_],
                        [logprobs_],
                        [sampled_token_ranks_],
                    )
                else:
                    logprobs = None
                if do_prompt_logprobs:
                    if self.current_idx == 0:
                        assert self.prompt_logprobs_raw is not None
                        prompt_logprobs = self.prompt_logprobs_raw[req_idx]
                    else:
                        prompt_logprobs = None
                else:
                    prompt_logprobs = None
                new_token_id = token_ids[token_idx]
                output = EngineCoreOutput(
                    request_id=f"request-{req_idx}",
                    new_token_ids=[new_token_id],
                    new_logprobs=logprobs,
                    new_prompt_logprobs_tensors=prompt_logprobs,
                )
                if token_idx == len(token_ids) - 1:
                    output.finish_reason = FinishReason.LENGTH
                    self.request_finished[req_idx] = True
                if not self.ignore_eos and new_token_id == self.eos_token_id:
                    output.finish_reason = FinishReason.STOP
                    self.request_finished[req_idx] = True
                if new_token_id in (self.stop_token_ids or ()):
                    output.finish_reason = FinishReason.STOP
                    output.stop_reason = new_token_id
                    self.request_finished[req_idx] = True
                outputs.append(output)

        self.current_idx += 1
        return outputs
