"""Engine test utils"""
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import (
    BaseTokenizerGroup)
from vllm.v1.engine import EngineCoreOutput

GeneralTokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# Number of sample logprobs to request when testing sample logprobs
NUM_SAMPLE_LOGPROBS = 5
# Number of prompt logprobs to request when testing prompt logprobs
NUM_PROMPT_LOGPROBS = 7

TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

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
    shape: Tuple,
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
) -> torch.Tensor:
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
    """
    # Calculate the final number of logprobs required
    total_logprobs = num_logprobs + 1

    # Generate random indices using torch
    choice_tensor = torch.randperm(upper - lower)[:total_logprobs] + lower

    # Ensure the sampled token ID is included in the tensor
    choice_tensor[0] = sampled_token_id

    return choice_tensor


def _create_random_top_token_test_matrix(
    shape: Tuple[int, int],
    lower: int,
    upper: int,
    tokens_list: List[int],
) -> torch.Tensor:
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
      2D num_tokens x num_logprobs torch Tensor of token ids
    """
    num_elements = shape[0] * shape[1]
    choice_tensor = torch.randperm(upper - lower)[:num_elements] + lower
    return torch.cat((torch.tensor(tokens_list, dtype=torch.int).unsqueeze(-1),
                      choice_tensor.view(shape)),
                     dim=1)


def generate_dummy_sample_logprobs(
    sampled_tokens_list: List,
    num_logprobs: int,
    tokenizer: PreTrainedTokenizer,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate dummy sample logprobs

    Generate a test data structure which imitates the list of sample logprobs
    which would be assembled in the engine core during decode phase.

    Args:
      sampled_tokens_list: list of sampled tokens
      num_logprobs: return `num_logprobs` or `num_logprobs+1` logprobs per token
      tokenizer: model tokenizer to use for detokenization

    Returns
      List of (logprobs vector, top token ids vector) torch Tensor tuples; each
      pair of vectors have the same length which is either `num_logprobs` or
      `num_logprobs+1`
    """
    res = []
    for sampled_token_id in sampled_tokens_list:
        res.append(
            (_create_random_top_logprob_test_vector(num_logprobs + 1, -100, 0),
             _create_random_top_token_test_vector(num_logprobs, 0,
                                                  len(tokenizer.vocab) - 1,
                                                  sampled_token_id)))
    return res


def generate_dummy_prompt_logprobs(
    prompt_tokens_list: List,
    num_logprobs: int,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate dummy prompt logprobs

    Generate a test data structure which imitates the torch Tensors of prompt
    logprobs which would be assembled in the engine core during chunked
    prefill.

    Args:
      prompt_tokens_list: list of prompt tokens
      num_logprobs: return `num_logprobs` logprobs per token
      tokenizer: model tokenizer to use for detokenization

    Returns
      Single Tuple of (logprobs matrix, top token ids matrix) torch Tensor,
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
    return (_create_random_top_logprob_test_matrix(
        (num_prompt_logprobs, num_logprobs + 1), -100, 0),
            _create_random_top_token_test_matrix(
                (num_prompt_logprobs, num_logprobs), 0,
                len(tokenizer.vocab) - 1, prompt_tokens_list[1:]))


@dataclass
class DummyOutputProcessorTestVectors:
    """Dummy test vectors for output processor tests"""
    tokenizer: GeneralTokenizerType
    tokenizer_group: BaseTokenizerGroup
    vllm_config: EngineArgs
    full_tokens: List[List[int]]  # Prompt + generated tokens
    prompt_tokens: List[List[int]]
    generation_tokens: List[List[int]]
    # Each request is associated with a tuple of (top logprobs,top tokens)
    # prompt logprobs tensors
    prompt_logprobs: List[Tuple[torch.Tensor, torch.Tensor]]
    # Each request is associated with a sample logprobs; a request's
    # sample logprobs are a list of (top logprobs,top tokens)
    # sample logprobs tensors at each sequence position
    generation_logprobs: List[List[Tuple[torch.Tensor, torch.Tensor]]]
    prompt_strings: List[str]
    prompt_strings_len: List[int]
    generation_strings: List[str]


class MockEngineCore:
    """Mock engine core outputs form premade tokens lists."""

    def __init__(
        self,
        tokens_list: List[List[int]],
        generated_logprobs_raw: Optional[List[List[Tuple[
            torch.Tensor, torch.Tensor]]]] = None,
        prompt_logprobs_raw: Optional[List[Tuple[torch.Tensor,
                                                 torch.Tensor]]] = None,
    ) -> None:
        self.tokens_list = tokens_list
        self.current_idx = 0
        self.generated_logprobs_raw = generated_logprobs_raw
        self.do_logprobs = generated_logprobs_raw is not None
        self.prompt_logprobs_raw = prompt_logprobs_raw
        self.do_prompt_logprobs = prompt_logprobs_raw is not None

    def get_outputs(self) -> List[EngineCoreOutput]:
        do_logprobs = self.do_logprobs
        do_prompt_logprobs = self.do_prompt_logprobs
        token_idx = self.current_idx

        outputs = []
        for req_idx, token_ids in enumerate(self.tokens_list):
            if len(token_ids) > token_idx:
                if do_logprobs:
                    assert self.generated_logprobs_raw is not None
                    (logprobs, logprobs_token_ids) = (
                        self.generated_logprobs_raw[req_idx][token_idx])
                    logprobs = [logprobs]
                    logprobs_token_ids = [logprobs_token_ids]
                else:
                    logprobs = None
                    logprobs_token_ids = None
                if do_prompt_logprobs:
                    if self.current_idx == 0:
                        assert self.prompt_logprobs_raw is not None
                        prompt_logprobs = self.prompt_logprobs_raw[req_idx][0]
                        prompt_logprobs_token_ids = self.prompt_logprobs_raw[
                            req_idx][1]
                    else:
                        (prompt_logprobs,
                         prompt_logprobs_token_ids) = (torch.empty(0, 0),
                                                       torch.empty(0, 0))
                else:
                    (prompt_logprobs, prompt_logprobs_token_ids) = (None, None)
                output = EngineCoreOutput(
                    request_id=f"request-{req_idx}",
                    new_token_ids=[token_ids[token_idx]],
                    new_logprobs=logprobs,
                    new_logprobs_token_ids=logprobs_token_ids,
                    new_prompt_logprobs=prompt_logprobs,
                    new_prompt_logprobs_token_ids=prompt_logprobs_token_ids,
                    finished=False)
                if token_idx == len(token_ids) - 1:
                    output.finished = True
                    output.finish_reason = "stopped"
                outputs.append(output)

        self.current_idx += 1
        return outputs
