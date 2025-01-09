"""Engine test utils"""
import random
from typing import List, Tuple

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from vllm.outputs import RequestOutput
from vllm.v1.engine import EngineCoreRequest

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


def _decode_token(
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
    return tokenizer.batch_decode([tok_id])[0]


def validate_requests_logprobs(
    requests: List[EngineCoreRequest],
    request_outputs: List[RequestOutput],
    tokenizer: PreTrainedTokenizer,
) -> None:
    """Validate detokenizer logprobs output

    For each sample or prompt logprob, the logprob's
    `decoded_token` member should match the result of
    detokenizing the logprob's token id.

    Fails upon mismatch.

    Requires that `requests` and `request_outputs` have
    the same ordering with respect to requests (i.e.
    the data structure pertaining to a given request
    id appears at the same index in both lists and
    both lists have the same length.)

    Args:
      requests: list of detokenizer input requests
      request_outputs: list of detokenizer outputs
    """
    for req, req_out in zip(requests, request_outputs):
        logprobs = req.sampling_params.logprobs
        prompt_logprobs = req.sampling_params.prompt_logprobs
        if logprobs is not None and logprobs > 0:
            # Validate sample logprobs
            for comp in req_out.outputs:
                # For each completion
                for lp_dict in comp.logprobs:
                    # For each sampled token offset
                    for tok_id, lp in lp_dict.items():
                        # For each top logprob,
                        # compare each `decoded_token` to the result
                        # of decoding the logprob's token id
                        assert lp.decoded_token == _decode_token(
                            tok_id,
                            tokenizer), "sample logprob decoded token mismatch"

        if prompt_logprobs is not None and prompt_logprobs > 0 and len(
                req_out.prompt_logprobs) > 0:
            # Validate prompt logprobs
            assert req_out.prompt_logprobs[
                0] is None  # always true for prompt logprobs
            for plp_dict in req_out.prompt_logprobs[1:]:
                # For each prompt token offset
                assert plp_dict is not None
                for tok_id, plp in plp_dict.items():
                    # For each top logprob,
                    # compare each `decoded_token` to the result
                    # of decoding the logprob's token id
                    assert plp.decoded_token == _decode_token(
                        tok_id,
                        tokenizer), "prompt logprob decoded token mismatch"
