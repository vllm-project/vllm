"""Engine test utils"""
import random
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from transformers.tokenization_utils import PreTrainedTokenizer

from vllm.outputs import RequestOutput
from vllm.v1.engine.detokenizer import DetokenizerRequest

random.seed(42)


def _create_random_top_logprob_test_vector(
    num_logprobs: int,
    lower: float,
    upper: float,
) -> npt.NDArray:
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
      1D length-`num_logprobs` np array of float logprob values
    """
    return np.random.rand(num_logprobs) * (upper - lower) + lower


def _create_random_top_logprob_test_matrix(
    shape: Tuple,
    lower: float,
    upper: float,
) -> npt.NDArray:
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
      2D num_tokens x num_logprobs np array of float logprob values
    """
    return np.random.rand(*shape) * (upper - lower) + lower


def _create_random_top_token_test_vector(
    num_logprobs: int,
    lower: int,
    upper: int,
    sampled_token_id: int,
    adjust_num_logprobs: bool,
) -> npt.NDArray:
    """Create a random vector of top logprob token indices

    Use to create fake sample logprobs for testing. The sampled token
    ID must always be one of the top logprobs, which this dummy test
    vector generator enforces. OpenAI API
    compatible engines must be able to return an addition sample
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
      1D length-x np array of token ids where x is
      `num_logprobs+1` if `adjust_num_logprobs` and
      `num_logprobs` otherwise
    """
    choice_list = list(range(lower, upper))
    res = np.random.choice(choice_list, (num_logprobs +
                                         (1 if adjust_num_logprobs else 0), ),
                           replace=False)
    res[-1] = sampled_token_id
    return res


def _create_random_top_token_test_matrix(
    shape: Tuple,
    lower: int,
    upper: int,
) -> npt.NDArray:
    """Create a random matrix of top logprob token indices

    Use to create fake prompt logprobs for testing.

    Token ids are generated randomly and sampled without
    replacement.

    Args:
      shape: (num_tokens,num_logprobs) tuple representing
             matrix shape
      lower: lower range of token ids
      upper: upper range of token ids

    Returns:
      2D num_tokens x num_logprobs np array of token ids
    """
    choice_list = list(range(lower, upper))
    res = np.random.choice(choice_list, (shape[0], shape[1]), replace=False)
    return res


def generate_dummy_sample_logprobs(
    sampled_tokens_list: List,
    num_logprobs: int,
    tokenizer: PreTrainedTokenizer,
) -> List[Tuple[npt.NDArray, npt.NDArray]]:
    """Generate dummy sample logprobs

    Generate a test data structure which imitates the list of sample logprobs
    which would be assembled in the engine core during decode phase.

    Args:
      sampled_tokens_list: list of sampled tokens
      num_logprobs: return `num_logprobs` or `num_logprobs+1` logprobs per token
      tokenizer: model tokenizer to use for detokenization

    Returns
      List of (logprobs vector, top token ids vector) np array tuples; each pair
      of vectors have the same length which is either `num_logprobs` or
      `num_logprobs+1`
    """
    res = []
    for sampled_token_id in sampled_tokens_list:
        num_logprobs_adjustment = random.choice([0, 1])
        res.append((_create_random_top_logprob_test_vector(
            num_logprobs + num_logprobs_adjustment, -100, 0),
                    _create_random_top_token_test_vector(
                        num_logprobs, 0,
                        len(tokenizer.vocab) - 1, sampled_token_id,
                        num_logprobs_adjustment > 0)))
    return res


def generate_dummy_prompt_logprobs(
    prompt_tokens_list: List,
    num_logprobs: int,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Generate dummy prompt logprobs

    Generate a test data structure which imitates the np arrays of prompt
    logprobs which would be assembled in the engine core during chunked
    prefill.

    Args:
      prompt_tokens_list: list of prompt tokens
      num_logprobs: return `num_logprobs` logprobs per token
      tokenizer: model tokenizer to use for detokenization

    Returns
      Single Tuple of (logprobs matrix, top token ids matrix) np arrays,
      where both matrices have dimensions
      num_prompt_tokens x num_logprobs
    """
    num_prompt_tokens = len(prompt_tokens_list)
    return (_create_random_top_logprob_test_matrix(
        (num_prompt_tokens, num_logprobs), -100, 0),
            _create_random_top_token_test_matrix(
                (num_prompt_tokens, num_logprobs), 0,
                len(tokenizer.vocab) - 1))


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
    return tokenizer.convert_ids_to_tokens([tok_id],
                                           skip_special_tokens=False)[0]


def validate_requests_logprobs(
    requests: List[DetokenizerRequest],
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
        if req.logprobs is not None and req.logprobs > 0:
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

        if req.prompt_logprobs is not None and req.prompt_logprobs > 0 and len(
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
