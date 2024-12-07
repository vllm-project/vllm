"""Engine test utils"""
import random
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from transformers.tokenization_utils import PreTrainedTokenizer

from vllm.outputs import RequestOutput
from vllm.v1.engine.detokenizer import DetokenizerRequest

random.seed(42)

def _create_random_top_logprob_vector(
    num_logprobs: int,
    lower: float,
    upper: float,
) -> npt.NDArray:
    return np.random.rand(num_logprobs) * (upper - lower) + lower


def _create_random_top_logprob_matrix(
    shape: Tuple,
    lower: float,
    upper: float,
) -> npt.NDArray:
    return np.random.rand(*shape) * (upper - lower) + lower


def _create_random_top_token_vector(
    num_logprobs: int,
    lower: int,
    upper: int,
    sampled_token_ids: Optional[npt.NDArray],
    adjust_num_logprobs: bool,
) -> npt.NDArray:
    choice_list = list(range(lower, upper))
    res = np.random.choice(choice_list, (num_logprobs +
                                         (1 if adjust_num_logprobs else 0), ),
                           replace=False)
    if sampled_token_ids is not None:
        res[-1] = sampled_token_ids
    return res


def _create_random_top_token_matrix(
    shape: Tuple,
    lower: int,
    upper: int,
    sampled_token_ids: Optional[npt.NDArray],
    adjust_num_logprobs: bool,
) -> npt.NDArray:
    choice_list = list(range(lower, upper))
    res = np.random.choice(choice_list, (shape[0], shape[1] +
                                         (1 if adjust_num_logprobs else 0)),
                           replace=False)
    if sampled_token_ids is not None:
        res[:, -1] = sampled_token_ids
    return res


def _generate_dummy_sample_logprobs(
    sampled_tokens_list: List,
    num_logprobs: int,
    tokenizer: PreTrainedTokenizer,
) -> List[Tuple[npt.NDArray, npt.NDArray]]:
    res = []
    for sampled_token_id in sampled_tokens_list:
        num_logprobs_adjustment = random.choice([0, 1])
        res.append(
            (_create_random_top_logprob_vector(
                num_logprobs + num_logprobs_adjustment, -100, 0),
             _create_random_top_token_vector(num_logprobs, 0,
                                             len(tokenizer.vocab) - 1,
                                             np.array([sampled_token_id]),
                                             num_logprobs_adjustment > 0)))
    return res


def _generate_dummy_prompt_logprobs(
    tokens_list: List,
    num_logprobs: int,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[npt.NDArray, npt.NDArray]:
    num_tok = len(tokens_list)
    return (_create_random_top_logprob_matrix((num_tok, num_logprobs), -100,
                                              0),
            _create_random_top_token_matrix((num_tok, num_logprobs), 0,
                                            len(tokenizer.vocab) - 1, None,
                                            False))


def _decode_token(
    tok_id: int,
    tokenizer: PreTrainedTokenizer,
) -> str:
    return tokenizer.convert_ids_to_tokens([tok_id],
                                           skip_special_tokens=False)[0]


def _validate_requests_logprobs(requests: List[DetokenizerRequest],
                                request_outputs: List[RequestOutput],
                                tokenizer: PreTrainedTokenizer,
) -> None:
    # Validate logprob detokenization
    for req, req_out in zip(requests, request_outputs):
        if req.logprobs is not None and req.logprobs > 0:
            for comp in req_out.outputs:
                for lp_dict in comp.logprobs:
                    for tok_id, lp in lp_dict.items():
                        assert lp.decoded_token == _decode_token(
                            tok_id,
                            tokenizer), "sample logprob decoded token mismatch"

        if req.prompt_logprobs is not None and req.prompt_logprobs > 0 and len(
                req_out.prompt_logprobs) > 0:
            # Validate prompt logprobs
            assert req_out.prompt_logprobs[0] is None
            for plp_dict in req_out.prompt_logprobs[1:]:
                for tok_id, plp in plp_dict.items():
                    assert plp.decoded_token == _decode_token(
                        tok_id,
                        tokenizer), "prompt logprob decoded token mismatch"