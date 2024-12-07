import random
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.detokenizer import Detokenizer, DetokenizerRequest

random.seed(42)
NUM_SAMPLE_LOGPROBS = 5
NUM_PROMPT_LOGPROBS = 7

TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


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
                                request_outputs: List[RequestOutput]):
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


FULL_STRINGS = [
    "My name is Robert from Neural Magic and I love working on vLLM so much!",
    "Red Hat is the best open source company by far across Linux, K8s, and AI.",
    "Nick is the name of my brother in addition to my colleague from Red Hat.",
]

STOP_STRINGS = ["I love working on", "company by far", "brother in"]

FULL_TOKENS = [tokenizer(text).input_ids for text in FULL_STRINGS]
PROMPT_LEN = 5
PROMPT_TOKENS = [
    tokenizer(text).input_ids[:PROMPT_LEN] for text in FULL_STRINGS
]
PROMPT_LOGPROBS_RAW: List[Tuple[npt.NDArray, npt.NDArray]] = [
    _generate_dummy_prompt_logprobs(tokens_list=tokens_list,
                                    num_logprobs=NUM_PROMPT_LOGPROBS,
                                    tokenizer=tokenizer)
    for tokens_list in PROMPT_TOKENS
]
# PROMPT_LOGPROBS = [
#     _new_logprobs_detokenized(logprobs=logprobs, tokenizer=tokenizer)
#     for logprobs in PROMPT_LOGPROBS_RAW
# ]
GENERATION_TOKENS = [
    tokenizer(text).input_ids[PROMPT_LEN:] for text in FULL_STRINGS
]
GENERATION_LOGPROBS_RAW = [
    _generate_dummy_sample_logprobs(sampled_tokens_list=tokens_list,
                                    num_logprobs=NUM_SAMPLE_LOGPROBS,
                                    tokenizer=tokenizer)
    for tokens_list in GENERATION_TOKENS
]
# GENERATION_LOGPROBS = [
#     _new_logprobs_detokenized(logprobs=logprobs, tokenizer=tokenizer)
#     for logprobs in GENERATION_LOGPROBS_RAW
# ]
PROMPT_STRINGS = [
    tokenizer.decode(prompt_tokens,
                     skip_special_tokens=True,
                     tokenizer=tokenizer) for prompt_tokens in PROMPT_TOKENS
]
PROMPT_STRINGS_LEN = [len(prompt_string) for prompt_string in PROMPT_STRINGS]
GENERATION_STRINGS = [
    text[prompt_len:]
    for text, prompt_len in zip(FULL_STRINGS, PROMPT_STRINGS_LEN)
]


class MockEngineCore:
    """Mock outputs form premade tokens lists."""

    def __init__(
        self,
        generated_tokens_list: List[List[int]],
        prompt_tokens_list: List[List[int]],
        generated_logprobs_raw: Optional[List[List[Tuple[npt.NDArray,
                                                         npt.NDArray]]]],
        prompt_logprobs_raw: Optional[List[Tuple[npt.NDArray, npt.NDArray]]],
    ) -> None:
        self.generated_tokens_list = generated_tokens_list
        self.prompt_tokens_list = prompt_tokens_list
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
        for req_idx, generated_token_ids in enumerate(
                self.generated_tokens_list):
            if len(generated_token_ids) > token_idx:
                if do_logprobs:
                    assert self.generated_logprobs_raw is not None
                    logprobs = [
                        self.generated_logprobs_raw[req_idx][token_idx]
                    ]
                else:
                    logprobs = None
                if do_prompt_logprobs:
                    if self.current_idx == 0:
                        assert self.prompt_logprobs_raw is not None
                        prompt_logprobs = self.prompt_logprobs_raw[req_idx][0]
                        prompt_logprobs_token_ids = self.prompt_logprobs_raw[
                            req_idx][1]
                    else:
                        (prompt_logprobs, prompt_logprobs_token_ids) = ([], [])
                else:
                    (prompt_logprobs, prompt_logprobs_token_ids) = (None, None)
                output = EngineCoreOutput(
                    request_id=f"request-{req_idx}",
                    new_token_ids=[generated_token_ids[token_idx]],
                    finished=False,
                    logprobs=logprobs,
                    prompt_logprobs=prompt_logprobs,
                    prompt_logprobs_token_ids=prompt_logprobs_token_ids,
                )
                if token_idx == len(generated_token_ids) - 1:
                    output.finished = True
                    output.finish_reason = "stopped"
                outputs.append(output)

        self.current_idx += 1
        return outputs


@pytest.mark.parametrize(
    "request_output_kind",
    [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
def test_incremental_detokenization(request_output_kind: RequestOutputKind):
    detokenizer = Detokenizer(TOKENIZER_NAME)
    engine_core = MockEngineCore(GENERATION_TOKENS)

    # Make N requests.
    requests = [
        DetokenizerRequest(
            request_id=f"request-{idx}",
            prompt=prompt,
            prompt_token_ids=prompt_tokens,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            output_kind=request_output_kind,
            stop=[],
            include_stop_str_in_output=False,
        ) for idx, (
            prompt,
            prompt_tokens) in enumerate(zip(PROMPT_STRINGS, PROMPT_TOKENS))
    ]

    # Add requests to the detokenizer.
    for request in requests:
        detokenizer.add_request(request)

    gen_strings = {}
    gen_tokens = {}
    while True:
        # Mock output from the EngineCore.
        outputs = engine_core.get_outputs()
        if len(outputs) == 0:
            break

        # Step the Detokenizer.
        request_outputs, requests_to_abort = detokenizer.step(outputs)
        assert len(requests_to_abort) == 0

        # Update tracking.
        for request_output in request_outputs:
            request_id = request_output.request_id
            new_text = request_output.outputs[0].text
            new_tokens = request_output.outputs[0].token_ids
            if request_id not in gen_strings:
                gen_strings[request_id] = new_text
                gen_tokens[request_id] = new_tokens
            else:
                gen_strings[request_id] += new_text
                gen_tokens[request_id].extend(new_tokens)

    # Confirmed tracked values matches what we expected.
    for idx, (ref_gen_str, ref_gen_toks) in enumerate(
            zip(GENERATION_STRINGS, GENERATION_TOKENS)):
        gen_str = gen_strings[f"request-{idx}"]
        gen_toks = gen_tokens[f"request-{idx}"]

        assert gen_str == ref_gen_str, f"{gen_str=}, {ref_gen_str=}"
        assert gen_toks == ref_gen_toks, f"{gen_toks=}, {ref_gen_toks=}"

    assert detokenizer.get_num_unfinished_requests() == 0
    assert not detokenizer.has_unfinished_requests()


@pytest.mark.parametrize("include_stop_str_in_output", [True, False])
@pytest.mark.parametrize("logprobs,prompt_logprobs",
                         [(None, None), (NUM_SAMPLE_LOGPROBS, None),
                          (None, NUM_PROMPT_LOGPROBS),
                          (NUM_SAMPLE_LOGPROBS, NUM_PROMPT_LOGPROBS)])
def test_stop_string(
    include_stop_str_in_output: bool,
    logprobs: Optional[int],
    prompt_logprobs: Optional[int],
) -> None:
    do_generated_logprobs = logprobs is not None
    do_prompt_logprobs = prompt_logprobs is not None
    detokenizer = Detokenizer(TOKENIZER_NAME)
    engine_core = MockEngineCore(generated_tokens_list=GENERATION_TOKENS,
                                 prompt_tokens_list=PROMPT_TOKENS,
                                 generated_logprobs_raw=GENERATION_LOGPROBS_RAW
                                 if do_generated_logprobs else None,
                                 prompt_logprobs_raw=PROMPT_LOGPROBS_RAW
                                 if do_prompt_logprobs else None)

    # Make N requests.
    requests = [
        DetokenizerRequest(
            request_id=f"request-{idx}",
            prompt=prompt,
            prompt_token_ids=prompt_tokens,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            output_kind=RequestOutputKind.DELTA,
            stop=STOP_STRINGS,
            include_stop_str_in_output=include_stop_str_in_output,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
        ) for idx, (
            prompt,
            prompt_tokens) in enumerate(zip(PROMPT_STRINGS, PROMPT_TOKENS))
    ]

    # Add requests to the detokenizer.
    for request in requests:
        detokenizer.add_request(request)

    gen_strings = {}
    aborted = []
    i = 0
    while True:
        # Mock output from the EngineCore.
        outputs = engine_core.get_outputs()
        if len(outputs) == 0:
            break

        # Step the Detokenizer.
        request_outputs, requests_to_abort = detokenizer.step(outputs)
        for request_output in request_outputs:
            # If aborted, we should not get a request output.
            assert request_output.request_id not in aborted
        aborted.extend(requests_to_abort)

        # Validate logprob detokenization
        _validate_requests_logprobs(requests, request_outputs)

        # Update tracking.
        for request_output in request_outputs:
            if request_output.finished:
                assert request_output.outputs[0].finish_reason == "stop"

            request_id = request_output.request_id
            new_text = request_output.outputs[0].text
            if request_id not in gen_strings:
                gen_strings[request_id] = new_text
            else:
                gen_strings[request_id] += new_text
        i += 1

    # Confirmed tracked values matches what we expected.
    for idx, (ref_gen_str,
              stop_str) in enumerate(zip(GENERATION_STRINGS, STOP_STRINGS)):

        # Request should be aborted.
        request_id = f"request-{idx}"
        assert request_id in aborted

        # Collected values that were generated.
        gen_str = gen_strings[request_id]

        # Construct reference strings.
        stop_str_idx = ref_gen_str.find(stop_str)
        ref_str_exc_stop = ref_gen_str[:stop_str_idx]
        ref_str_inc_stop = ref_gen_str[:stop_str_idx] + stop_str

        if include_stop_str_in_output:
            assert gen_str == ref_str_inc_stop, (
                f"{gen_str=}, {ref_str_inc_stop=}")
        else:
            assert gen_str == ref_str_exc_stop, (
                f"{gen_str=}, {ref_str_exc_stop=}")

    assert detokenizer.get_num_unfinished_requests() == 0
    assert not detokenizer.has_unfinished_requests()
