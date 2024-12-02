import random
from typing import Dict, List, Optional, Union

import pytest
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from vllm.sampling_params import RequestOutputKind
from vllm.sequence import Logprob, PromptLogprobs, SampleLogprobs
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.detokenizer import Detokenizer, DetokenizerRequest

random.seed(42)
NUM_SAMPLE_LOGPROBS = 5
NUM_PROMPT_LOGPROBS = 7

TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _duplicate_logprob_with_decode(
    logprob: Logprob,
    token_id: int,
    tokenizer: PreTrainedTokenizer,
) -> Logprob:
    return Logprob(logprob.logprob, logprob.rank,
                   tokenizer.decode(token_id, skip_special_tokens=True))


def _generate_dummy_single_logprob(
    num_logprobs: int,
    is_sample_logprobs: bool,
    tokenizer: PreTrainedTokenizer,
) -> Dict[int, Logprob]:
    adjusted_num_logprobs = (num_logprobs + random.choice([0, 1])
                             if is_sample_logprobs else num_logprobs)
    return {
        random.randint(0,
                       len(tokenizer.vocab) - 1):
        Logprob(random.uniform(-100, 0), idx, None)
        for idx in range(adjusted_num_logprobs)
    }


def _generate_dummy_logprobs(
    tokens_list: List,
    num_logprobs: int,
    is_sample_logprobs: bool,
    tokenizer: PreTrainedTokenizer,
) -> Union[SampleLogprobs, PromptLogprobs]:
    return [
        _generate_dummy_single_logprob(num_logprobs, is_sample_logprobs,
                                       tokenizer) for _ in tokens_list
    ]


def _new_logprobs_detokenized(
    logprobs: Union[SampleLogprobs, PromptLogprobs],
    tokenizer: PreTrainedTokenizer,
) -> Union[SampleLogprobs, PromptLogprobs]:
    return [{
        tok_id: _duplicate_logprob_with_decode(lp, tok_id, tokenizer)
        for tok_id, lp in lp_dict.items()
    } for lp_dict in logprobs]


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
PROMPT_LOGPROBS_RAW = [
    _generate_dummy_logprobs(tokens_list=tokens_list,
                             num_logprobs=NUM_PROMPT_LOGPROBS,
                             is_sample_logprobs=False,
                             tokenizer=tokenizer)
    for tokens_list in PROMPT_TOKENS
]
PROMPT_LOGPROBS = [
    _new_logprobs_detokenized(logprobs=logprobs, tokenizer=tokenizer)
    for logprobs in PROMPT_LOGPROBS_RAW
]
GENERATION_TOKENS = [
    tokenizer(text).input_ids[PROMPT_LEN:] for text in FULL_STRINGS
]
GENERATION_LOGPROBS_RAW = [
    _generate_dummy_logprobs(tokens_list=tokens_list,
                             num_logprobs=NUM_SAMPLE_LOGPROBS,
                             is_sample_logprobs=True,
                             tokenizer=tokenizer)
    for tokens_list in GENERATION_TOKENS
]
GENERATION_LOGPROBS = [
    _new_logprobs_detokenized(logprobs=logprobs, tokenizer=tokenizer)
    for logprobs in GENERATION_LOGPROBS_RAW
]
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
        generated_logprobs_raw: Optional[SampleLogprobs],
        prompt_logprobs_raw: Optional[PromptLogprobs],
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
        self.current_idx += 1

        outputs = []
        for req_idx, (generated_token_ids, prompt_token_ids) in enumerate(
                zip(self.generated_tokens_list, self.prompt_tokens_list)):
            if len(generated_token_ids) > token_idx:
                output = EngineCoreOutput(
                    request_id=f"request-{req_idx}",
                    new_token_ids=[generated_token_ids[token_idx]],
                    finished=False,
                    logprobs=[self.generated_logprobs_raw[req_idx][token_idx]]
                    if do_logprobs else None,
                    prompt_logprobs=self.prompt_logprobs_raw[req_idx]
                    if do_prompt_logprobs else None,
                    prompt_logprobs_token_ids=prompt_token_ids[req_idx]
                    if do_prompt_logprobs else None,
                )
                if token_idx == len(generated_token_ids) - 1:
                    output.finished = True
                    output.finish_reason = "stopped"
                outputs.append(output)

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
