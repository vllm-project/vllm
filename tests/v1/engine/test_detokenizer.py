from typing import List

import pytest
from transformers import AutoTokenizer

from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.detokenizer import Detokenizer, DetokenizerRequest

TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

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
GENERATION_TOKENS = [
    tokenizer(text).input_ids[PROMPT_LEN:] for text in FULL_STRINGS
]
PROMPT_STRINGS = [
    tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    for prompt_tokens in PROMPT_TOKENS
]
PROMPT_STRINGS_LEN = [len(prompt_string) for prompt_string in PROMPT_STRINGS]
GENERATION_STRINGS = [
    text[prompt_len:]
    for text, prompt_len in zip(FULL_STRINGS, PROMPT_STRINGS_LEN)
]


class MockEngineCore:
    """Mock outputs form premade tokens lists."""

    def __init__(self, tokens_list: List[List[int]]):
        self.tokens_list = tokens_list
        self.current_idx = 0

    def get_outputs(self) -> List[EngineCoreOutput]:
        token_idx = self.current_idx
        self.current_idx += 1

        outputs = []
        for req_idx, token_ids in enumerate(self.tokens_list):
            if len(token_ids) > token_idx:
                output = EngineCoreOutput(request_id=f"request-{req_idx}",
                                          new_token_ids=[token_ids[token_idx]],
                                          finished=False)
                if token_idx == len(token_ids) - 1:
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
def test_stop_string(include_stop_str_in_output: bool):
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
            output_kind=RequestOutputKind.DELTA,
            stop=STOP_STRINGS,
            include_stop_str_in_output=include_stop_str_in_output,
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
