from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import pytest
import torch
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from tests.v1.engine.utils import (generate_dummy_prompt_logprobs,
                                   generate_dummy_sample_logprobs,
                                   validate_requests_logprobs)
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.detokenizer import Detokenizer

# Number of sample logprobs to request when testing sample logprobs
NUM_SAMPLE_LOGPROBS = 5
# Number of prompt logprobs to request when testing prompt logprobs
NUM_PROMPT_LOGPROBS = 7
# Use Mistral instruct tokenizer
TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

FULL_STRINGS = [
    "My name is Robert from Neural Magic and I love working on vLLM so much!",
    "Red Hat is the best open source company by far across Linux, K8s, and AI.",
    "Nick is the name of my brother in addition to my colleague from Red Hat.",
]
STOP_STRINGS = ["I love working on", "company by far", "brother in"]
PROMPT_LEN = 5


@dataclass
class DummyTestVectors:
    """Dummy test vectors for detokenizer tests"""
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
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


@pytest.fixture(scope="module")
def dummy_test_vectors() -> DummyTestVectors:
    """Generate dummy test vectors for detokenizer tests.
    
    Returns:
      DummyTestVectors instance
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    # Tokenize prompts under test & create dummy generated tokens
    prompt_tokens = [
        tokenizer(text).input_ids[:PROMPT_LEN] for text in FULL_STRINGS
    ]
    generation_tokens = [
        tokenizer(text).input_ids[PROMPT_LEN:] for text in FULL_STRINGS
    ]
    # Generate prompt strings
    prompt_strings = [
        tokenizer.decode(prompt_tokens,
                         skip_special_tokens=True,
                         tokenizer=tokenizer)
        for prompt_tokens in prompt_tokens
    ]
    prompt_strings_len = [
        len(prompt_string) for prompt_string in prompt_strings
    ]
    return DummyTestVectors(
        tokenizer=tokenizer,
        full_tokens=[tokenizer(text).input_ids for text in FULL_STRINGS],
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        prompt_strings=prompt_strings,
        prompt_strings_len=prompt_strings_len,
        generation_strings=[
            text[prompt_len:]
            for text, prompt_len in zip(FULL_STRINGS, prompt_strings_len)
        ],
        prompt_logprobs=[
            generate_dummy_prompt_logprobs(prompt_tokens_list=tokens_list,
                                           num_logprobs=NUM_PROMPT_LOGPROBS,
                                           tokenizer=tokenizer)
            for tokens_list in prompt_tokens
        ],
        generation_logprobs=[
            generate_dummy_sample_logprobs(sampled_tokens_list=tokens_list,
                                           num_logprobs=NUM_SAMPLE_LOGPROBS,
                                           tokenizer=tokenizer)
            for tokens_list in generation_tokens
        ])


class MockEngineCore:
    """Mock outputs form premade tokens lists."""

    def __init__(
        self,
        generated_tokens_list: List[List[int]],
        prompt_tokens_list: List[List[int]],
        generated_logprobs_raw: Optional[List[List[Tuple[torch.Tensor,
                                                         torch.Tensor]]]],
        prompt_logprobs_raw: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
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
                    new_token_ids=[generated_token_ids[token_idx]],
                    finished=False,
                    logprobs=logprobs,
                    logprobs_token_ids=logprobs_token_ids,
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
@pytest.mark.parametrize("logprobs,prompt_logprobs",
                         [(None, None), (NUM_SAMPLE_LOGPROBS, None),
                          (None, NUM_PROMPT_LOGPROBS),
                          (NUM_SAMPLE_LOGPROBS, NUM_PROMPT_LOGPROBS)])
def test_incremental_detokenization(
    request_output_kind: RequestOutputKind,
    logprobs: Optional[int],
    prompt_logprobs: Optional[int],
    dummy_test_vectors: DummyTestVectors,
) -> None:
    generation_tokens = dummy_test_vectors.generation_tokens
    prompt_tokens = dummy_test_vectors.prompt_tokens
    # Determine whether sample/prompt logprobs are enabled
    do_generated_logprobs = logprobs is not None
    do_prompt_logprobs = prompt_logprobs is not None
    detokenizer = Detokenizer(TOKENIZER_NAME)
    # Build mock engine core, which emulates sampling & logprobs
    engine_core = MockEngineCore(
        generated_tokens_list=generation_tokens,
        prompt_tokens_list=prompt_tokens,
        generated_logprobs_raw=dummy_test_vectors.generation_logprobs
        if do_generated_logprobs else None,
        prompt_logprobs_raw=dummy_test_vectors.prompt_logprobs
        if do_prompt_logprobs else None)

    # Make N requests.
    requests = [
        EngineCoreRequest(request_id=f"request-{idx}",
                          prompt=prompt,
                          prompt_token_ids=prompt_tokens,
                          arrival_time=0,
                          mm_inputs=None,
                          mm_hashes=None,
                          mm_placeholders=None,
                          eos_token_id=None,
                          lora_request=None,
                          sampling_params=SamplingParams(
                              skip_special_tokens=False,
                              spaces_between_special_tokens=False,
                              output_kind=request_output_kind,
                              stop=[],
                              include_stop_str_in_output=False,
                              logprobs=logprobs,
                              prompt_logprobs=prompt_logprobs))
        for idx, (prompt, prompt_tokens) in enumerate(
            zip(dummy_test_vectors.prompt_strings, prompt_tokens))
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

        # Validate logprob detokenization
        validate_requests_logprobs(requests, request_outputs,
                                   dummy_test_vectors.tokenizer)

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
            zip(dummy_test_vectors.generation_strings, generation_tokens)):
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
    dummy_test_vectors: DummyTestVectors,
) -> None:
    prompt_tokens = dummy_test_vectors.prompt_tokens
    do_generated_logprobs = logprobs is not None
    do_prompt_logprobs = prompt_logprobs is not None
    detokenizer = Detokenizer(TOKENIZER_NAME)
    engine_core = MockEngineCore(
        generated_tokens_list=dummy_test_vectors.generation_tokens,
        prompt_tokens_list=prompt_tokens,
        generated_logprobs_raw=dummy_test_vectors.generation_logprobs
        if do_generated_logprobs else None,
        prompt_logprobs_raw=dummy_test_vectors.prompt_logprobs
        if do_prompt_logprobs else None)

    # Make N requests.
    requests = [
        EngineCoreRequest(
            request_id=f"request-{idx}",
            prompt=prompt,
            prompt_token_ids=prompt_tokens,
            arrival_time=0,
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            eos_token_id=None,
            lora_request=None,
            sampling_params=SamplingParams(
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
                output_kind=RequestOutputKind.DELTA,
                stop=STOP_STRINGS,
                include_stop_str_in_output=include_stop_str_in_output,
                logprobs=logprobs,
                prompt_logprobs=prompt_logprobs,
            )) for idx, (prompt, prompt_tokens) in enumerate(
                zip(dummy_test_vectors.prompt_strings, prompt_tokens))
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
        validate_requests_logprobs(requests, request_outputs,
                                   dummy_test_vectors.tokenizer)

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
    for idx, (ref_gen_str, stop_str) in enumerate(
            zip(dummy_test_vectors.generation_strings, STOP_STRINGS)):

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
