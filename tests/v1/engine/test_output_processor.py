from typing import Dict, List, Optional

import pytest
import math

from tests.v1.engine.utils import (STOP_STRINGS,
                                   DummyOutputProcessorTestVectors,
                                   MockEngineCore, _decode_token)
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.sequence import PromptLogprobs, SampleLogprobs
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor


@pytest.mark.parametrize(
    "request_output_kind",
    [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
def test_incremental_detokenization(request_output_kind: RequestOutputKind,
                                    dummy_test_vectors):
    output_processor = OutputProcessor(dummy_test_vectors.tokenizer_group,
                                       log_stats=False)
    engine_core = MockEngineCore(
        tokens_list=dummy_test_vectors.generation_tokens)

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
                          )) for idx, (prompt, prompt_tokens) in enumerate(
                              zip(dummy_test_vectors.prompt_strings,
                                  dummy_test_vectors.prompt_tokens))
    ]

    # Add requests to the detokenizer.
    for request in requests:
        output_processor.add_request(request)

    gen_strings = {}
    gen_tokens = {}
    while True:
        # Mock output from the EngineCore.
        outputs = engine_core.get_outputs()
        if len(outputs) == 0:
            break

        # Step the Detokenizer.
        processed_outputs = output_processor.process_outputs(outputs)
        request_outputs = processed_outputs.request_outputs
        requests_to_abort = processed_outputs.reqs_to_abort
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
            zip(dummy_test_vectors.generation_strings,
                dummy_test_vectors.generation_tokens)):
        gen_str = gen_strings[f"request-{idx}"]
        gen_toks = gen_tokens[f"request-{idx}"]

        assert gen_str == ref_gen_str, f"{gen_str=}, {ref_gen_str=}"
        assert gen_toks == ref_gen_toks, f"{gen_toks=}, {ref_gen_toks=}"

    assert output_processor.get_num_unfinished_requests() == 0
    assert not output_processor.has_unfinished_requests()


def _validate_logprobs(
    gen_tokens: Dict[str, List[int]],
    gen_logprobs: Dict[str, Optional[SampleLogprobs]],
    gen_prompt_logprobs: Dict[str, Optional[PromptLogprobs]],
    gen_cumulative_logprob: Dict[str, float],
    dtv: DummyOutputProcessorTestVectors,
    request_id_list: List[str],
    num_sample_logprobs: Optional[int],
    num_prompt_logprobs: Optional[int],
) -> None:
    for req_idx, req_id in enumerate(request_id_list):
        new_tokens = gen_tokens[req_id]
        logprobs = gen_logprobs[req_id]
        prompt_logprobs = gen_prompt_logprobs[req_id]
        cumulative_logprob = gen_cumulative_logprob[req_id]
        prompt_token_ids = dtv.prompt_tokens[req_idx]
        ref_logprobs = dtv.generation_logprobs[req_idx]
        ref_prompt_logprobs = dtv.prompt_logprobs[req_idx]

        if num_sample_logprobs:
            # Validate sample logprobs

            # Require num sampled tokens to match num
            # sampled logprobs - especially important
            # to check since the detokenizer can cause
            # a request to finish early due to a stop
            # string being hit
            assert len(new_tokens) == len(logprobs)

            ref_cumulative_logprob = 0.0
            for sampled_token, pos_logprob_dict in zip(new_tokens, logprobs):
                # For each position in the completion sequence,
                # ensure the actual sampled token is among the
                # logprobs
                assert sampled_token in pos_logprob_dict
                ref_cumulative_logprob += pos_logprob_dict[
                    sampled_token].logprob
                for lp_tok in pos_logprob_dict:
                    # Confirm that sample logprob decoded token matches
                    # the logprob token id at this sequence position
                    decoded_token = pos_logprob_dict[lp_tok].decoded_token
                    ref_decoded_token = _decode_token(lp_tok, dtv.tokenizer)
                    assert decoded_token == ref_decoded_token, (
                        f"Sampled logprob token id {lp_tok} decodes to"
                        f" {ref_decoded_token} but Logprob decoded"
                        f" token is {decoded_token} instead.")

            # Assert that cumulative logprobs are correct
            assert math.isclose(cumulative_logprob, ref_cumulative_logprob)
        else:
            # Sample logprobs disabled for this request
            assert logprobs is None
            assert cumulative_logprob is None

        if num_prompt_logprobs:
            # Validate prompt logprobs

            # Require num prompt tokens to match num
            # prompt logprobs
            assert len(prompt_token_ids) == len(prompt_logprobs)
            assert prompt_logprobs[0] is None

            for prompt_token, pos_logprob_dict in zip(prompt_token_ids[1:],
                                                      prompt_logprobs[1:]):
                # For each position in the prompt sequence,
                # ensure the actual prompt token is among the
                # logprobs
                assert prompt_token in pos_logprob_dict
                for plp_tok in pos_logprob_dict:
                    # Confirm that prompt logprob decoded token matches
                    # the logprob token id at this sequence position
                    decoded_token = pos_logprob_dict[plp_tok].decoded_token
                    ref_decoded_token = _decode_token(plp_tok, dtv.tokenizer)
                    assert decoded_token == ref_decoded_token, (
                        f"Prompt logprob token id {plp_tok} decodes to"
                        f" {ref_decoded_token} but Logprob decoded"
                        f" token is {decoded_token} instead.")
        else:
            # Prompt logprobs disabled for this request
            assert prompt_logprobs is None


@pytest.mark.parametrize(
    "request_output_kind",
    [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
@pytest.mark.parametrize("num_sample_logprobs", [None, 5])
@pytest.mark.parametrize("num_prompt_logprobs", [None, 5])
def test_logprobs_processor(request_output_kind: RequestOutputKind,
                            num_sample_logprobs: Optional[int],
                            num_prompt_logprobs: Optional[int],
                            dummy_test_vectors):
    output_processor = OutputProcessor(dummy_test_vectors.tokenizer_group,
                                       log_stats=False)
    engine_core = MockEngineCore(
        tokens_list=dummy_test_vectors.generation_tokens,
        generated_logprobs_raw=dummy_test_vectors.generation_logprobs
        if num_sample_logprobs else None,
        prompt_logprobs_raw=dummy_test_vectors.prompt_logprobs
        if num_prompt_logprobs else None)

    # Make N requests.
    request_id_list = [
        f"request-{idx}"
        for idx in range(len(dummy_test_vectors.prompt_strings))
    ]
    requests = [
        EngineCoreRequest(request_id=request_id_list[idx],
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
                              logprobs=num_sample_logprobs,
                              prompt_logprobs=num_prompt_logprobs,
                          )) for idx, (prompt, prompt_tokens) in enumerate(
                              zip(dummy_test_vectors.prompt_strings,
                                  dummy_test_vectors.prompt_tokens))
    ]

    # Add requests to the detokenizer.
    for request in requests:
        output_processor.add_request(request)

    gen_tokens = {}
    gen_logprobs = {}
    gen_prompt_logprobs = {}
    gen_cumulative_logprobs = {}
    while True:
        # Mock output from the EngineCore.
        outputs = engine_core.get_outputs()
        if len(outputs) == 0:
            break

        # Step the logprobs processor.
        processed_outputs = output_processor.process_outputs(outputs)
        request_outputs = processed_outputs.request_outputs
        requests_to_abort = processed_outputs.reqs_to_abort
        assert len(requests_to_abort) == 0

        # Update tracking.
        for request_output in request_outputs:
            request_id = request_output.request_id
            new_tokens = request_output.outputs[0].token_ids
            prompt_logprobs = request_output.prompt_logprobs
            logprobs = request_output.outputs[0].logprobs
            gen_cumulative_logprobs[request_id] = request_output.outputs[
                0].cumulative_logprob
            if request_id not in gen_logprobs:
                # Start tracking sample and prompt logprobs for this request
                gen_tokens[request_id] = new_tokens
                gen_logprobs[request_id] = logprobs
                gen_prompt_logprobs[request_id] = prompt_logprobs
            else:
                # Extend logprobs tracker
                gen_tokens[request_id].extend(new_tokens)
                lp = gen_logprobs[request_id]
                plp = gen_prompt_logprobs[request_id]
                if lp:
                    lp.extend(logprobs)
                if plp:
                    plp.extend(prompt_logprobs)

    # Confirmed tracked logprobs match what we expect
    _validate_logprobs(gen_tokens, gen_logprobs, gen_prompt_logprobs,
                       gen_cumulative_logprobs, dummy_test_vectors,
                       request_id_list, num_sample_logprobs,
                       num_prompt_logprobs)

    assert output_processor.get_num_unfinished_requests() == 0
    assert not output_processor.has_unfinished_requests()


@pytest.mark.parametrize("include_stop_str_in_output", [True, False])
@pytest.mark.parametrize("num_sample_logprobs", [None, 5])
@pytest.mark.parametrize("num_prompt_logprobs", [None, 5])
def test_stop_string(include_stop_str_in_output: bool,
                     num_sample_logprobs: Optional[int],
                     num_prompt_logprobs: Optional[int], dummy_test_vectors):
    output_processor = OutputProcessor(dummy_test_vectors.tokenizer_group,
                                       log_stats=False)
    engine_core = MockEngineCore(
        tokens_list=dummy_test_vectors.generation_tokens,
        generated_logprobs_raw=dummy_test_vectors.generation_logprobs
        if num_sample_logprobs else None,
        prompt_logprobs_raw=dummy_test_vectors.prompt_logprobs
        if num_prompt_logprobs else None)

    # Make N requests.
    request_id_list = [
        f"request-{idx}"
        for idx in range(len(dummy_test_vectors.prompt_strings))
    ]
    requests = [
        EngineCoreRequest(
            request_id=request_id_list[idx],
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
                logprobs=num_sample_logprobs,
                prompt_logprobs=num_prompt_logprobs,
            )) for idx, (prompt, prompt_tokens) in enumerate(
                zip(dummy_test_vectors.prompt_strings,
                    dummy_test_vectors.prompt_tokens))
    ]

    # Add requests to the detokenizer.
    for request in requests:
        output_processor.add_request(request)

    gen_strings = {}
    gen_tokens = {}
    gen_logprobs = {}
    gen_prompt_logprobs = {}
    gen_cumulative_logprobs = {}
    aborted = []
    while True:
        # Mock output from the EngineCore.
        outputs = engine_core.get_outputs()
        if len(outputs) == 0:
            break

        # Step the Detokenizer.
        processed_outputs = output_processor.process_outputs(outputs)
        request_outputs = processed_outputs.request_outputs
        requests_to_abort = processed_outputs.reqs_to_abort
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
            new_tokens = request_output.outputs[0].token_ids
            prompt_logprobs = request_output.prompt_logprobs
            logprobs = request_output.outputs[0].logprobs
            gen_cumulative_logprobs[request_id] = request_output.outputs[
                0].cumulative_logprob
            if request_id not in gen_strings:
                gen_strings[request_id] = new_text
                gen_tokens[request_id] = new_tokens
                gen_logprobs[request_id] = logprobs
                gen_prompt_logprobs[request_id] = prompt_logprobs
            else:
                gen_strings[request_id] += new_text
                gen_tokens[request_id].extend(new_tokens)
                lp = gen_logprobs[request_id]
                plp = gen_prompt_logprobs[request_id]
                if lp:
                    lp.extend(logprobs)
                if plp:
                    plp.extend(prompt_logprobs)

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

    # Confirmed tracked logprobs match what we expect
    _validate_logprobs(gen_tokens, gen_logprobs, gen_prompt_logprobs,
                       gen_cumulative_logprobs, dummy_test_vectors,
                       request_id_list, num_sample_logprobs,
                       num_prompt_logprobs)

    assert output_processor.get_num_unfinished_requests() == 0
    assert not output_processor.has_unfinished_requests()


def test_iteration_stats(dummy_test_vectors):
    output_processor = OutputProcessor(dummy_test_vectors.tokenizer_group,
                                       log_stats=True)
    engine_core = MockEngineCore(dummy_test_vectors.generation_tokens)

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
            sampling_params=SamplingParams(),
        ) for idx, (prompt, prompt_tokens) in enumerate(
            zip(dummy_test_vectors.prompt_strings,
                dummy_test_vectors.prompt_tokens))
    ]

    # Add all requests except one to the OutputProcessor.
    num_active = len(dummy_test_vectors.generation_tokens) - 1
    for request in requests[:num_active]:
        output_processor.add_request(request)
    inactive_request = requests[num_active]

    # First iteration has 2 prefills.
    outputs = engine_core.get_outputs()[:num_active]
    processed_outputs = output_processor.process_outputs(outputs)
    iteration_stats = processed_outputs.iteration_stats
    total_prompt_tokens = sum([
        len(prompt_tokens)
        for prompt_tokens in dummy_test_vectors.prompt_tokens[:num_active]
    ])

    assert iteration_stats.num_prompt_tokens == total_prompt_tokens
    assert iteration_stats.num_generation_tokens == num_active

    # Just decodes in this step.
    outputs = engine_core.get_outputs()[:num_active]
    processed_outputs = output_processor.process_outputs(outputs)
    iteration_stats = processed_outputs.iteration_stats

    assert iteration_stats.num_prompt_tokens == 0
    assert iteration_stats.num_generation_tokens == num_active

    # Add a new request - prefill and 2 decodes in this step.
    output_processor.add_request(inactive_request)
    num_active += 1
    outputs = engine_core.get_outputs()[:num_active]
    processed_outputs = output_processor.process_outputs(outputs)
    iteration_stats = processed_outputs.iteration_stats
    total_prompt_tokens = len(dummy_test_vectors.prompt_tokens[num_active - 1])

    assert iteration_stats.num_prompt_tokens == total_prompt_tokens
    assert iteration_stats.num_generation_tokens == num_active

    # Just decodes in this step.
    outputs = engine_core.get_outputs()[:num_active]
    processed_outputs = output_processor.process_outputs(outputs)
    iteration_stats = processed_outputs.iteration_stats

    assert iteration_stats.num_prompt_tokens == 0
    assert iteration_stats.num_generation_tokens == num_active
