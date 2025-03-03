# SPDX-License-Identifier: Apache-2.0

import math
import time
from typing import Optional

import pytest

from tests.v1.engine.utils import (NUM_PROMPT_LOGPROBS_UNDER_TEST,
                                   NUM_SAMPLE_LOGPROBS_UNDER_TEST,
                                   STOP_STRINGS,
                                   DummyOutputProcessorTestVectors,
                                   MockEngineCore)
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.sequence import PromptLogprobs, SampleLogprobs
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.metrics.stats import IterationStats


def _ref_convert_id_to_token(
    tokenizer: AnyTokenizer,
    token_id: int,
) -> str:
    """Reference impl of logprobs detokenization.

    Args:
      tokenizer: tokenizer used by the model under test
      token_id: convert this token id

    Returns:
      String representation of input token id
    """
    return tokenizer.convert_ids_to_tokens(token_id) or ""


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
    gen_tokens: dict[str, list[int]],
    gen_logprobs: dict[str, Optional[SampleLogprobs]],
    gen_prompt_logprobs: dict[str, Optional[PromptLogprobs]],
    gen_cumulative_logprob: dict[str, float],
    dtv: DummyOutputProcessorTestVectors,
    request_id_list: list[str],
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
        if num_sample_logprobs is not None:
            # Validate sample logprobs
            assert logprobs is not None, (f"Request {req_id} requires sample"
                                          " logprobs but sample logprobs are"
                                          " None.")
            # Require num sampled tokens to match num
            # sampled logprobs - especially important
            # to check since the detokenizer can cause
            # a request to finish early due to a stop
            # string being hit
            num_new_tokens = len(new_tokens)
            len_sample_logprobs = len(logprobs)
            assert num_new_tokens == len_sample_logprobs, (
                f"Request {req_id} has {num_new_tokens}"
                " completion tokens but has"
                f" {len_sample_logprobs} sample logprobs.")
            ref_cumulative_logprob = 0.0
            for idx, (sampled_token,
                      pos_logprob_dict) in enumerate(zip(new_tokens,
                                                         logprobs)):
                # Break out the reference log probability value &
                # logprob token id tensors associated with this
                # position in the completion. Also break out the
                # sampled token ranks
                (ref_pos_logprob_toks, ref_pos_logprob_vals,
                 ref_sampled_token_rank) = ref_logprobs[idx]
                # For each position in the completion sequence,
                # ensure the actual sampled token is among the
                # logprobs
                assert sampled_token in pos_logprob_dict, (
                    f"Sampled token {sampled_token} not"
                    f" present in logprob at index {idx}")

                # Validate number of sample logprobs
                num_lp_toks = len(pos_logprob_dict)
                assert (num_lp_toks == num_sample_logprobs
                        or num_lp_toks == num_sample_logprobs +
                        1), ("Valid numbers of sample logprobs are"
                             f" {num_sample_logprobs} or"
                             f" {num_sample_logprobs+1} but"
                             f" {num_lp_toks} logprobs found at"
                             f" position {idx}. Logprobs dict:"
                             f" {pos_logprob_dict}")

                # Validate sampled token logprob rank
                smp_lp = pos_logprob_dict[sampled_token]
                smp_lp_rank = smp_lp.rank
                assert (ref_sampled_token_rank == smp_lp_rank), (
                    "Sampled token logprob rank"
                    f" {smp_lp_rank} does not match"
                    " correct value"
                    f" {ref_sampled_token_rank}"
                    f" in Logprob {smp_lp}")

                # Validate that the logprob processor yields
                # the correct log probabilities and valid
                # rankings
                rank_one_appears = False
                for jdx in range(1, len(ref_pos_logprob_toks)):
                    # Iterate over the (logprob val,logprob tok id)
                    # pairs expected by the test fixture at this
                    # position in the completion.
                    ref_lp_val = ref_pos_logprob_vals[jdx]
                    ref_tok_id = ref_pos_logprob_toks[jdx]
                    assert ref_tok_id in pos_logprob_dict, (
                        f"Expected token {ref_tok_id} to be"
                        f" in logprob dict but it is not.")

                    # Extract actually-generated logprob
                    # info
                    lp = pos_logprob_dict[ref_tok_id]
                    lp_val = lp.logprob
                    lp_rank = lp.rank

                    # A "top" (rank 1) logprob must be
                    # present
                    rank_one_appears = (True
                                        if lp_rank == 1 else rank_one_appears)

                    # Rank must be >= 1
                    assert lp_rank >= 1, (f"Logprob {lp} has invalid"
                                          f" rank {lp_rank} < 1."
                                          f" Logprob dict: {pos_logprob_dict}")

                    # Validate log probability
                    assert math.isclose(lp_val, ref_lp_val), (
                        f"Token id {ref_tok_id} appears in logprobs dict"
                        f" at position {idx} in completion with log"
                        f" probability {lp_val} but {ref_lp_val} was"
                        f" expected. Logprob: {lp}")

                assert rank_one_appears, (f"No Logprob has rank 1"
                                          " in the following Logprob"
                                          f" dict: {pos_logprob_dict}")

                # Validate logprobs detokenization
                for lp_tok in pos_logprob_dict:
                    # Confirm that sample logprob decoded token matches
                    # the logprob token id at this sequence position
                    decoded_token = pos_logprob_dict[lp_tok].decoded_token
                    ref_decoded_token = _ref_convert_id_to_token(
                        dtv.tokenizer, lp_tok)
                    assert decoded_token == ref_decoded_token, (
                        f"Sampled logprob token id {lp_tok} decodes to"
                        f" {ref_decoded_token} but Logprob decoded"
                        f" token is {decoded_token} instead"
                        f" (at position {idx})")

                ref_cumulative_logprob += pos_logprob_dict[
                    sampled_token].logprob
            # Assert that cumulative logprobs are correct
            assert math.isclose(cumulative_logprob, ref_cumulative_logprob)
        else:
            # Sample logprobs disabled for this request
            assert logprobs is None
            assert cumulative_logprob is None

        if num_prompt_logprobs is not None:
            # Validate prompt logprobs
            assert prompt_logprobs is not None, (
                f"Request {req_id} requires prompt"
                " logprobs but prompt logprobs are"
                " None.")
            # Require num prompt tokens to match num
            # prompt logprobs
            num_prompt_tokens = len(prompt_token_ids)
            len_prompt_logprobs = len(prompt_logprobs)
            assert num_prompt_tokens == len_prompt_logprobs, (
                f"Request {req_id} has {num_prompt_tokens}"
                " prompt tokens but has"
                f" {len_prompt_logprobs} prompt logprobs.")
            # First prompt logprob is None
            first_plp_dict = prompt_logprobs[0]
            assert first_plp_dict is None, (
                f"Request {req_id} first prompt logprob"
                f" should be None but has following value"
                f" instead: {first_plp_dict}")
            # Break out the reference prompt log prob value &
            # logprob token id matrices for the whole prompt.
            # Also break out the prompt token rank vector
            (ref_prompt_logprob_toks, ref_prompt_logprob_vals,
             ref_prompt_token_ranks) = ref_prompt_logprobs
            for idx, (prompt_token, pos_logprob_dict) in enumerate(
                    zip(prompt_token_ids[1:], prompt_logprobs[1:])):

                # Break out the reference prompt log prob value
                # vector, prompt logprob token id vector, and
                # prompt token rank at the current position.
                (ref_pos_prompt_logprob_toks, ref_pos_prompt_logprob_vals,
                 ref_pos_prompt_token_rank) = (ref_prompt_logprob_toks[idx, :],
                                               ref_prompt_logprob_vals[idx, :],
                                               ref_prompt_token_ranks[idx])

                # For each position in the prompt sequence,
                # ensure the actual prompt token is among the
                # logprobs
                assert prompt_token in pos_logprob_dict, (
                    f"Prompt token {prompt_token} not"
                    f" present in logprob at index {idx}")
                # Validate number of prompt logprobs
                num_plp_toks = len(pos_logprob_dict)
                assert (num_plp_toks == num_prompt_logprobs
                        or num_plp_toks == num_prompt_logprobs +
                        1), ("Valid numbers of prompt logprobs are"
                             f" {num_prompt_logprobs} or"
                             f" {num_prompt_logprobs+1} but"
                             f" {num_plp_toks} logprobs found at"
                             f" position {idx}. Logprobs dict:"
                             f" {pos_logprob_dict}")

                # Validate prompt token logprob rank
                prmpt_tok_lp = pos_logprob_dict[prompt_token]
                prmpt_tok_lp_rank = prmpt_tok_lp.rank
                ref_prmpt_tok_lp_rank = ref_pos_prompt_token_rank
                assert (ref_prmpt_tok_lp_rank == prmpt_tok_lp_rank), (
                    "Prompt token logprob rank"
                    f" {prmpt_tok_lp_rank} does not match"
                    " correct value"
                    f" {ref_prmpt_tok_lp_rank}"
                    f" in Logprob {prmpt_tok_lp}")

                # Validate that the logprob processor yields
                # the correct prompt log probs and valid
                # rankings
                rank_one_appears = False
                for jdx in range(1, len(ref_pos_prompt_logprob_toks)):
                    # Iterate over the (logprob val,logprob tok id)
                    # pairs expected by the test fixture at this
                    # position in the completion.
                    ref_plp_val = float(ref_pos_prompt_logprob_vals[jdx])
                    ref_tok_id = int(ref_pos_prompt_logprob_toks[jdx])
                    assert ref_tok_id in pos_logprob_dict, (
                        f"Expected token {ref_tok_id} to be"
                        f" in logprob dict but it is not.")

                    # Extract actually-generated logprob
                    # info
                    plp = pos_logprob_dict[ref_tok_id]
                    plp_val = plp.logprob
                    plp_rank = plp.rank

                    # A "top" (rank 1) logprob must be
                    # present
                    rank_one_appears = (True
                                        if plp_rank == 1 else rank_one_appears)

                    # Rank must be >= 1
                    assert plp_rank >= 1, (
                        f"Logprob {plp} has invalid"
                        f" rank {plp_rank} < 1."
                        f" Logprob dict: {pos_logprob_dict}")

                    # Validate log probability
                    assert math.isclose(plp_val, ref_plp_val), (
                        f"Token id {ref_tok_id} appears in logprobs dict"
                        f" at position {idx} in completion with log"
                        f" probability {plp_val} but {ref_plp_val} was"
                        f" expected. Logprob: {plp}")

                assert rank_one_appears, (f"No Logprob has rank 1"
                                          " in the following Logprob"
                                          f" dict: {pos_logprob_dict}")

                # Validate prompt logprob detokenization
                for plp_tok in pos_logprob_dict:
                    # Confirm that prompt logprob decoded token matches
                    # the logprob token id at this sequence position
                    decoded_token = pos_logprob_dict[plp_tok].decoded_token
                    ref_decoded_token = _ref_convert_id_to_token(
                        dtv.tokenizer, plp_tok)
                    assert decoded_token == ref_decoded_token, (
                        f"Prompt logprob token id {plp_tok} decodes to"
                        f" {ref_decoded_token} but Logprob decoded"
                        f" token is {decoded_token} instead"
                        f" (at position {idx})")
        else:
            # Prompt logprobs disabled for this request
            assert prompt_logprobs is None


@pytest.mark.parametrize(
    "request_output_kind",
    [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
@pytest.mark.parametrize("num_sample_logprobs",
                         [None, NUM_SAMPLE_LOGPROBS_UNDER_TEST])
@pytest.mark.parametrize("num_prompt_logprobs",
                         [None, NUM_PROMPT_LOGPROBS_UNDER_TEST])
def test_logprobs_processor(request_output_kind: RequestOutputKind,
                            num_sample_logprobs: Optional[int],
                            num_prompt_logprobs: Optional[int],
                            dummy_test_vectors):
    output_processor = OutputProcessor(dummy_test_vectors.tokenizer_group,
                                       log_stats=False)
    engine_core = MockEngineCore(
        tokens_list=dummy_test_vectors.generation_tokens,
        generated_logprobs_raw=None if num_sample_logprobs is None else
        dummy_test_vectors.generation_logprobs,
        prompt_logprobs_raw=None
        if num_prompt_logprobs is None else dummy_test_vectors.prompt_logprobs)

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
@pytest.mark.parametrize("num_sample_logprobs",
                         [None, NUM_SAMPLE_LOGPROBS_UNDER_TEST])
@pytest.mark.parametrize("num_prompt_logprobs",
                         [None, NUM_PROMPT_LOGPROBS_UNDER_TEST])
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
    engine_core_timestamp = time.monotonic()

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
    iteration_stats = IterationStats()
    output_processor.process_outputs(outputs, engine_core_timestamp,
                                     iteration_stats)
    total_prompt_tokens = sum([
        len(prompt_tokens)
        for prompt_tokens in dummy_test_vectors.prompt_tokens[:num_active]
    ])

    assert iteration_stats.num_prompt_tokens == total_prompt_tokens
    assert iteration_stats.num_generation_tokens == num_active

    # Just decodes in this step.
    outputs = engine_core.get_outputs()[:num_active]
    iteration_stats = IterationStats()
    output_processor.process_outputs(outputs, engine_core_timestamp,
                                     iteration_stats)

    assert iteration_stats.num_prompt_tokens == 0
    assert iteration_stats.num_generation_tokens == num_active

    # Add a new request - prefill and 2 decodes in this step.
    output_processor.add_request(inactive_request)
    num_active += 1
    outputs = engine_core.get_outputs()[:num_active]
    iteration_stats = IterationStats()
    output_processor.process_outputs(outputs, engine_core_timestamp,
                                     iteration_stats)
    total_prompt_tokens = len(dummy_test_vectors.prompt_tokens[num_active - 1])

    assert iteration_stats.num_prompt_tokens == total_prompt_tokens
    assert iteration_stats.num_generation_tokens == num_active

    # Just decodes in this step.
    outputs = engine_core.get_outputs()[:num_active]
    iteration_stats = IterationStats()
    output_processor.process_outputs(outputs, engine_core_timestamp,
                                     iteration_stats)

    assert iteration_stats.num_prompt_tokens == 0
    assert iteration_stats.num_generation_tokens == num_active
