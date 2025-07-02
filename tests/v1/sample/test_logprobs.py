# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Generator

import pytest
import torch

from tests.v1.sample.utils import (
    BatchLogprobsComposition, BatchLogprobsSpecType,
    assert_incr_detok_str_matches_non_incr_detok_str,
    compute_correct_cumulative_logprob, get_test_batch)
from vllm import SamplingParams

from ...conftest import HfRunner, VllmRunner

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DTYPE = "half"

NONE = BatchLogprobsComposition.NONE
SAMPLE = BatchLogprobsComposition.SAMPLE
PROMPT = BatchLogprobsComposition.PROMPT
SAMPLE_PROMPT = BatchLogprobsComposition.SAMPLE_PROMPT


@pytest.fixture(
    scope="module",
    # Parameterize APC
    params=[False, True])
def vllm_model(vllm_runner, request) -> Generator[VllmRunner, None, None]:
    with vllm_runner(
            MODEL,
            dtype=DTYPE,
            max_logprobs=7,
            # Very small number of batched tokens to ensure
            # that we test chunking.
            max_num_batched_tokens=16,
            max_num_seqs=16,
            max_model_len=128,
            enforce_eager=True,
            #TODO: enable this once we support it for
            # prompt logprobs.
            enable_prefix_caching=request.param,
            gpu_memory_utilization=0.4,  # up to 2 alive concurrently
    ) as vllm_model:
        yield vllm_model


@pytest.fixture(scope="module")
def hf_model(hf_runner) -> Generator[HfRunner, None, None]:
    with hf_runner(MODEL, dtype=DTYPE) as hf_model:
        yield hf_model


def _repeat_logprob_config(
    test_prompts,
    logprob_prompt_logprob_list: BatchLogprobsSpecType,
) -> BatchLogprobsSpecType:
    """Ensure each test prompt has a logprob config.

    A logprob config specifies the optional (i.e.
    may-be-`None`) number of sample logprobs and
    the optional number of prompt logprobs.

    If more test prompts than logprob configs are
    provided, the provided logprob configs are
    tiled to match the number of test prompts.

    If fewer test prompts than logprob configs
    are provided, the list of logprob configs
    is truncated to match the number of test
    prompts.

    Otherwise, the list of logprob configs
    is returned as-is.

    Args:
      test_prompts: list of prompts under test
      logprob_prompt_logprob_list: list of
                            (optional num sample logprob,
                             optional num prompt logprob)
                             tuples

    Returns:
      list of
      (optional num sample logprob,optional num prompt logprob)
      tuples which is either identical to
      `logprob_prompt_logprob_list`, or else repeats
      `logprob_prompt_logprob_list` enough times to match the
      number of `test_prompts`, or else is truncated to match
      the number of `test_prompts`
    """
    num_test_prompts = len(test_prompts)
    # Make sure there is a logprobs configuration for each test prompt
    logprob_prompt_logprob_list = list(
        itertools.islice(itertools.cycle(logprob_prompt_logprob_list),
                         num_test_prompts))
    # Now the number of prompts should match the number of sample params combos
    assert num_test_prompts == len(logprob_prompt_logprob_list)
    return logprob_prompt_logprob_list


def _run_and_validate(
    vllm_model: VllmRunner,
    test_prompts: list[str],
    vllm_sampling_params: SamplingParams,
    hf_logprobs: list[list[torch.Tensor]],
    hf_outputs: list[tuple[list[int], str]],
    logprob_prompt_logprob_list: BatchLogprobsSpecType,
    temperature: float,
    max_tokens: int,
    do_apc: bool,
) -> None:
    vllm_results = vllm_model.model.generate(
        test_prompts, sampling_params=vllm_sampling_params)

    for vllm_result, hf_logprob, hf_output, logprob_prompt_logprob in zip(
            vllm_results, hf_logprobs, hf_outputs,
            logprob_prompt_logprob_list):

        # Extract request-level (prompt)logprobs config
        num_top_logprobs, num_top_prompt_logprobs = logprob_prompt_logprob

        # Test whether sampled token output is consistent between vLLM and HF
        # vLLM prompt+completion should match HF output
        if temperature == 0.0:
            assert (vllm_result.prompt_token_ids +
                    vllm_result.outputs[0].token_ids == hf_output[0])
        else:
            # Sampled tokens won't match if not greedy
            assert (vllm_result.prompt_token_ids == hf_output[0]
                    [:len(vllm_result.prompt_token_ids)])

        # Validate sample logprobs
        if num_top_logprobs is not None:
            assert num_top_logprobs is not None
            # Confirm that the structure of the sample logprobs in the result is
            # correct
            assert vllm_result.outputs[0].logprobs is not None
            assert len(vllm_result.outputs[0].logprobs) == max_tokens
            for logprobs, token_id in zip(vllm_result.outputs[0].logprobs,
                                          vllm_result.outputs[0].token_ids):
                assert logprobs is not None

                # Confirm that the output token appears among the logprobs
                assert token_id in logprobs
                token_in_topk = logprobs[token_id].rank <= num_top_logprobs

                # If the output token is not included in the top K
                # logprob, it can return 1 more data
                if token_in_topk and num_top_logprobs != 0:
                    assert len(logprobs) == num_top_logprobs
                else:
                    assert len(logprobs) == num_top_logprobs + 1

                if num_top_logprobs > 0:
                    # We should have an entry for each of the topk ranks
                    all_ranks = {lp.rank for lp in logprobs.values()}
                    assert all(r in all_ranks
                               for r in range(1, num_top_logprobs + 1))

            output_text = vllm_result.outputs[0].text
            output_string_from_most_likely_tokens_lst: list[str] = []
            for top_logprobs in vllm_result.outputs[0].logprobs:
                top_logprob = next(iter(top_logprobs.values()))
                output_string_from_most_likely_tokens_lst.append(
                    top_logprob.decoded_token)

            output_string_from_most_likely_tokens = "".join(
                output_string_from_most_likely_tokens_lst)
            assert_incr_detok_str_matches_non_incr_detok_str(
                output_text, output_string_from_most_likely_tokens,
                "The output text from the top logprob for each token "
                "position should be the same as the output text in the "
                "result.")

            # Compare vLLM sample logprobs to HF
            vllm_sample_logprobs = vllm_result.outputs[0].logprobs
            for i, top_logprobs in enumerate(vllm_sample_logprobs):
                for token_id, sample_logprob in top_logprobs.items():
                    if temperature == 0.0 or i == 0:
                        logprob = sample_logprob.logprob
                        torch.testing.assert_close(
                            logprob,
                            hf_logprob[i][-1][token_id].item(),
                            atol=1e-2,
                            rtol=1e-2)
                    assert isinstance(
                        sample_logprob.decoded_token,
                        str), ("The token should be decoded by the time it is"
                               " returned to the user.")

            # At this point we know the sample logprobs are correct for this
            # request. Validate that cumulative_logprob is actually the sum.
            # For each request, assert that the returned cumulative logprob
            # matches the correct value, which is computed below.
            torch.testing.assert_close(
                vllm_result.outputs[0].cumulative_logprob,
                compute_correct_cumulative_logprob(vllm_result.outputs[0]),
                atol=1e-6,
                rtol=1e-6)
        else:
            # Logprobs disabled for this request; should be None
            assert vllm_result.outputs[0].logprobs is None

        # Validate prompt logprobs
        if num_top_prompt_logprobs is not None:
            # Confirm that structure of prompt logprobs in result is correct
            assert vllm_result.prompt_logprobs is not None
            # - The first prompt logprob is always None
            assert vllm_result.prompt_logprobs[0] is None
            # - Prompt logprobs are returned for all indices in
            #   the prompt
            assert len(vllm_result.prompt_logprobs) == len(
                vllm_result.prompt_token_ids)
            for prompt_logprobs, prompt_token_id in zip(
                    vllm_result.prompt_logprobs[1:],
                    vllm_result.prompt_token_ids[1:]):
                assert prompt_logprobs is not None

                # Confirm that the prompt token appears among the logprobs
                assert prompt_token_id in prompt_logprobs
                token_in_topk = prompt_logprobs[
                    prompt_token_id].rank <= num_top_prompt_logprobs

                # If the prompt token is not included in the top K
                # logprob, it can return 1 more data
                if token_in_topk and num_top_prompt_logprobs != 0:
                    assert len(prompt_logprobs) == num_top_prompt_logprobs
                else:
                    assert len(prompt_logprobs) == num_top_prompt_logprobs + 1

                if num_top_prompt_logprobs > 0:
                    # We should have an entry for each of the topk ranks
                    all_ranks = {lp.rank for lp in prompt_logprobs.values()}
                    assert all(r in all_ranks
                               for r in range(1, num_top_prompt_logprobs + 1))

            # Compare prompt logprobs to HF
            # The first prompt logprob is always None, so we compare it from
            # 1:.
            vllm_prompt_logprobs = vllm_result.prompt_logprobs[1:]
            for i, vllm_prompt_logprob_dict in enumerate(vllm_prompt_logprobs):
                for token_id, logprob in vllm_prompt_logprob_dict.items():
                    torch.testing.assert_close(
                        logprob.logprob,
                        hf_logprob[0][i][token_id].item(),
                        atol=2e-2,
                        rtol=2e-2)
        else:
            assert vllm_result.prompt_logprobs is None


@pytest.mark.parametrize("batch_logprobs_composition",
                         [NONE, SAMPLE, PROMPT, SAMPLE_PROMPT])
@pytest.mark.parametrize("temperature", [0.0, 2.0])
def test_get_logprobs_and_prompt_logprobs(
        hf_model, vllm_model,
        batch_logprobs_composition: BatchLogprobsComposition,
        temperature: float, example_prompts: list[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Test V1 Engine logprobs & prompt logprobs

    Exercise a variety of combinations of `logprobs` and `prompt_logprobs`
    settings and validate that
    * The generated logprobs and prompt logprobs are consistent with the
      configuration settings, in terms of whether or not the logprobs
      (of either type) were requested and how many were requested
    * The generated logprobs are consistent with the generated tokens
    * The generated (prompt)logprobs are consistent with HuggingFace
      (prompt)logprobs, as a reference

    batch_logprobs_composition controls the logprobs configurations for
    requests in the batch under test.

    APC tests run two test iterations so that cache hits occur.

    To save time, only test one APC-enabled scenario
    (sample & prompt logprobs enabled, temperature>0.0).

    Args:
      hf_model: HuggingFace reference model fixture
      vllm_model: vLLM model fixture
      batch_logprobs_composition: logprobs configuration for test batch
      temperature: "temperature" sampling parameter
      example_prompts: example prompt fixture
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        do_apc = vllm_model.model.llm_engine.cache_config.enable_prefix_caching
        if do_apc and (temperature < 2.0
                       or batch_logprobs_composition != SAMPLE_PROMPT):
            # Skip some test-cases to save time.
            pytest.skip()
        test_prompts = example_prompts

        max_tokens = 5
        hf_outputs = hf_model.generate_greedy(
            test_prompts,
            max_tokens=max_tokens,
        )
        hf_logprobs = hf_model.generate_greedy_logprobs(
            test_prompts,
            max_tokens=max_tokens,
        )

        # Batch has mixed sample params
        # (different logprobs/prompt logprobs combos)
        logprob_prompt_logprob_list = get_test_batch(
            batch_logprobs_composition)

        # Ensure that each test prompt has a logprob config for testing
        logprob_prompt_logprob_list = _repeat_logprob_config(
            test_prompts, logprob_prompt_logprob_list)
        # Generate SamplingParams
        vllm_sampling_params = [
            SamplingParams(max_tokens=max_tokens,
                           logprobs=num_lp,
                           prompt_logprobs=num_plp,
                           temperature=temperature,
                           seed=1984)
            for num_lp, num_plp in logprob_prompt_logprob_list
        ]
        for _ in range(2 if do_apc else 1):
            _run_and_validate(
                vllm_model=vllm_model,
                test_prompts=test_prompts,
                vllm_sampling_params=vllm_sampling_params,
                hf_logprobs=hf_logprobs,
                hf_outputs=hf_outputs,
                logprob_prompt_logprob_list=logprob_prompt_logprob_list,
                temperature=temperature,
                max_tokens=max_tokens,
                do_apc=do_apc)


def test_max_logprobs(monkeypatch: pytest.MonkeyPatch):
    """vLLM v1 engine should fail a request with `logprobs > max_logprobs`
    Should also fail for `prompt_logprobs > max_logprobs`
    APC should not matter as this test checks basic request validation.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        runner = VllmRunner(
            "facebook/opt-125m",
            max_logprobs=1,
            enable_prefix_caching=False,
            # 2 other llms alive during whole session
            gpu_memory_utilization=0.15,
            max_model_len=256)
        vllm_sampling_params = SamplingParams(logprobs=1)
        # should pass
        runner.generate(["Hello world"], sampling_params=vllm_sampling_params)

        bad_sampling_params = SamplingParams(logprobs=2)
        with pytest.raises(ValueError):
            runner.generate(["Hello world"],
                            sampling_params=bad_sampling_params)


def test_none_logprobs(vllm_model, example_prompts,
                       monkeypatch: pytest.MonkeyPatch):
    """Engine should return `logprobs` and `prompt_logprobs` as `None`

    Args:
      vllm_model: vLLM model fixture
      example_prompts: list of example prompts (test fixture)
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        max_tokens = 5

        sampling_params_logprobs_none = SamplingParams(
            max_tokens=max_tokens,
            logprobs=None,
            prompt_logprobs=None,
            temperature=0.0,
        )
        results_logprobs_none = vllm_model.model.generate(
            example_prompts,
            sampling_params=sampling_params_logprobs_none,
        )

        for i in range(len(results_logprobs_none)):
            # Check sample logprobs are None
            assert results_logprobs_none[i].outputs[0].logprobs is None
            assert results_logprobs_none[i].outputs[
                0].cumulative_logprob is None
            # Check prompt logprobs are None
            assert results_logprobs_none[i].prompt_logprobs is None


def test_zero_logprobs(vllm_model, example_prompts,
                       monkeypatch: pytest.MonkeyPatch):
    """Engine should return sampled token and prompt token logprobs

    Args:
      vllm_model: vLLM model fixture
      example_prompts: list of example prompts (test fixture)
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        max_tokens = 5

        sampling_params_logprobs_zero = SamplingParams(max_tokens=max_tokens,
                                                       logprobs=0,
                                                       prompt_logprobs=0,
                                                       temperature=0.0)
        results_logprobs_zero = vllm_model.model.generate(
            example_prompts, sampling_params=sampling_params_logprobs_zero)

        for i in range(len(results_logprobs_zero)):
            # Check that there is one sample logprob dict for each
            # sample token
            logprobs = results_logprobs_zero[i].outputs[0].logprobs
            prompt_logprobs = results_logprobs_zero[i].prompt_logprobs
            sampled_token_ids = results_logprobs_zero[i].outputs[0].token_ids
            prompt_token_ids = results_logprobs_zero[i].prompt_token_ids
            assert logprobs is not None
            assert len(sampled_token_ids) == len(logprobs)
            assert results_logprobs_zero[i].outputs[
                0].cumulative_logprob is not None
            # Check that there is one prompt logprob dict for each
            # prompt token
            assert prompt_logprobs is not None
            assert len(prompt_token_ids) == len(prompt_logprobs)
