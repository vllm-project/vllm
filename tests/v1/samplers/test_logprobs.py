from typing import List, Tuple

import pytest
import torch

from tests.kernels.utils import override_backend_env_variable
from vllm import SamplingParams

from ...conftest import VllmRunner

MODELS = ["facebook/opt-125m"]


def _get_test_batch(batch_logprobs_composition: str) -> List[Tuple]:
    """Generate logprobs configs for a batch of requests
    
    A given request's logprobs configuration is (1) num_sample_logprobs and (2)
    num_prompt_logprobs. The batch logprobs configuration is the list of request
    logprobs configs.

    batch_logprobs_composition == "NONE" yields a batch with no sample or prompt
    logprobs

    batch_logprobs_composition == "SAMPLE" yields a batch with some requests
    configured for sample logprobs only, and others configured for no logprobs

    batch_logprobs_composition == "PROMPT" yields a batch with some requests
    configured for prompt logprobs only, and others configured for no logprobs

    batch_logprobs_composition == "SAMPLE_PROMPT" yields a batch with some
    requests configured for sample logprobs and prompt logprobs, some configured
    for only sample logprobs or only prompt logprobs, and some configured for
    no logprobs

    Args:
      batch_logprobs_composition: types of logprobs configs to include in batch

    Returns:

      List of (Optional[num_sample_logprobs], Optional[num_prompt_logprobs])
      tuples
    """
    if batch_logprobs_composition == "NONE":
        # No requests with sample or prompt logprobs
        return [(None, None), (0, None), (None, 0), (0, 0)]
    elif batch_logprobs_composition == "SAMPLE":
        return [
            (None, None),
            (None, 0),
            (0, None),
            (0, 0),
            (5, None),
            (3, 0),
        ]
    elif batch_logprobs_composition == "PROMPT":
        return [
            (None, 0),
            (0, None),
            (0, 0),
            (None, 6),
            (0, 5),
        ]
    elif batch_logprobs_composition == "SAMPLE_PROMPT":
        return [
            (None, 0),
            (0, None),
            (0, 0),
            (5, None),
            (3, 0),
            (6, 3),
            (None, 6),
            (0, 5),
        ]
    else:
        raise ValueError("Invalid logprobs batch configuration for test.")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype",
                         ["half"])  # needed for comparing logprobs with HF
# @pytest.mark.parametrize("detokenize", [True, False])
@pytest.mark.parametrize("max_num_batched_tokens", [128, 256, 1024])
@pytest.mark.parametrize("batch_logprobs_composition",
                         ["NONE", "SAMPLE", "PROMPT", "SAMPLE_PROMPT"])
def test_get_logprobs_and_prompt_logprobs(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
    # detokenize: bool,
    batch_logprobs_composition: str,
    max_num_batched_tokens: int,
    example_prompts,
    monkeypatch,
):
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

    Args:
      hf_runner
      vllm_runner
      model
      dtype
      detokenize: if False, return generated tokens bypassing detokenizer
      batch_logprobs_composition: logprobs configuration for test batch
      example_prompts
      monkeypatch
    """
    detokenize = True

    test_prompts = example_prompts

    # LLM engine v1
    monkeypatch.setenv("VLLM_USE_V1", "1")
    override_backend_env_variable(monkeypatch, "FLASH_ATTN")

    max_num_seqs = 128
    max_num_batched_tokens = 128
    max_model_len = 128

    max_tokens = 5
    with hf_runner(model, dtype=dtype) as hf_model:
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
    logprob_prompt_logprob_list = _get_test_batch(batch_logprobs_composition)

    # We rely on there being more prompts than combinations of
    # logprobs & prompt logprobs which we want to test
    assert len(test_prompts) >= len(logprob_prompt_logprob_list)
    # Make sure there is a sample params for each prompt
    num_extra_params = len(test_prompts) - len(logprob_prompt_logprob_list)
    if num_extra_params > 0:
        logprob_prompt_logprob_list = (
            logprob_prompt_logprob_list +
            logprob_prompt_logprob_list[-num_extra_params:])
    # Now the number of prompts should match the number of sample params combos
    assert len(test_prompts) == len(logprob_prompt_logprob_list)
    # Generate SamplingParams
    vllm_sampling_params = [
        SamplingParams(max_tokens=max_tokens,
                       logprobs=lp,
                       prompt_logprobs=plp,
                       temperature=0.0,
                       detokenize=detokenize)
        for lp, plp in logprob_prompt_logprob_list
    ]

    with vllm_runner(
            model,
            dtype=dtype,
            max_logprobs=7,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            enforce_eager=True,
    ) as vllm_model:
        vllm_results = vllm_model.model.generate(
            test_prompts, sampling_params=vllm_sampling_params)

    for vllm_result, hf_logprob, hf_output, logprob_prompt_logprob in zip(
            vllm_results, hf_logprobs, hf_outputs,
            logprob_prompt_logprob_list):

        # Extract request-level (prompt)logprobs config
        num_top_logprobs = logprob_prompt_logprob[0]
        num_top_prompt_logprobs = logprob_prompt_logprob[1]

        # Test whether sampled token output is consistent between vLLM and HF
        # vLLM prompt+completion should match HF output
        assert (vllm_result.prompt_token_ids +
                vllm_result.outputs[0].token_ids == hf_output[0])

        # Validate sample logprobs
        if num_top_logprobs is not None and num_top_logprobs > 0:
            assert num_top_logprobs is not None
            # Confirm that the structure of the sample logprobs in the result is
            # correct
            assert vllm_result.outputs[0].logprobs is not None
            assert len(vllm_result.outputs[0].logprobs) == max_tokens
            for logprobs in vllm_result.outputs[0].logprobs:
                assert logprobs is not None
                # If the output token is not included in the top X
                # logprob, it can return 1 more data
                assert (len(logprobs) == num_top_logprobs
                        or len(logprobs) == num_top_logprobs + 1)
            output_text = vllm_result.outputs[0].text
            output_string_from_most_likely_tokens_lst: List[str] = []
            for top_logprobs in vllm_result.outputs[0].logprobs:
                top_logprob = next(iter(top_logprobs.values()))
                output_string_from_most_likely_tokens_lst.append(
                    top_logprob.decoded_token)

            if detokenize:
                output_string_from_most_likely_tokens = "".join(
                    output_string_from_most_likely_tokens_lst)
                assert output_text == output_string_from_most_likely_tokens, (
                    "The output text from the top logprob for each token "
                    "position should be the same as the output text in the "
                    "result.")
            else:
                assert output_text == ''
                assert output_string_from_most_likely_tokens_lst == (
                    [None] * max_tokens)

            # Compare vLLM sample logprobs to HF
            vllm_sample_logprobs = vllm_result.outputs[0].logprobs
            for i, top_logprobs in enumerate(vllm_sample_logprobs):
                for token_id, sample_logprob in top_logprobs.items():
                    logprob = sample_logprob.logprob
                    torch.testing.assert_close(
                        logprob,
                        hf_logprob[i][-1][token_id].item(),
                        atol=1e-2,
                        rtol=1e-2)
                    if detokenize:
                        assert isinstance(sample_logprob.decoded_token, str), (
                            "The token should be decoded by the time it is"
                            " returned to the user.")
        else:
            # Logprobs disabled for this request; should be None
            assert vllm_result.outputs[0].logprobs is None

        # Validate prompt logprobs
        if (num_top_prompt_logprobs is not None
                and num_top_prompt_logprobs > 0):
            # Confirm that structure of prompt logprobs in result is correct
            assert vllm_result.prompt_logprobs is not None
            # - The first prompt logprob is always None
            assert vllm_result.prompt_logprobs[0] is None
            # - Prompt logprobs are returned for all indices in
            #   the prompt
            assert len(vllm_result.prompt_logprobs) == len(
                vllm_result.prompt_token_ids)
            for prompt_logprobs in vllm_result.prompt_logprobs[1:]:
                assert prompt_logprobs is not None
                # - If the prompt token is not included in the top X
                #   logprob, it can return 1 more data
                assert (len(prompt_logprobs) == num_top_prompt_logprobs
                        or len(prompt_logprobs) == num_top_prompt_logprobs + 1)

            # Compare prompt logprobs to HF
            # The first prompt logprob is always None, so we compare it from
            # 1:.
            vllm_prompt_logprobs = vllm_result.prompt_logprobs[1:]
            for i, vllm_prompt_logprob_dict in enumerate(vllm_prompt_logprobs):
                for token_id, logprob in vllm_prompt_logprob_dict.items():
                    torch.testing.assert_close(
                        logprob.logprob,
                        hf_logprob[0][i][token_id].item(),
                        atol=1e-2,
                        rtol=1e-2)
        else:
            assert vllm_result.prompt_logprobs is None


def test_max_logprobs(monkeypatch):
    """vLLM v1 engine should fail a request with `logprobs > max_logprobs`
    
    Should also fail for `prompt_logprobs > max_logprobs`
    
    Args:
      monkeypatch
    """
    # LLM engine v1
    monkeypatch.setenv("VLLM_USE_V1", "1")
    override_backend_env_variable(monkeypatch, "FLASH_ATTN")

    runner = VllmRunner("facebook/opt-125m", max_logprobs=1)
    vllm_sampling_params = SamplingParams(logprobs=1)
    # should pass
    runner.generate(["Hello world"], sampling_params=vllm_sampling_params)

    bad_sampling_params = SamplingParams(logprobs=2)
    with pytest.raises(ValueError):
        runner.generate(["Hello world"], sampling_params=bad_sampling_params)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("detokenize", [True, False])
def test_none_logprobs(vllm_runner, model, detokenize: bool, example_prompts,
                       monkeypatch):
    """Engine should return `logprobs` and `prompt_logprobs` as `None`
    
    Args:
      vllm_runner
      model
      detokenize: whether to feed generated tokens to detokenizer
      example_prompts
      monkeypatch
    """

    # LLM engine v1
    monkeypatch.setenv("VLLM_USE_V1", "1")
    override_backend_env_variable(monkeypatch, "FLASH_ATTN")

    max_num_seqs = 256
    max_num_batched_tokens = None
    max_tokens = 5

    with vllm_runner(
            model,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
    ) as vllm_model:
        sampling_params_logprobs_none = SamplingParams(max_tokens=max_tokens,
                                                       logprobs=None,
                                                       prompt_logprobs=None,
                                                       temperature=0.0,
                                                       detokenize=detokenize)
        results_logprobs_none = vllm_model.model.generate(
            example_prompts, sampling_params=sampling_params_logprobs_none)

    for i in range(len(results_logprobs_none)):
        # Check sample logprobs are None
        assert results_logprobs_none[i].outputs[0].logprobs is None
        assert results_logprobs_none[i].outputs[0].cumulative_logprob is None
        # Check prompt logprobs are None
        assert results_logprobs_none[i].prompt_logprobs is None
