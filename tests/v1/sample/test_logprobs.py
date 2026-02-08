# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import math
from collections.abc import Generator
from typing import get_args

import pytest
import torch

from tests.utils import large_gpu_mark
from tests.v1.sample.utils import (
    BatchLogprobsComposition,
    BatchLogprobsSpecType,
    assert_incr_detok_str_matches_non_incr_detok_str,
    compute_correct_cumulative_logprob,
    get_test_batch,
)
from vllm import SamplingParams
from vllm.config.model import LogprobsMode
from vllm.distributed import cleanup_dist_env_and_memory

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
    params=[False, True],
)
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
        enable_chunked_prefill=True,
        enforce_eager=True,
        # TODO: enable this once we support it for
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
        itertools.islice(itertools.cycle(logprob_prompt_logprob_list), num_test_prompts)
    )
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
    vllm_results = vllm_model.llm.generate(
        test_prompts, sampling_params=vllm_sampling_params
    )

    for vllm_result, hf_logprob, hf_output, logprob_prompt_logprob in zip(
        vllm_results, hf_logprobs, hf_outputs, logprob_prompt_logprob_list
    ):
        # Extract request-level (prompt)logprobs config
        num_top_logprobs, num_top_prompt_logprobs = logprob_prompt_logprob

        # Test whether sampled token output is consistent between vLLM and HF
        # vLLM prompt+completion should match HF output
        if temperature == 0.0:
            assert (
                vllm_result.prompt_token_ids + vllm_result.outputs[0].token_ids
                == hf_output[0]
            )
        else:
            # Sampled tokens won't match if not greedy
            assert (
                vllm_result.prompt_token_ids
                == hf_output[0][: len(vllm_result.prompt_token_ids)]
            )

        # Validate sample logprobs
        if num_top_logprobs is not None:
            assert num_top_logprobs is not None
            # Confirm that the structure of the sample logprobs in the result is
            # correct
            assert vllm_result.outputs[0].logprobs is not None
            assert len(vllm_result.outputs[0].logprobs) == max_tokens
            for logprobs, token_id in zip(
                vllm_result.outputs[0].logprobs, vllm_result.outputs[0].token_ids
            ):
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
                    assert all(r in all_ranks for r in range(1, num_top_logprobs + 1))

            output_text = vllm_result.outputs[0].text
            output_string_from_most_likely_tokens_lst: list[str] = []
            for top_logprobs in vllm_result.outputs[0].logprobs:
                top_logprob = next(iter(top_logprobs.values()))
                output_string_from_most_likely_tokens_lst.append(
                    top_logprob.decoded_token
                )

            output_string_from_most_likely_tokens = "".join(
                output_string_from_most_likely_tokens_lst
            )
            assert_incr_detok_str_matches_non_incr_detok_str(
                output_text,
                output_string_from_most_likely_tokens,
                "The output text from the top logprob for each token "
                "position should be the same as the output text in the "
                "result.",
            )

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
                            rtol=1e-2,
                        )
                    assert isinstance(sample_logprob.decoded_token, str), (
                        "The token should be decoded by the time it is"
                        " returned to the user."
                    )

            # At this point we know the sample logprobs are correct for this
            # request. Validate that cumulative_logprob is actually the sum.
            # For each request, assert that the returned cumulative logprob
            # matches the correct value, which is computed below.
            torch.testing.assert_close(
                vllm_result.outputs[0].cumulative_logprob,
                compute_correct_cumulative_logprob(vllm_result.outputs[0]),
                atol=1e-6,
                rtol=1e-6,
            )
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
            assert len(vllm_result.prompt_logprobs) == len(vllm_result.prompt_token_ids)
            for prompt_logprobs, prompt_token_id in zip(
                vllm_result.prompt_logprobs[1:], vllm_result.prompt_token_ids[1:]
            ):
                assert prompt_logprobs is not None

                # Confirm that the prompt token appears among the logprobs
                assert prompt_token_id in prompt_logprobs
                token_in_topk = (
                    prompt_logprobs[prompt_token_id].rank <= num_top_prompt_logprobs
                )

                # If the prompt token is not included in the top K
                # logprob, it can return 1 more data
                if token_in_topk and num_top_prompt_logprobs != 0:
                    assert len(prompt_logprobs) == num_top_prompt_logprobs
                else:
                    assert len(prompt_logprobs) == num_top_prompt_logprobs + 1

                if num_top_prompt_logprobs > 0:
                    # We should have an entry for each of the topk ranks
                    all_ranks = {lp.rank for lp in prompt_logprobs.values()}
                    assert all(
                        r in all_ranks for r in range(1, num_top_prompt_logprobs + 1)
                    )

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
                        rtol=2e-2,
                    )
        else:
            assert vllm_result.prompt_logprobs is None


@pytest.mark.parametrize(
    "batch_logprobs_composition", [NONE, SAMPLE, PROMPT, SAMPLE_PROMPT]
)
@pytest.mark.parametrize("temperature", [0.0, 2.0])
def test_get_logprobs_and_prompt_logprobs(
    hf_model,
    vllm_model,
    batch_logprobs_composition: BatchLogprobsComposition,
    temperature: float,
    example_prompts: list[str],
) -> None:
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
    do_apc = vllm_model.llm.llm_engine.cache_config.enable_prefix_caching
    if do_apc and (temperature < 2.0 or batch_logprobs_composition != SAMPLE_PROMPT):
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
    logprob_prompt_logprob_list = get_test_batch(batch_logprobs_composition)

    # Ensure that each test prompt has a logprob config for testing
    logprob_prompt_logprob_list = _repeat_logprob_config(
        test_prompts, logprob_prompt_logprob_list
    )
    # Generate SamplingParams
    vllm_sampling_params = [
        SamplingParams(
            max_tokens=max_tokens,
            logprobs=num_lp,
            prompt_logprobs=num_plp,
            temperature=temperature,
            seed=1984,
        )
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
            do_apc=do_apc,
        )


def test_max_logprobs():
    """vLLM v1 engine should fail a request with `logprobs > max_logprobs`
    Should also fail for `prompt_logprobs > max_logprobs`
    APC should not matter as this test checks basic request validation.
    """
    runner = VllmRunner(
        "facebook/opt-125m",
        max_logprobs=1,
        enable_prefix_caching=False,
        # 2 other llms alive during whole session
        gpu_memory_utilization=0.15,
        max_model_len=256,
    )
    vllm_sampling_params = SamplingParams(logprobs=1)
    # should pass
    runner.generate(["Hello world"], sampling_params=vllm_sampling_params)

    bad_sampling_params = SamplingParams(logprobs=2)
    with pytest.raises(ValueError):
        runner.generate(["Hello world"], sampling_params=bad_sampling_params)


def test_none_logprobs(vllm_model, example_prompts):
    """Engine should return `logprobs` and `prompt_logprobs` as `None`

    Args:
      vllm_model: vLLM model fixture
      example_prompts: list of example prompts (test fixture)
    """
    max_tokens = 5

    sampling_params_logprobs_none = SamplingParams(
        max_tokens=max_tokens,
        logprobs=None,
        prompt_logprobs=None,
        temperature=0.0,
    )
    results_logprobs_none = vllm_model.llm.generate(
        example_prompts,
        sampling_params=sampling_params_logprobs_none,
    )

    for i in range(len(results_logprobs_none)):
        # Check sample logprobs are None
        assert results_logprobs_none[i].outputs[0].logprobs is None
        assert results_logprobs_none[i].outputs[0].cumulative_logprob is None
        # Check prompt logprobs are None
        assert results_logprobs_none[i].prompt_logprobs is None


def test_zero_logprobs(vllm_model, example_prompts):
    """Engine should return sampled token and prompt token logprobs

    Args:
      vllm_model: vLLM model fixture
      example_prompts: list of example prompts (test fixture)
    """
    max_tokens = 5

    sampling_params_logprobs_zero = SamplingParams(
        max_tokens=max_tokens, logprobs=0, prompt_logprobs=0, temperature=0.0
    )
    results_logprobs_zero = vllm_model.llm.generate(
        example_prompts, sampling_params=sampling_params_logprobs_zero
    )

    for i in range(len(results_logprobs_zero)):
        # Check that there is one sample logprob dict for each
        # sample token
        logprobs = results_logprobs_zero[i].outputs[0].logprobs
        prompt_logprobs = results_logprobs_zero[i].prompt_logprobs
        sampled_token_ids = results_logprobs_zero[i].outputs[0].token_ids
        prompt_token_ids = results_logprobs_zero[i].prompt_token_ids
        assert logprobs is not None
        assert len(sampled_token_ids) == len(logprobs)
        assert results_logprobs_zero[i].outputs[0].cumulative_logprob is not None
        # Check that there is one prompt logprob dict for each
        # prompt token
        assert prompt_logprobs is not None
        assert len(prompt_token_ids) == len(prompt_logprobs)


def test_all_logprobs(example_prompts):
    """Engine should return all vocabulary logprobs and prompt logprobs

    Args:
      example_prompts: list of example prompts (test fixture)
    """
    runner = VllmRunner(
        "facebook/opt-125m",
        max_logprobs=-1,
        enable_prefix_caching=False,
        # 2 other llms alive during whole session
        gpu_memory_utilization=0.15,
        max_model_len=256,
    )

    sampling_params_logprobs_all = SamplingParams(
        max_tokens=5, logprobs=-1, prompt_logprobs=-1
    )
    results_logprobs_all = runner.llm.generate(
        example_prompts, sampling_params=sampling_params_logprobs_all
    )
    vocab_size = runner.llm.llm_engine.model_config.get_vocab_size()

    for i in range(len(results_logprobs_all)):
        logprobs = results_logprobs_all[i].outputs[0].logprobs
        prompt_logprobs = results_logprobs_all[i].prompt_logprobs
        assert logprobs is not None
        for logprob in logprobs:
            assert len(logprob) == vocab_size
        assert prompt_logprobs is not None
        assert prompt_logprobs[0] is None
        for prompt_logprob in prompt_logprobs[1:]:
            assert len(prompt_logprob) == vocab_size


@pytest.mark.parametrize("logprobs_mode", get_args(LogprobsMode))
def test_logprobs_mode(logprobs_mode: LogprobsMode):
    """Test with LLM engine with different logprobs_mode.
    For logprobs, we should have non-positive values.
    For logits, we should expect at least one positive values.
    """
    from vllm import LLM

    llm = LLM(
        "facebook/opt-125m",
        max_logprobs=5,
        enable_prefix_caching=False,
        # 2 other llms alive during whole session
        gpu_memory_utilization=0.05,
        max_model_len=16,
        logprobs_mode=logprobs_mode,
    )
    vllm_sampling_params = SamplingParams(logprobs=1)
    results = llm.generate(["Hello world"], sampling_params=vllm_sampling_params)

    total_token_with_logprobs = 0
    positive_values = 0
    for output in results[0].outputs:
        for logprobs in output.logprobs:
            for token_id in logprobs:
                logprob = logprobs[token_id]
                if logprobs_mode in ("raw_logprobs", "processed_logprobs"):
                    assert logprob.logprob <= 0
                if logprob.logprob > 0:
                    positive_values = positive_values + 1
                total_token_with_logprobs = total_token_with_logprobs + 1
    assert total_token_with_logprobs >= len(results[0].outputs)
    if logprobs_mode in ("raw_logits", "processed_logits"):
        assert positive_values > 0
    del llm


class TestCorrectDecodedToken:
    """Unit tests for _correct_decoded_token method in LogprobsProcessor.

    This method handles UTF-8 decoding issues where incomplete byte sequences
    result in the Unicode replacement character "�" (U+FFFD). This commonly
    happens with byte-fallback tokenization when multi-byte UTF-8 characters
    are split across tokens.
    """

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        from unittest.mock import Mock

        tokenizer = Mock()
        return tokenizer

    @pytest.fixture
    def processor_with_empty_logprobs(self, mock_tokenizer):
        """Create a LogprobsProcessor with empty logprobs."""
        from vllm.v1.engine.logprobs import LogprobsProcessor

        processor = LogprobsProcessor(
            tokenizer=mock_tokenizer,
            logprobs=[],
            prompt_logprobs=None,
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=None,
        )
        return processor

    @pytest.fixture
    def processor_with_previous_logprobs(self, mock_tokenizer):
        """Create a LogprobsProcessor with previous logprobs."""
        from vllm.v1.engine.logprobs import LogprobsProcessor

        processor = LogprobsProcessor(
            tokenizer=mock_tokenizer,
            logprobs=[{123: None}],  # Previous token ID is 123
            prompt_logprobs=None,
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=None,
        )
        return processor

    def test_correction_with_previous_token_in_list(
        self, processor_with_empty_logprobs
    ):
        """Test correction using previous token in the same list.

        Scenario: Token at idx=1 ends with "�", but when decoded with
        the previous token (idx=0), it forms a valid UTF-8 sequence.
        Example: token[0]="�", token[1]="�" -> together form "polarized"
        """
        processor = processor_with_empty_logprobs
        tokens = [100, 101, 102]  # token IDs

        # Mock tokenizer behavior:
        # - decode([102]) returns "�" (ends with replacement char)
        # - decode([101, 102]) returns "valid" (no replacement char)
        processor.tokenizer.decode.side_effect = lambda ids: (
            "valid" if ids == [101, 102] else "�"
        )

        result = processor._correct_decoded_token(2, tokens)
        assert result == "valid"
        processor.tokenizer.decode.assert_called_with([101, 102])

    def test_correction_with_previous_logprob_token(
        self, processor_with_previous_logprobs
    ):
        """Test correction using previous logprob token.

        Scenario: Cannot correct with previous token in list (idx=0),
        but can correct with previous logprob token.
        """
        processor = processor_with_previous_logprobs
        tokens = [100]  # single token

        # Mock tokenizer behavior:
        # - decode([100]) returns "�" (ends with replacement char)
        # - decode([123, 100]) returns " "polarized" (no replacement char)
        # Token 123 is from previous logprobs
        def mock_decode(ids):
            if ids == [123, 100]:
                return ' "polarized"'
            return "�"

        processor.tokenizer.decode.side_effect = mock_decode

        result = processor._correct_decoded_token(0, tokens)
        assert result == ' "polarized"'

    def test_correction_at_idx_zero_no_previous_logprobs(
        self, processor_with_empty_logprobs
    ):
        """Test correction at idx=0 with no previous logprobs.

        Scenario: First token in list, no previous logprobs available.
        Should return empty string as fallback.
        """
        processor = processor_with_empty_logprobs
        tokens = [100]

        # Mock tokenizer always returns "�"
        processor.tokenizer.decode.return_value = "�"

        result = processor._correct_decoded_token(0, tokens)
        assert result == ""

    def test_correction_at_idx_zero_with_previous_logprobs(
        self, processor_with_previous_logprobs
    ):
        """Test correction at idx=0 with previous logprobs available.

        Scenario: First token in list, but previous logprobs exist.
        Should try correction with previous logprob token.
        """
        processor = processor_with_previous_logprobs
        tokens = [200]

        # Mock tokenizer behavior
        def mock_decode(ids):
            if ids == [123, 200]:
                return "corrected"
            return "�"

        processor.tokenizer.decode.side_effect = mock_decode

        result = processor._correct_decoded_token(0, tokens)
        assert result == "corrected"

    def test_no_correction_needed_returns_fallback(
        self, processor_with_previous_logprobs
    ):
        """Test fallback to empty string when no correction works.

        Scenario: All correction attempts still end with "�".
        Should return empty string as final fallback.
        """
        processor = processor_with_previous_logprobs
        tokens = [100, 101, 102]

        # Mock tokenizer always returns text ending with "�"
        processor.tokenizer.decode.return_value = "still�"

        result = processor._correct_decoded_token(2, tokens)
        assert result == ""

    def test_middle_token_correction(self, processor_with_previous_logprobs):
        """Test correction for a token in the middle of the list.

        Scenario: Token at idx=5 in a longer list needs correction.
        """
        processor = processor_with_previous_logprobs
        tokens = [10, 20, 30, 40, 50, 60, 70, 80]

        # Mock tokenizer behavior for middle token
        def mock_decode(ids):
            if ids == [50, 60]:
                return "olar"
            return "�"

        processor.tokenizer.decode.side_effect = mock_decode

        result = processor._correct_decoded_token(5, tokens)
        assert result == "olar"

    def test_multiple_consecutive_replacement_chars(
        self, processor_with_previous_logprobs
    ):
        """Test handling of multiple consecutive replacement characters.

        Scenario: Sequence like ["�", "�", "p"] where first two should
        become empty strings.
        """
        processor = processor_with_previous_logprobs

        # Test first replacement char
        tokens = [100, 101, 102]
        processor.tokenizer.decode.return_value = "still�"
        result1 = processor._correct_decoded_token(0, tokens)
        assert result1 == ""

        # Test second replacement char
        result2 = processor._correct_decoded_token(1, tokens)
        assert result2 == ""

    def test_correction_with_multibyte_utf8(self, processor_with_previous_logprobs):
        """Test correction involving multi-byte UTF-8 characters.

        Scenario: Byte-fallback tokenization splits multi-byte UTF-8
        characters (e.g., curly quotes, Chinese characters, emojis).
        Example from user: "�", "�" -> "", "\""
        """
        processor = processor_with_previous_logprobs
        tokens = [200, 201]

        # Mock tokenizer behavior for multi-byte UTF-8 correction
        def mock_decode(ids):
            # When decoding first token (idx=0) with previous logprob token
            if ids == [123, 200]:
                return ' "'  # Space + left curly quote
            # When decoding second token (idx=1) with previous token in list
            elif ids == [200, 201]:
                return '"'  # Right curly quote
            # When decoding second token (idx=1) with previous logprob + prev token
            elif ids == [123, 200, 201]:
                return ' ""'  # Full sequence
            return "�"

        processor.tokenizer.decode.side_effect = mock_decode

        # First token correction (idx=0)
        # Will call decode([123, 200]) since idx=0 uses previous logprob token
        result1 = processor._correct_decoded_token(0, tokens)
        assert result1 == ' "'

        # Second token correction (idx=1)
        # Will call decode([200, 201]) since idx>0 uses previous token in list
        result2 = processor._correct_decoded_token(1, tokens)
        assert result2 == '"'

    def test_real_world_opt125m_scenario(self, mock_tokenizer):
        """Test the real-world scenario from user's example.

        User's example with facebook/opt-125m:
        Before: [" the", " term", " �", "�", "p", "olar", "ized", "�", "�", ...]
        After: [" the", " term", "", " "", "p", "olar", "ized", "", "\"", ...]
        """
        from vllm.v1.engine.logprobs import LogprobsProcessor

        # Simulate the sequence of tokens
        processor = LogprobsProcessor(
            tokenizer=mock_tokenizer,
            logprobs=[],
            prompt_logprobs=None,
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=None,
        )

        # Token IDs representing the problematic sequence
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # placeholder IDs

        # Mock decode behavior simulating the real scenario
        def mock_decode(ids):
            # Simulate cases where individual tokens decode to "�"
            # but combinations decode correctly
            if len(ids) == 1:
                if ids[0] == 3 or ids[0] == 4 or ids[0] == 8 or ids[0] == 9:
                    return "�"
            elif len(ids) == 2:
                if ids == [2, 3]:
                    return " term�"  # Still ends with �, need more context
                elif ids == [3, 4]:
                    return ' "'  # Corrected to space + left curly quote
                elif ids == [7, 8]:
                    return "ized�"  # Still ends with �
                elif ids == [8, 9]:
                    return '"'  # Corrected to right curly quote
            elif len(ids) == 3:
                if ids == [1, 2, 3]:
                    return " the term�"  # Still ends with issue
                elif ids == [2, 3, 4]:
                    return ' term "'  # With all context
            return "normal_text"

        mock_tokenizer.decode.side_effect = mock_decode

        # Test token at index 2 (should fail to correct, return "")
        # Token 3 individually is "�"
        # decode([2, 3]) = " term�" (still ends with �)
        # No previous logprobs, so fallback to ""
        result = processor._correct_decoded_token(2, tokens)
        assert result == ""

        # Test token at index 3 (should correct to " "")
        # Token 4 individually is "�"
        # decode([3, 4]) = " "" (corrected!)
        processor.logprobs = [{2: None}]  # Add previous logprob
        result = processor._correct_decoded_token(3, tokens)
        assert result == ' "'


def test_verify_tokens_integration():
    """Integration test for _verify_tokens with real model.

    This test validates that _verify_tokens correctly identifies and
    corrects tokens ending with the replacement character "�".
    Uses facebook/opt-125m which is known to produce these issues.
    """
    runner = VllmRunner(
        "facebook/opt-125m",
        max_logprobs=0,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.15,
        max_model_len=256,
    )

    # Use a prompt that triggers multi-byte UTF-8 issues
    # Based on user's example: "In this example,"
    test_prompts = ["In this example,"]

    sampling_params = SamplingParams(
        max_tokens=16,
        temperature=0,
        logprobs=0,
    )

    results = runner.llm.generate(test_prompts, sampling_params=sampling_params)

    # Verify that decoded tokens don't contain replacement characters
    for result in results:
        assert result.outputs[0].logprobs is not None
        for logprob_dict in result.outputs[0].logprobs:
            for token_id, logprob_info in logprob_dict.items():
                decoded_token = logprob_info.decoded_token
                # Decoded tokens should not end with replacement character
                # They should either be corrected or empty string
                assert not decoded_token.endswith("�"), (
                    f"Token {token_id} decoded to '{decoded_token}' which "
                    f"ends with replacement character"
                )
                # Decoded tokens should not contain lone replacement characters
                assert decoded_token != "�", (
                    f"Token {token_id} is a lone replacement character"
                )


def test_utf8_edge_cases_with_real_model():
    """Test various UTF-8 edge cases with a real model.

    Tests prompts that are likely to trigger byte-fallback tokenization
    and multi-byte UTF-8 splitting.
    """
    runner = VllmRunner(
        "facebook/opt-125m",
        max_logprobs=1,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.15,
        max_model_len=256,
    )

    # Prompts with various multi-byte UTF-8 characters
    test_prompts = [
        'Smart quotes: "Hello"',  # Curly quotes
        "Em dash — test",  # Em dash
        "Ellipsis… continues",  # Ellipsis
        "Chinese: 你好",  # Chinese characters
        "Emoji: 😀 🎉",  # Emojis
        'Mixed: "quoted" — with symbols',  # Mixed
    ]

    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0,
        logprobs=1,
    )

    results = runner.llm.generate(test_prompts, sampling_params=sampling_params)

    for i, result in enumerate(results):
        prompt = test_prompts[i]
        assert result.outputs[0].logprobs is not None

        # Check that no decoded tokens end with replacement character
        for logprob_dict in result.outputs[0].logprobs:
            for token_id, logprob_info in logprob_dict.items():
                decoded_token = logprob_info.decoded_token
                assert not decoded_token.endswith("�"), (
                    f"Prompt: '{prompt}'\n"
                    f"Token {token_id} decoded to '{decoded_token}' which "
                    f"ends with replacement character"
                )


def test_correct_decoded_token_preserves_valid_tokens():
    """Test that valid tokens (not ending with �) are not modified.

    The _correct_decoded_token method should only be called for tokens
    ending with "�", but this test verifies the broader _verify_tokens
    logic doesn't affect valid tokens.
    """
    runner = VllmRunner(
        "facebook/opt-125m",
        max_logprobs=2,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.15,
        max_model_len=256,
    )

    # Simple prompt with standard ASCII characters
    test_prompts = ["Hello world, this is a test."]

    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0,
        logprobs=2,
    )

    results = runner.llm.generate(test_prompts, sampling_params=sampling_params)

    for result in results:
        assert result.outputs[0].logprobs is not None

        # All decoded tokens should be valid strings
        for logprob_dict in result.outputs[0].logprobs:
            for token_id, logprob_info in logprob_dict.items():
                decoded_token = logprob_info.decoded_token
                # Valid tokens should be non-empty strings (or empty if corrected)
                assert isinstance(decoded_token, str)
                # Should not contain replacement character
                assert "�" not in decoded_token


@pytest.mark.parametrize("logprobs_mode", get_args(LogprobsMode))
@pytest.mark.parametrize(
    "model_setup",
    [
        pytest.param(
            (
                "eagle",
                "meta-llama/Llama-3.2-1B-Instruct",
                {
                    "method": "eagle",
                    "model": "nm-testing/Llama3_2_1B_speculator.eagle3",
                    "num_speculative_tokens": 3,
                },
                0,
            ),
            marks=large_gpu_mark(min_gb=32),
            id="eagle0",
        ),
        pytest.param(
            (
                "eagle",
                "meta-llama/Llama-3.2-1B-Instruct",
                {
                    "method": "eagle",
                    "model": "nm-testing/Llama3_2_1B_speculator.eagle3",
                    "num_speculative_tokens": 3,
                },
                3,
            ),
            marks=large_gpu_mark(min_gb=32),
            id="eagle3",
        ),
        pytest.param(
            (
                "ngram",
                "meta-llama/Llama-3.2-1B-Instruct",
                {
                    "method": "ngram",
                    "prompt_lookup_max": 5,
                    "prompt_lookup_min": 3,
                    "num_speculative_tokens": 3,
                },
                3,
            ),
            marks=large_gpu_mark(min_gb=32),
            id="ngram",
        ),
    ],
)
def test_spec_decode_logprobs(
    logprobs_mode: LogprobsMode,
    model_setup: tuple[str, str, dict, int],
):
    """Spec decode logprobs should match those of the base model.

    Args:
        logprobs_mode: logprobs mode.
        model_setup: Tuple of (method, base model name,
            speculative_config dict, top_logprobs).
    """
    from vllm import LLM

    method, model_name, spec_config, top_logprobs = model_setup

    prompt = "Hello world " * 50
    sampling_params = SamplingParams(
        temperature=0, logprobs=top_logprobs, max_tokens=10, ignore_eos=False
    )
    penalty_sampling_params = SamplingParams(
        temperature=0,
        logprobs=top_logprobs,
        max_tokens=10,
        ignore_eos=False,
        presence_penalty=-1.0,
    )

    max_model_len = 256

    # Run base LLM.
    ref_llm = LLM(
        model=model_name,
        max_logprobs=5,
        max_model_len=max_model_len,
        seed=42,
        logprobs_mode=logprobs_mode,
        gpu_memory_utilization=0.4,
        enable_prefix_caching=False,
    )
    ref_results = ref_llm.generate(
        [prompt, prompt], [sampling_params, penalty_sampling_params]
    )
    # Collect logprobs outputs from reference LLM.
    ref_logprobs = []
    for results in ref_results:
        for output in results.outputs:
            for logprobs in output.logprobs:
                ref_logprobs.extend(logprobs.values())
    del ref_llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    # Run spec decode LLM.
    # Add max_model_len to spec_config if not present
    spec_config_with_len = {**spec_config, "max_model_len": max_model_len}
    spec_llm = LLM(
        model_name,
        speculative_config=spec_config_with_len,
        max_logprobs=5,
        max_model_len=max_model_len,
        seed=42,
        logprobs_mode=logprobs_mode,
        gpu_memory_utilization=0.4,
        # Force prefill chunking
        enable_chunked_prefill=True,
        max_num_batched_tokens=32,
        enable_prefix_caching=False,
    )
    spec_results = spec_llm.generate(
        [prompt, prompt], [sampling_params, penalty_sampling_params]
    )
    # Collect logprobs outputs from spec decode LLM.
    spec_logprobs = []
    for results in spec_results:
        for output in results.outputs:
            for logprobs in output.logprobs:
                spec_logprobs.extend(logprobs.values())
    del spec_llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    # Per-token logprobs are expected to be the same.
    assert len(ref_logprobs) == len(spec_logprobs)
    for ref_logprob, spec_logprob in zip(ref_logprobs, spec_logprobs):
        assert math.isclose(
            ref_logprob.logprob, spec_logprob.logprob, rel_tol=5e-2, abs_tol=1e-1
        )
        assert ref_logprob.rank == spec_logprob.rank
        assert ref_logprob.decoded_token == spec_logprob.decoded_token


def test_prompt_logprobs_with_chunking_and_preemption():
    """Test that prompt logprobs are correctly returned when using
    both chunked prefill and preemption.

    This test ensures that the num_prompt_logprobs tracking persists
    across preemptions and prefill chunks.
    """

    # Create prompts that will trigger chunking and preemption
    prompts = [
        "The following numbers of the sequence "
        + ", ".join(str(i) for i in range(10))
        + " are:",
        "In one word, the capital of France is ",
    ] + [f"Tell me about the number {i}: " for i in range(32)]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=40,
        min_tokens=20,
        prompt_logprobs=2,  # Request prompt logprobs
    )

    with VllmRunner(
        "Qwen/Qwen3-0.6B",
        max_model_len=512,
        enable_chunked_prefill=True,
        max_num_batched_tokens=48,  # Force prefill chunking
        num_gpu_blocks_override=32,  # Force preemptions
        disable_log_stats=False,
        gpu_memory_utilization=0.25,
    ) as vllm_model:
        metrics_before = vllm_model.llm.get_metrics()

        # Generate with prompt logprobs using generate_w_logprobs which
        # returns (output_ids, output_str, output_logprobs, prompt_logprobs)
        outputs = vllm_model.generate_w_logprobs(
            prompts, sampling_params=sampling_params, include_prompt_token_ids=True
        )

        # Verify that all outputs have prompt logprobs
        for i, output in enumerate(outputs):
            _, _, _, prompt_token_ids, prompt_logprobs = output
            assert prompt_logprobs is not None and len(prompt_logprobs) > 0, (
                f"Output {i} missing prompt logprobs"
            )
            assert len(prompt_logprobs) == len(prompt_token_ids), (
                "Unexpected number of prompt logprob positions"
            )

            # Each position should have the requested number of logprobs
            for pos, logprobs_dict in enumerate(prompt_logprobs):
                if logprobs_dict is not None:  # First token may be None
                    assert (
                        sampling_params.prompt_logprobs
                        <= len(logprobs_dict)
                        <= sampling_params.prompt_logprobs + 1
                    ), (
                        f"Output {i} position {pos} has {len(logprobs_dict)} "
                        f"logprobs, expected {sampling_params.prompt_logprobs}"
                    )

        # Check that we actually had preemptions
        metrics_after = vllm_model.llm.get_metrics()
        preemptions_before = next(
            (m.value for m in metrics_before if m.name == "vllm_num_preemptions"), 0
        )
        preemptions_after = next(
            (m.value for m in metrics_after if m.name == "vllm_num_preemptions"), 0
        )
        preemptions = preemptions_after - preemptions_before
        assert preemptions > 0, "Test did not trigger any preemptions"

        print(f"Test passed with {preemptions} preemptions")
