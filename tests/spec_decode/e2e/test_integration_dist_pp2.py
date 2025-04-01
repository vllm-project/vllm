# SPDX-License-Identifier: Apache-2.0
"""Tests which cover integration of the speculative decoding framework with
pipeline parallelism.
"""

from typing import Optional

import pytest
import torch

from vllm.platforms import current_platform

from .conftest import run_equality_correctness_test_tp


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[
        # Skip cuda graph recording for fast test.
        "--enforce-eager",
        "--pipeline-parallel-size",
        "2",

        # precision
        "--dtype",
        "bfloat16",
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("model, test_llm_kwargs",
                         [("JackFram/llama-68m", [
                             "--speculative-model",
                             "JackFram/llama-68m",
                             "--num-speculative-tokens",
                             "5",
                             "--speculative-draft-pipeline-parallel-size",
                             "1",
                         ]),
                          ("ibm-granite/granite-3b-code-instruct", [
                              "--speculative-model",
                              "ibm-granite/granite-3b-code-instruct",
                              "--num-speculative-tokens",
                              "5",
                              "--speculative-draft-pipeline-parallel-size",
                              "1",
                          ])])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seed", [1])
def test_draft_model_pp_lt_target_model_pp2(model, common_llm_kwargs,
                                            per_test_common_llm_kwargs,
                                            baseline_llm_kwargs,
                                            test_llm_kwargs, batch_size: int,
                                            seed: int):
    """Verify spec decode works well with smaller pp for draft models.
    """
    if current_platform.is_rocm():
        pytest.skip("hip is not well-supported yet")
    run_equality_correctness_test_tp(model,
                                     common_llm_kwargs,
                                     per_test_common_llm_kwargs,
                                     baseline_llm_kwargs,
                                     test_llm_kwargs,
                                     batch_size,
                                     max_output_len=32,
                                     seed=seed,
                                     temperature=0.0)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[
        # Skip cuda graph recording for fast test.
        "--enforce-eager",
        "--pipeline-parallel-size",
        "2",

        # precision
        "--dtype",
        "bfloat16",
    ]])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [["--enable-chunked-prefill", "False"],
     [
         "--enable-chunked-prefill", "True", "--max-num-batched-tokens", "4",
         "--max-num-seqs", "4"
     ]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("model, test_llm_kwargs", [("JackFram/llama-68m", [
    "--speculative-model",
    "JackFram/llama-68m",
    "--num-speculative-tokens",
    "3",
    "--speculative-draft-pipeline-parallel-size",
    "1",
])])
@pytest.mark.parametrize("logprobs", [None, 2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_chunked_prefill_pp2(model, common_llm_kwargs,
                                         per_test_common_llm_kwargs,
                                         baseline_llm_kwargs, test_llm_kwargs,
                                         logprobs: Optional[int],
                                         batch_size: int, seed: int):
    """Verify spec decode works well with same and different PP size for
    the draft model with chunked prefill.
    """
    if logprobs:
        test_llm_kwargs.extend(
            ["--disable-logprobs-during-spec-decoding", "False"])
    run_equality_correctness_test_tp(model,
                                     common_llm_kwargs,
                                     per_test_common_llm_kwargs,
                                     baseline_llm_kwargs,
                                     test_llm_kwargs,
                                     batch_size,
                                     max_output_len=32,
                                     seed=seed,
                                     temperature=0.0,
                                     logprobs=logprobs)
