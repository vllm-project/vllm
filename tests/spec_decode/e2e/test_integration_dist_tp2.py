"""Tests which cover integration of the speculative decoding framework with
tensor parallelism.
"""

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
        "--tensor-parallel-size",
        "2"
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("test_llm_kwargs", [
    [
        "--speculative-model",
        "JackFram/llama-68m",
        "--num-speculative-tokens",
        "3",
    ],
    [
        "--speculative-model",
        "[ngram]",
        "--num-speculative-tokens",
        "5",
        "--ngram-prompt-lookup-max",
        "3",
    ],
])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_target_model_tp_gt_1(common_llm_kwargs, per_test_common_llm_kwargs,
                              baseline_llm_kwargs, test_llm_kwargs,
                              batch_size: int, output_len: int, seed: int):
    """Verify greedy equality when tensor parallelism is used.
    """
    if current_platform.is_rocm():
        pytest.skip("hip is not well-supported yet")
    run_equality_correctness_test_tp("JackFram/llama-68m",
                                     common_llm_kwargs,
                                     per_test_common_llm_kwargs,
                                     baseline_llm_kwargs,
                                     test_llm_kwargs,
                                     batch_size,
                                     output_len,
                                     seed,
                                     temperature=0.0)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[
        # Skip cuda graph recording for fast test.
        "--enforce-eager",
        "--tensor_parallel_size",
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
                             "--num_speculative-tokens",
                             "5",
                             "--speculative-draft-tensor-parallel-size",
                             "1",
                         ]),
                          ("ibm-granite/granite-3b-code-instruct", [
                              "--speculative-model",
                              "ibm-granite/granite-3b-code-instruct",
                              "--num_speculative-tokens",
                              "5",
                              "--speculative-draft-tensor-parallel-size",
                              "1",
                          ])])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seed", [1])
def test_draft_model_tp_lt_target_model_tp2(model, common_llm_kwargs,
                                            per_test_common_llm_kwargs,
                                            baseline_llm_kwargs,
                                            test_llm_kwargs, batch_size: int,
                                            seed: int):
    """Verify spec decode works well with smaller tp for draft models.
    """
    run_equality_correctness_test_tp(model,
                                     common_llm_kwargs,
                                     per_test_common_llm_kwargs,
                                     baseline_llm_kwargs,
                                     test_llm_kwargs,
                                     batch_size,
                                     max_output_len=32,
                                     seed=seed,
                                     temperature=0.0)
