# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests which cover integration of the speculative decoding framework with
tensor parallelism.
"""

import json
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
        "--tensor-parallel-size",
        "2"
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("test_llm_kwargs", [
    [
        "--speculative_config",
        json.dumps({
            "model": "JackFram/llama-68m",
            "num_speculative_tokens": 3,
        }),
    ],
    [
        "--speculative_config",
        json.dumps({
            "model": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 3,
        }),
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
@pytest.mark.parametrize(
    "model, test_llm_kwargs",
    [("JackFram/llama-68m", [
        "--speculative_config",
        json.dumps({
            "model": "JackFram/llama-68m",
            "num_speculative_tokens": 5,
            "draft_tensor_parallel_size": 1,
        }),
    ]),
     ("ibm-granite/granite-3b-code-instruct", [
         "--speculative_config",
         json.dumps({
             "model": "ibm-granite/granite-3b-code-instruct",
             "num_speculative_tokens": 5,
             "draft_tensor_parallel_size": 1,
         }),
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
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [["--enable-chunked-prefill", "False"],
     [
         "--enable-chunked-prefill", "True", "--max-num-batched-tokens", "4",
         "--max-num-seqs", "4"
     ]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("model, test_llm_kwargs",
                         [("JackFram/llama-68m", [
                             "--speculative_config",
                             json.dumps({
                                 "model": "JackFram/llama-68m",
                                 "num_speculative_tokens": 3,
                             }),
                         ]),
                          ("JackFram/llama-68m", [
                              "--speculative_config",
                              json.dumps({
                                  "model": "JackFram/llama-68m",
                                  "num_speculative_tokens": 3,
                                  "draft_tensor_parallel_size": 1,
                              }),
                          ])])
@pytest.mark.parametrize("logprobs", [None])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_chunked_prefill_tp2(model, common_llm_kwargs,
                                         per_test_common_llm_kwargs,
                                         baseline_llm_kwargs, test_llm_kwargs,
                                         logprobs: Optional[int],
                                         batch_size: int, seed: int):
    """Verify spec decode works well with same and different TP size for
    the draft model with chunked prefill.
    """
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
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [["--enable-chunked-prefill", "False"],
     [
         "--enable-chunked-prefill", "True", "--max-num-batched-tokens", "4",
         "--max-num-seqs", "4"
     ]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("model, test_llm_kwargs",
                         [("JackFram/llama-68m", [
                             "--speculative_config",
                             json.dumps({
                                 "model": "JackFram/llama-68m",
                                 "num_speculative_tokens": 3,
                                 "disable_logprobs": False,
                             }),
                         ]),
                          ("JackFram/llama-68m", [
                              "--speculative_config",
                              json.dumps({
                                  "model": "JackFram/llama-68m",
                                  "num_speculative_tokens": 3,
                                  "draft_tensor_parallel_size": 1,
                                  "disable_logprobs": False,
                              }),
                          ])])
@pytest.mark.parametrize("logprobs", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_chunked_prefill_tp2_with_logprobs(
        model, common_llm_kwargs, per_test_common_llm_kwargs,
        baseline_llm_kwargs, test_llm_kwargs, logprobs: Optional[int],
        batch_size: int, seed: int):
    """Verify spec decode works well with same and different TP size for
    the draft model with chunked prefill.
    """
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
