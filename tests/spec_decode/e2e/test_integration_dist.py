"""Tests which cover integration of the speculative decoding framework with
tensor parallelism.
"""

import pytest
import torch

from tests.nm_utils.utils_skip import should_skip_test_group
from vllm.utils import is_hip

from .conftest import run_greedy_equality_correctness_test

if should_skip_test_group(group_name="TEST_SPEC_DECODE"):
    pytest.skip("TEST_SPEC_DECODE=DISABLE, skipping spec decode group",
                allow_module_level=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,
        "tensor_parallel_size": 2,

        # Use AsyncLLM engine, so that the engine runs in its own process.
        # Otherwise, since vLLM does not follow true SPMD, the test runner
        # process will have both the engine and the rank0 worker. NCCL is not
        # cleaned up properly, and its server host thread leaks, causing the
        # second run of the test to fail with internal NCCL error.
        "use_async": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 3,
    },
    {
        "speculative_model": "[ngram]",
        "num_speculative_tokens": 5,
        "ngram_prompt_lookup_max": 3,
    },
])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_target_model_tp_gt_1(baseline_llm_generator, test_llm_generator,
                              batch_size: int, output_len: int):
    """Verify greedy equality when tensor parallelism is used.
    """
    if is_hip():
        pytest.skip("hip is not well-supported yet")
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=output_len,
                                         force_output_len=True)
