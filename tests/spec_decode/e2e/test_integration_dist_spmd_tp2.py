"""
"""

import pytest
import torch
import os

from vllm.utils import is_hip

from .conftest import run_greedy_equality_correctness_test


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,
        # Required for spec decode.
        "use_v2_block_manager": True,
        "tensor_parallel_size": 2,

        # Use a small model for a fast test.
        "model": "JackFram/llama-68m",

        # Currently required for SPMD.
        "distributed_executor_backend": "ray",
    }])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
        "speculative_draft_tensor_parallel_size": 1,
    },
])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seed", [1])
def test_spmd_tp2(test_llm_generator, baseline_llm_generator, batch_size: int):
    """Verify ray accelerated dag + spec decode works
    """

    assert os.getenv("VLLM_USE_RAY_SPMD_WORKER",
                     "0") == "1", "test requires SPMD worker"
    assert os.getenv("VLLM_USE_RAY_COMPILED_DAG",
                     "0") == "1", "test currently requires Ray compiled dags"

    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len=32,
                                         force_output_len=True)
