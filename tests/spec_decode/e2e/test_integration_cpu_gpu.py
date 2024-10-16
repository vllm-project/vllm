"""Tests which cover integration of the speculative decoding framework with
other features, e.g. cuda graphs.
"""

import pytest

from .conftest import run_equality_correctness_test

MAIN_MODEL = "facebook/opt-125m"


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Required for spec decode.
        "use_v2_block_manager": True,

        # Verify equality when cuda graphs allowed.
        "enforce_eager": False,
        "model_name": "facebook/opt-125m",
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            # Identical models.
            "speculative_model": "facebook/opt-125m",
            "num_speculative_tokens": 5,
            "cpu_draft_worker": True,
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("output_len", [32])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_cuda_graph(vllm_runner, common_llm_kwargs,
                                per_test_common_llm_kwargs,
                                baseline_llm_kwargs, test_llm_kwargs,
                                batch_size: int, output_len: int, seed: int):
    """Verify spec decode equality when cuda graphs are enabled.
    """
    run_equality_correctness_test(vllm_runner,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  max_output_len=output_len,
                                  seed=seed,
                                  temperature=0.0)
