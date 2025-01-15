"""Tests which cover integration of the speculative decoding framework with
tensor parallelism.
"""

import openai
import pytest
import torch

from .conftest import run_equality_correctness_test_tp

MAIN_MODEL = "JackFram/llama-68m"
SPEC_MODEL = "JackFram/llama-68m"


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[
        # Skip cuda graph recording for fast test.
        "--enforce_eager",
        "--tensor-parallel-size",
        "4",
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    [
        "--speculative-model",
        f"{SPEC_MODEL}",
        "--num-speculative-tokens",
        "5",
    ],
])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        #TODO(wooyeon): add spec_draft_dp=2 case
        [
            "--speculative-draft-tensor-parallel-size",
            "1",
        ],
    ])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seed", [1])
def test_draft_model_tp_lt_target_model_tp4(common_llm_kwargs,
                                            per_test_common_llm_kwargs,
                                            baseline_llm_kwargs,
                                            test_llm_kwargs, batch_size: int,
                                            seed: int):
    """Verify spec decode works well with smaller tp for draft models.
    """
    run_equality_correctness_test_tp(MAIN_MODEL,
                                     common_llm_kwargs,
                                     per_test_common_llm_kwargs,
                                     baseline_llm_kwargs,
                                     test_llm_kwargs,
                                     batch_size,
                                     max_output_len=32,
                                     seed=seed,
                                     temperature=0.0)


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[

        # Skip cuda graph recording for fast test.
        "--enforce-eager",
        "--tensor-parallel-size",
        "4",
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        [
            "--speculative-model",
            f"{SPEC_MODEL}",
            "--num-speculative-tokens",
            "5",

            # Artificially limit the draft model max model len; this forces vLLM
            # to skip speculation once the sequences grow beyond 32-k tokens.
            "--speculative-max-model-len",
            "32",
        ],
    ])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize(
    "output_len",
    [
        # This must be a good bit larger than speculative_max_model_len so that
        # we can test the case where all seqs are skipped, but still small to
        # ensure fast test.
        64,
    ])
@pytest.mark.parametrize("seed", [1])
def test_skip_speculation(common_llm_kwargs, per_test_common_llm_kwargs,
                          baseline_llm_kwargs, test_llm_kwargs,
                          batch_size: int, output_len: int, seed: int):
    """Verify job failure with RuntimeError when all sequences skip speculation.
    We do this by setting the max model len of the draft model to an
    artificially low value, such that when the sequences grow beyond it, they
    are skipped in speculative decoding.

    TODO: fix it to pass without raising Error. (#5814)
    """
    with pytest.raises(
        (openai.APIConnectionError, openai.InternalServerError)):
        run_equality_correctness_test_tp(MAIN_MODEL,
                                         common_llm_kwargs,
                                         per_test_common_llm_kwargs,
                                         baseline_llm_kwargs,
                                         test_llm_kwargs,
                                         batch_size,
                                         output_len,
                                         seed,
                                         temperature=0.0)
