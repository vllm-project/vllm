# SPDX-License-Identifier: Apache-2.0
"""A basic correctness check for TPUs

Run `pytest tests/v1/tpu/test_basic.py`.
"""
import pytest

from vllm.platforms import current_platform

from ...conftest import VllmRunner

MODELS = [
    # "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Llama-3.1-8B",
    # TODO: Add models here as necessary
]

TENSOR_PARALLEL_SIZES = [1]

# TODO: Enable when CI/CD will have a multi-tpu instance
# TENSOR_PARALLEL_SIZES = [1, 4]


@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This is a basic test for TPU only")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("tensor_parallel_size", TENSOR_PARALLEL_SIZES)
def test_models(
    monkeypatch,
    model: str,
    max_tokens: int,
    enforce_eager: bool,
    tensor_parallel_size: int,
) -> None:
    prompt = "The next numbers of the sequence " + ", ".join(
        str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        with VllmRunner(
                model,
                max_model_len=8192,
                enforce_eager=enforce_eager,
                gpu_memory_utilization=0.7,
                max_num_seqs=16,
                tensor_parallel_size=tensor_parallel_size) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                      max_tokens)
    output = vllm_outputs[0][1]
    assert "1024" in output
