"""Compare the outputs of HF and distributed vLLM when using greedy sampling.
vLLM will allocate all the available memory, so we need to run the tests one
by one. The solution is to pass arguments (model name) by environment
variables.
Run:
```sh
TEST_DIST_MODEL=facebook/opt-125m pytest \
    test_flashinfer_distributed.py
TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf \
    test_flashinfer_distributed.py
```
"""
import os

import pytest
import torch

MODELS = [
    os.environ["TEST_DIST_MODEL"],
]


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [True])
def test_models(hf_runner, vllm_runner, example_prompts, model: str,
                dtype: str, max_tokens: int, enforce_eager: bool) -> None:
    try:
        import flash_attn  # noqa: F401
        import flashinfer  # noqa: F401
    except ImportError:
        pytest.skip(
            "Cannot use Flashinfer backend because the flashinfer package "
            "is not found. Please install both flashinfer and flash attention "
            "for running the test.")

    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    vllm_model = vllm_runner(model,
                             dtype=dtype,
                             tensor_parallel_size=2,
                             enforce_eager=enforce_eager)
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")
