"""Compare the outputs of HF and vLLM for BART models using greedy sampling.

Run `pytest tests/models/encoder_decoder/language/test_bart.py`.
"""
import pytest

from tests.utils import multi_gpu_test

from ....conftest import DecoderPromptType
from .conftest import compare_hf_vllm_logprobs


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("facebook/bart-base",
                     marks=[pytest.mark.core_model, pytest.mark.cpu_model]),
        pytest.param("facebook/bart-large-cnn"),
    ],
)
@pytest.mark.parametrize("dtype", ["float", "bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("decoder_prompt_type", list(DecoderPromptType))
def test_models(hf_runner, vllm_runner, example_encoder_decoder_prompts, model,
                dtype, max_tokens, num_logprobs, decoder_prompt_type) -> None:

    compare_hf_vllm_logprobs(
        hf_runner,
        vllm_runner,
        example_encoder_decoder_prompts[decoder_prompt_type],
        decoder_prompt_type,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
        hf_tokens_to_skip=int(decoder_prompt_type == DecoderPromptType.NONE))


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("distributed_executor_backend", ["ray", "mp"])
@pytest.mark.parametrize("model", ["facebook/bart-large-cnn"])
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("decoder_prompt_type", [DecoderPromptType.CUSTOM])
def test_models_distributed(hf_runner, vllm_runner,
                            example_encoder_decoder_prompts,
                            distributed_executor_backend, model, dtype,
                            max_tokens, num_logprobs,
                            decoder_prompt_type) -> None:
    compare_hf_vllm_logprobs(
        hf_runner,
        vllm_runner,
        example_encoder_decoder_prompts[decoder_prompt_type],
        decoder_prompt_type,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=2,
        distributed_executor_backend=distributed_executor_backend,
    )
