"""Compare the outputs of HF and vLLM for T5 models using greedy sampling.
Based on tests/models/encoder_decoder/language/test_bart.py.

Run `pytest tests/models/encoder_decoder/language/test_t5.py`.
"""
import pytest

from tests.utils import multi_gpu_test
from vllm.attention.selector import (_Backend,
                                     global_force_attn_backend_context_manager)

from ....conftest import DecoderPromptType
from .conftest import compare_hf_vllm_logprobs


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("google-t5/t5-small"),
        pytest.param("google/flan-t5-base"),
    ],
)
@pytest.mark.parametrize("vllm_kwargs", [{"max_model_len": 512}])
@pytest.mark.parametrize("dtype", ["float", "bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
# TODO custom prompt here generate high entropy output, causing
# differences in sampled tokens.
@pytest.mark.parametrize("decoder_prompt_type",
                         [DecoderPromptType.NONE, DecoderPromptType.EMPTY_STR])
def test_models(hf_runner, vllm_runner, example_encoder_decoder_prompts, model,
                dtype, max_tokens, num_logprobs, decoder_prompt_type,
                vllm_kwargs) -> None:
    # Model only supported on xformers backend as of now.
    with global_force_attn_backend_context_manager(_Backend.XFORMERS):
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
            vllm_runner_kwargs=vllm_kwargs)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("distributed_executor_backend", ["ray", "mp"])
@pytest.mark.parametrize("model", ["google/t5-small"])
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("decoder_prompt_type", [DecoderPromptType.NONE])
def test_models_distributed(hf_runner, vllm_runner,
                            example_encoder_decoder_prompts,
                            distributed_executor_backend, model, dtype,
                            max_tokens, num_logprobs,
                            decoder_prompt_type) -> None:
    with global_force_attn_backend_context_manager(_Backend.XFORMERS):
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
