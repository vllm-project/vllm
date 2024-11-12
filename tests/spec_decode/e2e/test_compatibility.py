import pytest

from vllm import SamplingParams

from .conftest import get_output_from_llm_generator


@pytest.mark.parametrize("common_llm_kwargs", [{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "speculative_model": "JackFram/llama-68m",
    "num_speculative_tokens": 5,
}])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            # Speculative max model len > overridden max model len should raise.
            "max_model_len": 128,
            "speculative_max_model_len": 129,
        },
        {
            # Speculative max model len > draft max model len should raise.
            # https://huggingface.co/JackFram/llama-68m/blob/3b606af5198a0b26762d589a3ee3d26ee6fa6c85/config.json#L12
            "speculative_max_model_len": 2048 + 1,
        },
        {
            # Speculative max model len > target max model len should raise.
            # https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/f5db02db724555f92da89c216ac04704f23d4590/config.json#L12
            "speculative_max_model_len": 4096 + 1,
        },
    ])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_xfail_spec_max_model_len(test_llm_generator):
    """Verify that speculative decoding validates speculative_max_model_len.
    """
    output_len = 128
    temperature = 0.0

    prompts = [
        "Hello, my name is",
    ]

    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=temperature,
    )

    with pytest.raises(ValueError, match="cannot be larger than"):
        get_output_from_llm_generator(test_llm_generator, prompts,
                                      sampling_params)


@pytest.mark.parametrize("common_llm_kwargs",
                         [{
                             "model": "meta-llama/Llama-2-7b-chat-hf",
                             "speculative_model": "JackFram/llama-68m",
                             "num_speculative_tokens": 5,
                             "enable_chunked_prefill": "True",
                         }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    {
        "tensor_parallel_size": 2,
        "speculative_draft_tensor_parallel_size": 2,
    },
    {
        "tensor_parallel_size": 4,
        "speculative_draft_tensor_parallel_size": 4,
    },
    {
        "tensor_parallel_size": 8,
        "speculative_draft_tensor_parallel_size": 8,
    },
])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_xfail_chunked_prefill_draft_model_tp_not_one(
        test_llm_generator):
    """Verify that speculative decoding fails if chunked prefill is enabled for 
    draft model with tensor parallelism of more than 1.
    """
    output_len = 128
    temperature = 0.0

    prompts = [
        "Hello, my name is",
    ]

    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=temperature,
    )

    with pytest.raises(ValueError, match="with tensor parallel size 1"):
        get_output_from_llm_generator(test_llm_generator, prompts,
                                      sampling_params)
