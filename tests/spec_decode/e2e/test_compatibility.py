import pytest

from vllm import SamplingParams

from .conftest import get_output_from_llm_generator


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-68m",
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,

        # Required for spec decode.
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            # Expect failure as spec decode not supported by
            # Ray backend.
            "worker_use_ray": True,
        },
    ])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_xfail_ray(test_llm_generator):
    """Verify that speculative decoding with Ray fails.
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

    try:
        with pytest.raises(
                AssertionError,
                match="Speculative decoding not yet supported for "):
            get_output_from_llm_generator(test_llm_generator, prompts,
                                          sampling_params)
    finally:
        # we need to free up ray resource,
        # so that latter test could use the gpu we allocated here
        import ray
        ray.shutdown()


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-68m",
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,

        # Required for spec decode.
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    {
        "enable_chunked_prefill": True,
    },
])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_xfail_chunked_prefill(test_llm_generator):
    """Verify that speculative decoding with chunked prefill fails.
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

    with pytest.raises(ValueError,
                       match="Speculative decoding and chunked prefill"):
        get_output_from_llm_generator(test_llm_generator, prompts,
                                      sampling_params)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,

        # Required for spec decode.
        "use_v2_block_manager": True
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


@pytest.mark.parametrize("common_llm_kwargs", [{
    "model": "JackFram/llama-68m",
    "speculative_model": "JackFram/llama-68m",
    "num_speculative_tokens": 5,
}])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_xfail_block_manager_v1(test_llm_generator):
    """Verify that speculative decoding with block manager v1 fails.
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

    with pytest.raises(ValueError,
                       match="Speculative decoding requires usage of the V2"):
        get_output_from_llm_generator(test_llm_generator, prompts,
                                      sampling_params)
