import pytest

from vllm import SamplingParams


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        "model": "JackFram/llama-68m",
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,

        # Skip real loading for fast test.
        "load_format": "dummy",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    {
        "tensor_parallel_size": 1,
    },
])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode(test_llm_generator):
    output_len = 128
    temperature = 0.0

    prompts = [
        "Hello, my name is",
        #"The president of the United States is",
        #"The capital of France is",
        #"The future of AI is",
    ]

    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=temperature,
    )

    #with pytest.raises(
    #        AssertionError,
    #        match="Speculative decoding not yet supported for GPU backend"):
    get_token_ids_from_llm_generator(test_llm_generator, prompts,
                                     sampling_params)

@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        "model": "JackFram/llama-68m",
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,

        # Skip real loading for fast test.
        "load_format": "dummy",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    {
        # Expect failure as spec decode not supported by
        # Ray backend.
        "tensor_parallel_size": 2,
    },
])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_xfail(test_llm_generator):
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

    with pytest.raises(
            AssertionError,
            match="Speculative decoding not yet supported for "):
        get_token_ids_from_llm_generator(test_llm_generator, prompts,
                                         sampling_params)

def get_token_ids_from_llm_generator(llm_generator, prompts, sampling_params):
    for llm in llm_generator:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        token_ids = [output.outputs[0].token_ids for output in outputs]
        del llm

    return token_ids

