from vllm import LLM, SamplingParams


def test_fp8_offline_inference():
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM
    llm = LLM(
            model="/data/models/llama-2-7b-chat-hf",
            kv_cache_dtype="fp8",
            quantization_param_path = \
                    "./tests/fp8_kv/llama2-7b-fp8-kv/kv_cache_scales.json"
            )

    prompt = "London is the capital of"

    # Generate model response
    out = llm.generate(prompt, sampling_params)[0].outputs[0].text

    assert out == (" England and the United Kingdom."
                   " It is located in the southeastern part of")
