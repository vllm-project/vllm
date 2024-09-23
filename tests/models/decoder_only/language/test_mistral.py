"""Compare the outputs of HF and vLLM for Mistral models using greedy sampling.

Run `pytest tests/models/test_mistral.py`.
"""
import pytest

from vllm import LLM, SamplingParams

from ...utils import check_logprobs_close

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.3",
    # Mistral-Nemo is to big for CI, but passes locally
    # "mistralai/Mistral-Nemo-Instruct-2407"
]

SAMPLING_PARAMS = SamplingParams(max_tokens=512, temperature=0.0, logprobs=5)
SYMBOLIC_LANG_PROMPTS = [
    "勇敢な船乗りについての詩を書く",  # japanese
    "寫一首關於勇敢的水手的詩",  # chinese
]

# for function calling
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type":
                    "string",
                    "description":
                    "The city to find the weather for, e.g. 'San Francisco'"
                },
                "state": {
                    "type":
                    "string",
                    "description":
                    "the two-letter abbreviation for the state that the city is"
                    " in, e.g. 'CA' which would mean 'California'"
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["city", "state", "unit"]
        }
    }
}]
MSGS = [{
    "role":
    "user",
    "content": ("Can you tell me what the temperate"
                " will be in Dallas, in fahrenheit?")
}]
EXPECTED_FUNC_CALL = (
    '[{"name": "get_current_weather", "arguments": '
    '{"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]')


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    # TODO(sang): Sliding window should be tested separately.
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    with vllm_runner(model, dtype=dtype,
                     tokenizer_mode="mistral") as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS[1:])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_mistral_format(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with vllm_runner(
            model,
            dtype=dtype,
            tokenizer_mode="auto",
            load_format="safetensors",
            config_format="hf",
    ) as hf_format_model:
        hf_format_outputs = hf_format_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    with vllm_runner(
            model,
            dtype=dtype,
            tokenizer_mode="mistral",
            load_format="mistral",
            config_format="mistral",
    ) as mistral_format_model:
        mistral_format_outputs = mistral_format_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=hf_format_outputs,
        outputs_1_lst=mistral_format_outputs,
        name_0="hf",
        name_1="mistral",
    )


@pytest.mark.parametrize("model", MODELS[1:])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("prompt", SYMBOLIC_LANG_PROMPTS)
def test_mistral_symbolic_languages(
    model: str,
    dtype: str,
    prompt: str,
) -> None:
    prompt = "hi"
    msg = {"role": "user", "content": prompt}
    llm = LLM(model=model,
              dtype=dtype,
              max_model_len=8192,
              tokenizer_mode="mistral",
              config_format="mistral",
              load_format="mistral")
    outputs = llm.chat([msg], sampling_params=SAMPLING_PARAMS)
    assert "�" not in outputs[0].outputs[0].text.strip()


@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("model", MODELS[1:])  # v1 can't do func calling
def test_mistral_function_calling(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model,
                     dtype=dtype,
                     tokenizer_mode="mistral",
                     config_format="mistral",
                     load_format="mistral") as vllm_model:
        outputs = vllm_model.model.chat(MSGS,
                                        tools=TOOLS,
                                        sampling_params=SAMPLING_PARAMS)

        assert outputs[0].outputs[0].text.strip() == EXPECTED_FUNC_CALL
