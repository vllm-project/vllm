import pytest

import vllm
from vllm.prompt_adapter.request import PromptAdapterRequest

MODEL_PATH = "bigscience/bloomz-560m"
PA_PATH = 'stevhliu/bloomz-560m_PROMPT_TUNING_CAUSAL_LM'


def do_sample(llm, pa_name: str, pa_id: int):

    prompts = [
        "Tweet text : @nationalgridus I have no water and the bill is \
        current and paid. Can you do something about this? Label : ",
        "Tweet text : @nationalgridus Looks good thanks! Label : "
    ]
    sampling_params = vllm.SamplingParams(temperature=0.0,
                                          max_tokens=3,
                                          stop_token_ids=[3])

    outputs = llm.generate(prompts,
                           sampling_params,
                           prompt_adapter_request=PromptAdapterRequest(
                               pa_name, pa_id, PA_PATH, 8) if pa_id else None)

    # Print the outputs.
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_twitter_prompt_adapter(enforce_eager: bool):
    llm = vllm.LLM(MODEL_PATH,
                   enforce_eager=enforce_eager,
                   enable_prompt_adapter=True,
                   max_prompt_adapter_token=8)

    expected_output = ['complaint', 'no complaint']

    assert do_sample(llm, "twitter_pa", pa_id=1) == expected_output
