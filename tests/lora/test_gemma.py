import pytest

import vllm
from tests.nm_utils.utils_skip import should_skip_test_group
from vllm.lora.request import LoRARequest

if should_skip_test_group(group_name="TEST_LORA"):
    pytest.skip("TEST_LORA=DISABLE, skipping lora test group",
                allow_module_level=True)

MODEL_PATH = "google/gemma-7b"


def do_sample(llm, lora_path: str, lora_id: int) -> str:
    prompts = [
        "Quote: Imagination is",
        "Quote: Be yourself;",
        "Quote: So many books,",
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None)
    # Print the outputs.
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


# UPSTREAM SYNC: skip this test in nm-automation
@pytest.mark.skip("Flaky test in NM automation")
def test_gemma_lora(gemma_lora_files):
    llm = vllm.LLM(MODEL_PATH,
                   max_model_len=1024,
                   enable_lora=True,
                   max_loras=4)

    expected_lora_output = [
        "more important than knowledge.\nAuthor: Albert Einstein\n",
        "everyone else is already taken.\nAuthor: Oscar Wilde\n",
        "so little time\nAuthor: Frank Zappa\n",
    ]

    output1 = do_sample(llm, gemma_lora_files, lora_id=1)
    for i in range(len(expected_lora_output)):
        assert output1[i].startswith(expected_lora_output[i])
    output2 = do_sample(llm, gemma_lora_files, lora_id=2)
    for i in range(len(expected_lora_output)):
        assert output2[i].startswith(expected_lora_output[i])
