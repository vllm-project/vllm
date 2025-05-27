# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

rope_theta = 1000000
original_max_position_embeddings = 32768
factor = 4.0

# Use yarn to extend context
hf_overrides = {
    "rope_theta": rope_theta,
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": original_max_position_embeddings,
    },
    "max_model_len": int(original_max_position_embeddings * factor),
}

llm = LLM(model="Qwen/Qwen3-0.6B", hf_overrides=hf_overrides)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=128,
)

conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
]
outputs = llm.chat(conversation, sampling_params, use_tqdm=False)


def print_outputs(outputs):
    print("\nGenerated Outputs:\n" + "-" * 80)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\n")
        print(f"Generated text: {generated_text!r}")
        print("-" * 80)


print_outputs(outputs)
