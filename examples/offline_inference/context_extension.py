# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script demonstrates how to extend the context length
of a Qwen model using the YARN method (rope_parameters)
and run a simple chat example.

Usage:
    python examples/offline_inference/context_extension.py
"""

from vllm import LLM, SamplingParams


def create_llm():
    rope_theta = 1000000
    original_max_position_embeddings = 32768
    factor = 4.0

    # Use yarn to extend context
    hf_overrides = {
        "rope_parameters": {
            "rope_theta": rope_theta,
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": original_max_position_embeddings,
        },
        "max_model_len": int(original_max_position_embeddings * factor),
    }

    llm = LLM(model="Qwen/Qwen3-0.6B", hf_overrides=hf_overrides)
    return llm


def run_llm_chat(llm):
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
    return outputs


def print_outputs(outputs):
    print("\nGenerated Outputs:\n" + "-" * 80)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\n")
        print(f"Generated text: {generated_text!r}")
        print("-" * 80)


def main():
    llm = create_llm()
    outputs = run_llm_chat(llm)
    print_outputs(outputs)


if __name__ == "__main__":
    main()
