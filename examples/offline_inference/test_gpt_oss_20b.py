# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple offline inference test for openai/gpt-oss-20b."""

from vllm import LLM, SamplingParams

llm = LLM(model="openai/gpt-oss-20b", tensor_parallel_size=1)
prompts = ["What is the capital of France?"]
sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
