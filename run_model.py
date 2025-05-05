# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

prompts = [
    "The Swiss Alps are", "The president of the USA is",
    "The Boston Bruins are"
]

sampling_params = SamplingParams(temperature=0.80,
                                 top_p=0.95,
                                 max_tokens=40,
                                 min_tokens=10)
llm = LLM('nvidia/Llama-3.3-70B-Instruct-FP4',
          quantization='nvfp4',
          max_model_len=2048,
          enforce_eager=True)

# Print the outputs.
output = llm.generate(prompts, sampling_params)
for o in output:
    print(o.outputs[0].text)
    print("\n")
