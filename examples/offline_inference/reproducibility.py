# SPDX-License-Identifier: Apache-2.0
import os

from vllm import LLM, SamplingParams

# vLLM does not guarantee the reproducibility of the results by default,
# for the sake of performance. You need to do the following to achieve
# reproducible results:
# 1. Turn off multiprocessing to make the scheduling deterministic.
#    NOTE(woosuk): This is not needed and will be ignored for V0.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# 2. Fix the global seed for reproducibility. The default seed is None, which is
# not reproducible.
SEED = 42

# NOTE(woosuk): Even with the above two settings, vLLM only provides
# reproducibility when it runs on the same hardware and the same vLLM version.
# Also, the online serving API (`vllm serve`) does not support reproducibility
# because it is almost impossible to make the scheduling deterministic in the
# online serving setting.

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    llm = LLM(model="facebook/opt-125m", seed=SEED)
    outputs = llm.generate(prompts, sampling_params)
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


if __name__ == "__main__":
    main()
