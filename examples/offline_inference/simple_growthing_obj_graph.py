# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

from vllm import LLM, SamplingParams

# Enable object graph analysis by setting environment variable
os.environ["VLLM_OBJ_GRAPH_DIR"] = "./vllm_obj_graph"

# Sample prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of artificial intelligence is",
]
# Create sampling parameters object
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create LLM instance
    llm = LLM(model="facebook/opt-125m", tensor_parallel_size=1)

    # Start object graph analysis
    llm.start_object_graph()

    # Generate text from prompts. The output is a list of RequestOutput objects
    # containing the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Stop object graph analysis
    llm.stop_object_graph()

    # Print output results
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Add buffer time to wait for processes (if multiprocessing is enabled)
    # to complete writing object graph analysis output.
    time.sleep(10)
    print(f"Completed! Results saved to: {os.environ['VLLM_OBJ_GRAPH_DIR']}")
    print("You can check the generated files to analyze memory growth")


if __name__ == "__main__":
    main()
