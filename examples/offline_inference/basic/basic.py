# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

# Sample prompts.
prompts = [
    "hello, can you tell me the answer of 1 + 1?",
    
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=50)


prompt_token_ids = [
    TokensPrompt(
        prompt_token_ids=[0, 128803, 33310, 14, 588, 440, 4575, 678, 270, 3287, 294, 223, 19, 940, 223, 19, 33, 128804, 128799],
    ), # hello, can you tell me the answer of 1 + 1?
    TokensPrompt(
        prompt_token_ids=[0, 128803, 33310, 14, 1205, 344, 223, 20, 940, 223, 20, 33, 128804, 128799],
    ), # hello, what is 2 + 2?
    TokensPrompt(
        prompt_token_ids=[0, 128803, 9602, 344, 223, 21, 940, 223, 21, 33, 8033, 1801, 678, 16, 128804, 128799],
    ), # what is 3 + 3? please show me. 
]

"""
Prompt: hello, can you tell me the answer of 1 + 1?
Output: Hello! The answer to 1 + 1 is **2**. \n\nIf you have any more questions, feel free to ask! 😊
"""

"""
Prompt: hello, what is 2 + 2?
Output: Hello! 2 + 2 equals 4. 😊
"""

"""
Prompt: what is 3 + 3? please show me.
Output: Let's add 3 and 3 together:\n\n3 + 3 = 6\n\nSo, 3 plus 3 equals 6."
"""
def main():
    # Create an LLM.
    llm = LLM(model="/home/vllm-dsv32/DeepSeek-V3.2-Preview-Fix", tensor_parallel_size=8,  enforce_eager=True)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompt_token_ids, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
