# SPDX-License-Identifier: Apache-2.0
import os
from urllib.request import urlopen

from vllm import LLM, SamplingParams

os.environ["VLLM_ATTENTION_BACKEND"] = "DUAL_CHUNK_FLASH_ATTN"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


def load_prompt() -> str:
    # Test cases with various lengths can be found at:
    #
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/64k.txt
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/200k.txt
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/600k.txt
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/1m.txt

    with urlopen(
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/600k.txt",
        timeout=5,
    ) as response:
        prompt = response.read().decode("utf-8")
    return prompt


# Processing the prompt.
def process_requests(llm: LLM, prompts: list[str]) -> None:
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        detokenize=True,
        max_tokens=256,
    )
    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt_token_ids = output.prompt_token_ids
        generated_text = output.outputs[0].text
        print(
            f"Prompt length: {len(prompt_token_ids)}, "
            f"Generated text: {generated_text!r}"
        )


# Create an LLM.
def initialize_engine() -> LLM:
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-1M",
        max_model_len=1048576,
        tensor_parallel_size=4,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=131072,
    )
    return llm


def main():
    llm = initialize_engine()
    prompt = load_prompt()
    process_requests(llm, [prompt])


if __name__ == "__main__":
    main()
