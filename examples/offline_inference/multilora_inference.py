# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use the multi-LoRA functionality
for offline inference.

Requires HuggingFace credentials for access to Llama2.
"""

from typing import Optional

from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


def create_test_prompts(
    lora_path: str,
) -> list[tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.

    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be run after all requests with the
    first adapter have finished.
    """
    return [
        (
            "A robot may not injure a human being",
            SamplingParams(
                temperature=0.0, logprobs=1, prompt_logprobs=1, max_tokens=128
            ),
            None,
        ),
        (
            "To be or not to be,",
            SamplingParams(
                temperature=0.8, top_k=5, presence_penalty=0.2, max_tokens=128
            ),
            None,
        ),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
            SamplingParams(
                temperature=0.0,
                logprobs=1,
                prompt_logprobs=1,
                max_tokens=128,
                stop_token_ids=[32003],
            ),
            LoRARequest("sql-lora", 1, lora_path),
        ),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
            SamplingParams(
                temperature=0.0,
                logprobs=1,
                prompt_logprobs=1,
                max_tokens=128,
                stop_token_ids=[32003],
            ),
            LoRARequest("sql-lora2", 2, lora_path),
        ),
    ]


def process_requests(
    engine: LLMEngine,
    test_prompts: list[tuple[str, SamplingParams, Optional[LoRARequest]]],
):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    print("-" * 50)
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(
                str(request_id), prompt, sampling_params, lora_request=lora_request
            )
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)
                print("-" * 50)


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(
        model="meta-llama/Llama-2-7b-hf",
        enable_lora=True,
        max_loras=1,
        max_lora_rank=8,
        max_cpu_loras=2,
        max_num_seqs=256,
    )
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine()
    lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
    test_prompts = create_test_prompts(lora_path)
    process_requests(engine, test_prompts)


if __name__ == "__main__":
    main()
