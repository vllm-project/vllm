"""
This example shows how to use the multi-LoRA functionality for offline inference.

Requires HuggingFace credentials for access to Llama2.
"""
import os

from typing import Optional, List, Tuple

# from huggingface_hub import snapshot_download
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest

PROMPT_TEMPLATE = """I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.\n"\n##Instruction:\nconcert_singer contains tables such as stadium, singer, concert, singer_in_concert. Table stadium has columns such as Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average. Stadium_ID is the primary key.\nTable singer has columns such as Singer_ID, Name, Country, Song_Name, Song_release_year, Age, Is_male. Singer_ID is the primary key.\nTable concert has columns such as concert_ID, concert_Name, Theme, Stadium_ID, Year. concert_ID is the primary key.\nTable singer_in_concert has columns such as concert_ID, Singer_ID. concert_ID is the primary key.\nThe Stadium_ID of concert is the foreign key of Stadium_ID of stadium.\nThe Singer_ID of singer_in_concert is the foreign key of Singer_ID of singer.\nThe concert_ID of singer_in_concert is the foreign key of concert_ID of concert.\n\n###Input:\n{query}\n\n###Response:"""  # noqa: E501

inputs = [
    PROMPT_TEMPLATE.format(query="How many singers do we have?"),
    PROMPT_TEMPLATE.format(
        query=
        "What is the average, minimum, and maximum age of all singers from France?"  # noqa: E501
    ),
    PROMPT_TEMPLATE.format(
        query=
        "Show name, country, age for all singers ordered by age from the oldest to the youngest."  # noqa: E501
    ),
]


def create_test_prompts(
        lora_path: str
) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.

    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """

    return [
        (
            "who are you",
            SamplingParams(temperature=0.3,
                           top_p=0.75,
                           top_k=40,
                           repetition_penalty=1.3),
            None,
        ),
        # (
        #     "To be or not to be,",
        #     SamplingParams(
        #         temperature=0.8, top_k=5, presence_penalty=0.2, max_tokens=128
        #     ),
        #     None,
        # ),
        (
            inputs[0],
            SamplingParams(
                temperature=0.0,
                logprobs=1,
                prompt_logprobs=1,
                max_tokens=128,
            ),
            LoRARequest("sql-lora2222", 4, lora_path),
        ),
        (
            inputs[1],
            SamplingParams(
                temperature=0.0,
                logprobs=1,
                prompt_logprobs=1,
                max_tokens=128,
            ),
            LoRARequest("sql-lora2222", 3, lora_path),
        ),
        (
            inputs[2],
            SamplingParams(
                temperature=0.0,
                logprobs=1,
                prompt_logprobs=1,
                max_tokens=128,
            ),
            LoRARequest("sql-lora2222", 3, lora_path),
        ),
    ]


def process_requests(
    engine: LLMEngine,
    test_prompts: List[Tuple[str, SamplingParams, Optional[LoRARequest]]],
):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for output in request_outputs:
            if output.finished:
                prompt = output.prompt
                generated_text = output.outputs[0].text.strip()
                print(
                    f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


model_llm2 = "/home/sobey/SSD/Llama-2-7B-fp16-hf"
model_glm3 = "/home/sobey/Code/Code_leejee/chatglmv3.optimize/chatglm3_backup"
model_baichuan = "/home/sobey/SSD/baichuan-7B"


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
        model=model_baichuan,
        enable_lora=True,
        max_loras=8,
        max_lora_rank=64,
        max_cpu_loras=8,
        max_num_seqs=256,
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
    )
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine()
    # lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
    #lora_path = "/home/sobey/Code/Code_leejee/test_data/llama-2-7b-sql-lora-test"
    #lora_path = "/home/sobey/Code/DB-GPT-Hub/dbgpt_hub/output/adapter/20240319_chatglmv3_spider"
    lora_path="/home/sobey/Code/DB-GPT-Hub/dbgpt_hub/output/adapter/20240319_baichuan_spider"
    test_prompts = create_test_prompts(lora_path)
    process_requests(engine, test_prompts)


if __name__ == "__main__":
    main()
