"""
This example shows how to use the multi-LoRA functionality
for offline inference.

Requires HuggingFace credentials for access to Llama2.
"""

from typing import List, Optional, Tuple

from huggingface_hub import list_models, snapshot_download

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

NUM_LORAS = 20

def fetch_lora_metadata(keyword: str = "lora") -> List[dict]:
    """Fetch available LoRA adapters from Hugging Face Hub."""
    models = list_models(filter=f"{keyword}", cardData=True)
    lora_metadata = []
    for model in models:
        if "lora" in model.tags:  # Ensure the model is tagged as LoRA
            if model.cardData and "example_prompt" in model.cardData:
                lora_metadata.append({
                    "name": model.modelId,
                    "description": model.cardData.get("description"),
                    "example_prompt": model.cardData["example_prompt"],
                    "repo_id": model.modelId
                })
        if len(lora_metadata) == NUM_LORAS:
            break
    return lora_metadata


def download_lora_weights(metadata: List[dict]) -> List[str]:
    """Download LoRA weights using Hugging Face snapshot_download."""
    paths = []
    for lora in metadata:
        path = snapshot_download(repo_id=lora["repo_id"])
        paths.append(path)
    return paths


def create_test_prompts(
        metadata: str
) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.
    """
    prompts = []
    for idx, lora in enumerate(metadata, start=1):
        example_prompt = lora.get("example_prompt", "Provide a relevant response.")
        prompts.append(
            (
                example_prompt,
                SamplingParams(temperature=0.7, max_tokens=128),
                LoRARequest(f"lora-adapter-{idx}", idx, lora["repo_id"]),
            )
        )
    return prompts


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams,
                                              Optional[LoRARequest]]]):
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

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model="meta-llama/Llama-2-7b-hf",
                             enable_lora=True,
                             max_loras=3,  # TODO: set this depending on adapter size
                             max_lora_rank=8,
                             max_cpu_loras=5,
                             max_num_seqs=256)
    return LLMEngine.from_engine_args(engine_args)

def main():
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine()
    lora_metadata = fetch_lora_metadata()
    download_lora_weights(lora_metadata)
    test_prompts = create_test_prompts(lora_metadata)
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    main()
