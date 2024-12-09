"""
This example shows how to use the multi-LoRA functionality
for offline inference.

Requires HuggingFace credentials for access to Llama2.
"""

from typing import List, Optional, Tuple
import numpy as np
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams, LoraPolicy
from vllm.lora.request import LoRARequest
from faker import Faker
import pandas as pd
import torch

OUT_DIR = "out"
TOTAL_LORAS = 8
NUM_REQUESTS = 96

def create_test_prompts(
    distribution: str
) -> List[Tuple[str, int]]:
    """Create a list of test prompts with their sampling parameters.

    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """
    fake = Faker()
    sentence = f"lora:"

    # TODO Atindra: Instead of hardcoding i, make i follow a distribution and generate 100 or so requests
    # NOTE: LoRA 0 means no LoRA in VLLM
    '''
    prompts = []
    for _ in range(10):
        for i in range(TOTAL_LORAS):
            prompts.append((
                sentence,
                SamplingParams(temperature=0.0,
                            logprobs=1,
                            prompt_logprobs=1,
                            max_tokens=64,
                            stop_token_ids=[128001]),
                LoRARequest(f"lora{i}", i, f"{base_path}/lora{i}")
            ))

    return prompts
    '''
    num_requests=NUM_REQUESTS
    if distribution == "uniform":
        # lora_ids_list = np.random.randint(0, TOTAL_LORAS, size=num_requests)
        lora_ids_list = np.array(list(range(TOTAL_LORAS)) * (num_requests // TOTAL_LORAS))
    elif distribution == "normal":
        # Center the normal distribution around the middle of the LoRA range
        mean = (TOTAL_LORAS - 1)/2
        std_dev = TOTAL_LORAS/6 # This ensures ~99.7% of values fall within range ("68-95-99.7 rule")
        lora_ids_list = np.random.normal(mean, std_dev, size=num_requests)

        # Clip values to ensure they're within valid range and convert to integers
        lora_ids_list = np.clip(lora_ids_list, 0, TOTAL_LORAS-1)
        lora_ids_list = np.round(lora_ids_list).astype(int)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    prompts = []
    for lora_id in lora_ids_list:
        prompts.append((
            sentence,
            lora_id
        ))
        
    return prompts

def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams,
                                              Optional[LoRARequest]]],
                     distribution: str) -> None:
    """Continuously process a list of prompts and handle the outputs."""
    metrics_list = []

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
                metrics = request_output.metrics
                output = request_output.outputs[0]
                metrics_list.append({
                    "request_id": request_output.request_id,
                    "lora_policy": engine.lora_config.lora_policy,
                    "num_iters_before_lora_reschedule": engine.lora_config.num_iters_before_reschedule,
                    "distribution": distribution,
                    "max_loras": engine.lora_config.max_loras,
                    "lora_name": request_output.lora_request.lora_name,
                    "lora_id": request_output.lora_request.lora_int_id,
                    "arrival_time": metrics.arrival_time,
                    "last_token_time": metrics.last_token_time,
                    "first_scheduled_time": metrics.first_scheduled_time,
                    "time_in_queue": metrics.time_in_queue,
                    "finished_time": metrics.finished_time,
                    "scheduler_time": metrics.scheduler_time,
                    "model_forward_time": metrics.model_forward_time,
                    "model_execute_time": metrics.model_execute_time,
                    "prompt": request_output.prompt,
                    "prompt_num_tokens": len(request_output.prompt_token_ids),
                    "output": output.text,
                    "output_num_tokens": len(output.token_ids),
                })

    return metrics_list


def initialize_engine(max_loras: int, lora_policy: LoraPolicy, num_iters_before_lora_reschedule: int) -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model="meta-llama/Llama-3.2-1B",
                             enable_lora=True,
                             max_loras=max_loras,
                             max_lora_rank=8,
                             max_cpu_loras=TOTAL_LORAS,
                             max_num_seqs=256,
                             enforce_eager=True,
                             disable_async_output_proc=True,
                             lora_policy=lora_policy,
                             num_iters_before_lora_reschedule=num_iters_before_lora_reschedule,
                             max_model_len=256,
                             )
    return LLMEngine.from_engine_args(engine_args)

import time

def main():
    """Main function that sets up and runs the prompt processing."""
    metrics_list = []
    for distribution in ["uniform"]: # , "normal"]:
        test_prompts = create_test_prompts(distribution)
        for lora_policy in [LoraPolicy.NAIVE]:
            for num_iters_before_lora_reschedule in [1, 2, 4, 8, 16, 32, 64]:
                for max_loras in [1, 2, 4, 8]:
                    try:
                        engine = initialize_engine(max_loras, lora_policy, num_iters_before_lora_reschedule)
                        prompts = [(
                            sentence,
                            SamplingParams(temperature=0.0,
                                logprobs=1,
                                prompt_logprobs=1,
                                max_tokens=16,
                                stop_token_ids=[128001]),
                            LoRARequest(f"lora{lora_id}", lora_id + 1, f"{OUT_DIR}/lora{lora_id}")
                        ) for sentence, lora_id in test_prompts]
                        metrics_list.extend(process_requests(engine, prompts, distribution))
                        del engine
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Error: {e}")
                        time.sleep(10)
                        continue
                    metrics_df = pd.DataFrame(metrics_list)
                    metrics_df.to_csv(f"{OUT_DIR}/metrics_tmp.csv", index=False)
                    with open(f"{OUT_DIR}/status.csv", "a") as f:
                        f.write(f"{distribution},{lora_policy},{num_iters_before_lora_reschedule},{max_loras}\n")


if __name__ == '__main__':
    main()
