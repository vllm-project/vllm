"""
This example shows how to use Ray Core for running offline batch inference
distributively on a multi-nodes cluster. In contrast to the
`offline_inference_distributed.py` example, you don't need to manually
set the batch size.

Learn more about Ray Core in https://docs.ray.io/en/latest/ray-core/walkthrough.html
"""

from typing import List

import ray

from vllm import LLM, RequestOutput, SamplingParams

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Set tensor parallelism per instance.
tensor_parallel_size = 1

# Set number of instances. Each instance will use tensor_parallel_size GPUs.
num_workers = 2

# Sample prompts.
prompts = [
    "Hello, I'm running on the first GPU",
    "Hello, I'm running on the second GPU",
]


def split_dataset(dataset: List[str], num_splits: int) -> List[List[str]]:
    """
    Splits the dataset into the specified number of parts.
    """
    avg_size = len(dataset) // num_splits
    splits = [
        dataset[i * avg_size:(i + 1) * avg_size] for i in range(num_splits)
    ]

    # Allocate remaining items if not evenly divisible
    remainder = len(dataset) % num_splits
    if remainder:
        splits[-1].extend(dataset[-remainder:])
    return splits


# Split the dataset into parts for each worker
data_splits = split_dataset(prompts, num_workers)


def inference_task(data_split: List[str]) -> List[RequestOutput]:
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tensor_parallel_size=tensor_parallel_size,
    )
    results = llm.generate(data_split, sampling_params)
    return results


# Define and submit the remote Ray tasks
inference_task_remote = ray.remote(
    num_gpus=tensor_parallel_size)(inference_task)
futures = [inference_task_remote.remote(split) for split in data_splits]

# Gather the results
results = ray.get(futures)

# Flatten the list of results
flattened_results = []
for split_result in results:
    flattened_results.extend(split_result)

# Print results.
for output in flattened_results:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
