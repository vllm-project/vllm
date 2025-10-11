# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates RLHF with GPU buffer offloading during sleep_mode.
Run: VLLM_HOST_IP=127.0.0.1 python cuda_graph_sleep_mode.py
"""

import logging
import os
import time

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FooLLM(LLM):
    """Configure vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)


# Load training model on GPU 0
train_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to("cuda:0")

# Setup Ray with vLLM on GPUs 1-2
os.environ.update(
    {
        "CUDA_VISIBLE_DEVICES": "1,2",
        "VLLM_HOST_IP": "127.0.0.1",
        "CUDA_GRAPH_MEMORY_POOL_SLEEP_MODE": "1",
        "VLLM_USE_V1": "1",
    }
)

ray.init()
pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
scheduling = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

# Launch vLLM with sleep mode enabled
llm = ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=scheduling)(FooLLM).remote(
    model="facebook/opt-125m",
    enforce_eager=False,
    enable_sleep_mode=True,
    worker_extension_cls="rlhf_utils.WorkerExtension",
    tensor_parallel_size=1,
    distributed_executor_backend="ray",
)

# Initial generation test
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0)
outputs = ray.get(llm.generate.remote(prompts, sampling_params))

logger.info("-" * 50)
for output in outputs:
    logger.info("Prompt: %r | Generated: %r", output.prompt, output.outputs[0].text)
logger.info("-" * 50)


# Training loop with sleep/wake cycles
for step in range(50):
    logger.info("Training Step %d/50", step + 1)

    # Determine sleep level: level 1 every 10 steps
    sleep_level = 1 if (step + 1) % 10 == 0 else 0

    if sleep_level:
        logger.info("Sleep level %d -> Wake up", sleep_level)
        ray.get(llm.sleep.remote(level=sleep_level))
        ray.get(llm.wake_up.remote())
        # Verify generation after wake
        test_out = ray.get(
            llm.generate.remote(
                ["Quick test: Who is the father of Harry Porter?"], sampling_params
            )
        )
        logger.info("Test output: %s...", test_out[0].outputs[0].text[:50])

    time.sleep(1)  # Simulate training work

# Final generation test after training loop
logger.info("-" * 50)
logger.info("Post-training generation test")
outputs_final = ray.get(llm.generate.remote(prompts, sampling_params))
for output in outputs_final:
    logger.info("Prompt: %r | Generated: %r", output.prompt, output.outputs[0].text)
logger.info("-" * 50)
