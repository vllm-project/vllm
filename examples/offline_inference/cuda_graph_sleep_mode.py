# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) with
GPU buffer offloading during sleep_mode.

Run using : VLLM_HOST_IP=127.0.0.1 python cuda_graph_sleep_mode.py
"""

import os

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.logger import init_logger

# Initialize logger
logger = init_logger(__name__)


class FooLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        # Remove the top-level CUDA_VISIBLE_DEVICES variable set by Ray
        # so that vLLM can manage its own device placement within the worker.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)


# Load the OPT-125M model onto GPU 0 for the training workload.
train_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
train_model.to("cuda:0")

# Initialize Ray and set the visible devices. The vLLM engine will
# be placed on GPUs 1 and 2.
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
ray.init()

# Create a placement group that reserves GPU 1–2 for the vLLM inference engine.
# Learn more about Ray placement groups:
# https://docs.ray.io/en/latest/placement-groups.html
pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

# Launch the vLLM inference engine with cudagraphs and sleep mode enabled.
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(FooLLM).remote(
    model="facebook/opt-125m",
    enforce_eager=False,
    enable_sleep_mode=True,
    worker_extension_cls="rlhf_utils.WorkerExtension",
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
)

# Generate text from the prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)

outputs = ray.get(llm.generate.remote(prompts, sampling_params))

logger.info("%s", "-" * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    logger.info("Prompt: %r\nGenerated text: %r", prompt, generated_text)
    logger.info("%s", "-" * 50)


# Simulate a training loop for 20 steps
for step in range(20):
    logger.info("--- Training Step %d/20", step + 1)

    # Test different sleep levels at different steps
    if (step + 1) % 5 == 0:
        # Determine which sleep level to test
        if (step + 1) <= 10:
            sleep_level = 1
            logger.info("Step %d: Testing level 1 (weights only)", step + 1)
        else:
            sleep_level = 2
            logger.info("Step %d: Testing level 2 (weights + buffers)", step + 1)

        # Sleep is called on a ray actor that internally calls sleep async on
        # each worker. ray.get() is a blocking call.
        logger.info("--- Entering sleep mode level %d ---", sleep_level)

        # Execute sleep (memory tracking is already done inside the worker's sleep method)
        ray.get(llm.sleep.remote(level=sleep_level))

        logger.info(
            "--- Step %d: Waking up from sleep level %d ---", step + 1, sleep_level
        )
        ray.get(llm.wake_up.remote())

        logger.info("--- Step %d: Wake up complete ---", step + 1)

        # Generate text after wake up to verify model still works
        if sleep_level == 2:
            logger.info("--- Testing generation ---")
            test_outputs = ray.get(
                llm.generate.remote(["Quick test: The answer is"], sampling_params)
            )
            for output in test_outputs:
                logger.info("Test output: %s...", output.outputs[0].text[:50])

    # Simulate some training work
    import time

    time.sleep(1)

# Generate text with the model to ensure it's still working.
outputs_after_loop = ray.get(llm.generate.remote(prompts, sampling_params))
logger.info("%s", "-" * 50)
logger.info("--- Generating text after training loop ---")
for output in outputs_after_loop:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    logger.info("Prompt: %r\nGenerated text: %r", prompt, generated_text)
    logger.info("%s", "-" * 50)
