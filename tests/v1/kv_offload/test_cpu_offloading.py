# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import pytest

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

CPU_BLOCK_SIZES = [16, 48]


@pytest.mark.parametrize("cpu_block_size", CPU_BLOCK_SIZES)
def test_cpu_offloading(cpu_block_size: int) -> None:
    """
    Tests OffloadingConnector with CPUOffloadingSpec.
    """

    # configure OffloadingConnector (spec_name=CPUOffloadingSpec by default)
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "num_cpu_blocks": 100,
            "block_size": cpu_block_size
        },
    )

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        gpu_memory_utilization=0.5,
        kv_transfer_config=kv_transfer_config,
    )

    prompts = ["Hi " * 100]
    sampling_params = SamplingParams(temperature=0, max_tokens=20)

    # run generation - this should trigger saving KV cache
    start_time = time.time()
    llm.generate(prompts, sampling_params, use_tqdm=False)
    cold_time = time.time() - start_time

    # run generation again - should hit the GPU prefix cache
    start_time = time.time()
    llm.generate(prompts, sampling_params, use_tqdm=False)
    gpu_hit_time = time.time() - start_time

    # reset prefix cache to avoid GPU hit.
    llm.reset_prefix_cache()

    # sleep for a sec to make sure CPU finished storing
    time.sleep(1)

    # run generation again - this should trigger loading from CPU
    start_time = time.time()
    llm.generate(prompts, sampling_params, use_tqdm=False)
    cpu_hit_time = time.time() - start_time

    print("Generation times:")
    print(f"    Cold: {cold_time * 1000:.2f}ms")
    print(f"    GPU hit: {gpu_hit_time * 1000:.2f}ms")
    print(f"    CPU hit: {cpu_hit_time * 1000:.2f}ms")
