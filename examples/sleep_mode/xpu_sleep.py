#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test script to actually instantiate XPUWorker and test sleep/wakeup functionality.
This test creates a real XPUWorker instance and calls the methods.
"""

import torch

from vllm import LLM, SamplingParams
from vllm.utils import GiB_bytes


def run_xpu_sleep():
    model = "Qwen/Qwen3-0.6B"
    free, total = torch.xpu.mem_get_info()
    used_bytes_baseline = total - free  # in case other process is running
    llm = LLM(model, enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    # Put the engine to deep sleep
    llm.sleep(level=2)

    free_gpu_bytes_after_sleep, total = torch.xpu.mem_get_info()
    used_bytes = total - free_gpu_bytes_after_sleep - used_bytes_baseline
    assert used_bytes < 3 * GiB_bytes

    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")
    free_gpu_bytes_wake_up_w, total = torch.xpu.mem_get_info()
    used_bytes = total - free_gpu_bytes_wake_up_w - used_bytes_baseline
    assert used_bytes < 4 * GiB_bytes

    # now allocate kv cache
    llm.wake_up(tags=["kv_cache"])
    output2 = llm.generate(prompt, sampling_params)
    # cmp output
    assert output[0].outputs[0].text == output2[0].outputs[0].text


if __name__ == "__main__":
    run_xpu_sleep()
