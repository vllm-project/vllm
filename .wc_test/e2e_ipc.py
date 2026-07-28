# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E check: load Qwen3-0.6B from the weight cache daemon via CUDA IPC."""

import time

from vllm import LLM, SamplingParams

start = time.perf_counter()
llm = LLM(
    model="/disk3/models/Qwen3-0.6B/",
    load_format="ipc_cache",
    model_loader_extra_config={
        "socket_dir": "/disk3/lsy/vllm/.wc_test",
        "fallback": False,
    },
    gpu_memory_utilization=0.4,
    enforce_eager=True,
)
print(f"LLM init took {time.perf_counter() - start:.2f}s")

prompts = [
    "The capital of France is",
    "1+1=",
]
outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=24))
for out in outputs:
    print(repr(out.prompt), "->", repr(out.outputs[0].text))
