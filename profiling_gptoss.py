#! python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

os.environ["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_MXFP4_USE_MARLIN"] = "1"
os.environ["VLLM_TUNED_CONFIG_FOLDER"] = "/home/ubuntu/qwen_32B_max_lora_8"
# os.environ['VLLM_DISABLE_SHARED_EXPERTS_STREAM'] = '1'


import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

prompts = [
    # "What are frogs?" * 1,
    "What are frogs?" * 400,
]

# jeeejeee/gpt-oss-20b-lora-adapter-text2sql
# ./gpt-oss-20b-coding-rank32-lr1e-5
# lora_path = "/home/ubuntu/gpt-oss-20b-coding-rank32-lr1e-5"
lora_path = "/home/ubuntu/tuned-qwen3-32b"
max_num_seqs = 16

# Set to false for baseline profiling w/out lora
use_lora = True
max_loras = 16

if use_lora:
    llm = LLM(
        model="Qwen/Qwen3-32B-FP8",
        trust_remote_code=True,
        tensor_parallel_size=4,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=0.95,
        enable_lora=True,
        max_loras=8,
        max_lora_rank=32,
        max_cpu_loras=max_loras,
        # enforce_eager=True
        speculative_config={
            "model": "/home/ubuntu/qwen-32B-eagle2",
            "method": "eagle",
            "num_speculative_tokens": 5,
        },
        # fully_sharded_loras=True
    )

    # Check model dimensions
    model_config = llm.llm_engine.model_config.hf_config
    """
    print(f"\n=== Model Configuration ===")
    print(f"Model hidden_size: {model_config.hidden_size}")
    print(f"Model intermediate_size: {model_config.intermediate_size}")
    print(f"Model num_experts: {getattr(model_config, 'num_local_experts', 'N/A')}")
    """

    for i in range(1, max_loras + 1):
        lora_req = LoRARequest(lora_name=f"lora{i}", lora_int_id=i, lora_path=lora_path)
        llm.llm_engine.add_lora(lora_req)

        lora_req = [
            LoRARequest(
                lora_name="lora1",
                lora_int_id=1,
                lora_path=lora_path,
            )
        ]
else:
    llm = LLM(
        model="deepseek-ai/DeepSeek-V2-Lite",
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_seqs=max_num_seqs,
        # speculative_config = {
        # "model": "/mnt/models/spec-decoding/oss_20b_eagle/09_01",
        # "method": "eagle",
        # "num_speculative_tokens": 3
        # },
        # gpu_memory_utilization=0.8,
        # enforce_eager=True
    )
    lora_req = None

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=5)

# warmup
if lora_req:
    llm.generate(prompts[:max_num_seqs], sampling_params, lora_request=lora_req)
else:
    llm.generate(prompts[:max_num_seqs], sampling_params)

llm.reset_prefix_cache()

lora_req = [LoRARequest(lora_name="lora2", lora_int_id=2, lora_path=lora_path)]

lora_req2 = [LoRARequest(lora_name="lora3", lora_int_id=3, lora_path=lora_path)]


torch.cuda.cudart().cudaProfilerStart()
if lora_req:
    llm.generate(prompts[:max_num_seqs], sampling_params, lora_request=lora_req)
    llm.generate(prompts[:max_num_seqs], sampling_params, lora_request=lora_req2)
else:
    llm.generate(prompts[:max_num_seqs], sampling_params)
torch.cuda.cudart().cudaProfilerStop()
