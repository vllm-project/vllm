# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shape configurations for IR op benchmarks.
"""

import torch

NUM_TOKENS = [1, 2, 4, 16, 64, 256, 1024, 4096, 16384]
COMMON_HIDDEN_SIZES = [
    2048,  # Llama 3.2 1B, Qwen 3 MoE 30B-A3B, Gemma 3n
    3072,  # Gemma 7B/9B
    4096,  # Llama 3 8B, Qwen 3 8B, Mistral 7B
    5120,  # Llama 4 Scout 17B-16E
    7168,  # DeepSeek V3
    8192,  # Llama 3 70B
    16384,  # Llama 3 405B
]

# Each entry maps an op name to a list of kwarg dicts that will be passed
# to that op's registered input generator via op.generate_inputs(**kwargs).
SHAPE_CONFIGS: dict[str, list[dict]] = {
    "rms_norm": [
        {"num_tokens": n, "hidden_size": d, "dtype": dtype}
        for dtype in [torch.float16, torch.bfloat16, torch.float32]
        for d in COMMON_HIDDEN_SIZES
        for n in NUM_TOKENS
    ],
    "dynamic_group_quant_fp8": [
        {
            "num_tokens": n,
            "hidden_size": h,
            "dtype": dtype,
            "group_size": g,
            "column_major_scales": cm,
        }
        for dtype in [torch.bfloat16]
        for n in [1, 64, 1024]
        for h in [2048, 8192]
        for g in [128]
        for cm in [False, True]
    ],
}
