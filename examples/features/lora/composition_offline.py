# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compose two compatible, linear-target LoRA adapters for offline generation.

From the repository root, run:
    .venv/bin/python examples/features/lora/composition_offline.py \
        --model BASE_MODEL --adapters FIRST_ADAPTER SECOND_ADAPTER \
        --weights 0.7 0.3 --max-lora-rank 64 --target-modules q_proj v_proj

Both adapters must target the selected base model. Supported NVIDIA configurations
use CUDA graphs by default. For CPU and other backends, add --enforce-eager.
On CPU hosts, set VLLM_CPU_KVCACHE_SPACE to a suitable cache size in GiB for the
available memory.
"""

import argparse

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest, LoRARoutingRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapters", nargs=2, required=True)
    parser.add_argument("--weights", nargs=2, type=float, default=[0.7, 0.3])
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--target-modules", nargs="+", required=True)
    args = parser.parse_args()
    llm = LLM(
        model=args.model,
        enable_lora=True,
        enforce_eager=args.enforce_eager,
        max_loras=2,
        max_loras_per_request=2,
        max_lora_rank=args.max_lora_rank,
        lora_target_modules=args.target_modules,
    )
    mixture = LoRARoutingRequest(
        routing_name="composition",
        routing_int_id=100,
        lora_requests=tuple(
            LoRARequest(f"adapter-{index}", index, path)
            for index, path in enumerate(args.adapters, start=1)
        ),
        lora_weights=tuple(args.weights),
    )
    for output in llm.generate(
        ["Explain low-rank adaptation briefly."],
        SamplingParams(temperature=0, max_tokens=64),
        lora_request=mixture,
    ):
        print(output.outputs[0].text)


if __name__ == "__main__":
    main()
