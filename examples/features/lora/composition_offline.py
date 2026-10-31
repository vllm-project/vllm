# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compose two compatible, linear-target LoRA adapters for offline generation.

From the repository root, run:
    .venv/bin/python examples/features/lora/composition_offline.py \
        --model BASE_MODEL --adapters FIRST_ADAPTER SECOND_ADAPTER \
        --weights 0.7 0.3 --max-lora-rank 64

Both adapters must target the selected base model. Eager execution and the
two-adapter per-request limit are enabled by this example. On CPU hosts, set
VLLM_CPU_KVCACHE_SPACE to a suitable cache size in GiB for the available memory.
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
    args = parser.parse_args()
    llm = LLM(
        model=args.model,
        enable_lora=True,
        enforce_eager=True,
        max_loras=2,
        max_loras_per_request=2,
        max_lora_rank=args.max_lora_rank,
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
