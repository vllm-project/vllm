# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

from PIL import Image

from vllm import LLM, SamplingParams

# Usage:
# python3 examples/offline_inference/florence2_v1_smoke.py \
#   --model /path/to/florence2
#
# This smoke script intentionally keeps a minimal single-image flow.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Florence2 v1 smoke test")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="<CAPTION>")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )

    image = Image.open(args.image).convert("RGB")
    outputs = llm.generate(
        {
            "encoder_prompt": {
                "prompt": args.prompt,
                "multi_modal_data": {"image": image},
            },
            "decoder_prompt": "",
        },
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
        ),
    )

    print(outputs[0].outputs[0].text)
