# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parse an ordered set of document page images with Unlimited-OCR-2."""

import argparse

from PIL import Image

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--attention-backend",
        choices=["FLASH_ATTN", "FLASHINFER"],
        default="FLASH_ATTN",
    )
    args = parser.parse_args()
    images = []
    for path in args.images:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    llm = LLM(
        model="baidu/Unlimited-OCR-2",
        attention_config={"backend": args.attention_backend},
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        limit_mm_per_prompt={"image": len(images)},
        mm_encoder_attn_backend="TORCH_SDPA",
    )
    output = llm.generate(
        {
            "prompt": "<PAGE><image>" * len(images) + "Multi page parsing.",
            "multi_modal_data": {"image": images},
        },
        SamplingParams(
            temperature=0, max_tokens=args.max_tokens, skip_special_tokens=False
        ),
    )
    print(output[0].outputs[0].text)


if __name__ == "__main__":
    main()
