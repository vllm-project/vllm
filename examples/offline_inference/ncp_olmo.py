# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run an NCP-OLMo Hugging Face checkpoint with vLLM."""

import argparse
import os


def build_attention_config(mode: str) -> dict[str, object] | None:
    if mode == "auto":
        return None
    if mode == "flash-attn":
        return {"backend": "FLASH_ATTN", "flash_attn_version": 3}
    if mode == "mixed-flashinfer":
        return {
            "backend": "FLASH_ATTN",
            "flash_attn_version": 3,
            "use_trtllm_attention": False,
            "backend_per_kind": {
                "full_attention": "FLASHINFER",
                "sliding_window": "FLASH_ATTN",
            },
        }
    raise ValueError(f"Unsupported attention mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path or Hub ID of a pure HF NCP-OLMo model")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attention-mode",
        choices=("auto", "flash-attn", "mixed-flashinfer"),
        default="auto",
        help=(
            "Attention backend selection. The mixed mode uses FlashInfer for "
            "full-attention layers and FlashAttention for sliding-window layers."
        ),
    )
    parser.add_argument(
        "--flashinfer-sampler",
        action="store_true",
        help="Use FlashInfer top-k/top-p sampling kernels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.flashinfer_sampler:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        attention_config=build_attention_config(args.attention_mode),
        seed=args.seed,
    )
    output = llm.generate(
        [args.prompt],
        SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
        use_tqdm=False,
    )[0]
    print(output.outputs[0].text)


if __name__ == "__main__":
    main()
