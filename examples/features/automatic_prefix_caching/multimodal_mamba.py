# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare incremental-image cache reuse with a cold run on hybrid Mamba models.

Use --strict to reproduce fixed-configuration numerical comparisons. Exact
equality is not a general batch-invariance guarantee. No latency is measured.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

import vllm
from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform


def make_prompt(tokenizer, image, count, repeats, tail):
    items = [{"type": "text", "text": "Task description.\n\nHistory:\n"}]
    for index in range(count):
        items.extend(
            [
                {"type": "text", "text": f"\n--- Turn {index + 1} ---\n"},
                {"type": "image", "image": image},
            ]
        )
    items.append(
        {
            "type": "text",
            "text": "\n"
            + "Follow these instructions carefully. " * tail
            + "\nWhat next?",
        }
    )
    prompt = TextPrompt(
        prompt=tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. " * repeats,
                },
                {"role": "user", "content": items},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
    )
    if count:
        prompt["multi_modal_data"] = {"image": [image] * count}
    return prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--legacy-hashes", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("short", "padded", "long_tail", "text"),
        default=("short", "padded", "long_tail", "text"),
    )
    args = parser.parse_args()
    if args.legacy_hashes and not args.mtp:
        parser.error("--legacy-hashes requires --mtp")
    if args.max_tokens < 1 or args.image_size < 1:
        parser.error("--max-tokens and --image-size must be positive")
    llm = LLM(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        kv_cache_dtype="auto",
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=4,
        kv_cache_memory_bytes=2 * 1024**3,
        gpu_memory_utilization=0.26,
        limit_mm_per_prompt={"image": 5, "video": 0},
        mm_processor_kwargs={"max_pixels": args.image_size**2},
        speculative_config=(
            {
                "method": "mtp",
                "num_speculative_tokens": 3,
                "disable_lookahead_block_hashes": args.legacy_hashes,
            }
            if args.mtp
            else None
        ),
        seed=0,
    )
    image = Image.new("RGB", (args.image_size, args.image_size), (128, 128, 128))
    tokenizer = llm.get_tokenizer()
    generations = []
    comparisons = []

    def generate(name, prompt, skip=False):
        output = llm.generate(
            [prompt],
            SamplingParams(
                temperature=0,
                max_tokens=args.max_tokens,
                ignore_eos=True,
                logprobs=10,
                skip_reading_prefix_cache=skip,
            ),
            use_tqdm=False,
        )[0]
        row = {
            "name": name,
            "cached_tokens": output.num_cached_tokens,
            "tokens": list(output.outputs[0].token_ids),
            "logprobs": [
                {token: value.logprob for token, value in step.items()}
                for step in output.outputs[0].logprobs
            ],
        }
        generations.append(row)
        assert len(row["tokens"]) == args.max_tokens
        assert not skip or row["cached_tokens"] == 0
        return row

    for case, repeats, tail in (
        ("short", 30, 0),
        ("padded", 300, 0),
        ("long_tail", 300, 500),
        ("text", 300, 500),
    ):
        if case not in args.cases:
            continue
        assert llm.reset_prefix_cache()
        producer_count = 0 if case == "text" else 4
        sibling_count = 0 if case == "text" else 5
        producer = make_prompt(tokenizer, image, producer_count, repeats, tail)
        sibling = make_prompt(tokenizer, image, sibling_count, repeats, tail)
        generate(case + "_prime", producer)
        replay = generate(case + "_replay", producer)
        hit = generate(case + "_sibling", sibling)
        cold = generate(case + "_cold", sibling, skip=True)
        if case != "short":
            assert replay["cached_tokens"] > 0
            assert hit["cached_tokens"] > 0
        comparisons.append(
            {
                "case": case,
                "cached_tokens": hit["cached_tokens"],
                "token_ids_equal": hit["tokens"] == cold["tokens"],
                "top10_logprobs_equal": hit["logprobs"] == cold["logprobs"],
            }
        )
    config_path = Path(args.model) / "config.json"
    config = llm.llm_engine.vllm_config
    result = {
        "arguments": {**vars(args), "output": str(args.output)},
        "runtime": {
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": [
                current_platform.get_device_name(i)
                for i in range(torch.accelerator.device_count())
            ],
            "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
            "block_size": config.cache_config.block_size,
            "model_config_sha256": (
                hashlib.sha256(config_path.read_bytes()).hexdigest()
                if config_path.is_file()
                else None
            ),
        },
        "comparisons": comparisons,
        "generations": generations,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))
    if args.strict and any(
        not row["token_ids_equal"] or not row["top10_logprobs_equal"]
        for row in comparisons
    ):
        raise SystemExit("Strict cached-versus-cold comparison failed")


if __name__ == "__main__":
    main()
