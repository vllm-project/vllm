# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark EAGLE3 tree-based speculative decoding with sampling args and output saving.

Features:
- Supports min_p, temperature, top_p, top_k sampling arguments
- Saves output text, tokens, and configuration to files
- Validates min_p works with speculative decoding
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch

from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig
from vllm.platforms import current_platform

DEFAULT_PROMPT = "Write a short paragraph about GPUs."


def generate_linear_tree(num_tokens: int) -> str:
    """Generate a linear tree with num_tokens draft tokens.

    Linear tree: [(0,), (0,0), (0,0,0), ...]
    Each token depends on the previous one in a single chain.
    """
    if num_tokens <= 0:
        return "[]"
    tree = []
    for i in range(num_tokens):
        # Path is all zeros of length i+1: (0,), (0,0), (0,0,0), ...
        tree.append(tuple([0] * (i + 1)))
    return str(tree)


def generate_branching_tree(max_depth: int, top_k: int = 3) -> str:
    """Generate a branching tree with multiple candidates at each level.

    This is based on TRT-LLM's eagle_choices format:
    - At level 1: top_k branches [0], [1], [2]
    - At level 2: each level-1 node expands to top_k children
    - Continues until max_depth

    Example for max_depth=2, top_k=3:
    [(0,), (1,), (2,), (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]

    For better performance with limited compute, we use a narrowing tree:
    - Level 1: top_k branches
    - Level 2+: top 2 children per parent (prevents exponential growth)
    """
    if max_depth <= 0:
        return "[]"

    tree = []
    # Level 1: generate top_k first-level nodes
    for i in range(top_k):
        tree.append((i,))

    if max_depth == 1:
        return str(tree)

    # Level 2+: expand each node with up to 2 children (to limit tree size)
    prev_level_nodes = tree[:]
    for depth in range(2, max_depth + 1):
        new_nodes = []
        children_per_parent = min(2, top_k)  # Limit branching to prevent explosion
        for parent in prev_level_nodes:
            for child_idx in range(children_per_parent):
                new_node = parent + (child_idx,)
                new_nodes.append(new_node)
        tree.extend(new_nodes)
        prev_level_nodes = new_nodes

    return str(tree)


def _build_llm(args: argparse.Namespace, speculative_config: dict | None) -> LLM:
    attention_config = None
    if args.attention_backend:
        attention_config = AttentionConfig(backend=args.attention_backend)
    return LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        attention_config=attention_config,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
    )


def _build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    """Build SamplingParams with all supported arguments."""
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k_sample,  # Use top_k_sample to avoid conflict with tree top_k
        min_p=args.min_p,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )


def _run_once(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    save_outputs: bool = False,
) -> tuple[int, list[dict]]:
    """Run generation and optionally collect outputs."""
    outputs = llm.generate(prompts, sampling_params)
    token_count = 0
    results = []

    for i, output in enumerate(outputs):
        token_ids = list(output.outputs[0].token_ids)
        text = output.outputs[0].text
        token_count += len(token_ids)

        if save_outputs:
            results.append(
                {
                    "prompt_idx": i,
                    "prompt": prompts[i],
                    "output_text": text,
                    "output_tokens": token_ids,
                    "num_tokens": len(token_ids),
                    "finish_reason": str(output.outputs[0].finish_reason),
                }
            )

    return token_count, results


def _save_results(
    args: argparse.Namespace,
    results: list[dict],
    sampling_params: SamplingParams,
    spec_config: dict | None,
    tree: str,
    tokens_per_s: float,
    total_tokens: int,
    is_baseline: bool = False,
):
    """Save results to output directory."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "baseline" if is_baseline else "eagle3"

    # Configuration file - make JSON-safe by converting complex objects to strings
    def make_json_safe(obj):
        """Convert non-serializable objects to strings."""
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_safe(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)  # Convert complex objects to string

    config = {
        "timestamp": timestamp,
        "model": args.model,
        "draft_model": args.draft_model if not is_baseline else None,
        "method": args.method if not is_baseline else None,
        "tree_type": args.tree_type,
        "tree": tree,
        "num_spec_tokens": args.num_spec_tokens,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "num_iters": args.num_iters,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k_sample": args.top_k_sample,
            "min_p": args.min_p,
            "seed": args.seed,
        },
        "speculative_config": make_json_safe(spec_config),
        "results": {
            "tokens_per_s": tokens_per_s,
            "total_tokens": total_tokens,
        },
    }

    config_path = output_dir / f"{prefix}_config_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to: {config_path}")

    # Results file (text and tokens)
    if results:
        results_path = output_dir / f"{prefix}_outputs_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved outputs to: {results_path}")

        # Plain text file for easy viewing
        text_path = output_dir / f"{prefix}_text_{timestamp}.txt"
        with open(text_path, "w") as f:
            for i, r in enumerate(results):
                f.write(f"=== Prompt {i} ===\n")
                f.write(f"Input: {r['prompt']}\n")
                f.write(f"Output ({r['num_tokens']} tokens):\n")
                f.write(f"{r['output_text']}\n")
                f.write(
                    f"Token IDs: {r['output_tokens'][:20]}...\n"
                    if len(r["output_tokens"]) > 20
                    else f"Token IDs: {r['output_tokens']}\n"
                )
                f.write("\n")
        print(f"Saved text to: {text_path}")


def _benchmark(
    args: argparse.Namespace,
    speculative_config: dict | None,
    tree: str,
    is_baseline: bool = False,
) -> tuple[float, int, list[dict]]:
    prompts = [args.prompt] * args.batch_size
    sampling_params = _build_sampling_params(args)
    llm = _build_llm(args, speculative_config)

    # Warmup
    for _ in range(args.warmup_iters):
        _run_once(llm, prompts, sampling_params, save_outputs=False)

    total_tokens = 0
    total_time_s = 0.0
    all_results = []

    for iter_idx in range(args.num_iters):
        start = time.perf_counter()
        # Save outputs on last iteration
        save_outputs = (iter_idx == args.num_iters - 1) and args.output_dir
        tokens, results = _run_once(
            llm, prompts, sampling_params, save_outputs=save_outputs
        )
        total_tokens += tokens
        total_time_s += time.perf_counter() - start
        if save_outputs:
            all_results = results

    tokens_per_s = total_tokens / total_time_s if total_time_s > 0 else 0.0

    # Save results if output directory specified
    if args.output_dir:
        _save_results(
            args,
            all_results,
            sampling_params,
            speculative_config,
            tree,
            tokens_per_s,
            total_tokens,
            is_baseline,
        )

    return tokens_per_s, total_tokens, all_results


def main() -> None:
    if not current_platform.is_cuda():
        raise SystemExit("CUDA is required for this benchmark.")

    parser = argparse.ArgumentParser(
        description="Benchmark EAGLE3 tree-based speculative decoding with sampling args."
    )
    # Model arguments
    parser.add_argument("--model", required=True, help="Target model path.")
    parser.add_argument("--draft-model", required=True, help="EAGLE3 draft model path.")
    parser.add_argument(
        "--method",
        default="eagle3",
        choices=("eagle", "eagle3"),
        help="Speculative method to use.",
    )

    # Tree configuration
    parser.add_argument(
        "--num-spec-tokens",
        type=int,
        default=2,
        help="Number of speculative tokens (tree depth for linear, max depth for branching).",
    )
    parser.add_argument(
        "--spec-token-tree",
        default=None,
        help="Speculative token tree (string literal). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--tree-type",
        default="branching",
        choices=["linear", "branching"],
        help="Type of tree: linear (single chain) or branching (multiple candidates per level).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top candidates at first level for branching tree.",
    )

    # Sampling arguments (min_p support!)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for greedy.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling.",
    )
    parser.add_argument(
        "--top-k-sample",
        type=int,
        default=-1,
        help="Top-k sampling. -1 means disabled.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Min-p sampling threshold. 0.0 means disabled. Values like 0.1 filter low-probability tokens.",
    )

    # Benchmark configuration
    parser.add_argument(
        "--attention-backend",
        default="TRITON_ATTN",
        help="Attention backend (e.g. TREE_ATTN or TRTLLM_ATTN).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run baseline (no spec decode) for comparison.",
    )

    # Output saving
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output text, tokens, and configuration. If not set, outputs are not saved.",
    )
    parser.add_argument(
        "--inspect-min-p",
        action="store_true",
        help="Print detailed token probability info to inspect min_p filtering.",
    )

    args = parser.parse_args()

    # Generate tree if not provided
    tree = args.spec_token_tree
    if tree is None:
        if args.tree_type == "branching":
            tree = generate_branching_tree(args.num_spec_tokens, args.top_k)
        else:
            tree = generate_linear_tree(args.num_spec_tokens)

    # Count total draft tokens in tree
    import ast

    tree_list = ast.literal_eval(tree)
    num_tree_tokens = len(tree_list)

    print(f"=== Configuration ===")
    print(f"Tree type: {args.tree_type}")
    print(f"Max depth: {args.num_spec_tokens}")
    print(f"Total draft tokens: {num_tree_tokens}")
    print(f"Tree: {tree[:100]}..." if len(tree) > 100 else f"Tree: {tree}")
    print(f"\n=== Sampling Parameters ===")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")
    print(f"top_k: {args.top_k_sample}")
    print(f"min_p: {args.min_p}")
    if args.min_p > 0:
        print(
            f"  ↳ min_p enabled! Tokens with p < {args.min_p} * max_p will be filtered."
        )
    if args.output_dir:
        print(f"\n=== Output ===")
        print(f"Results will be saved to: {args.output_dir}")

    spec_config = {
        "method": args.method,
        "model": args.draft_model,
        "num_speculative_tokens": num_tree_tokens,  # Use actual tree size
        "speculative_token_tree": tree,
        "max_model_len": args.max_model_len,
    }

    print(f"\n=== Running EAGLE3 tree decode ===")
    tokens_per_s, total_tokens, results = _benchmark(
        args, spec_config, tree, is_baseline=False
    )
    print(f"tokens/sec: {tokens_per_s:.2f}")
    print(f"total tokens: {total_tokens}")

    # Display sample output
    if results:
        print(f"\n=== Sample Output (prompt 0) ===")
        print(f"Text: {results[0]['output_text'][:200]}...")

    if args.compare_baseline:
        print(f"\n=== Running Baseline (no spec decode) ===")
        baseline_tps, baseline_tokens, _ = _benchmark(
            args, None, tree, is_baseline=True
        )
        print(f"tokens/sec: {baseline_tps:.2f}")
        print(f"total tokens: {baseline_tokens}")
        if baseline_tps > 0:
            speedup = tokens_per_s / baseline_tps
            print(f"\n=== Summary ===")
            print(f"Speedup: {speedup:.2f}x")
            if speedup < 1.0:
                print("WARNING: Speculative decoding is slower than baseline!")
            elif speedup >= 2.0:
                print("EXCELLENT: Achieved 2x+ speedup!")


if __name__ == "__main__":
    main()
