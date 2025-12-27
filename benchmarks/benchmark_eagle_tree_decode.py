# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import time

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


def _run_once(llm: LLM, prompts: list[str], sampling_params: SamplingParams) -> int:
    outputs = llm.generate(prompts, sampling_params)
    token_count = 0
    for output in outputs:
        token_count += len(output.outputs[0].token_ids)
    return token_count


def _benchmark(
    args: argparse.Namespace, speculative_config: dict | None
) -> tuple[float, int]:
    prompts = [args.prompt] * args.batch_size
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
    llm = _build_llm(args, speculative_config)

    for _ in range(args.warmup_iters):
        _run_once(llm, prompts, sampling_params)

    total_tokens = 0
    total_time_s = 0.0
    for _ in range(args.num_iters):
        start = time.perf_counter()
        total_tokens += _run_once(llm, prompts, sampling_params)
        total_time_s += time.perf_counter() - start

    tokens_per_s = total_tokens / total_time_s if total_time_s > 0 else 0.0
    return tokens_per_s, total_tokens


def main() -> None:
    if not current_platform.is_cuda():
        raise SystemExit("CUDA is required for this benchmark.")

    parser = argparse.ArgumentParser(
        description="Benchmark EAGLE3 tree-based speculative decoding."
    )
    parser.add_argument("--model", required=True, help="Target model path.")
    parser.add_argument("--draft-model", required=True, help="EAGLE3 draft model path.")
    parser.add_argument(
        "--method",
        default="eagle3",
        choices=("eagle", "eagle3"),
        help="Speculative method to use.",
    )
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
    print(f"=== Tree Configuration ===")
    print(f"Tree type: {args.tree_type}")
    print(f"Max depth: {args.num_spec_tokens}")
    print(f"Total draft tokens: {num_tree_tokens}")
    print(f"Tree: {tree[:100]}..." if len(tree) > 100 else f"Tree: {tree}")

    spec_config = {
        "method": args.method,
        "model": args.draft_model,
        "num_speculative_tokens": num_tree_tokens,  # Use actual tree size
        "speculative_token_tree": tree,
        "max_model_len": args.max_model_len,
    }

    tokens_per_s, total_tokens = _benchmark(args, spec_config)
    print("=== EAGLE3 tree decode ===")
    print(f"tokens/sec: {tokens_per_s:.2f}")
    print(f"total tokens: {total_tokens}")

    if args.compare_baseline:
        baseline_tps, baseline_tokens = _benchmark(args, None)
        print("=== Baseline (no spec decode) ===")
        print(f"tokens/sec: {baseline_tps:.2f}")
        print(f"total tokens: {baseline_tokens}")
        if baseline_tps > 0:
            print(f"speedup: {tokens_per_s / baseline_tps:.2f}x")


if __name__ == "__main__":
    main()
