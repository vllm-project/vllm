#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import OrderedDict
from typing import List, Optional, Tuple, Union
import argparse
import json

# Third Party
from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import torch

# Constants
DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B"
DEFAULT_TOKENS_PER_GB = 8200  # Default for Llama-3.1; More details here: https://docs.lmcache.ai/getting_started/kv_cache_calculator.html
DEFAULT_POOL_SIZES_GB: List[Union[int, float, str]] = [
    1,
    2,
    4,
    8,
    16,
    32,
    50,
    100,
    200,
    500,
    "unlimited",
]


class LRUTokenPool:
    """
    Token pool with LRU eviction policy based on token count limit.
    """

    def __init__(self, max_tokens: float) -> None:
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.requests: OrderedDict[int, List[int]] = OrderedDict()

    def longest_prefix_len(self, tokens: List[int]) -> Tuple[int, int]:
        """
        Find longest prefix match and update LRU ordering.
        For request i (1-indexed):
        y[i] = y[i-1] + (len(tokens[i]) - max_shared_prefix(tokens[i], any previous))
        """
        best_len = 0
        best_id = -1

        for req_id, req_tokens in self.requests.items():
            common_len = 0
            for i in range(min(len(tokens), len(req_tokens))):
                if tokens[i] == req_tokens[i]:
                    common_len += 1
                else:
                    break

            if common_len > best_len:
                best_len = common_len
                best_id = req_id

        # Update LRU ordering
        if best_id != -1:
            self.requests.move_to_end(best_id)

        return best_len, best_id

    def longest_common_substring(
        self,
        request_id: int,
        token_tensor: torch.Tensor,
        tokens: List[int],
        *,
        chunk_len: int = 4,
        stride_r: int = 4,
        chunk_batch: int = 512,
    ) -> Tuple[int, float]:
        """
        For token_tensor[request_id], chunk it and check whether each chunk
        appears contiguously in any previous request (token_tensor[:request_id]).
        Returns (total_tokens_matched, elapsed_seconds).
        """
        assert token_tensor.ndim == 2, "Expected [N, T] tensor"
        N, T = token_tensor.shape
        assert 0 <= request_id < N, "request_id out of range"

        if request_id == 0 or T < chunk_len:
            return 0, 0

        r = token_tensor[request_id]  # [T]
        r = r[: len(tokens)]
        Xprev = token_tensor[:request_id]  # [request_id, T]

        # Sliding windows for previous rows
        Xw = Xprev.unfold(dimension=1, size=chunk_len, step=1)  # [R, W, L]
        # Chunks of r
        r_chunks = r.unfold(dimension=0, size=chunk_len, step=stride_r)  # [C, L]
        if r_chunks.numel() == 0:
            return 0, 0

        total_matched_chunks = 0

        # Process in mini-batches to control memory
        for b in range(0, r_chunks.size(0), chunk_batch):
            rc = r_chunks[b : b + chunk_batch]  # [B, L]
            eq = Xw[:, :, None, :] == rc[None, None, :, :]
            full = eq.all(dim=-1)  # [R, W, B]
            # Count how many unique chunks matched (across all previous rows)
            matched_chunk_indices = torch.unique(full.nonzero(as_tuple=True)[2])
            total_matched_chunks += matched_chunk_indices.numel()

        total_tokens_matched = total_matched_chunks * chunk_len

        return total_tokens_matched, 0

    def add_request(
        self,
        request_id: int,
        tokens: List[int],
        token_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Add a request to the pool, evicting LRU entries if necessary.
        """
        # Evict until we have space
        while self.current_tokens + len(tokens) > self.max_tokens and self.requests:
            old_id, old_tokens = self.requests.popitem(last=False)
            self.current_tokens -= len(old_tokens)

            # substring matching case
            if token_tensor is not None:
                token_tensor[old_id, :] = 0

        # Add new request
        self.requests[request_id] = tokens
        self.current_tokens += len(tokens)


def load_and_tokenize_inputs(
    jsonl_path: str, tokenizer_name: str = DEFAULT_TOKENIZER
) -> Tuple[List[List[int]], torch.Tensor]:
    """
    Load and tokenize inputs from a JSONL file.

    Returns:
        Tuple of (tokenized_sequences_list, tokenized_sequences_tensor)
        - tokenized_sequences_list: List of token lists
        - tokenized_sequences_tensor: Padded 2D tensor (sequences, tokens)
          Sequences are padded with 0s to match the longest sequence.
    """
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Reading and tokenizing inputs from: {jsonl_path}")
    tokenized_sequences = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Tokenizing"):
        try:
            data = json.loads(line.strip())
            input_text = data.get("input", "")
            tokens = tokenizer.encode(input_text)
            tokenized_sequences.append(tokens)
        except Exception as e:
            print(f"Warning: Failed to process line: {e}")
            tokenized_sequences.append([])

    if tokenized_sequences:
        max_length = max(len(seq) for seq in tokenized_sequences)
        num_sequences = len(tokenized_sequences)

        # Create padded tensor (pad with 0s)
        tokenized_tensor = torch.zeros((num_sequences, max_length), dtype=torch.long)
        for i, seq in enumerate(tokenized_sequences):
            if seq:
                tokenized_tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    else:
        tokenized_tensor = torch.tensor([], dtype=torch.long)

    return tokenized_sequences, tokenized_tensor


def calculate_hit_rate(
    token_sequences: List[List[int]],
    pool_size: Optional[int] = None,
    token_tensor: Optional[torch.Tensor] = None,
    method: str = "prefix",
) -> float:
    # Use float('inf') for unlimited case to avoid eviction
    max_tokens = float("inf") if pool_size is None else pool_size
    pool = LRUTokenPool(max_tokens)

    total_tokens = 0
    hit_tokens = 0

    total_lcs_time_s = 0.0
    lcs_calls = 0

    for idx, tokens in tqdm(list(enumerate(token_sequences))):
        total_tokens += len(tokens)

        if method == "prefix":
            if idx > 0:
                common, _ = pool.longest_prefix_len(tokens)
                hit_tokens += common
            pool.add_request(idx, tokens)
        elif method == "substring" and token_tensor is not None:
            if idx > 0:
                common, elapsed = pool.longest_common_substring(
                    idx, token_tensor, tokens
                )
                hit_tokens += common
                total_lcs_time_s += elapsed
                lcs_calls += 1
            pool.add_request(idx, tokens, token_tensor)
        else:
            raise ValueError(f"Invalid method: {method}")

    if method == "substring":
        avg_ms = (total_lcs_time_s / lcs_calls * 1000.0) if lcs_calls > 0 else 0.0
        print(
            f"  [Timing] longest_common_substring: total {total_lcs_time_s:.3f}s, "
            f"calls {lcs_calls}, avg {avg_ms:.2f} ms"
        )

    return hit_tokens / total_tokens if total_tokens > 0 else 0.0


def analyze_hit_rates_across_pool_sizes(
    token_sequences: List[List[int]],
    pool_sizes_gb: List[Union[int, float, str]],
    tokens_per_gb: int,
    token_tensor: Optional[torch.Tensor] = None,
) -> Tuple[List[float], List[float], List[str]]:
    print("\nAnalyzing hit rates across pool sizes...")
    print("=" * 60)

    prefix_hit_rates = []
    substring_hit_rates = []
    x_labels = []

    for size_gb in pool_sizes_gb:
        if size_gb == "unlimited":
            size_tokens = None
            x_labels.append("∞")
            pool_desc = "unlimited"
            token_desc = ""
        else:
            size_tokens = int(size_gb * tokens_per_gb)
            x_labels.append(str(int(size_gb)))
            pool_desc = f"{size_gb}GB"
            token_desc = f" ({size_tokens:,} tokens)"

        print(f"Testing pool size: {pool_desc}{token_desc}")

        # For every pool size round, we should start from fresh
        tensor_copy = token_tensor.clone() if token_tensor is not None else None

        prefix_hit_rate = calculate_hit_rate(
            token_sequences, size_tokens, tensor_copy, method="prefix"
        )
        prefix_hit_rates.append(prefix_hit_rate)
        print(f"  Prefix: {prefix_hit_rate:.4f} ({prefix_hit_rate * 100:.2f}%)")

        substring_hit_rate = calculate_hit_rate(
            token_sequences, size_tokens, tensor_copy, method="substring"
        )
        substring_hit_rates.append(substring_hit_rate)
        print(
            f"  Substring: {substring_hit_rate:.4f} ({substring_hit_rate * 100:.2f}%)\n"
        )

    print("=" * 60)
    return prefix_hit_rates, substring_hit_rates, x_labels


def plot_hit_rates(
    prefix_hit_rates: List[float],
    substring_hit_rates: List[float],
    x_labels: List[str],
    output_path: str,
) -> None:
    """
    Generate and save the hit rate vs pool size plot comparing both methods.
    """
    plt.figure(figsize=(12, 7))

    # Plot prefix
    plt.plot(
        range(len(prefix_hit_rates)),
        prefix_hit_rates,
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
        label="Prefix Matching",
    )

    # Plot substring
    plt.plot(
        range(len(substring_hit_rates)),
        substring_hit_rates,
        marker="s",
        linewidth=2,
        markersize=8,
        color="#A23B72",
        label="Substring Matching",
    )

    plt.xlabel("Pool Size (GB)", fontsize=12, fontweight="bold")
    plt.ylabel("Hit Rate", fontsize=12, fontweight="bold")
    plt.title(
        "Cache Hit Rate vs Pool Size: Prefix vs Substring Matching",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(range(len(x_labels)), x_labels, rotation=45)
    plt.grid(True, alpha=0.3, linestyle="--")

    # Set y-axis limit based on max of both methods
    max_rate = max(max(prefix_hit_rates), max(substring_hit_rates))
    plt.ylim(0, min(1.0, max_rate * 1.1))
    plt.legend(loc="best", fontsize=10)

    # Annotate prefix matching rates
    for i, (rate, label) in enumerate(zip(prefix_hit_rates, x_labels, strict=False)):
        plt.annotate(
            f"{rate * 100:.1f}%",
            xy=(i, rate),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#2E86AB",
        )

    # Annotate substring matching rates
    for i, (rate, label) in enumerate(zip(substring_hit_rates, x_labels, strict=False)):
        plt.annotate(
            f"{rate * 100:.1f}%",
            xy=(i, rate),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#A23B72",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze prefix cache hit rates across different pool sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i trace.jsonl
  %(prog)s -i trace.jsonl -o custom_output.png
  %(prog)s -i trace.jsonl --pool-sizes 1 2 4 8 16 unlimited
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file (trace.jsonl)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="prefix_cache_hit_rate.png",
        help="Path to output plot file (PNG) (default: prefix_cache_hit_rate.png)",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"HuggingFace tokenizer model name (default: {DEFAULT_TOKENIZER})",
    )

    parser.add_argument(
        "--tokens-per-gb",
        type=int,
        default=DEFAULT_TOKENS_PER_GB,
        help=f"Conversion factor from GB to tokens "
        f"(default: {DEFAULT_TOKENS_PER_GB}). "
        "This should be adjusted when using a different tokenizer.",
    )

    parser.add_argument(
        "--pool-sizes",
        nargs="+",
        default=None,
        help='Pool sizes in GB to test (space-separated, can include "unlimited"). '
        f"Default: {' '.join(map(str, DEFAULT_POOL_SIZES_GB))}",
    )

    return parser.parse_args()


def parse_pool_sizes(
    pool_sizes_input: Optional[List[str]],
) -> List[Union[int, float, str]]:
    if pool_sizes_input is None:
        return DEFAULT_POOL_SIZES_GB

    parsed_sizes: List[Union[int, float, str]] = []
    for size in pool_sizes_input:
        if size.lower() == "unlimited":
            parsed_sizes.append("unlimited")
        else:
            try:
                parsed_sizes.append(float(size))
            except ValueError:
                raise ValueError(
                    f"Invalid pool size: {size}. Must be a number or 'unlimited'"
                ) from None

    return parsed_sizes


def main() -> None:
    args = parse_arguments()

    # Parse pool sizes
    pool_sizes_gb = parse_pool_sizes(args.pool_sizes)

    print("Configuration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Tokens per GB: {args.tokens_per_gb}")
    print(f"  Pool sizes: {pool_sizes_gb}\n")

    # Load and tokenize inputs
    token_sequences, token_tensor = load_and_tokenize_inputs(args.input, args.tokenizer)
    print(f"Loaded {len(token_sequences)} requests")
    print(f"Token tensor shape: {token_tensor.shape} (padded with 0s)")
    print(f"First sequence: {token_tensor[0]}")

    # Analyze hit rates using both methods
    prefix_hit_rates, substring_hit_rates, x_labels = (
        analyze_hit_rates_across_pool_sizes(
            token_sequences,
            pool_sizes_gb,
            args.tokens_per_gb,
            token_tensor,
        )
    )

    # Generate comparison plot
    plot_hit_rates(prefix_hit_rates, substring_hit_rates, x_labels, args.output)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
