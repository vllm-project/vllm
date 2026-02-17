# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure FFN activation memory waste from CUDA graph batch padding.

Runs a LlamaMLP forward pass with actual vs padded batch sizes and
reports the peak activation memory difference. This is the memory
that split_ffn saves by running FFN eagerly with real batch sizes.

Usage:
    python benchmarks/measure_ffn_padding_overhead.py
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Minimal Llama-style MLP for memory measurement."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        # Fused gate+up projection (like MergedColumnParallelLinear)
        self.gate_up_proj = nn.Linear(
            hidden_size, 2 * intermediate_size, bias=False, dtype=dtype
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = torch.nn.functional.silu(gate) * up
        x = self.down_proj(x)
        return x


def measure_peak_activation(
    mlp: nn.Module,
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> int:
    """Run MLP forward and return peak memory allocated (bytes)."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure baseline (weights already loaded)
    baseline = torch.cuda.memory_allocated()

    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    _ = mlp(x)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    return peak - baseline


def main():
    dtype = torch.bfloat16

    # Model configs: (name, hidden_size, intermediate_size, tp_size)
    configs = [
        ("Llama-3.2-1B", 2048, 8192, 1),
        ("Llama-3.1-8B", 4096, 14336, 1),
        ("Llama-3.1-70B", 8192, 28672, 1),
        ("Llama-3.1-70B/TP8", 8192, 28672, 8),
    ]

    # CUDA graph capture sizes (typical vLLM pattern)
    # Actual batch → padded to next capture size
    test_cases = [
        (1, 1),
        (3, 4),
        (5, 8),
        (13, 16),
        (17, 24),
        (25, 32),
        (50, 56),
        (100, 104),
        (200, 200),
        (250, 256),
    ]

    print(
        f"{'Model':<20} {'Actual':>6} {'Padded':>6} {'Waste':>3} "
        f"{'Actual MB':>10} {'Padded MB':>10} {'Saved MB':>10} {'Saved%':>7}"
    )
    print("-" * 85)

    for name, hidden, intermediate, tp in configs:
        per_gpu_intermediate = intermediate // tp
        mlp = SimpleMLP(hidden, per_gpu_intermediate, dtype).cuda()
        mlp.eval()

        with torch.no_grad():
            for actual, padded in test_cases:
                waste = padded - actual
                if waste == 0 and actual != 1:
                    continue  # skip no-padding cases (uninteresting)

                mem_actual = measure_peak_activation(mlp, actual, hidden, dtype)
                mem_padded = measure_peak_activation(mlp, padded, hidden, dtype)
                saved = mem_padded - mem_actual
                saved_pct = saved / mem_padded * 100 if mem_padded > 0 else 0

                print(
                    f"{name:<20} {actual:>6} {padded:>6} {waste:>3} "
                    f"{mem_actual / 1e6:>10.2f} {mem_padded / 1e6:>10.2f} "
                    f"{saved / 1e6:>10.2f} {saved_pct:>6.1f}%"
                )

        # Clean up
        del mlp
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
