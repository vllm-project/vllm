# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark FlashInfer GDN recurrent-state preparation and writeback."""

import torch

from vllm.model_executor.layers.mamba.ops.gdn_state_io import (
    gather_gdn_initial_state,
    scatter_gdn_final_state,
)
from vllm.triton_utils import triton


def _bench(function) -> list[float]:
    return [
        value * 1000
        for value in triton.testing.do_bench(
            function,
            warmup=100,
            rep=300,
            quantiles=[0.2, 0.5, 0.8],
        )
    ]


def benchmark_case(batch: int, cache_dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    heads, rows, columns = 64, 128, 128
    cache = torch.randn(
        32,
        heads,
        rows,
        columns,
        device="cuda",
        dtype=cache_dtype,
    )
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    has_initial_state = torch.arange(batch, device="cuda") % 2 == 0
    final_state = torch.randn(
        batch,
        heads,
        rows,
        columns,
        device="cuda",
        dtype=torch.float32,
    )

    def gather_pytorch() -> torch.Tensor:
        output = cache[indices]
        output[~has_initial_state] = 0
        return output.float()

    def scatter_pytorch() -> None:
        cache[indices] = final_state.to(cache.dtype)

    gather_reference = gather_pytorch()
    gather_actual = gather_gdn_initial_state(cache, indices, has_initial_state)
    torch.testing.assert_close(gather_actual, gather_reference, atol=0, rtol=0)

    scatter_reference = cache.clone()
    scatter_reference[indices.long()] = final_state.to(cache.dtype)
    scatter_actual = cache.clone()
    scatter_gdn_final_state(scatter_actual, indices, final_state)
    torch.testing.assert_close(scatter_actual, scatter_reference, atol=0, rtol=0)

    gather_baseline_us = _bench(gather_pytorch)
    gather_fused_us = _bench(
        lambda: gather_gdn_initial_state(cache, indices, has_initial_state)
    )
    scatter_baseline_us = _bench(scatter_pytorch)
    scatter_fused_us = _bench(
        lambda: scatter_gdn_final_state(cache, indices, final_state)
    )
    print(
        f"batch={batch:2d} dtype={str(cache_dtype):14s} "
        f"gather={gather_baseline_us[1]:7.2f}->{gather_fused_us[1]:7.2f} us "
        f"({gather_baseline_us[1] / gather_fused_us[1]:4.2f}x) "
        f"scatter={scatter_baseline_us[1]:7.2f}->{scatter_fused_us[1]:7.2f} us "
        f"({scatter_baseline_us[1] / scatter_fused_us[1]:4.2f}x)"
    )


def main() -> None:
    for cache_dtype in (torch.bfloat16, torch.float32):
        for batch in (1, 4, 16):
            benchmark_case(batch, cache_dtype)


if __name__ == "__main__":
    main()
