# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Benchmark ReLUSquaredActivation: custom CUDA kernel vs forward_native, both
# eager and under torch.compile (Inductor fuses relu+square into one kernel).

import itertools

import torch
import torch.nn.functional as F

import vllm.model_executor.layers.activation  # noqa: F401
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed

# Capped so the largest tensor stays under 2**31 elements: the shared activation
# kernel computes the per-token pointer offset (blockIdx.x * d) in 32-bit, which
# overflows for tensors with >2**32 elements. Realistic token counts are well
# below this; the kernel-vs-native gap is already clear at these sizes.
batch_size_range = [1, 16, 128]
seq_len_range = [1, 16, 64, 1024]
intermediate_size = [3072, 9728, 12288]
configs = list(itertools.product(batch_size_range, seq_len_range, intermediate_size))


@default_vllm_config()
def benchmark_relu_squared(
    batch_size: int,
    seq_len: int,
    intermediate_size: int,
    provider: str,
    dtype: torch.dtype,
):
    device = "cuda"
    num_tokens = batch_size * seq_len
    set_random_seed(42)
    torch.set_default_device(device)

    x = torch.randn(num_tokens, intermediate_size, dtype=dtype, device=device)
    out = torch.empty_like(x)

    def native(x: torch.Tensor) -> torch.Tensor:
        return torch.square(F.relu(x))

    # Verify the custom kernel matches the native implementation before timing.
    ref = native(x)
    torch.ops._C.relu_squared(out, x)
    torch.testing.assert_close(out, ref)

    if provider == "custom":
        # Custom CUDA kernel — single fused kernel.
        fn = lambda: torch.ops._C.relu_squared(out, x)
    elif provider == "native":
        # forward_native, eager — relu and square as separate ops.
        fn = lambda: native(x)
    elif provider == "native_compiled":
        # forward_native under torch.compile — Inductor fuses relu+square.
        # This is the real production baseline (custom ops are off when
        # Inductor is enabled), so it is the comparison reviewers care about.
        compiled = torch.compile(native)
        compiled(x)  # warm up / trigger compilation before timing
        fn = lambda: compiled(x)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return ms, max_ms, min_ms


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark ReLUSquaredActivation: custom kernel vs native."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["half", "bfloat16", "float"],
        default="bfloat16",
    )
    args = parser.parse_args()

    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    perf_report = triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "seq_len", "intermediate_size"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["custom", "native_compiled", "native"],
            line_names=[
                "Custom Kernel",
                "Native (torch.compile)",
                "Native (eager)",
            ],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")],
            ylabel="ms",
            plot_name="relu_squared-eager-performance",
            args={},
        )
    )

    perf_report(
        lambda batch_size, seq_len, intermediate_size, provider: benchmark_relu_squared(
            batch_size, seq_len, intermediate_size, provider, dtype
        )
    ).run(print_data=True)
