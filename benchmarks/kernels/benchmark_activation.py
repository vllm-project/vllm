# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# benchmark custom activation op performance
import random

import torch

from vllm.model_executor.layers.activation import (
    FastGELU,
    FatreluAndMul,
    GeluAndMul,
    MulAndSilu,
    NewGELU,
    QuickGELU,
    SiluAndMul,
    SwigluOAIAndMul,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, FlexibleArgumentParser


@torch.inference_mode()
def bench(
    func_name: str,
    num_tokens: int,
    dim: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    warmup: int = 10,
    num_iters: int = 100,
):
    current_platform.seed_everything(seed)
    torch.set_default_device(device)

    if func_name == "silu_and_mul":
        layer = SiluAndMul()
    elif func_name == "mul_and_silu":
        layer = MulAndSilu()
    elif func_name == "gelu":
        layer = GeluAndMul(approximate="none")
    elif func_name == "gelu_tanh":
        layer = GeluAndMul(approximate="tanh")
    elif func_name == "fatrelu":
        threshold = random.uniform(0, 1)
        layer = FatreluAndMul(threshold)
    elif func_name == "swigluoai_and_mul":
        layer = SwigluOAIAndMul()
    elif func_name == "new_gelu":
        layer = NewGELU()
    elif func_name == "fast_gelu":
        layer = FastGELU()
    elif func_name == "quick_gelu":
        layer = QuickGELU()

    x = torch.randn(num_tokens, dim, dtype=dtype)
    compiled_layer = torch.compile(layer.forward_native)
    t = triton.testing.do_bench(lambda: layer(x), warmup=warmup, rep=num_iters)
    t_compiled = triton.testing.do_bench(
        lambda: compiled_layer(x), warmup=warmup, rep=num_iters
    )

    print(f"Benchmark results for {func_name}:")
    print(f"  Input shape: {x.shape}, dtype: {dtype}, device: {device}")
    print(f"  Custom OP: {t:.4f} ms")
    print(f"  Compiled: {t_compiled:.4f} ms")
    if (
        isinstance(t, (int, float))
        and isinstance(t_compiled, (int, float))
        and t_compiled not in [0, None]
    ):
        print(f"  Speedup: {t_compiled / t:.2f}x")
    else:
        print("  Speedup: N/A (invalid benchmark results)")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the layernorm kernel.")
    parser.add_argument(
        "--func-name",
        type=str,
        choices=[
            "mul_and_silu",
            "silu_and_mul",
            "gelu",
            "gelu_tanh",
            "fatrelu",
            "swigluoai_and_mul",
            "new_gelu",
            "fast_gelu",
            "quick_gelu",
        ],
        default="mul_and_silu",
    )
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="half"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-warmup-iters", type=int, default=10)
    parser.add_argument(
        "--num-iters", type=int, default=200, help="Number of benchmark iterations. "
    )
    args = parser.parse_args()
    print(args)
    if args is not None:
        bench(
            func_name=args.func_name,
            num_tokens=args.num_tokens,
            dim=args.dim,
            dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
            seed=args.seed,
            device="cuda",
            warmup=args.num_warmup_iters,
            num_iters=args.num_iters,
        )
    else:
        print(
            "Error: Failed to parse arguments. Please check your command line inputs."
        )
