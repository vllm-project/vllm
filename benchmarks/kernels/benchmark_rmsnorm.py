import itertools
from typing import Optional, Tuple, Union

import torch
import triton
from flashinfer.norm import fused_add_rmsnorm, rmsnorm
from torch import nn

from vllm import _custom_ops as vllm_ops


class HuggingFaceRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual


def rmsnorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    naive_norm = HuggingFaceRMSNorm(x.shape[-1], eps=eps)
    naive_norm.weight = nn.Parameter(weight)
    naive_norm = naive_norm.to(x.device)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    output = naive_norm(x, residual)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_flashinfer(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        fused_add_rmsnorm(x, residual, weight, eps)
        output = (x, residual)
    else:
        output = rmsnorm(x, weight, eps)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_vllm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        vllm_ops.fused_add_rms_norm(x, residual, weight, eps)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        vllm_ops.rms_norm(out, x, weight, eps)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def calculate_diff(batch_size, seq_len, hidden_size, use_residual=True):
    dtype = torch.bfloat16
    x = torch.randn(batch_size,
                    seq_len,
                    hidden_size,
                    dtype=dtype,
                    device="cuda")
    weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x) if use_residual else None

    output_naive = rmsnorm_naive(
        x.clone(), weight,
        residual.clone() if residual is not None else None)
    output_flashinfer = rmsnorm_flashinfer(
        x.clone(), weight,
        residual.clone() if residual is not None else None)
    output_vllm = rmsnorm_vllm(
        x.clone(), weight,
        residual.clone() if residual is not None else None)

    if use_residual:
        output_naive = output_naive[0]
        output_flashinfer = output_flashinfer[0]
        output_vllm = output_vllm[0]

    print(f"Naive output={output_naive}")
    print(f"FlashInfer output={output_flashinfer}")
    print(f"VLLM output={output_vllm}")

    if torch.allclose(output_naive, output_flashinfer, atol=1e-2,
                      rtol=1e-2) and torch.allclose(
                          output_naive, output_vllm, atol=1e-2, rtol=1e-2):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [2**i for i in range(0, 7, 2)]
seq_length_range = [2**i for i in range(6, 11, 1)]
head_num_range = [32, 48]
configs = list(
    itertools.product(head_num_range, batch_size_range, seq_length_range))


def get_benchmark(use_residual):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["head_num", "batch_size", "seq_len"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["huggingface", "flashinfer", "vllm"],
            line_names=["HuggingFace", "FlashInfer", "vLLM"],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")],
            ylabel="us",
            plot_name=
            f"rmsnorm-perf-{'with' if use_residual else 'without'}-residual",
            args={},
        ))
    def benchmark(head_num, batch_size, seq_len, provider):
        dtype = torch.bfloat16
        hidden_size = head_num * 128  # assuming head_dim = 128

        x = torch.randn(batch_size,
                        seq_len,
                        hidden_size,
                        dtype=dtype,
                        device="cuda")
        weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
        residual = torch.randn_like(x) if use_residual else None

        quantiles = [0.5, 0.2, 0.8]

        if provider == "huggingface":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_naive(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        elif provider == "flashinfer":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_flashinfer(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_vllm(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Hidden size (2nd dimension) of the sequence",
    )
    parser.add_argument("--use-residual",
                        action="store_true",
                        help="Whether to use residual connection")
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/rmsnorm/",
        help="Path to save rmsnorm benchmark results",
    )

    args = parser.parse_args()

    # Run correctness test
    calculate_diff(batch_size=args.batch_size,
                   seq_len=args.seq_len,
                   hidden_size=args.hidden_size,
                   use_residual=args.use_residual)

    # Get the benchmark function with proper use_residual setting
    benchmark = get_benchmark(args.use_residual)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
