# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# fmt: off
# ruff: noqa: E501
import argparse
import copy
import itertools
import time

import torch

from vllm import _custom_ops as ops

from vllm.triton_utils import triton


def get_inputs(m, n, k):
    def _make_sparse_tensors(dtype,
                             m, n, k):
        a = torch.randn((m, k), device='cuda', dtype=dtype)        
        zero_sparse_b = torch.Tensor([0, 0, 1, 1]).to('cuda').to(dtype).tile((n, k // 4)).t()

        # print(f"zero_sparse_b : shape={zero_sparse_b.shape}, \nsnapshot={zero_sparse_b}")

        b = prune_to_2_4(zero_sparse_b.t()).t()

        torch.testing.assert_close(zero_sparse_b, b, rtol=1e-1, atol=1e0)

        compressed_zero_sparse_b, e = ops.cutlass_sparse_compress(zero_sparse_b.t())

        # print(f"compressed_zero_sparse_b : shape={compressed_zero_sparse_b.shape}, \nsnapshot={compressed_zero_sparse_b}")
        # print(f"e : shape={e.shape}, \nsnapshot={e}")

        check_compress_decompress_invariance(dtype, zero_sparse_b, compressed_zero_sparse_b, e)

        return a, zero_sparse_b, compressed_zero_sparse_b, e

    scale_a = torch.randn((1, 1), device="cuda", dtype=torch.float32)
    scale_b = torch.randn((1, 1), device="cuda", dtype=torch.float32)

    a, zero_sparse_b, compressed_zero_sparse_b, e = _make_sparse_tensors(torch.bfloat16, m, n, k)
    return a, zero_sparse_b, compressed_zero_sparse_b, e, scale_a, scale_b


def torch_sparse_tensor_wise_scaled_mm(act, sparsed_w, scale_a, scale_b=None, output=None):
    # scale_b is absorbed into weights
    assert scale_b == None

    scale_a = scale_a.to(torch.bfloat16)
    return torch.mm(act * scale_a, sparsed_w, out=output)


# adpated from test_custlass_2of4_sparse, will be moved to utils
def prune_to_2_4(tensor):
    # Reshape tensor to [N, 4] where N is number of groups of 4
    original_shape = tensor.shape
    reshaped = tensor.reshape(-1, 4)

    # Get indices of top 2 absolute values in each group of 4
    _, indices = torch.topk(torch.abs(reshaped), k=2, dim=1)

    # Create binary mask
    mask = torch.zeros_like(reshaped)
    mask.scatter_(dim=1,
                  index=indices,
                  src=torch.ones_like(indices, dtype=mask.dtype))

    # Apply mask and reshape back
    pruned = reshaped * mask

    # Turn all -0.0 to 0.0
    pruned[pruned == -0.0] = 0.0

    return pruned.reshape(original_shape)


# adpated from test_custlass_2of4_sparse, will be moved to utils
def check_compress_decompress_invariance(dtype: torch.dtype, b: torch.Tensor,
                                         b_compressed: torch.Tensor,
                                         b_metadata: torch.Tensor):

    # For float16 and bfloat16, cutlass_scaled_sparse_mm's output must be the
    # same dtype as its inputs. This line addresses that constraint while
    # arbitrarily using bfloat16 for the int8/fp8 cases.
    out_dtype = torch.float16 if dtype is torch.float16 else torch.bfloat16

    eye = torch.eye(b.shape[0], device='cuda', dtype=dtype)
    eye_scale = torch.ones(1, device='cuda', dtype=torch.float32)
    b_decomp = ops.cutlass_scaled_sparse_mm(eye,
                                            b_compressed,
                                            b_metadata,
                                            eye_scale,
                                            eye_scale,
                                            out_dtype=out_dtype)

    torch.testing.assert_close(b.to(dtype=out_dtype), b_decomp)


def test_cutlass_sparse_subset2():
    m, n, k = 512, 256, 64

    def _make_sparse_tensors(dtype,
                             m, n, k):
        a = torch.randn((m, k), device='cuda', dtype=dtype)        
        zero_sparse_b = torch.Tensor([0, 0, 1, 1]).to('cuda').to(dtype).tile((n, k // 4)).t()

        print(f"zero_sparse_b : shape={zero_sparse_b.shape}, \nsnapshot={zero_sparse_b}")

        b = prune_to_2_4(zero_sparse_b.t()).t()

        torch.testing.assert_close(zero_sparse_b, b, rtol=1e-1, atol=1e0)

        compressed_zero_sparse_b, e = ops.cutlass_sparse_compress(zero_sparse_b.t())

        print(f"compressed_zero_sparse_b : shape={compressed_zero_sparse_b.shape}, \nsnapshot={compressed_zero_sparse_b}")
        print(f"e : shape={e.shape}, \nsnapshot={e}")

        check_compress_decompress_invariance(dtype, zero_sparse_b, compressed_zero_sparse_b, e)

        return a, zero_sparse_b, compressed_zero_sparse_b, e

    def sparse_encode(b):
        sparse_encoded_b = torch.sparse.to_sparse_semi_structured(b)
        return sparse_encoded_b

    a, b, compressed_b, e = _make_sparse_tensors(torch.bfloat16, m, n, k)

    scale_a = torch.Tensor([[1,],]).to('cuda')
    scale_b = torch.Tensor([[1,],]).to('cuda')

    out = ops.cutlass_scaled_sparse_mm(a,
                                       compressed_b,
                                       e,
                                       scale_a,
                                       scale_b,
                                       out_dtype=torch.bfloat16)

    ref_baseline = torch.mm(a * scale_a.to(torch.bfloat16), b * scale_b.to(torch.bfloat16))

    sparse_encoded_b = sparse_encode((b * scale_b.to(torch.bfloat16)).t()).t()

    # NOTE(yiakwy) : torch._scaled_mm does not support bfloat16
    ref = torch.mm(a * scale_a.to(torch.bfloat16), sparse_encoded_b)

    torch.testing.assert_close(ref, ref_baseline, rtol=1e-1, atol=1e0)
    torch.testing.assert_close(ref, out, rtol=1e-1, atol=1e0)


m = [512,]
n = [128,]
k = [64,] 

configs = list(itertools.product(m, n, k))

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=configs,
        x_log=False,
        line_arg="provider",
        line_vals=["cutlass", "torch"],
        line_names=["Cutlass_Sparse_scaled_MM", "Torch_sparse_mm_ref"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="sparse scaled mm performance",
        args={},
    )
)
def benchmark(M, N, K,provider, ):
    a, b, compressed_b, e, scale_a, scale_b = get_inputs(M, N, K)

    out = torch.zeros((M, N), dtype=torch.float16, device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    if provider == "cutlass":
        fn = lambda : ops.cutlass_scaled_sparse_mm(a,
                                                   compressed_b,
                                                   e,
                                                   scale_a,
                                                   scale_b,
                                                   out_dtype=torch.bfloat16)
    else:
        def sparse_encode(b):
            sparse_encoded_b = torch.sparse.to_sparse_semi_structured(b)
            return sparse_encoded_b

        scale_b = scale_b.to(torch.bfloat16)
        sparse_encoded_b = sparse_encode((b * scale_b.to(torch.bfloat16)).t()).t()

        fn = lambda : torch_sparse_tensor_wise_scaled_mm(a, sparse_encoded_b, scale_a, output=out)
    
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return ms * 1000, min_ms * 1000, max_ms * 1000


if __name__ == "__main__":
    benchmark.run(print_data=True)
    pass