# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUPTI comparison of SenseNova Loop mHC normalization, projection and merge."""

import json
import statistics

import torch
import torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.kernels.mhc.sensenova_loop import (
    mhc_merge_triton,
    mhc_project_triton,
    mhc_rmsnorm_triton,
)


def norm_reference(x):
    flat = x.flatten(-2).float()
    return flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)


def merge_reference(residual, branch, post, mix):
    mixed = mix.to(residual.dtype) @ residual
    update = post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)
    return mixed + update


def main():
    torch.manual_seed(0)
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__)))
    for streams in (2, 4):
        for tokens in (1, 8, 32, 128, 2048):
            x = torch.randn(tokens, streams, 4096, device="cuda", dtype=torch.bfloat16)
            normalized = norm_reference(x)
            weight = (
                torch.randn(2 * streams + streams**2, streams * 4096, device="cuda")
                * 0.01
            )
            bias = torch.randn(weight.shape[0], device="cuda") * 0.01
            branch = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
            post = torch.randn(tokens, streams, device="cuda")
            mix = torch.randn(tokens, streams, streams, device="cuda")
            for op, reference, candidate, args, rtol, atol in (
                ("normalize", norm_reference, mhc_rmsnorm_triton, (x,), 1e-6, 1e-7),
                (
                    "project",
                    F.linear,
                    mhc_project_triton,
                    (normalized, weight, bias),
                    1e-5,
                    2e-6,
                ),
                (
                    "merge",
                    merge_reference,
                    mhc_merge_triton,
                    (x, branch, post, mix),
                    0,
                    0,
                ),
            ):
                expected = (
                    F.linear(*(arg.double() for arg in args)).float()
                    if op == "project"
                    else reference(*args)
                )
                torch.testing.assert_close(
                    candidate(*args), expected, rtol=rtol, atol=atol
                )
                for backend, fn in (("reference", reference), ("triton", candidate)):
                    for _ in range(5):
                        fn(*args)
                    samples = bench_gpu_time_with_cupti(
                        fn,
                        input_args=args,
                        use_cuda_graph=True,
                        cold_l2_cache=True,
                        dry_run_iters=5,
                        repeat_iters=20,
                    )
                    print(
                        json.dumps(
                            dict(
                                op=op,
                                backend=backend,
                                tokens=tokens,
                                streams=streams,
                                median_us=statistics.median(samples) * 1000,
                            )
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
