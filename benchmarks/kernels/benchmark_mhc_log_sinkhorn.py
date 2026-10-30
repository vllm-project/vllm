# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare log-space mHC Sinkhorn with its PyTorch reference on CUDA.

Run: .venv/bin/python benchmarks/kernels/benchmark_mhc_log_sinkhorn.py
CUPTI measures GPU work with graph replay and cold L2; compilation is excluded.
"""

import json
import statistics
import time

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.kernels.mhc.sensenova_loop import mhc_log_sinkhorn_triton


def reference(mix):
    for _ in range(20):
        mix = mix - mix.logsumexp(-1, keepdim=True)
        mix = mix - mix.logsumexp(-2, keepdim=True)
    return mix.exp()


def main():
    torch.manual_seed(0)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}))
    for streams in (2, 4):
        for tokens in (1, 8, 128, 2048):
            logits = torch.randn(tokens, 2 * streams + streams**2, device="cuda")
            mix = logits[:, 2 * streams :].unflatten(-1, (streams, streams))
            start = time.perf_counter()
            actual = mhc_log_sinkhorn_triton(mix, 20)
            torch.accelerator.synchronize()
            first_call_s = time.perf_counter() - start
            torch.testing.assert_close(actual, reference(mix), rtol=1e-5, atol=1e-7)
            for name, fn in (
                ("torch", reference),
                ("triton", lambda x: mhc_log_sinkhorn_triton(x, 20)),
            ):
                for _ in range(5):
                    fn(mix)
                samples = bench_gpu_time_with_cupti(
                    fn,
                    input_args=(mix,),
                    use_cuda_graph=True,
                    cold_l2_cache=True,
                    dry_run_iters=5,
                    repeat_iters=30,
                )
                print(
                    json.dumps(
                        {
                            "streams": streams,
                            "tokens": tokens,
                            "backend": name,
                            "median_us": statistics.median(samples) * 1000,
                            "first_triton_call_s": first_call_s,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
