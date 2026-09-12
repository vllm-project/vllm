# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
import time
from pathlib import Path

import torch

mode = sys.argv[1]
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
for _ in range(50):
    torch.mm(a, b, out=c)
torch.accelerator.synchronize()


def run(count):
    start = time.perf_counter()
    for _ in range(count):
        torch.mm(a, b, out=c)
    torch.accelerator.synchronize()
    return time.perf_counter() - start


if mode == "calibrate":
    count = min(20000, max(100, round(100 / run(100) * 12)))
    Path("/tmp/gpu-benchmark-iterations").write_text(str(count))
    print(json.dumps({"benchmark_iterations": count}))
else:
    count = int(Path("/tmp/gpu-benchmark-iterations").read_text())
    seconds = run(count)
    print(
        "GPU_BENCHMARK "
        + json.dumps(
            {
                "sampling": int(mode),
                "iterations": count,
                "seconds": seconds,
                "iterations_per_second": count / seconds,
            }
        ),
        flush=True,
    )
