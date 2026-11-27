# ruff: noqa
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TEMPORARY diagnostic, not for merge: which BF16 GEMM kernel serves each
(weight shape, rows) on this device, its output bits and its time.

Read the job log with: grep DPXDIAG
"""

import hashlib
import json
import subprocess
import time

import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")

# (out_features, in_features) of the dense layers of two Llama sizes.
SHAPES = {
    "1b_qkv": (3072, 2048),
    "1b_o": (2048, 2048),
    "1b_gate_up": (16384, 2048),
    "1b_down": (2048, 8192),
    "1b_lm_head": (128256, 2048),
    "70b_qkv": (10240, 8192),
    "70b_o": (8192, 8192),
    "70b_gate_up": (57344, 8192),
    "70b_down": (8192, 28672),
}
ROWS = [
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    12,
    16,
    17,
    24,
    32,
    48,
    64,
    96,
    128,
    129,
    192,
    256,
    512,
]
REPEATS, ITERS = 5, 60


def out(kind: str, **fields) -> None:
    print("DPXDIAG " + json.dumps({"kind": kind, **fields}), flush=True)


def sha(t: torch.Tensor) -> str:
    return hashlib.sha1(t.float().cpu().numpy().tobytes()).hexdigest()[:12]


def timed(fn) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    best = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
        best.append((time.perf_counter() - start) / ITERS * 1e6)
    return sorted(best)[len(best) // 2]


def shell(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        ).stdout[-4000:]
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def test_dpx_gemm_diag(monkeypatch):
    from vllm.model_executor.determinism.batch_invariant import linear_batch_invariant
    from vllm.model_executor.layers import utils as layer_utils
    from vllm.platforms import rocm

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    out(
        "device",
        name=props.name,
        compute_units=props.multi_processor_count,
        total_gib=round(total / 2**30, 2),
        torch=torch.__version__,
        gfx950=rocm.on_gfx950(),
        mi3xx=rocm.on_mi3xx(),
    )
    for cmd in (
        ["amd-smi", "partition", "--current"],
        ["amd-smi", "static", "--partition"],
        ["rocm-smi", "--showhw"],
    ):
        out("shell", cmd=" ".join(cmd), stdout=shell(cmd))

    # Which kernel vLLM's dispatch takes: count calls into each backend.
    calls: dict[str, int] = {}
    raw_linear = F.linear

    def counting(name, fn):
        def wrapper(*args, **kwargs):
            calls[name] = calls.get(name, 0) + 1
            return fn(*args, **kwargs)

        return wrapper

    for name in ("wvSplitK", "wvSplitKrc", "LLMM1"):
        monkeypatch.setattr(ops, name, counting(name, getattr(ops, name)))
    monkeypatch.setattr(F, "linear", counting("torch", raw_linear))
    try:
        from aiter.tuned_gemm import tgemm

        monkeypatch.setattr(tgemm, "mm", counting("aiter_tgemm", tgemm.mm))
    except Exception:  # noqa: BLE001
        pass

    kernels = {
        "vllm": layer_utils.rocm_unquantized_gemm_impl,
        "torch": raw_linear,
        "bi": linear_batch_invariant,
    }
    cu = props.multi_processor_count
    gen = torch.Generator().manual_seed(1234)
    for label, (m, k) in SHAPES.items():
        weight = (torch.randn(m, k, generator=gen) * 0.02).to(torch.bfloat16).cuda()
        acts = torch.randn(max(ROWS), k, generator=gen).to(torch.bfloat16).cuda()
        for n in ROWS:
            x = acts[:n].contiguous()
            calls.clear()
            kernels["vllm"](x, weight)
            row = {
                "shape": label,
                "m": m,
                "k": k,
                "n": n,
                "vllm_kernel": "+".join(sorted(calls)) or "none",
                "fits_krc_128": bool(layer_utils.wvsplitkrc_dispatch(n, k, m, 128)[1]),
                "fits_krc_256": bool(layer_utils.wvsplitkrc_dispatch(n, k, m, 256)[1]),
                "fits_krc_here": bool(layer_utils.wvsplitkrc_dispatch(n, k, m, cu)[1]),
            }
            for name, fn in kernels.items():
                y = fn(x, weight)
                row[f"sha_{name}"] = sha(y)
                row[f"row0_{name}"] = sha(y[0])
                row[f"us_{name}"] = round(
                    timed(lambda fn=fn, x=x, w=weight: fn(x, w)), 1
                )
            out("gemm", **row)
        del weight, acts
        torch.cuda.empty_cache()


def test_dispatch_overhead():
    """Host cost of vLLM's GEMM dispatch against calling the kernel directly."""
    from vllm.model_executor.layers import utils as layer_utils
    from vllm.platforms import rocm

    cu = torch.cuda.get_device_properties(0).multi_processor_count
    gen = torch.Generator().manual_seed(7)
    for label, (m, k) in (("1b_qkv", (3072, 2048)), ("70b_o", (8192, 8192))):
        weight = (torch.randn(m, k, generator=gen) * 0.02).to(torch.bfloat16).cuda()
        for n in (1, 9, 12, 64):
            x = torch.randn(n, k, generator=gen).to(torch.bfloat16).cuda()
            row = {
                "shape": label,
                "n": n,
                "us_torch_direct": round(
                    timed(lambda x=x, w=weight: F.linear(x, w)), 1
                ),
                "us_impl": round(
                    timed(
                        lambda x=x, w=weight: layer_utils.rocm_unquantized_gemm_impl(
                            x, w
                        )
                    ),
                    1,
                ),
                "us_custom_op": round(
                    timed(
                        lambda x=x, w=weight: torch.ops.vllm.rocm_unquantized_gemm(
                            x, w, None
                        )
                    ),
                    1,
                ),
            }
            if 10 <= n <= 128 and rocm.on_gfx950():
                for grid in sorted({cu, 256}):
                    try:
                        y = ops.wvSplitKrc(x, weight, grid, None)
                        row[f"sha_krc_{grid}"] = sha(y)
                        row[f"us_krc_{grid}"] = round(
                            timed(
                                lambda x=x, w=weight, g=grid: ops.wvSplitKrc(
                                    x, w, g, None
                                )
                            ),
                            1,
                        )
                    except Exception as e:  # noqa: BLE001
                        row[f"krc_{grid}_error"] = f"{type(e).__name__}: {e}"[:200]
                row["sha_torch"] = sha(F.linear(x, weight))
            out("overhead", **row)
        del weight
