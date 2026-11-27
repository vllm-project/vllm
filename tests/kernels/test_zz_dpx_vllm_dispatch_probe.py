# ruff: noqa
#!/usr/bin/env python3
"""dpx_vllm_dispatch_probe.py - vLLM's own ROCm GEMM kernels against hipBLASLt on the same shapes, on THIS device.

Purpose: re-tune the thresholds of vllm/model_executor/layers/utils.py::rocm_unquantized_gemm_impl for a partition
(rows <= 5 -> wvSplitK; 10..128 rows and CuNeeded <= cu_count -> wvSplitKrc; weight bytes <= L3 cap for wvSplitK at
2..5 rows; everything else -> F.linear = hipBLASLt). They were tuned on full MI350/MI355 devices.

Needs an importable vllm with the ROCm extension (vllm._custom_ops). Run on an idle device:

    python dpx_vllm_dispatch_probe.py --out dpx_vllm.jsonl [--triton]
    grep DPXVLLM job.log

or copy it to tests/kernels/test_zz_dpx_vllm_dispatch_probe.py (pytest collects test_dpx_vllm_dispatch_probe).

Per (weight m x k, rows n) it times, with HIP-graph replay over rotating weight copies (cold weights, no launch
overhead; median of 5 rounds):
  hipblaslt        F.linear
  wvSplitK         n <= 5, timed even where the weight-size guard would skip it, so the guard can be placed
  wvSplitKrc@CU    with this device's CU count, when the current rule admits it
  wvSplitKrc@256   with the full-device grid (256 workgroups) whenever the full device would admit it: on a 128-CU
                   partition this is the case the current rule refuses
  triton_bi        the batch-invariant Triton matmul (--triton)
and prints which one the current dispatch takes, which one is fastest, and a sha1 of each output (fixed CPU inputs).
"""

import argparse
import hashlib
import json
import statistics
import sys

import torch
import torch.nn.functional as F

LLC_TARGET_BYTES = 400 * 1024**2  # above the 256 MB MI355X memory-side cache


def n_rot(weight_bytes, lo=2, hi=32):
    return max(lo, min(hi, -(-LLC_TARGET_BYTES // weight_bytes)))


def timeit_graph(calls, inner=2, replays=6, rounds=5, warmup=3, budget_us=None):
    """Median over rounds of the mean time per call (microseconds). calls: list of zero-arg callables, one
    per rotating slot. All launches are captured into one HIP graph and replayed."""
    for c in calls:
        c()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(len(calls) * inner):
            calls[i % len(calls)]()
    torch.cuda.synchronize()
    per = len(calls) * inner
    if budget_us:  # keep very slow candidates from eating the run: one probe replay decides the replay count
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        g.replay()
        torch.cuda.synchronize()
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        one = s.elapsed_time(e) * 1e3
        replays = max(1, min(replays, int(budget_us / max(one, 1.0) / rounds)))
        warmup = min(warmup, replays)
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
    for _ in range(warmup):
        g.replay()
    for s, e in ev:
        s.record()
        for _ in range(replays):
            g.replay()
        e.record()
    torch.cuda.synchronize()
    samples = [s.elapsed_time(e) * 1e3 / (replays * per) for s, e in ev]
    del g
    return statistics.median(samples), min(samples), max(samples)


MODELS = {
    "Llama-3.2-1B": {
        "qkv": (3072, 2048),
        "o": (2048, 2048),
        "gate_up": (16384, 2048),
        "down": (2048, 8192),
        "lm_head": (128256, 2048),
    },
    "Qwen3-0.6B": {
        "qkv": (4096, 1024),
        "o": (1024, 2048),
        "gate_up": (6144, 1024),
        "down": (1024, 3072),
        "lm_head": (151936, 1024),
    },
    "Llama-3.1-70B": {
        "qkv": (10240, 8192),
        "o": (8192, 8192),
        "gate_up": (57344, 8192),
        "down": (8192, 28672),
        "lm_head": (128256, 8192),
    },
    "gpt-oss-120b": {"qkv": (5120, 2880), "o": (2880, 4096), "lm_head": (201088, 2880)},
}
ROWS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 17, 24, 32, 33, 48, 64, 96, 128]
DESIGN_CU = 256  # full MI355X


def wvsplitkrc_dispatch(n, k, m, cu_count):
    """Copy of vllm/model_executor/layers/utils.py::wvsplitkrc_dispatch (main 8c97cff059)."""
    N_p2 = 1 << (n - 1).bit_length()
    GrpsShrB = min(N_p2 // 16, 4)
    CuNeeded = ((m + 64 - 1) // 64) * ((k + 512 - 1) // 512) * GrpsShrB
    chunkk = (
        2 if (N_p2 != 16 and CuNeeded * 2 <= cu_count and (k + 255) // 256 <= 11) else 1
    )
    k_rnd = (k + 512 // chunkk - 1) // (512 // chunkk)
    return (
        chunkk,
        (N_p2 * m * k_rnd <= 128 * 1024 * 12 and CuNeeded <= cu_count),
        CuNeeded,
    )


def rc_ok(n, m, k, cu):
    return (
        10 <= n <= 128
        and k % 8 == 0
        and k > 512
        and m % 16 == 0
        and wvsplitkrc_dispatch(n, k, m, cu)[1]
    )


def sha(t):
    return hashlib.sha1(t.cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:12]


def emit(kind, **f):
    print("DPXVLLM " + json.dumps({"kind": kind, **f}), flush=True)


def main(argv=None):
    import vllm
    import vllm._custom_ops as ops

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--shapes", help="comma list of MxK; overrides --models")
    ap.add_argument("--rows", default=",".join(str(r) for r in ROWS))
    ap.add_argument("--triton", action="store_true")
    ap.add_argument("--budget-ms", type=float, default=500)
    a = ap.parse_args(argv)
    rows = [int(r) for r in a.rows.split(",")]
    if a.shapes:
        layers = [
            ("custom", s, int(s.split("x")[0]), int(s.split("x")[1]))
            for s in a.shapes.split(",")
        ]
    else:
        layers = [
            (mo, la, m, k)
            for mo in a.models.split(",")
            for la, (m, k) in MODELS[mo].items()
        ]
    props = torch.cuda.get_device_properties(0)
    cu = props.multi_processor_count
    free, total = torch.cuda.mem_get_info()
    try:  # the wvSplitK weight cap vLLM derives from the L3 size reported by amdsmi (halves on a DPX/NPS2 partition)
        from vllm.model_executor.layers.utils import _skinny_gemm_max_weight_bytes

        cap = _skinny_gemm_max_weight_bytes()
    except Exception:  # noqa: BLE001
        cap = 256 * 1024 * 1024
    bi = None
    if a.triton:
        from vllm.model_executor.determinism.batch_invariant import (
            linear_batch_invariant as bi,
        )
    dev = dict(
        name=props.name,
        compute_units=cu,
        total_gib=round(total / 2**30, 1),
        free_gib=round(free / 2**30, 1),
        torch=torch.__version__,
        vllm=vllm.__file__,
        wvsplitk_weight_cap_bytes=cap,
    )
    emit("device", **dev)
    fout = open(a.out, "w") if a.out else None
    if fout:
        fout.write(json.dumps({"kind": "device", **dev}) + "\n")
    for model, layer, m, k in layers:
        g = torch.Generator().manual_seed(m * 31 + k)
        base = torch.empty(m, k, dtype=torch.bfloat16, device="cuda")
        step = max(1, (64 << 20) // k)
        for i in range(0, m, step):
            base[i : i + step] = (
                (torch.randn(min(step, m - i), k, generator=g) * 0.05)
                .to(torch.bfloat16)
                .cuda()
            )
        Ws = [base] + [base.roll(r, 0).contiguous() for r in range(1, n_rot(m * k * 2))]
        X = torch.randn(512, k, generator=g).to(torch.bfloat16).cuda()
        for n in rows:
            x = X[:n].contiguous()
            K = {"hipblaslt": lambda W: F.linear(x, W)}
            if n <= 5 and k % 8 == 0 and m > 8:
                K["wvSplitK"] = lambda W: ops.wvSplitK(W, x, cu)
            if rc_ok(n, m, k, cu):
                K["wvSplitKrc@%d" % cu] = lambda W: ops.wvSplitKrc(x, W, cu, None)
            if DESIGN_CU != cu and rc_ok(n, m, k, DESIGN_CU):
                K["wvSplitKrc@%d" % DESIGN_CU] = lambda W: ops.wvSplitKrc(
                    x, W, DESIGN_CU, None
                )
            if bi is not None:
                K["triton_bi"] = lambda W: bi(x, W, None)
            if rc_ok(n, m, k, cu):
                cur = "wvSplitKrc@%d" % cu
            elif n <= 5 and k % 8 == 0 and m > 8 and (n <= 1 or m * k * 2 <= cap):
                cur = "wvSplitK"
            else:
                cur = "hipblaslt"
            rec = {
                "model": model,
                "layer": layer,
                "m": m,
                "n": n,
                "k": k,
                "dispatch": cur,
                "cu_needed": wvsplitkrc_dispatch(n, k, m, cu)[2] if n > 1 else None,
                "t_us": {},
                "sha": {},
            }
            ref = F.linear(x, Ws[0]).float()
            for name, fn in K.items():
                y = fn(Ws[0])
                torch.cuda.synchronize()
                rec["sha"][name] = sha(y)
                if ((y.float() - ref).abs().max() / ref.abs().max()).item() > 0.05:
                    rec["t_us"][name] = None
                    continue
                try:
                    rec["t_us"][name] = round(
                        timeit_graph(
                            [(lambda W=W, fn=fn: fn(W)) for W in Ws],
                            budget_us=a.budget_ms * 1e3,
                        )[0],
                        2,
                    )
                except RuntimeError as e:
                    rec["t_us"][name] = None
                    rec.setdefault("err", {})[name] = str(e)[:160]
            ok = {k_: v for k_, v in rec["t_us"].items() if v}
            rec["best"] = min(ok, key=ok.get)
            rec["dispatch_over_best"] = (
                round(ok[cur] / ok[rec["best"]], 3) if cur in ok else None
            )
            emit("gemm", **rec)
            if fout:
                fout.write(json.dumps({"kind": "gemm", **rec}) + "\n")
                fout.flush()
        del Ws, X, base
        torch.cuda.empty_cache()


def test_dpx_vllm_dispatch_probe():
    if not torch.cuda.is_available() or "rocm" not in torch.__version__:
        import pytest

        pytest.skip("ROCm only")
    main(["--triton"])


if __name__ == "__main__":
    main()
