# ruff: noqa
#!/usr/bin/env python3
"""dpx_aiter_probe.py - what AITER's tuned bf16 GEMM (vLLM's gfx950 fall-through when VLLM_ROCM_USE_AITER=1)
picks on THIS device, against hipBLASLt (F.linear) and against the pick AITER makes for a full 256-CU device.

AITER keys its tuned table on (gfx, cu_num); the shipped gfx950 rows are all cu_num=256, so a 128-CU DPX
partition misses every row and falls back to torch/hipBLASLt. This probe prints, per (weight m x k, rows n):
  libtype@dev   backend AITER picks with this device's CU count
  libtype@256   backend AITER picks for cu_num=256 (forced)
  t_us          hipblaslt (F.linear) | aiter@dev (tgemm.mm) | aiter@256 (tgemm.mm with the 256-CU row forced)
  sha           output hash of each (fixed CPU-generated inputs)
Needs torch (ROCm) + aiter. Run on an idle device:  python dpx_aiter_probe.py --out dpx_aiter.jsonl ; grep DPXAITER
Timing: HIP-graph replay over rotating weight copies, median of 5 rounds (same harness as dpx_gemm_probe.py).
"""

import argparse, hashlib, json, logging, statistics, sys
import torch
import torch.nn.functional as F

LLC_TARGET_BYTES = 400 * 1024**2


def n_rot(weight_bytes, lo=2, hi=32):
    return max(lo, min(hi, -(-LLC_TARGET_BYTES // weight_bytes)))


def timeit_graph(calls, inner=2, replays=6, rounds=5, warmup=3):
    for c in calls:
        c()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(len(calls) * inner):
            calls[i % len(calls)]()
    torch.cuda.synchronize()
    per = len(calls) * inner
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
    for _ in range(warmup):
        g.replay()
    for s, e in ev:
        s.record()
        for _ in range(replays):
            g.replay()
        e.record()
    torch.cuda.synchronize()
    del g
    return statistics.median(s.elapsed_time(e) * 1e3 / (replays * per) for s, e in ev)


SHAPES = {
    "1b_o": (2048, 2048),
    "1b_gate_up": (16384, 2048),
    "q06_qkv": (4096, 1024),
    "q06_gate_up": (6144, 1024),
    "70b_qkv": (10240, 8192),
    "70b_o": (8192, 8192),
    "70b_gate_up": (57344, 8192),
    "70b_down": (8192, 28672),
    "mixtral_o": (6144, 6144),
}
ROWS = [6, 8, 16, 32, 64, 128, 256, 512]


def sha(t):
    return hashlib.sha1(t.cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:12]


def emit(kind, **f):
    print("DPXAITER " + json.dumps({"kind": kind, **f}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    ap.add_argument("--shapes", default=",".join(SHAPES))
    a = ap.parse_args(argv)
    logging.getLogger("aiter").setLevel(logging.ERROR)
    import aiter.ops.opus._arch as arch
    from aiter.tuned_gemm import get_GEMM_A16W16_config, tgemm

    real = arch._device_arch_and_cu
    gfx, cu = real(torch.device("cuda:0"))
    free, total = torch.cuda.mem_get_info()
    dev = dict(
        name=torch.cuda.get_device_properties(0).name,
        gfx=gfx,
        aiter_cu=cu,
        torch=torch.__version__,
        free_gib=round(free / 2**30, 1),
        total_gib=round(total / 2**30, 1),
    )
    emit("device", **dev)
    fout = open(a.out, "w") if a.out else None
    if fout:
        fout.write(json.dumps({"kind": "device", **dev}) + "\n")

    def with_cu(c, fn):
        arch._device_arch_and_cu = lambda d: (gfx, c)
        try:
            return fn()
        finally:
            arch._device_arch_and_cu = real

    for label in a.shapes.split(","):
        m, k = SHAPES[label]
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
        for n in [int(r) for r in a.rows.split(",")]:
            x = X[:n].contiguous()
            cfg = {
                c: get_GEMM_A16W16_config(
                    M=n,
                    N=m,
                    K=k,
                    bias=False,
                    dtype="torch.bfloat16",
                    otype="torch.bfloat16",
                    gfx=gfx,
                    cu_num=c,
                )
                for c in {cu, 256}
            }
            K = {
                "hipblaslt": lambda W: F.linear(x, W),
                "aiter@dev": lambda W: tgemm.mm(x, W, None),
            }
            if cu != 256:
                K["aiter@256"] = lambda W: with_cu(256, lambda: tgemm.mm(x, W, None))
            rec = {
                "shape": label,
                "m": m,
                "n": n,
                "k": k,
                "libtype@dev": cfg[cu]["libtype"],
                "libtype@256": cfg[256]["libtype"],
                "kernel@256": str(cfg[256].get("kernelName"))[:80],
                "t_us": {},
                "sha": {},
            }
            ref = F.linear(x, Ws[0]).float()
            for name, fn in K.items():
                try:
                    y = fn(Ws[0])
                    torch.cuda.synchronize()
                    rec["sha"][name] = sha(y)
                    if ((y.float() - ref).abs().max() / ref.abs().max()).item() > 0.05:
                        rec["t_us"][name] = None
                        rec.setdefault("err", {})[name] = "wrong result"
                        continue
                    rec["t_us"][name] = round(
                        timeit_graph([(lambda W=W, fn=fn: fn(W)) for W in Ws]), 2
                    )
                except Exception as e:  # noqa: BLE001
                    rec["t_us"][name] = None
                    rec.setdefault("err", {})[name] = repr(e)[:200]
            emit("gemm", **rec)
            if fout:
                fout.write(json.dumps({"kind": "gemm", **rec}) + "\n")
                fout.flush()
        del Ws, X, base
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


def test_dpx_aiter_probe():
    import pytest

    if not torch.cuda.is_available() or "rocm" not in torch.__version__:
        pytest.skip("ROCm only")
    pytest.importorskip("aiter")
    main([])
