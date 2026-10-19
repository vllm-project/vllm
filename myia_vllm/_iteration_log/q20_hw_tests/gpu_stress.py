"""Q20 hardware test — GPU VRAM pattern test + compute stress.

Runs on a FREE GPU (GPU 2) in a throwaway container; the production engine
(GPUs 0/1) is never touched. Purpose: detect VRAM bit-rot / driver-level
errors on the card under sustained load, in the context of the 2026-10-18/19
GPU anomalies (CUDA unspecified launch failure + driver-level VRAM leak +
nvlddmkm event bursts during engine wedges).

Run:
  docker run --rm --gpus device=2 -v d:/vllm/myia_vllm/_iteration_log/q20_hw_tests:/q20:ro \
      vllm/vllm-openai:v0.31.0 python3 /q20/gpu_stress.py

Verdict: "RESULT: errs = 0" = VRAM pattern clean. Any corrupted byte or a
process death (driver reset) = hardware/driver fault evidence.
"""
import time

import torch


def main() -> None:
    assert torch.cuda.is_available(), "no CUDA device visible"
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {name} — {props.total_memory / 2**30:.1f} GiB VRAM", flush=True)

    # Phase 1 — VRAM pattern test: fill + verify 18 x 1 GiB blocks, 6 passes.
    # A scalar fill + equality check exercises the full write/read path; any
    # bit corruption shows up as a block that fails its known pattern.
    n_blocks = 18
    blocks = [torch.empty(2**30, dtype=torch.uint8, device="cuda") for _ in range(n_blocks)]
    print(f"allocated {n_blocks} GiB", flush=True)

    errs = 0
    for it in range(6):
        t0 = time.time()
        for i, b in enumerate(blocks):
            b.fill_((i * 7 + it) % 251)
        for i, b in enumerate(blocks):
            expect = (i * 7 + it) % 251
            if not bool((b == expect).all()):
                bad = int((b != expect).sum())
                errs += bad
                print(f"  PASS {it} BLOCK {i}: {bad} corrupted bytes", flush=True)
        torch.cuda.synchronize()
        print(f"pass {it}: ok ({time.time() - t0:.1f}s)", flush=True)

    # Phase 2 — compute stress: 60 s of sustained bf16 matmul (SM + power draw).
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    t0 = time.time()
    n = 0
    while time.time() - t0 < 60:
        a = a @ a
        a = a / a.norm()  # keep values bounded across iterations
        n += 1
    torch.cuda.synchronize()
    print(f"matmul stress: {n} iterations in 60s, finite={bool(torch.isfinite(a).all())}", flush=True)

    # Phase 3 — D2D copy ringing 30 s (copy engines + memory controllers).
    src = torch.empty(2**30, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    t0 = time.time()
    while time.time() - t0 < 30:
        dst.copy_(src)
    torch.cuda.synchronize()
    print("d2d copy 30s: ok", flush=True)

    print(f"RESULT: errs = {errs}", flush=True)


if __name__ == "__main__":
    main()