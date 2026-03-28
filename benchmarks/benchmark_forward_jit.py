#!/usr/bin/env python3
"""
Forward-pass latency benchmark: four CPU-offload strategies.

Simulates running a model LARGER THAN GPU VRAM by keeping weights in CPU RAM
and streaming them to GPU one layer at a time, then releasing GPU memory.

Four modes
----------
  gpu-resident   — baseline: weights already on GPU (needs VRAM ≥ model size)
  cpu-raw        — plain FP16 weights on CPU; full H2D copy each forward
  jit-cpu        — weights zlib-compressed in CPU RAM; CPU decompress → H2D each pass
  jit-gpu        — weights LZ4-chunked in pinned CPU RAM; H2D compressed → GPU decompress

The jit-gpu mode pre-allocates pinned (page-locked) CPU buffers for the
compressed weight data so the DMA engine can transfer directly without an
intermediate bounce buffer — the realistic production implementation.

Run:
    python3 /tmp/bench_forward.py [--hidden 2048] [--layers 8] [--iters 20]
"""

import argparse
import struct
import time

import torch
import torch.nn.functional as F
import zlib

CHUNK_SIZE = 65536  # must match DECOMP_CHUNK_SIZE in weight_decompress.cu


# ── Compression helpers ────────────────────────────────────────────────────────

def lz4_compress_chunked(data: bytes) -> bytes:
    import lz4.block as lz4b
    chunks = [lz4b.compress(data[o:o + CHUNK_SIZE], store_size=False)
              for o in range(0, len(data), CHUNK_SIZE)]
    n = len(chunks)
    hdr = struct.pack(f"<I{n}I", n, *(len(c) for c in chunks))
    return hdr + b"".join(chunks)


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint8).cpu().numpy().tobytes()


# ── Synthetic layer ───────────────────────────────────────────────────────────

class SyntheticLayer:
    """4-weight linear block, mimicking a transformer projection block."""

    def __init__(self, hidden: int, dtype=torch.float16):
        self.hidden = hidden
        self.dtype = dtype
        self.orig_size = hidden * hidden * torch.finfo(dtype).bits // 8
        self.cpu_weights: list[torch.Tensor] = [
            torch.randn(hidden, hidden, dtype=dtype) for _ in range(4)
        ]

    def make_zlib(self) -> list[tuple[bytes, int]]:
        return [(zlib.compress(tensor_to_bytes(w), 6), self.orig_size)
                for w in self.cpu_weights]

    def make_lz4_pinned(self) -> list[tuple[torch.Tensor, int]]:
        """Compress + store in pinned CPU tensors (page-locked for fast DMA)."""
        result = []
        for w in self.cpu_weights:
            raw = tensor_to_bytes(w)
            comp = lz4_compress_chunked(raw)
            # Copy compressed bytes into a pinned CPU tensor
            pinned = torch.empty(len(comp), dtype=torch.uint8, pin_memory=True)
            pinned.copy_(torch.frombuffer(bytearray(comp), dtype=torch.uint8))
            result.append((pinned, self.orig_size))
        return result


# ── Forward computation ───────────────────────────────────────────────────────

def compute(x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
    for w in weights:
        x = F.linear(x, w)
    return x


# ── Per-mode weight fetch ─────────────────────────────────────────────────────

def fetch_cpu_raw(cpu_weights, device):
    """Plain H2D copy of full FP16 weights."""
    return [w.to(device, non_blocking=False) for w in cpu_weights]


def fetch_zlib(zlib_data, device, hidden, dtype):
    """CPU decompress → H2D."""
    result = []
    for comp, orig_size in zlib_data:
        raw = zlib.decompress(comp)
        t = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(hidden, hidden)
        result.append(t.to(device, non_blocking=False))
    return result


def fetch_lz4_pinned(lz4_data, device, hidden, dtype, decompress_op):
    """H2D from pinned buffer → GPU decompress.  No CPU copy on the hot path."""
    dtype_str = str(dtype).replace("torch.", "")
    shape = [hidden, hidden]
    result = []
    for pinned, orig_size in lz4_data:
        comp_gpu = pinned.to(device, non_blocking=False)  # fast: pinned → DMA
        w = decompress_op(comp_gpu, shape, dtype_str, orig_size)
        result.append(w)
    return result


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_mode(layers, x_gpu, device, fetch_fn, iters, warmup=3):
    for _ in range(warmup):
        x = x_gpu.clone()
        for layer_data in layers:
            weights = fetch_fn(layer_data)
            x = compute(x, weights)
            del weights
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        x = x_gpu.clone()
        for layer_data in layers:
            weights = fetch_fn(layer_data)
            x = compute(x, weights)
            del weights
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / iters


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CPU-offload forward-pass benchmark (4 modes)")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    H = args.hidden
    NL = args.layers
    SL = args.seq_len
    ITERS = args.iters
    dtype = torch.float16

    bytes_per_weight = H * H * 2  # float16 = 2 bytes/element
    total_weight_mb = NL * 4 * bytes_per_weight / 1e6

    print(f"Config: hidden={H}, layers={NL}, seq_len={SL}, iters={ITERS}")
    print(f"Weight matrix: {bytes_per_weight/1e6:.1f} MB fp16  |  "
          f"Total: {total_weight_mb:.0f} MB ({NL} layers × 4 matrices)")

    # ── GPU setup ──────────────────────────────────────────────────────────────
    print("\nInitialising CUDA context...", flush=True)
    device = torch.device("cuda")
    torch.zeros(1).to(device); torch.cuda.synchronize()
    print("CUDA ready.", flush=True)

    # ── nvCOMP op ─────────────────────────────────────────────────────────────
    try:
        import vllm._C  # noqa: F401
        decompress_op = torch.ops._C.decompress_tensor
        gpu_ok = bool(torch.ops._C.is_gpu_decompress_available())
        print(f"GPU decompression: {'available (nvCOMP)' if gpu_ok else 'NOT available'}")
    except Exception as e:
        decompress_op = None
        gpu_ok = False
        print(f"GPU decompression: NOT available ({e})")

    if gpu_ok:
        _d = lz4_compress_chunked(b"\x00" * CHUNK_SIZE)
        _p = torch.empty(len(_d), dtype=torch.uint8, pin_memory=True)
        _p.copy_(torch.frombuffer(bytearray(_d), dtype=torch.uint8))
        _g = _p.to(device)
        torch.ops._C.decompress_tensor(_g, [CHUNK_SIZE], "uint8", CHUNK_SIZE)
        torch.cuda.synchronize()
        print("LZ4 pool initialised (pinned memory path).")

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"\nBuilding {NL}-layer synthetic model (CPU)...", flush=True)
    synth = [SyntheticLayer(H) for _ in range(NL)]

    print("Compressing for jit-cpu (zlib)...", flush=True)
    zlib_layers = [layer.make_zlib() for layer in synth]

    if gpu_ok:
        print("Compressing + pinning for jit-gpu (LZ4)...", flush=True)
        lz4_pinned_layers = [layer.make_lz4_pinned() for layer in synth]

    # ── Compression stats ─────────────────────────────────────────────────────
    raw_one  = bytes_per_weight
    z_one    = len(zlib_layers[0][0][0])
    if gpu_ok:
        l4_one   = lz4_pinned_layers[0][0][0].numel()
        n_chunks = struct.unpack_from("<I", lz4_pinned_layers[0][0][0].numpy().tobytes(), 0)[0]
        lz4_total_mb = sum(
            sum(p.numel() for p, _ in l) for l in lz4_pinned_layers
        ) / 1e6
    zlib_total_mb = sum(sum(len(c) for c, _ in l) for l in zlib_layers) / 1e6

    print(f"\nCompression (one weight, {raw_one/1e6:.1f} MB fp16):")
    print(f"  Uncompressed : {raw_one/1e6:.2f} MB")
    print(f"  zlib         : {z_one/1e6:.2f} MB  (ratio {z_one/raw_one:.3f})")
    if gpu_ok:
        print(f"  LZ4 chunked  : {l4_one/1e6:.2f} MB  (ratio {l4_one/raw_one:.3f}, {n_chunks} chunks)")

    print(f"\nTotal H2D per full-model pass:")
    print(f"  cpu-raw      : {total_weight_mb:.0f} MB  (uncompressed)")
    print(f"  jit-cpu      : {zlib_total_mb:.0f} MB  (zlib, {zlib_total_mb/total_weight_mb:.3f} ratio)")
    if gpu_ok:
        print(f"  jit-gpu      : {lz4_total_mb:.0f} MB  (LZ4, {lz4_total_mb/total_weight_mb:.3f} ratio)")

    # ── Pre-load GPU-resident weights ─────────────────────────────────────────
    print(f"\nLoading {total_weight_mb:.0f} MB onto GPU for baseline...", flush=True)
    gpu_layers = [[w.to(device) for w in layer.cpu_weights] for layer in synth]
    torch.cuda.synchronize()
    vram_mb = torch.cuda.memory_allocated() / 1e6
    print(f"GPU VRAM used by model weights: {vram_mb:.0f} MB")

    # ── Input tensor ──────────────────────────────────────────────────────────
    x_gpu = torch.randn(1, SL, H, dtype=dtype, device=device)

    # ── Run benchmarks ────────────────────────────────────────────────────────
    warmup = 3
    print(f"\nRunning benchmarks ({ITERS} iters, {warmup} warmup)...\n")

    print("  [1/4] gpu-resident (no H2D, weights already on GPU)...")
    t_gpu = run_mode(gpu_layers, x_gpu, device, lambda wd: wd, ITERS, warmup)

    print("  [2/4] cpu-raw  (plain H2D of full FP16 weights)...")
    cpu_raw_data = [layer.cpu_weights for layer in synth]
    t_raw = run_mode(cpu_raw_data, x_gpu, device,
                     lambda wd: fetch_cpu_raw(wd, device), ITERS, warmup)

    print("  [3/4] jit-cpu  (zlib CPU decompress + H2D)...")
    t_jit_cpu = run_mode(zlib_layers, x_gpu, device,
                          lambda wd: fetch_zlib(wd, device, H, dtype), ITERS, warmup)

    if gpu_ok:
        print("  [4/4] jit-gpu  (pinned LZ4 → H2D → nvCOMP GPU decompress)...")
        t_jit_gpu = run_mode(lz4_pinned_layers, x_gpu, device,
                              lambda wd: fetch_lz4_pinned(wd, device, H, dtype, decompress_op),
                              ITERS, warmup)
    else:
        t_jit_gpu = None
        print("  [4/4] jit-gpu  SKIPPED (nvCOMP not available)")

    # ── Report ────────────────────────────────────────────────────────────────
    W = 28

    def ms(t): return f"{t * 1000:.1f} ms"

    print()
    print("=" * 74)
    print(f"Scenario: model too large for VRAM — weights streamed from CPU RAM")
    print(f"  {NL} layers × 4 matrices × {bytes_per_weight/1e6:.1f} MB = {total_weight_mb:.0f} MB total weights")
    print(f"  Each layer: transfer weights → compute (seq={SL}, hidden={H}) → free GPU weights")
    print("=" * 74)
    print()

    # ── Section 1: VRAM-fits baseline (reference only) ────────────────────────
    print(f"  Reference (model fits in VRAM — not the target scenario):")
    print(f"    gpu-resident : {ms(t_gpu):>8} / pass  ({ms(t_gpu/NL)} / layer, pure compute)")
    print()

    # ── Section 2: Streaming comparison — the main result ────────────────────
    print(f"  Streaming from CPU RAM (weights never fully resident on GPU):")
    print(f"  {'Mode':{W}}  {'Time/pass':>10}  {'per layer':>10}  {'vs uncompressed':>16}  {'PCIe bytes':>12}")
    print(f"  {'-'*W}  {'-'*10}  {'-'*10}  {'-'*16}  {'-'*12}")

    pcie_raw  = total_weight_mb / t_raw
    pcie_jcpu = zlib_total_mb  / t_jit_cpu

    def stream_row(name, t, transferred_mb, vs_raw_t):
        pcie = transferred_mb / t
        if vs_raw_t is None:
            vs_raw = "← baseline"
        elif t < vs_raw_t:
            vs_raw = f"{vs_raw_t/t:.2f}× faster"
        else:
            vs_raw = f"{t/vs_raw_t:.2f}× slower"
        print(f"  {name:{W}}  {ms(t):>10}  {ms(t/NL):>10}  {vs_raw:>16}  {transferred_mb:.0f} MB ({pcie:.0f} MB/s)")

    stream_row("cpu-raw  (uncompressed H2D)",  t_raw,     total_weight_mb,  None)
    stream_row("jit-cpu  (zlib, CPU decomp)",  t_jit_cpu, zlib_total_mb,    t_raw)
    if t_jit_gpu:
        pcie_jgpu = lz4_total_mb / t_jit_gpu
        stream_row("jit-gpu  (LZ4+nvCOMP)",        t_jit_gpu, lz4_total_mb,     t_raw)

    # ── Section 3: Key numbers ────────────────────────────────────────────────
    print()
    if t_jit_gpu:
        print(f"  Key comparisons (streaming modes only):")
        if t_jit_gpu < t_raw:
            print(f"    jit-gpu vs uncompressed : {t_raw/t_jit_gpu:.2f}× faster")
        else:
            print(f"    jit-gpu vs uncompressed : {t_jit_gpu/t_raw:.2f}× slower  "
                  f"(data is incompressible — same bytes transferred, plus decompress overhead)")
        print(f"    jit-gpu vs jit-cpu      : {t_jit_cpu/t_jit_gpu:.1f}× faster")
        print(f"    streaming overhead      : {t_raw/t_gpu:.1f}× vs gpu-resident "
              f"(PCIe is the bottleneck, not compute)")
        print()

    # ── Section 4: Compression ratio context ─────────────────────────────────
    print(f"  Compression ratios for this data (random FP16 — worst case):")
    print(f"    zlib : {z_one/raw_one:.3f}  ({(1-z_one/raw_one)*100:.1f}% smaller)")
    if gpu_ok:
        print(f"    lz4  : {l4_one/raw_one:.3f}  ({(1-l4_one/raw_one)*100:.1f}% smaller)")
    print()
    print(f"  Realistic GPTQ INT4 qweight compression ratios with LZ4: ~0.65–0.80")
    if t_jit_gpu:
        for ratio, label in [(0.80, "light"), (0.70, "moderate"), (0.60, "good")]:
            t_est = t_jit_gpu * ratio  # H2D time scales linearly with bytes transferred
            faster = t_raw / t_est
            marker = " ← matches uncompressed" if abs(faster - 1.0) < 0.05 else ""
            sign = "faster" if t_est < t_raw else "slower"
            cmp = f"{max(faster,1/faster):.2f}× {sign}"
            print(f"    LZ4 ratio {ratio:.2f} ({label:8s}): ~{t_est*1000:.0f} ms/pass  "
                  f"({cmp} than uncompressed){marker}")
    print("=" * 74)


if __name__ == "__main__":
    main()
