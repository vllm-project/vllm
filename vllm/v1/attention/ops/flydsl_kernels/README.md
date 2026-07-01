# FlyDSL kernels for TurboQuant (vendored)

This directory contains FlyDSL **kernel sources** vendored into vLLM. The
kernel files are bundled here so a single `git checkout` of vllm-pr is enough
to read and test them. The **FlyDSL framework** itself (the MLIR compiler
stack that compiles these kernels at runtime) is a separate dependency that
must be installed once per machine.

## What's vendored

- `tq_decode_v4.py` — TurboQuant 4-bit MSE-key decode attention kernel for
  AMD MI355X (gfx950 / CDNA4). Implements:
    - LDS-resident centroids (one-time DMA in CTA prologue)
    - `buffer_load_to_lds` cross-tile prefetch with LDS ping-pong
    - `mfma_f32_16x16x32_bf16` (CDNA4 wide-K) for QK and PV
    - `ds_read_tr16_b64` hardware V-transpose (with cross-lane LDS race fence)
    - Flash-Attention-2 style online softmax with split-K reduction

- `fp8_g32_decode_v4.py` — fp8_g32 (FP4 E2M1 codes + UE8M0 per-group-of-32
  scales) decode attention kernel. A direct port of `tq_decode_v4.py`: the
  MFMA layouts, online-softmax loop, HW/SW V-transpose, output store and
  split-K reducer are reused verbatim. Only the dequant + cache addressing
  differ:
    - AoS cache slot `[num_blocks, BS, Hk, padded_slot]` (K codes@0, K
      scales@64, V codes@68, V scales@132 for D=128/group=32)
    - fixed FP4 value table reused via the LDS centroid LUT
    - K/V dequant = `FP4_value[nibble] * 2^(scale_byte-127)` where the pow2
      scale is `bitcast_f32(byte << 23)` (no `exp2`); each lane owns exactly
      one group so it loads one K-scale + one V-scale byte
    - Q is Hadamard-rotated + FP8-E4M3-haircut in the launcher, fed bf16
  Launcher: `vllm/v1/attention/ops/flydsl_fp8_g32_decode_v4.py`. Enable with
  `VLLM_FP8_G32_DECODE_V4=1` (eligible: HEAD_SIZE=128, GQA {8,16}, no
  sinks/SWA). Parity test: `tests/kernels/turboquant_v4/test_fp8_g32_v4_parity.py`.

The canonical upstream copies live at `<FlyDSL repo>/kernels/tq_decode_v4.py`
and `<FlyDSL repo>/kernels/fp8_g32_decode_v4.py`.
This directory is updated by re-copying that file when a new tested snapshot
is ready. **Do not edit the vendored copy in-place** — make changes in the
FlyDSL repo, validate, then re-vendor.

## Requirements

- AMD MI355X (gfx950 / CDNA4) — kernel uses CDNA4-only intrinsics
- ROCm 7.x with `/opt/rocm/bin/hipcc`
- Python 3.12 (FlyDSL build is Python-version-pinned)
- Linux x86_64 with glibc ≥ 2.35

## Step 1 — Install the FlyDSL framework (one-time per machine)

The kernel above is just the source; you also need the FlyDSL compiler stack
to JIT-compile it. Build it once:

```bash
git clone <flydsl-repo-url> /opt/FlyDSL
cd /opt/FlyDSL
git checkout 41500b0    # tested SHA — newer commits may also work
mkdir -p build-fly && cd build-fly
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON
ninja -j$(nproc)
```

Build takes ~5 minutes on a typical workstation. The output you need lives at
`/opt/FlyDSL/build-fly/python_packages/flydsl/`.

## Step 2 — Point vLLM at the FlyDSL install

```bash
export VLLM_FLYDSL_ROOT=/opt/FlyDSL
export VLLM_FLYDSL_PKGS=/opt/FlyDSL/build-fly/python_packages
```

Verify FlyDSL is importable:

```bash
python3 -c "import sys; sys.path.insert(0, '$VLLM_FLYDSL_PKGS'); import flydsl; print('FlyDSL OK')"
```

## Step 3 — Enable the v4 decode kernel

```bash
export VLLM_ROCM_TQ_FLYDSL_DECODE=1            # master enable
export VLLM_TQ_FLYDSL_HW_TR=1      # use HW V-transpose (default; safe post-fence-fix)
```

Eligible TQ layers (`HEAD_SIZE=128`, GQA in {8, 16}, `MSE_BITS=4`, no sinks,
no FP8) will route to `tq_decode_v4.py`. If the framework is missing or the
layer is ineligible, vLLM logs a warning and falls back to Triton v3.

Confirmation appears in the server log on startup:

```
TurboQuant has flash attn: True, decode kernel: v4(flydsl)
FlyDSL v4 launcher invoked: B=... Hk=... ... hw_v_transpose=True
```

If you see this instead, FlyDSL was not found:

```
WARNING: VLLM_ROCM_TQ_FLYDSL_DECODE requested but FlyDSL is unavailable; falling back to v3.
```

## Full launch example (Qwen2.5-72B, MI355X TP=4)

```bash
VLLM_ROCM_TQ_FLYDSL_DECODE=1 \
VLLM_TQ_FLYDSL_HW_TR=1 \
VLLM_FLYDSL_ROOT=/opt/FlyDSL \
VLLM_FLYDSL_PKGS=/opt/FlyDSL/build-fly/python_packages \
HSA_NO_SCRATCH_RECLAIM=1 \
vllm serve Qwen/Qwen2.5-72B \
    --tensor-parallel-size 4 \
    --kv-cache-dtype turboquant_4bit_nc \
    --attention-backend ROCM_AITER_UNIFIED_ATTN \
    --block-size 32 \
    --gpu-memory-utilization 0.88 \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

## Tests

Unit / regression tests for the kernel live alongside vLLM's other kernel
tests at `tests/kernels/turboquant_v4/`:

```bash
pytest tests/kernels/turboquant_v4/ -v
```

Three tests are included:

| Test | What it checks |
|---|---|
| `test_v4_vs_v3_fast.py` | Per-element diff vs Triton v3 reference (max ≤ 1.95e-3) |
| `test_v4_hwtr_long_seq_regression.py` | Bit-exact HW vs SW V-transpose on Qwen3-32B production shape (post-fence-fix regression guard) |
| `test_v4_norm_correction_equivalence.py` | `norm_correction={True, False}` toggle equivalence |

## Validated results

Reference numbers from the development team's MI355X (gfx950) box, ROCm 7.x,
TP=4:

**GSM8K accuracy (Qwen3-32B):**

| Configuration | Accuracy | vs BF16 |
|---|---:|---:|
| BF16 baseline | 0.6892 | — |
| TQ Triton v3 | 0.7074 | +1.8 pp |
| **TQ FlyDSL v4 (HW transpose ON, post-fence-fix)** | **0.6907** | **+0.1 pp** |

**Throughput (Qwen2.5-72B, 32K prompt / 1K out, N=80):**

| Configuration | Output tok/s | vs BF16 | vs v3 |
|---|---:|---:|---:|
| BF16 baseline | 304 | — | +39% |
| TQ Triton v3 | 219 | −28% | — |
| **TQ FlyDSL v4** | **332.9** | **+9.6%** | **+52.0%** |

## Why isn't the FlyDSL framework vendored?

FlyDSL ships as ~330 MB of compiled MLIR shared libraries pinned to a specific
Python + glibc + ROCm combination. Vendoring would balloon the repo and
silently break on slightly different host stacks. A 5-minute one-time source
build is more portable.
