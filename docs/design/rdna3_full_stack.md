# RDNA3 Full Stack Acceleration for vLLM

> _"Every instruction counts when you're chasing the memory wall."_

This document describes the complete performance stack for AMD RDNA3 (gfx1100)
in vLLM. Three independent acceleration layers that compose multiplicatively —
each targeting a different bottleneck in the inference pipeline.

---

## The Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                        vLLM Inference Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────┐   ┌──────────────────┐   ┌───────────────────┐  │
│  │  Layer 1:     │   │  Layer 2:        │   │  Layer 3:         │  │
│  │  W4A16 WMMA   │──▶│  Triton Prefill  │──▶│  HIP INT8/INT4    │  │
│  │  (GEMM)       │   │  (Attention FP16)│   │  (Attention Quant) │  │
│  └───────────────┘   └──────────────────┘   └───────────────────┘  │
│       ▲                      ▲                      ▲               │
│       │                      │                      │               │
│  Weights are the        Attention is the      KV cache is the      │
│  bottleneck at          bottleneck at         bottleneck at        │
│  short context          medium context        long context         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Native W4A16 WMMA Kernel

**What**: Hand-written HIP kernel for 4-bit weight × 16-bit activation GEMM,
targeting `v_wmma_f32_16x16x16_bf16_w32` on gfx1100.

**Why**: Triton's W4A16 kernel on RDNA3 leaves 60% of WMMA throughput on the
table due to suboptimal dequant scheduling and wave occupancy.

**Result**: Saturates the gfx1100 WMMA unit. Scalar path + WMMA v2 (2-wave
cooperative with double-buffered LDS) hits the compute ceiling.

**Files**: [`csrc/libtorch_stable/quantization/gptq/README_RDNA3.md`](../../csrc/libtorch_stable/quantization/gptq/README_RDNA3.md)

---

## Layer 2: Triton Prefill Tuning (3-Tier Adaptive)

**What**: Launch parameter overrides for `triton_unified_attention` on gfx11.
Adapts BLOCK_M and num_warps to sequence length.

**Why**: Triton defaults assume wave64 (CDNA). On wave32 RDNA3, the default
BLOCK_M=16 with 4 warps gives pathetically low arithmetic intensity (~4
FLOPs/byte) — deeply memory-bound even when it shouldn't be.

**Result**: Up to **3.7× faster** FP16 prefill by matching tile size to the
memory/compute crossover point.

| Sequence length | Tile config | Speedup vs default |
|----------------|-------------|-------------------|
| ≤ 1024 | M32, 2 warps | 1.7× |
| 1025 – 8192 | M64, 4 warps | 2.3× |
| > 8192 | M128, 8 warps | 2.2× |

**Files**: [`vllm/v1/attention/ops/README_RDNA3_TRITON_PREFILL_TUNING.md`](../../vllm/v1/attention/ops/README_RDNA3_TRITON_PREFILL_TUNING.md)

---

## Layer 3: HIP INT8/INT4 Prefill Kernels

**What**: Native HIP attention kernels for quantized KV cache. Bypass Triton
entirely with cooperative 4-wave WMMA and fused dequant.

**Why**: On gfx1100, INT8→FP16 is 1 native instruction (`v_cvt_f16_i16_e32`).
FP8→FP16 is ~20 emulated instructions. INT4 nibble unpack is 2 instructions.
Combined with 50-75% bandwidth reduction from quantized cache, these kernels
dominate at long context.

**Result**:

| Mode | vs FP16 Triton | VRAM saving | ISA used |
|------|---------------|-------------|----------|
| INT8 per-token-head | **+66%** | 50% | `v_wmma_f32_16x16x16_bf16` |
| INT4 per-token-head | **+58%** | 75% | `v_wmma_i32_16x16x16_iu8` |

Both are **8× faster** than the equivalent Triton quantized attention path.

**Files**: [`csrc/attention/README_RDNA3_HIP_KERNELS.md`](../../csrc/attention/README_RDNA3_HIP_KERNELS.md)

---

## Benchmark Results

### Test Setup

- **Model**: Qwen3.6-27B GPTQ W4A16, TP2
- **Hardware**: 2× RX 7900 XTX (48 GB total VRAM)
- **Config**: chunked prefill, max_num_batched_tokens=2048, enforce_eager

### Kernel-Level (Attention Microbenchmark, ctx=8000 ql=512)

| Kernel | Time | vs Triton baseline |
|--------|------|--------------------|
| Triton INT8 per-token-head | ~25 ms | — |
| **HIP INT8** | **3.03 ms** | **8.3×** |
| **HIP INT4** | **3.05 ms** | **8.2×** |
| Triton FP16 (default M16w4) | 11.2 ms | — |
| Triton FP16 (tuned M64w4) | **3.0 ms** | **3.7×** |

### End-to-End TTFT (Single Request, Prefill → First Token)

| Prompt length | Baseline (FP16) | INT8 full stack | INT4 full stack |
|--------------|-----------------|-----------------|-----------------|
| ~2K tokens | 2.81s | 2.74s (-2.5%) | 2.78s (-1%) |
| ~8K tokens | 11.15s | 10.99s (-1.4%) | 11.25s (+1%) |
| ~16K tokens | 23.74s | 22.52s (-5.1%) | 22.96s (-3.3%) |
| ~24K tokens | 38.44s | **34.84s (-9.4%)** | **35.49s (-7.7%)** |

Note: GEMM (W4A16 linear layers) dominates ~80% of total inference time.
Attention is ~20%. The 8× kernel speedup on 20% of runtime yields ~10% E2E.
The longer the context, the more attention dominates → larger E2E gains.

### KV Cache Capacity (The Killer Metric)

With identical VRAM budget (33.3 GB available for KV after model load):

| KV dtype | Max context | Concurrent @8K | Concurrent @32K |
|----------|-------------|----------------|-----------------|
| FP16 | 136K tokens | 17 requests | 4 requests |
| INT8 | 273K tokens | **34 requests (2×)** | **8 requests (2×)** |
| INT4 | **545K tokens** | **68 requests (4×)** | **17 requests (4×)** |

**INT4 serves 4× more concurrent users with the same hardware.**
Or equivalently: supports 4× longer context (545K vs 136K tokens).

### Throughput Under Load (8 Concurrent Requests, ~2K prompt + 50 gen)

| Config | Total throughput |
|--------|-----------------|
| Baseline FP16 | 673 tok/s |
| INT8 full stack | 672 tok/s |
| INT4 full stack | 655 tok/s |

At low concurrency (8 reqs), throughput is GEMM-bound and similar. The real
throughput advantage appears at high concurrency (34-68 reqs) where FP16
would OOM but INT8/INT4 can still serve.

### Quality

| Mode | Cosine vs FP16 reference | Notes |
|------|--------------------------|-------|
| INT8 per-token-head | 1.000000 | Exact (symmetric, no precision loss) |
| INT4 per-token-head | 0.999995 | Q→int8 quantization noise (~0.001) |

Both produce coherent outputs at temperature=0 on code generation,
mathematical reasoning, translation, and factual QA.

---

### Updated sweep: prefill + decode tok/s, 1K → 128K

Apples-to-apples sweep across all KV cache modes on the same hardware
session. CUDA graphs enabled, batch=1.

#### Prefill tok/s

| Seqlen | FP16¹ | FP8¹ | INT8 per-tensor (Triton, removed)¹ | INT8 per-token-head (HIP) | INT4 per-token-head (HIP) |
|--------|-------|------|------------------------------------|---------------------------|---------------------------|
| 1K     | 1405  | 1420 | 1377                               | **1462**                  | 1461                      |
| 4K     | 1381  | 1401 | 1331                               | 1433                      | **1450**                  |
| 8K     | 1343  | 1340 | 1233                               | 1373                      | **1388**                  |
| 16K    | 1277  | 1291 | 1112                               | 1322                      | **1333**                  |
| 32K    | 1180  | 1204 | 929                                | 1233                      | **1240**                  |
| 64K    | —     | —    | —                                  | **1101**                  | 1095                      |
| 128K   | —     | —    | —                                  | **868**                   | 864                       |

#### Decode tok/s (128 output tokens)

| Seqlen | INT8 per-tensor (Triton, removed)¹ | INT8 per-token-head (split-KV) | INT4 per-token-head (split-KV) |
|--------|-----------------------------------|--------------------------------|--------------------------------|
| 1K     | 51.8                              | **43.3**                       | 43.5                           |
| 4K     | 49.7                              | 25.2                           | **25.3**                       |
| 8K     | 45.9                              | 16.2                           | **16.3**                       |
| 16K    | 43.6                              | **9.5**                        | 9.5                            |
| 32K    | 37.1                              | **5.2**                        | 5.2                            |
| 64K    | —                                 | **2.7**                        | 2.7                            |
| 128K   | —                                 | **1.4**                        | 1.4                            |

¹ FP16 / FP8 / per-tensor columns: prior-session data, same hardware.

### Key findings

1. **All HIP kernels converge to ±1% in prefill** — the K-loop is HBM-bound,
   not LDS-bound. Scale-array overhead (per-token-head) and scale folding
   (per-tensor) are both latency-hidden behind the ~21 global loads per
   K-loop iteration. INT4 / INT8 / FP8 all run at the same speed.

2. **INT8 per-token-head wins prefill at long context** — beats per-tensor
   Triton by +12 % at 16K, +33 % at 32K. The HIP kernel's cooperative
   4-wave K/V loads + native int8→fp16 cast (1 instr) eliminate the
   Triton path's tile heuristic and register pressure issues.

3. **Decode is weight-bound, not KV-bound** — at 128K context, KV cache
   reads are ~3.8 GB/step (INT8) or ~1.9 GB/step (INT4), versus 6.75 GB
   for the W4A16 model weights. Per the breakdown, KV cache is <1 % of
   the 714 ms/token decode time. INT4 ≡ INT8 in decode throughput
   because the difference is invisible behind weight bandwidth.

4. **Per-token-head split-KV wins decode** — dedicated kernel that splits
   each query across `NUM_KV_SPLITS` segments and reduces partial outputs.
   M=1 doesn't saturate the 96 CUs on gfx1100 without splitting.
   Per-tensor decode (now removed) had no equivalent path and lost
   5–12 % at long contexts.

5. **INT4 vs INT8: choose by VRAM, not speed** — INT4 saves 75 % VRAM
   (vs 50 % for INT8) but offers no measured speed advantage. Use INT4
   only when you need the extra context length or concurrency headroom.

Layer 1 (W4A16 WMMA) improves decode and short-prompt latency where GEMM
dominates. Layer 2+3 improve prefill where attention dominates.

---

## ISA Foundation

All optimizations are grounded in verified gfx1100 ISA analysis:

| Instruction | What | Cycles |
|---|---|---|
| `v_wmma_f32_16x16x16_bf16_w32` | 16×16 bf16 matmul | 16 |
| `v_wmma_i32_16x16x16_iu8_w32` | 16×16 int8 matmul | 16 |
| `v_cvt_f16_i16_e32` | INT8 → FP16 | 1 |
| `v_cvt_f16_u16_e32` | UINT8 → FP16 | 1 |
| `v_dot2_f32_bf16` | 2× bf16 FMA (scalar path) | 1* |

*half-rate on gfx1100 (measured 2.01× vs full-rate expectation).

Full ISA reference: [`csrc/libtorch_stable/quantization/gptq/README_RDNA3_FULL_ISA.md`](../../csrc/libtorch_stable/quantization/gptq/README_RDNA3_FULL_ISA.md)
Conversion cost analysis: [`csrc/libtorch_stable/quantization/gptq/README_RDNA3_CVT_ISA.md`](../../csrc/libtorch_stable/quantization/gptq/README_RDNA3_CVT_ISA.md)

---

## Usage

```bash
# Full stack: W4A16 model + INT4 KV cache + all tuning active
vllm serve Qwen3.6-27B-GPTQ-W4A16 \
  --kv-cache-dtype int4_per_token_head \
  --dtype float16 \
  --enforce-eager \
  --enable-chunked-prefill \
  --tensor-parallel-size 2

# Or INT8 for simpler setup (no RHT rotation needed)
vllm serve <model> --kv-cache-dtype int8_per_token_head
```

The W4A16 WMMA kernel auto-dispatches when a GPTQ model is loaded on ROCm
gfx1100. Triton tuning activates automatically on gfx11. HIP attention kernels
dispatch when the compiled op is available and conditions are met.

---

## Build Requirements

- ROCm 6.x+ with gfx1100 target
- `PYTORCH_ROCM_ARCH=gfx1100` during build
- Python 3.12+, PyTorch 2.4+

```bash
MAX_JOBS=$(nproc) PYTORCH_ROCM_ARCH=gfx1100 python3 setup.py build_ext --inplace
```

---

## Architecture Decisions

1. **Separate TUs per kernel variant** — hipcc optimizer interactions between
   templates cause VGPR inflation. Each kernel gets its own `.cu` file.

2. **Centering zero-point in loader** (INT4) — subtracting zp during the
   cooperative load (16 integer subs) eliminates ~96 VALU/tile from the inner
   loop. Makes INT4 within 1-8% of INT8 despite asymmetric quantization.

3. **Phase 1/Phase 2 split** — cached prefix reads quantized paged cache;
   current chunk reads raw fp16 (not yet quantized). Avoids double-quantize
   and keeps the critical path clean.

4. **3-tier prefill over fixed tuning** — no single BLOCK_M is optimal across
   all sequence lengths. The crossover from memory-bound to compute-bound
   happens at ~1024 tokens on gfx1100.

5. **INT8 WMMA for INT4 Q×K** — `v_wmma_i32_16x16x16_iu8` operates directly
   on centered int8 K values. Zero conversion needed. Q quantized to int8
   once (amortized over all K tiles).

---

## Custom All-Reduce over PCIe P2P

`e59a122a91` enables the custom all-reduce on gfx11 and `f1be45cd60` keeps
`register_graph_buffers()` unconditional.
`RocmPlatform.use_custom_allreduce_graph_registration()` returns `False` on
gfx11, so captured collectives copy into the buffer registered at init instead
of writing through the graph's own pointers.

That only holds when the driver's peer access is backed by a root complex that
actually supports P2PDMA. Measured on two boxes with the same GPUs (RX 7900
XTX, gfx1100) and the same model artifact:

| box | peer access | TP | custom AR | output |
| --- | --- | --- | --- | --- |
| EPYC 7252 | native | 4 | on | correct — 2 days serving |
| i9-13900K / Z790 | only after force-adding `8086:a700` to the kernel P2PDMA allowlist | 2 | on | garbage |

On the second box, with custom AR enabled and a sanity check in the same engine
start:

| algorithm | sane completions |
| --- | --- |
| 1stage — the only one the C++ picks at `world_size == 2` | 2/10 |
| 2stage — `VLLM_CUSTOM_ALLREDUCE_ALGO=2stage` | 0/10 |
| `--disable-custom-all-reduce` | 9/9 |

Refuted as causes, each by measurement: the `max_size` cap (8 MiB and 96 MiB
both corrupt), `dtype` (fp16 and bf16), `trust_remote_code`, the model artifact
(two different checkpoints), and the graph-registration bug above — the gfx11
hook already forces `registered=False`. The build is refuted too: the failing
box corrupts identically on `v0.26.1rc1.dev333` (July image) and on
`v0.28.1rc1.dev667`, a build of this branch that is _newer_ than the one
serving correctly on the EPYC box.

Two firmware knobs on the failing box were tried and refuted as well, both
individually and together, each verified in the register and in `lspci` before
the run:

| knob | root ports | result |
| --- | --- | --- |
| ACS P2P Request/Completion Redirect cleared (`ACSCtl` `0x1d` -> `0x11`) | both | garbage |
| AtomicOp Egress Blocking cleared (`DevCtl2` bit 7, set only on one port) | `00:06.0` | garbage |

Neither is surprising on reflection: they are permission bits, not ordering
guarantees. They say whether a transaction may pass, not that it arrives in
order carrying current data.

What is left is the link itself, and it can be measured directly. A HIP probe
that allocates on both GPUs and drives the same bytes in each direction, with
every launch error-checked and every buffer verified locally first:

| operation | 0 -> 1 | 1 -> 0 |
| --- | --- | --- |
| local fill, read back on the same GPU (control) | ok | ok |
| kernel WRITES into the peer's buffer | ok | ok |
| kernel READS the peer's buffer | all zeros | all zeros |
| `hipMemcpyPeer` scheduled on the **source** stream (push) | ok | ok |
| `hipMemcpyPeer` scheduled on the **destination** stream (pull) | all zeros | all zeros |

Symmetric, and identical for cached and uncached allocations. This root complex
forwards peer writes and answers peer reads with zeros — not with an error, so
nothing is logged and no PCIe error counter moves. Forcing `8086:a700` into
`pci_p2pdma_whitelist` decides only whether Linux sets the mapping up; it
cannot make the hardware route read completions between root ports.

That is the whole corruption. `cross_device_reduce_1stage` is a pull:
`packed_reduce` walks `dp.ptrs[i][idx]` across every peer. Those loads return
zeros, so each rank reduces its own contribution against zero. The barrier is
unaffected because it only _writes_ to the peer
(`__scoped_atomic_store_n(..., __MEMORY_SCOPE_SYSTEM)` on the ROCm branch, then
a device-scope spin on memory the peer wrote), which is why the engine neither
hangs nor produces NaN — it produces confident, wrong numbers. 2stage corrupts
for the same reason: it also reads.

It also explains the false green. `rocm-bandwidth-test -a -v` reports PASS
between the two GPUs because HSA schedules the async copy on the **source**
agent, so the benchmark only ever exercises a push. Asked for one direction
with `-s 1 -d 2` it validates the `[1][1]` diagonal — a device copying to
itself — and prints `N/A` for the cell that was the point of the run.

### The fix: push instead of pull

A push all-reduce works on this hardware. Each rank writes its contribution
into a slot the peer owns, then reduces reading only its own memory. Measured
against RCCL on the same box, per all-reduce:

| payload | RCCL | push | |
| --- | --- | --- | --- |
| 16 KiB | 51.5 us | 8.4 us | 6.1x |
| 64 KiB | 70.0 us | 20.8 us | 3.4x |
| 256 KiB | 151.4 us | 93.1 us | 1.6x |
| 1 MiB | 507.5 us | 375.9 us | 1.35x |
| 4 MiB | 1946.6 us | 1572.0 us | 1.24x |

The inbox has to be uncached. With a cacheable peer mapping the writes are
absorbed by the _writer's_ L2 and never reach the wire; at 4 MiB that test
still passed, which is a size-dependent false green that only the
deliberately-wrong control caught.

vLLM already ships a push collective: QuickReduce writes into every peer's
buffer in phase 1A and reduces out of `buffer_list[rank]`. It was gated to
gfx94/gfx95, and it returns wrong results on RDNA3 for an unrelated reason —
`BufferResource` builds word 3 of the buffer descriptor as `0x00020000`, the
gfx9 encoding. On gfx1100 that makes `buffer_load_dwordx4` return zeros:

| word 3 | result on gfx1100 |
| --- | --- |
| `0x00020000` (what QuickReduce used) | 4095/4096 zeros |
| `0x31014000` (RDNA gfx10+) | ok |

With the descriptor selected per architecture, QuickReduce is bit-exact against
RCCL at every size tested, in fp16 and bf16, and all four codecs (FP, INT8,
INT6, INT4) compute correctly — their errors grow in the expected order,
`4e-3 / 1.8e-2 / 7e-2`. End to end on the 27B at TP2, six distinct prompts,
streaming, TTFT kept separate:

| all-reduce | decode (median) | TTFT |
| --- | --- | --- |
| PYNCCL (`--disable-custom-all-reduce`) | 46.26 tok/s | 148 ms |
| QuickReduce, FP codec | **49.23 tok/s** | 139 ms |
| QuickReduce, INT8 codec | 42.50 tok/s | 130 ms |

FP is the one to use: quantizing the payload costs more than the link saves at
decode sizes, even on a 3.5 GB/s link. The pull collective stays off on gfx11.

Three traps worth writing down:

1. **Nothing validates P2P on ROCm.** `CustomAllreduce.__init__` skips the
   functional test — `if not current_platform.is_rocm() and not _can_p2p(...)`
   — and the `fully_connected` gate only applies when `world_size > 2`. At TP2
   on gfx11 the collective is enabled purely on the driver's
   `can_device_access_peer` flag, with nothing behind it. The check that would
   have caught this is a peer _read_, which is what the collective needs and
   what nothing tests.

2. **A bulk D2D content check does not cover this.** `rocm-bandwidth-test`
   passes on the failing box because it pushes. A control that cannot fail
   reads exactly like a control that passed.

3. **Throughput alone will not catch it.** The corrupted engine benchmarks
   _faster_: raising `max_size` lets the prefill collectives through instead of
   falling back to PYNCCL, so the speedup and the corruption arrive together.
   Any measurement of a collective on this hardware has to include an output
   sanity check in the same engine start, and the checker has to be proven
   against a known-bad sample first. A detector that greps for `!!!!` calls the
   generic garbage here clean, and one built on word fractions calls
   `ductductduct...` clean too.

## Branch Structure

### Independent PR branches (each targets `main`)

```
main
├── perf/rdna3_w4a16_squashed              Layer 1 — W4A16 WMMA GEMM kernel
├── perf/rdna3_triton_prefill_tuning       Layer 2 — Triton prefill 3-tier adaptive
│
├── refactor/prefill-fastpath-per-token-head-v2    (upstream dependency, NOT in main)
│   └── feat/rdna3_int8_int4_hip_kernels           Layer 3 — HIP INT8/INT4 prefill kernels
│
└── perf/rdna3_full_stack                  Integration branch (all layers, local testing only)
```

### PR dependency chain

| Branch | PR target | Blocker |
|--------|-----------|---------|
| `perf/rdna3_w4a16_squashed` | `main` | None |
| `perf/rdna3_triton_prefill_tuning` | `main` | None |
| `refactor/prefill-fastpath-per-token-head-v2` | `main` | None (upstream) |
| `feat/rdna3_int8_int4_hip_kernels` | `main` | Requires `refactor/prefill-fastpath-pth-v2` merged first |
| `perf/rdna3_full_stack` | — | Not for PR; integration branch for E2E testing |

> **Note:** `int8_per_tensor` (formerly Layer 4) was evaluated and removed.
> Benchmarks showed it tied per-token-head HIP in prefill (HBM-bound, scale
> overhead latency-hidden) and lost in decode (no dedicated split-KV kernel).
> Per-token-head wins on all axes — use `int8_per_token_head` exclusively.

### Rebuilding the integration branch (`perf/rdna3_full_stack`)

The integration branch can be reconstructed from `main` plus the five
dependency branches.  Merge order matters — later merges depend on context
introduced by earlier ones.

```bash
git checkout -b perf/rdna3_full_stack_rebuild main

# Layer 1 — clean merge
git merge perf/rdna3_w4a16_squashed

# Layer 2 — CONFLICT in triton_unified_attention.py (see §Conflict 1)
git merge perf/rdna3_triton_prefill_tuning

# Upstream dep for Layer 3 — clean merge
git merge refactor/prefill-fastpath-per-token-head-v2

# Layer 3 — CONFLICT in csrc/ops.h, csrc/torch_bindings.cpp (see §Conflict 2)
git merge feat/rdna3_int8_int4_hip_kernels

# Own commits (docs, multi HEAD_SIZE) — cherry-pick from old full_stack
git cherry-pick <docs-commits> <multi-headsize-commit>
```

### Known merge conflicts and resolutions

These conflicts are **expected and unavoidable**.  Each PR branch targets
`main` independently — they cannot be pre-aligned to each other without
rewriting pushed history, and doing so would break them against `main`
(where the other branches don't exist yet).

#### Conflict 1: Layer 1 × Layer 2 — `triton_unified_attention.py`

**File**: `vllm/v1/attention/ops/triton_unified_attention.py` (2 hunks)

**Cause**: Layer 1 (`w4a16_squashed`) adds a simple RDNA3 prefill override
(`BLOCK_M=32`, `num_warps=2`).  Layer 2 (`triton_prefill_tuning`) replaces
that same block with the full 3-tier adaptive logic (M32/M64/M128 with
2/4/8 warps).  Both insert at the same location after `BLOCK_M = 16 if ...`.

**Resolution**: Take Layer 1's simple `BLOCK_M=32, num_warps=2` for both
hunks. Validated empirically post-V7 WMMA: 27B prefill at 8192 tokens
(8× the 3-tier "long" threshold) runs in 14.483 s and beats Hybrid 1.93×
— attention is _not_ the bottleneck at long context on gfx1100; the V7
W4A16 GEMM kernel is. The 3-tier preventive scaling (M64/M128 + 4/8 warps
for "compute-bound long") was designed pre-V7 when the balance was
different. M32+2warps scales fine because once GEMM saturates the WMMA
unit, larger attention tiles only add register pressure for no payoff.

#### Conflict 2: Layer 3 — `csrc/ops.h`, `csrc/torch_bindings.cpp`

**Files**: `csrc/ops.h`, `csrc/torch_bindings.cpp` (1 hunk each)

**Cause**: Layer 1 adds GPTQ RDNA3 function declarations after
`gptq_shuffle`.  Layer 3 adds HIP INT8/INT4 attention declarations at
the same insertion point.  Git can't merge two independent additions at
the same location.

**Resolution**: Keep both blocks.  Order doesn't matter — they are
independent function declarations.  Convention: GPTQ ops first, then
attention ops.

### Why the branches can't be conflict-free

Each PR branch is designed to merge cleanly into `main` on its own.
They share no common ancestor beyond `main` itself, yet they touch
overlapping files:

- `triton_unified_attention.py` is modified by Layers 1 and 2
- `csrc/ops.h` / `torch_bindings.cpp` are modified by Layers 1 and 3
- `triton_attn.py` is modified by the upstream refactor

Making them conflict-free against each other would require either:

1. **Rewriting history** on pushed branches — breaks collaboration
2. **Pre-aligning to unreleased code** — breaks the branch against `main`
3. **Merging in a fixed order** and rebasing later branches — creates
   artificial dependencies between independent features

Option 3 is what `perf/rdna3_full_stack` effectively does as an
integration branch.  The individual PR branches stay clean against
`main`, and the documented conflict resolutions above make reconstruction
deterministic.
