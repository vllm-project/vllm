# QuickReduce on gfx1100, and why the other collective corrupts

The design writeup lives in `docs/design/rdna3_full_stack.md`, section
*Custom All-Reduce over PCIe P2P*. This directory holds the probes that
produced it, the rebuilt kernel, and the steps to put it back.

## One paragraph

On a desktop root complex (i9-13900K / Z790, the two cards on **different**
root ports) PCIe peer **writes** work and peer **reads return zeros** — not an
error, so nothing is logged. vLLM's `CustomAllreduce` is a pull: every rank
reads every peer's buffer. Those reads come back zero, each rank reduces its
own contribution against zero, and the model emits confident garbage without
hanging or producing NaN. QuickReduce pushes instead and works — once its
buffer descriptor is built with the RDNA encoding.

## The two source fixes (already in this branch)

1. `csrc/quickreduce/base.h` — word 3 of the buffer resource descriptor is
   `0x00020000` (gfx9). RDNA replaced those fields; on gfx1100 that constant
   makes every `buffer_load_dwordx4` return zero. Now selected per
   architecture: `0x31014000` on gfx10+.
2. `vllm/distributed/device_communicators/` — `quick_all_reduce.py` admits
   `gfx11`; `custom_all_reduce.py` refuses to run there.

Fix 1 needs a rebuild. Fix 2 is pure Python.

## Numbers (27B, TP2, six distinct prompts, streaming, TTFT separate)

| all-reduce | decode (median) | TTFT |
| --- | --- | --- |
| PYNCCL (`--disable-custom-all-reduce`) | 46.26 tok/s | 148 ms |
| QuickReduce, FP codec | **49.23 tok/s** | 139 ms |
| QuickReduce, INT8 codec | 42.50 tok/s | 130 ms |

Use **FP**. Quantizing the all-reduce payload costs more than it saves at
decode sizes, even on a 3.5 GB/s link, and FP is bit-exact against RCCL.

## Serving flags that are NOT optional on this box

`--enable-prefix-caching` and `--kv-cache-dtype int8_per_token_head` are
required to reach 400k of context with this model. Do not drop them when
copying a launch line from a benchmark.

Then, for QuickReduce:

```text
VLLM_QR_RDNA3_LIB=/tmp/qr_build/libqr_rdna3.so   # only for the live patch, see below
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP
VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB=0
VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=16
VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0
```

`MAX_SIZE` matters: the default is 2 GiB, and QuickReduce allocates `2 *
max_size` up front, which starves the engine before it can size its KV cache.
`MIN_SIZE=0` keeps every collective on the push path. `CAST_BF16_TO_FP16=0`
keeps the model's numerics in bf16.

## ⚠️ The live patch dies with the container

Until an image is built from this branch, the running box uses a **rebuilt
kernel injected at runtime**, not the one inside the image. Recreating the
container silently reverts to the in-image QuickReduce, which returns zeros —
and the failure looks like a bad model, not a bad deploy. To redo it:

```bash
docker exec vllm-vllm1-1 mkdir -p /tmp/qr_build
docker cp libqr_rdna3.gfx1100.torch2.11.0-d0c8b1f.rocm7.2.3.so vllm-vllm1-1:/tmp/qr_build/libqr_rdna3.so
docker cp live_patch.py vllm-vllm1-1:/tmp/qr_build/
docker exec vllm-vllm1-1 python /tmp/qr_build/live_patch.py   # backs up to *.p2pbak
```

`live_patch.py` loads the rebuilt library, routes the `qr_*` ops to it, admits
gfx11, and disables the pull collective. Reverting is `cp *.p2pbak` back.

The `.so` here is untracked on purpose: it is only valid for the exact torch
and ROCm it was built against, which is why its name carries them. If those
move, rebuild rather than reuse.

To rebuild the library from source (needs the patched `csrc/quickreduce/`):

```bash
python -c "from torch.utils.hipify.hipify_python import hipify; \
  hipify(project_directory='.', output_directory='.', includes=['*'], \
         is_pytorch_extension=True, show_detailed=False)"
T=/usr/local/lib/python3.12/dist-packages/torch
hipcc -O3 -std=c++17 -fPIC -shared -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 \
  -D_GLIBCXX_USE_CXX11_ABI=1 --offload-arch=gfx1100 \
  -I. -I$T/include -I$T/include/torch/csrc/api/include \
  qr_rdna3.hip -o libqr_rdna3.so \
  -L$T/lib -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip
```

## probes/

Each is self-checking: it also runs a case that must fail, so a probe that
cannot fail is visible as one that never fails.

| file | question it answers |
| --- | --- |
| `dir.hip` | do peer reads work? do peer writes? both directions, cached and uncached |
| `who.hip` | is it the direction, or who moves the data? push vs pull for the same copy |
| `bufres.hip` | which buffer-descriptor word 3 works on gfx1100 |
| `push_ar.hip` | a push all-reduce prototype: correctness and bandwidth |
| `bw.hip` | where the push bandwidth goes: peer write vs local uncached read |
| `test_qr.py` | rebuilt QuickReduce vs RCCL, all four codecs (`torchrun --nproc_per_node=2`) |
| `rccl_bench.py` | RCCL all-reduce latency baseline |
| `bench.py` | end-to-end decode rate, distinct prompts, TTFT kept apart |

Build a `.hip` probe with
`hipcc --offload-arch=gfx1100 -O2 -o <name> <name>.hip`. They need both GPUs
idle, so stop the engine first.

## Two false greens this cost

- `rocm-bandwidth-test -a -v` reports PASS between the two GPUs. HSA schedules
  the copy on the **source** agent, so it only ever tests a push. Asked for one
  direction with `-s 1 -d 2` it validates the `[1][1]` diagonal — a device
  copying to itself — and prints `N/A` for the cell that was the point.
- A garbage detector built on word fractions calls `ductductduct...` clean.
  Prove the checker against a known-bad sample before trusting its green.
