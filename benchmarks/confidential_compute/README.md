# Confidential Computing bridge microbench

Under NVIDIA GPU Confidential Computing (CC) every CPU↔GPU copy goes through an
encrypted bounce buffer, and `cudaMemcpyAsync` (`tensor.copy_(...,
non_blocking=True)`) becomes **host-synchronous**: the calling thread blocks
until the copy *and everything already queued on that stream* has finished.
Serving loops that were tuned on the assumption that these copies are async
lose their CPU/GPU overlap. Defaults like async scheduling can then give no
benefit at all, or even hurt. We call that a *policy inversion*.

This directory holds a small, self-contained harness that measures that
behaviour on your hardware and shows its effect on a vLLM-style decode loop.
It needs only PyTorch, plus `nvidia-ml-py` for CC detection. It does not
import vLLM, so the scripts can be copied to any machine.

## Quick start

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
pip install -r benchmarks/confidential_compute/requirements.txt   # skip if torch is already installed
bash benchmarks/confidential_compute/run.sh            # full suite, ~5-10 min
bash benchmarks/confidential_compute/run.sh --quick    # smoke run, ~1 min
```

Results land in `./cc_bench_results/<host>_<gpu>_cc-<on|off>_<time>.{json,md}`.
The Markdown file is also printed, and it starts with a **Findings** section in
plain language, for example:

> Pinned H2D with non_blocking=True BLOCKS the host until the busy stream
> drains (19987us of a 20000us kernel). An H2D on the compute stream stalls the
> scheduler for ~one forward step.

## The CC-off vs CC-on comparison

The interesting output is the *difference* between CC off and CC on on the
same hardware.

1. Run the harness once with CC **off** and once with CC **on**, on the same
   GPU, driver, and torch build.
2. Compare the two runs:

   ```bash
   python benchmarks/confidential_compute/compare.py \
       cc_bench_results/*cc-off*.json cc_bench_results/*cc-on*.json -o cc_compare.md
   ```

How to switch modes depends on the platform. Check your vendor's CC deployment
guide. Some common routes:

- **Hopper / Blackwell (single GPU):** use `nvidia_gpu_tools.py` from
  [NVIDIA/nvtrust](https://github.com/NVIDIA/nvtrust) (`--set-cc-mode=on|off
  --reset-after-cc-mode-switch`) on the host. Then start the guest
  (TDX / SEV-SNP CVM) with the GPU passed through.
- **Multi-GPU:** protected PCIe (PPCIe) or Blackwell NVLink encryption (NVLE).
  The harness records the mode as `multi_gpu=protected_pcie|nvle`.
- Inside the guest, `nvidia-smi conf-compute -f` shows the CC feature state, and
  `nvidia-smi conf-compute -grs` shows the GPU ready state. `run.sh` prints the
  first of these.

The harness reads the state through NVML (`nvmlSystemGetConfComputeState`), the
same source vLLM#52226 uses. If NVML is unavailable, label the run yourself
with `--cc-label on|off`. *DevTools* mode reports `cc_feature != 0` and is
labelled `on`. Do not treat it as a CC-off baseline.

## What each test measures

| test | question | what to look at |
| --- | --- | --- |
| `semantics` | Does a `non_blocking` copy return immediately, or block the host until queued GPU work drains? Covers the same stream, an idle stream, and an idle stream that event-waits on the busy one. | `call returns (us)` compared with `busy_us` (20 ms by default), plus the `verdict` column |
| `bandwidth` | Copy latency and bandwidth as a function of size, H2D and D2H, pinned vs pageable memory. | Fixed per-copy overhead at 4K–64K (decode-sized copies) and peak GB/s at 16M–256M. `host-issue / total` ≈ 1 means the host thread was stuck inside the copy calls. |
| `concurrency` | Does aggregate bandwidth scale with N streams on one thread, N threads with one stream each, or N processes with one CUDA context each? | `scaling vs n=1`. Under CC, one thread with many streams cannot overlap copies, because every call blocks. Threads and processes show whether the secure-copy channel itself is the limit. |
| `decode` | A vLLM-style decode loop: per step, CPU prep, an H2D of 64 KiB, a GPU "forward", then a D2H of 4 KiB. It runs in four modes (listed below), at several forward lengths. | `speedup vs sync` and `GPU busy` |

The four decode modes:

- `sync_sched`: no overlap; this is `--no-async-scheduling`.
- `async_sched`: vLLM's default. The CPU prepares step N+1 while step N runs. The H2D runs on the compute stream, and the D2H is issued by the scheduler thread on a copy stream.
- `async_staged_h2d`: as above, plus H2D into a staging buffer on a dedicated prep stream, then a D2D on the compute stream. This is `StagedH2DCopier` in vLLM#52226.
- `async_staged_h2d_d2h_worker`: as above, plus the D2H readback is issued from a worker thread. This is `AsyncD2HCopyWorker` / deferred readback in vLLM#52226, and mirrors TensorRT-LLM#8463.

The decode loop checks every token it reads back (the GPU computes
`input + 1`). A pipeline race raises an error instead of reporting a
misleadingly fast number.

### Expected pattern

This is what the bounce-buffer model predicts. Your run confirms or refutes it.

| | CC off | CC on |
| --- | --- | --- |
| `semantics h2d/pinned/same_stream` | `async`, a few µs | `blocks_on_busy_work`, ≈ `busy_us` |
| `semantics h2d/pinned/idle_stream` | `async` | only its own transfer, which is the premise of staged H2D |
| `semantics d2h/pinned/event_wait_stream` | `async` | blocks until the forward finishes |
| `semantics d2d/device/same_stream` | `async` | `async` (device-only copies are not encrypted) |
| `decode async_sched` speedup vs sync | > 1, up to `(cpu+fwd)/max(cpu,fwd)` | ≈ 1 (**policy inversion**) |
| `decode async_staged_h2d_d2h_worker` | ≈ `async_sched` | recovers most of the CC-off overlap |

If `h2d/pinned/idle_stream` also blocks under CC, the driver serialises copies
across the whole device. In that case staged H2D cannot help on that driver.
That result is worth reporting to NVIDIA, together with the JSON from the run.

For reference, vLLM#52226 reports these end-to-end gains on B300 with CC on
(Qwen3.5-397B FP8, TP4, ISL/OSL 1024/1024): output tok/s rises by 12–32% with
the staged-H2D and D2H-worker changes. At low concurrency, CC goes from about
71% to 93–95% of CC-off throughput.

## Knobs

`python benchmarks/confidential_compute/cc_bridge_bench.py --help` lists all
options. The most useful ones:

```bash
# only the decode emulation, matching a model whose decode step is ~6 ms with ~1 ms of CPU prep
cc_bridge_bench.py --tests decode --fwd-us 6000 --cpu-us 1000

# decode-sized copies only, with a longer busy kernel
cc_bridge_bench.py --tests semantics,bandwidth --sizes 1K,4K,16K,64K --busy-us 50000

# copy-channel scaling with more workers, on GPU 3
cc_bridge_bench.py --tests concurrency --workers 1,2,4,8,16 --device 3
```

`--fwd-us` stands in for the decode step time, so small values model low
concurrency. `--cpu-us` stands in for the scheduler and input-prep work per
step. Take both from a CC-off `vllm bench serve` run (TPOT, plus a profile of
`execute_model`) to make the emulation match your deployment.

## From microbench to `vllm serve`

The microbench explains *why* a default should change. Confirm that the change
helps end to end with a fixed A/B test, keeping the model, hardware, ISL/OSL,
and concurrency the same and toggling one thing at a time:

```bash
vllm serve $MODEL --async-scheduling      # vLLM default when compatible
vllm serve $MODEL --no-async-scheduling
vllm bench serve --model $MODEL --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 --max-concurrency 4
```

Run the matrix {CC off, CC on} × {async, sync} × {without, with vLLM#52226},
and attach the `compare.py` output to the PR.

## Limitations

- The decode loop is a model, not vLLM. It has no CUDA graphs, no sampler, no
  NCCL, and its "forward" is a `torch.cuda._sleep` spin kernel. It isolates
  copy/overlap behaviour and is not a throughput predictor.
- The busy loop for `--cpu-us` holds the GIL, as the Python scheduler does.
  Values above Python's 5 ms thread switch interval delay the D2H worker
  thread.
- One GPU per run. Multi-GPU (PPCIe / NVLE) collectives are out of scope.
