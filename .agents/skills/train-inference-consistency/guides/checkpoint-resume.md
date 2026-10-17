# Checkpoint, resume and memory budget

Use this guide to get checkpoint save, load and resume right in long RL runs, and to budget host (CPU) memory peaks and growth. For per-phase GPU memory peaks see [Locating GPU OOM and staging peaks by lifecycle](#locating-gpu-oom-and-staging-peaks-by-lifecycle).

## Find by symptom

| Symptom | Where to go |
| --- | --- |
| OOM while loading a checkpoint | [Load](#4-load-enough-capacity-does-not-mean-enough-for-the-transient-peak) |
| Train-inference mismatch on the first step after restore | [13](#13-stale-quantization-scale-on-the-first-step-after-restore) |
| Resume is set but training starts from step 0 | [14](#14-resume_modeauto-starts-from-step-0) |
| Host RSS grows every step | [3.3](#33-locating-slowly-grows-every-step) |
| Planning host memory for a checkpoint in the hundreds of GB | [4](#4-capacity-planning) |
| vLLM fails to start when colocated with training (KV budget too small) | [5](#5-gpu-memory-set-gpu_memory_utilization-per-mode-when-training-and-rollout-are-colocated) |
| GPU OOM during init / export / wake | [Locating GPU OOM](#locating-gpu-oom-and-staging-peaks-by-lifecycle) |

## Terms

- **pinned memory:** page-locked host memory for direct GPU DMA (`cudaHostAlloc` / `cudaHostRegister`), cached by PyTorch's caching host allocator.
- **cgroup:** limit `memory.max`, usage `memory.current`, split in `memory.stat` into anon, file (page cache) and shmem.
- **RSS and PSS:** RSS counts shared pages fully in every process; PSS splits them, so only summed PSS approximates the physical total.
- **DCP / Megatron dist_ckpt:** PyTorch and Megatron distributed checkpoint formats; each rank writes its shard, `.metadata` last.
- **staging / D2H:** the host buffer GPU tensors are copied into before writing to disk.
- **Ray object store (plasma):** Ray's shared memory for passing objects, mapped into each worker and charged via PSS.
- **old-logprob:** the logprob training recomputes for rollout tokens with the pre-update policy.
- **Ray OOM threshold:** `RAY_memory_usage_threshold`, the usage fraction at which Ray kills workers. Default 0.95.

## Key rules

- Check file, memory and derived state separately; complete files can still mismatch on the first step after restore.
- Capture the first exception (errno, rank, file, offset), not the last message.
- Completion marker last; clear the old directory before overwriting; all ranks check and fail together; a missing file raises instead of falling back to a fresh optimizer.
- Under EP4 + FSDP2, ranks' expert shards in DCP share key and offset, so restore reads another rank's experts. Put the EP rank in the key (port the reference fix), add a 4-GPU bitwise read-back unit test, and raise on EP size mismatch.
- Rotation deletes only tracked, complete, idle directories; in long runs keep one checkpoint every 20 steps outside rotation.
- Set rollout `gpu_memory_utilization` per training mode ([5](#5-gpu-memory-set-gpu_memory_utilization-per-mode-when-training-and-rollout-are-colocated)).
- Save fixes pass [the acceptance in 3.2](#32-reproduce-and-discriminate). Before a long run on a new save path or layout, run to the first save and do one kill→resume.
- Resume is proven by matching per-rank digests in a new process, a byte-identical first export, an aligned first old-logprob and one non-zero update; `resume_mode=auto` alone proves nothing.
- Read host memory as anon/file/shmem; `memory.current` includes page cache, and Ray kills on it.
- Pinned buffers round up to a power of 2 (Nemotron: 146.9 GiB pinned for 88.2 GiB of optimizer state, +67%); allocate large long-lived pinned blocks at exact size.
- The save step is the peak; a higher Ray threshold or smaller microbatch does not fix headroom.
- For per-step growth, use the per-process PSS floor and a small CPU-only repro before changing code.
- Verify per save path that the host cache is released after save.

## Checkpoint and resume

### 1. Three independent contracts

Classify the problem before calling it "the checkpoint is broken":

| Contract | Typical failure |
| --- | --- |
| File: serialization and publish | Disk write error; an incomplete directory treated as complete |
| Memory: transient peaks of save and load | Killed by Ray after the write finishes; cgroup OOM during load |
| Derived state: the first export after restore | Files complete, but the first step after restore mismatches ([stale quantization scale](#13-stale-quantization-scale-on-the-first-step-after-restore)) |

The third class is a derived-state lifecycle bug, not file corruption.

### 2. Which state exact resume needs

| State | Requirement |
| --- | --- |
| Model parameters or LoRA adapter shards | Saved per rank; strict load that rejects missing/unexpected keys ([training forward](training-forward.md)) |
| Optimizer state, including the CPU-offloaded part | Same step as the model. Nemotron chunked optimizer offload: 88.2 GiB per rank, 94.7 GB per rank on disk; DS4.1 saves LoRA parameters and Adam `exp_avg` / `exp_avg_sq` in FP32 |
| LR scheduler | Check the saved value, not the source default (DS4.1 checks its constant, no-warmup schedule per rank) |
| RNG | Four streams: CPU, CUDA, Python, NumPy |
| Counters and data position | global step, optimizer step, data cursor; DS4.1 also saves the cumulative count of effective-signal steps |
| Run contract | Hashes of code, data and recipe, compared item by item on restore; reject a change in total steps or data by default. To extend training, declare a new phase explicitly; do not edit the old contract file or add a generic ignore switch |
| Completion marker | `.metadata` / tracker (e.g. `latest_checkpointed_iteration.txt`), written last, see 3.4 |

Not in the checkpoint, rebuilt after restore:

- **Parameter-derived caches:** quantization scale, packed/transposed/fused weight, version markers (see 5).
- **Records outside training:** the metrics jsonl is overwritten on resume.

### 3. Save

#### 3.1 Capture the first error, do not fix the last error message

Keep the writer callback's original exception / errno, rank, path, offset, requested / returned bytes and exception chain, which DCP and Ray wrappers may drop; also filesystem/mount, free space, quota and cgroup OOM events, not just the final `ActorDiedError`. `unexpected pos` can come from disk space, quota, permissions or I/O failure; the first exception tells them apart.

#### 3.2 Reproduce and discriminate

A fast repro keeps the real tensors, serializer and concurrency, skips rollout/backward and repeats only save, for a bounded number of rounds sized to the failure probability; stop at the first failure and keep the scene. DS4 model-only save failed on the 3rd attempt on one path and the 6th on another; Nemotron with host cache release failed on the 2nd-4th save.

**Acceptance for a save fix: N consecutive saves + kill→resume→save again.** Also run it before the first long run of a new save path, parallel layout or node count; running to the first save point is enough.

Order of discrimination:

1. Same buffer/serializer, target filesystem versus node-local disk as the control; read the errno of the first failure.
2. Control concurrency, pinned/pageable and memory peak; if the serializer changed along with the filesystem, the result cannot be attributed to one factor.
3. Check growth of threads, processes and file handles, separately from the write error.
4. Only then read kernel / filesystem client code; a matching historical issue is supporting evidence, not proof of a fix.

#### 3.3 Reliable publish

Serialize on node-local disk and close the file; copy to a **temporary name on the target filesystem**, handle short copies, flush/fsync as durability requires, verify integrity, then rename within the same filesystem. Publish metadata/tracker/completion marker only after all ranks finish. A move across filesystems is not an atomic rename.

#### 3.4 Commit protocol: write the completion marker last

| Rule | If you do not |
| --- | --- |
| Write `.metadata` / tracker last, after every rank's shards and auxiliary files (optimizer, RNG, scheduler) succeed and are confirmed collectively | A rank fails, the marker already exists, and restore reads incomplete state |
| Before overwriting a step directory, delete all old files first (uncommit) | Old optimizer / RNG silently pair with the new model |
| Before restore, all ranks check file completeness together and fail together | The rank missing a file exits; the others hang until the NCCL timeout |
| Ranks with no shard to write still join the same collective | Collective order diverges across ranks and hangs |
| A requested load with a missing file raises an error, not just a log line | Training silently continues with a fresh optimizer |

Acceptance: a save failing midway (Nemotron hit a real ENOSPC) makes all ranks exit together without advancing the tracker, and training resumes directly; new code loads an old-code checkpoint. Never kill a job mid-write; verify the checkpoint is complete first.

#### 3.5 Verify a written checkpoint

Check the expected shard set, non-empty sizes, leftover temporary files, checksums where needed, and do one real full restore. A correct save digest is not a restore test (DS4.1 checkpoints 20/24 had correct save digests but were never restored in a new process, and the report says so).

### 4. Load: enough capacity does not mean enough for the transient peak

Load may copy the full optimizer into anonymous memory, and FSDP CPU offload may create a second set of pinned storage. Accept by loading model + optimizer + RNG/extra at real scale, not one small shard.

Read a checkpoint with the image that wrote it: DS4.1 host PyTorch 2.11 cannot deserialize the DTensor `_FlatLayout` written by the image's 2.13; it reads only inside the image with `torch.load(weights_only=True, mmap=True, map_location="cpu")`. Do not modify the checkpoint or turn off `weights_only` to make it load.

### 5. Derived state after restore

DS4 chain: initial HF load saves FP8 source scales → DCP restores the trained BF16 parameters → the first export still uses the old registry → the actor re-quantizes by parameter version, but rollout uses the old scales. The first resumed step mismatches; once the next optimizer hook clears the cache it matches. The prediction "it recovers on the next step" plus a before/after-fix comparison on the same checkpoint locates the cause.

Fix it in the lifecycle that owns the state: after load/update, invalidate or rebuild parameter-derived caches (scale, packed/transposed/fused weight, version markers), reusing an existing hook without side effects such as a real optimizer update. Do not hide it by waiting a step, disabling resume or clearing all caches by hand. DS4's minimal patch calls the existing post-update hook ([Megatron-LM #197](https://github.com/ISEEKYAN/Megatron-LM/pull/197)). List every parameter-derived cache and its invalidation code, add the missing calls and one regression test.

### 6. How to prove resume is correct

1. **Before launch.** For node-local checkpoints, every rank's node in the new allocation has its shard; the tracker / latest pointer points to a complete checkpoint; the launcher's resolved `resume_from_path` and global step are as expected. Confirm the restored step before the first training log line; `resume_mode=auto` is not evidence.
2. **New-process restore.** All worker PIDs changed; per-rank adapter / optimizer / scheduler / RNG digests match the save-time digests item by item.
3. **First export and old-logprob.** Exported files are byte-identical to those of the saving step; also check the exporter scale / policy fingerprint and the full valid logprob. Checking only that the second step returns to 0 misses the bug in 5.
4. **One non-zero update.** A zero-advantage step only exercises the flow (two DS4.1 resume segments had zero gradient on all ranks at step 2, proving nothing about restored adapter values).
5. **Gate coverage.** step0 cold load, first export/old-logprob after restoring a non-zero step, one later update, address / state validity under Graph and sleep/wake, new code reading an old checkpoint.
6. **What acceptance compares.** Restored per-rank digests equal save-time digests; first exported files are byte-identical to the saving step's; per-step `probs_diff` after restore is 0 under BI=1 (Nemotron: 0 on the first step after kill→`resume_mode=auto`). The resumed trajectory need not match an uninterrupted run bitwise (rollout sampling is not bitwise across processes). There is no "metrics are close" threshold; reward curves are reference only, on the same step range and basis.

### 7. Rotation, disk and time cost

| Item | Record |
| --- | --- |
| Size per checkpoint and save time | DS4 full32: 99 files, about 3.2 TiB, save about 269 s. Nemotron-H 30B-A3B, 4×GB200: 353 GB (EP4 + FSDP2) or 429 GB (PP4 dist_opt), save 72-152 s, about 2-3% of step time when saving every 5 steps. Layout-specific; state these for your model in the plan |
| Rotation | The mLite engine ignores `max_actor_ckpt_to_keep` |
| Checkpoint used for eval | Rotation may have deleted the step to evaluate; a checkpoint from another chain cannot substitute |
| Resume on a different node after node failure | Copy checkpoint, dequantization cache, pinned image and dependency cache to the new node before launch; disable automatic requeue; change only the node selection; write a transfer record |
| Paths for two-arm experiments | BI=0 / BI=1 output and checkpoint paths are separate; renaming the run breaks a resume chain in progress ([recipe and acceptance](recipe-and-acceptance.md#rules)) |

## Memory budget: host and GPU

Host memory fails in two ways in long RL runs: phase **peaks** (save / load / sleep) hitting the limit, and slow per-step **growth**; they are measured differently (1-4). GPU memory is a third class: with training and rollout colocated, rollout `gpu_memory_utilization` is set per training mode (5).

### 1. Who holds host memory

| Source | Mechanism |
| --- | --- |
| Pinned buffer of optimizer offload | The caching host allocator rounds up to a power of 2, so the footprint is tens of percent above the nominal state; summed across ranks it pushes residency before save close to the limit |
| Parameter shards of FSDP CPU offload | Each rank's shard stays resident on CPU; doubles if load creates a second pinned copy |
| D2H staging during save | Stays in the pinned cache pool; whether `torch.accelerator.empty_host_cache()` releases it must be verified per save path |
| CPU backup of serving sleep | L1 sleep keeps one full backup per rank; the pinned cache is not returned after wake |
| Ray object store | plasma mappings count into each process's PSS as shared pages; node shmem does not grow with it (3) |

`param_offload` trades CPU pressure for GPU memory; optimizer offload is separate. Change one offload switch at a time and measure the phase peak before and after.

### 2. Limits and who does the killing

- **cgroup.** The job limit `memory.max` is usually below physical memory; budget against it. A cgroup OOM shows an `oom_kill` event and process SIGKILL, with no Python stack.
- **Ray memory monitor.** Kills workers at `RAY_memory_usage_threshold`=0.95 by default; log line `exceeded threshold of <bytes>B (95.0% ...)`. Its usage figure includes page cache. Raising the threshold does not fix headroom: Nemotron at 0.985 was still killed during save.
- Distinguish GPU allocation exception, host cgroup OOM and Ray kill. A CUDA allocator allocation-failure / retry warning is not a terminal OOM and the step may complete, but it does show memory pressure.

### 3. How to measure

#### 3.1 One record per phase

Log one line at load, save, sleep, wakeup, reset KV cache and generation start / end, with monotonic time, node / rank / PID and policy version. Per node, record cgroup `memory.current` / peak / events and `memory.stat` anon/file/shmem; sample process RSS/PSS, threads and FDs as needed, not with high-frequency scans of all processes.

`generate_sequences` includes wake/init. An error inside it does not prove token generation started, let alone a backward/optimizer OOM.

#### 3.2 Pitfalls in the readings

| Reading | Problem |
| --- | --- |
| `memory.current` | Includes page cache, which masks per-step growth; but it is what Ray and the save peak see |
| `free`, `psutil.virtual_memory().used` | System-visible memory, not process RSS or job cgroup usage |
| `scontrol` FreeMem | Excludes reclaimable cache |
| `ru_maxrss`, Slurm MaxRSS | Cumulative peak, not current RSS |
| VmPin/VmLck | Cannot alone infer CUDA pinned |
| Sum of per-process RSS | Counts shared mappings repeatedly; use PSS |

Pinned memory from cudaHostAlloc shows up as shmem in cgroup `memory.stat` (measured on Nemotron).

#### 3.3 Locating "slowly grows every step"

1. Use node `MemTotal−MemAvailable` and cgroup anon/file/shmem to find which class grows.
2. Sample PSS from `/proc/<pid>/smaps_rollup` per process once a minute, recording only comm / argv0; take each step's minimum as the floor and find processes that rise monotonically. Within-phase peaks swing with offload and are useless here.
3. For a rising process read `smaps`. On someone else's job or a running job, only read `/proc`; do not attach or send signals.
4. Reproduce the candidate mechanism in a small CPU-only process, then decide where to change code.

A floor that drops after resume (Nemotron run 1: about 455 back to about 433 GiB) shows the growth accumulates over the process lifetime.

### 4. Capacity planning

Budget the save-step peak as the sum of what is alive at the same time, against cgroup limit × Ray threshold:

```text
peak  ≈ resident pinned optimizer offload (actual footprint after allocator rounding)
      + save D2H staging (and the part of it left in the cache pool)
      + page cache of the newly written checkpoint files
      + CPU backup of serving sleep and unreleased pinned cache
      + resident amount of the other processes (rollout, driver, Ray workers, etc.; take the floor per 3.3)
limit = cgroup memory.max × RAY_memory_usage_threshold
```

Use measured values wherever possible.

- **The save-step peak is closest to the limit.** The long-run plan states, per training mode, the measured cgroup peak at the save step and in the kill→resume segment, and which load is still unmeasured.
- **Estimate from headers before running.** Safetensors headers alone give the expanded size per dtype (full DS4.1 checkpoint: about 1421.66 GiB as BF16, including auxiliary content). It is not peak RSS; add serializer copies and mapped source pages.
- **A slope only hints at headroom risk.** Continue from a verified checkpoint and observe long enough; do not extrapolate an OOM step linearly from a short timeline.
- **On OOM, do not shrink the microbatch or raise the guard first.** Find the phase and which objects are alive together.

### 5. GPU memory: set gpu_memory_utilization per mode when training and rollout are colocated

The training process keeps part of its GPU memory during rollout. Nemotron single node 4×GB200:

| Training mode | Held per GPU by training during rollout | rollout `gpu_memory_utilization` |
| --- | --- | --- |
| EP4 + FSDP2 (mLite; BI=1 arm default `ACTOR_OPTIMIZER=fsdp2`) | Not measured separately | 0.65 (verified) |
| dist_opt (BI=1 arm optional `ACTOR_OPTIMIZER=dist_opt`; numbers from PP4 dist_opt measurements) | 54-64 GiB (FP32 quantization master resident on GPU) | 0.7, 0.6 and 0.65 fail to start; measured 0.55 |
| EP4 + Megatron (mbridge) dist_opt + chunked optimizer offload (BI=0 arm, QAT + R3) | Not measured separately | 0.45 (current recipe value) |

FlashInfer autotune warmup needs about 8 GiB more, another direct cause of failing to start. In a two-arm comparison, differences such as gpu_mem 0.45/0.65 are confounders and are listed first ([recipe and acceptance](recipe-and-acceptance.md)).

Rules:

- Write the value into the recipe per training mode, not as one global default; the recipe's hardware comment lists only verified topologies.
- Check the resolved value inside the container and the actual KV budget in the vLLM log, not the outer shell variable.
- `param_offload=True` saves GPU memory but moves parameters, changing the `data_ptr` of the quantization master so the pre-export check errors; as with DS4, enable only optimizer offload.
- What sleep level 1/2 release, and why wake may still find no GPU memory: [rollout engine §5](rollout-engine.md#5-lifecycle-and-gpu-memory-ledger) and [Locating GPU OOM](#locating-gpu-oom-and-staging-peaks-by-lifecycle).

## Locating GPU OOM and staging peaks by lifecycle

Use this when the shard estimate fits but init, export or wake still OOMs. Find which step allocates and what is still alive, and change that boundary; do not lower the batch or change sharding first, and never blame update for a failure before update.

### 1. The minimum to record

```text
phase / process / rank / topology
requested bytes / allocated / reserved / device free
host RSS / container or Ray limit
last completed phase / first failing call
```

Mark missing fields as not obtained; do not guess. If output just stops with no error, check CPU/GPU/I/O, log timestamps and phase (reading weights, compiling, collective, generating); slow is not a hang.

### 2. The memory peak is set by "the objects alive at the same time"

Parameters divided by GPU count estimates only part of the steady state and misses these peaks:

| Phase | Objects and boundaries to look for | Minimal verification |
| --- | --- | --- |
| Init / FSDP wrap | Full materialization before sharding; largest single parameter; CPU-resident/tied parameters | Whether meta → shard → load takes effect; per-rank peak |
| forward / backward | Full module from all-gather, activation, workspace | Measure logprob and actor update separately; frozen parameters may be gathered too |
| Export / merge / quantize | A single full tensor, fused groups, lazy generator references | The actual export set; release timing after yield/exception |
| compile / graph capture | autotune workspace and graph pool | Find the allocation in the log; eager only isolates, recheck the production graph separately |
| KV planning / wake | KV budget, trainer residency, allocator free cache | Resolved GMU/TP and allocator metrics, not the configured value |
| checkpoint / host | Full CPU copy, optimizer gather, shared mappings, Ray guard | Per-process and node usage; RSS does not sum to the physical total |

A small model with the same special layers, attention period and scale layout verifies the mechanism; only the full model verifies capacity, such as the all-gather peak of a huge single parameter.

### 3. Reserved is high, so why can wake still have no GPU memory

After update's temporaries are freed, PyTorch keeps the blocks as reserved-but-unallocated. Another allocator such as CuMem cannot use them.

`empty_cache()` releases free cache only, not live tensors, and not reserved memory held by active blocks or a graph's private pool. `gc.collect()` helps only when collecting a reference cycle drops a tensor's last owner; high reserved alone does not prove GC is needed.

Compare allocated/reserved/device-free and the same allocation before and after cleanup. If allocated stays high, find the live owner; if reserved-unallocated is small, empty-cache cannot explain a workspace OOM of tens of GiB. Clean up where the resource is handed back (end of update or before wake), chosen by the real call path and a retest.

### 4. How to tell a fix from a config downgrade

- Lifecycle error: fix materialization/release timing, the completion contract, or the wrong gather set.
- A legitimate peak exceeds capacity: pick a topology that carries it or shrink the declared configuration, and record the support boundary.
- Configuration not propagated: confirm the resolved value inside the container; editing the shell variable again is not a new comparison.
- Killed by Ray/container policy: account for actual headroom before judging the limit; do not raise the guard by default.
- Long duration: time by phase and optimize only large shares; GPU utilization is not evidence of completion.

Do not add a resident full-tensor dump to chase an OOM; use the phase record above to decide the next step. For the full-tensor boundary of export quantization see [precision and quantization](precision-quant.md#41-fsdp-path-verl).

### Evidence boundary of historical experiments

- A configured path is not an executed path: record a marker/count and treat zero hits as an invalid verification.
- Before comparing tensors, align request, token position, rank/owner and valid region; equal byte hashes alone do not prove the same objects.
- Verify a root cause with a single-variable intervention and a negative control that actually triggers the failure.
- Keep failed, invalid and superseded records with reasons. A small model locates the mechanism but does not extrapolate to full-model capacity or performance; the profiler explains structure, latency comes from independent timing.
- Bind every result to the actual patch stack and imported-file hashes; overlaying only new files keeps the old overlay, so restore and apply by manifest ([code and delivery](../practices/code-and-delivery.md)).

## Pitfalls

### 13. Stale quantization scale on the first step after restore

- **Symptom:** After DS4 restores from a non-zero step, the first rollout mismatches and the next step matches again by itself.
- **Cause and Fix:** See [5](#5-derived-state-after-restore).
- **How to verify:** On the same checkpoint, compare the first export's scale fingerprint and the old-logprob before and after the fix.

### 14. resume_mode=auto starts from step 0

- **Symptom:** A job with `resume_mode=auto` retrained from step 0; the log said `latest_checkpointed_iteration.txt` was not found.
- **Cause:** The checkpoint is on node-local NVMe and the newly allocated nodes have no tracker / shards. Same class: shards on the NVMe of the original eight nodes with the global `data.pt` on only one, so the launcher must check the exact node set.
- **Fix and verification:** Check before launch per item 1 of [6](#6-how-to-prove-resume-is-correct).
