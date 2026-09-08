# EC connector benchmark — handoff

Everything here measures the **encoder-cache (EC) connector** on branch
`p2p_nixl_ec_connector`: whether offloading/transferring encoder outputs beats
recomputing them, and how the CPU/NIXL transport compares to the reference
shared-filesystem one.

Read the **Traps** section before running anything. Most of the wasted time in
this effort came from those, not from the code under test.

---

## 1. The target design (six arms)

Defined by the project owner; arms 3 and 5 are the ones that motivated the
proxy patch shipped here.

| # | arm | topology | connector | what the decode instance receives |
|---|---|---|---|---|
| 1 | `baseline` | one instance | none | n/a — vanilla, encodes and decodes itself |
| 2 | `offload` | one instance | `ECCPUConnector`, `ec_role: ec_both` | n/a — local CPU-mmap offload, value comes from **reuse** |
| 3 | `cpu-data` | encoder + decode | `ECCPUConnector` + NIXL | **pixels** + uuid + `ec_transfer_params` |
| 4 | `cpu-grid` | encoder + decode | `ECCPUConnector` + NIXL | **grid** (`image_grid_thw`) + uuid + params |
| 5 | `example-data` | encoder + decode | `ECExampleConnector` | **pixels** + uuid + params |
| 6 | `example-grid` | encoder + decode | `ECExampleConnector` | **grid** + uuid + params |

3-vs-4 and 5-vs-6 isolate what the grid substitution saves, with the transport
working in both. 4-vs-6 compares the transports.

### Why arms 3 and 5 needed a proxy change

The grid substitution and the transfer params are **independent**, but
`--no-rewrite` disabled both:

- `disagg_epd_proxy.py:255` assigns the uuid only when rewriting
- `:317` attaches `ec_transfer_params` only when a uuid exists
- `:507`/`:551` strip the pixels only when rewriting

`ECCPUConnector` can locate a peer's entry **only** through
`ec_transfer_params` (`cpu/scheduler/__init__.py:181` returns early when the
request has none), because the entry lives in the producer's private
`/dev/shm` region. `ECExampleConnector` needs no params — it rendezvouses by
content hash on shared storage.

So the shipped `--no-rewrite` produced a "CPU connector" arm that transferred
**nothing**. `proxy_pass_mm_data.patch` (in this directory, applied to the
worktree) adds `--pass-mm-data`: keep the pixels, still assign the uuid, still
carry the params. Verified at runtime — modes resolve to `data` /
`no-rewrite` / `rewrite`.

**This is a local modification to in-tree example code.** Any arm-3 or arm-5
number depends on it. Say so when reporting.

---

## 2. Files

| file | role |
|---|---|
| `gen_workload.py` | builds the image pool + `custom_image` JSONL + `manifest.json`. Real photos, Lanczos-enlarged, JPEG. |
| `run_bench.py` | server lifecycle, load driving, log accounting, gates, tables. Both topologies. |
| `ec_log_stats.py` | shared log parsers (EC transfers, encoder inputs, proxy `STAGE` lines, rewrite counts, queue slices) |
| `queue_sampler.sh` | samples `vllm:num_requests_{running,waiting}` from every instance's `/metrics` once per second |
| `patches/sitecustomize.py` | descriptor-count instrumentation, wraps `_coalesce_runs` without touching the connector. **Never yet exercised.** |
| `phase0_hit_check.py` | correctness gate: proves a repeat pass reuses encoder outputs |
| `micro_swap_blocks.py` | descriptor-layout microbenchmark (owned by this effort; a previous agent rewrote it) |
| `proxy_pass_mm_data.patch` | the `--pass-mm-data` proxy mode |
| `results_grid_vs_pixels.html` | write-up of the grid-vs-pixels result (unpublished; needs a claude.ai login) |
| `results_archive/` | pulled result JSONs + run log + workload manifest, so they survive the pod |

### Key `run_bench.py` structures

- `EPD_CONFIGS` — arm name → `(rewrite: bool, transform device, connector)`.
  **Must become `(payload_mode, device, connector)`** with `payload_mode` in
  `{grid, data, pixels}` for the six-arm design. Not yet done.
- `EPD_EXPECT_LOADS` — arms where the consumer must load encodings. Asserted
  both ways: an arm not in the set must load nothing.
- `--encoder-devices` — list length is the encoder count; a repeated device
  means those encoders share a GPU and split its memory. Accepts MIG UUIDs.
- Per-encoder port / log / pid / `engine_id` / side-channel port are derived
  from the index, so N encoders never collide.

---

## 3. Environment

Single pod, intra-node, two H100 80GB. **No MIG** (needs pod permissions the
owner does not currently have).

```
pod                vllm-omer-2         (oc, not kubectl)
python             /vllm-workspace/venv-vllm/bin/python
vLLM checkout      /vllm-workspace/vllm
bench scripts      /vllm-workspace/bench
workload pool      /vllm-workspace/bench/wl
results            /vllm-workspace/bench_results
logs               /vllm-workspace/logs/epd_*.log
```

`/vllm-workspace` survives a pod recreate. **`/tmp` does not**, and the pod has
been deleted three times mid-session — keep everything under
`/vllm-workspace`.

### Setup after a recreate

`setup_bench.sh` (recreate from this doc if lost) runs under
`set -euo pipefail`, checks the HF credential *before* downloading, then
asserts manifest + workload + exact pool count. Steps:

1. model: `snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")`, ~16 GB
2. credential: LSDIR is gated; see Traps
3. pool: `gen_workload.py --photo-source hf-tar:ofsoundof/LSDIR:shard-00.tar.gz
   --out-dir /vllm-workspace/bench/wl --pool-size 96 --buckets 2048x2048:1.0
   --allow-upscale --num-requests 400 --reuse zipf:1.1 --self-check`

---

## 4. Known-good configuration (each was learned the hard way)

| setting | why |
|---|---|
| `--encoder-max-num-batched-tokens 65536` | 8192 against 5,350 tokens/image forbids batching two image requests, pinning the encoder at ~25 req/s. Every `rate=inf` measurement then becomes queue depth. |
| `--max-concurrency 1,4,8` | **never** `--request-rate inf` alone: it floods, and TTFT becomes queue depth ÷ service rate (we measured 20 s TTFTs and called them latency) |
| `--enable-mm-embeds` on the decode instance | without it every grid request returns HTTP 400. **Undocumented in the example's README**; `tests/v1/ec_connector/integration/run_epd_correctness_test.sh:186` sets it |
| `--mm-processor-cache-type lru` pinned on the encoder | with `shm` the engine keeps no receiver cache, the processed data is refilled in the *worker*, and the connector reports no grid — the change under test silently becomes a no-op |
| `--limit-mm-per-prompt '{"video":0}'` | drops the encoder-cache floor from 32768 to 16384 embeddings |
| `--enable-logging-iteration-details` | otherwise `encoder inputs:` never prints and the accounting reads zero |
| `VLLM_LOGGING_LEVEL=DEBUG` | the connector's transfer accounting is on debug lines |
| `--mm-encoder-only` on encoders | drops ~16 GB to ~1.4 GB. **Only valid when something consumes the embeddings** — on a connector-less encoder the instance has no language model, cannot serve the proxy's request, and EngineCore dies |
| `--enforce-eager` on encoders | the example README states encoder instances only work in eager mode (not enforced in code — I looked) |
| `cd /tmp` before launching | from `/vllm-workspace`, `import vllm` resolves the repo directory as a namespace package and top-level attributes vanish |

`ec_enable_nixl` goes **inside `ec_connector_extra_config`**, read via
`get_from_extra_config` (`cpu/scheduler/__init__.py:78`). It is *not* a
top-level `ECTransferConfig` field, and that config rejects unknown keys.

---

## 5. Results so far

### Validated

**Grid vs pixels, EPD, `ECExampleConnector`, 2048² images** (commit
`0be69a42c9`; two reps, order alternated, queue-verified):

| concurrency | throughput ratio | TTFT ratio |
|---|---|---|
| 1 | 1.28 / 1.23 | 0.69 / 0.72 |
| 2 | 1.24 / 1.24 | 0.69 / 0.70 |
| 4 | 1.14 / 1.15 | 0.77 / 0.75 |
| 8 | 1.12 / 1.13 | 0.74 / 0.72 |
| 16 | 1.04 / 1.07 | 0.90 / 0.78 |
| 32 | 1.02 / 1.02 | 0.88 / 0.94 |

Reproduces PR #50390's published 1.18–1.31× at low concurrency. The decay is
decode-GPU saturation: decode queue climbs 0 → 2 → 5 → 17 while the encoder
queue stays 0. Encoder queue 0 everywhere, so c=1–8 measure work; c=16/32 are
partly queue.

**Phase 0.5 descriptor fix** (committed): descriptor build 4.24 ms → 0.041 ms
per image; DMA 4.4 → 52.8 GB/s when entries coalesce. Caveat: that is a
*fresh-region* best case — a fragmented region still yields ~1,196 descriptors.

### Suspect — do not quote without re-running

**Five-arm run** (`results_archive/`, commit on pod at the time):

| arm | c=1 | c=4 | c=8 | transfers |
|---|---|---|---|---|
| `example` | 41.8 / 40.7 | 118.4 / 113.8 | 144.8 / 138.8 | 79 |
| `cpu` (no params) | 38.8 / 37.6 | 83.1 / 79.7 | 95.9 / 93.7 | **0** |
| `example-grid` | 53.7 / 52.5 | 133.8 / 128.3 | 165.5 / 163.9 | 79 |
| `cpu-grid` | 52.9 / 51.3 | 131.4 / 124.1 | 158.8 / 157.6 | 79 (NIXL) |

- `none` arm **invalid**: 3 of 120 requests completed (see Traps).
- `cpu` is the old broken arm — superseded by `cpu-data`.
- `cpu-grid` decode TTFB is much better and flatter than `example-grid`
  (277 vs 426–447 ms at c=8) but end-to-end throughput is slightly worse, and
  its encode-leg stage timing is much worse (373 vs 190 ms at c=8). **Cause
  unattributed.** DMA is eliminated: saves cost 32.7 ms total at 50 GB/s and
  occur only at c=1. Leading candidate is producer-side read-serving
  (side-channel grants + NIXL bookkeeping, polled per engine step), which
  would scale with concurrency as observed. The encoder-queue column was
  broken in that run, so service time and queue time were not separated.

**Never measured:** anything at N>1 encoders; `patches/sitecustomize.py`;
`--frag` fragmentation-over-time; MuirBench or any real multi-image dataset.

---

## 6. Open bugs and design gaps

1. **No `ECExampleConnector` save counter.** It logs `Save cache successful for
   mm_hash %s` (`example_connector.py:123`); `ec_log_stats.summarize()` counts
   only its loads. Every `save_*` zero for an example-connector arm means "not
   parsed", not "no saves". This made a published table misleading.
2. **The EC region is never reset between load points.** Servers start once per
   arm; c=1 pays the saves, c=4/c=8 run against a fully warm region (working
   set 2.81 GiB vs a 3.5 GiB region, so after the first pass **no further saves
   ever occur**). The encoder's processor cache is warm by then too. Arm-to-arm
   comparison at a given concurrency is still fair; the within-arm scaling
   curve is confounded. No reset endpoint exists — only `/reset_prefix_cache`
   and `/reset_mm_cache`. Fix by restarting servers per load point, and/or
   sizing the region below the working set
   (`manifest.expected.fragmentation_arm_ec_cpu_bytes`, 1.4 GiB) so eviction
   forces continuous saves.
3. **`EPD_CONFIGS` still carries a boolean `rewrite`** — needs the three-way
   payload mode for arms 3/5.
4. **Arms 1–2 use one GPU, arms 3–6 use two.** Direct throughput comparison
   credits disaggregation with the extra hardware.
   `--tensor-parallel-size` exists for a resource-matched variant; the owner
   had not chosen TP=1, TP=2, or both.
5. **A single multi-image request can exceed the consumer's encoder cache.**
   `encoder_cache_size` is its own scheduler field, floored by the largest item
   (`encoder_cache_manager.py:333`). `can_allocate` checks per-item compute and
   then *cumulative* cache. 4 × 2048² = 21,316 embeds > 16,384 → the request
   chunks across steps, confounding a fan-out measurement. The manifest reports
   `max_embeds_per_request`; keep it under the cache or raise the cache.
6. **Connector loads consume cache space but not compute budget**
   (`scheduler.py:1748`) — externally loaded items still compete for encoder
   cache.

---

## 7. Traps (all cost real time here)

1. **The pod runs different code than you think.** It has been found on
   `main`, on another feature branch, and on three different commits of this
   one. **Verify before every run** by content, not by SHA:
   `grep get_from_extra_config vllm/distributed/ec_transfer/ec_connector/cpu/scheduler/__init__.py`
   and `grep ECCPUConnector .../factory.py`. Probing
   `config/ec_transfer.py` for `ec_enable_nixl` proves nothing — it moved.
2. **Log-text parsing is a dependency on an unpromised string.** The proxy
   renamed `Rewrote N image item(s)` → `media item(s)` and a gate then reported
   "the change never engaged" while it was working perfectly. Prefer
   `/metrics` and the bench tool's JSON; when you must parse logs, assert the
   pattern matched *something*.
3. **`pkill -f <pattern>` matches its own shell.** Cost four separate
   incidents, including a watcher killing itself. Use a bracket
   (`[p]attern`) and exclude your own process group.
4. **Zombies count as alive to `pgrep`.** A `<defunct>` child made every
   teardown look hung and escalate to SIGKILL — which is how multi-GiB
   `/dev/shm` files leak, since the region's cleanup is best-effort and does
   not survive SIGKILL. Filter `stat !~ /^Z/`.
5. **Never wait on a success string.** A wait loop watching only for
   `self-check OK` hid a failed pool build for 25 hours. Wait for process exit,
   then inspect the outcome.
6. **`subprocess.run(capture_output=True)` never returns for a launched
   daemon** — it waits for inherited pipes to close. Use DEVNULL for
   fire-and-forget launches.
7. **Kill the process *group*.** Signalling the parent leaves EngineCore and
   workers holding the GPU and the port, which presents as the next arm's
   server never becoming healthy. Launch under `setsid`, have the setsid'd
   shell record its own `$$` as the group id (reading it back with `ps` races),
   and refuse to signal a group equal to your own.
8. **`--decode-port` must sit outside the encoder range**, which grows from
   `--encoder-port` with the encoder count. There is a check now.
9. **`HF_TOKEN` in the environment overrides a stored `hf auth login`**, so a
   login inside the pod looks like a no-op. LSDIR (`ofsoundof/LSDIR`, gated,
   *academic research only*) needs a token whose account accepted the terms —
   run with `env -u HF_TOKEN` to use the stored one. The pod's env token
   belongs to a different account.
10. **`val1.tar.gz` in LSDIR contains `X2`/`X3`/`X4` downscaled copies beside
    `HR`.** Without a path filter a pool silently builds from downscaled
    images. Train shards are flat HR.
11. **Check `completed` against `num_prompts`.** A run where 117 of 120
    requests failed produced a tidy 17 ms TTFT table. Gate added.
12. **`oc` credentials expire**; every pod probe fails at once when they do.

### About upscaling the images

Deliberate and validated. Every measured quantity is pixel-count driven, and
EC entry size is purely resolution (5,329 embeds × 7,168 B = 38.2 MB at
2048²). Measured: an upscaled 2048² and a native 2048² crop are **both 0.54 MB**
at JPEG q85, and the 96-image pool lands at 0.17 MB/MP against 0.13 for a
native crop. A mosaic alternative was 40% heavier than native, i.e. worse. Do
not "fix" this back to refusing upscaling.

---

## 8. Suggested next steps

1. Add the example-connector save counter (open bug 1) — cheap, and it makes
   the two connectors comparable.
2. Convert `EPD_CONFIGS` to the three-way payload mode and wire the six arms;
   arms 1–2 are `--topology single` with `--arms recompute` / `connector`.
3. Decide the region-vs-working-set question deliberately: run the reuse arms
   both with a region larger than the working set (all hits) and smaller
   (continuous eviction). They answer different questions; conflating them
   caused open bug 2.
4. Restart servers per load point, or report saves per point.
5. Then the fan-out story the owner is most interested in: `--images-per-request N`
   against N encoders (`--encoder-devices 0,0,0,0`), where the proxy issues one
   encode call per image so width caps per-request parallelism. Expect the
   curve to flatten at N = width; that flattening is the evidence the mechanism
   is real. MIG would make per-slice attribution clean but is not required —
   several encoders share a GPU with `--mm-encoder-only`.
6. `patches/sitecustomize.py` has never run. Exercise it with `--frag` before
   trusting any fragmentation claim.

## 9. Reproducing a run

```bash
# one arm, EPD, grid, 1 encoder + 1 decode
cd /vllm-workspace/bench && /vllm-workspace/venv-vllm/bin/python run_bench.py \
    --pod "" --workload-dir /vllm-workspace/bench/wl \
    --out-dir /vllm-workspace/bench_results/example-grid \
    --topology epd --epd-configs example-grid \
    --encoder-devices 0 --decode-gpu 1 \
    --request-rates inf --max-concurrency 1,4,8 --num-prompts 120 \
    --encoder-max-num-batched-tokens 65536

# the monolithic baseline
... --topology single --arms recompute --gpu 0 --max-concurrency 1,4,8
```

Always `--dry-run` first: it prints every launch command and the load command,
and has caught a port collision and a stale-code path that would each have
wasted a full run.
