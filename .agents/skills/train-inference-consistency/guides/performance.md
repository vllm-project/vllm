# Performance and profiling

Record performance gates and correctness gates independently: low throughput does not mean the numerics are wrong, and correct numerics do not mean performance meets the target. BI kernel choices and costs are in [BI kernels](bi-kernels.md); the statistical protocol for paired acceptance is in [Pairing and computation](recipe-and-acceptance.md#pairing-and-computation).

## 1. Workload and measurement basis

### Three throughput measurement bases

- **Per-request tok/s**: 1/TPOT; decode latency.
- **Engine total tok/s**: completed tokens of all concurrent requests / common timing window. With multiple engines use the full common window and total completed tokens, not a sum of per-engine throughput; with unsynchronized node clocks, do not compute the window from timestamps.
- **RL one-step gen time (makespan)**: one batch of rollouts from the first request in to the last request out. This is what RL cares about; long tails and scheduling can move it opposite to the other two.

State which basis you report; compare ratios only on the same basis.

### Translate the rollout target into actual kernel workload

"batch=128, n=8, input 2K, output 8K" must at least specify:

- Is 128 prompts, expanded requests, or per-engine concurrency? 128 prompts with 8 siblings each is 1024 requests.
- Is 2K/8K the maximum or a fixed length? Record input token mean/max/percentiles, total output, and completed requests.
- `max_num_seqs` limits simultaneously active sequences; `max_num_batched_tokens` is the token budget per scheduling pass; neither equals the total request count.
- Requests per DP engine, prefill chunk size, and the active decode batch over time; a global 128 is not each rank's kernel M.
- Prefix cache cold, warm, or naturally hit; the n siblings of a prompt affect hits and prefill.
- Natural EOS vs ignore EOS / fixed length, sampling parameters, seed, request order, maximum model length, scheduling policy.

### replay bench: equal-workload replay of one RL step

Use the same request count and prompt×n as one RL step, take each request's output length from one real rollout, replay with ignore EOS at fixed lengths, and measure makespan. Measure the repeat noise of one config first, then compare arms; draw no conclusion from a difference smaller than the noise. With the same length table, token count, request count and long tail are identical, so the difference belongs to the execution path. Fixed output suits equal-workload comparison, not quality.

In natural generation, different numerical paths or scheduling can change token counts even with the same weights and seed; record output token receipts and add a frozen-token replay if needed. A short 2K+64 profile can locate startup/local hotspots but **cannot** represent the long-term decode, cache/page pressure, scheduling tail or full rollout speed of 2K+8K. Use different experiment IDs for the short profile, the full stress test and the training E2E.

### Metrics are cumulative values

When one engine runs multiple stages (concurrency or parameter sweep), `vllm:iteration_tokens_total_*`, `vllm:spec_decode_num_*` and `vllm:num_preemptions_total` in `/metrics` are cumulative since process start. Always compute per-stage cost as `after - before`, and note in the script that the first stage is right only because before is all zero, which makes the method look correct. Cross-check the prefill amount with an independent client-side basis (prompt_tokens/cached_tokens in the request records); if the two disagree, the counter basis is wrong.

When the verification method itself fails (a regex that never matches, a hardcoded version number, an unsubtracted cumulative value), the numbers look plausible but measure something else. Reconcile every new script's first numbers against an independent basis.

## 2. Fair A/B

### Do not mix the two kinds of A/B

- **Causal comparison of the BI switch**: same software/model/scheduling/cache, only BI changes. If the switch selects a different kernel automatically, record the actual branch and add no other optimizations.
- **Each side's best available config**: BI=0 uses the strong default / verified best path for that platform and model; BI=1 uses the fastest path that meets the agreed consistency. Backends may differ; disclose every difference.

Do not force BI=0 onto FA2, a specific MoE, non-default attention, or Graph off to shrink the gap. Do not call an experiment that locally disables switches inside BI=1 "native BI=1". A replaced path is a separately named candidate that passes the full contract gate. Give common optimizations (cutting shared launches, enabling a supported network path) to both sides and report the common gain and the reduction in BI overhead separately. BI=1 faster than BI=0 is not automatically fake, but it triggers an audit of config, workload, cache and timing.

### What to fix

Weights, input, generated tokens/lengths, output cap, concurrency, nodes, software, warmup and timing boundaries. If output receipts and workload differ, report only each side's own throughput. Separate the fresh-process first rollout from warm steady state.

**Same fabric**: before comparing communication bottlenecks, confirm both arms use the same transport; see [Same-fabric fairness](debugging-mismatch.md#same-fabric-fairness). **Same node**: small gains need same-node pairing, crossed order and multiple rounds; counterbalance across node pools and do not pick the fastest cross-pool combination. **JIT pollution**: a JIT kernel specialized by token count recompiles for every new prefill size under BI=0; verify zero compile lines during the bench. **BI=0 silently running the BI kernel**: a fork whose gate only checks whether the `.so` exists makes BI=0 run the BI kernel; grep `vllm::batch_invariant::` in nsys.

### Report speed switches and numerical switches separately

One switch (PC, cudagraph, BI) changes both the numerical path and throughput. Before reporting "probs_diff with BI on/off" or "throughput with BI on/off", list the confounders that change with it, rule out or fix each, then compare; see [Paired acceptance](recipe-and-acceptance.md#paired-acceptance).

### Computation and reporting

```text
output tokens/ms = completed output tokens / (wall_seconds * 1000)
BI throughput deficit = 1 - throughput_BI1 / throughput_BI0
BI time overhead      = time_BI1 / time_BI0 - 1    # matched work only
```

Latency reduction `1-new/old` and throughput gain `old/new-1` differ; 16.35% lower throughput is not 16.35% more time. Report numerator, denominator, request / completed count, raw receipts, config differences, sample count and variation; take the median over pairs and report single pairs separately. Gains of A and B do not add; measure the combination, and do not claim it beats the best single item from a few observations. List skipping the old-logprob forward separately, only when the RL algorithm allows it; it is not a gain of the BI kernel. Every ratio must be recomputable from explicit artifacts.

## 3. Breakdown and ablation

### Two stages: remove structural defects first, then fine-tune

Once proxy inference consistency passes, check feasibility: sweep token row counts on real operator shapes, count launches and host gaps, look for per-token launches, synchronous readbacks, repeated copies and abnormal workspace, and confirm the target Graph captures/replays. After backward is wired in, measure training forward and forward+backward (by token count) on the main topologies fixed at C1 against the native training path on the same build, and check them against the C1 budget, so a scheme fast at decode but slow in training/prefill is not chosen. After full consistency, fine-tune by E2E profile -> micro benchmark (screen high-share shapes, then sweep all target shapes and boundaries; all must be correct; weight performance by real call frequency) -> multi-round paired E2E on the same workload. New or replacement kernels do not use Triton; use CuTe DSL, CUDA/CUTLASS or an existing implementation; historical Triton data is only a comparison. Final delivery still needs full acceptance, see [validation](../workflow/validation.md). Fix a structural problem first and re-check the numerics; do not turn this stage into endless tuning.

### Break down the overhead first

Time model load/JIT, rollout prefill/decode, old-logprob, backward/update and offload/wake separately. Kernel times may overlap and do not add up to wall time. Keep the nsys window, rank/GPU normalization, Graph node tracing and sampling interval consistent.

Profile the effective config and actual kernels, not just the BI switch. In DS4 the same quantization `.so` loads under both BI0 and BI1, but different MoE layouts select different schedulers. For an NVFP4 MoE model, account for dense linear, expert GEMM, expert routing/indexing, all-to-all and attention separately; one vendor name does not describe the path (one Nemotron run mixed FlashInfer FP8 linear, Humming NVFP4 dense GEMM, Humming indexed experts and FlashInfer NVLink one-sided all-to-all). A run that fails on a backend guard (a BI=0 trial expected Humming but got FlashInfer CuTe DSL) calls for re-auditing the backend and build path and does not enter the performance denominator.

### Collect enough attribution information in one profile

1. Verify the actual serving config and input/output statistics. For inference-only use vLLM serve or an existing client, not training; use the mp/Ray executor actually needed.
2. Add low-overhead, off-by-default NVTX at semantic module boundaries: attention, indexer/Top-K, index preparation, MLA, MHC, shared/routed MoE, norm, linear, sampling, communication. Do not apply the DS4 module list to dense models.
3. Use nsys first for the timeline and hotspots, with CUDA Graph node tracing on; graph launches without node details do not support per-kernel comparison.
4. Count calls, GPU time, CPU launch/sync, allocation and shape per path; aggregate by rank, phase and module.
5. Pick representative prefill, steady decode, mixed and tail windows, then ncu a few kernels/shapes for bandwidth, occupancy, memory access and compute; ncu-perturbed wall time is not throughput.
6. Measure throughput on the same full workload without profiler or heavy dumps, and record whether profile and timing used the same code and config.

A kernel name is not a module identity: copy/fill/cast, `index_select`, generic elementwise, GEMM, and different shapes of one template appear in many layers. NVTX/launch correlation and shape are needed to tell "the indexer's gather" from other gathers. Parent/child NVTX inclusive times do not add; the sum of concurrent kernel GPU durations is not wall time; label multi-GPU sum, per-GPU mean and critical-path time separately.

When a wrapper/overlay also rewrites the kernel selector, confirm who launched a kernel with **three layers of evidence**: per-worker class, object count and import source at initialization; input shape, module name and dispatch at the real forward callsite; launch correlation and GPU kernel in the final profiler. Overridden linear objects or a Humming shape do not prove what launched at replay, and overlay env vars set where the worker does not inherit the entry point falsely suggest the candidate is enabled. Graph may hide ordinary Python CPU op events, so a `cpu_op` count of 0 alone does not prove the callsite did not execute. When evidence conflicts, run a minimal probe that keeps the numerical path; do not fall back to an unverified default operator for throughput.

For a Roofline bound, use the peak compute of the actual dtype and the bandwidth of the memory level studied; L2 / shared-memory / non-Tensor-Core kernels cannot borrow the HBM / Tensor Core bound, and do not infer the metric from chart coordinates.

### Pin "where is it slow" to a process and a call stack

The frontend and control plane are in scope. Split the makespan difference by process: py-spy the engine, worker and frontend once each and see which function holds the main-loop time; use asyncio loop-lag to see whether the frontend is saturated. With R3, most of the frontend main loop is in `output_handler`; without R3 the frontend is almost idle, which moved suspicion from the GPU to the frontend object count. Optimize only parts with a large share; GPU utilization is not evidence of completion.

### Single-operator replay and synthetic micro benchmark

A synthetic shape matrix proves algorithm boundaries; real rollout replay supplies score distribution, ties, masks, non-contiguous layouts, cold/hot buffers and call frequency. Weight by sampled shapes/frequencies and keep high percentiles and rare boundaries; report decode small M and prefill large M separately. Time with and without workspace allocation; if the original allocates per call and the candidate preallocates, do not compare only the kernel body. A micro A/B fixes input and resources and checks the candidate bitwise against the baseline; change one thing at a time, covering at least the batch sizes actually used (1/4/32/128), different neighbor compositions, prefill/decode, Graph, mask/tie/boundaries; mark shapes not run as not covered. When Triton is forbidden, a "temporary benchmark" does not bypass that.

### Ablation method: make A/B really differ in only one variable

These rules apply to intermittent numerical leaks (the batch-invariance kind) and to performance attribution; the examples come from the DS4 batch-invariance investigation. Process rules are in [Experiment discipline](../practices/evidence-and-experiments.md#experiment-discipline).

1. **A single round of 0-diff does not count.** For a leak with a hit rate below one half, a single all-green round has probability above one half. Re-check a statistical conclusion with a deterministic reproducer, especially "turned clean"; 0-diff is necessary but not sufficient.
2. **For "turning off X fixes it", first check whether X's switch also changes another branch.** Example: with cudagraph mode NONE, the budget split function also took a different branch, so the eager experiment moved two variables. Hidden coupling cannot be attributed to the switch.
3. **A switch that changes speed also changes the scheduling composition.** Eager is several times slower than graphs, so batch composition changes. Before a statistical comparison, pin each step's batch composition with offline manual steps; otherwise "clean when off" may just mean it was not hit. Turning off the cache is the same: a prefill that never ran before now runs.
4. **The target path must actually be exercised.** A long-prompt filler looks "heavier", but the step exceeds `max_cudagraph_capture_size` and the graph does not run, so identical results across settings are an illusion. A negative control records the target branch's execution count; zero hits is marked VACUOUS and does not count as verification.

Before localizing, draw a 2×2 (or higher-dimensional) matrix to separate the variables, then choose which cells to run.

### Ablations must be able to identify interactions

With optimizations A and B, measure `base / A / B / A+B`; with network/Graph/cache switches, add the corresponding controls instead of running the full combination mechanically. Reuse A+B data only under identical weights, input, software and timing. The first round after restoring checkpoint N is usually step N+1, and adjacent names do not prove the same policy: fix the checkpoint content digest and export fingerprint, and do not mix BI0/BI1 or R3/no-R3 branches. For rollout-only timing, exit cleanly after completion without optimizer/save, and record that all requests completed. Top-K and index post-processing can usually be ablated orthogonally, but shared buffers, layout or Graph create interactions, so re-accept the combination.

### Segment-by-segment isolation: add the total difference back up

Cut a long path into arms at boundaries, each enabled up to one segment, and measure what each segment adds:

| Arm | Enabled up to | What it answers |
| --- | --- | --- |
| All off | No capture | Baseline |
| A0 | Capture only (hook on, nothing lands on the data plane) | Cost of the capture kernel/sync point |
| A | Capture + worker data plane + packing, not sent to the scheduler | Worker-side processing cost |
| E | worker→engine IPC on, scheduler discards | Inter-process transfer cost |
| B | No capture, return path only (fake data) | Return path cost, independent of capture |
| All on | Current path | Total difference |

Arm differences should roughly sum to all-on minus all-off; otherwise there is an interaction or a dirty cut. This localized the Nemotron-H R3 return path: A0 and A are almost level with baseline, E adds a little, B is close to all on, so about 80% of the overhead is on the per-step engine→frontend return path, not the capture hook.

Measure same-config repeat noise first and ignore arm differences below it.

### Optimization priorities and correctness gates

1. Remove idle/repeated work: launch grids mismatched with the real workload, repeated moves, invalid tokens.
2. Reduce host scheduling/sync and kernel launches; restore a usable Graph with the output contract unchanged.
3. Fuse boundaries such as gather/dequant/index post-processing, keeping the original rounding, traversal order and padding/sentinel.
4. Only then change the kernel algorithm, chunking or collectives; these may change the reduction order and need new equivalence evidence.

Example: a quantization kernel launches 1024 blocks per expert while decode has few valid rows. Shrink the grid and the stride together; under Graph capture use a host-known capacity bound, not the device token count. If dequant/gather are element-wise independent, re-chunking may keep bit-level equivalence; fusing into sparse attention must keep the scale rounding and KV traversal/reduction order.

Each candidate first runs a same-input kernel comparison (real dtype/shape, non-contiguous or masked layout, empty experts, tail, extreme values, ties, reused dirty buffers, Graph replay), then the fixed-token engine gates, then actual rollout/training. Formatting checks do not replace compiling and executing; rerun measurements on the final code, since a JIT refactor that looks behavior-preserving may fail to compile. A candidate with no gain or a regression keeps its conclusion and is withdrawn from the default path.

### Delivery

Each item lists the actual kernel/path, correctness before and after, micro latency, rollout and full-step results, independent sample count, Graph/topology scope, raw receipts and unsupported configs. State whether quality evaluation and long-term stability are done; kernel speedup does not replace an end-to-end conclusion.

## 4. Lever results table

Measured switches: directions and dead ends not to retry.

| Model / arm | Winners | Dead ends (measured, do not retry) | Structural floor |
| --- | --- | --- | --- |
| DS4-Flash GB200 DP8/EP8 BI=0 | FlashInfer trtllm MoE (`--moe-backend auto` + new FlashInfer); `cudagraph_mode=FULL_AND_PIECEWISE` | deepep_low_latency, `--api-server-count>1`, NCCL_ALGO/PROTO, max-num-batched-tokens 4k/8k, linear backend fitrt/cudnn (cutlass crashes), DBO (forces the V1 runner, which does not accept PIECEWISE) | MoE decode is weight-bound (gemm1 near HBM roofline); DBO rereads weights and loses; overlap capped at the shared experts' few µs |
| DS4-Flash BI=1 | Deterministic top-k + fusion, two PRs together (each alone has almost no effect); DeepEP LL over MNNVL (`VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1` + nvl8 overlay) is the biggest lever; after that the shared experts aux stream helps only under MNNVL | DBO, TP2/DP4, TP4/DP2, the skinny bi_gemm, async-scheduling (already default), splitting the shared MLP to fill the combine window | MoE backend locked to DEEPGEMM (the only BI fp8 MoE) -> a2a locked to deepep LL; per-layer dispatch+combine is serial; cross-layer overlap breaks BI; finally the TTFT floor |
| Nemotron-H BI=0 | PC off (when R3 forces PC on, upstream #58851 is best); R3 finish-only + per-step packed IPC (overlay), makespan returns close to the R3-off floor; then packing per-step logprobs into a list for the frontend (overlay) unsaturates the frontend and BI=0 is 16% faster than BI=1 | Raising the zero-copy threshold (no effect); `max_num_seqs` 256 (slower); PR 57968 keys-only as it stands | PC on (Mamba align mode) has a fixed cost: extra eager launches per decode step, 8K prefill split into two segments |
| Nemotron-H 256K training update (both arms) | BI=1 per-GPU per-micro-batch token cap 64K -> 128K: update only about 5% faster per token (13.1 -> 12.4 s / million tokens), actor reserved 99 -> 168 GiB (184 GiB per GPU, cap 170); small gain, thin margin | Less recompute (some layers, only Mamba layers, none): both arms OOM at 32K-64K per GPU. Per-GPU token cap for BI=0 (no difference across 32K-192K): with CP>1 the recipe disables dynamic batching, each micro-batch is one sequence, and the parameter has no effect | Update is 14-18% of the step, rollout about 80% (sandbox execution, slowest trajectories). Compute the share before tuning; confirm in the resolved config that a parameter takes effect |

BI kernel choice: [BI kernels](bi-kernels.md).
