# Rollout engine and sampling

In RL the rollout engine (vLLM) pauses, sleeps and wakes next to the training process, its sampling randomness must be auditable, prefix cache and speculative decoding affect whether the same token sequence is bitwise identical, and RL-only features such as R3 constrain the engine config. Section 6 is a debugging procedure: save one rollout and replay the old-logprob forward. Measurement bases and A/B methods are in [Performance and profiling](performance.md); BI kernels and cudagraph are in [BI kernels](bi-kernels.md).

## 1. Sampling seed collision

**Mechanism.** Each vLLM DP engine resets the global RNG with the same `seed` at the end of warmup. Requests without a seed sample from the global stream, so the n-th request samples the same tokens on every engine. When GRPO spreads the siblings of one prompt across DP engines, many samples in a group repeat: before the fix, Nemotron had only 64-70 distinct responses out of 256 and up to 43% of sibling pairs identical; after the fix, 256/256. The fraction of zero-advantage groups barely changes with the fix; zero advantage mostly comes from task difficulty, not seed collision.

**Fix.** Reset the global RNG by `(seed, data_parallel_index)`; index 0 is bitwise identical to the old behavior (vllm-project/vllm#59788, merged). Use `data_parallel_index`, not `data_parallel_rank`: dense (non-MoE) models run independent engines under DP, and `reconfigure_for_independent_dp_rank()` resets `data_parallel_rank` to 0 (`vllm/v1/engine/core.py`), so 4 engines get the same seed stream and n=4 gets worse. Nemotron is MoE and goes through `DPEngineCoreProc` with a correct rank, so its 1024/1024 verification on verl holds, but only the index version fixes dense models.

**How to accept.** In the rollout dump, count distinct samples per group and the fraction of fully identical sibling pairs. With different seeds the baseline is about 0.1% identical sibling pairs, all short answers of <=8 tokens (the model outputs `\boxed{}` directly), and about 1-3% share a prefix of >=16 tokens. Anything above this is RNG correlation.

**Do not do this.** verl `rollout.full_determinism=True` gives the n responses in a group the same seed. Do not enable it.

**Greedy cannot verify the fix.** Under greedy, n repeats are identical (Nemotron: 32 of 32), and the score is a multiple of 1/number of problems (AIME with 30 problems: multiples of 1/30; 13/30 -> 19-22/30). A greedy A/B before and after the seed fix shows no difference because greedy does not sample. Rising entropy and the three config curves overlapping are visible only on sampling runs.

## 2. Prefix cache semantics

**Premise.** "Enabling PC does not change the numerics" requires the same request to be bitwise identical on hit, miss, and PC off. For pure attention models KV block reuse guarantees this. For stateful layers (Mamba2 conv/ssm state) it depends on which states the engine copies.

**The Mamba2 pitfall.** vLLM's Mamba align mode copies only the conv and ssm states. Under BI=1, Mamba2 uses exact replay (ported from vllm#55905, which stores three extra chunk buffers x/dt/B) and needs 5 states, so BI=1 Mamba2 originally rejected PC. The fix copies every state in each layer's declared state list; after that, hit/miss/PC off are bitwise identical. Early versions had a convolution window alignment bug in state export (written aligned to the tail, read from the head).

**Cost.** PC under BI=0 has a measurable cost: in Mamba align mode each decode step adds a batch of eager launches, and an 8K prefill is split into two segments. The approach in upstream #58851 is the best.

**Confounders between the two arms.** BI=1 Mamba2 rejects PC while the BI=0 R3 AuxOutput forces PC on, so PC on/off is a confounder in the BI=0 vs BI=1 throughput comparison. List confounders first, then compare; see [paired acceptance](recipe-and-acceptance.md#paired-acceptance).

**Relation to cudagraph.** A PC hit changes the token count of a step. Piecewise cudagraph chooses between the graph and eager fallback by total token count vs `max_cudagraph_capture_size`. Each path is batch-invariant on its own, but the two are not bitwise equal. Under BI, pin `FULL_AND_PIECEWISE`; see [BI kernels](bi-kernels.md).

**Checks for stateful layers and caches.** Mamba/SSM must also save chunk boundary state and partial input buffers; check aligned/unaligned split, cross-chunk and restore separately. Cover indexing and capacity of long-context KV/page/workspace near the boundaries.

When testing prefill, confirm whether the prefix cache was hit; during diagnosis, disable or isolate it if needed so the target path executes. Verify the production cache's hit, reuse, and preemption/recompute paths separately. Do not make a comparison pass by clearing the cache and then claim the original lifecycle is fixed.

For performance fixes on the cache path, use bitwise equality with "cache off" as the gold standard: hit positions outside chunk boundaries, every DP rank, recomputed hits after forced preemption, and chunks truncated by the scheduling budget; confirm in the trace that the target branch executed. Then add a hook that checks the device-side state index against the host prediction at every step, plus a per-problem quality evaluation. Two fixes that each pass alone may silently break when stacked; the combination must pass the gate on its own.

## 3. MTP / speculative decoding

- The Nemotron checkpoint has `num_nextn_predict_layers: 1`, and the MTP block is not quantized (bf16). It does not take part in training; the training side must explicitly skip its mapping ([training forward](training-forward.md)).
- Usable with BI=0. BI=1 + MTP requires the logprob from speculative verification to be bitwise identical to the full forward; without verifying this you cannot claim true on-policy.
- If used, use it in both arms; otherwise neither the throughput nor the logprob comparison holds.

## 4. R3 (routed experts replay) feature constraints

- **What is returned.** One `(num_moe_layers, top_k)` row of uint8 routed expert ids per token (Nemotron: 52 layers x 6 = 312 B), returned to the trainer with the request's CompletionOutput for route replay. 0 is a valid expert id; check distinctness, not non-zero.
- **PC must be on.** The R3 AuxOutput is keyed by KV block hash; with PC off there is no key. This is why the BI=0 arm in §2 is forced to enable PC; the keys-only form (PR 57968) does not lift it.
- **Why per-step return is expensive.** 312 B exceeds the 256 B msgpack zero-copy threshold, so each request's row per step becomes a separate frame, and the frontend accumulates millions of small objects per request before concatenating them at the end. The overhead is on the engine->frontend return path, not in the capture hook. The fix returns once at request finish (finish-only) and packs each step into one ndarray on worker->engine. `output_processor` already exposed `routed_experts` only at finish, so semantics are unchanged.
- **There is another floor below R3.** msgpack unpacking of three small ndarrays per token for sampling logprobs is a fixed frontend cost (about 25-30 µs/token), larger than everything else left after R3 is fixed. BI=1 does not see it when the engine is the bottleneck; BI=0 has a faster engine and hits it. Two arms tying on e2e is often due to this, not the kernel. Check the frontend asyncio busy fraction before comparing the arms ([Performance and profiling](performance.md)).
- **What does not change.** R3 adds no GPU->CPU sync, async scheduling is kept, capture writes into a statically preallocated buffer, and cudagraph is unaffected. flashinfer_cutedsl and Humming are both modular MoE (they receive the router's top-k), so moving R3 into the kernel saves nothing.
- **Limits of the equivalence check.** BI=1 cannot run the equivalence check (Mamba2 BI rejects PC while AuxOutput forces PC, until the §2 fix lands). BI=0 compares sha with strictly serial single-request greedy, but BI=0 routing is itself not batch-invariant, and an unpatched baseline also repeatedly shows partial inequality against the same reference. So first measure the drift with several baseline-vs-baseline runs; the patch counts as equivalent only if it fully equals one clean reference. Bitwise comparison of concurrent greedy dumps is not evidence.

## 5. Lifecycle and GPU memory ledger

**Three pause modes** (`EngineCore.pause_scheduler`): `abort` marks all running requests `FINISHED_ABORTED`, queues new requests, and optionally clears the cache; `wait` queues new requests but keeps stepping until running requests finish (not supported by the inproc engine); `keep` pauses everything, and the Future completes when the output queue is empty. In all three modes the Future returned by pause is the barrier; until it completes, workers still step. In `abort` mode the client receives aborted outputs and must resend them. Requests that arrive during pause are queued and released at resume, using the **new** weights. These are "orphan requests" (pause issued before the rollout tail finishes, or the agent loop sending during pause), and their tokens belong to no single policy version. Before each step's pause, confirm in-flight is 0; after resume, confirm the queue is empty. The HTTP endpoints `/sleep`, `/wake_up`, `/is_sleeping` and pause/resume are dev endpoints and need `VLLM_SERVER_DEV_MODE=1`, otherwise 404.

**Three sleep levels** (`EngineCore.sleep`, `Worker.sleep`):

| level | What it does | GPU memory | Host memory |
| --- | --- | --- | --- |
| 0 | Only pauses scheduling | Unchanged | Unchanged |
| 1 | Offloads weights to CPU, drops the KV cache, clears the prefix cache | Frees KV and weights | One weight backup per rank (DS4.1 78.61 GiB); the pinned cache is not returned after wake, see [Checkpoint and resume](checkpoint-resume.md) |
| 2 | Discards all GPU memory, saves only `named_buffers` to CPU | All freed | Small |

After L2, `wake(weights)` only reallocates GPU memory; the contents are uninitialized. "Reserved is very high but still no GPU memory" after wake is an allocator boundary problem: PyTorch's reserved-but-unallocated blocks cannot be used by another allocation path such as CuMem; see [Checkpoint and resume](checkpoint-resume.md#3-reserved-is-high-so-why-can-wake-still-have-no-gpu-memory).

**Set gpu_memory_utilization by training mode.** The training process does not fully release GPU memory during rollout. Write the value into the recipe per training mode; check the resolved value inside the container and the KV budget in the vLLM log, not the outer setting. Measured values and rules per mode are in [Checkpoint and resume](checkpoint-resume.md#5-gpu-memory-set-gpu_memory_utilization-per-mode-when-training-and-rollout-are-colocated).

**Metrics are cumulative.** `vllm:*_total` counts since process start; when one engine runs several phases, always use `after − before`. Why the first phase happens to be right and how to cross-check are in [Performance and profiling](performance.md#metrics-are-cumulative-values).

**DP traffic lands on rank 0.** When the Rust frontend sends one request at a time, all of them land on rank 0 (engine selection is `min_by_key(routing_score)`, which takes the first on equal load). "No events on rank 1" then means no traffic, not broken routing. Use `X-data-parallel-rank` to pin requests to each rank in turn. The two frontends interpret this header differently: Python in hybrid mode validates it as a local index in `[0, dp_local_size)`, Rust uses the global rank; do not send it to a single-rank service. Python hybrid/external LB with a dense model does not start (`DPAsyncMPClient` asserts `stats_update_address`, upstream #32252); test Python hybrid with an MoE model.

---

## 6. Save one rollout and replay the old-logprob forward

### When to use

For a difference that shows up once across many RL steps, save the failing input batch and matching policy early. Rerunning the rollout changes tokens, scheduling, lengths, cache, packing and later training data at once, hiding whether an intervention worked.

1. Save the framework batch and the policy identity after the real rollout ends and before the old-logprob forward; append the trainer scores after the comparison.
2. Verify that the save hook executed, that sequence / valid token counts match the log, and that the data reads back.
3. Use the framework's existing training-only/repeat entry point to read the same batch; restore its weights.
4. Return after old-logprob; forbid the optimizer step and regeneration; record the skipped stages and the weight fingerprints before and after.
5. Run A/A first, then change one layout factor. If A/A itself is unstable, first check randomness, mutable buffers, async streams and runtime state.

### Minimal matrix

| Variant | Change | Answers |
| --- | --- | --- |
| A | Original packing | Can the frozen input reproduce it |
| B | Exact repeat of A | Is it run-to-run nondeterministic |
| C | Sequence permutation/packing | Does it depend on layout |
| D | micro-batch size or target position | Does it depend on call shape/neighbors |
| E | Separate process or equivalent node | Does it depend on initialization/device state |

Run first the variants that distinguish the current hypotheses; the full matrix is not needed every time. After reordering, restore and compare by UID and response token position; the physical row no longer represents the same request. C failing does not prove the packing algorithm wrong; it may trigger a CP, page, tile or expert boundary error.

### Keep the trigger condition when shrinking the input

Keep the target sample and its necessary neighbors first, then reduce length and request count. After every reduction, rerun the criterion to confirm the failure remains. A CP problem may depend on the remainder from the end of the previous request to the start of the rank; a Graph problem may depend on short fillers and the capture bucket. Keeping only the failing request often makes the problem disappear.

Freezing the rollout and the weights does not freeze all runtime state. Training-mode dropout, FP8 amax history, RNG, route replay, caches and workspace must each be kept or reset according to the failure condition. A fresh eval forward is not a substitute for the original old-logprob path.

A resident model saves checkpoint loading, but state must not leak across variants. First verify A/A between a new process and a reused process; when native libraries, autotune or compile config are involved, restart the corresponding workers.

### Leaving the replay stage

Once the original reproducer is closed, return to normal generation, real backward/update and the next rollout step. Training-only success supports only the frozen-policy forward conclusion. If the error appears in backward or in long-running state, save that state and build a minimal reproducer for that stage.

This rollout replay is a debugging procedure. Nemotron's SSD exact replay is an algorithm in the model implementation that recomputes the partial chunk to match prefill. Do not call them by the same name.
