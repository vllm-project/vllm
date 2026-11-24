# Experimental NIXL GPU Device API DCP

The opt-in prototype implements MLA DCP output/LSE exchange on two or more
GPUs using direct stores to mapped peers and NIXL device puts to unmapped peers.
It targets both CUDA IPC mappings and cross-node mlx5 GPU-initiated RDMA, with
three communication kernel launches per invocation. NIXL execution is disabled
by default; the device operators are included in CUDA source builds.

## Build and select

CUDA builds fetch NIXL 1.5.2 device headers and UCX
`65e342a82dd2609a28c8c2b8c45ed7fe2cb33fa0`, matching that NIXL release's wheel
build. The UCX GPUNetIO submodule is pinned by the same revision. Install
`rdma-core-devel` (RPM) or `libibverbs-dev` (Debian/Ubuntu) in the build
environment. No separate NIXL build flag is needed. The CUDA source includes
both mapped-memory and GPU-initiated RDMA device support.

The runtime needs `nixl==1.5.2` and the corresponding CUDA wheel (for example,
`nixl-cu13==1.5.2` for CUDA 13). NIXL/UCX runtime libraries come from that
package; vLLM does not link a second UCX runtime. Avoid overriding these with
an unrelated `LD_LIBRARY_PATH` or plugin directory. Device setup verifies
`nixl._bindings.HAVE_UCX_GPU_DEVICE_API`, mapped accessibility and supported
RDMA queue geometry. Importing NIXL successfully alone does not qualify RDMA
on a particular device/driver configuration.

At runtime use `--dcp-comm-backend a2a`, a DCP size greater than 1, PCP size 1, an MLA
model, and FP16 or BF16 attention output. Set the flags explicitly for comparisons:

| Path | `VLLM_USE_DIRECT_DCP_A2A` | `VLLM_USE_NIXL_DCP_A2A` |
| --- | --- | --- |
| NCCL | 0 | 0 |
| Direct P2P | 1 | 0 |
| NIXL GPU Device API | 0 | 1 |

NIXL defaults to 0. When enabled, it overrides automatic direct selection if
the direct flag is unset. Explicitly enabling both raises an error. Missing SDK
support, unsupported configurations and capacity misses fail instead of falling
back to another transport. For an isolated A2A comparison, set
`VLLM_USE_DIRECT_DCP_Q_GATHER=0` and `VLLM_USE_DIRECT_DCP_KV_GATHER=0` in every arm.

Set the NIXL flag before invoking `vllm serve`. Validation runs on the assembled
engine configuration after model-specific DCP defaults are resolved, so CLI
default discovery does not require temporarily disabling the flag.

For example, with the source-built runtime and a local DeepSeek-V2-Lite snapshot:

```bash
VLLM_USE_DIRECT_DCP_A2A=0 VLLM_USE_NIXL_DCP_A2A=1 \
  .venv/bin/vllm serve /path/to/DeepSeek-V2-Lite --dtype bfloat16 \
  --tensor-parallel-size 2 --decode-context-parallel-size 2 \
  --prefill-context-parallel-size 1 --dcp-comm-backend a2a \
  --trust-remote-code --enable-expert-parallel --max-model-len 163840 \
  --kv-cache-memory-bytes 51539607552
```

This example uses two local GPUs. For cross-node placement, follow the
[multiprocessing deployment instructions](../serving/parallelism_scaling.md#running-vllm-with-multiprocessing)
and set the NIXL/direct flags consistently on every worker. Use a working GPU
RDMA device transport and ensure the DCP group spans hosts. For example, two
hosts with two visible GPUs each use TP=DCP=4 and PP=1, with `--nnodes 2`,
`--node-rank 0/1`, a shared `--master-addr`, and `--headless` on the second host.
Confirm `nixl_put` and `rc_mlx5_gda` for remote peers in the workspace logs;
mapped local peers should report `mapped_p2p`.

For TP groups spanning separate NVLink domains, also set
`VLLM_ALLREDUCE_USE_SYMM_MEM=0`. This disables the separate PyTorch symmetric-memory
all-reduce initialization, which can stall on this topology even with
`--disable-custom-all-reduce`. It does not disable NIXL DCP's mapped-peer copies.
Use the same all-reduce setting in the NCCL and NIXL comparison arms.

## Projection and workspace sizing

NCCL and NIXL share the MLA projection-order helper. Eligible
batches with at least eight query tokens project before exchange; smaller
batches combine latent outputs before projecting local heads. Dimensions and
eligibility come from the model's absorbed weights, not a model-name check.
The common MLA path and KimiK3's specialized path initialize their workspaces
after weight preparation and before CUDA graph capture.

The existing symmetric P2P backend retains combine-then-project and its
original workspace allocation. NIXL adopts NCCL's eight-query-token heuristic;
the threshold has not been tuned for mixed NVLink/RDMA groups. Query-token count
can differ from request batch size when using speculative decoding.

NIXL reserves a full-capacity projected workspace and a small latent workspace
when projection first is available. Otherwise it retains full latent capacity.
Each workspace has bounded send staging, two receive slots and independent
microbatch state. Logged workspace bytes exclude projection weights,
intermediate outputs, CUDA graph pools and NIXL/UCX internal resources.

## Protocol and lifetime

Setup allocates fixed registered send storage and one registered receive
allocation per workspace. Each ubatch has two receive payload slots, one epoch
and per-source/channel readiness counters. Metadata and layout are exchanged
through the CPU process group. A device probe accepts CUDA IPC or mlx5 GDA
endpoints and rejects unsupported transports, CPU-proxy paths, aliased RDMA
queues and queues too small for the configured batch. That probe also caches
`nixlGetPtr` for each remote receive allocation in a persistent GPU tensor.
A non-null pointer enables direct writes; a null pointer retains NIXL puts.
This tests mapped accessibility, not hostname or NVLink connectivity. There is
no additional PyTorch symmetric-memory allocation or rendezvous. Logs report
hosts, device transports, per-peer copy paths (`self`, `mapped_p2p`, `nixl_put`)
and workspace bytes; that byte count excludes NIXL/UCX's own queue allocations.

| Launch | Work |
| --- | --- |
| 1 | Advance the GPU epoch and expected counters for active channels. |
| 2 | Each CTA owns one destination/channel, packs directly into mapped peer memory or stages tiles for NIXL puts, then publishes readiness. |
| 3 | Acquire each source's channel counter and combine output/LSE in FP32. |

The default is four channels per peer and a 64-KiB tile target. Small inputs
activate fewer channels. Packing supports strided inputs, converts LSE to FP32
and masks empty/padded rows in the same kernel. For mapped peers, those packing
stores go directly to the peer's receive slot, bypassing send staging and NIXL
put submission. Every producing thread performs a system fence, then the CTA
synchronizes before its leader publishes readiness with a system-scope release
store to the peer's counter. Self transfers use the same direct-write path.

Unmapped peers retain the registered send-buffer path: every producer publishes
its stores before the posting warp exposes them to the NIC. Batches contain at
most four puts, ending in a nondeferred put whose completion is polled. The
ready atomic uses the same memory-view element and channel as the payload.
Both paths share the receive layout, counters and acquire-wait/combine kernel.
The merge retains the NCCL A2A path's source order and exp/log formulation,
including both LSE bases and optional returned LSE.

Payload slots alternate with the epoch. There is no consumed-acknowledgement
kernel: before a sender can write generation `g+2` over slot `g`, it must have
combined generation `g+1` from every rank. Each peer can send `g+1` only after
finishing its own combine of `g`. This argument requires ordered all-rank
participation and completion of the previous combine before the next dispatch
on the same ubatch. It does not permit independent asynchronous sends.

Each channel reuses at most four registered send tile slots after the existing
batch completion and block synchronization. Receive storage retains the full
payload for two generations; it cannot be reused merely because a put completed.
Outgoing atomic operations also complete before the send kernel returns.
Application waits and completion polls trap after 30 seconds; a backend call
that blocks internally still requires an external job/process timeout.
Communication errors are fatal to the worker.

The steady-state path has no application host NIXL transfer/progress calls. Epochs advance
on the GPU during eager execution and every graph replay; there is no captured
host-side generation constant. Ubatch slots are independent. Calls on the same
slot must remain stream ordered, and ranks must issue matching shapes, channel
configuration, ubatch order and collectives.
Pointers and registrations are never resized while graphs can reference them.
Model-parallel teardown synchronizes GPU work and peers before releasing views
and registrations, then waits for all peers to release their views before the
owning tensors can be freed. Graphs must not be replayed after teardown.

## Validation and limits

There is no NIXL-specific upper bound on DCP group size. The usual TP/DCP and
attention-head divisibility constraints still apply, as do transport, queue and
workspace capacity checks. Configuration acceptance does not qualify a topology.

The revised SDK, shared projection and bounded staging implementation passed
H100 functional checks with NIXL 1.5.2, its matching UCX release build,
Torch 2.14.0+cu130 and NVIDIA driver 580.95.05:

| Placement | DCP size | Qualified paths |
| --- | --- | --- |
| One node | 2 and 8 | Mapped peers, including untouched send-staging sentinels. |
| Two nodes, one GPU each | 2 | GPU-initiated RDMA. |
| Two nodes, two GPUs each | 4 | One mapped peer and two RDMA peers per rank. |
| Two nodes, eight GPUs each | 16 | Seven mapped peers and eight RDMA peers per rank. |

Checks include exact NCCL payload/merge comparisons and 1,024 changing-input
CUDA graph replays per native case. Stress tests record samples from every
replay on the GPU and check them after the loop, without per-replay NCCL
rendezvous. Cross-node checks also cover delayed consumers and verify the
three-launch protocol on every rank. Shared-projection tests exercise both
projection orders, empty shards, bounded staging reuse, graph replay and
non-power-of-two projected widths. The 16-rank test also covers the full
16,384-token projected workspace capacity with strided inputs.

Bounded DeepSeek-V2-Lite BF16 model checks passed at TP=DCP2 locally (eager
and graph execution), TP=DCP2 across hosts, and mixed TP=DCP4. Batches of
1, 7 and 8 requests were queued completely before stepping the scheduler,
with 32 generated tokens per request and two repetitions. NIXL and NCCL
produced identical tokens and saved logprobs; local symmetric P2P passed the
existing token/logprob bounds. Worker evidence confirms projection preparation
and the small latent/full projected workspace layouts. These are bounded
correctness checks, not model-wide accuracy or performance measurements.
They do not establish invariance to HTTP arrival timing or batch composition.

Use separate compilation caches when changing topology: a cached local
all-reduce fusion may be invalid for an RDMA-only process group. Disable the
separate symmetric-memory all-reduce initializer in both arms as described
above. Bounded DeepSeek-V3 FP8 model checks also passed at mixed TP=DCP16:
NIXL matched same-order NCCL tokens and saved logprobs for nine prompts and
288 generation steps, with exact restoration of the original NCCL path.
Changing projection order produced BF16 differences that failed the original
cross-order bounds; these checks do not establish task-accuracy equivalence.
No performance improvement is established for the revised implementation; its
full-model projection-order performance comparison remains incomplete.
KimiK3/GLM model runs and GPU generations other than H100 remain unqualified.
SM100 compilation passed, but is not Blackwell execution.

The existing distributed DCP suite includes `test_distributed_nixl_a2a_matches_reference`
for two/eight GPUs, one/four channels, FP16/BF16 output, FP32 packed/transposed
LSE, empty shards, exact payload/LSE comparisons with NCCL, and 1,024 recorded
changing-input graph replays interleaved with smaller eager calls and ubatches.
Local tests require mapped-peer selection and verify that send staging for
mapped destinations stays untouched throughout these calls. Cross-node runs
must separately exercise the NIXL-put path and both paths in the same group.
Run it with the source-built image before model performance measurements:

```bash
.venv/bin/python -m pytest tests/distributed/test_dcp_direct_a2a_lse_reduce.py \
  --noconftest -k distributed_nixl_a2a_matches_reference -v -s
```

That pytest launcher uses local GPUs. Cross-node qualification must launch ranks
on distinct physical hosts and verify `rc_mlx5_gda` in the workspace's transport
report, exact transport/merge results, repeated graph replay and a trace with
exactly the three stages above. GPU/NIC/driver/firmware and GPUDirect support
remain runtime prerequisites; successful host UCX communication is insufficient.

The workspace retains registered packing storage for unmapped destinations;
mapped destinations bypass it, without reducing the reserved workspace size.
Fusing packing and submission removes a launch but limits packing to the bounded
send-owner grid. A separate packing kernel could outperform it for some payloads;
three launches alone do not establish a speedup. CUDA IPC may use PCIe or NVLink,
and NIXL is not assumed to match the specialized direct kernel's local bandwidth.
Establish actual transport and numerical correctness separately from timing.
Storage transfers, host-staged fallback and NIXL-EP are outside this prototype.
Microbatch tests interleave serialized calls; overlapping multi-stream traffic
and end-to-end dual batch overlap have not been qualified.
