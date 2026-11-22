# Experimental NIXL GPU Device API DCP

The opt-in prototype implements MLA DCP output/LSE exchange on two or more
GPUs using direct stores to mapped peers and NIXL device puts to unmapped peers.
It targets both CUDA IPC mappings and cross-node mlx5 GPU-initiated RDMA, with
three kernel launches per invocation. NIXL is disabled by default and requires
an optional source build. H100 functional qualification covers mapped, RDMA
and mixed groups, including CUDA graph replay. NCCL, direct P2P and NIXL can
share a source-built image.

## Build and select

Build NIXL 1.5.0 (`1683cf3b7f3d11674c03c5e861cea22339876c96`) against UCX
`589be40811c7b53591cbbfe8a3739b5984987cdc` with CUDA IPC device support,
multithreading and installed headers. For East-West traffic, enable UCX verbs
and GDA (`--with-verbs --with-gda --enable-mt`) and include its GPUNetIO submodule
at `f44728fb023ba61911b705dba289b1650ce2eb28`. The compile-time
`UCT_RC_MLX5_GDA_SUPPORTED` check must pass. Use matching headers and libraries
at build and runtime; the setup probe inspects this pinned UCX device ABI.
`nixl._bindings.HAVE_UCX_GPU_DEVICE_API` must be true. The ordinary KV-transfer
NIXL wheel does not by itself establish device API support.

Add these options to the usual vLLM source build (substitute install prefixes):

```bash
export CMAKE_ARGS="-DVLLM_BUILD_NIXL_DCP=ON -DNIXL_DCP_INCLUDE_DIR=/opt/nixl/include -DNIXL_DCP_UCX_INCLUDE_DIR=/opt/ucx/include"
uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e .
```

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

Outgoing payload and atomic operations complete within the send kernel, so its
registered source and request storage can be reused on the next invocation.
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
workspace capacity checks. Configuration acceptance does not qualify a topology:
DCP16 has not yet been validated on GPUs.

The mixed implementation passed H100 functional qualification on the pinned
NIXL/UCX build with Torch 2.14.0+cu130 and NVIDIA driver 580.95.05:

| Placement | DCP size | Qualified paths |
| --- | --- | --- |
| One node | 2 and 8 | Mapped peers, including untouched send-staging sentinels. |
| Two nodes, one GPU each | 2 | GPU-initiated RDMA. |
| Two nodes, two GPUs each | 4 | One mapped peer and two RDMA peers per rank. |

Checks include exact NCCL payload/merge comparisons and 1,024 changing-input
CUDA graph replays per case. Stress tests record samples from every replay on
the GPU and check them after the loop, without per-replay NCCL rendezvous.
Cross-node checks also cover delayed consumers
and verify the three-launch protocol on every rank. Schema and FakeTensor
`torch.library.opcheck` checks passed for the native operators; their isolated
mapped-peer setup does not replace the distributed protocol checks.

DeepSeek-V2-Lite BF16 model qualification passed locally at TP=DCP2/8 and across
two hosts at TP=DCP2/4, using six prompts and the existing token/logprob gates.
The cross-node TP4 run exercised mapped and RDMA peers in each workspace, with
the separate symmetric-memory all-reduce initializer disabled in both arms as
described above. These are bounded generation checks, not a model-wide accuracy
evaluation. No performance improvement is established for the mapped-peer
optimization. Cross-node groups larger than four and GPU generations other than
H100 remain unqualified.

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
