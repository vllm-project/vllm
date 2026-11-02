# Experimental NIXL GPU Device API DCP

The opt-in prototype implements MLA DCP output/LSE exchange on two to eight
GPUs using NIXL device puts and atomic notifications. It targets both CUDA IPC
peers and cross-node mlx5 GPU-initiated RDMA, with three kernel launches per
invocation. H100 validation covers local CUDA IPC and two cross-node ranks
using mlx5 GPU RDMA, including CUDA graph replay and DeepSeek-V2-Lite BF16
model comparisons. NCCL, direct P2P and NIXL can share a source-built image.
This remains an experimental prototype; larger and mixed RDMA groups require
separate qualification.

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

At runtime use `--dcp-comm-backend a2a`, a DCP size of 2..8, PCP size 1, an MLA
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

The current CLI has a startup limitation: setting the NIXL flag before argument
parser construction validates a temporary default `ParallelConfig` with DCP=1
and raises an error. The model experiments use an external harness wrapper that
disables the flag during parser construction and restores it before creating
the real engine configuration. Direct `vllm serve` activation needs this startup
issue resolved; the wrapper is not part of this vLLM branch.

## Protocol and lifetime

Setup allocates fixed registered send storage and one registered receive
allocation per workspace. Each ubatch has two receive payload slots, one epoch
and per-source/channel readiness counters. Metadata and layout are exchanged
through the CPU process group. A device probe accepts CUDA IPC or mlx5 GDA
endpoints and rejects unsupported transports, CPU-proxy paths, aliased RDMA
queues and queues too small for the configured batch. Remote mapped pointers
are not required. Logs report hosts, selected device transports and workspace
bytes; that byte count excludes NIXL/UCX's own queue allocations.

| Launch | Work |
| --- | --- |
| 1 | Advance the GPU epoch and expected counters for active channels. |
| 2 | Each CTA owns one destination/channel, packs its tiles, submits NIXL puts in batches, then publishes and completes a channel-ready atomic. |
| 3 | Acquire each source's channel counter and combine output/LSE in FP32. |

The default is four channels per peer and a 64-KiB tile target. Small inputs
activate fewer channels. Packing supports strided inputs, converts LSE to FP32
and masks empty/padded rows in the same kernel. Every producer publishes its
stores before the posting warp exposes them to the NIC. Batches contain at most
four puts, ending in a nondeferred put whose completion is polled. Readiness
uses the same memory-view element and channel as the payload. CUDA IPC copying
threads also publish their writes before signalling. The merge retains the
NCCL A2A path's source order and exp/log formulation, including both LSE bases
and optional returned LSE.

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

Completed H100 checks include local two/eight-rank functional coverage,
two-host RDMA payload and reduction checks, and 1,024 changing-input CUDA graph
replays per RDMA test case. The two-host model qualification used one GPU per
host with TP=DCP=2: all six tested generated sequences matched NCCL exactly,
and the log-probability comparison passed its existing numerical gates.
Two repeated RDMA model-performance runs covered concurrency 32/128/512 with
5,632 requests and no request errors. These are bounded qualification results,
not a broad model-accuracy evaluation or a guarantee of faster communication.
Cross-node DCP greater than two and groups mixing local and remote peers have
not yet been qualified. The local-P2P/remote-NIXL hybrid is not implemented.

The existing distributed DCP suite includes `test_distributed_nixl_a2a_matches_reference`
for two/eight GPUs, one/four channels, FP16/BF16 output, FP32 packed/transposed
LSE, empty shards, exact payload/LSE comparisons with NCCL, and 1,024 recorded
changing-input graph replays interleaved with smaller eager calls and ubatches.
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

Registered packing storage remains necessary for this tiled, strided layout.
Fusing packing and submission removes a launch but limits packing to the bounded
send-owner grid. A separate packing kernel could outperform it for some payloads;
three launches alone do not establish a speedup. CUDA IPC may use PCIe or NVLink,
and NIXL is not assumed to match the specialized direct kernel's local bandwidth.
Establish actual transport and numerical correctness separately from timing.
Storage transfers, host-staged fallback and NIXL-EP are outside this prototype.
