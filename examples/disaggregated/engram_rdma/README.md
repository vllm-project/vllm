# Experimental remote Engram owner and reader

vLLM ships the application-side owner, lease protocol, checkpoint validation,
reader and provider under `vllm.distributed.engram_rdma`. A separately built
GPU GET transport is still required. The current reader expects the legacy
ABI-3 NIXL/GPUNETIO shim, not an ordinary NIXL wheel or the proposed interface
in [NIXL #2544](https://github.com/ai-dynamo/nixl/pull/2544). The tested
[legacy research snapshot](https://github.com/foraxe/nixl/tree/b5322e46f6e686ff16bdb3afb8a1a940a64e1163/research/vllm_engram_repro)
contains the matching NIXL source and build instructions for DOCA 3.1.

The owner allocates pinned host memory for the full table. The owner and
readers must be connected over a trusted network: the control protocol is not
authenticated or encrypted, and the owner ready file contains memory keys.
Keep this file private and transfer it securely. Use distinct ports per owner
and reader instance. This is a prototype, not a fault-tolerant service.

## Start an owner

For a transport-only check without model weights, generate the same small
fixture on both nodes. This fixture is not a runnable language model:

```bash
python examples/disaggregated/engram_rdma/make_tiny_checkpoint.py --output "$MODEL"
```

Set the matching NIXL/DOCA library paths and GPUNETIO device configuration for
the host. A ready/stop file must not already exist. On the owner node:

```bash
export NIXL_PLUGIN_DIR="$NIXL_PLUGIN_DIRECTORY"
export LD_LIBRARY_PATH="$NIXL_LIBRARY_DIRECTORY:${LD_LIBRARY_PATH:-}"
export GPUNETIO_DEVICE_API_RDMA="$RDMA_DEVICE"
export GPUNETIO_DEVICE_API_GID_INDEX="$GID_INDEX"
export GPUNETIO_DEVICE_API_OOB_INTERFACE="$NETWORK_INTERFACE"
python -m vllm.distributed.engram_rdma.owner \
  --checkpoint "$MODEL" --library "$ENGRAM_NIXL_SHIM" --owner-gpu 0 \
  --rdma-port "$OWNER_RDMA_PORT" --control-host "$OWNER_BIND_IP" \
  --advertise-host "$OWNER_REACHABLE_IP" --control-port "$OWNER_CONTROL_PORT" \
  --ready-file "$OWNER_READY_FILE" --stop-file "$OWNER_STOP_FILE" \
  --max-pinned-gib "$PINNED_BUDGET_GIB" --storage-layout row264
```

The owner validates checkpoint metadata, creates one pinned row264 backing,
registers both layer tables and writes the mode-0600 ready file. It stays alive
until it receives SIGTERM/SIGINT or the stop file appears. It then rejects new
readers and waits for all leases before deregistration. The current control
protocol does not automatically revoke a lease after a worker crashes.

## Configure vLLM readers

On each reader node, securely copy the ready file and set the same transport
library configuration plus:

```bash
export ENGRAM_SHARED_READY="$OWNER_READY_FILE"
export ENGRAM_SHARED_LIBRARY="$ENGRAM_NIXL_SHIM"
export ENGRAM_SHARED_MODEL="$MODEL"
export ENGRAM_SHARED_READER_PORT_BASE="$READER_PORT_BASE"
```

Use a unique reader port base per vLLM instance. TP ranks derive their ports
from the visible GPU ordinal. Launch vLLM with `--engram-config
'{"cpu_offload":true,"rdma_provider":"vllm.distributed.engram_rdma.provider"}'`,
`--load-format safetensors --safetensors-load-strategy lazy` and a nonzero
`--shutdown-timeout`. Follow the model's regular P/D setup separately. The
provider verifies checkpoint identity and sampled content before attaching;
it uses one READ lane per rank. See the Engram feature documentation for graph
and model limitations.

## Verify and stop

Run the reader against the same checkpoint on a reader node:

```bash
python examples/disaggregated/engram_rdma/verify_rows.py \
  --model "$MODEL" --ready "$OWNER_READY_FILE" \
  --library "$ENGRAM_NIXL_SHIM" --reader-port-base "$TEST_READER_PORT_BASE"
```

The verifier checks both table layers, padding, exact packed rows, the
514-query boundary, alternating CUDA streams, queued close and reader lease
release. It copies only validation rows to the CPU and is not a performance
benchmark. The self-contained GPU adapter smoke uses a fake reader and does not
test RDMA:

```bash
python examples/disaggregated/engram_rdma/test_provider_gpu.py
```

Stop all vLLM workers and wait for their reader leases to release before
requesting owner shutdown. If a worker is forcibly terminated or a transport
error prevents safe retirement, the owner retains the backing and may not exit;
do not force-release leases while device or NIC work may still reference it.
