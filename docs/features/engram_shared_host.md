# Shared host-backed Engram tables (experimental)

DeepSeek-V4.1-Flash has two immutable FP8/E8M0 Engram tables totaling
202,758,032,400 bytes (188.833 GiB). Independently launched prefill and decode
instances normally allocate a TP-sharded copy in each instance. This opt-in
mode places one complete generation in a persistent owner and lets workers
import CUDA Fabric handles. Their Engram kernels read selected rows through
device-local virtual mappings; the row output remains in each worker's HBM.
P/D KV transfer is configured separately.

The CUDA provider uses `cuMemCreate` with `HOST_NUMA` placement and `FABRIC`
handles. It can place chunks of one table on multiple GPU-local Grace NUMA
nodes while mapping them as one contiguous VA in each process. Consumer GPU
mappings request read access from the CUDA driver. Workers do not allocate,
initialize, or load table replicas.

## Scope and prerequisites

- CUDA with HOST_NUMA VMM, Fabric handles, and a common NVIDIA IMEX channel for
  owner and consumers. Cross-host use requires the same NVLink fabric domain.
- A trusted, reachable TCP control network and the same private token file on
  every participant. The control channel is not encrypted; use cluster network
  isolation. FABRIC handles are accepted only by processes with IMEX access.
- A local DeepSeek-V4.1-Flash safetensors checkpoint directory on the owner and
  consumers. Consumers need the mixed shards for their small Engram projection
  weights, but the two large table tensors are skipped before `get_tensor`.
- Identical `config.json` and `model.safetensors.index.json` bytes across owner
  and consumers. The owner records SHA256 of both table tensors; consumers
  validate the config/index hashes, layout, encoding and explicit generation.
  Config/index hashes do not independently verify all checkpoint weight bytes.
- DP size 1 within each serving instance, CPU Engram offload, and explicit lazy
  safetensors loading. TP can differ between independent instances, subject to
  the usual model and KV connector limits.
- In-place weight reload and weight transfer are unsupported: a new checkpoint
  cannot change the owner's immutable Engram generation in place. Start a new
  owner generation and new consumers for an updated checkpoint.

## Start the owner

The complete single-host P TP2 + D TP2 NIXL example, including owner,
consumer launch commands and a routed request, is in
`examples/disaggregated/shared_engram/`. SGLang consumers can use the same
provider with their own launch recipe in
[SGLang's VMM port](https://github.com/foraxe/sglang/tree/nyx/engram-vmm-port-99fd0104)
(intended for [SGLang #47478](https://github.com/sgl-project/sglang/pull/47478)). The two
engines use separate KV-transfer paths; the shared contract is the Engram
owner generation and FABRIC handles.

Create a private random token file, then start the owner on a host whose GPUs
can access both Grace NUMA nodes. Keep it running while workers use its
generation. The owner prints `ENGRAM_OWNER_READY` after both real tables and
their SHA256 digests are complete.

```bash
umask 077
openssl rand -hex 32 > /shared/private/engram.token
python -m vllm.distributed.engram_share.vmm.owner \
  --model-dir /models/DeepSeek-V4.1-Flash \
  --listen-host OWNER_ADDRESS --listen-port 56000 \
  --token-file /shared/private/engram.token --device 0
```

The ready line contains a generation ID. Give that ID, the owner address, and
access to the token file to each independent worker. For example, add these
settings to otherwise normal `vllm serve` arguments:

```bash
--engram-config '{"cpu_offload":true,"shared_host_endpoint":"OWNER_ADDRESS:56000","shared_host_token_file":"/shared/private/engram.token","shared_host_generation":"GENERATION_ID"}' \
--load-format safetensors --safetensors-load-strategy lazy
```

The owner exchanges mapping handles at startup. It does not serve row requests.
Workers use their own TP head ranges and existing Engram lookup/dequantization
kernel. The backend does not depend on a vLLM DP process group or Mooncake
Store Put/Get.

## Lifetime and failure behavior

Workers retain imported mappings through CUDA Graph use. During normal worker
shutdown, vLLM first stops model execution and synchronizes the device, then
releases mappings and owner leases. For serving, set a nonzero
`--shutdown-timeout` so the engine drains rather than force-killing its workers.
The owner stops accepting new attachments during shutdown and waits for all
leases to detach. A lost control socket leaves an uncertain lease: the owner
does not infer that GPU reads stopped, and it does not transparently fail over
to another generation. An operator must establish that those consumer
processes are gone before forcing owner teardown. The owner logs attach and
detach events with a bounded client identity (host, PID, device, layer). On
shutdown, `ENGRAM_OWNER_DRAIN_WAIT` reports active and uncertain lease IDs and
their recorded clients every 60 seconds until all leases are released. An
uncertain lease is not cleared automatically by a lost TCP connection.

## Validation status

The VMM-only port from pinned main `6b3488cc97ee` passed local four-GB200
validation with Torch 2.14 and an older precompiled native wheel: a separate
consumer matched seven real checkpoint boundary rows per layer and changed-ID
CUDA Graph replay; the full TP2 model captured decode graphs at batches 1 and
2 and generated two 32-token outputs; and P TP2 to D TP2 NIXL serving
completed with one owner and eight normal detaches. The SGLang VMM port also
completed its independent P/D route against this owner. These are same-host
correctness gates, not a stable speedup or cross-host NVL72 claim.

Cross-host import and serving on two NVL72 hosts have not been run because only
one host is available. The local result establishes the mechanism and
single-node serving behavior, not NVL72 reachability. Active-graph owner loss,
sleep/reload, non-FP8 table encodings, and production SLO acceptance remain
unqualified.
