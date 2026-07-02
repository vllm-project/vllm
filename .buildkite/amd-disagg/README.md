# ROCm Disaggregated Inference CI (`.buildkite/amd-disagg/`)

A config-driven launcher + SLURM/Buildkite glue for vLLM disaggregated
prefill/decode (P/D) serving over the **MoRIIO** KV connector on ROCm. Supports
single- and multi-node `xP`/`yD` topologies and two parallelism modes via
**`WIDE_EP_MODE`**. The first CI target is **1P1D TP8** (`WIDE_EP_MODE=0`, 2 nodes).

| `WIDE_EP_MODE` | Mode | What runs | Topology |
|-----------|------|-----------|----------|
| `0` (default) | `tp` | independent **TP8** server per node | xP prefill + yD decode servers, each registered to the proxy |
| `1` | `ep` | **DP + Expert Parallel** | prefill = one DP group of `xP*GPUS_PER_NODE` across xP nodes; decode = one DP group of `yD*GPUS_PER_NODE` across yD nodes (1 master + children each) |

## Files

### Launcher (cluster-agnostic; runs inside `rocm/vllm-ci`)
| File | Purpose |
|------|---------|
| `vllm_disagg.sh` | The launcher. Roles: `proxy`, `prefill`, `decode`, `bench`, `accuracy`. |
| `cluster.sh` | **Single** config (sourced): `WIDE_EP_MODE`, topology, ports, RDMA/NIC, vLLM env, benchmark/accuracy defaults, plus the site defaults (model dir, fallback IPs, NIC list, log path, partition — tagged `[site]`). All `${VAR:-default}`; precedence is **env > built-in**. Edit the `[site]` defaults for a new cluster. |
| `models.yaml` | **Per-model** perf flags (split by mode `tp`/`ep` and role) plus an `env:` block of model/arch-specific env (e.g. AITER kernel toggles). |
| `apply_moriio_2pd_patches.sh` | Applies vLLM PR #39276 (MoRIIO multi-node DP). Auto-run when `WIDE_EP_MODE=1` and `xP>1`/`yD>1`. |
| `moriio_toy_proxy_server.py` | MoRIIO toy proxy (self-contained copy of the in-repo example). Default front door. |

### toy proxy vs. vLLM router (`ROUTER_TYPE`)

The client-facing gateway is selectable via **`ROUTER_TYPE`** (default `toy`):

| `ROUTER_TYPE` | What runs | Client port |
|---------------|-----------|-------------|
| `toy` (default) | `moriio_toy_proxy_server.py` as a background process **inside** the rank-0 container | `PROXY_PORT` (10001) |
| `vllm-router` | a **separate** `vllm/vllm-router` container on the rank-0 node (started by the SLURM job) | `ROUTER_PORT` (30000) |

Note: There are few existing issues while running vllm-router with DP mode, once that is fixed can be switched as default.

Both use the identical MoRIIO discovery mechanism — prefill/decode register to
`PROXY_IP:PROXY_PING_PORT` (`36367`) — so the prefill/decode serve commands and
`--kv-transfer-config` are unchanged. Only the HTTP front door differs, abstracted
behind `GATEWAY_PORT` (which `bench`/`accuracy` target). With `vllm-router`, rank 0's
node runs **two** containers: the router (`VLLM_ROUTER_IMAGE`) and the usual main
container (`IMAGE`) running `vllm_disagg.sh node`.

Relevant knobs (in `cluster.sh`, env-overridable): `ROUTER_TYPE`, `ROUTER_PORT`,
`ROUTER_POLICY` (`round_robin`), `VLLM_ROUTER_IMAGE`
(`vllm/vllm-router:nightly`), `GATEWAY_PORT`, `ROUTER_DP_LOCAL`
(intra-node data-parallel size the router routes across; `GPUS_PER_NODE` for
wideEP, `1` for TP).

```bash
# opt in to the production router (1P1D):
ROUTER_TYPE=vllm-router NODES=2 \
  bash .buildkite/amd-disagg/run-slurm-disagg-test.sh
```

### SLURM / Buildkite CI glue
| File | Purpose |
|------|---------|
| `run_xPyD_disagg.slurm` | The **directly-`sbatch`-able** SLURM job: selects nodes, resolves node IPs, fans out ONE container per node via a single `srun`, and hands off to `vllm_disagg.sh node` (rank-based prefill/decode self-select via `$SLURM_PROCID`). Health-gates and runs the workload. Passes `cluster.sh` as `CLUSTER_ENV` and bind-mounts the host scripts dir into the container. Debug by hand: `sbatch -N2 --gres=gpu:8 -p amd-rccl run_xPyD_disagg.slurm`. |
| `run-slurm-disagg-test.sh` | Thin **foreground submitter** (CI glue): `sbatch run_xPyD_disagg.slurm`, polls for completion, and propagates the job exit code (incl. the GSM8K accuracy gate). The single command the pipeline step calls. |
| `pipeline.disagg.yaml` | The Buildkite step snippet to paste into `.buildkite/test-amd.yaml`. |

## CI flow (SLURM)

```
Buildkite step (SLURM login node)
  └─ run-slurm-disagg-test.sh        # sbatch + poll + propagate rc
       └─ run_xPyD_disagg.slurm       # selects nodes, resolves IPADDRS, single srun fan-out
            └─ srun (1 task/node): docker run … vllm_disagg.sh node
                 ├─ rank 0 .. xP-1       -> prefill  (rank 0 also: proxy + orchestrator)
                 ├─ rank xP .. xP+yD-1   -> decode
                 └─ rank 0: health-gate both → bench|accuracy → sentinel → exit rc
```

Each node runs one container with `--network host` and RDMA passthrough
(`/dev/infiniband`, `--cap-add IPC_LOCK`). The host scripts dir is bind-mounted
into the container (`DISAGG_SCRIPTS_DIR` → `CONTAINER_SCRIPTS`), so script edits
take effect without rebuilding the image. Each node self-selects its role from
its global rank (`$SLURM_PROCID`) — no per-role `srun` fan-out and no manually
passed `NODE_RANK`.

## Quick start — 1P1D TP8 on SLURM

```bash
# from the SLURM login node (defaults: NODES=2, WIDE_EP_MODE=0, TP8)
IMAGE=rocm/vllm-ci:<commit> SLURM_PARTITION=amd-rccl \
  bash .buildkite/amd-disagg/run-slurm-disagg-test.sh

# …or submit the job directly (manual/debug, no CI streaming):
sbatch -N2 --gres=gpu:8 -p amd-rccl .buildkite/amd-disagg/run_xPyD_disagg.slurm
```

The launcher owns topology, host/port, EP backbone (`--enable-expert-parallel`,
`--all2all-backend mori`, `--data-parallel-*`, `--headless`/`--data-parallel-start-rank`,
`--api-server-count`) and the `--kv-transfer-config` block. `models.yaml` only
holds model-specific perf knobs (mem fraction, eager/cudagraph) and model/arch
env under `env:`.

Model env precedence: the launcher exports each `models.yaml` `env:` entry only
if it is not already set, so **caller/inline env > `models.yaml` env:**. Cluster
transport/engine env (e.g. `VLLM_MORIIO_CONNECTOR_READ_MODE`, `VLLM_USE_V1`,
`HSA_NO_SCRATCH_RECLAIM`) stays in `cluster.sh`.

## Topology model

`IPADDRS` is the ordered, comma-separated node IP list — prefill IPs first, then
decode IPs. `NODE_RANK` is the global 0-based rank (`$SLURM_PROCID` under SLURM):

```
rank 0          -> prefill master (+ proxy co-located)
rank 1..xP-1    -> prefill child   (WIDE_EP_MODE=1) / independent prefill TP server (WIDE_EP_MODE=0)
rank xP         -> decode  master
rank xP+1..end  -> decode  child   (WIDE_EP_MODE=1) / independent decode TP server (WIDE_EP_MODE=0)
```

For manual 1P1D, `NODE_RANK` defaults per role (prefill→0, decode→xP) and
`IPADDRS` falls back to `PREFILL_IP,DECODE_IP`.

## Quick start — 1P1D, TP8 (`WIDE_EP_MODE=0`)

```bash
# prefill node
./vllm_disagg.sh proxy                 # start first
./vllm_disagg.sh prefill
# decode node
NODE_RANK=1 ./vllm_disagg.sh decode
# anywhere
./vllm_disagg.sh bench
```

## 2P2D — DP/EP (`WIDE_EP_MODE=1`)

4 nodes (2 prefill + 2 decode), 8 GPUs each → prefill DP=16, decode DP=16.
Run one invocation per node with its global `NODE_RANK`:

```bash
export WIDE_EP_MODE=1 xP=2 yD=2
export IPADDRS=ipP0,ipP1,ipD0,ipD1     # prefill master, prefill child, decode master, decode child

NODE_RANK=0 ./vllm_disagg.sh prefill   # prefill master (+ run ./vllm_disagg.sh proxy here too)
NODE_RANK=1 ./vllm_disagg.sh prefill   # prefill child (--headless, start-rank 8)
NODE_RANK=2 ./vllm_disagg.sh decode    # decode master
NODE_RANK=3 ./vllm_disagg.sh decode    # decode child  (--headless, start-rank 8)

# on the prefill master, after both masters are up:
./vllm_disagg.sh proxy
./vllm_disagg.sh bench
```

`apply_moriio_2pd_patches.sh` (PR #39276) is applied automatically because
`xP>1`/`yD>1`. Control with `APPLY_MORIIO_PATCH=auto|1|0`.

## Switching modes

`WIDE_EP_MODE` is the single switch. Turn EP off to run plain TP8:

```bash
WIDE_EP_MODE=0 ./vllm_disagg.sh prefill     # TP8
WIDE_EP_MODE=1 ./vllm_disagg.sh prefill     # DP/EP
```

…or persist it in `cluster.sh`. Inline `--wide-ep-mode 0|1` overrides both.

## Notes / gotchas

- **Proxy first, on the prefill master.** `PROXY_PING_PORT` must stay `36367`
  (toy proxy hardcodes its zmq discovery there; `vllm-router` must use the same
  via `--vllm-discovery-address`). `proxy_ip` = prefill master.
- **`ROUTER_TYPE=vllm-router`** swaps the toy proxy for an external
  `vllm/vllm-router` container on rank 0 ; clients then hit
  `ROUTER_PORT`. Nothing else in the P/D topology changes.
- **PR #39276 is mandatory for multi-node DP** (`WIDE_EP_MODE=1`, `xP>1`/`yD>1`); the
  launcher aborts if the patch is required but unavailable.
- **Same model on both sides** — single shared `MODEL_NAME`/`MODEL_DIR`.
- **EP masters only** expose the API server + KV transfer; children are `--headless`.

