# ConnexConnector Usage Guide

`ConnexConnector` moves KV cache blocks from prefill to decode instances with
[Connex](https://github.com/HeterInfra/connex), a communication library whose
endpoints keep a stable logical identity while instances join, leave and fail.
The connector itself is provided by the `connex-ai` package; vLLM only
registers it.

What it adds over a plain P/D transfer:

- **Pull with a renewable hold.** The decode instance reads the prompt KV with
  one-sided RDMA reads (connex-rs over UCX with GPUDirect RDMA, or Mooncake
  Transfer Engine). The prefill instance keeps the blocks under a transfer id
  until the decode instance releases them (or a lease expires), so if the
  decode instance fails mid-generation, the request can be resumed on another
  decode instance that pulls the same KV again instead of re-running prefill.
  This reuse needs a resume that extends the original prompt token for token:
  the Connex PD router does that for `/v1/completions` requests (token-ID
  resume, the default). Chat requests, and completions resumed from text
  (`--no-token-ids`), are resumed from text and run prefill again once tokens
  have been emitted, even while the hold exists.
- **Failure reporting per request.** A prefill instance that dies during a
  transfer fails only the affected requests (load errors, handled by
  `kv_load_failure_policy`), detected within milliseconds by connex-rs.
- **Pool membership.** Instances can register their role in a Connex
  front-router, which the Connex PD router (`connex.serving.pd_router`) uses
  to add and remove prefill and decode instances at run time.

## Installation

```bash
pip install "connex-ai[rs]"        # connex-rs data plane (RDMA hosts)
# or
pip install "connex-ai[mooncake]"  # Mooncake Transfer Engine data plane
```

## Usage

Prefill instance (the two instances must use different GPUs: the decode
instance refuses to pull from a prefill instance on its own GPU, and two
instances on one GPU may also run out of memory):

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3-8B --port 8100 \
    --kv-transfer-config '{"kv_connector":"ConnexConnector","kv_role":"kv_producer","kv_connector_extra_config":{"ctrl_port":17100}}'
```

Decode instance:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Meta-Llama-3-8B --port 8200 \
    --kv-transfer-config '{"kv_connector":"ConnexConnector","kv_role":"kv_consumer"}'
```

Router (follows the same `kv_transfer_params` protocol as the NIXL proxy
example: prefill with `do_remote_decode`, then decode with the returned
parameters; it also resumes requests whose decode instance fails):

```bash
python -m connex.serving.pd_router --port 8000 \
    --prefill http://127.0.0.1:8100 --decode http://127.0.0.1:8200
```

## Configuration

`kv_connector_extra_config` keys:

| key | default | meaning |
| --- | --- | --- |
| `backend` | unset | `connex-rs` or `mooncake`; leave unset to pick one automatically (connex-rs on RDMA hosts, else Mooncake) |
| `host` | `VLLM_HOST_IP` | address peers use to reach this instance |
| `ctrl_port` | `kv_port` | KV control port; TP rank `r` uses `ctrl_port + r` |
| `retain_s` | `60` | prefill hold lease, renewed by every pull |
| `release` | `on_finish` | `after_pull` frees prefill blocks right after the first pull |
| `front_router`, `http_url`, `replica` | unset | register this instance in a Connex front-router |

## Limitations

- Full-attention models; prefill and decode must use the same tensor-parallel
  size and block size (rank `r` pulls from rank `r`).
- No hybrid (Mamba / sliding-window) KV groups and no pipeline parallelism
  inside an instance yet.
