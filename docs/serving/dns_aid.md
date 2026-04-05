# DNS-AID Endpoint Discovery

vLLM can publish an [RFC 9460](https://www.rfc-editor.org/rfc/rfc9460) SVCB
ServiceMode record to a DNS zone when it starts. AI agents that implement the
[DNS-AID](https://dns-aid.org) standard can then discover vLLM endpoints and
their model capabilities with a single DNS lookup — no bootstrap API call
required.

## Installation

DNS-AID support is an optional dependency:

```bash
pip install "vllm[dns-aid]"
```

## Quick Start

### Environment variables

```bash
export DNS_AID_ZONE="agents.example.internal"
export DNS_AID_SERVER="ns1.example.internal"   # read by the dns-aid library

vllm serve meta-llama/Llama-3-70b-instruct --dns-aid-enabled
```

### CLI flags

```bash
vllm serve meta-llama/Llama-3-70b-instruct \
  --dns-aid-enabled \
  --dns-aid-zone agents.example.internal \
  --dns-aid-name llama3-70b
```

## Published SVCB Record

On startup vLLM registers a record of the form:

```
_{agent-name}._agents.{zone}  SVCB  1  {target}  (
    alpn="h2"
    port={port}
    # custom hints:
    model="meta-llama/Llama-3-70b-instruct"
    context_len="131072"
    quant="fp8"
    max_batch="256"
    framework="vllm"
    api_base="/v1"
)
```

Example record (default name derived from model):

```
_meta-llama-llama-3-70b-instruct._agents.agents.example.internal
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--dns-aid-enabled` | `false` | Enable DNS-AID registration |
| `--dns-aid-zone` | `None` | DNS zone for the record (required) |
| `--dns-aid-name` | slugified model name | Override the agent name label |

## Environment Variables

The dns-aid library reads its own configuration from environment variables.
vLLM-specific overrides use CLI flags; DNS transport settings are configured
via the library's env vars.

| Variable | Source | Description |
|----------|--------|-------------|
| `DNS_AID_ZONE` | vLLM | Fallback if `--dns-aid-zone` is not set |
| `DNS_AID_NAME` | vLLM | Fallback if `--dns-aid-name` is not set |
| `DNS_AID_SERVER` | dns-aid library | DNS server for dynamic updates |
| `DNS_AID_PORT` | dns-aid library | DNS server port (default: 53) |

## Published Hints

| Hint | Source |
|------|--------|
| `model` | Full model name (e.g. `meta-llama/Llama-3-70b-instruct`) |
| `context_len` | `max_model_len` from model config |
| `quant` | Quantization method (e.g. `fp8`, `awq`) or `"none"` |
| `max_batch` | `max_num_seqs` from scheduler config |
| `framework` | Always `"vllm"` |
| `api_base` | Always `"/v1"` |

## Agent Name and DNS Labels

The default agent name is the model name converted to a DNS-safe label:
lowercase, special characters replaced with hyphens.  If the resulting label
exceeds the 63-octet DNS limit, it is truncated with a short hash suffix to
preserve uniqueness.  You can always override this with `--dns-aid-name`.

## Target Hostname

The SVCB `target` is set to `--host` when it looks like a resolvable FQDN
(contains a dot and is not an IP address). Otherwise `socket.getfqdn()` is
used.

## Multi-GPU / Tensor Parallel

In tensor-parallel (or any multi-rank) configuration, only **global rank 0**
registers the DNS record. This ensures exactly one record per deployment
regardless of TP/PP/DP topology.

## TTL

The record TTL is 60 seconds.  This short value ensures clients discover
replacement endpoints quickly after a rolling restart or scale-down.  The
dns-aid library handles periodic refresh; vLLM itself does not re-register.

## Graceful Deregistration

The record is removed on both:

- **ASGI shutdown** (e.g. `SIGINT`): via the lifespan `finally` block.
- **SIGTERM**: the existing vLLM SIGTERM handler raises `KeyboardInterrupt`,
  which triggers the ASGI lifespan shutdown path.

## Troubleshooting

**Server starts but no record appears**

- Check that `--dns-aid-zone` (or `DNS_AID_ZONE`) is set.
- Verify the dns-aid library is installed: `python -c "import dns_aid"`.
- Look for `DNS-AID:` log lines at `INFO` level.

**Warning: 'dns-aid' library is not installed**

Install the optional dependency: `pip install "vllm[dns-aid]"`.

**Warning: no zone is configured**

Set `--dns-aid-zone <zone>` or export `DNS_AID_ZONE=<zone>`.
