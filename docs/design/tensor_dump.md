# Tensor dumping

Tensor dumping records module inputs and outputs to diagnose inference differences.
It is disabled by default and requires Model Runner V2 with `--enforce-eager`.
Microbatch execution (including dual batch overlap) is not supported.

## Enable capture

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_TENSOR_DUMP_DIR=/tmp/vllm-dumps \
VLLM_TENSOR_DUMP_CONFIG='{"patterns":["*self_attn*"],"max_steps":2}' \
vllm serve MODEL --enforce-eager
```

`VLLM_TENSOR_DUMP_DIR` opts workers into dumping and enables the serving control
routes. Ray forwards these `VLLM_` variables using its existing environment policy.
`VLLM_TENSOR_DUMP_CONFIG` is an optional JSON object:

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Whether to capture real model forwards. |
| `patterns` | `["*"]` | Shell-style globs matched against dump names, such as `*self_attn*`. |
| `ranks` | `null` | Distributed ranks to capture; null selects all. |
| `max_steps` | `1` | Maximum captured forwards per worker between resets; must be positive. |

Warmup/dummy forwards do not consume steps. Failed forwards can leave partial
records and consume a step. Step numbers count captured forwards only, not all
forwards, requests or generated tokens. This captures the main model; speculative draft
models, logits computation and sampling are outside this capture scope. Hooks
observe modules invoked through PyTorch's
`module(...)`; direct `module.forward(...)` calls and kernel-local values are not
automatically captured.

## Read the files

Each worker creates a unique directory containing its distributed rank, PID and
a random suffix. Numbered `.pt` files contain `name`, `step`, `rank` and `value`.
Names identify the module and input/output, with paths for nested tensors.
Repeated module calls produce separate files. Tensor snapshots are detached CPU
copies; later in-place operations cannot change them.

Every captured step also writes a `batch` record with request IDs, input IDs,
positions, sequence lengths, query boundaries and real/padded token counts.
Module tensors retain their original shapes; padding and parallel shards are
not automatically trimmed or gathered. Use the batch record and deployment
topology when comparing runs.

```python
record = torch.load(path, weights_only=True)
print(record["name"], record["step"], record["value"])
```

## Explicit capture and runtime control

For a tensor inside a model function, insert a call during the instrumented forward:

```python
from vllm.v1.worker.gpu.tensor_dump import dump

dump("attention.q", query)
```

This uses the same filters and limits. Calls outside an active capture do nothing.

The serving API exposes `POST /dumper/configure`, `/dumper/get_state` and
`/dumper/reset` only when the directory is set at startup. Configure accepts the
JSON fields above and replaces the complete configuration; omitted fields use
their defaults. Reset restarts the step budget while keeping step numbers increasing
and retaining the configuration, hooks and existing files. It does not overwrite
earlier dumps. Responses contain
worker states; with data-parallel serving, the existing collective RPC returns
states from the first engine. Configure/reset still reach the engines served by
that client. These routes use the API server's authentication when an API key is
configured; without an API key they are unauthenticated. The output directory
cannot be changed remotely. Runtime control is available on the Python frontend.

```bash
curl -X POST http://localhost:8000/dumper/configure \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true,"patterns":["*self_attn*"],"max_steps":2}'
curl -X POST http://localhost:8000/dumper/reset \
  -H 'Authorization: Bearer YOUR_API_KEY'
```

Capture synchronizes device-to-host copies and writes files synchronously, so use
narrow filters and a small step budget. Dumps can contain prompts, activations and
other sensitive data. The default pattern captures every module, including repeated
parent/child values. File-write errors propagate to the inference call. This is a
debugging tool, not a production telemetry path.
