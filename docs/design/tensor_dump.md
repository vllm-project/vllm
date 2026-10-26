# Tensor dumping

Tensor dumping records module inputs and outputs to diagnose inference differences.
It is disabled by default and requires Model Runner V2 with `--enforce-eager`.
Microbatch execution (including dual batch overlap) is not supported.

## Enable capture

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve MODEL --enforce-eager \
  --tensor-dump-config '{"output_dir":"/tmp/vllm-dumps","patterns":["*self_attn*"],"max_steps":2}'
```

Like `--profiler-config`, `--tensor-dump-config` accepts a JSON object or individual
keys such as `--tensor-dump-config.max_steps 2`. Setting `output_dir` opts workers
into dumping. The configuration travels
with `VllmConfig` to workers, including Ray workers; no dumper environment variables
are required. The directory is local to each worker, so use a shared filesystem
if dumps from multiple nodes should be collected in one place.

| Field | Default | Meaning |
| --- | --- | --- |
| `output_dir` | `null` | Worker output directory; null disables dumping. Startup only. |
| `enabled` | `true` | Whether to capture real model forwards. |
| `patterns` | `["*"]` | Shell-style globs matched against dump names, such as `*self_attn*`. |
| `ranks` | `null` | Distributed ranks to capture; null selects all. |
| `max_steps` | `1` | Maximum captured forwards per worker between resets; must be positive. |
| `start_step` | `0` | Number of real forwards to skip after startup or reset, including forwards while disabled. |
| `layers` | `null` | Layer indices selected from module paths containing `layers.N`; null selects all modules. |
| `phase` | `"all"` | Select `prefill`, `decode`, `mixed`, or all batches. Mixed batches are not split. |

Offline callers can pass the same configuration through `LLM`:

```python
llm = LLM(
    model=MODEL,
    enforce_eager=True,
    tensor_dump_config={"output_dir": "/tmp/vllm-dumps", "max_steps": 2},
)
```

Warmup/dummy forwards do not consume steps. Failed forwards can leave partial
records and consume a step. Step numbers count captured forwards only, not all
forwards, requests or generated tokens. This captures the main model; speculative draft
models, logits computation and sampling are outside this capture scope. Hooks
observe modules invoked through PyTorch's
`module(...)`; direct `module.forward(...)` calls and kernel-local values are not
automatically captured.

`start_step` uses a separate forward counter; filtering does not consume the
`max_steps` capture budget. Phase classification uses MRV2's per-request
`is_prefilling_np`: all prefill requests means prefill, none means decode, and
both means mixed. A one-token prefill remains prefill, and speculative decode
remains decode even when it processes multiple tokens. Layer, phase, rank, and
name filters apply together. Reset also restarts the forward counter and delay.

## Read the files

Each worker creates a unique directory containing its distributed rank, PID and
a random suffix. Numbered `.pt` files contain `name`, `step`, `rank` and `value`.
Names identify the module and input/output, with paths for nested tensors.
Repeated module calls produce separate files. Tensor snapshots are detached CPU
copies; later in-place operations cannot change them.

Every captured step also writes a `batch` record with request IDs, input IDs,
positions, sequence lengths, query boundaries and real/padded token counts.
`parallel_groups` records each TP, PP, DP, DCP and PCP group's local rank and
ordered global ranks, plus the EP group when initialized. These describe process
groups, not tensor partition dimensions; an EP group can exist without expert
parallel execution being enabled. Uninitialized groups in standalone use produce
an empty mapping.
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

No dedicated HTTP endpoint is added. Offline callers can use
`llm.collective_rpc("dumper_control", args=("get_state",))` and the existing
Python development `/collective_rpc` endpoint supports the same worker method.
HTTP access requires `VLLM_SERVER_DEV_MODE=1`; this enables other development
endpoints too. This development endpoint is not protected by `--api-key` in this
baseline; bind to loopback or isolate access in a trusted debugging environment.
Configure accepts the
capture fields above, excluding `output_dir`, and replaces the capture options;
omitted fields use their defaults. Reset restarts the step budget while keeping step numbers increasing
and retaining the configuration, hooks and existing files. It does not overwrite
earlier dumps. Responses contain
worker states; with data-parallel serving, the existing collective RPC returns
states from the first engine. Configure/reset still reach the engines served by
that client. The output directory cannot be changed through `dumper_control`.
Runtime HTTP control is available on the Python frontend.

```bash
curl -X POST http://localhost:8000/collective_rpc \
  -H 'Content-Type: application/json' \
  -d '{"method":"dumper_control","args":["configure",{"enabled":true,"patterns":["*self_attn*"],"max_steps":2}]}'
curl -X POST http://localhost:8000/collective_rpc \
  -H 'Content-Type: application/json' \
  -d '{"method":"dumper_control","args":["reset"]}'
```

Capture synchronizes device-to-host copies and writes files synchronously, so use
narrow filters and a small step budget. Dumps can contain prompts, activations and
other sensitive data. The default pattern captures every module, including repeated
parent/child values. File-write errors propagate to the inference call. This is a
debugging tool, not a production telemetry path.
