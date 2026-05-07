# Capture Consumers

Capture consumers are a pluggable system for observing and routing
hidden-state activations produced inside vLLM's forward pass. Consumers
receive captured tensors at request finalization and can do anything
they want with them — stream them to disk, feed them into a training
loop, ship them to a dashboard, or simply log that a capture occurred.

This page is the user-facing guide. For the internal design and
runtime mechanics, see
[Capture Consumers Design](../design/capture_consumers.md).

## What Capture Consumers Do

Each capture consumer is a plugin registered under the
`vllm.capture_consumers` Python entry-point group. Once enabled, the
engine routes activations from specific `(layer, hook)` points to the
consumer as requests are processed. Consumers can be triggered in two
ways:

- **Global capture**: the consumer declares a `CaptureSpec` that
  applies to every request. Used by observability probes, reward
  trainers, dashboards — anything that wants to see every request
  without clients opting in.
- **Per-request capture**: the client opts in by setting
  `SamplingParams.capture[consumer_name]`. Used by the built-in
  filesystem consumer so callers choose a tag, layers, and positions
  per request.

The two modes compose: a single request can trigger a global consumer
*and* a per-request consumer, and `RequestOutput.capture_results`
returns a per-consumer result dict.

## Built-in Consumers

vLLM ships two consumers, registered in its own `pyproject.toml` via
the same entry-point group third-party plugins use.

### `filesystem`

Streams captured activations to raw `.bin` files with sidecar JSON.
Implemented at `vllm.v1.capture.consumers.filesystem.FilesystemConsumer`.

- `reads_client_spec = True` — captures are always per-request. The
  filesystem consumer has no global spec.
- `location = "worker"` — runs in the engine-core subprocess, so it
  can stream bytes to disk without crossing a process boundary.
- Writes incrementally to `{path}.bin.tmp` as chunks arrive and does
  an atomic `os.replace` on finalize, so readers never see a partial
  file.

**Engine-side parameters** (set via `--capture-consumers` / YAML / the
Python API):

| Field | Type | Default | Purpose |
|---|---|---|---|
| `root` | `str` | required | Root directory for all captures. |
| `writer_threads` | `int` | `4` | Writer thread pool size. |
| `queue_size` | `int` | `1024` | Per-thread bounded queue capacity. |
| `timeout_seconds` | `float` | `180.0` | Per-write timeout; failures become `partial_error`. |
| `on_collision` | `"overwrite" \| "error" \| "suffix"` | `"overwrite"` | What to do when the target `.bin` already exists. |
| `fd_cache_size` | `int` | `256` | Per-thread LRU file-descriptor cache. |

**Per-request client spec** (`FilesystemCaptureRequest`):

```python
@dataclass
class FilesystemCaptureRequest:
    request_id: str                      # filename stem, slugged
    tag: str                             # grouping label, slugged
    hooks: dict[str, Any]                # hook name -> layer selector
    positions: str | list[int]           # position selector
```

Client `hooks` values may be a list of ints, the literal string
`"all"`, or a dict `{"layers": [...], "ranges": [[a, b], ...]}`.
`positions` accepts `"last_prompt"`, `"all_prompt"`,
`"all_generated"`, `"all"`, or an explicit `list[int]`.

**On-disk layout**:

```
{root}/{tag_slug}/{request_id_slug}/{layer_idx}_{hook_name}.bin
{root}/{tag_slug}/{request_id_slug}/{layer_idx}_{hook_name}.json
```

`tag_slug` and `request_id_slug` are produced by the admission
validator — characters outside `[a-zA-Z0-9._-]` are replaced with
`_`, and `..` / leading `/` are rejected outright.

**Payload**: raw tensor bytes in the model's residual dtype. `bf16`
is stored as raw uint16 bytes; readers should round-trip through
`torch.uint16.view(torch.bfloat16)`.

**Sidecar JSON**: written atomically alongside the `.bin` file on
finalize. Contains `request_id`, `layer`, `hook`, plus any fields
the framework propagates via the per-finalize `sidecar` dict.

### `logging`

Minimal observation consumer. Logs one line per finalized capture:
`"capture key=... rows=N dtype=..."`. Discards the actual tensor.
Implemented at `vllm.v1.capture.consumers.logging.LoggingConsumer`.

- `reads_client_spec = False` — activated by its global spec.
- `location = "worker"`.

**Parameters**:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `hooks` | `dict[str, list[int]]` | required | Hook name to layer indices. |
| `positions` | position selector | `"last_prompt"` | Which positions to capture. |
| `level` | `str` | `"INFO"` | Python logging level. |

## Enabling Consumers

A consumer is enabled when its name is referenced in any of the
config surfaces below. Names are the entry-point names from
`pyproject.toml` (e.g. `filesystem`, `logging`, or whatever a
third-party plugin registers).

### CLI (`vllm serve`)

`--capture-consumers` takes the shorthand `name:key=value,key=value`
and can be repeated to register multiple consumers:

```bash
vllm serve meta-llama/Llama-3-8B \
    --capture-consumers filesystem:root=/mnt/nas/activations \
    --capture-consumers logging
```

The shorthand only accepts flat scalar values — no nested dicts or
lists in values. For richer configuration, use a YAML config file.

### YAML config

`--config path/to.yaml` maps keys onto the same `EngineArgs` fields
as the CLI, so `capture_consumers` in YAML is a list of shorthand
strings — one per consumer:

```yaml
model: meta-llama/Llama-3-8B
capture_consumers:
  - filesystem:root=/mnt/nas/activations,writer_threads=4
  - logging
```

The shorthand is the same `name:key=value,key=value` form the CLI
accepts, with the same flat-scalar limitation. For richer parameters
(nested dicts, multi-layer hook maps) use the Python API below or
build a `CaptureConsumersConfig` and pass it via
`EngineArgs.capture_consumers_config_override`.

To run multiple instances of the same consumer type, disambiguate
them with `instance_name` via the Python API:

```python
llm = LLM(
    model="meta-llama/Llama-3-8B",
    capture_consumers=[
        {"name": "filesystem", "instance_name": "primary",
         "params": {"root": "/mnt/nas/primary"}},
        {"name": "filesystem", "instance_name": "mirror",
         "params": {"root": "/mnt/nas/mirror"}},
    ],
)
```

`RequestOutput.capture_results` is keyed by `instance_name` when
present, otherwise by the entry-point `name`.

### Python `LLM(...)`

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-8B",
    capture_consumers=[
        {"name": "filesystem", "params": {"root": "/tmp/captures"}},
        {"name": "logging", "params": {"hooks": {"post_mlp": [0]}}},
    ],
)
```

Dict entries become `CaptureConsumerSpec`s on `VllmConfig` and flow
through the engine end-to-end.

The list also accepts pre-constructed `CaptureConsumer` instances
(e.g. a driver-side consumer that needs a live Python model or
optimizer); these must have `location = "driver"`. The LLM
constructor validates and stashes such instances, but the plumbing
between `LLM` and `VllmConfig` for pre-constructed instances is
currently incomplete — see
[Capture Consumers Design — Known Limitations](../design/capture_consumers.md#known-limitations).
Use the dict form unless you are working on that plumbing.

### Per-Request Capture

For consumers that set `reads_client_spec = True` (the filesystem
consumer, and any third-party consumer that opts in), clients drive
the capture by attaching a dict to `SamplingParams.capture`:

```python
from vllm import SamplingParams
from vllm.v1.capture.consumers.filesystem import FilesystemCaptureRequest

sampling_params = SamplingParams(
    max_tokens=16,
    capture={
        "filesystem": FilesystemCaptureRequest(
            request_id="probe_0001",
            tag="mnist-probe-v1",
            hooks={"post_mlp": [12]},
            positions="last_prompt",
        ),
    },
)
```

The key is the consumer's entry-point name (or its `instance_name`
if configured). The value is whatever the consumer accepts — the
consumer's own `validate_client_spec` parses it. For the filesystem
consumer, passing a dict with the same fields also works.

Requests that omit `capture` only receive captures from consumers
with global specs.

### OpenAI-compatible API

Send the per-request spec in the `extra_body.capture` field:

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3-8B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "extra_body": {
            "capture": {
                "filesystem": {
                    "request_id": "probe_train_0001",
                    "tag": "capital-probe",
                    "hooks": {"post_mlp": [12, 16, 20, 24]},
                    "positions": "last_prompt",
                },
            },
        },
    },
    timeout=60,
).json()
```

Validation happens at admission time; an invalid spec returns HTTP
400 with a descriptive error.

## Reading Results

On request completion, `RequestOutput.capture_results` is a
`dict[str, CaptureResult]` keyed by consumer instance name:

```python
from vllm import LLM, SamplingParams
from vllm.v1.capture.consumers.filesystem import FilesystemCaptureRequest

llm = LLM(
    model="meta-llama/Llama-3-8B",
    capture_consumers=[{"name": "filesystem", "params": {"root": "/tmp"}}],
)

sampling_params = SamplingParams(
    max_tokens=16,
    capture={
        "filesystem": FilesystemCaptureRequest(
            request_id="req1",
            tag="demo",
            hooks={"post_mlp": [0]},
            positions="last_prompt",
        ),
    },
)

[output] = llm.generate(["Hello"], sampling_params)
result = output.capture_results.get("filesystem")
if result is not None and result.status == "ok":
    for path in result.payload:
        print("wrote", path)
```

`CaptureResult` fields:

- `status`: `"pending"`, `"ok"`, `"partial_error"`, `"error"`, or
  `"not_requested"`.
- `error`: a human-readable message when `status != "ok"`.
- `payload`: consumer-specific. Filesystem returns a `list[str]` of
  written paths; other consumers return whatever they like.

On the OpenAI-compatible HTTP path, results are attached to the
response body as `capture_results`, mirroring the structure above.

## Limits

- **Tensor/pipeline parallelism**: the filesystem consumer rejects
  any request under `tensor_parallel_size > 1` or
  `pipeline_parallel_size > 1`. Third-party consumers that collect
  residuals should do the same — multi-rank collection is out of
  scope for the current framework.
- **Prefix-cache hits**: positions below
  `CaptureContext.num_computed_tokens` were served from the prefix
  cache and never forwarded through the model. Consumers reject such
  positions at admission; enforcement is the consumer's
  responsibility.
- **Hook coverage**: only the decoder architectures that wire the
  `apply_layer_steering` / `maybe_capture_residual` pair fire hooks.
  See [Activation Steering](steering.md) for the list of covered
  architectures — capture coverage matches the steering list.
- **Capture failures don't abort generation**: consumer errors
  surface as `partial_error` / `error` on the corresponding
  `CaptureResult`; text generation always completes.

## Writing a Consumer Plugin

Third-party consumers ship as separate Python packages. See
[Plugin Authoring Guide](../capture_consumers/plugin_authoring.md)
for the worked examples (quick-start consumer, driver-side training
loop, streaming consumer, tests).

Example plugins live under `examples/capture_consumers/`:

- `minimal_plugin/` — the simplest `CaptureConsumer` subclass; records
  the sum of every captured tensor.
- `activation_reward_producer/` — a direct `CaptureSink` that returns
  a cosine-alignment reward plus diagnostic fields on
  `CaptureResult.payload`. Designed for RL loops; the README covers
  vector drift, detection via the diagnostic payload, and the
  frozen-scorer deployment pattern.
