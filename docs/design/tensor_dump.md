# General tensor dumper

The dumper ports SGLang's general debugging framework at commit
`3e914e69d5cea497d458d1f90359b326a12ba654`. It does not depend on SGLang.
The framework adapter translates vLLM batch tensors and parallel ranks; the
capture, filtering, gradients, parameter dumps, source patching and grafting
contracts follow that implementation.

## Capture arbitrary modules

```bash
DUMPER_ENABLE=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
DUMPER_NON_INTRUSIVE_MODE=all \
DUMPER_DIR=/tmp/dumper \
DUMPER_EXP_NAME=baseline \
vllm serve MODEL --enforce-eager
```

`all` captures tensor inputs and outputs of every named module. `core`, the
upstream default, captures only recognized batch fields. `off` disables automatic
hooks while retaining explicit `dumper.dump(...)` calls.

Use `DUMPER_FILTER` to limit captures, for example:

```bash
export DUMPER_FILTER="search('self_attn', name) is not None and step < 2"
```

Each `.pt` file contains `value` and `meta`, with tensor name, step, rank and
available parallel topology. Files are compatible with SGLang dump readers.
`DUMPER_INCLUDE_PARALLEL_RANK_IN_FILENAME=1` also places parallel ranks in filenames.
Set an explicit, shared experiment name for independently scheduled DP workers;
automatic experiment naming requires all distributed ranks to participate.

## Runtime control

Set `DUMPER_SERVER_PORT=reuse` at startup to expose `/dumper/configure`,
`/dumper/get_state` and `/dumper/reset` on the serving API. These use vLLM's
existing worker collective RPC rather than another transport. The serving API's
authentication also applies to these routes.

```bash
curl -X POST http://localhost:8000/dumper/configure \
  -H 'Content-Type: application/json' \
  -d '{"enable": true, "non_intrusive_mode": "all"}'
```

Choose `DUMPER_NON_INTRUSIVE_MODE=all` **at startup** when enabling automatic
captures dynamically. Changing that setting does not reinstall existing hooks.
As in SGLang, `reset` removes hooks as well as resetting counters; it is not a
"clear files and resume" operation.

A numeric `DUMPER_SERVER_PORT` starts the upstream-style standalone HTTP/ZMQ
control transport. It is unauthenticated and requires an isolated, trusted
network. Filters and source patches are developer-supplied code, not a sandbox.
Dump files may contain sensitive prompts, activations and model parameters.

## Explicit captures and comparison

Import `dumper` from `vllm.utils.debug_utils.dumper`. The upstream APIs include
`dump`, `dump_dict`, `dump_model`, `set_ctx`, `ctx`, `step` and `configure`.
Gradient and parameter captures use the corresponding `DUMPER_ENABLE_*`
settings. Source patching uses `DUMPER_SOURCE_PATCHER_CONFIG`; grafting uses
the `DUMPER_GRAFTER_*` settings and requires coordinated baseline/target processes.

The lightweight comparator is available as
`python -m vllm.utils.debug_utils.dump_comparator` with optional `polars` installed.
The full SGLang comparator, including unsharding and visualization, is not ported.

Serving integration currently requires eager Model Runner V2. Ordinary Python
hooks do not execute during CUDA Graph replay. GPU serving, distributed controls
and grafting require separate integration validation beyond the CPU contracts.
