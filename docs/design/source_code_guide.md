# Source Code Guide

This guide is a reading map for contributors who are new to the vLLM
codebase. It describes the package layout, follows a request through the main
runtime components, and records the compatibility contract used while source
files move to clearer ownership domains.

The layout is intended to make source navigation easier and to provide a
reference for future package evolution. It does not change the public API or
the behavior of the inference engine.

[TOC]

## Package map

The Python package is organized around four conceptual domains:

```text
vllm/
├── foundation/             # Shared services with no inference ownership
│   ├── config/
│   ├── system/
│   ├── observability/
│   ├── utilities/
│   ├── integrations/
│   ├── extensibility/
│   └── devtools/
├── frontend/               # External representations and serving interfaces
│   ├── compat/engine/
│   ├── entrypoints/
│   └── processing/
├── runtime/                # State and policy for an inference run
│   ├── modeling/
│   ├── generation/
│   └── execution/
├── backends/               # Hardware and execution mechanisms
│   ├── platform/
│   ├── compiler/
│   ├── compute/
│   └── distributed/
├── v1/                     # Runtime vertical stack, kept in place
├── model_executor/         # Model implementation stack, kept in place
├── models/                 # Next-generation model implementations
├── third_party/            # Vendored code with path-sensitive probes
└── vllm_flash_attn/        # Build-system and extension-module namespace
```

### Frontend

`vllm.frontend` owns the boundary between user-facing representations and
runtime representations. It contains offline and online entrypoints, protocol
servers, request types, tokenization, prompt rendering, multimodal input
processing, and output parsers.

The main subpackages are:

- `frontend/entrypoints`: `LLM`, CLI commands, API servers, and launchers.
- `frontend/processing`: input contracts, tokenizers, renderers, multimodal
  processing, reasoning parsers, tool parsers, and public output types.
- `frontend/compat`: public engine interfaces whose implementation is provided
  by the V1 runtime.

Structured-output token constraints remain under `vllm/v1/structured_output`
because they participate directly in token generation. Reasoning and tool-call
parsers belong to the frontend because they interpret output representations.

### Runtime

`vllm.runtime` owns state and policy associated with an inference run. The
current physical package contains shared modeling state, generation data, and
forward-pass context. The complete V1 runtime vertical stack remains under
`vllm.v1` and is documented conceptually as:

- engine and lifecycle: `vllm/v1/engine`
- scheduling and KV cache: `vllm/v1/core`
- execution: `vllm/v1/executor` and `vllm/v1/worker`
- generation: `vllm/v1/sample`, `vllm/v1/spec_decode`, and
  `vllm/v1/structured_output`
- observability and recovery: `vllm/v1/metrics` and
  `vllm/v1/fault_tolerance`

### Backends

`vllm.backends` contains mechanisms selected by runtime policy:

- `backends/platform`: device detection and platform implementations.
- `backends/compiler`: graph compilation, passes, and CUDA graph support.
- `backends/compute`: IR, Python kernels, scalar types, and DSL helpers.
- `backends/distributed`: collectives, device communicators, transfer
  connectors, expert parallelism, and Ray integration.

Model architecture implementations and layer kernels stay under
`vllm.model_executor` for now. Their registry and dynamic import paths make
them a separate migration problem.

### Foundation

`vllm.foundation` contains shared services that do not own a stage of
inference. These include configuration, environment access, logging, tracing,
general utilities, Hugging Face integration, plugins, and profiling helpers.

## Reading the main request path

A useful first pass through the code follows one request from a public API to
an output:

```text
vllm.frontend.entrypoints
    -> vllm.frontend.processing.inputs
    -> vllm.frontend.compat.engine
    -> vllm.v1.engine
    -> vllm.v1.core (scheduler and KV cache)
    -> vllm.v1.executor
    -> vllm.v1.worker
    -> vllm.model_executor
    -> vllm.v1.attention and vllm.backends.compute
    -> vllm.v1.sample
    -> vllm.frontend.processing.outputs
```

For offline inference, start with
[`LLM`](../../vllm/frontend/entrypoints/llm.py). For online serving, start with
the CLI and launchers under `vllm/frontend/entrypoints`. Both paths eventually
reach `EngineCore`, which coordinates scheduling, KV-cache management, and
worker execution.

## Legacy import compatibility

New package paths are canonical for internal imports. Existing import paths
remain available through generated compatibility shims so downstream users do
not need to update imports as part of this reorganization.

Representative mappings are:

| Legacy path | Canonical path |
| - | - |
| `vllm.config` | `vllm.foundation.config` |
| `vllm.utils` | `vllm.foundation.utilities` |
| `vllm.transformers_utils` | `vllm.foundation.integrations.transformers_utils` |
| `vllm.plugins` | `vllm.foundation.extensibility.plugins` |
| `vllm.entrypoints` | `vllm.frontend.entrypoints` |
| `vllm.inputs` | `vllm.frontend.processing.inputs` |
| `vllm.multimodal` | `vllm.frontend.processing.multimodal` |
| `vllm.engine` | `vllm.frontend.compat.engine` |
| `vllm.platforms` | `vllm.backends.platform` |
| `vllm.compilation` | `vllm.backends.compiler` |
| `vllm.distributed` | `vllm.backends.distributed` |
| `vllm.kernels` | `vllm.backends.compute.kernels` |
| `vllm.lora` | `vllm.runtime.modeling.lora` |

The complete mapping is stored in
[`tools/package_refactor/mapping.json`](../../tools/package_refactor/mapping.json).

Single-file moves and leaf modules use a `sys.modules` alias, so importing the
legacy and canonical paths returns the same module object. Package roots use
lazy `__getattr__` delegation to avoid eager imports and circular dependencies.
For package roots, the two module objects can differ, but exported classes,
functions, and constants are the same objects.

The compatibility contract covers public Python APIs, CLI commands, plugin
group names, dynamic registries, and importable classes. It does not make old
paths canonical for new internal code.

## Packages kept in place

Some packages have intentionally not moved:

- `vllm.v1` is a tightly connected runtime stack with dynamic backend paths.
- `vllm.model_executor` and `vllm.models` are coupled to model registry strings.
- `vllm.third_party` participates in path-based optional-feature probes.
- `vllm.vllm_flash_attn` is tied to compiled extension names.
- Root custom-op modules retain their established registration namespace.

Moving these packages safely requires separate design discussions and focused
build or registry changes.

## Validation

The reorganization was compared with commit
`648b7468b8e10dc7baf05554f6772136f7a52a46`. Validation included public and
semi-public import sweeps, model and attention registry probes, focused unit
tests, scheduler and KV-cache tests, and local Qwen3-0.6B inference.

For Qwen3-0.6B, greedy, seeded sampling, batched prompts, shared-prefix prompts,
and a 128-token decode produced identical token IDs, text, finish reasons, and
token counts. A separate live 32-token greedy run also matched the saved
baseline exactly.

These results show that no behavioral difference was observed on the tested
surface. They are not a proof that every optional backend, hardware platform,
or downstream integration is equivalent.

## Reference status

This layout is a source-navigation aid and an architectural reference. The
compatibility layer allows package ownership to become clearer without forcing
downstream users to migrate immediately. Further physical moves should be
proposed and validated independently, especially for the V1 runtime, model
registry, compiled extensions, and kernel implementations.
