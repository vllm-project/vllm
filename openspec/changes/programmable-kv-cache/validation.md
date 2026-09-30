# Implementation validation

This is an execution record, not a claim that the full change is complete.
Local dates use Asia/Shanghai. All Python commands use the repository `.venv`.

## October 1: control record foundation

Implemented immutable correlated operation snapshots, explicit engine/epoch/path
validation, whole-envelope deduplication, ID/content conflicts, bounded admission,
terminal TTL, progress/terminal transitions and bounded event replay. Completions
carry an operation incarnation so a late completion cannot update reused IDs.
Added fixed native payload, target and capability records, including strict
action versions, unknown-field rejection, group-specific content validation,
and explicit follow-growth scope. Native actions are not yet connected to serving
or physical execution.

The complete relevant baseline command was:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/test_kv_hint_control.py \
  tests/v1/test_request.py \
  tests/v1/kv_offload/cpu/test_manager.py \
  tests/v1/kv_offload/tiering/test_kvcr_tier.py \
  tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py \
  tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py \
  tests/v1/kv_connector/unit/test_mooncake_store_worker.py \
  tests/v1/kv_connector/unit/test_mooncake_store_layout.py \
  tests/v1/kv_connector/unit/test_mooncake_store_connector.py \
  tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py -q
```

Result: **412 passed, 1 skipped**, 146.83 seconds. The Mooncake suites use mocks;
this does not prove real Store transfer or cross-engine reuse. The KVCR module was
initially skipped because its optional package was absent; it was installed and
tested separately below.

After strengthening stale-completion handling, reran the full control suite:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/test_kv_hint_control.py -q
```

Result: **36 passed**, 10.64 seconds.

After adding native payload/target validation, the same complete control suite
passed **73 tests**, 20.10 seconds.

Existing real CPU/GPU worker transfer correctness:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/kv_offload/cpu/test_gpu_worker.py -q
```

Result: **40 passed, 1 skipped**, 13.81 seconds. The skipped case is ROCm-specific.
This verifies existing transfer behavior, not new request-free execution.

Cloned the official `ai-dynamo/kvcr` repository into `../kvcr`, pinned by the
checkout at `317e62f301300c101bf9c911e020fb89b35f1583`, and installed it editable
using `.venv/bin/uv pip install --python .venv/bin/python -e ../kvcr`.

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/kv_offload/tiering/test_kvcr_tier.py -q -rs
```

Result: **20 passed**, 6.09 seconds; no skip. This covers the existing adapter
contract and framework-reference callbacks with its existing test doubles.

Ruff 0.14.0 formatting, normal pre-commit hooks, and mypy 3.10/3.12 checks passed for
the changed files. OpenSpec strict validation also passed. GitHub HTTPS timed out, so Git fetching uses the
existing authenticated SSH transport through per-command configuration. A slow
ShellCheck binary download required a retry; its installation subsequently
finished. The final hook run uses the normal configuration with no hook disabled.
Buildkite runs the new control suite in the existing V1 job.

## Environment and outstanding integration

Editable precompiled vLLM and all `requirements/test/cuda.in` dependencies are
installed. Python 3.12.7, uv 0.12.21, Torch 2.13.0+cu132, Transformers 5.17.0;
local A100 GPUs 0 and 3, driver 595.71.05.

Qwen/Qwen3.8-27B is pinned to revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The supplied download function is used
without executing its unrelated model list. Direct Hugging Face access timed out;
the public mirror works for metadata. Xet downloads failed; ordinary HTTP download
is being retried. Model validation requires every indexed weight shard, not just
the presence of `config.json`.

Pending: real Mooncake deployment/transfer, native control execution, model output
evaluation, full workload comparisons and the external large-model matrix.

## October 1: logical state and independent retention

Added a bounded session/continuation index with captured parent prefixes,
group-aware range resolution, reverse membership, revision-checked mutation,
task-tree cleanup, explicit history pruning, paginated queries and association
events. Known empty content remains distinguishable from unavailable history.
These records contain no reusable physical block IDs.

Added independent priority/TTL declarations, snapshot and explicit follow-growth
scope, release and coordinated logical task cleanup. Closing a followed prefix
preserves another owner's final scope and original expiry. Following declarations
cannot attach to a reused continuation identity. These modules still await engine
request/cache hooks and the common command executor.

The new state suite tests independent shared-prefix lifetimes, branch stability,
group boundaries, stale writes/cleanup, atomic capacity refusal, revision-safe
pagination, event gaps, explicit pruning and seeded declaration interleavings.
Existing full prefix-cache, KV-manager, deferred-free and scheduler suites run
alongside the control/state suites:

```bash
CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python -m pytest \
  tests/v1/test_kv_hint_control.py tests/v1/test_kv_hint_state.py \
  tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/core/test_deferred_block_free.py \
  tests/v1/core/test_scheduler.py -q
```

Initial result: **503 passed, 1 failed**, 354.74 seconds. The existing CPU-marked
`test_diffusion_read_deferral_keeps_a_longer_pp_wait` constructs PP=3 without
workers, but the real platform count sees only the two allocated GPUs. Corrected
that test's hardware isolation by supplying a mocked device count of three;
kept all PP scheduling assertions intact and did not expose additional GPUs.
The complete suite is rerun after this fixture correction.

Final result: **504 passed**, 324.09 seconds. Normal pre-commit hooks and mypy
3.10/3.12 checks passed for the changed state/retention and test files.
