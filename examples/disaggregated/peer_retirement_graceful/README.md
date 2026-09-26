# Graceful Prefill Retirement with UCX v1.22.x

This is the **version-pinned experimental implementation**, replacing the
earlier CPU-only sketch. It exposes explicit D-side cleanup using existing
vLLM/NIXL resource-release functions. It does not add a production vLLM API or
automatically change Kubernetes, Docker, a router, or any running service.

Supported here: **normal drain**, vLLM **0.26.0**, one API process per engine,
unidirectional NIXL pull, separate TP2 P/D instances. The validated transport
was NIXL 1.4.1 with UCX v1.22.x commit
`5288e74cd40be622e109489de7bb80d821341bd5`, `num_threads=8`, and
`UCX_CUDA_IPC_CACHE=n`. A newer checkout or similarly named version is not an
equivalent validated build. Source patching refuses an unexpected file hash.

## Motivation

On a single multi-GPU host, a surviving Decode may retain imported CUDA IPC
resources after Prefill exits. The exited P's GPU memory then remains occupied
and cannot be reliably reused. Stopping the entire D pool can release it, but
also interrupts unrelated work and prevents independent N-P/M-D scaling.

Disabling the UCX mapping cache is not sufficient while live references remain.
The missing operation is retiring the exact old P generation on every relevant
D worker after work has drained. It must not depend on a replacement P loading
and handshaking first. See [the GPU evidence](VALIDATION.md).

## Files and Contract

| File | Purpose |
| --- | --- |
| `peer_lab.py` | Worker-thread prepare/commit, exact-generation fence, busy checks and existing NIXL cleanup |
| `peer_lab_api.py` | Authenticated experimental middleware; all-rank checks and D admission fence |
| `patch_vllm.py` | Hash-pinned v0.26.0 guard patches; no native library changes |
| `prepare_sources.py` | Fetch or copy two public, SHA-256-checked source files for CPU tests |
| `test_retirement.py` | Runtime/API tests using actual pinned vLLM methods and a fake native boundary |
| `Dockerfile` | Optional isolated image layer over a caller-provided matching native stack |

The source patches block retired-generation handshakes, reads and heartbeats,
prevent TTL cleanup while the worker is busy or the generation is fenced, and
remove notification-only receive metadata that otherwise prevents quiescence.
They do not change weights, kernels, scheduling policy or normal request data.

The successful lifecycle is:

1. Stop new router admissions to the selected P. Keep P alive for admitted work.
2. Wait for **complete responses** from that P's already admitted requests.
   This example does not support cleanup after a disconnected/aborted response.
3. Stop new admissions to one D being cleaned and wait until it has no active
   HTTP requests. Route background work to the other D. This is stricter than
   waiting only for KV handoff; active local generation on the target D was
   not covered by the GPU acceptance run.
4. Snapshot and verify that all ranks agree on the D generation and that their
   Python-visible connector queues are empty. Check native transfer/notification
   error metrics and engine logs before cleanup; an HTTP 200 is not sufficient.
5. Prepare the exact old P generation on all ranks, then commit on all ranks.
   The worker releases prepared descriptor-list handles before invoking
   `remove_remote_agent()` through `_cleanup_remote_engine()`.
6. Verify all acknowledgements. Repeat for every D that imported from this P.
7. Normally stop P only after cleanup, then measure its GPU memory. Confirm D
   process identities are unchanged and another P can perform fresh transfers.

`cleanup_returned` deliberately reports `native_release_verified=false` and
`gpu_memory_measured=false`. It is not a native close completion certificate.
The external GPU observations are necessary to establish memory reclamation.

## Safety Limits

- Use isolated, single-API instances and private loopback-only management. Never
  expose `/peer_lab/control` through the serving gateway. The middleware blocks
  alternate inference/development paths rather than letting them bypass fencing.
- Both `--worker-extension-cls peer_lab.WorkerExtension` and
  `--middleware peer_lab_api.AdmissionFence` are required on D. Enabling only
  the worker extension is not a safe admission protocol.
- Partial rank success or timeout keeps the whole-D admission freeze. A native
  release may have partially succeeded, so it is not blindly retried. This
  prototype has no automated recovery from unknown native completion.
- Zero active HTTP requests plus Python-visible quiescence is not a native
  all-thread barrier. Abrupt P death/OOM, incomplete-transfer cancellation,
  forced unmapping, multi-API and bidirectional/push operation are unsupported.
- The middleware requires a dedicated management key of at least 32 characters.
  Generate it locally, keep it out of logs/source control and never reuse a
  production credential. `PEER_LAB_RETIRE_ENABLED=0` keeps cleanup disabled.
- The body limit is 32 MiB. A 1,024-generation tombstone cap refuses further
  retirements rather than silently forgetting fences. These are experimental
  guardrails, not an unbounded production lifecycle service.
- No call to `uct_cuda_ipc_destroy_cache_by_iface_address()` is added. Existing
  native reference lifetimes and UCX cache policy still determine unmapping.

## CPU Tests

From the repository root, in an isolated environment:

```bash
uv venv --python 3.12
uv pip install pytest starlette ruff pre-commit
EXAMPLE=examples/disaggregated/peer_retirement_graceful
.venv/bin/python "$EXAMPLE/prepare_sources.py"
.venv/bin/python -m pytest -c /dev/null -p no:cacheprovider \
  --confcutdir="$EXAMPLE" "$EXAMPLE/test_retirement.py" -q
.venv/bin/python -m ruff check "$EXAMPLE"
.venv/bin/python -m ruff format --check "$EXAMPLE"
```

The preparation step alone accesses the network. Alternatively pass
`--source-dir /path/to/v0.26.0/vllm/distributed/kv_transfer/kv_connector/v1/nixl`.
Tests are then offline and need neither CUDA nor a full vLLM installation. The
source files are ignored, not vendored into this PR; both hashes are checked
before preparing fixtures. These method-level tests mock NIXL and do not prove
GPU release or replace upstream's full-import connector tests.

## Isolated Image and API Exercise

Prepare a **non-production** parent image with stock vLLM 0.26.0, NIXL 1.4.1
and the intended UCX build. Pin its digest. This Dockerfile only packages the
tested Python integration; it does not fetch, upgrade or build UCX:

```bash
docker build --pull=false --network=none \
  --build-arg BASE_IMAGE='<verified-parent-image@sha256:digest>' \
  -t peer-retirement-lab:local \
  examples/disaggregated/peer_retirement_graceful
```

Keep the working P/D launch configuration and exact model revision unchanged.
For the isolated D append `--host 127.0.0.1`,
`--worker-extension-cls peer_lab.WorkerExtension`, and
`--middleware peer_lab_api.AdmissionFence`; use one API process. Set:

```bash
export UCX_CUDA_IPC_CACHE=n
export UCX_TLS=tcp,cuda_ipc,cuda_copy
export PEER_LAB_TP_SIZE=2
export PEER_LAB_RETIRE_ENABLED=1
export PEER_LAB_ADMIN_KEY='<new-private-lab-key-at-least-32-characters>'
```

Before transferring, verify `ucx_info -v`, `ucx_info -f`, and the worker's actual
loaded libraries. The authenticated `libraries` operation returns worker paths,
not proof by itself that a given ABI or native teardown is safe.

After normal P/D traffic establishes the peer, and **only after the drain and
error checks above**, use the isolated D's loopback HTTP port:

```bash
curl --fail-with-body --max-time 40 http://127.0.0.1:8200/peer_lab/control \
  -H "Authorization: Bearer $PEER_LAB_ADMIN_KEY" \
  -H 'Content-Type: application/json' -d '{"phase":"snapshot"}'
```

Use the exact old P engine ID, D engine ID, and one unique operation ID. Send
the following payload first with `phase=prepare`, then `phase=commit` to the
same management URL and with the same authenticated headers:

```json
{
  "phase": "prepare",
  "target_engine_id": "old-prefill-generation",
  "decode_engine_id": "this-decode-generation",
  "operation_id": "unique-retirement-operation"
}
```

Check HTTP status **and** that every TP rank has the expected generation,
operation and result (`prepared` then `cleanup_returned`). A timeout is an
unknown outcome, not permission to continue. Do not delete P early, omit a D,
force-unmap, or erase a failure fence to make the test pass.

## Related Work

- #50047 cleans a replaced pull peer after successful same-address handshake
  by a different engine ID. This example instead explicitly retires a drained
  P even when no replacement will be started. Both reuse peer cleanup.
- #56341 handles the opposite direction: P-side cleanup after D death.
- #55471 recovers locally missing metadata while P remains alive.

The example is opt-in and version-pinned so reviewers can inspect the tested
contract without silently changing current-main serving behavior. A production
implementation needs integration with the upstream lifecycle API, native
completion semantics and additional CPU/GPU regression coverage.
