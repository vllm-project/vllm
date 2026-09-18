# Decode-side peer retirement: CPU design prototype

This is an RFC experiment, **not a working `/drop_peer` endpoint or a vLLM
serving patch**. It adds no production hooks and does not change engine behavior.
The goal is to review the safety contract before integrating it with the
scheduler, workers, and router.

## Deployment and operational problem

Our deployment uses separate Kubernetes Prefill/Decode Pods on a single
multi-GPU NVIDIA B300 host, with TP=2 per engine and an external P/D proxy.
Decode pulls Prefill KV through NIXL/UCX. The same-host data path uses CUDA IPC
over NVLink-connected GPUs; the configured transports also include `cuda_copy`
and `tcp`. The deployed engine baseline is vLLM 0.26.0.

We have observed memory remaining on a retired Prefill's GPU after its process
has exited. Restarting Prefill alone has not reliably reclaimed that capacity;
our current operational workaround has been to stop **all Decode instances**.
That prevents reliable independent P scale-down/replacement and couples a P
rolling update to an outage of the D pool. Kubernetes can issue the scale
command, but cannot necessarily make the old GPU capacity reusable.

Surviving D workers retaining imported IPC mappings or transport references is
the working hypothesis. This CPU experiment does not reproduce that native
condition or prove a driver bug. We have not identified each native mapping
holder in a controlled GPU reproduction. The goal is to release every holder's
references, not to prescribe that every D must always be stopped.

Peer state and allocation references are distinct layers, not necessarily two
independent bugs. Removing NIXL metadata may initiate backend disconnect, but
does not by itself prove that native CUDA IPC mappings are closed or that every
D/rank has released its references. The target is reclaiming the retired P's
capacity, not merely clearing its registry entry.

## Proposed contract

1. The router declares an exact old P generation retired and stops new routing
   to it. One request failure is not a sufficient death signal.
2. Block new remote reads for that generation. A replacement at the same address
   must have a distinct generation.
3. Serialize cleanup with the connector worker and handshake callbacks. Return
   `busy` while target receives, callbacks, or unattributed handles remain.
4. After draining, release remote descriptor lists and remove remote agents via
   the existing worker cleanup method. Do not free active transfer handles.
5. Make retries idempotent, and preserve partial cleanup errors rather than
   mistaking missing Python dictionaries for successful native cleanup.
6. Require acknowledgement from every relevant D worker/rank. Python cleanup
   return, native endpoint close, and observed GPU memory reclamation are
   different milestones.

`peer_retirement.py` models this contract. Its admission gate is only exercised
by the tests; it is **not connected to the real scheduler**. Callers must be
serialized on the connector worker thread. This prototype does not expire
tombstones, persist state, cancel native transfers, or support arbitrary
bidirectional/push connector state. A crashed peer whose transfer never finishes
will remain `busy`; safe cancellation and escalation are still open work.

## What runs without a GPU

`source_harness.py` AST-extracts these unmodified methods from the checkout:

- `NixlBaseConnectorWorker._cleanup_remote_engine`
- `NixlBaseConnectorWorker._evict_stale_engines`
- `NixlBaseConnectorWorker._engines_with_inflight_transfers`

The pinned source revision is
`0136df94b0d75732be6c66f620f51289c888e41d`, with `base_worker.py` SHA-256
`5f9a4f45edd38fe73eede0e8b7be8147b02c141cea2f2efc0f352e38c5380102`.
The harness refuses a different hash so a rebase requires source re-review.
This is a main-revision method experiment, **not a vLLM 0.26.0 regression test**.

NIXL calls, imported allocations, topology, worker initialization, and callback
events are simulated. The harness imports no torch/CUDA/native NIXL and does not
run an engine. A fake delayed native close specifically demonstrates why
`remove_remote_agent()` returning is not proof that GPU memory is reclaimed.

The example-local tests are separate from the ordinary connector suite because
this prototype intentionally avoids vLLM's platform initialization. Once the
feature is integrated, its regression tests should move into the normal suite.

## Reproduce

From the repository root, prepare an isolated CPU environment:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python pytest==9.1.1 ruff==0.14.0
```

Run the example tests without loading vLLM's GPU-dependent root conftest or
project-wide pytest options:

```bash
.venv/bin/python -m pytest -c /dev/null -p no:cacheprovider \
  --confcutdir=examples/disaggregated/peer_retirement_cpu \
  examples/disaggregated/peer_retirement_cpu/test_peer_retirement.py -v
.venv/bin/python examples/disaggregated/peer_retirement_cpu/source_harness.py
.venv/bin/python -m ruff check examples/disaggregated/peer_retirement_cpu
.venv/bin/python -m ruff format --check examples/disaggregated/peer_retirement_cpu
```

Initial CPU result: **31 passed**, on macOS with Python 3.12.14 and pytest 9.1.1.
Coverage includes cleanup order, pending receives/handshakes, the callback gap
after a handshake future is removed, orphan handles, healthy-peer isolation,
duplicate calls, partial native failures, multiple simulated D/rank holders,
and delayed close. This is not model-serving or GPU validation.

Expected demo output:

```text
SIMULATION ONLY: no GPU, no vLLM server, no native NIXL loaded
P exited; simulated allocation resident: True
D Python cleanup result: cleanup_returned
Before simulated native close: True
After simulated native close: False
Real native release verified: False
```

## Open work before implementation review

- Coordinate API/metadata changes with the existing peer-lifecycle PRs below.
- Add real scheduler/worker admission fencing and all-rank acknowledgements.
- Resolve stuck native transfers without freeing resources still in use.
- Validate actual loaded NIXL/UCX libraries and native endpoint/cache teardown.
- On isolated GPUs, reproduce P retirement with surviving D holders and measure
  old-P GPU memory before and after cleanup without restarting healthy D.
- Check graceful drain, crashed P, same-address replacement, concurrent handshakes,
  healthy local D generation, streaming/non-streaming output, and performance.

## Related work

- [#56341](https://github.com/vllm-project/vllm/pull/56341) is the P-side
  `/drop_peer` draft for dead D; its description explicitly excludes the
  dead-P/live-D direction explored here. This example does not duplicate that
  HTTP endpoint.
- [#50047](https://github.com/vllm-project/vllm/pull/50047) addresses dead/stale
  peer recovery, while [#55471](https://github.com/vllm-project/vllm/pull/55471)
  covers locally invalidated pull metadata with a still-live producer.
- [#54689](https://github.com/vllm-project/vllm/pull/54689) protects peers still
  referenced by in-flight reads; the eventual feature must preserve that rule.

AI assistance was used for source analysis, prototype code, tests, and writing.
The submitting human has confirmed review and local test execution. GPU
validation and production integration remain pending; this draft is not ready
to merge as a serving fix.
