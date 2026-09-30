# Follow-up: Completion, Processing Plans, and RDT

Thank you for the detailed response. We agree that coverage tracking is a
more general completion mechanism than counting or cold-load call matching.
We would like to clarify the input-layout contract and implementation scope,
and explore a shared reload execution layer.

## Slots: Agreed Limitations and Existing Work

We agree with these observations about the current slot-based interface:

- Loader-observed slots do not directly cover sharded RDT or `nccl_m2n`
  writes into parameters.
- Matching slots against cold-load arrivals constrains reload to the
  observed calling convention. A fused checkpoint and separately sent
  Q/K/V tensors can have different calls even when their destination
  coverage is equivalent.
- Slots prove that logical load applications occurred, not that every
  intended destination byte was written correctly.
- With `load_format="dummy"`, ordinary initialization provides no cold-load
  arrivals from which to build the expected slots.

EPLB is no longer simply an unsupported item in our implementation. We support
expert placement changes between updates, with placement held stable during
each update. The separate coordination PR is
[EPLB reload gating and synchronization, #59063](https://github.com/vllm-project/vllm/pull/59063).
It includes DeepSeek V4.1 dynamic expert mapping support.

We have also run the DSV4.1 trace reload workflow successfully on H200:
checkpoint A -> reload B matched a fresh load of B after resetting the
prefix/KV cache. The recorded configuration uses a reduced five-layer model,
TP=1, DP=4, expert parallelism, and
`FLASHINFER_CUTLASS_MXFP4_BF16`. This validates that configuration, not every
EPLB/backend combination. The model result alone should not be read as proof
of arbitrary concurrent EPLB movement.
See the [validation report](https://github.com/new-TonyWang/vllm/blob/e494daa3e497122416908c8ce636e97bb3fd463b/docs/design/weight-update/fp8-reload-results.md#dsv41-h200-day0-reload-validation).
Some configurations still need coordinated collectives; we do not claim
unrestricted support.

For partial updates, tracking a `weight_version` per weight is a good idea.
It would make the intentional mixture of updated and unchanged weights
explicit, rather than implying that one engine-level generation means every
weight was updated.

For dummy initialization, our earlier proposal was a metadata-only cold-load
probe: run the weight-loading/routing path with placeholder tensors, mocking
the data-moving operations such as `copy_` so no real weight copy occurs.
This builds the load contract without a real checkpoint load. The existing
proposal is
[dummy-load manifest probing, #50564](https://github.com/vllm-project/vllm/pull/50564).
We agree that this is additional work rather than a capability automatically
provided by observing an ordinary dummy initialization.

## Coverage: Which Layout Is Being Covered?

We agree with replacing raw element counts with destination-region coverage.
However, we should distinguish two cases.

### Incoming Data Already Has the Runtime Layout

For ordinary BF16 weights where incoming and runtime layouts coincide,
coverage can track writes directly into the destination tensor. Once the
required region is covered, the weight is complete.

Even here, dtype equality alone is not sufficient: TP slicing, permutations,
or model-specific layout transforms may still be necessary.

### Incoming Data Needs Conversion

For most quantized paths, the received checkpoint-format data is not the
runtime representation. For example, BF16 input can become FP8 weights and
scales, followed by backend-specific packing or permutation.

In this case, the completion ledger must cover the **received input in its
loading layout**, not merely the final runtime tensor:

```text
receive checkpoint-format pieces
    -> write loading-layout staging
    -> record input coverage
    -> required inputs complete
    -> run the conversion/PWAL reload path
    -> land runtime outputs into bound live tensors
    -> refresh derived state
```

Staging may use temporary buffers, or safely reuse live storage when its
capacity and aliasing allow it. Coverage must describe the input view in
either case, rather than confusing its geometry with the runtime target.

Completion may also require several related inputs together: weight,
weight scale, activation scale, and any backend-specific metadata. Full
coverage of one weight alone does not prove that the conversion can run.

This does not require a separate coverage algorithm for each backend. It does
require each supported conversion path to expose its input contract, runtime
outputs, and postprocessing lifecycle. That is the role of our processing
plans, such as
[Fp8MoEProcessingPlan](https://github.com/new-TonyWang/vllm/blob/e494daa3e497122416908c8ce636e97bb3fd463b/vllm/model_executor/layers/quantization/utils/fp8_processing.py)
and the corresponding Humming/Marlin plans. The conversion should be shared
with cold PWAL, not independently reimplemented for coverage.

### Coverage's Position in the Reload Stack

Coverage should be the low-level data-copy and region-tracking engine between
weight transport and backend processing. It is not a replacement for
`weight_loader`, PWAL, or a quantization backend. Higher-level reload logic can
build slot/key and per-tensor semantics on top of it.

Its upstream inputs are the transport-specific arrival events:

```text
file/loader path, NCCL/IPC, or RDT scatter
    -> destination parameter or loading-layout staging view
    -> coverage copies the data and records the received region
```

Its downstream outputs are readiness events for the reload executor:

```text
required input regions complete
    -> invoke the module's processing plan / PWAL
    -> land runtime-format outputs into bound live tensors
    -> refresh derived state
    -> publish module completion
```

The higher-level layer can then map low-level regions to logical progress:

```text
coverage regions
    -> slot/key or per-tensor completion
    -> per-parameter and per-shard progress
    -> user-visible reload status
```

In the common case, cold loading and reload use the same weight-loading logic.
That makes it possible to observe the cold-load `slot`/`key` contract once and
reuse it during reload to determine which logical shards have arrived. This is
also the most useful layer for reporting progress because it preserves the
semantic identity of a Q/K/V shard, an MoE expert, or a quantization input.

If a transport or model path genuinely uses a different `weight_loader` during
reload, the reload can bypass the logical slot/key layer and submit its
destination regions directly to the lower-level coverage engine. This gives us
a general fallback without requiring every transport to reproduce the cold
loader call sequence.

However, we should verify how often this case occurs in practice before making
it a central design constraint. The fact that RDT writes directly into
parameters does not by itself prove that cold load and reload use different
`weight_loader`s; it may instead mean that the same loading geometry is
replayed through a different transport. We should identify concrete model or
backend examples where the logical loader contract actually differs between
cold load and reload, and distinguish those from cases where only the
transport path differs.

For a direct runtime-layout update, coverage can record writes directly on the
live destination. For a quantized or packed path, it records the
checkpoint-format input regions in temporary staging, then triggers that
backend's processing plan once all required inputs are complete. The plan
owns the conversion details; coverage only decides whether the inputs are
complete and hands the executor the corresponding readiness event.

This gives the stack a clear division of responsibility:

```text
transport:
    decide what data to send and perform the transfer

coverage:
    record which input regions arrived and publish readiness

processing plan / PWAL:
    convert checkpoint layout to runtime layout

reload executor:
    enforce dependencies, landing, refresh, and failure handling
```

The common coverage interface can therefore cover central loader writes and
RDT scatters without requiring every model to be rewritten. Individual
loading paths and backend processing plans still need explicit integration
where they expose non-standard input regions or conversion behavior.

### Runtime-Layout Transport

We should also account for transports that send weights already in the
runtime layout. The proposed
[direct reload mode](https://github.com/aoshen02/vllm/pull/59) is an example:
the sender provides tensors in the layout consumed by the running model, so
the receiver can copy them directly into the live parameters and buffers
without moving modules to meta or re-running PWAL.

This is a different path from checkpoint-layout coverage:

```text
checkpoint-layout transport:
    receive input regions
    -> coverage complete
    -> PWAL / processing plan
    -> landing and refresh

runtime-layout transport:
    receive runtime regions
    -> coverage complete
    -> copy into live storage
    -> identity/layout validation
    -> publish completion
```

Coverage should support both modes. In runtime-layout mode it remains the
low-level copy and completeness layer, while the reload executor validates
that the destination `data_ptr`, shape, stride, and dtype remain unchanged.
The direct mode must be selected only when the model does not require
post-load conversion or derived-state refresh; otherwise skipping PWAL would
leave the runtime representation stale or undefined.

The sender-side completeness contract is also insufficient for a generic
reload abstraction. Coverage should still identify which destination regions
arrived, report missing regions, and preserve transport-independent progress
information. A backend can then declare that its runtime-layout inputs require
no processing plan, while checkpoint-layout inputs continue through the
normal staging and PWAL path.

## MLA and Cross-Module Dependencies

Input coverage alone does not establish cross-module readiness. MLA
postprocessing must wait until the projection it reads, such as `kv_b_proj`,
has finished conversion and landing.

Our approach explicitly registers the relevant dependency states. A module
becomes ready only after its own required inputs and all declared dependencies
are complete; then its module-level reload/PWAL-derived processing runs.
These are actual data dependencies, not simply every child in the module tree.

See
[dependency registration](https://github.com/new-TonyWang/vllm/blob/e494daa3e497122416908c8ce636e97bb3fd463b/vllm/model_executor/model_loader/reload/model.py),
[readiness scheduling](https://github.com/new-TonyWang/vllm/blob/e494daa3e497122416908c8ce636e97bb3fd463b/vllm/model_executor/model_loader/reload/trace.py#L898),
and the
[MLA reload implementation](https://github.com/new-TonyWang/vllm/blob/e494daa3e497122416908c8ce636e97bb3fd463b/vllm/model_executor/model_loader/reload/mla.py).

Modulewise's deferred-attention phase addresses the MLA ordering case too.
Explicit dependencies allow that readiness contract to be represented in a
shared execution layer without making completion depend on transport order.

## RDT and a Shared Reload Execution Layer

Sharded RDT currently rejects EPLB because expert rearrangement invalidates
its baked replay plan. Our EPLB-aware loader path can complement RDT for
those configurations today. Making RDT itself EPLB-aware would still require
invalidating or updating its source-routing and scatter plan; our support
does not automatically make an existing RDT bake valid.

RDT bake and our cold-load arrival observation are closely related:
both execute model loading once to discover the loading contract.
RDT additionally records source slicing operations and destination geometry.
Could we factor the common discovery and execution concepts into a
transport-independent reload layer?

The division we propose is:

```text
RDT:
    producer selection, source slicing, packing, transport, receive buffers

shared reload execution:
    input targets and coverage
    dependency readiness and conversion
    runtime identity/layout checks and landing
    derived-state refresh and failure handling
```

RDT replay should be able to submit region writes and completion events
directly, without replaying or fabricating `weight_loader` calls. The
loader-driven front end could submit logical applications and observed
regions to the same executor.

This would let **RDT own the transfer plan while the reload layer owns
reliable reload execution**. We would be happy to design the input-coverage
contract and the shared discovery interface together.

## What's More

The reload lifecycle could expose a few small features that would make
user easier to operate and debug.

### Reload Progress and Partial Results

Each reload request could return a structured progress report instead of only
success or failure. For example:

```text
reload_id
status: in_progress | completed | failed
updated:
    parameters and regions completed by this request
pending:
    parameters or shards still waiting for input
failed:
    parameters, regions, or conversion steps that failed
progress:
    received regions / required regions
```

For a successful request, the response would identify which parameters or
parameter shards were updated and which remain incomplete. For a failed
request, it would include the failure reason and the affected regions. This
would make coverage state visible to the caller and provide a progress-bar-like
view for multi-step or asynchronous reloads.

The report should distinguish data arrival from runtime readiness:

```text
received
    -> coverage complete
    -> PWAL/processing complete
    -> landed and refreshed
```

Receiving a shard does not necessarily mean that the corresponding runtime
weight has already been updated, especially for quantized paths that wait for
all checkpoint-format inputs before running PWAL.

### Reload Memory Monitoring

Streaming reloads can accumulate staging buffers when a required weight or
shard never arrives. Repeated reload requests may therefore increase memory
usage even though no additional module can become ready.

The reload executor could track, per update and per module:

```text
staging bytes
temporary conversion bytes
receive-buffer bytes
pending parameter/shard regions
age of the oldest incomplete region
```

When the retained reload memory exceeds a configured threshold, the API
should warn the user with the affected modules and missing regions. The
warning should make clear whether memory is held by incomplete coverage,
pending PWAL conversion, or transport receive buffers.

An implementation could expose both a soft warning threshold and a hard
limit:

```text
soft limit:
    emit a warning and include the memory report in reload status

hard limit:
    reject or abort new reload work and release incomplete staging state
```

This monitoring is complementary to coverage tracking: coverage identifies
why a module is not ready, while memory accounting identifies the operational
cost of waiting for it. Both should be tied to the same reload transaction so
that completed staging is released promptly and an aborted transaction cannot
leave buffers retained indefinitely.
