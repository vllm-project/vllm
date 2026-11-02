# The Speculative Round Contract (Experimental)

This page is for authors of [custom speculators](README.md#custom-speculators-experimental).
It describes how a speculator controls a whole speculative round with the V2 model
runner. The round has five parts: the plan of rows the target verifies, which of
them it accepts, how many candidates the next round may take, how many requests the
scheduler admits, and the kernels that verify and commit trees.

Every hook has a default that keeps the standard behaviour. A speculator overrides
only the hooks it needs.

!!! warning
    This is an experimental interface. It may change between releases.

## Who owns what

| The speculator owns | vLLM owns |
| --- | --- |
| The rows the target verifies each step: chain, tree or mask (`BaseSpeculator.plan`) | The target forward, its CUDA graphs and its parallelism |
| Which rows are accepted (`BaseSpeculator.accept`) | KV cache and recurrent-state memory, sized at startup |
| The candidate budget for the next round (`SpeculatorConfig.candidate_budget`) | Scheduling, preemption and the order of the waiting queue |
| A cap on running requests (`SpeculatorConfig.admit_limit`) | Committing the accepted path to the KV cache and recurrent state |
| Its own drafting (`BaseSpeculator.propose`) | Deciding what each layer of the target can verify |
| Optional verify and commit kernels (`SpeculatorConfig.spec_kernel_providers`) | The in-tree kernels, used for every key a speculator does not provide |

## Startup

### Declare resources

`SpeculatorConfig.resources(max_num_seqs) -> SpecResources` declares what speculation
holds. vLLM sizes memory and CUDA graphs for it once, at startup.

| `SpecResources` field | Meaning |
| --- | --- |
| `max_candidates: int` | Rows per request, the root included. Must equal `num_speculative_tokens + 1`; the scheduler reserves this many KV slots per decoding request. |
| `capture_shapes: tuple[PlanShape, ...]` | The target-forward shapes to capture. `PlanShape(n)` is `n` rows for every request. `PlanShape(None)` is a variable-length decode graph that takes 1 to `max_candidates` rows per request. |
| `max_branching: int = 1` | 1 for chains. Above 1, plans may branch. |
| `custom_mask: bool = False` | Plans may carry a visibility mask. Needs `max_branching > 1`. |

The default is a chain of up to K drafts: `max_candidates` is K + 1, with one
recurrent-state row per draft. `SpecResources.plan_kind` follows from the fields:
`"mask"` with `custom_mask`, `"tree"` when `max_branching > 1`, else `"chain"`.

### Plan kinds

Plan kinds are ordered `"none" < "chain" < "tree" < "mask"` (`PlanKind` in
`capabilities.py`):

- `"none"`: the layer cannot verify drafts at all.
- `"chain"`: each verify row follows the one before it.
- `"tree"`: each row has a parent and sees its ancestors.
- `"mask"`: each row sees the rows its mask names.

### Per-layer capabilities

After the attention backends are built, each worker reports what its layers verify
as a `SpecCapabilities`: a `plan_kind` and one `SpecLimit(plan_kind, reason)` per
layer kind or setting that lowers it. The engine core reduces the reports of every rank: the model verifies the lowest plan
kind of any layer on any rank.

Each kind of layer answers in its own way:

- **Attention layers.** The metadata builder answers
  `spec_verify_limit(*, kv_cache_dtype, sliding_window, has_sinks,
  logits_soft_cap, has_alibi, chunk_size, rswa_window, kv_sharing_target)` for the
  layer's features (`vllm/v1/attention/backend.py`) with the plan kind and the
  reason the startup log shows. The default is `"chain"`. A builder that returns
  `"tree"` reads `CommonAttentionMetadata.tree_ancestors` and applies every feature
  of the layer by position (`CommonAttentionMetadata.positions`, the plan
  positions). A builder that returns `"mask"` accepts any mask over earlier rows
  in `tree_ancestors`.
- **Recurrent layers** (`MambaBase` subclasses). A layer class verifies more than
  chains only when the class itself sets `spec_layer_variant`, for example `"gdn"`
  or `"mamba2"`. Subclasses do not inherit it, so a subclass with its own forward
  falls back to chains. A class with `spec_chain_supported = False` (Mamba1) reports
  `"none"`. Above chains, a recurrent layer verifies the highest plan kind that its
  kernels in the registry support (see [Kernel registry](#kernel-registry)). A
  recurrent state folds in every ancestor, so it cannot hide one. A recurrent layer
  therefore caps masks at trees, unless its verify kernel itself supports `"mask"`.
- **Runner settings.** These lower trees and masks to chains: pipeline parallelism,
  prefill and decode context parallelism, batch-sharded sampling, micro-batching,
  and a model state without tree support.

### Downgrade to chain

The engine core calls `SpeculatorConfig.adapt_resources(declared, caps) ->
SpecResources` with the reduced capabilities. The result may only take less than
`declared`, and its plan kind must be one the model verifies
(`validate_effective_resources` in `capabilities.py`). By default, a speculator
that branches runs as a chain of the same length, and a mask speculator runs its
plans as trees on a model that verifies trees: its plans then carry no mask, and
rows see their ancestors.

A tree speculator should override `adapt_resources` to pick its own chain, because a
64-row tree is not a 64-token chain. A mask speculator whose masks only make sense as
masks should override it to pick its own tree.

The engine logs one line per limit that lowers the declared plan kind:

```text
spec decode: tree -> chain (<layer kind>, <n> layers: <reason>)
```

The scheduler, the runner, the CUDA graph manager and the speculator then run with
the adapted resources (see [Limits](#limits)). The speculator receives them in
`BaseSpeculator.bind_resources(resources)`, once per worker, before capture. With
`resources.max_branching == 1`, every batch is `PlanInputs.chain_only`.

A speculator that must not run with a lower plan kind raises in its
`adapt_resources`. Startup also fails when a layer verifies no plans at all (plan
kind `"none"`).

## Every step, in call order

Steps 1 and 2 run in the scheduler process. State a config keeps there stays
there. Steps 3 to 8 run on every worker.

### 1. Admission: `SpeculatorConfig.admit_limit`

```python
def admit_limit(self, view: BudgetView) -> int
```

This returns the most requests that may be running once this step admits waiting
ones. The value must be at least 1. The scheduler calls it once per `schedule()`
that admits requests. The call comes after the running requests are scheduled
(`view.num_scheduled_reqs` counts them) and before any waiting request is admitted.
The scheduler takes the smaller of the answer and `max_num_active_seqs`, so this
hook can lower admission but never raise it. Running requests are never preempted
for it. Waiting requests, preempted ones included, are still admitted in queue
order. Under async scheduling, `view.num_running` includes the requests of the step
still executing.

The default sets no limit, and the scheduler calls only an override.

### 2. Candidate budget: `SpeculatorConfig.candidate_budget`

```python
def candidate_budget(self, view: BudgetView) -> int
```

This returns the rows per request, root included, that the drafts proposed after
this step may take. The value must be in `[1, max_candidates]`. The scheduler calls
it once per `schedule()` that schedules requests, after allocation. The drafts
proposed now are verified next step, so the budget binds one step later. This
matches the Dynamic SD schedule.

A budget of 1 proposes no drafts. The scheduler sets `skip_drafts`, and the runner
does not run the drafter after this step. Under data parallelism the drafter is
skipped only when every rank skips it. The next step is a plain decode. It replays
the uniform-1 graph when `PlanShape(1)` is in `capture_shapes`. A drafter with its
own KV leaves the skipped tokens' draft KV unwritten. The target verifies every
draft, so outputs do not change.

The default is K + 1, or the Dynamic SD schedule's K + 1 at this batch size. The
scheduler caps the answer at the effective `max_candidates`. `BudgetView` holds
host integers read after this step's requests are scheduled.

On a model with recurrent (Mamba, GDN) layers, a request holds one recurrent-state
block per draft it verifies, so a budget that drops drafts under load frees blocks
for more running requests (`view.num_free_blocks` shows the pool).

### 3. Plan: `BaseSpeculator.plan`

```python
def plan(self, inputs: PlanInputs) -> tuple[VerifyPlan, PlanBound] | None
```

The runner calls this once per scheduled batch that carries drafts, before it builds
the batch's inputs (not under adaptive verification, see [Limits](#limits)). Return
`None`, the default, to verify every scheduled
draft as a chain.

`PlanInputs` holds `req_ids`, `idx_mapping` (a device tensor), `idx_mapping_np`,
`scheduled_rows` and `chain_only`. `scheduled_rows[r]` is 1 plus the drafts the
scheduler reserved for request `r`. `chain_only` is set when the runner can only
verify chains this step: when a prefill is in the batch, when a logits processor
reads each row's prefix (penalties, bad words, a thinking budget, a custom
processor), with structured output, and in the modes listed under [Limits](#limits).

A `VerifyPlan` holds device tensors per scheduled request, with `R` requests and
`C` rows:

| Field | Shape and meaning |
| --- | --- |
| `parent` | `[R, C]` int32. `parent[r, 0] == -1`. `parent[r, i]` is the row candidate `i` continues, in `[0, i)`. Rows are ordered parent first. A chain has `parent[r, i] == i - 1`. |
| `count` | `[R]` int32 in `[1, C]`. 1 is a plain decode. Entries past `count[r]` are ignored. |
| `tokens` | `[R, C]` int32 candidate tokens. `tokens[r, 0]` is ignored. |
| `draft_probs` | `[R, C, vocab]` float32 for probabilistic verification, else `None`. |
| `is_chain` | A host flag. Set it when every `parent[r, i] == i - 1` by construction. The runner then skips ancestor sets and KV compaction. It never derives this flag from the tensors. |
| `positions` | `[R, C]` int32 offsets from `num_computed_tokens`, replacing each row's depth. Positions do not enter a recurrence, so any model takes them. |
| `mask` | `[R, C, C]` bool. `mask[r, i, j]` is whether row `i` sees row `j`. The diagonal is taken as true, entries with `j > i` are ignored, and the committed prefix is always visible. Needs `custom_mask`. `parent` still gives the paths the commit accepts along. Recurrent layers read `parent`, not the mask. |

Row 0 is the root, the request's last committed token. Each row writes its KV at the
chain slot of its row index. After acceptance, the commit moves the accepted path
into place. Plan buffers belong to the speculator. They must stay valid until the
commit returns, so that CUDA graph replays read stable addresses.

`PlanBound` gives the host integers the runner needs before the device plan exists.
`rows[r]` is how many rows request `r` verifies, at most `scheduled_rows[r]`. The
drafts past it leave the batch and count as rejected. `rows=None` verifies every
scheduled row. The batch's graph shape follows from the rows. The device `count` may
stop short of the bound, and rows past it are never accepted.

A plan that is not a chain needs `max_branching > 1` in the bound resources, and a
batch that is not `chain_only`. The runner checks each plan against the effective
resources and rejects a plan with more than `max_candidates` rows, or with a mask
the resources do not allow.

### 4. Verify

The runner builds the inputs from the plan and runs the target forward. Each row's
position is `num_computed_tokens` plus its depth, or plus `positions`. With M-RoPE,
the shift is the same on every section. Attention sees each row's ancestors, or its
mask. The forward replays the captured graph for the batch's shape.

### 5. Accept: `BaseSpeculator.accept`

```python
def accept(
    self, plan: VerifyPlan, output: VerifyOutput, sampler: Sampler
) -> AcceptResult | None
```

The runner calls this on the sampler's stream for every verified batch. A large
batch is verified in request ranges, with one call per range, and `plan` and
`output` cover that range. `VerifyOutput` carries the logits of the verified rows in
plan order, with their positions and draft tokens. `sampler` applies the sampling
parameters (`apply_sampling_params`) and holds the sampling states. It is an
internal interface and may change.

Return `None`, the default, to let vLLM verify. vLLM checks a chain with its
rejection sampler. For a plan that is not a chain, it walks the tree on each row's
argmax, which only greedy requests accept.

`AcceptResult` has these fields:

- `path`: `path[r, :path_len[r]]` lists the accepted rows from the root down. Each
  row is a child of the previous one, so `path[r, j] >= j`.
- `path_len`: the number of accepted rows per request.
- `sampled`: `sampled[r, :path_len[r]]` holds the committed tokens. They are the
  accepted candidates' tokens followed by the bonus token, sampled from the logits
  of the last accepted row.
- `processed_logits`: optional.

The runner takes each committed token's logprobs from its row on the path.

### 6. Commit

The runner commits the KV and the recurrent state along `path`. Every other row is
rejected. Whether the commit moves anything depends on the plan's `is_chain`, not on
the result: a chain commits nothing. A plan that is not a chain runs the bound
`commit` kernels in the registry.

### 7. Observe: `BaseSpeculator.observe_step`

```python
def observe_step(self, step: StepObservation) -> None
```

The runner calls this once per step that ran the target forward and sampled. The
call comes after the commit and before `propose`. Steps without drafts (budget 1) and
prefill steps are observed too. Capture and profiling runs are not. With pipeline
parallelism, only the last rank calls it.

`StepObservation` (`step_observation.py`) holds the step's shape and counts. The
record holds no copies, and the runner does not sync for it. Reading
`num_sampled` on the host adds a sync. The default does nothing, and the runner
builds no record for a speculator that does not override the hook.

### 8. Propose: `BaseSpeculator.propose`

The runner calls `propose(..., num_speculative_tokens=)` with the number of drafts
this step's budget allows. It does not call it when the drafter is skipped (step 2).
`propose` returns the draft tokens. The scheduler schedules them, and step 3 of the
next step plans over them.

## Kernel registry

Every kernel that speculative decoding adds is in a `KernelRegistry`
(`vllm/v1/worker/gpu/spec_decode/kernels.py`). A key is a layer kind and an op,
written as `"<layer kind>/<op>"` in logs:

| Op | Keys | What it does |
| --- | --- | --- |
| `verify` | `mamba:gdn@slots`, `mamba:mamba2@slots` | Runs the convolution and recurrence of a recurrent layer's verify rows, inside the captured graphs. The reference entry is the layer's own kernels. A bound provider kernel replaces them. |
| `commit` | `layout:paged_token`, `layout:mamba_state@slots` | Moves what the accepted path wrote to where the next step reads it: attention KV, or the recurrent state and conv window. |

`@slots` is the recurrent-state mode.

`ReferenceKernels` holds the in-tree kernels. A speculator names provider factories
in `SpeculatorConfig.spec_kernel_providers`, a tuple of qualified names. Providers
overlay the reference kernels in order: a later provider wins a key, and each
override is logged once. A provider is constructed with `vllm_config`, has a `name`,
and returns `{key: SpecKernel}` from `kernels()`.

A provider may supply any subset of keys. A layer verifies the highest plan kind
that every kernel it needs supports for it. `KernelRegistry.required_keys(layer)`
lists those kernels. For a recurrent layer, they are its `verify` kernel and its
state's `commit` kernel. If a kernel is missing, or it declines the layer, the layer
verifies chains. A missing kernel is never called.

A `SpecKernel` implements these methods:

- `supports(layer: SpecLayerSpec) -> frozenset[PlanKind]` returns the plan kinds the
  kernel runs for `layer`. An empty set declines the layer, for example on its state
  dtype, head size or conv width. It is called before `bind`.
- `bind(ctx: KernelContext, layers: Sequence[SpecLayerSpec]) -> None` is called once
  per worker, before capture, with every layer the kernel runs on. Allocate memory
  here. Raise here on anything the kernel cannot run, not at the first step.
- `verify(rows) -> None`, for `mamba:*` keys, writes every row's output to
  `rows.out` and keeps the in-tree state layout. Row `i` of request `b` stores its
  state at `state_indices[b, i]` and continues from row `parents[i]`.
- `commit(step: CommitStep) -> None`, for `layout:*` keys, moves what the rows on
  `step.result.path` wrote to where the next step reads it.

**Capture rules.** `verify` runs inside every forward with verify rows, CUDA graph
captures included. It must not sync with the host, and it must not allocate memory
sized by data. When capturing, the runner runs each bound `commit` kernel once per
capture shape on identity paths, so compilation and launch failures surface at
startup.

```python
from vllm.v1.worker.gpu.spec_decode.kernels import SpecKernel


class MyGDNVerify(SpecKernel):
    def supports(self, layer):
        return frozenset({"chain", "tree"})

    def verify(self, rows) -> None: ...


class MyProvider:
    name = "my_provider"

    def __init__(self, vllm_config): ...

    def kernels(self):
        return {("mamba:gdn@slots", "verify"): MyGDNVerify()}


class MySpeculatorConfig(SpeculatorConfig):
    spec_kernel_providers = ("my_package.MyProvider",)
```

## Example: a chain speculator

A chain speculator needs only the config and `propose`. The optional hooks shown
here are a budget that stops drafting under load, and an admission cap.

```python
# my_package/config.py
from vllm.v1.worker.gpu.spec_decode.eagle.config import Eagle3SpeculatorConfig


class MyChainConfig(Eagle3SpeculatorConfig):
    speculator_class = "my_package.speculator.MyChainSpeculator"

    def candidate_budget(self, view):
        # No drafts when many requests are running: K + 1 rows, else 1.
        k_plus_1 = self.num_speculative_tokens + 1
        return 1 if view.num_running > 128 else k_plus_1

    def admit_limit(self, view):
        return max(1, view.max_num_seqs // 2)
```

```python
# my_package/speculator.py
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator


class MyChainSpeculator(EagleSpeculator):
    def observe_step(self, step):
        # Host-only bookkeeping. Do not read step.num_sampled on the host.
        self.last_num_spec_tokens = step.num_spec_tokens
```

Register the config with `register_speculative_method` as shown in
[Custom Speculators](README.md#custom-speculators-experimental).

## Example: a tree speculator

A tree speculator also declares branching resources, chooses its chain for models
that cannot verify trees, and returns a plan each step. `accept` is optional: the
default walks the tree on each row's argmax, which only greedy requests accept.

```python
from vllm.v1.worker.gpu.spec_decode.round import (
    PlanBound,
    PlanShape,
    SpecResources,
    VerifyPlan,
)


class MyTreeConfig(MyChainConfig):
    speculator_class = "my_package.speculator.MyTreeSpeculator"

    def resources(self, max_num_seqs):
        # The root and K candidates: run with num_speculative_tokens=15.
        rows = self.num_speculative_tokens + 1
        return SpecResources(
            max_candidates=rows,
            # Graphs are captured only at declared shapes: include the chain's.
            capture_shapes=(PlanShape(1), PlanShape(5), PlanShape(rows)),
            max_branching=4,
        )

    def adapt_resources(self, declared, caps):
        if caps.plan_kind == "chain":
            # Our best chain is 4 drafts, not 15.
            return SpecResources(
                max_candidates=5,
                capture_shapes=(PlanShape(1), PlanShape(5)),
            )
        return declared


class MyTreeSpeculator(MyChainSpeculator):
    def bind_resources(self, resources):
        super().bind_resources(resources)
        self.trees = resources.max_branching > 1
        # Allocate the plan buffers here: they must keep their addresses.

    def plan(self, inputs):
        if not self.trees or inputs.chain_only:
            return None  # vLLM verifies the scheduled drafts as a chain
        n = len(inputs.req_ids)
        # Fill self.parent[:n], self.count[:n] and self.tokens[:n] on device.
        plan = VerifyPlan(
            parent=self.parent[:n],
            count=self.count[:n],
            tokens=self.tokens[:n],
        )
        return plan, PlanBound(rows=None)
```

## Coverage by model family

A model verifies the lowest plan kind of any of its layers. The table shows what each
family verifies with the in-tree backends and kernels. A plugin attention backend
whose builder answers `"tree"` or `"mask"`, or a kernel provider, can raise a row.

| Family | Layers | Plan kind | What decides it |
| --- | --- | --- | --- |
| Dense attention (Llama, Qwen3, Mistral) | FlashAttention, FlashInfer, Triton | chain | The builders keep the default `spec_verify_limit`. |
| Any attention model on FlexAttention | full and sliding-window attention | mask | `FlexAttentionMetadataBuilder` verifies masks, and so trees. A sliding window measures plan rows by position. |
| Sinks, soft cap, ALiBi, chunked local attention, R-SWA, KV sharing, quantized KV (for example gpt-oss, Gemma 2, Llama 4) | any in-tree backend | chain | FlexAttention chains with these features and names the feature in the startup log. |
| MLA (DeepSeek V3, Kimi K2) | MLA backends | chain | MLA builders keep the default. |
| Sparse attention with an indexer (DeepSeek V3.2, GLM-5) | indexer | chain | The indexer builder keeps the default. |
| GDN hybrids (Qwen3-Next, Qwen3.5) | GDN | tree | `mamba:gdn@slots/verify` and `layout:mamba_state@slots/commit`. The attention layers must also verify trees. |
| Mamba2 hybrids (Nemotron-H, Zamba2, Falcon-H1, Granite hybrids) | Mamba2 | tree | `mamba:mamba2@slots/verify`. The attention layers must also verify trees. |
| KDA and other GDN subclasses (Kimi Linear, OLMo hybrid) | own layer classes | chain | No `spec_layer_variant` on the subclass. |
| Linear attention (MiniMax-Text-01, Bailing) | `LinearAttention` | chain | No `spec_layer_variant`. |
| Short conv (LFM2) | short conv | chain | No `verify` kernel for `mamba:short_conv@slots`. |
| Mamba1 (Jamba, FalconMamba) | Mamba1 | none | `spec_chain_supported = False`. Speculative decoding refuses to start. |

## Limits

- A plan has at most 64 rows per request (`MAX_PLAN_ROWS` in `plan_kernels.py`): the
  ancestor set of a row is one int64.
- With pipeline parallelism, trees and masks run as chains. The tree path is not
  broadcast between stages.
- Prefill and decode context parallelism, batch-sharded sampling and micro-batching
  also run as chains.
- `plan` is not called under adaptive verification, which sets its own budget.
- Overlapping the drafter with the target forward is not part of the contract.
- The KV and recurrent-state pools are sized for the declared resources, even when
  the model runs with less.
- Replay and recovery recurrent-state modes keep no state slot per draft row, so
  their layers verify chains.
