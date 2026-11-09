# The Speculative Round Contract (Experimental)

This page is for authors of [custom speculators](README.md#custom-speculators-experimental)
on the V2 model runner. A speculator may choose the rows the target verifies, how
many candidates the next round takes, the accepted path through a tree, and the
layers and kernels that verify and commit trees. Every part has a default, and a speculator
overrides only the parts it needs. vLLM describes and checks what each part is
given and returns; the tree kernels themselves come from the speculator.

!!! warning
    This is an experimental interface. It may change between releases.

## Who owns what

| The speculator owns | vLLM owns |
| --- | --- |
| The rows the target verifies each step: chain or tree (`BaseSpeculator.plan`) | The target forward, its CUDA graphs and its parallelism |
| The candidate budget for the next round (`SpeculatorConfig.candidate_budget`) | Sampling the target's token at every verify row, and the rejection sampler for chains |
| The accepted path through a tree (`BaseSpeculator.tree_accept`) | Checking that path against the plan |
| Its own drafting (`BaseSpeculator.propose`) | KV cache and recurrent-state memory, sized at startup |
| The tree commits: KV compaction and recurrent state (`SpeculatorConfig.spec_kernels`) | A description of each KV cache group's layout for the KV commit |
| Tree-verifying layers: an attention backend or a recurrent layer swapped in with `PluggableLayer.register_oot` | Each plan row's parent and position in the attention metadata |
| | Scheduling, preemption and the order of the waiting queue |
| | Deciding what each layer of the target can verify |

## Startup

### Declare resources

`SpeculatorConfig.resources(max_num_seqs) -> SpecResources` declares what speculation
holds. vLLM sizes memory and CUDA graphs for it once, at startup.

| `SpecResources` field | Meaning |
| --- | --- |
| `capture_rows: tuple[int, ...]` | Rows per request, the root included, of each decode graph to capture; at most `num_speculative_tokens + 1`. The largest, `max_rows`, caps the rows any request verifies. |
| `max_branching: int = 1` | 1 for chains. Above 1, plans may branch. |

The scheduler reserves `num_speculative_tokens + 1` KV slots per decoding request.
The default is a chain of up to K drafts: `capture_rows` is `(K + 1,)`.
`SpecResources.plan_kind` is `"tree"` (each row has a parent and sees its
ancestors) when `max_branching > 1`, else `"chain"` (each row follows the one
before it).

### Startup rule

After the attention backends are built, each worker checks its target
layers and reports a `SpecCapabilities`: a `plan_kind` and one
`SpecLimit(plan_kind, reason)` per layer kind or setting that lowers it. The model
verifies the lowest plan kind of any layer on any rank.

Each layer, its backend and the model state answer for themselves with a
decline reason (None: trees); vLLM knows no model family.

- **Attention layers** verify trees when
  `AttentionBackend.tree_decline_reason(layer, kv_cache_spec)` returns None
  (default: chains; `causal_full_attention` helps a backend decline windows)
  and the registry holds the commit of the layer's layout, e.g.
  `layout:paged_token/commit`. Such a backend reads
  `CommonAttentionMetadata.tree_parents` (each token's parent row within its
  request, -1 for its first) and the plan `positions`, and builds the mask it
  needs (a row sees its context, itself and its ancestors) in its own metadata
  builder. No in-tree backend verifies trees and vLLM has no KV commit
  kernel, so in tree every model verifies chains.
- **The speculator** verifies chains unless it overrides `BaseSpeculator.tree_accept`.
- **Recurrent layers** (`MambaBase` subclasses) verify trees when
  `MambaBase.tree_decline_reason()` returns None (no in-tree layer's does: a
  plugin swaps in its own layer with `PluggableLayer.register_oot`) and the
  registry holds a `layout:recurrent_state/commit` kernel. Row `i` of a request
  stores its state at `spec_state_indices_tensor[b, i]` and continues from the
  state of its parent row (`KernelContext.tree_parents`).
- **The model state** verifies chains when `ModelState.tree_decline_reason()`
  says it cannot resume from a tree path (recurrent align prefix caching,
  recovery or replay). On a tree step the runner passes each request's last
  accepted row as `postprocess_state(..., accepted_row=)`; it omits the keyword
  on chain steps, so an override without it keeps working.
- **Runner settings** listed under [Limits](#limits) lower trees to chains.

### Downgrade to chain

The engine core calls `SpeculatorConfig.adapt_resources(declared, caps) ->
SpecResources`. The result may only shrink `declared` (no larger `max_branching`,
no undeclared `capture_rows`) and its plan kind must be one the model verifies. By
default a branching speculator runs as a chain of the same length; a tree
speculator should pick its own chain, since a 64-row tree is not a 64-token chain.
Dropping the largest `capture_rows` entry shortens the chain. To refuse a lower
plan kind, raise in `adapt_resources`. Each lowering is logged, for example
`spec decode: tree -> chain (<layer kind>, <n> layers: <reason>)`.

Everything then runs with the adapted resources; the speculator receives them in
`BaseSpeculator.bind_resources(resources)`, once per worker, before capture. With
`max_branching == 1`, every batch is `PlanInputs.chain_only`.

## Every step, in call order

Step 1 runs in the scheduler process. State a config keeps there stays there.
Steps 2 to 6 run on every worker.

### 1. Candidate budget: `SpeculatorConfig.candidate_budget`

```python
def candidate_budget(self, view: BudgetView) -> int
```

This returns the rows per request, root included, in `[1, num_speculative_tokens +
1]`, that the drafts proposed after this step may take. The scheduler calls it once
per `schedule()` that schedules requests; the budget binds one step later, as the
Dynamic SD schedule does, and is capped at the effective `max_rows`. The default is
K + 1, or the Dynamic SD schedule's.

A budget of 1 proposes no drafts (`skip_drafts`): the runner skips the drafter
(under data parallelism, only when every rank skips it) and the next step is a
plain decode. A budget that stops drafting under load:

```python
from vllm.v1.worker.gpu.spec_decode.eagle.config import Eagle3SpeculatorConfig


class MyChainConfig(Eagle3SpeculatorConfig):
    def candidate_budget(self, view):
        return 1 if view.num_running > 128 else self.num_speculative_tokens + 1
```

### 2. Plan: `BaseSpeculator.plan`

```python
def plan(self, inputs: PlanInputs) -> VerifyPlan | None
```

The runner calls this once per batch with drafts, before it builds the batch's
inputs (not under adaptive verification). Return `None`, the default, to verify
every scheduled draft as a chain.

`PlanInputs` holds `req_ids`, `idx_mapping` (a device tensor), `idx_mapping_np`,
`scheduled_rows` and `chain_only`. `scheduled_rows[r]` is 1 plus the drafts the
scheduler reserved for request `r`. `chain_only` is set when the runner can only
verify chains this step: when a prefill is in the batch, when a logits processor
reads each row's prefix (penalties, bad words, a thinking budget, a custom
processor), with structured output, when a request replays a trace, and in the
modes listed under [Limits](#limits).

A `VerifyPlan` holds device tensors per scheduled request, with `R` requests and
`C` rows:

| Field | Shape and meaning |
| --- | --- |
| `parent` | `[R, C]` int32. `parent[r, 0] == -1`. `parent[r, i]` is the row candidate `i` continues, in `[0, i)`. Rows are ordered parent first. A chain has `parent[r, i] == i - 1`. |
| `tokens` | `[R, C]` int32 candidate tokens. `tokens[r, 0]` is ignored. None exactly for a chain, which verifies the drafted tokens. |
| `rows` | Host `[R]` int32, or `None`. `rows[r]` is how many rows request `r` verifies, at most `scheduled_rows[r]`; the drafts past it leave the batch and count as rejected. `None` verifies every scheduled row. Entries past a request's rows are ignored. |
| `single_branch` | A host flag. Set it when every `parent[r, i] == i - 1` by construction. The runner then uses the rejection sampler and skips the commit kernels. It never derives this flag from the tensors. |

Row 0 is the root, the request's last committed token. Plan buffers belong to the
speculator and must stay valid until the commit returns. A plan that is not a chain
needs `max_branching > 1` and a batch that is not `chain_only`; a plan has at most
`num_speculative_tokens + 1` rows. The kernels index each tensor of a plan or an
`AcceptResult` row by row, so its last dimension must have unit stride (a slice of
whole rows, not of columns); vLLM raises a `ValueError` otherwise.

### 3. Verify

The runner builds the inputs from the plan and runs the target forward. Each row's
position is `num_computed_tokens` plus its depth. With M-RoPE, the shift is the same
on every section. Attention gets each row's parent (`tree_parents`) and the tree
backend masks each row to its ancestors. The forward replays the captured graph
for the batch's shape.

### 4. Accept: `BaseSpeculator.tree_accept`

```python
def tree_accept(self, plan: VerifyPlan, target: TargetTokens, width: int) -> AcceptResult
```

Chains go through the rejection sampler. For a tree, vLLM gives every row the
target's token (argmax, or a draw from the row's processed logits) and calls
`tree_accept` once per sampling chunk. `TargetTokens` is flat over the chunk's
rows in plan order: `tokens` (int64, the target's token per row), `drafted`
(int64, each row's token; -1 for a row that must not be accepted) and
`row_start` (`[R + 1]` int32). A typical walk goes from the root to the child
whose drafted token is the target's token at the current row, until none
matches. The result, `AcceptResult`:

- `path`: `path[r, :path_len[r]]` lists the accepted rows from the root down, so
  `path[r, j] >= j`.
- `path_len`: the number of accepted rows per request.
- `sampled`: `[R, width]`; `sampled[r, :path_len[r]]` holds the committed tokens;
  the last is the bonus token.

vLLM checks the result on device, without a host sync: `path[r, 0] == 0`, each
accepted row's parent is the row before it on the path, and `1 <= path_len[r] <=
rows[r]`. A failed check is a device assert.

### 5. Commit

The runner commits the KV and the recurrent state along `path`. Every other row is
rejected. Whether the commit moves anything depends on the plan's `single_branch`, not on
the result: a chain commits nothing. A plan that is not a chain runs the bound
`commit` kernels in the registry.

### 6. Propose: `BaseSpeculator.propose`

The runner calls `propose(..., num_speculative_tokens=)` with the number of drafts
this step's budget allows. It does not call it when the drafter is skipped (step 1).
`propose` returns the draft tokens. The scheduler schedules them, and step 2 of the
next step plans over them.

## Kernel registry

Every kernel that speculative decoding adds is in a `KernelRegistry`
(`vllm/v1/worker/gpu/spec_decode/kernels.py`). A key is a layer kind and an op,
written as `"<layer kind>/<op>"` in logs. The one op is `commit`, keyed
`layout:paged_token` (the attention KV) or `layout:recurrent_state` (a recurrent
layer's state): it moves what the accepted path wrote to where the next step reads it.

The registry holds no kernel in tree. A speculator names, in
`SpeculatorConfig.spec_kernels`, the qualified name of a factory that takes
`vllm_config` and returns `{key: SpecKernel}`. A layer verifies trees when every kernel in `KernelRegistry.required_keys(layer)` supports it; a
missing or declining kernel means chains.

A `SpecKernel` implements these methods:

- `supports(layer: SpecLayerSpec) -> bool` says whether the kernel runs `layer`, on
  chains and trees alike. `False` declines the layer, for example on its state
  dtype, head size or conv width. It is called before `bind`.
- `bind(ctx: KernelContext, layers: Sequence[SpecLayerSpec]) -> None` is called once
  per worker, before capture, with every layer the kernel runs on. Allocate memory
  here. Raise here on anything the kernel cannot run, not at the first step.
- `commit(step: CommitStep) -> None` moves what the rows on
  `step.result.path` wrote to where the next step reads it. For request `b`
  with context `seq_lens[b] - rows[b]`, the KV of row `path[b, j]` moves to
  position `context + j`, for `1 <= j < path_len[b]`.

**KV layouts.** The `layout:paged_token` commit kernel is bound with
`ctx.kv_layouts`, one `PagedKVLayout` per KV cache group of its layers; vLLM
raises at bind for a cache it cannot describe.

| `PagedKVLayout` field | Meaning |
| --- | --- |
| `caches` | The group's cache tensors, one per distinct storage: KV-sharing layers appear once. |
| `shape` | `(num_blocks, block_size, num_planes, num_heads, head_dim)`. K and V are packed in one row (`num_planes == 1`); MLA has `num_heads == 1`. |
| `strides` | Element strides in the order of `shape`; `strides[4] == 1`. |
| `dtype` | The cache dtype. |
| `block_table` | `[max_num_reqs, max_blocks]` int32, the group's block table. |
| `table_block_size` | Tokens per `block_table` block; `block_size` (the backend's kernel block size) may divide it. |

Token `t` of request `b` is kernel token
`block_table[b, t // table_block_size] * table_block_size + t % table_block_size`,
split by `block_size` into block and slot.

**Capture rules.** `commit` and the tree attention backend run inside captured
graphs: no host syncs and no data-sized allocations. A `commit` kernel that needs a warm-up runs it in `bind`,
so failures surface at startup.

## Example: a tree speculator

A tree speculator declares branching resources, chooses its chain for models that
cannot verify trees, and returns a plan each step. Register the config with
`register_speculative_method` as shown in
[Custom Speculators](README.md#custom-speculators-experimental).

```python
from vllm.v1.spec_decode.resources import SpecResources
from vllm.v1.worker.gpu.spec_decode.eagle.config import Eagle3SpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.round import VerifyPlan


class MyTreeConfig(Eagle3SpeculatorConfig):
    speculator_class = "my_package.speculator.MyTreeSpeculator"

    def resources(self, max_num_seqs):
        # The root and K candidates: run with num_speculative_tokens=15.
        rows = self.num_speculative_tokens + 1
        return SpecResources(
            # Graphs are captured only at declared rows: include the chain's.
            capture_rows=(1, 5, rows),
            max_branching=4,
        )

    def adapt_resources(self, declared, caps):
        if caps.plan_kind == "chain":
            # Our best chain is 4 drafts, not 15.
            return SpecResources(capture_rows=(1, 5))
        return declared


class MyTreeSpeculator(EagleSpeculator):
    def bind_resources(self, resources):
        super().bind_resources(resources)
        self.trees = resources.max_branching > 1
        # Allocate the plan buffers here: they must keep their addresses.

    def plan(self, inputs):
        if not self.trees or inputs.chain_only:
            return None  # vLLM verifies the scheduled drafts as a chain
        n = len(inputs.req_ids)
        # Fill self.parent[:n] and self.tokens[:n] on device.
        return VerifyPlan(parent=self.parent[:n], tokens=self.tokens[:n])

    def tree_accept(self, plan, target, width):
        # Walk each request's tree with the target's tokens; return AcceptResult.
        ...
```

The config also names, in `spec_kernels`, a factory of a
`layout:paged_token/commit` kernel, and the model runs a tree attention backend. The
test plugin `tests/plugins/vllm_add_test_speculators` has PyTorch references of
the tree walk and the KV commit (`reference.py`) and of a tree attention backend
(`tree_attn.py`).

## Limits

- Pipeline parallelism (the tree path is not broadcast between stages), prefill and
  decode context parallelism, batch-sharded sampling and micro-batching run trees
  as chains.
- A speculator that overrides `plan` fails at startup under adaptive verification,
  which sets each request's verified drafts itself.
- Overlapping the drafter with the target forward is not part of the contract.
- The KV and recurrent-state pools are sized for the declared resources, even when
  the model runs with less.
- Align prefix caching, replay and recovery keep no recurrent state per draft
  row, so recurrent layers verify chains.
