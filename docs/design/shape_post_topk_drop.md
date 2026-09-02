# Design handoff: SHAPE post-top-k expert dropping in RIY

## Status

- **Purpose:** implementation handoff for another coding session
- **Target branch:** `riy-rebase-preserve-history`
- **Current branch head:** `ca2051351` (`test: isolate RIY router mask coverage`)
- **Public base commit:** `761e0aa7a01ca764fdbe0eef563f0e8855630fe4`
- **Primary model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`
- **Model revision:** `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`

This document specifies a new RIY routing mode required by SHAPE. Do not infer
its semantics from the current `--riy-expert-profile` behavior: the current
behavior is intentionally different and must remain available for backward
compatibility.

## Problem statement

SHAPE's existing export path defaults to `zero_weights`:

```python
# Shapley-Moe/pruning/save_model.py
METHOD_DEFAULT_STRATEGIES = {
    "shapley": "zero_weights",
    # ...
}
```

The original checkpoint keeps the full router. For every token and layer, the
router still selects its original top-k logical experts from all experts. If a
selected expert was pruned, that expert contributes zero; no lower-ranked expert
is selected as a replacement. The number of effective expert FFN evaluations is
therefore allowed to be less than `top_k`.

Example for `top_k = 8`:

```text
router top-k logical IDs: [4, 17, 23, 41, 58, 72, 91, 108]
pruned logical IDs:       [17, 58]
executed experts:         [4, 23, 41, 72, 91, 108]
effective experts/token:  6
```

The desired physical-pruning implementation must be numerically equivalent to
keeping all expert tensors allocated and zeroing the pruned expert tensors.

## Current RIY behavior

Load-time profile pruning currently does the following:

1. Builds a compact expert map and allocates weights only for retained experts.
2. Builds a router-logit mask with `-inf` for pruned experts.
3. Adds the mask before top-k selection.
4. Selects a full `top_k` from retained experts.

Relevant code:

```python
# vllm/model_executor/layers/fused_moe/router/base_router.py
if self.prune_logit_mask is not None:
    router_logits = router_logits + self.prune_logit_mask

topk_weights, topk_ids = self._compute_routing(...)
```

This is a **pre-top-k reroute** policy. It saves expert-weight memory but still
executes `top_k` experts per token. It is not the SHAPE policy.

An older RIY runtime-mask path applied a mask after top-k, but it retained all
expert weights and renormalized the remaining routing weights. That is also not
numerically equivalent to SHAPE's `zero_weights` policy.

## Required public contract

Add an explicit routing mode to the profile schema. Do not silently change the
meaning of existing profiles.

### Version 1 compatibility

Existing version-1 profiles have no `routing_mode` field and must retain current
behavior:

```json
{
  "version": 1,
  "pruned_experts": [[0, 3], [4, 7]]
}
```

Interpret as:

```text
routing_mode = pre_topk_mask
```

### Version 2 profiles

Version-2 profiles must require a `routing_mode` field:

```json
{
  "version": 2,
  "routing_mode": "post_topk_drop",
  "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
  "model_revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
  "pruned_experts": [[0, 3], [4, 7]]
}
```

Supported values:

| Mode | Selection semantics | Effective experts/token | Renormalize |
| --- | --- | --- | --- |
| `pre_topk_mask` | Mask pruned logits, then select top-k from retained experts | Always `top_k` | Existing router behavior |
| `post_topk_drop` | Select original top-k, then drop pruned slots without replacement | `0..top_k` | **No** |

Unknown profile versions and unknown routing modes must fail closed with a clear
error before model allocation.

Do not add a second CLI override in the first implementation. Keeping the mode
inside the profile makes experiment artifacts self-describing and avoids
profile/CLI precedence ambiguity.

## Required `post_topk_drop` semantics

For each MoE layer:

```text
full hidden state
  -> full router projection (all original logical experts)
  -> original top-k logical expert IDs and weights
  -> identify selected IDs absent from the compact expert map
  -> mark those slots invalid; do not choose replacements
  -> execute only valid physical experts
  -> invalid slots contribute exactly zero
```

### Router logits

Do not modify router logits in `post_topk_drop` mode. In particular, do not add
`-inf` before `_compute_routing()`.

### Expert IDs

The router continues to produce global logical IDs. The compact expert map maps:

```text
retained logical ID -> contiguous physical ID
pruned logical ID   -> -1
```

The MoE backend must skip `-1` expert-token pairs rather than indexing an expert
weight tensor.

### Routing weights

For a dropped slot, its contribution must be zero. Remaining routing weights
must remain unchanged.

Do **not** renormalize surviving routing weights. Renormalization would define a
new model and would not match the existing `zero_weights` results.

For numerical safety, represent a dropped slot as an invalid expert ID and zero
contribution. Do not propagate `-inf` into arithmetic that can produce NaNs.

### No replacement

If two of the original top-8 experts are pruned, execute six experts. Do not
select the ninth- and tenth-ranked experts.

If all top-k experts are pruned, the routed-expert contribution is zero. Shared
experts, if present, remain untouched.

### Allocation

Both routing modes continue to use the existing compact allocation path:

- `expert_filter`
- compact `expert_map`
- reduced `local_num_experts`
- skipped checkpoint loading for pruned experts

The new mode is not a return to runtime-only masking.

## Recommended internal design

Avoid extending the current anonymous tuple from `build_riy_prune_map()` with
more positional fields. Prefer a small typed plan object, for example:

```python
@dataclass(frozen=True)
class RiyLayerPrunePlan:
    routing_mode: Literal["pre_topk_mask", "post_topk_drop"]
    num_kept: int
    expert_filter: torch.Tensor
    expert_map: torch.Tensor
    pre_topk_logit_mask: torch.Tensor | None
    post_topk_drop_mask: torch.Tensor | None
```

Exact naming is flexible, but the mode and the two mask phases must not be
confusable.

Suggested flow:

1. Parse and validate the profile once.
2. Build a per-layer prune plan.
3. Pass `expert_filter` to `ExpertMapManager` in both modes.
4. Attach `pre_topk_logit_mask` to the router only for `pre_topk_mask`.
5. Attach a logical pruned-expert lookup for `post_topk_drop`.
6. After `_compute_routing()`, identify dropped top-k slots in
   `post_topk_drop` mode.
7. Preserve logical IDs until the existing expert-map-aware MoE path maps them.
8. Ensure dropped slots produce zero and no expert kernel work.

If the selected MoE backend cannot safely skip expert-map entries equal to
`-1`, fail closed rather than falling back to allocating or executing the
pruned expert.

## Validation differences by mode

The current check rejects profiles where `num_kept < top_k`. Keep this check for
`pre_topk_mask`, because that mode must refill a complete top-k.

Do not apply that check to `post_topk_drop`; selecting fewer effective experts
is the point of the mode. A layer must still retain at least one physical expert
unless the backend is explicitly extended to support zero-expert layers.

Existing restrictions remain unless separately implemented and tested:

- EPLB unsupported with profile pruning
- round-robin expert placement unsupported with `expert_filter`
- begin with single-GPU / TP=1 and the Triton MoE backend

## Backend scope

Implement and validate `post_topk_drop` for:

```text
--moe-backend triton
```

first. This backend successfully loaded the compact Qwen3 profile on B300 and
uses the modular expert-map-aware path.

Do not claim support for all MoE backends merely because model loading succeeds.
Backends that ignore `expert_map`, remap `-1` incorrectly, or still launch work
for invalid slots must be rejected or tested before being listed as supported.

## Metrics required for the thesis experiment

Add optional counters for the post-top-k mode:

```text
original_topk_slots
surviving_topk_slots
dropped_topk_slots
```

Derive and expose:

```text
average surviving experts per token
average dropped experts per token
expert compute reduction ratio
per-layer surviving experts per token
distribution of effective expert count from 0 through top_k
```

Definition:

```text
expert compute reduction ratio
  = 1 - surviving_topk_slots / original_topk_slots
```

Record original logical top-k IDs before physical mapping. Statistics must not
change model outputs and should be disabled for formal performance runs unless
their overhead is separately measured.

## Tests to add before implementation is accepted

### 1. Profile schema tests

- Version 1 defaults to `pre_topk_mask`.
- Version 2 requires `routing_mode`.
- Unknown versions fail.
- Unknown modes fail.
- Invalid expert IDs fail.
- Duplicate profile entries have defined handling (prefer reject or deduplicate
  explicitly and test it).

### 2. Router unit tests

Use deterministic logits where a pruned expert is inside the original top-k.

For `pre_topk_mask`:

- pruned expert must not appear in top-k;
- another retained expert fills the slot;
- output still contains `top_k` valid experts.

For `post_topk_drop`:

- original logical top-k selection must be unchanged;
- the pruned logical expert must still be observable in original top-k;
- it must be marked invalid for physical execution;
- no replacement expert appears;
- surviving weights are unchanged;
- no renormalization occurs.

### 3. Expert-map tests

Extend `tests/distributed/test_expert_placement.py`:

- retained logical IDs map to contiguous physical IDs;
- pruned logical IDs map to `-1`;
- different layers can retain different expert counts;
- `post_topk_drop` permits `num_kept < top_k`;
- unsupported EP placements fail closed.

### 4. Kernel/output equivalence test

Construct a small deterministic MoE layer and compare:

```text
Reference:
  all expert tensors allocated
  pruned expert tensors filled with zeros
  unmodified original top-k routing

Candidate:
  pruned expert tensors not allocated
  compact expert map
  post_topk_drop routing
```

Assert outputs are equal within the dtype-appropriate tolerance. Include cases
with zero, one, multiple, and all top-k slots dropped.

This is the load-bearing correctness test.

### 5. End-to-end Qwen3 smoke test

With greedy decoding and fixed prompts, compare the compact
`post_topk_drop` service against a full-allocation zero-weight reference using
the same profile.

Capture:

- generated token IDs
- selected logical expert IDs
- surviving expert counts
- model memory reported during load

Greedy token outputs should match. If logits are compared, document BF16
absolute/relative tolerance.

### 6. Physical-savings test

Confirm:

```text
Dense model weight memory > keep-0.8 > keep-0.6
```

Also verify that the number of allocated physical expert tensors matches the
per-layer profile.

### 7. Performance test

After correctness passes, compare Dense / keep-0.8 / keep-0.6 with the same:

- vLLM source commit
- compiled extension wheel
- MoE backend
- dtype
- GPU
- model revision
- max model length
- request dataset and seed

Report TTFT, TPOT, end-to-end latency, request throughput, output-token
throughput, and surviving experts/token.

## Acceptance criteria

The implementation is complete only when all are true:

- [ ] Existing version-1 profiles preserve current pre-top-k behavior.
- [ ] Version-2 `post_topk_drop` profiles leave router logits unchanged.
- [ ] Original top-k is computed before any prune decision.
- [ ] Pruned top-k slots are not replaced.
- [ ] Surviving routing weights are not renormalized.
- [ ] Pruned expert tensors are not allocated or loaded.
- [ ] Kernel execution safely skips invalid expert slots.
- [ ] Compact output matches a zero-weight full-allocation reference.
- [ ] Surviving-experts/token metrics are available.
- [ ] Dense, pre-top-k, and post-top-k modes are distinguishable in logs.
- [ ] Unsupported backends/configurations fail closed.
- [ ] Unit tests and the Qwen3 B300 smoke test pass.

## Files likely to change

Primary:

```text
vllm/model_executor/layers/fused_moe/riy.py
vllm/model_executor/layers/fused_moe/layer.py
vllm/model_executor/layers/fused_moe/router/base_router.py
vllm/model_executor/layers/fused_moe/expert_map_manager.py  # only if needed
```

Tests:

```text
tests/kernels/moe/test_routing.py
tests/distributed/test_expert_placement.py
# add a focused output-equivalence test near existing FusedMoE tests
```

Docs:

```text
docs/design/riy.md
README.riy.md
```

Related SHAPE repository changes after RIY support lands:

```text
tools/build_faithfulness_profiles.py
results/qwen3-30b-a3b/faithfulness_profiles/**/*.json
```

Update generated SHAPE profiles to version 2 with:

```json
"routing_mode": "post_topk_drop"
```

Do not change those profile artifacts until the RIY implementation and schema
are finalized.

## B300 handoff environment

Pod/container:

```text
Pod:       ds-2e02f903-1.ds-2e02f903-1-b5916d65-a-f2a6
Container: worker0
```

Paths:

```text
RIY source: /root/workspace/chuanwu/vllm-moe_pruning
Python venv: /root/workspace/chuanwu/venvs/shape-vllm
SHAPE repo:  /root/workspace/chuanwu/Shapley-Moe
Model:       /root/workspace/chuanwu/models/Qwen3-30B-A3B-Instruct-2507
Logs:        /root/workspace/chuanwu/logs
```

The environment is an isolated `uv venv`, not Conda. It does not modify the
company environment under `/root/workspace/mudi/env` or the company vLLM
processes. The last test service was stopped and the test GPU was released.

Installed runtime:

```text
RIY Python source: branch riy-rebase-preserve-history
Torch:             2.13.0+cu130
CUDA runtime:      13.0 (host toolkit 13.2)
Compiled wheel:    nightly e222c33f2, cu130
MoE backend used:  triton
```

Before implementation/testing on B300:

1. Pull the latest branch head (`ca2051351` or newer).
2. Confirm `git status` is clean.
3. Keep using the isolated venv.
4. Use a free GPU; do not touch other users' processes.
5. Start with `--tensor-parallel-size 1 --moe-backend triton`.

Useful commands:

```bash
# Unit tests already exercised on B300
/root/workspace/chuanwu/venvs/shape-vllm/bin/python -m pytest \
  tests/distributed/test_expert_placement.py -q

# Confirm the RIY CLI
/root/workspace/chuanwu/venvs/shape-vllm/bin/vllm serve --help=all \
  | grep -A2 riy-expert-profile
```

## Existing B300 observations (not final SHAPE results)

Using the current, semantically incorrect pre-top-k profile mode:

```text
Dense model weight memory:       56.93 GiB
keep≈0.8 model weight memory:    46.13 GiB
weight-memory reduction:         10.80 GiB (~18.97%)
```

This proves compact allocation and skipped weight loading work. It does **not**
validate SHAPE quality or compute reduction because the current mode refills a
full top-k. Do not publish these as `post_topk_drop` results.

## Isolation from company code

The local worktree is:

```text
/Users/alizen/Dev/vllm-moe_pruning-riy-rebase
```

It is an independent Git worktree on `riy-rebase-preserve-history`. Its base
commit is part of public vLLM history, and RIY changes are separate commits. On
B300, source and Python environment live under `/root/workspace/chuanwu/`.
Nothing in `/root/workspace/mudi/env` or other users' running services should be
modified.

## Non-goals for this task

- Do not change SHAPE scoring or expert selection.
- Do not reduce the router output dimension.
- Do not choose replacement experts in `post_topk_drop` mode.
- Do not renormalize surviving top-k weights.
- Do not claim support for every MoE backend.
- Do not add EPLB or round-robin placement support as part of this change.
- Do not run final throughput experiments before zero-weight equivalence passes.
