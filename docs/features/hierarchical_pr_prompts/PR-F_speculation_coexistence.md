# PR-F — Speculative decoding coexistence

You are implementing **PR-F** of the Colibri-parity hierarchical plan.
**Depends on PR-B** (stable residency). PR-C recommended.

## Goal

Make vLLM **speculative decoding** (MTP / EAGLE / draft-verify) work correctly
and not regress when hierarchical expert staging is on — Colibri’s
`SPEC_PIN` / int8-MTP lessons applied as **policy**, not a C rewrite.

## Context

- Hierarchical remaps `topk_ids` → slot ids in `ensure_and_remap`.
- Spec paths may run draft and verify forwards with different shapes/batches.
- Colibri lesson: draft and verify must compute the **same function**; pinning
  policies must not diverge mid-step.

## Requirements

### 1. Shared staging policy for draft + verify

- Ensure draft and target models/forwards share one `ExpertTierManager`
  (or explicitly documented separate managers with identical pin sets).
- During a speculative step, do **not** allow draft-only eviction of experts
  still needed by verify (protect union of draft+verify expert sets for the
  step).

### 2. Config / docs

- Document which speculative methods are supported with hierarchical.
- Add a flag or automatic behavior: “spec pin” — freeze pins / disable live
  LFRU repin for the duration of the speculative step when
  `tier_policy=balanced`.
- Warn if MTP/draft quant is known-bad combinations (document only unless an
  existing vLLM check exists).

### 3. Correctness tests

- Prefer extending existing spec + MoE tests with hierarchical enabled on a
  tiny fake MoE if full GPU e2e is heavy.
- Hardware: one Ornith or Mixtral smoke with speculation on/off comparing
  acceptance rate and tok/s; disable speculation when cold-cache makes it a
  net loss (document how to measure).

## Non-goals

- Implementing a new MTP head or grammar engine.
- Dual-SSD or atlas.

## Test plan

```bash
.venv/bin/python -m pytest tests/model_executor/offloader/test_hierarchical_offload.py -v
# Plus any existing speculative decoding tests that can toggle hierarchical
```

Acceptance: speculative decode does not desync experts between draft/verify;
PR reports acceptance rate and tok/s with hierarchical on.

## PR description must include

- Duplicate check, tests/results, AI-assist statement, support matrix.

## Done when

- Spec pin / expert-union protect documented and implemented; smoke evidence
  attached.
