---
name: review-models-inference-upstream-diff
description: Reviews the current branch against docs/cohere/code_notes/upstream-diff-models-and-inference.md for compliance, detects file restructuring, verifies custom models/inference code, and optionally updates the doc. Use when reviewing upstream diff compliance, after rebases or cherry-picks, or when the user asks whether models-and-inference customizations match the branch.
---

# Review Models & Inference Upstream Diff

## Purpose

Your job is to review the current repository and see if the upstream diffs from `docs/cohere/code_notes/upstream-diff-models-and-inference.md` are complied with the changes in the current branch. Also check if the custom code has been added. If there were changes such as restrcturing of the files, mention those files first, then let the reviewer review those file, after which an option needs to be provided, to update the `docs/cohere/code_notes/upstream-diff-models-and-inference.md` file based on the latest restrcturing.

Cross-reference `docs/cohere/code_notes/models-and-inference.md` for deeper behavior notes when a section needs more context.

## When to Use

- After rebase, upstream sync, or cherry-pick onto a new base (e.g. v0.21)
- Before PR review for models, pooling, reward, guided decoding, MoE routing, or spec-decode changes
- When the user asks whether models/inference fork diffs still match the doc
- After large upstream refactors that may have moved documented paths

## Scope

In-scope directories (models & inference extensions):

- `vllm/model_executor/models/` (especially `commandr*.py`, `cohere_reward.py`, `cohere2_moe.py`, `registry.py`)
- `vllm/model_executor/layers/` (pooler, fused_moe, quantization/online)
- `vllm/cohere/guided_decoding/`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/reasoning/`
- `examples/features/speculative_decoding/`
- Related tests under `tests/cohere/` and `tests/quantization/`

Out of scope unless referenced by the doc: CI workflows, Docker, generic runtime/scheduling (see other code notes).

## Workflow

### Step 1: Load the reference doc

Read `docs/cohere/code_notes/upstream-diff-models-and-inference.md` end-to-end.

Build an inventory table from each section:

| Section | Documented paths | Documented symbols / behavior |
|---------|------------------|-------------------------------|
| §1 Pooling | e.g. `pooler.py` | `DispatchPooler`, task-aware pooling |
| §2 Reward | `commandr.py`, `cohere_reward.py`, `registry.py` | `Cohere2ForRewardModel`, ranking head |
| §3 EAGLE | `commandr_eagle.py`, `kv_cache_utils.py` | `eagle_draft_model` prefix, draft KV grouping |
| §4 Guided decoding | `vllm/cohere/guided_decoding/*` | structural tags, tool grammar |
| §5 Spec decode | `spec_decode_offline.py` | `ngram-eagle`, acceptance metrics |
| §6 MoE routing | `commandr.py`, `fused_moe/*` | `SigmoidRenorm`, `token_choice_with_bias`, `norm_topk_prob` |
| §7 Performance | kernel benchmarks, compressed tensors | enablement deltas |
| §8 Hotspots | listed high-conflict files | validation checklist |

Use [reference.md](reference.md) for helper commands and a report template.

### Step 2: Detect file restructuring (report FIRST)

For every documented path, check existence on the current branch:

```bash
test -e <path> && echo OK || echo MISSING
```

When a path is **missing**, locate the successor before compliance checks:

1. `git log --follow --oneline -5 -- <old-path>` (if path existed on an older base)
2. `git ls-files | grep -i <basename>` or `Glob **/<basename>*`
3. Grep for section-specific symbols (e.g. `DispatchPooler`, `Cohere2ForRewardModel`)
4. Compare against upstream: `git diff <UPSTREAM_REF> --name-status -- <related-dir>`

Classify each restructuring:

| Type | Example | Action |
|------|---------|--------|
| **Renamed** | `pooler.py` → `pooler/` package | Map old → new paths |
| **Split** | single file → multiple modules | List all new files carrying the behavior |
| **Moved** | under `quantization/` → `quantization/online/` | Update doc paths |
| **Upstream absorbed** | behavior now in upstream without cohere markers | Flag for minimize-upstream-diff |
| **Removed** | custom code dropped | Flag as regression |

**Stop and present restructuring findings before deeper compliance review.**

Use this format:

```markdown
## File restructuring (review first)

| Doc path | Status | Current location(s) | Notes |
|----------|--------|---------------------|-------|
| `vllm/.../pooler.py` | RESTRUCTURED | `vllm/.../pooler/` | Upstream split into package |

**Reviewer action:** confirm the mapped paths still carry the documented behavior.
```

Wait for reviewer acknowledgment (or explicit "continue") before Step 3.

### Step 2b: Split-file porting check (mandatory for RESTRUCTURED / SPLIT rows)

When Step 2 finds a **Split** or **Renamed** row (especially monolithic file →
package), run the full procedure in
[`../_shared/split-file-porting-check.md`](../_shared/split-file-porting-check.md)
before marking any section compliant:

1. **Diff Cohere hunks** — compare `git diff <UPSTREAM_REF> -- <legacy-path>`
   vs `git diff <UPSTREAM_REF> -- <successor-package>/` for `cohere start` /
   `# cohere` / `// cohere` blocks.
2. **Confirm runtime imports** — grep importers (e.g.
   `compressed_tensors.py` importing `compressed_tensors_moe`) and resolve
   package `__init__.py` vs legacy sibling `.py` file.
3. **Flag duplicate implementations** — legacy monolithic file still on disk
   **and** successor package both defining the same entry symbol (e.g.
   `get_moe_method`, `CompressedTensorsWNA16MarlinMoEMethod`).

Add a **Split-file porting** table to the Step 2 report:

| Legacy path | Successor | Hunks only in legacy? | Runtime import | Status |
|-------------|-----------|----------------------|----------------|--------|
| `.../compressed_tensors_moe.py` | `.../compressed_tensors_moe/` | yes → **regression** | package | FAIL |

Treat **hunks only in legacy** as **Partial** or **Missing** in Step 3 — never
**Compliant** until ported. See [reference.md](reference.md) for commands.

Wait for reviewer acknowledgment on porting failures before Step 3.

### Step 3: Verify documented custom code (compliance)

For each section, after restructuring is mapped:

1. **Path check** — successor path(s) exist.
2. **Symbol check** — grep documented classes/functions/constants still present:
   ```bash
   rg -l 'Cohere2ForRewardModel|SigmoidRenorm|token_choice_with_bias' vllm/
   ```
3. **Behavior check** — read key hunks; confirm intent in the doc still holds (not just symbol names).
4. **Marker check** (modified upstream files only) — invoke `/check-cohere-markers` or spot-check `# cohere` / `// cohere` on fork diffs via:
   ```bash
   git diff <UPSTREAM_REF> -- <file>
   ```

Record per section:

- **Compliant** — paths mapped, symbols/behavior present
- **Partial** — behavior moved but doc or markers stale
- **Missing** — documented customization absent (regression)
- **Drift** — behavior changed in ways the doc does not describe

### Step 4: Find undocumented custom additions

Scan in-scope trees for fork-only code not covered by the doc:

```bash
# New files vs upstream under models/inference scope
git diff --name-status <UPSTREAM_REF> -- vllm/model_executor/ vllm/cohere/ vllm/reasoning/

# Cohere markers in scope
rg -l 'cohere start|# cohere|// cohere' vllm/model_executor/ vllm/cohere/ vllm/reasoning/ \
  --glob '!**/__pycache__/**'
```

Flag additions that are:

- **New customization** — should be documented
- **Already in** `models-and-inference.md` but not in `upstream-diff-models-and-inference.md` — doc sync gap
- **Incidental** — test-only or re-export; note but do not require doc update

Known post-v0.19 areas often missing from the short diff doc: `quantization/online/*`, `reload_weights`, `cohere2_moe.py` LoRA, SWA window, online FP8 PTPC.

### Step 5: Produce the review report

```markdown
# Models & Inference Upstream Diff Review

**Branch:** <branch> @ <sha>
**Upstream ref:** <UPSTREAM_REF>
**Reference doc:** docs/cohere/code_notes/upstream-diff-models-and-inference.md

## 1. File restructuring (reviewed first)
[table from Step 2]

## 2. Section compliance
| § | Topic | Status | Evidence |
|---|-------|--------|----------|
| 1 | Pooling | Compliant / Partial / Missing / Drift | ... |

## 3. Undocumented custom code
- [file/symbol] — suggested doc section

## 4. Recommended doc updates
- [bullet list of path/symbol/section edits]

## 5. Validation checklist (§8)
- [ ] reward models load
- [ ] pooling tasks dispatch
- [ ] EAGLE spec decode
- [ ] guided generation grammar + structural tags
- [ ] multimodal spec decode script
```

### Step 6: Offer doc update

After presenting the report, ask:

> The review found [N] restructuring change(s) and [M] doc drift item(s).  
> **Update `docs/cohere/code_notes/upstream-diff-models-and-inference.md` to reflect the latest restructuring?** (yes / no / section-by-section)

If **yes** or **section-by-section**:

1. Edit only `upstream-diff-models-and-inference.md` unless the user also requests `models-and-inference.md`.
2. Update paths, section numbers, hotspot list, and validation checklist to match current branch.
3. Keep behavior-focused prose; do not paste large code blocks.
4. If a customization was upstream-absorbed, note that in the doc and drop stale fork-only wording.
5. Summarize what changed in the doc in the response.

If **no**, leave docs unchanged and record open items for the reviewer.

## Upstream ref

Default: use `UPSTREAM_REF` from [`../_shared/upstream-file-collection.md`](../_shared/upstream-file-collection.md) Step B. Confirm with the user before diffing.

For v0.21 work, the merge-base tag against `upstream/main` is usually the right base.

## Related skills

- [`../_shared/split-file-porting-check.md`](../_shared/split-file-porting-check.md) — legacy vs package Cohere hunk diff, import resolution, duplicate impl flags (Step 2b)
- `/check-cohere-markers` — marker coverage on modified upstream files
- `/minimize-upstream-diff` — revert convergent or stale hunks before doc review
- `/check-docs-and-cursor-freshness` — broader docs/cursor freshness pass
- `/rebase-assistant` — Step 6b runs the same split-file check after rebase

## Anti-patterns

- Do not assume a missing path means the customization was removed — check restructuring first.
- Do not mark compliant based on symbol name alone; confirm behavior in code.
- Do not update the doc without showing the review report and getting user consent.
- Do not conflate `upstream-diff-models-and-inference.md` (short tracker) with `models-and-inference.md` (deep notes) — link between them when depth is needed.
