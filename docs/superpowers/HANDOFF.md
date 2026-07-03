# Session Handoff — SAE Eviction Policy Integration

**Date written:** 2026-07-03
**Branch:** `session_aware_eviction`
**Fork:** `git@github.com:InbarShapira/vllm.git` (origin)
**Upstream:** `https://github.com/vllm-project/vllm.git` (upstream)
**Current tip:** `3e68e6ae5` — freshly rebased on `upstream/main`

Read this file first when resuming on a new server or new Claude Code session.

## What's done

The SAE (Session-Aware Eviction) cache policy has been implemented, tested,
audited against the reference plugin, parity-fixed, and pushed to the fork.
No PR has been opened yet.

All 11 tasks of the implementation plan are complete:

- Design: [`docs/superpowers/specs/2026-07-01-sae-eviction-policy-integration-design.md`](specs/2026-07-01-sae-eviction-policy-integration-design.md)
- Plan: [`docs/superpowers/plans/2026-07-01-sae-eviction-policy-integration.md`](plans/2026-07-01-sae-eviction-policy-integration.md)
- User-facing docs: [`docs/features/kv_offloading_usage.md`](../features/kv_offloading_usage.md) (see the "Eviction Policy" section)

The branch has 18 commits on top of upstream. Notable milestones after the
initial 11-task implementation:

- `1e98d6d3b` — reference-parity fixes (three unintended divergences found
  and corrected via option-1 audit against `sae_kv_offload` plugin)
- `2d4221463` — renamed counters to `vllm:kv_offload_*` prefix + folded in
  design-doc parity edits
- `3e68e6ae5` — expanded `SAECachePolicy` docstring in ARC-style structured
  form (Data Structures → Algorithm Flow → Session Score → Tunables →
  Semantic Differences)

## Test status

Last full run (post-rebase on upstream): **69 pass** across:

```bash
.venv/bin/python -m pytest \
  tests/v1/kv_offload/cpu/policies/ \
  tests/v1/kv_offload/cpu/test_manager.py \
  tests/v1/kv_offload/cpu/test_manager_policy_metrics.py \
  tests/v1/kv_offload/cpu/test_spec_config_validation.py \
  tests/v1/kv_offload/cpu/test_sae_end_to_end.py
```

Pre-existing failures in `test_shared_offload_region.py` and
`test_gpu_worker.py` are macOS `/dev/shm` and no-GPU issues unrelated to
this work — confirmed against `upstream/main`.

## Reference-parity audit summary

The port intentionally differs from the reference plugin
(`/Users/iklamer/ai-native-systems/VSCodeProjects/sae_kv_offload/`) in
three documented ways (see the design doc's "Semantic differences"
section):

1. Session boundaries reconstructed from the call sequence, not
   `prepare_store` batches.
2. Per-batch position weighting on lookup dropped (current scheduler
   feeds keys one at a time).
3. `start_pos` always zero (no batch context at `insert` time); both
   `_score` and the admission gate use a fixed `pos_bonus = 30000.0`.

Three **unintended** divergences were found and fixed in commit
`1e98d6d3b`:

1. `touch` was bumping `hits` per key; now bumps once per unique sid.
2. Decay wasn't truncating float `hits` to int; now does, matching
   `manager.py:161` in the reference.
3. Admission gate was mixing ghost scores into the new-session
   score; now excludes them per the reference's explicit design
   choice.

Expected TTFT impact vs LRU: the core mechanism (session grouping +
hot-session protection + admission gate) is intact; #2 above (dropped
position weighting) is the biggest gap and will slightly erode SAE's
edge on prefix-heavy workloads. Empirical validation deferred.

## Where we paused

The current session was mid-brainstorm on a benchmark harness (using
the `superpowers:brainstorming` skill) when we pivoted to push the
branch. **The brainstorm is not saved to a spec file yet** — the
answers below need to be re-loaded into the next session so it can
resume from the same point rather than restart questioning.

### Brainstorm answers so far

Purpose: measure whether SAE is worth using vs LRU/ARC in vLLM.

1. **Scope:** LRU vs ARC vs SAE (in-tree only). No reference-plugin
   parity comparison in the benchmark — that was covered by the
   code-review audit.
2. **Drivers:** Both.
   - Synthetic driver against `CPUOffloadingManager` for iteration
     (fast, deterministic, characterizes behavior across workload
     shapes).
   - End-to-end via a vLLM serve + `benchmarks/benchmark_serving.py`
     variant for final numbers reviewers will believe.
3. **Synthetic workload shapes:** all four —
   - Session-structured hot/cold mix (Zipf-distributed sessions with
     shared prefix + varying tails)
   - Uniform random no-session (adversarial to SAE)
   - Bursty prefix reuse (long shared prefixes with per-request tails)
   - Cyclic revisit (sessions go cold long enough to test ghost
     persistence, then return)
4. **Metrics (apply to all 3 policies where meaningful):**
   - Hit rate (per policy) — primary
   - Eviction counter deltas — churn signal
   - SAE admission-gate denial rate — SAE-specific sanity check
   - TTFT (e2e only)
   - Throughput (e2e only)
   - Per-session hit-rate breakdown by hot/warm/cold band
5. **Location:** `benchmarks/kv_offload/` (new directory, mirrors how
   `benchmarks/attention_benchmarks/` is organized).
6. **Reporting:** JSON per run + markdown summary table (policy ×
   workload grid). No plotting dependencies.
7. **E2E target:** realistic 7B-class model on a single GPU (Llama-3-8B
   or Qwen-2.5-7B). CPU-only e2e was rejected.

### Still open

- Statistical rigor (N runs per config, warmup, mean/stddev vs mean/P95)
   — user rejected the question mid-flow to switch to server-startup
  testing. Recommended default: N=3, discard first as warmup, report
  mean + stddev. Confirm with user.
- Whether to include the reference plugin as a **fourth** benchmark
  target for reality-check purposes. Original scope said no; worth
  re-asking now that we're planning e2e infra anyway.
- Reference to any existing vLLM CI perf infrastructure worth reusing
  (`benchmarks/benchmark_serving.py` was noted but not yet inspected
  for reuse patterns).

### Next actions (in order)

1. Confirm the outstanding open questions above.
2. Propose 2–3 harness structural approaches (single script vs
   pytest-integrated vs Makefile-driven) — see superpowers brainstorming
   skill; presenting approaches is the step we didn't reach.
3. Present the design in sections; get approval on each.
4. Write the design doc to
   `docs/superpowers/specs/2026-07-03-sae-benchmark-harness-design.md`.
5. Spec self-review; user review gate.
6. Invoke `superpowers:writing-plans` to produce the implementation plan.
7. Only then start writing benchmark code.

## Server-startup smoke test (also from the paused session)

User expressed interest in verifying vLLM starts with SAE selected
before writing benchmark code. The minimal check:

```bash
.venv/bin/python -c "import vllm; print(vllm.__version__)"

.venv/bin/vllm serve facebook/opt-125m \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "cpu_bytes_to_use": 268435456,
      "eviction_policy": "sae"
    }
  }' \
  --disable-log-requests
```

Look for the INFO line `CPU offload: eviction_policy=sae` — that's the
log line from Task 9 confirming the SAE path is active. Then in a
second shell:

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/completions \
  -H "content-type: application/json" \
  -d '{"model": "facebook/opt-125m", "prompt": "Hello", "max_tokens": 8}'
```

The reference plugin lives at
`/Users/iklamer/ai-native-systems/VSCodeProjects/sae_kv_offload/` on the
old server — that path won't exist on the new server. If the audit
needs to be re-run, the plugin is publicly available at
<https://github.com/anthropics/sae_kv_offload> (or wherever it's
published) — check the design doc for the exact reference.

## Environment reminders

Per repo AGENTS.md — non-negotiable:

- Never use system `python3` or bare `pip` — everything goes through
  `uv` and `.venv/bin/python`.
- Every commit signed off (`git commit -s`), with `Assisted-by: Claude`
  trailer.
- `pre-commit run` on changed files before every commit.
- Line length 88 for Python. Google-style docstrings.

## Where to open the PR when ready

GitHub already surfaced the URL when we pushed:

```text
https://github.com/InbarShapira/vllm/pull/new/session_aware_eviction
```

Draft PR description guidance is in the message log immediately
preceding the fork/push work — the four sections a vLLM PR needs are
Summary, Duplicate-check evidence, Test plan, and AI-assistance
statement (per AGENTS.md).

**Recommendation from the paused session:** run the benchmark harness
before opening the PR, so real perf numbers can go in the description
rather than triggering weeks of "please add benchmarks" feedback.

## What to say to a fresh Claude Code session

Paste this to start:

> Resuming SAE eviction work on the vllm repo, branch
> `session_aware_eviction`. Read `docs/superpowers/HANDOFF.md` first —
> it captures the full state, what's committed, what's paused, and
> what comes next.
