## Overview

Replace #49555's slot-indexed Indexer Store with the current AuxOutput SHM
lifecycle, sharing KV-compatible hashes, asynchronous output ownership and
accepted-token handling with routed-expert return. Implements the selection
capture/return path described in #47280.

## Claims

- Add model-owned capture sources for DeepSeek V3.2/GLM-5.2, DeepSeek V4/V4.1,
  GLM-5.3 KPool, Qwen4Exp and Step5; record real producer/consumer layer mappings.
- Add fixed-width profiles, lossless Pool/QSA codecs, raw/packed terminal output,
  independent prompt offsets and startup memory diagnostics.
- Reuse the existing AuxOutput Scheduler/Worker lifecycle; preserve R3-only
  storage and support either artifact or both in one record.
- Allocate bounded GPU capture/snapshot storage before KV memory planning.

These are implementation claims pending behavioral validation, not a statement
that every model/backend has passed testing.

## Validation

Per user instruction, **no pytest, model evaluation or GPU/H20 E2E was run** for
this migration. Historical test results from the old PR do not apply.

Ruff, Python 3.11 syntax, diff whitespace and selected repository static rules
pass. Focused mypy of the four core AuxOutput modules passes with
`--follow-imports=skip`; full mypy is not passing in this environment. Local
system-Python Pydantic is 2.11.7 versus the repository's >=2.12 requirement;
the shared project venv has Pydantic 2.13.4 but no mypy. An unchanged-main
comparison also reports existing diagnostics. Rust tooling is unavailable.

Added CPU oracle/state-machine regressions and extended existing config,
HTTP/output and DeepSeek V4.1 replay-row tests. Full reproduction commands,
coverage gaps and actual check results are in
`.codex-plans/indexer-aux-output/REVIEW.md`.

## Details

Duplicate check: this updates the existing Indexer Replay PR #49555 rather than
opening a competing proposal. Open PRs referencing #47280 were checked.
#57968's Mooncake transport is separate and is not a dependency.

The former Slot Store cannot restore selections using the new AuxOutput prefix
hash lifecycle. This implementation uses one fixed combined Artifact Record,
the existing Block/Tail pool, background SHM writer, reference tracking, generation
and pending-output cleanup. Per-source hooks snapshot selections before shared
buffers are overwritten. The output process decodes only raw mode; packed mode
stays compact through HTTP encoding.

Scope/limitations: MRV2 generate runner with prefix caching; current AuxOutput
PP/DCP/PCP, adaptive-verification and remote-PD restrictions remain. MiniMax-M3
has been audited but its independent per-head Block selection has no adapter.
AMD QSA uses raw output. Rust's raw wire protocol retains the new fields, but
Rust HTTP delivery is not added. SHM retains immutable first-writer semantics.
CUDA Graph, real TP/DP, model-quality and performance claims need later evidence.

AI assistance was used for implementation and static review. The human submitter
must review every changed line and run/accept the relevant validation before
requesting maintainer review or publishing verified support claims.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill.
- [x] AI assistance was used during the creation of this PR.
- [x] **Design Fit:** Reuses the existing AuxOutput control/data plane.
- [ ] **Testing and Validation:** Deferred by the user; no model/GPU validation yet.
- [x] **Code Quality and Style:** Static checks and usage/design documentation prepared.
- [x] **Pull Request Contents:** Motivation, duplicate check, tradeoffs and validation gaps recorded.

</details>
