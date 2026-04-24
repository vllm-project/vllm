# Cohere Code Notes Overview

This directory contains deep technical notes for major Cohere-specific codepaths, intended as "future self" reference for implementation understanding, debugging, and maintenance.

Many notes were originally derived from analyzing the branch delta against upstream `v0.14.1`, then expanded into behavior-focused documentation.

## Read This First

- `ci-and-automation.md`: CI graph, dispatcher matrix logic, result upload semantics, rerere rebase automation.
- `build-and-packaging.md`: Docker/Makefile build strategy, wheel handoff, sccache WebDAV changes, ROCm deltas.
- `runtime-and-scheduling.md`: thinking-budget execution path, scheduler/model-runner coupling, structured-output safeguards, SHM lifecycle changes.
- `models-and-inference.md`: reward/pooler stack, Command-R/EAGLE integration points, KV grouping for draft layers, spec-decode tooling, SigmoidRenorm routing and `norm_topk_prob` for Cohere MoE.
- `tests-benchmarks-and-data.md`: Cohere test topology, hardware/model mapping configs, benchmark dataset extensions, reporting outputs.

## Why This Split Exists

The original top-level summary was useful but too shallow for detailed engineering work. These deep notes add:

- behavior-level intent (what runtime behavior changed),
- integration boundaries (what must stay consistent across files),
- typical failure modes (what tends to break first),
- change hot spots (files likely to conflict or drift from upstream).
