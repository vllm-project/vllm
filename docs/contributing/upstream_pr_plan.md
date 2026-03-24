# Upstream PR Plan for the Local Runtime Work

This repository contains a larger local-runtime product layer. That full change
set is not a realistic first upstream PR for `vllm-project/vllm`.

The first PR should be intentionally narrow and non-breaking.

## Recommended First PR

Title direction:

- `Make top-level vllm CLI help and dispatch avoid eager subcommand imports`

Scope:

- keep `vllm.entrypoints.cli` lightweight
- avoid importing every subcommand module before top-level argument parsing
- preserve existing command behavior once a specific subcommand is selected
- add regression tests for:
  - `vllm --help`
  - `vllm collect-env --help`
  - selected-command dispatch importing only the requested module

Why this is a good first PR:

- it is small enough to review
- it is easy to defend technically
- it addresses a real startup and import-safety problem
- it does not force a product direction change on upstream

## Changes That Should Not Be in the First PR

These should be proposed later, and likely in multiple PRs:

- repo bootstrap installer in `scripts/install.sh`
- standalone launcher in `scripts/vllm_launcher.py`
- local model alias catalog
- `pull`, `run`, `ls`, `aliases`, `inspect`, `ps`, `stop`, `logs`, `rm`
- changing `vllm serve` defaults
- rewriting the top-level project README around the local-runtime story

## Follow-up PR Sequence

1. CLI startup and lazy-import improvements
2. discussion or RFC for a local-runtime UX in upstream
3. narrow alias support, if upstream wants short model names
4. optional local service-management features
5. optional installer/bootstrap workflow, if upstream wants a source-checkout installer

## Before Opening the First Upstream PR

- run duplicate-work checks against upstream issues and open PRs
- run focused tests for the CLI changes
- verify behavior on Linux with the standard vLLM install path
- prepare a PR description that clearly states:
  - why the change is not duplicating an open PR
  - what tests were run
  - that AI assistance was used
