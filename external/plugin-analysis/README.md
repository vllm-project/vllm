# Plugin analysis workspace

This directory holds **git clones** used for the dllm-plugin evidence-driven analysis (reference plugins only).

- **Path:** `external/plugin-analysis/` (relative to the vLLM repo root used as context).
- **Clones here:** `bart-plugin`, `vllm-metal` — each subdirectory is its own repo. Do **not** run `git` for those clones in the parent vLLM tree; use each subdirectory as its own repo.
- **dLLM plugin (PRs / implementation):** use the working copy at the vLLM repo root: [`dllm-plugin/`](../dllm-plugin/) — **not** duplicated under `plugin-analysis/`.

Deliverables in this folder:

- **`ANALYSIS.md`** — evidence-driven repo comparison and RFC traceability.
- **`docs/MVP-ARCHITECTURE.md`** — MVP plugin architecture (components, field mapping, Mermaid diagrams).
