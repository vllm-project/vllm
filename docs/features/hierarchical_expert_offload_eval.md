# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eval / bakeoff notes template for hierarchical expert staging PRs.

Per AGENTS.md, model-affecting changes must report eval commands and results.
Fill this in when running on XPU hardware:

## Commands

```bash
# Unit (no GPU for most cases)
.venv/bin/python -m pytest tests/model_executor/offloader/test_hierarchical_offload.py -v

# Correctness vs baseline (GPU/XPU)
.venv/bin/python -m pytest tests/basic_correctness/test_hierarchical_offload.py -v

# Throughput bakeoff (hal / XPU default model)
.venv/bin/python benchmarks/hierarchical_tier_bakeoff.py \\
  --model /tank/nas/models/Mixtral-8x22B-Instruct-v0.1-AWQ \\
  --tier-num-slots 4 --tier-ram-gb 32 --max-tokens 64 --num-prompts 8 \\
  --colibri-tok-s 0.1 --output /tmp/tier_bakeoff.json
```

Hardware e2e for this feature uses **Mixtral-8x22B Instruct AWQ (Q4)**
(`MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ`). Use `--tier-num-slots 4`
(of 8 experts) so staging is exercised; raise to 8 for full residency.

## Results (fill on hardware)

| Setup | tok/s | TTFT proxy | device hit rate | Notes |
|-------|-------|------------|-----------------|-------|
| baseline (no offload) | | | n/a | Mixtral-8x22B Q4 |
| hierarchical slots=4 | | | | Mixtral-8x22B Q4 |
| Colibri reference | | | | |

AI assistance was used for this feature implementation.
"""
