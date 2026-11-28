# SPDX-License-Identifier: Apache-2.0
"""SHARED semantics vector table — single source of truth for the window rule.

This table is consumed verbatim by BOTH test suites:
  - vLLM (this repo):      tests/v1/core/test_kv_load_policy.py
  - LMCache-Ascend (PR #365): tests/host/test_prefix_policy.py (WindowPolicyTest)

Adding/editing a row here MUST be mirrored into both tests; the two
implementations are not allowed to drift semantically (interface differences
CLI vs env are OK, semantic differences are not).
"""

from __future__ import annotations

# (matched_tokens_input, tau1, tau2, expected_load)  ->  may the scheduler
# rely on this many external tokens?
WINDOW_VECTORS: list[tuple[int, int, int | None, bool]] = [
    # boundary: H == tau1 -> load
    (1024, 1024, 23296, True),
    # inside the window -> recompute
    (1024 + 256, 1024, 23296, False),
    (23296 - 256, 1024, 23296, False),
    # boundary: H == tau2 -> load
    (23296, 1024, 23296, True),
    # beyond tau2 -> load
    (23296 + 256, 1024, 23296, True),
    # well below tau1 -> load
    (256, 1024, 23296, True),
    # far above tau2 -> load
    (30000, 1024, 23296, True),
    # legacy single threshold (tau2=None): <= tau1 loads, else recompute
    (1024, 1024, None, True),
    (1024 + 1, 1024, None, False),
    # zero matched (nothing to suppress; scheduler gates on ext_tokens anyway)
    (0, 1024, 23296, True),
    # custom thresholds
    (512, 512, None, True),
    (513, 512, None, False),
    (2048, 512, 17920, False),
    (17920, 512, 17920, True),
]

# Invalid configurations that MUST be rejected (pydantic validation).
INVALID_CONFIGS: list[dict[str, object]] = [
    {"tau1": 0},
    {"tau1": -5},
    {"tau1": 1024, "tau2": 1024},
    {"tau1": 1024, "tau2": 512},
    {"mode": "bogus"},
]
