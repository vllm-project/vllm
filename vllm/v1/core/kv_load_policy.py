# SPDX-License-Identifier: Apache-2.0
"""KV load/recompute policy for vLLM v1 scheduler.

Target-agnostic decision module: it only answers *whether the scheduler may
rely on externally matched KV tokens*. The scheduler applies the decision in
the same place the request-level ``skip_reading_external_kv`` override lives
(see ``vllm/v1/core/sched/scheduler.py``): turning ``ext_tokens`` to 0 makes
the scheduler recompute that prefix — the same semantics a connector would use
to suppress retrieval.

Important: possession of policy semantics. The shared, single source of truth
for the window decision is this module. The LMCache-Ascend experimental stack
(PR #365) keeps a copy of the same semantics; both test suites consume the same
input/expected-output vector table (see tests/v1/core/__shapes__.py) to prevent
semantic drift. Interface differences (CLI config vs env vars) are allowed;
semantic differences are not.

Configuration enters through ``--kv-transfer-config``'s ``kv_load_policy``
object (``vllm/config/kv_transfer.py``). No new top-level CLI flags;
invalid values fail at argument-parsing time.
"""

from __future__ import annotations

from vllm.config.kv_transfer import KVLoadPolicyConfig


class StaticThresholdWindowPolicy:
    """Dual-threshold window on the matched-prefix length.

    Decision input is the **incremental external token count** reported by the
    connector at the scheduler decision point (matched tokens beyond any local
    prefix-cache hits). With ``tau2=None`` this degrades to the legacy
    single-threshold rule. Boundary semantics (fixed; see the shared test
    vector table):

      ``matched <= tau1 -> load``; ``tau1 < matched < tau2 -> recompute``;
      ``matched >= tau2 -> load`` (i.e. both ``== tau1`` and ``== tau2`` load).
    """

    def __init__(self, cfg: KVLoadPolicyConfig) -> None:
        self.tau1 = cfg.tau1
        self.tau2: int | None = cfg.tau2

    def should_load(self, *, matched_tokens: int) -> bool:
        """Return True if the scheduler may rely on ``matched_tokens`` KV."""
        if matched_tokens <= self.tau1:
            return True
        if self.tau2 is None:
            return False
        return matched_tokens >= self.tau2

    def describe(self) -> dict[str, object]:
        """One-line config snapshot for the debug startup log."""
        return {
            "mode": "static_threshold_window",
            "tau1": self.tau1,
            "tau2": self.tau2,
        }
