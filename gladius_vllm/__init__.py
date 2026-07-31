"""GLADIUS control-plane integration for vLLM's V1 scheduler.

Additive plugin package (not part of `vllm/`) that lets an external experience
control plane influence live scheduling decisions via a versioned, file-based
protocol (`policy_snapshot.json` in, `telemetry.jsonl` out). See
`gladius_vllm.scheduler.GladiusScheduler` for the entry point, wired in via
vLLM's existing `--scheduler-cls` plugin mechanism.
"""

from gladius_vllm.schema import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
