# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-version CRIU + cuda-checkpoint snapshot support for vllm startup.

This module is a scaffolding prototype — the real CRIU and cuda-checkpoint
subprocess calls are gated behind `VLLM_SNAPSHOT_ENABLED=1` and currently
run in dry-run mode (logging actions, not invoking the binaries). The goal
is to prove the integration points exist in the right places and the
protocol between snapshot creator, restored process, and CLI is sound.

See .startup-bench/design/criu_cuda_checkpoint_plan.md for the full
design.
"""

from vllm.snapshot.keying import compute_snapshot_key, snapshot_root

__all__ = ["compute_snapshot_key", "snapshot_root"]
