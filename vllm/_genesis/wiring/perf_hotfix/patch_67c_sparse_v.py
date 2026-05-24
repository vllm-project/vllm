# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 67c — Per-row vote sparse-V integration into P67.

================================================================
What this patch does
================================================================

P67c is a CONFIGURATION-only patch — no monkey-patch, no text-patch.
The P67 kernel (`vllm/_genesis/kernels/p67_multi_query_kernel.py`)
reads sparse-V env vars at launch time and passes them as constexpr
to the Triton split-M kernel. P67c just registers the metadata in
the dispatcher so operators see the option in apply matrix.

When env vars are set:
- `GENESIS_ENABLE_P67_SPARSE_V=1` — enable per-q_t sparse-V skip
- `GENESIS_P67_SPARSE_V_THRESHOLD=<float>` — skip threshold (default 0.001)
- `GENESIS_P67_SPARSE_V_SINK_TOKENS=<int>` — protected positions (default 4)

The kernel-side logic (constexpr-DCE'd when SPARSE_V=0):

```python
skip_pv_t = False
if SPARSE_V:
    tile_protected = start_n < SINK_TOKENS
    if not tile_protected:
        p_t_max = tl.max(P_t)  # uniform-scalar
        skip_pv_t = p_t_max < SPARSE_V_THRESHOLD

if skip_pv_t:
    acc_new_t = acc_old_t * alpha_t[:, None]  # decay only
else:
    # full V@P dot
```

================================================================
Bit-exact contract
================================================================

When threshold=0.0: `p_t_max < 0.0` is ALWAYS False (P_t >= 0).
Therefore SPARSE_V=1 + threshold=0 == SPARSE_V=0 (byte-equivalent
output). This matches TheTom #41422 PR contract.

When SPARSE_V=0 (default): the entire sparse-V block is constexpr-
DCE'd at compile time. Triton produces SASS byte-equivalent to
the pre-sparse-V P67 kernel.

================================================================
Why per-q_t (not per-tile uniform vote)
================================================================

P67 split-M has K_PLUS_1 independent q_t (e.g. 4 for MTP K=3). Each
q_t has its own attention distribution P_t. Per-q_t skip gives finer
granularity:
- One q_t can skip a tile while another q_t in same tile takes V@P
- More skip opportunities than per-tile uniform vote

V_tile load is shared across K_PLUS_1 q_t (single load per outer
iter), so we cannot skip V load itself. But the per-q_t V@P tl.dot
IS independent — we can skip individual ones.

Maximum theoretical savings: 1/K_PLUS_1 per skipped (q_t, tile) pair
of the V@P matmul cost. On 35B MTP K=3 with K_PLUS_1=4 and ~50% skip
rate on long-ctx: ~12.5% of V@P time saved.

================================================================
Safety
================================================================

- Default OFF (opt-in via `GENESIS_ENABLE_P67_SPARSE_V=1`).
- Constexpr-DCE invariant: SPARSE_V=0 path SASS byte-equivalent to
  pre-sparse-V P67 v17 split-M kernel.
- Threshold=0.0 invariant: SPARSE_V=1 + threshold=0 bit-exact.
- Requires P67 base (`GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1`).
  If P67 disabled, P67c skips with clear reason.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Synthesizes: PN26b proven uniform-scalar `if` pattern (Triton 3.6),
TRT-LLM #9821 sink protection design, TheTom #41422 threshold=0
bit-exact contract.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.p67c_sparse_v")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def apply() -> tuple[str, str]:
    """Apply P67c — sparse-V config validator (no monkey-patch)."""
    if not _env_truthy("GENESIS_ENABLE_P67_SPARSE_V"):
        return "skipped", (
            "opt-in: set GENESIS_ENABLE_P67_SPARSE_V=1 to engage per-q_t "
            "sparse-V skip in P67 split-M kernel"
        )
    # Verify P67 base is enabled — sparse-V is meaningless without it.
    if not _env_truthy("GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL"):
        return "skipped", (
            "P67 base kernel not enabled (GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL "
            "must be 1) — P67c is a no-op without the parent kernel"
        )

    threshold = os.environ.get("GENESIS_P67_SPARSE_V_THRESHOLD", "0.001")
    sink = os.environ.get("GENESIS_P67_SPARSE_V_SINK_TOKENS", "4")

    log.info(
        "[P67c] sparse-V config: threshold=%s, sink_tokens=%s. "
        "Kernel will pass these as constexpr at next launch.",
        threshold, sink,
    )

    return "applied", (
        f"P67c sparse-V config registered (threshold={threshold}, "
        f"sink_tokens={sink}). Per-q_t skip via uniform-scalar `if` in "
        "P67 split-M kernel. Bit-exact at threshold=0; constexpr-DCE'd "
        "to baseline at SPARSE_V=0."
    )


def is_applied() -> bool:
    """Passive (config-only) — applied iff env enabled."""
    return _env_truthy("GENESIS_ENABLE_P67_SPARSE_V") and _env_truthy(
        "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL"
    )
