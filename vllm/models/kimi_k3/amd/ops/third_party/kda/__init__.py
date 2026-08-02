# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# AMD/ROCm vendored copy of the Kimi-K3 KDA triton kernels.
#
# Provenance: mirror of vllm/models/kimi_k3/nvidia/ops/third_party/kda (tracker
# mke-tracker @ 7adebfcf9; FLA vendored per PRs #39/#86). Split per-vendor so
# AMD can carry gfx950-specific kernel changes without touching the NVIDIA copy.
#
# fla-org/flash-linear-attention#869 (the unmerged ROCm fixes our earlier
# amd_fla shim carried against FLA 0.5.0) is covered here by the *newer* vendored
# FLA rather than the literal patch:
#   - transpose-state-layout workaround: N/A (kernels rewritten; no
#     transpose_state_layout path remains),
#   - AMD autotune configs: present (is_amd num_warps/num_stages branches),
#   - OOB-mask correctness fix: present (all tl.load use mask=..., other=0).
# Validated on gfx950: no core-dump, gsm8k 94.1%.
# Validated on gfx942 (MI325X): tests/models/kimi_k3/test_kda.py and
# test_kda_metadata.py pass (45 passed, 6 skipped).
#
# Deltas vs the NVIDIA copy, which has since diverged forward:
#   - no PDL: the NVIDIA fused_recurrent kernels gate on
#     `current_platform.is_arch_support_pdl()` and issue gdc_wait /
#     gdc_launch_dependents, which have no ROCm equivalent,
#   - fused_recurrent decode launch config is a flat `BV = 32 if
#     use_gate_in_kernel else 8` rather than the NVIDIA GB300-tuned
#     BV/num_stages ladder; it has not been retuned for CDNA.
# Keep in sync with the NVIDIA copy on FLA updates; any further divergence
# should be an intentional, documented AMD-only change (a #869-style fix).

from .chunk import (
    chunk_kda,
    chunk_kda_fwd,
    chunk_kda_with_fused_gate,
    chunk_kda_with_fused_gate_fwd,
    fused_kda_gate,
    fused_kda_gate_chunk_cumsum,
)
from .fused_recurrent import (
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
)

__all__ = [
    "chunk_kda",
    "chunk_kda_fwd",
    "chunk_kda_with_fused_gate",
    "chunk_kda_with_fused_gate_fwd",
    "fused_kda_gate",
    "fused_kda_gate_chunk_cumsum",
    "fused_recurrent_kda",
    "fused_recurrent_kda_fwd",
    "fused_recurrent_kda_packed_decode",
]
