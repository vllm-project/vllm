# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evidence gate for attaching the gfx950 FFN kernel.

Compile success is not a residency or performance proof.  The first executor
must provide the post-link resource report and a real-layer eager comparison
before it can be selected by the host lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass

BLOCKS = 256
WAVE_SIZE = 64
MAX_LDS_BYTES = 96 * 1024


@dataclass(frozen=True)
class FfnViabilityEvidence:
    activation_mode: str
    tp_size: int
    hidden_size: int
    intermediate_size: int
    routed_experts: int
    shared_experts: int
    top_k: int
    grid_ctas: int
    threads_per_cta: int
    lds_bytes: int
    vgpr_spills: int
    resident_workgroups_per_cu: int
    real_layer_speedup: float

    def refusal_reasons(self) -> list[str]:
        reasons: list[str] = []
        expected = {
            "activation_mode": "bf16",
            "tp_size": 4,
            "hidden_size": 6144,
            "intermediate_size": 768,
            "routed_experts": 128,
            "shared_experts": 1,
            "top_k": 4,
            "grid_ctas": BLOCKS,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                reasons.append(f"{name} must be {value}")
        if (
            self.threads_per_cta <= 0
            or self.threads_per_cta > 1024
            or self.threads_per_cta % WAVE_SIZE
        ):
            reasons.append("threads_per_cta must be a legal whole-wave workgroup")
        if not 0 < self.lds_bytes <= MAX_LDS_BYTES:
            reasons.append(f"lds_bytes must be in (0, {MAX_LDS_BYTES}]")
        if self.vgpr_spills:
            reasons.append("post-link resource report must have no VGPR spills")
        if self.resident_workgroups_per_cu < 1:
            reasons.append("post-link occupancy must prove one resident CTA per CU")
        if self.real_layer_speedup <= 1.0:
            reasons.append("real-layer eager FFN must beat the native seam")
        return reasons

    @property
    def proven(self) -> bool:
        return not self.refusal_reasons()
