# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest

from vllm.models.minimax_m3.amd.specialized_decode.mono.viability import (
    FfnViabilityEvidence,
)


@pytest.fixture
def valid_evidence() -> FfnViabilityEvidence:
    return FfnViabilityEvidence(
        activation_mode="bf16",
        tp_size=4,
        hidden_size=6144,
        intermediate_size=768,
        routed_experts=128,
        shared_experts=1,
        top_k=4,
        grid_ctas=256,
        threads_per_cta=512,
        lds_bytes=96 * 1024,
        vgpr_spills=0,
        resident_workgroups_per_cu=1,
        real_layer_speedup=1.01,
    )


def test_complete_evidence_passes(valid_evidence: FfnViabilityEvidence):
    assert valid_evidence.proven
    assert valid_evidence.refusal_reasons() == []


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("activation_mode", "mxfp4", "activation_mode"),
        ("tp_size", 8, "tp_size"),
        ("routed_experts", 256, "routed_experts"),
        ("top_k", 8, "top_k"),
        ("grid_ctas", 128, "grid_ctas"),
        ("vgpr_spills", 1, "VGPR spills"),
        ("resident_workgroups_per_cu", 0, "resident CTA"),
        ("real_layer_speedup", 1.0, "must beat"),
    ],
)
def test_incomplete_or_glm_specific_evidence_is_refused(
    valid_evidence: FfnViabilityEvidence,
    field: str,
    value: object,
    reason: str,
):
    evidence = replace(valid_evidence, **{field: value})
    assert not evidence.proven
    assert any(reason in item for item in evidence.refusal_reasons())
