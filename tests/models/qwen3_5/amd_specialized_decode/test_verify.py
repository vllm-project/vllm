# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP verify steps on the host: the widths, which steps route to the kernels
(`runtime.routing.step_reason`) and the speculative-decoding refusals."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen3_5.amd.specialized_decode.runtime import (
    decode_widths,
    spec_reasons,
    verify_rows,
)
from vllm.models.qwen3_5.amd.specialized_decode.runtime.routing import step_reason

GDN = "model.layers.0.linear_attn"


def _vllm_config(sizes, spec=None):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=sizes),
        speculative_config=spec,
    )


def _spec(method="mtp", k=1, adaptive=False, dynamic=False):
    return SimpleNamespace(
        method=method,
        num_speculative_tokens=k,
        enable_adaptive_verification=adaptive,
        uses_dynamic_speculative_decoding=lambda: dynamic,
    )


def _md(
    reqs,
    q,
    *,
    decodes=0,
    prefills=0,
    uniform=True,
    spec=True,
):
    m = SimpleNamespace(
        num_prefills=prefills,
        num_decodes=decodes,
        num_spec_decodes=reqs if spec else 0,
        spec_sequence_masks=torch.ones(reqs, dtype=torch.bool) if spec else None,
        uniform_spec_sequence_length=q if uniform else None,
        spec_state_indices_tensor=torch.zeros(reqs, q, dtype=torch.int32),
        non_spec_state_indices_tensor=torch.zeros(reqs * q, dtype=torch.int32),
    )
    return {GDN: m}


def test_widths():
    sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 128]
    assert decode_widths(_vllm_config(sizes)) == (1, 2, 4, 8, 16, 24, 32, 48, 64)
    assert decode_widths(_vllm_config(sizes), 2) == (2, 4, 8, 16, 24, 32, 48, 64)
    assert decode_widths(_vllm_config(sizes), 4) == (4, 8, 16, 24, 32, 48, 64)
    assert decode_widths(_vllm_config(sizes), 3) == (24, 48)
    assert verify_rows(_vllm_config(sizes)) == 1
    assert verify_rows(_vllm_config(sizes, _spec(k=3))) == 4


@pytest.mark.parametrize("q", [2, 4])
def test_verify_routing(q):
    assert step_reason(_md(8, q), GDN, 8 * q, q) is None
    assert step_reason(_md(8, q), GDN, 16 * q, q) == "not uniform"
    assert step_reason(_md(8, q, uniform=False), GDN, 8 * q, q) == "not uniform"
    assert step_reason(_md(8, q, decodes=1), GDN, 8 * q, q) == "not pure verify"
    assert step_reason(_md(8, q, prefills=1), GDN, 8 * q, q) == "not pure decode"
    assert step_reason(_md(8, q, spec=False), GDN, 8 * q, q) == "not spec-verify"
    # a verify step without MTP widths runs stock
    assert step_reason(_md(8, q), GDN, 8 * q) == "spec-verify"


def test_spec_refusals():
    assert spec_reasons(None) == []
    assert spec_reasons(_spec(k=3)) == []
    assert spec_reasons(_spec(method="eagle3")) == [
        "speculative method 'eagle3' (MTP only)"
    ]
    assert spec_reasons(_spec(k=64)) == ["64 speculative tokens"]
    assert spec_reasons(_spec(adaptive=True, dynamic=True)) == [
        "adaptive verification",
        "dynamic speculative decoding",
    ]
