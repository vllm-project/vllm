# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron shared-context packed reads (CPU): the plan is the trainer's
mask, and the readout keeps every slot's logprobs."""

import math

import numpy as np
import pytest
import torch

from vllm.model_executor.models.nemotron_dllm_packed import (
    PackedReadPlan,
    build_packed_plan,
    packed_attention_mask,
    parse_sc_layout,
    write_packed_readout,
)

pytestmark = pytest.mark.cpu_test


def _plan_mask(p, T):
    """Dense mask implied by the plan's query segments + gathered keys."""
    allowed = np.zeros((T, T), dtype=bool)
    for i in range(len(p["cu_q"]) - 1):
        q0, q1 = p["cu_q"][i], p["cu_q"][i + 1]
        k0, k1 = p["cu_k"][i], p["cu_k"][i + 1]
        for k in p["kv_idx"][k0:k1]:
            allowed[q0:q1, k] = True
    return allowed


@pytest.mark.parametrize(
    "layouts,lens",
    [
        ([(5, [3, 4, 2], [[7, 8], [9], [1, 2, 3]])], [14]),
        ([None, (16, [5, 3], [[1], [2, 3]]), None], [7, 24, 3]),
        ([(0, [2, 2], [[1], [2]])], [4]),
    ],
)
def test_plan_matches_dense_mask(layouts, lens):
    qsl = np.concatenate([[0], np.cumsum(lens)])
    p = build_packed_plan(qsl, layouts)
    T = int(qsl[-1])
    assert (_plan_mask(p, T) == packed_attention_mask(qsl, layouts)).all()
    # question positions restart at |S|
    for i, layout in enumerate(layouts):
        if layout is None:
            continue
        s, ql, _ = layout
        at = qsl[i] + s
        for n in ql:
            idx = np.searchsorted(p["pos_idx"], at)
            assert list(p["pos_val"][idx : idx + n]) == list(range(s, s + n))
            at += n


def test_parse_layout_rejects_bad():
    assert parse_sc_layout(None, 5) is None
    with pytest.raises(ValueError):
        parse_sc_layout('{"s": 2, "q": [3], "c": [[1]]}', 6)
    with pytest.raises(ValueError):
        parse_sc_layout('{"s": 2, "q": [3], "c": [[]]}', 5)


def _dev(p):
    keys = [f for f in PackedReadPlan.__dataclass_fields__]
    return PackedReadPlan(
        **{
            k: (p[k] if isinstance(p[k], int) else torch.from_numpy(np.asarray(p[k])))
            for k in keys
        }
    )


def test_readout_preserves_logprobs_and_mass():
    torch.manual_seed(0)
    V = 50
    layouts = [None, (4, [3, 2, 2], [[10, 11, 12], [20, 21], [30, 31]])]
    qsl = np.array([0, 6, 17])
    plan = _dev(build_packed_plan(qsl, layouts))
    logits = torch.randn(2, V, dtype=torch.bfloat16)
    slot_logits = torch.randn(3, V, dtype=torch.bfloat16) * 3
    out = write_packed_readout(logits, plan, slot_logits)
    assert out.dtype == torch.float32
    assert torch.equal(out[0], logits[0].float())  # plain row unchanged
    lp_row = torch.log_softmax(out[1], -1)  # processed_logprobs
    assert abs(torch.logsumexp(out[1], -1).item()) < 1e-5
    true = torch.log_softmax(slot_logits.float(), -1)
    at = 0
    for j, cand in enumerate(layouts[1][2]):
        got = lp_row[at : at + len(cand)] + math.log(3)
        assert torch.allclose(got, true[j, cand], atol=1e-5)
        at += len(cand)
