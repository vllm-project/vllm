# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The guard against AITER's silently-wrong fp8 MLA decode kernel.

AITER dispatches `mla_a8w8_qh64_qseqlen4_gqaratio16` for fp8 query + fp8 KV at
gqa_ratio 16 and decode_qlen 3-4 when the call is not persistent, and that
kernel returns output uncorrelated with the reference while raising nothing
(cos_diff 0.840 / 0.898 against AITER's own 3e-2 bar, measured on gfx950 with
AITER's own op_tests/test_mla.py).

Each condition below is tested in BOTH directions: the combination fires, and
flipping any single variable stops it firing. The one-variable controls mirror
the controls that isolated the bug on hardware -- bf16 instead of fp8,
gqa_ratio 128 instead of 16, and qlen <= 2 -- so a guard that quietly widened
to catch working configurations would fail here.

See https://github.com/vllm-project/vllm/issues/66141
"""

import pytest

from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    _aiter_mla_decode_is_silently_wrong,
)

# The combination AITER gets wrong.
BAD = dict(
    q_is_fp8=True,
    kv_is_fp8=True,
    gqa_ratio=16,
    max_qo_len=3,
    persistent=False,
    is_gfx950=True,
)


def test_fires_on_the_broken_combination():
    assert _aiter_mla_decode_is_silently_wrong(**BAD)


@pytest.mark.parametrize("qlen", [3, 4])
def test_fires_at_both_broken_qlens(qlen):
    assert _aiter_mla_decode_is_silently_wrong(**{**BAD, "max_qo_len": qlen})


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"q_is_fp8": False}, "bf16 query selects a different AITER branch"),
        ({"kv_is_fp8": False}, "non-fp8 KV selects a different AITER branch"),
        ({"gqa_ratio": 128}, "gqa_ratio 128 measured clean (cos_diff 5.2e-4)"),
        ({"gqa_ratio": 32}, "gqa_ratio 32 is rejected loudly by AITER already"),
        ({"max_qo_len": 1}, "qlen 1 measured clean (cos_diff 5.1e-4)"),
        ({"max_qo_len": 2}, "qlen 2 measured clean (cos_diff 1.0e-3)"),
        ({"max_qo_len": 5}, "qlen > 4 takes the persistent-only branch"),
        ({"persistent": True}, "persistent is remapped to a supported kernel"),
        ({"is_gfx950": False}, "the remap chain is gfx950-only"),
    ],
)
def test_does_not_fire_when_one_variable_changes(change, reason):
    """Every condition must be necessary, or the guard refuses working configs."""
    assert not _aiter_mla_decode_is_silently_wrong(**{**BAD, **change}), reason


def test_guard_is_reachable_from_the_decode_path():
    """The call site passes exactly the keywords the predicate declares.

    Without this, renaming a field would leave the guard silently uncalled --
    which is the failure mode the guard exists to prevent.
    """
    import inspect

    from vllm.v1.attention.backends.mla import rocm_aiter_mla as mod

    src = inspect.getsource(mod.AiterMLAImpl._forward_mqa)
    assert "_aiter_mla_decode_is_silently_wrong(" in src, (
        "the guard is defined but never called from the decode path"
    )
    params = set(inspect.signature(_aiter_mla_decode_is_silently_wrong).parameters)
    for kw in params:
        assert f"{kw}=" in src, f"call site does not pass {kw!r}"
