# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rebuilding a fused-MoE kernel around weights already on the device.

GPU-free: ``assert_same_fp8_weight_layout`` is a pure function over two
``Fp8MoeBackend`` members, and the ``dry_run`` contract is exercised with a
hand-built method object, so neither needs a device, a real layer, or a
loaded checkpoint.
"""

import itertools
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    _SAME_WEIGHT_LAYOUT_PAIRS,
    Fp8MoeBackend,
    assert_same_fp8_weight_layout,
    rebuild_fp8_moe_kernel,
)

# Spelled out rather than derived from _SAME_WEIGHT_LAYOUT_PAIRS: a test that
# recomputes the rule from the rule under test cannot catch the rule changing.
# Widening the allow-list must fail this test until someone edits it here too.
_ALLOWED_PAIRS = [
    (Fp8MoeBackend.DEEPGEMM, Fp8MoeBackend.BATCHED_DEEPGEMM),
    (Fp8MoeBackend.TRITON, Fp8MoeBackend.BATCHED_TRITON),
    (Fp8MoeBackend.VLLM_CUTLASS, Fp8MoeBackend.BATCHED_VLLM_CUTLASS),
]
_ALLOWED_ORDERED = frozenset(
    itertools.chain.from_iterable(((a, b), (b, a)) for a, b in _ALLOWED_PAIRS)
)

_SELECT = "vllm.model_executor.layers.fused_moe.oracle.fp8.select_fp8_moe_backend"
_MAKE_KERNEL = "vllm.model_executor.layers.fused_moe.oracle.fp8.make_fp8_moe_kernel"


def test_same_weight_layout_pairs_match_the_documented_allow_list():
    """The oracle's tuple holds exactly the three documented pairs."""
    assert set(_SAME_WEIGHT_LAYOUT_PAIRS) == {
        frozenset(pair) for pair in _ALLOWED_PAIRS
    }


@pytest.mark.parametrize(
    ("old", "new"),
    list(itertools.product(Fp8MoeBackend, Fp8MoeBackend)),
    ids=lambda backend: backend.value,
)
def test_assert_same_fp8_weight_layout_matrix(old, new):
    """Only same-backend and the three layout-sharing pairs may switch.

    Sweeps the whole ordered cross-product, so a new Fp8MoeBackend member is
    exercised the moment it is added -- an unknown backend must be refused by
    default rather than assumed compatible. TRITON -> VLLM_CUTLASS must be
    refused in particular, even though both fall through
    convert_to_fp8_moe_kernel_format on the same branch: the allow-list is
    deliberately narrower than the converter.
    """
    if old is new or (old, new) in _ALLOWED_ORDERED:
        # Symmetric by construction (frozenset); asserted in both directions
        # by the cross-product rather than assumed.
        assert assert_same_fp8_weight_layout(old, new) is None
        return

    with pytest.raises(ValueError) as exc_info:
        assert_same_fp8_weight_layout(old, new)

    message = str(exc_info.value)
    # The message is the operator's only signal, so it must name both ends of
    # the refused switch and the way out of it.
    assert f"{old.value} -> {new.value}" in message
    assert "do not share a weight layout" in message
    assert "Pin moe_backend" in message


def _method(backend: Fp8MoeBackend) -> SimpleNamespace:
    """A stand-in for an FP8 MoE method, holding only what the rebuild reads."""
    return SimpleNamespace(
        moe=Mock(),
        fp8_backend=backend,
        experts_cls=object,
        moe_quant_config=Mock(),
        moe_kernel=Mock(),
    )


def test_dry_run_assigns_nothing():
    """A dry run selects and validates, but must not mutate the method."""
    method = _method(Fp8MoeBackend.DEEPGEMM)
    before = (method.fp8_backend, method.experts_cls, method.moe_kernel)

    with (
        patch(
            _SELECT, return_value=(Fp8MoeBackend.BATCHED_DEEPGEMM, type("New", (), {}))
        ),
        patch(_MAKE_KERNEL) as make_kernel,
    ):
        rebuild_fp8_moe_kernel(
            method,
            Mock(),
            weight_key=None,
            activation_key=None,
            dry_run=True,
        )

    assert (method.fp8_backend, method.experts_cls, method.moe_kernel) == before
    make_kernel.assert_not_called()


def test_dry_run_still_refuses_an_incompatible_layout():
    """The refusal has to happen on the dry run, or it is worthless.

    A caller uses dry_run precisely to reject an impossible change before it
    has mutated anything.
    """
    method = _method(Fp8MoeBackend.DEEPGEMM)

    with (
        patch(_SELECT, return_value=(Fp8MoeBackend.TRITON, object)),
        pytest.raises(ValueError, match="do not share a weight layout"),
    ):
        rebuild_fp8_moe_kernel(
            method,
            Mock(),
            weight_key=None,
            activation_key=None,
            dry_run=True,
        )

    assert method.fp8_backend is Fp8MoeBackend.DEEPGEMM


def test_dry_run_runs_the_same_checks_as_the_real_rebuild():
    """A dry run that passes must not be followed by a rebuild that asserts."""
    method = _method(Fp8MoeBackend.DEEPGEMM)
    method.moe_quant_config = None

    with (
        patch(_SELECT, return_value=(Fp8MoeBackend.BATCHED_DEEPGEMM, object)),
        pytest.raises(AssertionError),
    ):
        rebuild_fp8_moe_kernel(
            method,
            Mock(),
            weight_key=None,
            activation_key=None,
            dry_run=True,
        )


def test_rebuild_assigns_and_builds_around_the_new_backend():
    """The real rebuild moves all three fields and builds from the new pair."""
    method = _method(Fp8MoeBackend.DEEPGEMM)
    new_cls = type("NewExperts", (), {})
    layer = Mock()

    with (
        patch(_SELECT, return_value=(Fp8MoeBackend.BATCHED_DEEPGEMM, new_cls)),
        patch(_MAKE_KERNEL, return_value="rebuilt") as make_kernel,
    ):
        rebuild_fp8_moe_kernel(
            method,
            layer,
            weight_key=None,
            activation_key=None,
        )

    assert method.fp8_backend is Fp8MoeBackend.BATCHED_DEEPGEMM
    assert method.experts_cls is new_cls
    assert method.moe_kernel == "rebuilt"
    kwargs = make_kernel.call_args.kwargs
    assert kwargs["fp8_backend"] is Fp8MoeBackend.BATCHED_DEEPGEMM
    assert kwargs["experts_cls"] is new_cls
    assert kwargs["routing_tables"] is layer._expert_routing_tables.return_value


def test_allow_vllm_cutlass_is_forwarded_to_the_selection():
    """Whatever rule the caller selects under must reach the oracle.

    A method that loaded with allow_vllm_cutlass=True and rebuilds without it
    re-selects under a stricter rule, lands on a different backend, and then
    refuses a switch the allow-list explicitly permits.
    """
    method = _method(Fp8MoeBackend.VLLM_CUTLASS)

    with patch(
        _SELECT, return_value=(Fp8MoeBackend.BATCHED_VLLM_CUTLASS, object)
    ) as select:
        rebuild_fp8_moe_kernel(
            method,
            Mock(),
            weight_key=None,
            activation_key=None,
            allow_vllm_cutlass=True,
            dry_run=True,
        )

    assert select.call_args.kwargs["allow_vllm_cutlass"] is True
