# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enum-path tests for `prioritize_humming` (CPU-only).

`prioritize_humming` reorders two different kinds of entry: kernel *classes*
(`_POSSIBLE_KERNELS`, `_POSSIBLE_NVFP4_KERNELS`) and MoE backend *enum members*
(the `*MoeBackend` priority lists in `vllm/model_executor/layers/fused_moe/
oracle/`). Only the class path has coverage today, in
`test_scaled_mm_kernel_selection.py::test_humming_priority_only_reorders_humming_and_marlin`;
the `isinstance(kernel, Enum)` branch has none, so a member rename that stops
matching "humming"/"marlin" would turn the reorder into a silent no-op and leave
SM90 MoE on Marlin with no test failing.

Run `pytest tests/kernels/quantization/test_humming_priority_enum.py`.
"""

import pytest

from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import WNA16MoEBackend
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.quantization.utils.humming import prioritize_humming

pytestmark = pytest.mark.cpu_test

# A capability where Humming outranks Marlin, and one where it does not.
_PREFERRED_CAPABILITY = 90
_UNPREFERRED_CAPABILITY = 80

# One order per MoE backend enum that has both a Humming and a Marlin member,
# with Humming behind Marlin (the case the reorder is for). These are tuples,
# and every test builds its own list from one: a list in a `pytest.param` is a
# single object shared by every test parametrized on it, so a function that
# reordered its argument in place would corrupt the next test's input instead
# of failing this one.
_ENUM_LISTS = [
    pytest.param(
        (
            WNA16MoEBackend.FLASHINFER_TRTLLM,
            WNA16MoEBackend.MARLIN,
            WNA16MoEBackend.TRITON,
            WNA16MoEBackend.HUMMING,
        ),
        id="wna16",
    ),
    pytest.param(
        (
            NvFp4MoeBackend.VLLM_CUTLASS,
            NvFp4MoeBackend.MARLIN,
            NvFp4MoeBackend.HUMMING,
        ),
        id="nvfp4",
    ),
    pytest.param(
        (
            Mxfp4MoeBackend.TRITON,
            Mxfp4MoeBackend.MARLIN,
            Mxfp4MoeBackend.HUMMING,
        ),
        id="mxfp4",
    ),
]


@pytest.mark.parametrize("backends", _ENUM_LISTS)
def test_enum_members_are_matched_by_name(backends):
    """Humming must land directly ahead of Marlin for enum members too."""
    reordered = prioritize_humming(list(backends), _PREFERRED_CAPABILITY)

    names = [backend.name for backend in reordered]
    humming = names.index("HUMMING")
    assert names[humming + 1].endswith("MARLIN")


@pytest.mark.parametrize("backends", _ENUM_LISTS)
def test_enum_reorder_preserves_every_other_entry(backends):
    """Only Humming moves, the rest keep their relative order, and the caller's
    list is left alone -- the MoE oracles pass module-level backend lists, so
    reordering one in place would change the priority order for every later
    call in the process."""
    passed_in = list(backends)
    original = list(backends)
    reordered = prioritize_humming(passed_in, _PREFERRED_CAPABILITY)

    assert sorted(reordered, key=lambda b: b.name) == sorted(
        original, key=lambda b: b.name
    )
    assert [b for b in reordered if b.name != "HUMMING"] == [
        b for b in original if b.name != "HUMMING"
    ]
    assert reordered is not passed_in, "the reorder must return a new list"
    assert passed_in == original, "the input list must not be modified"


@pytest.mark.parametrize("backends", _ENUM_LISTS)
def test_enum_order_is_unchanged_where_humming_is_not_preferred(backends):
    assert prioritize_humming(list(backends), _UNPREFERRED_CAPABILITY) == list(backends)


def test_humming_goes_ahead_of_the_first_marlin_member():
    """`WNA16MoEBackend` has two Marlin members; Humming lands ahead of the
    first one, so both Marlin backends stay in their relative order behind it.
    """
    backends = [
        WNA16MoEBackend.MARLIN,
        WNA16MoEBackend.BATCHED_MARLIN,
        WNA16MoEBackend.HUMMING,
    ]

    assert prioritize_humming(backends, _PREFERRED_CAPABILITY) == [
        WNA16MoEBackend.HUMMING,
        WNA16MoEBackend.MARLIN,
        WNA16MoEBackend.BATCHED_MARLIN,
    ]


def test_enum_list_without_a_humming_member_is_unchanged():
    backends = [WNA16MoEBackend.MARLIN, WNA16MoEBackend.TRITON]

    assert prioritize_humming(backends, _PREFERRED_CAPABILITY) == backends
