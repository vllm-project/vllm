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
import torch

from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    _SAME_WEIGHT_LAYOUT_PAIRS,
    Fp8MoeBackend,
    assert_same_fp8_weight_layout,
    rebuild_fp8_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
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
_ROUNDUP = (
    "vllm.model_executor.layers.fused_moe.all2all_utils.maybe_roundup_layer_hidden_size"
)
_UNQUANTIZED_SELECT = (
    "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
    ".select_unquantized_moe_backend"
)


def _moe_config(batched: bool = False, hidden: int = 4096) -> SimpleNamespace:
    """A stand-in FusedMoEConfig holding only what the rebuilds read.

    No all2all backend flag is set, so ``maybe_roundup_layer_hidden_size``
    returns the hidden size unchanged and the padding check passes; tests of
    that check patch the rounding instead.
    """
    return SimpleNamespace(
        hidden_dim=hidden,
        hidden_dim_unpadded=hidden,
        in_dtype=torch.bfloat16,
        moe_parallel_config=SimpleNamespace(
            all2all_backend="deepep_high_throughput",
            use_batched_activation_format=batched,
            use_deepep_ht_kernels=False,
            use_deepep_ll_kernels=False,
            use_deepep_v2_kernels=False,
            use_nixl_ep_kernels=False,
        ),
    )


def _experts_cls(name: str, activation_format):
    """A stand-in experts class reporting one activation format."""
    return type(
        name, (), {"activation_format": staticmethod(lambda: activation_format)}
    )


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
    # "deepgemm" is not a value --moe-backend accepts; the fix it names must
    # be one the operator can type.
    assert "Pin moe_backend to deep_gemm or triton" in message


def _method(backend: Fp8MoeBackend) -> SimpleNamespace:
    """A stand-in for an FP8 MoE method, holding only what the rebuild reads."""
    return SimpleNamespace(
        moe=_moe_config(),
        fp8_backend=backend,
        experts_cls=object,
        moe_quant_config=Mock(),
        moe_kernel=Mock(),
    )


def _rebuild(method, dry_run: bool = False, **kwargs) -> None:
    rebuild_fp8_moe_kernel(
        method,
        Mock(),
        weight_key=None,
        activation_key=None,
        dry_run=dry_run,
        **kwargs,
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
        _rebuild(method, dry_run=True)

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
        _rebuild(method, dry_run=True)

    assert method.fp8_backend is Fp8MoeBackend.DEEPGEMM


def test_dry_run_runs_the_same_checks_as_the_real_rebuild():
    """A dry run that passes must not be followed by a rebuild that raises."""
    method = _method(Fp8MoeBackend.DEEPGEMM)
    method.moe_quant_config = None

    with (
        patch(_SELECT, return_value=(Fp8MoeBackend.BATCHED_DEEPGEMM, object)),
        pytest.raises(ValueError, match="before the weights have been loaded"),
    ):
        _rebuild(method, dry_run=True)


def test_dry_run_refuses_a_backend_with_no_experts_class():
    """select_fp8_moe_backend returns (NONE, None) off CUDA/ROCm."""
    method = _method(Fp8MoeBackend.NONE)

    with (
        patch(_SELECT, return_value=(Fp8MoeBackend.NONE, None)),
        pytest.raises(ValueError, match="selects no experts class"),
    ):
        _rebuild(method, dry_run=True)


def test_dry_run_refuses_a_backend_that_pads_the_hidden_size_differently():
    """The weights were allocated for the padding the load-time backend chose.

    A backend that rounds the unpadded hidden size differently builds a kernel
    that fails on the first forward, not at the rebuild, so the dry run has to
    compare the two.
    """
    method = _method(Fp8MoeBackend.DEEPGEMM)
    method.moe = _moe_config(hidden=3584)

    with (
        patch(_SELECT, return_value=(Fp8MoeBackend.BATCHED_DEEPGEMM, object)),
        patch(_ROUNDUP, return_value=4096),
        pytest.raises(ValueError, match="pads hidden size 3584 to 4096"),
    ):
        _rebuild(method, dry_run=True)

    assert method.fp8_backend is Fp8MoeBackend.DEEPGEMM


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


def test_rebuild_leaves_the_method_untouched_when_the_kernel_build_fails():
    """A failure inside the kernel build must not half-switch the method.

    The kernel is built into a local and the three fields are assigned only
    once it succeeds. Assigning first would leave fp8_backend and experts_cls
    naming a kernel that was never built while moe_kernel still runs the
    outgoing one -- a layer that reports the new role while serving the old,
    which the caller cannot detect and has no way to roll back.
    """
    method = _method(Fp8MoeBackend.DEEPGEMM)
    before = (method.fp8_backend, method.experts_cls, method.moe_kernel)

    with (
        patch(_SELECT, return_value=(Fp8MoeBackend.BATCHED_DEEPGEMM, object)),
        patch(_MAKE_KERNEL, side_effect=RuntimeError("no buffer")),
        pytest.raises(RuntimeError, match="no buffer"),
    ):
        _rebuild(method)

    assert (method.fp8_backend, method.experts_cls, method.moe_kernel) == before


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
        _rebuild(method, dry_run=True, allow_vllm_cutlass=True)

    assert select.call_args.kwargs["allow_vllm_cutlass"] is True


def test_fp8_moe_method_rebuilds_with_the_keys_it_loaded_with():
    """Fp8MoEMethod.rebuild_moe_kernel forwards the stashed selection inputs.

    The keys and the cutlass rule are stashed at load precisely so the
    re-selection runs under the same inputs; dropping the stash would only
    show up as an AttributeError on the first switch of a real engine.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

    method = object.__new__(Fp8MoEMethod)
    method.moe = _moe_config()
    method.fp8_backend = Fp8MoeBackend.DEEPGEMM
    method.experts_cls = object
    method.moe_quant_config = Mock()
    method.moe_kernel = Mock()
    method.weight_key = "weight-key"
    method.activation_key = "activation-key"
    method.allow_vllm_cutlass = False

    with patch(
        _SELECT, return_value=(Fp8MoeBackend.BATCHED_DEEPGEMM, object)
    ) as select:
        method.rebuild_moe_kernel(Mock(), dry_run=True)

    kwargs = select.call_args.kwargs
    assert kwargs["weight_key"] == "weight-key"
    assert kwargs["activation_key"] == "activation-key"
    assert kwargs["allow_vllm_cutlass"] is False


# ---------------------------------------------------------------- unquantized


def _unquantized_method(batched: bool) -> UnquantizedFusedMoEMethod:
    """A stand-in unquantized method whose config wants one activation format."""
    method = object.__new__(UnquantizedFusedMoEMethod)
    method.moe = _moe_config(batched=batched)
    method.unquantized_backend = UnquantizedMoeBackend.TRITON
    method.experts_cls = object
    method.moe_quant_config = "outgoing-config"
    method.moe_kernel = "outgoing"
    return method


_STANDARD = _experts_cls("StandardExperts", FusedMoEActivationFormat.Standard)
_BATCHED = _experts_cls("BatchedExperts", FusedMoEActivationFormat.BatchedExperts)


@pytest.mark.parametrize(
    ("old", "new"),
    list(itertools.product(UnquantizedMoeBackend, UnquantizedMoeBackend)),
    ids=lambda backend: backend.value,
)
def test_unquantized_layout_matrix(old, new):
    """Only same-backend and TRITON <-> BATCHED_TRITON may switch in place.

    The FlashInfer and AITER backends reshape weights at load, so the
    allow-list is the Triton pair alone; every other pair must be refused
    before anything is assigned, and a backend added later is refused by
    default. CPU is refused even against itself: its experts prepack the
    weights in a step a rebuild never runs.
    """
    method = _unquantized_method(batched=False)
    method.unquantized_backend = old
    switchable = {UnquantizedMoeBackend.TRITON, UnquantizedMoeBackend.BATCHED_TRITON}

    with patch(_UNQUANTIZED_SELECT, return_value=(new, _STANDARD)):
        if old is not new and not {old, new} <= switchable:
            with pytest.raises(ValueError, match="do not share a weight layout"):
                method.rebuild_moe_kernel(Mock(), dry_run=True)
        elif new is UnquantizedMoeBackend.CPU:
            with pytest.raises(ValueError, match="on CPU"):
                method.rebuild_moe_kernel(Mock(), dry_run=True)
        else:
            method.rebuild_moe_kernel(Mock(), dry_run=True)

    assert method.unquantized_backend is old
    assert method.experts_cls is object


def test_unquantized_dry_run_refuses_a_class_of_the_wrong_activation_format():
    """The oracle's LoRA branch returns TritonExperts whatever the format.

    Under a batched backend that class would pass the layout check (TRITON
    is the incumbent) and then trip the kernel's activation-format assert
    inside the real rebuild. The dry run has to refuse it first.
    """
    method = _unquantized_method(batched=True)

    with (
        patch(
            _UNQUANTIZED_SELECT, return_value=(UnquantizedMoeBackend.TRITON, _STANDARD)
        ),
        pytest.raises(ValueError, match="Standard activation format"),
    ):
        method.rebuild_moe_kernel(Mock(), dry_run=True)

    assert method.unquantized_backend is UnquantizedMoeBackend.TRITON
    assert method.experts_cls is object


def test_unquantized_dry_run_checks_the_prepare_finalize_format_not_the_pin():
    """A batched_triton pin keeps the oracle on BatchedTritonExperts.

    On CUDA the pin does not change the prepare/finalize, whose format follows
    the all2all backend alone, so a pinned engine moving to a standard backend
    must be refused here even though the oracle re-selects happily.
    """
    method = _unquantized_method(batched=False)
    method.unquantized_backend = UnquantizedMoeBackend.BATCHED_TRITON

    with (
        patch(
            _UNQUANTIZED_SELECT,
            return_value=(UnquantizedMoeBackend.BATCHED_TRITON, _BATCHED),
        ),
        pytest.raises(ValueError, match="hands it Standard"),
    ):
        method.rebuild_moe_kernel(Mock(), dry_run=True)


def test_unquantized_dry_run_refuses_a_backend_with_no_experts_class():
    """TPU and OOT select no class; the real rebuild would assert on None."""
    method = _unquantized_method(batched=False)

    with (
        patch(_UNQUANTIZED_SELECT, return_value=(UnquantizedMoeBackend.TRITON, None)),
        pytest.raises(ValueError, match="selects no experts class"),
    ):
        method.rebuild_moe_kernel(Mock(), dry_run=True)


def test_unquantized_dry_run_refuses_a_backend_that_pads_the_hidden_size_differently():
    method = _unquantized_method(batched=True)
    method.moe = _moe_config(batched=True, hidden=3584)

    with (
        patch(
            _UNQUANTIZED_SELECT,
            return_value=(UnquantizedMoeBackend.BATCHED_TRITON, _BATCHED),
        ),
        patch(_ROUNDUP, return_value=4096),
        pytest.raises(ValueError, match="pads hidden size 3584 to 4096"),
    ):
        method.rebuild_moe_kernel(Mock(), dry_run=True)


def test_unquantized_rebuild_assigns_and_builds_around_the_new_backend():
    """The real rebuild moves all four fields and builds from the new pair."""
    method = _unquantized_method(batched=True)
    layer = Mock()

    with (
        patch(
            _UNQUANTIZED_SELECT,
            return_value=(UnquantizedMoeBackend.BATCHED_TRITON, _BATCHED),
        ),
        patch.object(
            UnquantizedFusedMoEMethod,
            "_build_moe_kernel",
            return_value=("rebuilt-config", "rebuilt"),
        ) as build,
    ):
        method.rebuild_moe_kernel(layer)

    assert method.unquantized_backend is UnquantizedMoeBackend.BATCHED_TRITON
    assert method.experts_cls is _BATCHED
    assert method.moe_quant_config == "rebuilt-config"
    assert method.moe_kernel == "rebuilt"
    build.assert_called_once_with(layer, UnquantizedMoeBackend.BATCHED_TRITON, _BATCHED)


def test_unquantized_rebuild_leaves_the_method_untouched_when_the_kernel_build_fails():
    """Built into locals and assigned last, like the fp8 helper."""
    method = _unquantized_method(batched=True)
    before = (
        method.unquantized_backend,
        method.experts_cls,
        method.moe_quant_config,
        method.moe_kernel,
    )

    with (
        patch(
            _UNQUANTIZED_SELECT,
            return_value=(UnquantizedMoeBackend.BATCHED_TRITON, _BATCHED),
        ),
        patch.object(
            UnquantizedFusedMoEMethod,
            "_build_moe_kernel",
            side_effect=RuntimeError("no buffer"),
        ),
        pytest.raises(RuntimeError, match="no buffer"),
    ):
        method.rebuild_moe_kernel(Mock())

    assert (
        method.unquantized_backend,
        method.experts_cls,
        method.moe_quant_config,
        method.moe_kernel,
    ) == before
