# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extra coverage for NVFP4 fused-MoE w13 global-scale reconciliation.

Run `pytest tests/quantization/test_nvfp4_utils.py`.

`tests/quantization/test_modelopt.py` already unit-tests
``reconcile_nvfp4_moe_w13_scales`` on hand-picked scale pairs. This file adds
what that coverage does not reach:

* the effective-dequantization-scale invariant across the whole mismatch range
  reported in #54974 (gate/up ratios up to 10x, in both directions, not powers
  of two);
* the two safety properties the rescale relies on -- block scales only ever
  shrink, and the caller's tensors are never written through;
* the three production call sites (ModelOpt, Quark, compressed-tensors). Each
  must pass the reconciled block scales and the shared per-expert scale to
  ``convert_to_nvfp4_moe_kernel_format``, and each must leave
  ``NvFp4MoeBackend.HUMMING`` on the original gate-scale path, because
  Humming's converter re-reads ``layer.w13_weight_scale*`` and folds each
  half's ``scale_2`` itself.

CPU only: no CUDA device and no real kernel is touched.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.quantization import modelopt as modelopt_module
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_w4a4_nvfp4 as ct_nvfp4_module,
)
from vllm.model_executor.layers.quantization.quark import quark_moe as quark_module
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    reconcile_nvfp4_moe_w13_scales,
)

# float8_e4m3fn keeps 3 mantissa bits, so re-rounding a block scale costs at
# most half an ulp, i.e. 2**-4 relative. Every tolerance below is that bound;
# none of them is a fudge factor.
E4M3_HALF_ULP = 2**-4


def _mismatched_w13(
    num_experts: int,
    half_size: int,
    num_blocks: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a fused w13 whose gate and up global scales disagree per expert.

    Block scales stay in ``[0.25, 1.0]`` and rescale factors stay at or above
    0.1, so every reconciled block scale lands at or above ``2**-6`` and stays
    in the e4m3 *normal* range. That keeps the invariant test bounded by one
    half-ulp instead of by subnormal spacing.
    """
    g = torch.Generator().manual_seed(seed)
    gs_gate = torch.empty(num_experts).uniform_(0.05, 2.0, generator=g)
    ratio = torch.empty(num_experts).uniform_(0.1, 10.0, generator=g)
    w13_scale_2 = torch.stack([gs_gate, gs_gate * ratio], dim=1)
    blocks = torch.empty(num_experts, 2 * half_size, num_blocks).uniform_(
        0.25, 1.0, generator=g
    )
    return blocks.to(torch.float8_e4m3fn), w13_scale_2


def test_reconcile_preserves_each_half_effective_scale_over_reported_range():
    """``block_scale * scale_2`` survives the move onto one shared scale.

    This is the numerical claim of the fix: the up half keeps meaning
    ``block * gs_up`` rather than silently becoming ``block * gs_gate``. It is
    checked here over 8 experts whose gate/up ratios sweep 0.1x to 10x -- the
    range #54974 measured on a real checkpoint -- rather than on a few
    hand-chosen power-of-two pairs.
    """
    num_experts, half, num_blocks = 8, 6, 3
    w13_scale, w13_scale_2 = _mismatched_w13(num_experts, half, num_blocks, seed=0)
    before = w13_scale.float()
    expected_gate = before[:, :half] * w13_scale_2[:, 0].view(-1, 1, 1)
    expected_up = before[:, half:] * w13_scale_2[:, 1].view(-1, 1, 1)

    reconciled, shared = reconcile_nvfp4_moe_w13_scales(w13_scale, w13_scale_2)

    after = reconciled.float() * shared.view(-1, 1, 1)
    torch.testing.assert_close(
        after[:, :half], expected_gate, rtol=E4M3_HALF_ULP, atol=0
    )
    torch.testing.assert_close(after[:, half:], expected_up, rtol=E4M3_HALF_ULP, atol=0)


def test_reconcile_never_grows_a_block_scale():
    """Guard: block scales only shrink, so e4m3 cannot overflow.

    The shared scale is the per-expert *maximum*, which makes every rescale
    factor ``<= 1``. This holds before the fix as well (nothing was rescaled
    at all); it is here so that a later change to a smaller shared scale --
    ``torch.maximum`` -> ``torch.minimum`` fails this test -- cannot quietly
    push block scales towards e4m3's 448 ceiling.
    """
    w13_scale, w13_scale_2 = _mismatched_w13(8, 6, 3, seed=1)
    before = w13_scale.float()

    reconciled, _ = reconcile_nvfp4_moe_w13_scales(w13_scale, w13_scale_2)

    assert torch.all(reconciled.float() <= before)


def test_reconcile_does_not_write_through_to_its_arguments():
    """Guard: reconciliation is out of place.

    All three call sites hand ``layer.w13_weight_scale`` straight in and pass
    the *returned* tensor on, while Humming re-reads ``layer`` itself. If
    reconciliation ever started writing through to its argument, Humming would
    see rescaled block scales underneath the ``scale_2`` folding its own
    converter performs, double-applying the correction on the very backend the
    call sites deliberately skip.
    """
    w13_scale, w13_scale_2 = _mismatched_w13(4, 4, 2, seed=2)
    scale_before = w13_scale.clone()
    scale_2_before = w13_scale_2.clone()

    reconcile_nvfp4_moe_w13_scales(w13_scale, w13_scale_2)

    assert torch.equal(w13_scale.float(), scale_before.float())
    assert torch.equal(w13_scale_2, scale_2_before)


# Expert 0 has gate (1.0) != up (2.0); expert 1 is already tied. Every block
# scale starts at 1.0, so the reconciled tensor is exact and needs no
# tolerance: expert 0's gate half is halved onto the shared max of 2.0 and
# nothing else moves.
_MISMATCHED_SCALE_2 = torch.tensor([[1.0, 2.0], [1.0, 1.0]])
_EXPECTED_RECONCILED = torch.tensor(
    [
        [[0.5, 0.5], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    ]
)
_EXPECTED_SHARED_SCALE_2 = torch.tensor([2.0, 1.0])
_EXPECTED_GATE_ONLY_SCALE_2 = torch.tensor([1.0, 1.0])


def _make_modelopt_layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.w13_weight = torch.zeros(1)
    layer.w13_weight_scale = torch.ones((2, 4, 2), dtype=torch.float8_e4m3fn)
    layer.w13_weight_scale_2 = _MISMATCHED_SCALE_2.clone()
    layer.w13_input_scale = torch.zeros(1)
    layer.w2_weight = torch.zeros(1)
    layer.w2_weight_scale = torch.zeros(1)
    layer.w2_weight_scale_2 = torch.zeros(1)
    layer.w2_input_scale = torch.zeros(1)
    layer._expert_routing_tables = Mock(return_value=Mock())
    return layer


def _make_quark_layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.w13_weight = torch.zeros(1)
    layer.w13_weight_scale = torch.ones((2, 4, 2), dtype=torch.float8_e4m3fn)
    layer.w13_weight_scale_2 = _MISMATCHED_SCALE_2.clone()
    layer.w13_input_scale_2 = torch.zeros(1)
    layer.w2_weight = torch.zeros(1)
    layer.w2_weight_scale = torch.zeros(1)
    layer.w2_weight_scale_2 = torch.zeros(1)
    layer.w2_input_scale_2 = torch.zeros(1)
    layer._expert_routing_tables = Mock(return_value=Mock())
    return layer


def _make_compressed_tensors_layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.w13_weight_packed = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    layer.w2_weight_packed = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    layer.w13_weight_scale = torch.ones((2, 4, 2), dtype=torch.float8_e4m3fn)
    # compressed-tensors stores global scales as *divisors*, so the shared
    # per-expert scale the kernel needs is 1 / min(divisors), not 1 / max.
    layer.w13_weight_global_scale = 1.0 / _MISMATCHED_SCALE_2
    layer.w13_input_global_scale = torch.ones(1)
    layer.w2_weight_scale = torch.zeros(1)
    layer.w2_weight_global_scale = torch.ones(1)
    layer.w2_input_global_scale = torch.ones(1)
    layer._expert_routing_tables = Mock(return_value=Mock())
    return layer


_CALL_SITES = {
    "modelopt": (
        modelopt_module,
        "ModelOptNvFp4FusedMoE",
        _make_modelopt_layer,
    ),
    "quark": (
        quark_module,
        "QuarkNvfp4MoEMethod",
        _make_quark_layer,
    ),
    "compressed_tensors": (
        ct_nvfp4_module,
        "CompressedTensorsW4A4Nvfp4MoEMethod",
        _make_compressed_tensors_layer,
    ),
}


def _run_process_weights(call_site: str, backend: NvFp4MoeBackend) -> dict[str, Any]:
    """Drive ``process_weights_after_loading`` and capture the converter call."""
    module, cls_name, make_layer = _CALL_SITES[call_site]
    method_cls = getattr(module, cls_name)
    layer = make_layer()

    method = method_cls.__new__(method_cls)
    method.moe = Mock(is_act_and_mul=True)
    method.nvfp4_backend = backend
    method.experts_cls = Mock()
    method.use_a16 = False
    if call_site == "modelopt":
        # ModelOpt's loader consults per_token_activation (upstream added
        # per-token NVFP4 activation handling); the other methods do not.
        method.per_token_activation = False

    convert = Mock(return_value=tuple(torch.zeros(1) for _ in range(8)))
    with (
        patch.object(module, "convert_to_nvfp4_moe_kernel_format", convert),
        patch.object(
            method_cls, "get_fused_moe_quant_config", Mock(return_value=Mock())
        ),
        patch.object(module, "make_nvfp4_moe_kernel", Mock(return_value=Mock())),
    ):
        method.process_weights_after_loading(layer)

    return dict(convert.call_args.kwargs)


@pytest.mark.parametrize("call_site", sorted(_CALL_SITES))
def test_nvfp4_moe_call_site_passes_reconciled_w13_scales(call_site):
    """Every NVFP4 MoE method must hand the kernel the reconciled scales.

    ``reconcile_nvfp4_moe_w13_scales`` is only useful if its result actually
    reaches ``convert_to_nvfp4_moe_kernel_format``; each of the three methods
    computes it separately, so each is pinned separately here. For
    compressed-tensors this also pins the divisor-to-multiplier inversion:
    global scales are stored as divisors there, so the shared scale must come
    out as ``1 / min(divisors)``.
    """
    kwargs = _run_process_weights(call_site, NvFp4MoeBackend.VLLM_CUTLASS)

    torch.testing.assert_close(kwargs["w13_scale"].float(), _EXPECTED_RECONCILED)
    torch.testing.assert_close(kwargs["w13_scale_2"], _EXPECTED_SHARED_SCALE_2)


@pytest.mark.parametrize("call_site", sorted(_CALL_SITES))
def test_nvfp4_moe_call_site_leaves_humming_untouched(call_site):
    """Guard: Humming keeps the pre-existing gate-scale-only behaviour.

    ``convert_to_nvfp4_moe_kernel_format`` ignores the ``w13_scale`` /
    ``w13_scale_2`` it is given on the Humming path -- it calls
    ``convert_to_humming_moe_kernel_format(layer, ...)`` and then re-reads the
    scales off ``layer``, folding each half's own ``scale_2`` into bf16 block
    scales. Reconciling first would therefore double-apply the correction.

    This assertion holds both before and after the fix, so it is a guard
    against over-correction rather than a regression test for the bug.
    """
    kwargs = _run_process_weights(call_site, NvFp4MoeBackend.HUMMING)

    torch.testing.assert_close(
        kwargs["w13_scale"].float(), torch.ones((2, 4, 2), dtype=torch.float32)
    )
    torch.testing.assert_close(kwargs["w13_scale_2"], _EXPECTED_GATE_ONLY_SCALE_2)
