# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the LoRA-aware FlashInfer TRTLLM MXFP4 experts
(`TrtLlmMxfp4LoRAExperts`) against a float64 reference."""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
    TrtLlmMxfp4LoRAExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    _requested_kernel_cls,
    backend_to_kernel_cls,
    convert_weight_to_mxfp4_moe_kernel_format,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe
from vllm.utils.torch_utils import set_random_seed

# FlashInfer's TRTLLM MoE LoRA kernels only run on SM100-family GPUs.
pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_device_capability_family(100)
        or not has_flashinfer_trtllm_fused_moe(),
        reason="Requires an SM100-family GPU and FlashInfer TRTLLM fused MoE",
    ),
    pytest.mark.skip_global_cleanup,
]

E, N, K, T, TOPK = 16, 256, 512, 33, 4
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def _dequant_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """(E, R, K/2) packed nibbles + (E, R, K/32) e8m0 scales -> (E, R, K) f64."""
    nibbles = torch.stack([weight & 0xF, weight >> 4], dim=-1).flatten(-2).long()
    magnitude = torch.tensor(E2M1, dtype=torch.float64, device=weight.device)
    values = magnitude[nibbles & 7] * torch.where(nibbles >> 3 == 1, -1.0, 1.0)
    block_scale = torch.pow(2.0, scale.double() - 127.0)
    return values * block_scale.repeat_interleave(32, dim=-1)


def _make_experts(has_bias: bool, activation=MoEActivation.SILU):
    w13 = torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.randint(118, 122, (E, 2 * N, K // 32), dtype=torch.uint8, device="cuda")
    s2 = torch.randint(118, 122, (E, K, N // 32), dtype=torch.uint8, device="cuda")
    b13 = torch.randn(E, 2 * N, device="cuda") / 4 if has_bias else None
    b2 = torch.randn(E, K, device="cuda") / 4 if has_bias else None
    k13, k2, ks13, ks2, kb13, kb2 = convert_weight_to_mxfp4_moe_kernel_format(
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        None,
        w13.clone(),
        w2.clone(),
        s13.clone(),
        s2.clone(),
        None if b13 is None else b13.clone(),
        None if b2 is None else b2.clone(),
        _cache_permute_indices={},
    )
    config = make_dummy_moe_config(
        num_experts=E,
        experts_per_token=TOPK,
        hidden_dim=K,
        intermediate_size=N,
        activation=activation,
    )
    quant_config = mxfp4_w4a16_moe_quant_config(ks13, ks2, kb13, kb2)
    experts = TrtLlmMxfp4LoRAExperts(config, quant_config)
    reference = (_dequant_mxfp4(w13, s13), _dequant_mxfp4(w2, s2), b13, b2)
    return experts, (k13, k2), reference


def _make_inputs():
    x = torch.randn(T, K, device="cuda").to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(T)])
    weights = torch.rand(T, TOPK, device="cuda", dtype=torch.float32) + 0.1
    return x, ids.to(torch.int32), weights


def _reference(reference, x, ids, weights, gate_delta=None, up_delta=None):
    """Float64 MoE; the checkpoint w13 layout is [gate | up]."""
    w13, w2, b13, b2 = reference
    ids = ids.long()
    h = torch.einsum("tk,tjnk->tjn", x.double(), w13[ids])
    if b13 is not None:
        h = h + b13.double()[ids]
    gate, up = h[..., :N], h[..., N:]
    if gate_delta is not None:
        gate, up = gate + gate_delta.double(), up + up_delta.double()
    act = torch.nn.functional.silu(gate) * up
    out = torch.einsum("tjn,tjkn->tjk", act, w2[ids])
    if b2 is not None:
        out = out + b2.double()[ids]
    return act, (out * weights.double()[..., None]).sum(1)


def _run_lora_path(experts, kernel_weights, x, ids, weights, delta):
    """`delta` is (T, TOPK, 2I) in the [up | gate] order of apply_w13_lora."""
    lora_context = SimpleNamespace(w13_num_slices=2)
    gemm2, expert_weights, idx_map, act_permuted = experts.invoke_routed_moe(
        hidden_states=x,
        w1=kernel_weights[0],
        w2=kernel_weights[1],
        topk_ids_and_weights=(ids, weights),
        gemm1_lora_delta=experts.prepare_gemm1_lora_delta(delta, lora_context, ids),
        global_num_experts=E,
        a1q_scale=None,
        output=torch.empty(T, K, dtype=torch.bfloat16, device="cuda"),
    )
    assert act_permuted.dtype == torch.bfloat16
    act = experts._unpermute_activation(act_permuted, idx_map, T, TOPK, N)
    output = torch.empty(T, K, dtype=torch.bfloat16, device="cuda")
    no_w2_delta = torch.zeros(T, TOPK, K, dtype=torch.bfloat16, device="cuda")
    experts._finalize_with_w2_lora(
        output, gemm2, expert_weights, idx_map, no_w2_delta, T, TOPK
    )
    return act.view(T, TOPK, N), output


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.double(), expected.double()
    return float((actual - expected).norm() / expected.norm())


@pytest.mark.parametrize("has_bias", [False, True])
@torch.inference_mode()
def test_gemm1_lora_delta_matches_reference(has_bias: bool):
    set_random_seed(0)
    experts, kernel_weights, reference = _make_experts(has_bias)
    x, ids, weights = _make_inputs()
    gate_delta = (torch.randn(T, TOPK, N, device="cuda") / 2).to(torch.bfloat16)
    up_delta = (torch.randn(T, TOPK, N, device="cuda") / 2).to(torch.bfloat16)
    ref_act, ref_out = _reference(reference, x, ids, weights, gate_delta, up_delta)
    _, ref_base_out = _reference(reference, x, ids, weights)
    # The delta has to matter for the layout check below to mean anything.
    assert _rel_err(ref_out, ref_base_out) > 0.3

    delta = torch.cat([up_delta, gate_delta], dim=-1)
    act, out = _run_lora_path(experts, kernel_weights, x, ids, weights, delta)
    assert _rel_err(act, ref_act) < 0.02
    assert _rel_err(out, ref_out) < 0.02

    swapped = torch.cat([gate_delta, up_delta], dim=-1)
    _, out = _run_lora_path(experts, kernel_weights, x, ids, weights, swapped)
    assert _rel_err(out, ref_out) > 0.3


@pytest.mark.parametrize("has_bias", [False, True])
@torch.inference_mode()
def test_zero_delta_matches_base_path(has_bias: bool):
    set_random_seed(0)
    experts, kernel_weights, reference = _make_experts(has_bias)
    x, ids, weights = _make_inputs()
    base = torch.empty(T, K, dtype=torch.bfloat16, device="cuda")
    experts.invoke_routed_moe(
        hidden_states=x,
        w1=kernel_weights[0],
        w2=kernel_weights[1],
        topk_ids_and_weights=(ids, weights),
        gemm1_lora_delta=None,
        global_num_experts=E,
        a1q_scale=None,
        output=base,
    )
    _, ref_out = _reference(reference, x, ids, weights)
    assert _rel_err(base, ref_out) < 0.02

    zero = torch.zeros(T, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
    _, out = _run_lora_path(experts, kernel_weights, x, ids, weights, zero)
    assert _rel_err(out, base) < 0.02


@pytest.mark.parametrize("w13_num_slices", [1, 2])
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.SWIGLUOAI])
@torch.inference_mode()
def test_prepare_gemm1_lora_delta_layout(activation, w13_num_slices: int):
    set_random_seed(0)
    experts, _, _ = _make_experts(False, activation)
    ids = _make_inputs()[1]
    gate = torch.randn(T, TOPK, N, device="cuda").to(torch.bfloat16)
    up = torch.randn(T, TOPK, N, device="cuda").to(torch.bfloat16)
    if w13_num_slices == 2:
        # What apply_w13_lora(swap_w13_slices=True) writes for per-expert LoRA.
        pair, halves = (up, gate), (up, gate)
    else:
        # A fused (3D) adapter follows the checkpoint w13 row order.
        pair, halves = (gate, up), (gate, up)
    if activation == MoEActivation.SWIGLUOAI:
        delta = torch.stack(pair, dim=-1).flatten(-2)
    else:
        delta = torch.cat(halves, dim=-1)

    lora_context = SimpleNamespace(w13_num_slices=w13_num_slices)
    prepared = experts.prepare_gemm1_lora_delta(delta, lora_context, ids)
    assert torch.equal(prepared, torch.cat([up, gate], dim=-1))


def test_lora_experts_are_opt_in_for_bf16_activations():
    config = make_dummy_moe_config(num_experts=E, hidden_dim=K, intermediate_size=N)
    bf16 = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16
    mxfp8 = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8
    assert _requested_kernel_cls(bf16, config) == backend_to_kernel_cls(bf16)
    assert _requested_kernel_cls(mxfp8, config) == backend_to_kernel_cls(mxfp8)

    config.is_lora_enabled = True
    assert _requested_kernel_cls(bf16, config)[0] is TrtLlmMxfp4LoRAExperts
    with pytest.raises(ValueError, match="MXFP8 activations"):
        _requested_kernel_cls(mxfp8, config)
    marlin = Mxfp4MoeBackend.MARLIN
    assert _requested_kernel_cls(marlin, config) == backend_to_kernel_cls(marlin)
