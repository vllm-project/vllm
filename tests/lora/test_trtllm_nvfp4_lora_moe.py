# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
    TrtLlmNvFp4LoRAExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    quantize_nvfp4_per_token_input,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4DynamicToken,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe
from vllm.utils.torch_utils import set_random_seed

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


def _dequant_nvfp4(packed, block_scale, global_scale):
    nibbles = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2).long()
    magnitude = torch.tensor(E2M1, dtype=torch.float64, device=packed.device)
    values = magnitude[nibbles & 7] * torch.where(nibbles >> 3 == 1, -1.0, 1.0)
    scale = block_scale.view(torch.float8_e4m3fn).double()
    scale = scale.reshape(*values.shape[:-1], -1).repeat_interleave(16, dim=-1)
    return values * scale * global_scale


def _make_experts():
    w13 = torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device="cuda")
    s13 = (torch.rand(E, 2 * N, K // 16, device="cuda") + 0.5).to(torch.float8_e4m3fn)
    s2 = (torch.rand(E, K, N // 16, device="cuda") + 0.5).to(torch.float8_e4m3fn)
    gs13 = (torch.rand(E, device="cuda") + 0.5) / 64
    gs2 = (torch.rand(E, device="cuda") + 0.5) / 64
    a13 = torch.ones(E, device="cuda")
    a2 = torch.ones(E, device="cuda")

    config = make_dummy_moe_config(
        num_experts=E, experts_per_token=TOPK, hidden_dim=K, intermediate_size=N
    )
    layer = SimpleNamespace(activation=MoEActivation.SILU, moe_config=config)
    k13, ks13, kgs13, ka13, k2, ks2, kgs2, ka2 = convert_to_nvfp4_moe_kernel_format(
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        layer,
        w13.clone(),
        s13.clone(),
        gs13.clone(),
        a13.clone(),
        w2.clone(),
        s2.clone(),
        gs2.clone(),
        a2.clone(),
        is_act_and_mul=True,
    )
    quant_config = make_nvfp4_moe_quant_config(
        NvFp4MoeBackend.FLASHINFER_TRTLLM, ks13, ks2, kgs13, kgs2, ka13, ka2
    )
    experts = TrtLlmNvFp4LoRAExperts(config, quant_config, True)
    fake_layer = torch.nn.Module()
    fake_layer.w13_weight_scale_2 = kgs13
    fake_layer.w2_weight_scale_2 = kgs2
    fake_layer.w13_input_scale = ka13
    fake_layer.w2_input_scale = ka2
    experts.process_weights_after_loading(fake_layer)

    reference = (
        _dequant_nvfp4(w13, s13, gs13.double()[:, None, None]),
        _dequant_nvfp4(w2, s2, gs2.double()[:, None, None]),
    )
    return experts, (k13, k2), reference


def _make_inputs():
    x = torch.randn(T, K, device="cuda").to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(T)])
    weights = torch.rand(T, TOPK, device="cuda", dtype=torch.float32) + 0.1
    return x, ids.to(torch.int32), weights


def _quant_dequant(x):
    rows = x.flatten(0, -2)
    x_q, block_scale, per_token_scale = quantize_nvfp4_per_token_input(rows)
    rows = _dequant_nvfp4(x_q, block_scale, per_token_scale.double()[..., None])
    return rows.view(x.shape)


def _reference(reference, x, ids, weights, gate_delta=None, up_delta=None):
    w13, w2 = reference
    ids = ids.long()
    h = torch.einsum("tk,tjnk->tjn", _quant_dequant(x), w13[ids])
    gate, up = h[..., :N], h[..., N:]
    if gate_delta is not None:
        gate, up = gate + gate_delta.double(), up + up_delta.double()
    act = torch.nn.functional.silu(gate) * up
    act_q = _quant_dequant(act.to(torch.bfloat16))
    out = torch.einsum("tjn,tjkn->tjk", act_q, w2[ids])
    return act, (out * weights.double()[..., None]).sum(1)


def _run_base_path(experts, kernel_weights, x, ids, weights):
    output = torch.empty(T, K, dtype=torch.bfloat16, device="cuda")
    experts.invoke_routed_moe(
        hidden_states=x,
        w1=kernel_weights[0],
        w2=kernel_weights[1],
        topk_ids_and_weights=(ids, weights),
        gemm1_lora_delta=None,
        global_num_experts=E,
        a1q_scale=None,
        output=output,
    )
    return output


def _run_lora_path(experts, kernel_weights, x, ids, weights, delta):
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


@torch.inference_mode()
def test_gemm1_lora_delta_matches_reference():
    set_random_seed(0)
    experts, kernel_weights, reference = _make_experts()
    x, ids, weights = _make_inputs()
    gate_delta = (torch.randn(T, TOPK, N, device="cuda") / 2).to(torch.bfloat16)
    up_delta = (torch.randn(T, TOPK, N, device="cuda") / 2).to(torch.bfloat16)
    ref_act, ref_out = _reference(reference, x, ids, weights, gate_delta, up_delta)
    _, ref_base_out = _reference(reference, x, ids, weights)
    assert _rel_err(ref_out, ref_base_out) > 0.3

    delta = torch.cat([up_delta, gate_delta], dim=-1)
    act, out = _run_lora_path(experts, kernel_weights, x, ids, weights, delta)
    assert _rel_err(act, ref_act) < 0.02
    assert _rel_err(out, ref_out) < 0.05

    swapped = torch.cat([gate_delta, up_delta], dim=-1)
    _, out = _run_lora_path(experts, kernel_weights, x, ids, weights, swapped)
    assert _rel_err(out, ref_out) > 0.3


@torch.inference_mode()
def test_zero_delta_matches_base_path():
    set_random_seed(0)
    experts, kernel_weights, reference = _make_experts()
    x, ids, weights = _make_inputs()
    base = _run_base_path(experts, kernel_weights, x, ids, weights)
    _, ref_out = _reference(reference, x, ids, weights)
    assert _rel_err(base, ref_out) < 0.02

    zero = torch.zeros(T, TOPK, 2 * N, dtype=torch.bfloat16, device="cuda")
    _, out = _run_lora_path(experts, kernel_weights, x, ids, weights, zero)
    assert _rel_err(out, base) < 0.02


@pytest.mark.parametrize("w13_num_slices", [1, 2], ids=["fused-3d", "per-expert"])
@torch.inference_mode()
def test_prepare_gemm1_lora_delta_layout(w13_num_slices: int):
    set_random_seed(0)
    experts, _, _ = _make_experts()
    ids = _make_inputs()[1]
    gate = torch.randn(T, TOPK, N, device="cuda").to(torch.bfloat16)
    up = torch.randn(T, TOPK, N, device="cuda").to(torch.bfloat16)
    halves = (up, gate) if w13_num_slices == 2 else (gate, up)
    delta = torch.cat(halves, dim=-1)

    lora_context = SimpleNamespace(w13_num_slices=w13_num_slices)
    prepared = experts.prepare_gemm1_lora_delta(delta, lora_context, ids)
    assert torch.equal(prepared, torch.cat([up, gate], dim=-1))


def test_lora_experts_are_opt_in_for_per_token_activations():
    def select(moe_backend, lora, activation_key, routing=RoutingMethodType.TopK):
        config = make_dummy_moe_config(
            num_experts=E, experts_per_token=TOPK, hidden_dim=K, intermediate_size=N
        )
        config.moe_backend = moe_backend
        config.is_lora_enabled = lora
        config.routing_method = routing
        return select_nvfp4_moe_backend(config, kNvfp4Static, activation_key)

    trtllm, lora_cls = NvFp4MoeBackend.FLASHINFER_TRTLLM, TrtLlmNvFp4LoRAExperts
    per_token = kNvfp4DynamicToken
    assert select("flashinfer_trtllm", False, per_token)[1] is not lora_cls
    assert select("flashinfer_trtllm", True, per_token) == (trtllm, lora_cls)
    assert select("auto", True, per_token)[0] != trtllm
    with pytest.raises(ValueError, match="quantization scheme"):
        select("flashinfer_trtllm", True, kNvfp4Dynamic)
    with pytest.raises(ValueError, match="routing method"):
        select("flashinfer_trtllm", True, per_token, RoutingMethodType.Llama4)
