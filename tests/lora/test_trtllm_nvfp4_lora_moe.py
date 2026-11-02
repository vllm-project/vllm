# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config import LoRAConfig
from vllm.lora.layers.utils import LoRAMapping
from vllm.lora.punica_wrapper import get_punica_wrapper
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.lora_context import MoELoRAContext
from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
    TrtLlmNvFp4LoRAExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    _requested_kernel_cls,
    backend_to_kernel_cls,
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


def _make_experts(num_local: int = E, ep_rank: int = 0, clamp: float | None = None):
    L = num_local
    w13 = torch.randint(0, 256, (L, 2 * N, K // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (L, K, N // 2), dtype=torch.uint8, device="cuda")
    s13 = (torch.rand(L, 2 * N, K // 16, device="cuda") + 0.5).to(torch.float8_e4m3fn)
    s2 = (torch.rand(L, K, N // 16, device="cuda") + 0.5).to(torch.float8_e4m3fn)
    gs13 = (torch.rand(L, device="cuda") + 0.5) / 64
    gs2 = (torch.rand(L, device="cuda") + 0.5) / 64
    a13 = torch.ones(L, device="cuda")
    a2 = torch.ones(L, device="cuda")

    config = make_dummy_moe_config(
        num_experts=E,
        num_local_experts=L,
        experts_per_token=TOPK,
        hidden_dim=K,
        intermediate_size=N,
    )
    if L != E:
        config.moe_parallel_config = replace(
            FusedMoEParallelConfig.make_no_parallel(),
            ep_size=E // L,
            ep_rank=ep_rank,
            use_ep=True,
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
    quant_config.gemm1_clamp_limit = clamp
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
    with pytest.raises(ValueError, match="does not support"):
        select("flashinfer_trtllm", True, kNvfp4Dynamic)


def test_requested_kernel_cls_puts_lora_experts_first():
    config = make_dummy_moe_config(num_experts=E, hidden_dim=K, intermediate_size=N)
    trtllm = NvFp4MoeBackend.FLASHINFER_TRTLLM
    cutlass = NvFp4MoeBackend.FLASHINFER_CUTLASS
    assert _requested_kernel_cls(trtllm, config) == backend_to_kernel_cls(trtllm)

    config.is_lora_enabled = True
    assert _requested_kernel_cls(trtllm, config) == [
        TrtLlmNvFp4LoRAExperts,
        *backend_to_kernel_cls(trtllm),
    ]
    assert _requested_kernel_cls(cutlass, config) == backend_to_kernel_cls(cutlass)


def _make_lora(rank: int, slots: list[int], num_local: int = E):
    max_loras = 4
    g = 0.5 / rank**0.5

    def rand(*shape, scale):
        return (torch.randn(*shape, device="cuda") * scale).to(torch.bfloat16)

    w13_a = tuple(rand(max_loras, num_local, rank, K, scale=K**-0.5) for _ in range(2))
    w13_b = tuple(rand(max_loras, num_local, N, rank, scale=g) for _ in range(2))
    w2_a = (rand(max_loras, num_local, rank, N, scale=N**-0.5),)
    w2_b = (rand(max_loras, num_local, K, rank, scale=g),)
    adapter_enabled = torch.zeros(max_loras + 1, dtype=torch.int32, device="cuda")
    adapter_enabled[:max_loras] = 1

    punica = get_punica_wrapper(
        8192,
        256,
        "cuda",
        lora_config=LoRAConfig(max_loras=max_loras, max_lora_rank=rank),
    )
    mapping = tuple(s + 1 for s in slots)
    punica.update_metadata(
        LoRAMapping(mapping, mapping, is_prefill=False),
        list(range(1, max_loras + 1)),
        max_loras,
        32000,
    )
    context = MoELoRAContext(
        w13_lora_a_stacked=w13_a,
        w13_lora_b_stacked=w13_b,
        w2_lora_a_stacked=w2_a,
        w2_lora_b_stacked=w2_b,
        adapter_enabled=adapter_enabled,
        max_loras=max_loras,
        top_k=TOPK,
        w13_num_slices=2,
        fully_sharded=False,
        tp_rank=0,
        tp_size=1,
        local_num_experts=num_local,
        punica_wrapper=punica,
        use_tuned_config=False,
    )
    return context


def _lora_reference(reference, context, x, ids, weights, slots, offset, clamp):
    w13, w2 = reference
    num_local = w13.shape[0]
    xq, xd = _quant_dequant(x), x.double()
    s = torch.tensor(slots, device="cuda")
    has = (s >= 0).double()[:, None]
    s = s.clamp(min=0)
    out = torch.zeros(x.shape[0], K, dtype=torch.float64, device="cuda")
    for k in range(TOPK):
        local = ids[:, k].long() - offset
        mask = ((local >= 0) & (local < num_local)).double()[:, None]
        e = local.clamp(0, num_local - 1)
        h = torch.einsum("tk,tnk->tn", xq, w13[e])
        halves = []
        for i, half in enumerate((h[:, :N], h[:, N:])):
            a = context.w13_lora_a_stacked[i][s, e].double()
            b = context.w13_lora_b_stacked[i][s, e].double()
            delta = torch.einsum("tr,tnr->tn", torch.einsum("tk,trk->tr", xd, a), b)
            halves.append(half + has * delta)
        gate, up = halves
        if clamp is not None:
            gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
        act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        base = torch.einsum("tn,tkn->tk", _quant_dequant(act), w2[e])
        a2 = context.w2_lora_a_stacked[0][s, e].double()
        b2 = context.w2_lora_b_stacked[0][s, e].double()
        d2 = torch.einsum("tn,trn->tr", act.double(), a2)
        d2 = torch.einsum("tr,tkr->tk", d2, b2)
        out += mask * weights[:, k, None].double() * (base + has * d2)
    return out


def _apply(experts, kernel_weights, context, x, ids, weights, expert_map):
    experts.set_lora_context(context)
    output = torch.empty(x.shape[0], K, dtype=torch.bfloat16, device="cuda")
    empty = torch.empty(0, device="cuda")
    experts.apply(
        output,
        x,
        kernel_weights[0],
        kernel_weights[1],
        weights,
        ids,
        MoEActivation.SILU,
        E,
        expert_map,
        None,
        None,
        empty,
        empty,
        None,
        False,
    )
    return output


def _slots(pattern: str, T: int) -> list[int]:
    return {
        "none": [-1] * T,
        "one": [0] * T,
        "mixed": [(-1, 0, 1)[t % 3] for t in range(T)],
        "four": [t % 4 for t in range(T)],
    }[pattern]


@pytest.mark.parametrize(
    "T,rank,pattern,clamp,ep",
    [
        (1, 8, "one", None, False),
        (64, 16, "mixed", None, False),
        (64, 64, "four", None, False),
        (33, 16, "none", None, False),
        (64, 16, "mixed", 1.0, False),
        (64, 16, "mixed", None, True),
    ],
    ids=["1tok-r8", "mixed-r16", "4-adapters-r64", "no-lora", "clamp", "ep2"],
)
@torch.inference_mode()
def test_apply_with_lora_matches_reference(T, rank, pattern, clamp, ep):
    set_random_seed(0)
    num_local, ep_rank = (E // 2, 1) if ep else (E, 0)
    experts, kernel_weights, reference = _make_experts(num_local, ep_rank, clamp)
    x = torch.randn(T, K, device="cuda").to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(T)])
    ids = ids.to(torch.int32)
    weights = torch.rand(T, TOPK, device="cuda", dtype=torch.float32) + 0.1
    expert_map = None
    if ep:
        expert_map = torch.full((E,), -1, dtype=torch.int32, device="cuda")
        expert_map[num_local * ep_rank :] = torch.arange(num_local, device="cuda")
    slots = _slots(pattern, T)
    context = _make_lora(rank, slots, num_local)

    out = _apply(experts, kernel_weights, context, x, ids, weights, expert_map)
    offset = num_local * ep_rank
    ref = _lora_reference(reference, context, x, ids, weights, slots, offset, clamp)
    no_lora = [-1] * T
    ref_base = _lora_reference(
        reference, context, x, ids, weights, no_lora, offset, clamp
    )
    assert _rel_err(out, ref) < 0.05
    if pattern == "none":
        base = torch.empty_like(out)
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
        assert torch.equal(out, base)
    else:
        assert _rel_err(ref, ref_base) > 0.2


def test_apply_with_lora_cuda_graph_matches_eager():
    set_random_seed(0)
    T = 16
    experts, kernel_weights, _ = _make_experts()
    x = torch.randn(T, K, device="cuda").to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(T)])
    ids = ids.to(torch.int32)
    weights = torch.rand(T, TOPK, device="cuda", dtype=torch.float32) + 0.1
    context = _make_lora(16, _slots("mixed", T))
    with torch.inference_mode():
        eager = _apply(experts, kernel_weights, context, x, ids, weights, None)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _apply(experts, kernel_weights, context, x, ids, weights, None)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _apply(experts, kernel_weights, context, x, ids, weights, None)
        captured.zero_()
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(captured, eager)


@torch.inference_mode()
def test_apply_rejects_batches_above_the_launch_limit(monkeypatch):
    set_random_seed(0)
    T = 8
    experts, kernel_weights, _ = _make_experts()
    monkeypatch.setattr(experts, "_get_chunk_size", lambda: T - 1)
    x, ids, weights = _make_inputs()
    context = _make_lora(16, _slots("one", T))
    with pytest.raises(ValueError, match="launch limit"):
        _apply(experts, kernel_weights, context, x[:T], ids[:T], weights[:T], None)


@pytest.mark.parametrize(
    "field,value,supported",
    [
        ("routing_method", RoutingMethodType.Renormalize, True),
        ("routing_method", RoutingMethodType.RenormalizeNaive, True),
        ("routing_method", RoutingMethodType.DeepSeekV3, True),
        ("routing_method", RoutingMethodType.Sigmoid, True),
        ("moe_parallel_config", dict(ep_size=2, use_ep=True), True),
        ("routing_method", RoutingMethodType.Llama4, False),
        ("activation", MoEActivation.GELU, False),
        ("moe_parallel_config", dict(dp_size=2), False),
        ("moe_parallel_config", dict(enable_eplb=True), False),
    ],
    ids=[
        "renormalize",
        "renormalize-naive",
        "deepseek-v3",
        "sigmoid",
        "ep2",
        "llama4",
        "gelu",
        "dp2",
        "eplb",
    ],
)
def test_lora_never_falls_back_to_lora_unaware_experts(field, value, supported):
    config = make_dummy_moe_config(
        num_experts=E, experts_per_token=TOPK, hidden_dim=K, intermediate_size=N
    )
    config.moe_backend = "flashinfer_trtllm"
    config.is_lora_enabled = True
    if field == "moe_parallel_config":
        value = replace(FusedMoEParallelConfig.make_no_parallel(), **value)
    setattr(config, field, value)
    if supported:
        assert select_nvfp4_moe_backend(config, kNvfp4Static, kNvfp4DynamicToken) == (
            NvFp4MoeBackend.FLASHINFER_TRTLLM,
            TrtLlmNvFp4LoRAExperts,
        )
    else:
        with pytest.raises(ValueError, match="does not support"):
            select_nvfp4_moe_backend(config, kNvfp4Static, kNvfp4DynamicToken)
