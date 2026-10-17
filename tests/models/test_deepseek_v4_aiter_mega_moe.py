# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.models.deepseek_v4.amd import mega_moe as mega_moe_module
from vllm.models.deepseek_v4.amd.mega_moe import (
    DeepseekV4AiterMegaMoEExperts,
)
from vllm.models.deepseek_v4.amd.model import (
    DeepseekV4AiterMegaMoEMixtureOfExperts,
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4MoE,
)
from vllm.platforms import current_platform

HIDDEN = 64
INTER = 64


def _make_experts(
    monkeypatch,
    *,
    ep_rank=0,
    ep_size=1,
    num_experts=2,
    intermediate_size=INTER,
    max_num_batched_tokens=3000,
    swiglu_limit=None,
    fuse_shared_expert=False,
):
    monkeypatch.setattr(
        mega_moe_module,
        "get_ep_group",
        lambda: SimpleNamespace(
            rank_in_group=ep_rank, world_size=ep_size, cpu_group="ep"
        ),
    )
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens)
    )
    return DeepseekV4AiterMegaMoEExperts(
        vllm_config,
        num_experts=num_experts,
        top_k=2,
        hidden_size=HIDDEN,
        intermediate_size=intermediate_size,
        swiglu_limit=swiglu_limit,
        fuse_shared_expert=fuse_shared_expert,
    )


@pytest.mark.cpu_test
def test_experts_load_into_local_slots(monkeypatch):
    model = torch.nn.Module()
    model.ffn = torch.nn.Module()
    model.ffn.experts = _make_experts(monkeypatch, ep_rank=1, ep_size=2, num_experts=4)
    params = dict(model.named_parameters())
    mapping = fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=4,
        routed_experts_prefix="",
    )

    shapes = {
        ("w1", "weight"): (INTER, HIDDEN // 2),
        ("w3", "weight"): (INTER, HIDDEN // 2),
        ("w2", "weight"): (HIDDEN, INTER // 2),
        ("w1", "weight_scale"): (INTER, HIDDEN // 32),
        ("w3", "weight_scale"): (INTER, HIDDEN // 32),
        ("w2", "weight_scale"): (HIDDEN, INTER // 32),
    }
    shard_value = {"w1": 1, "w3": 2, "w2": 3}
    loaded = {}
    for expert in range(4):
        for (shard, suffix), shape in shapes.items():
            name = f"ffn.experts.{expert}.{shard}.{suffix}"
            weight = torch.full(
                shape, 10 * expert + shard_value[shard], dtype=torch.uint8
            )
            for param_name, weight_name, expert_id, shard_id in mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                param = params[mapped]
                if param.weight_loader(
                    param,
                    weight,
                    mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                ):
                    loaded[name] = mapped
                    break

    assert sorted({n.split(".")[2] for n in loaded}) == ["2", "3"]
    assert len(loaded) == 2 * len(shapes)
    for slot, expert in enumerate((2, 3)):
        for suffix in ("weight", "weight_scale"):
            w13 = params[f"ffn.experts.w13_{suffix}"][slot]
            w2 = params[f"ffn.experts.w2_{suffix}"][slot]
            assert torch.all(w13[:INTER] == 10 * expert + 1)
            assert torch.all(w13[INTER:] == 10 * expert + 2)
            assert torch.all(w2 == 10 * expert + 3)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens_across_dp, expected",
    [([4, 128], "decode"), ([4, 129], "prefill")],
)
def test_runtime_selected_by_max_tokens_across_dp(
    monkeypatch, tokens_across_dp, expected
):
    experts = _make_experts(monkeypatch)
    experts._runtimes = [(128, "decode"), (4096, "prefill")]
    monkeypatch.setattr(mega_moe_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        mega_moe_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(
                num_tokens_across_dp_cpu=torch.tensor(tokens_across_dp)
            )
        ),
    )
    # Every rank must take the same runtime, so a small local batch still gets
    # the prefill runtime when another DP rank is over the decode capacity.
    assert experts.select_runtime(4) == expected


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
def test_shared_expert_route_appended(monkeypatch):
    experts = _make_experts(
        monkeypatch, ep_rank=3, ep_size=8, num_experts=384, fuse_shared_expert=True
    )
    ids = torch.randint(0, 384, (37, 2), device="cuda", dtype=torch.int32)
    ids[::5, 1] = -1
    weights = torch.rand(37, 2, device="cuda")
    out_weights, out_ids = experts.append_shared_expert(weights, ids)

    shared = torch.full((37, 1), 3 * 49 + 48, device="cuda", dtype=torch.int32)
    expected_ids = torch.cat([torch.where(ids >= 0, ids + ids // 48, ids), shared], 1)
    torch.testing.assert_close(out_ids, expected_ids)
    torch.testing.assert_close(
        out_weights, torch.cat([weights, torch.ones(37, 1, device="cuda")], 1)
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
@pytest.mark.parametrize("hash_routing", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 37])
def test_mega_moe_routing_matches_reference(monkeypatch, hash_routing, num_tokens):
    num_experts, top_k, vocab = 16, 4, 64
    scaling = 2.5
    device = "cuda"
    torch.manual_seed(0)

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(num_experts, HIDDEN, device=device) * 0.1
            self.e_score_correction_bias = (
                None if hash_routing else torch.randn(num_experts, device=device)
            )
            self.tid2eid = (
                torch.stack(
                    [
                        torch.randperm(num_experts, device=device)[:top_k]
                        for _ in range(vocab)
                    ]
                ).to(torch.int32)
                if hash_routing
                else None
            )
            self.bias_vl = None

        def forward(self, x):
            return x.float() @ self.weight.t(), None

    captured = {}

    def experts(x, topk_weights, topk_ids):
        captured["weights"], captured["ids"] = topk_weights, topk_ids
        return 2 * x

    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.gate = Gate()
    moe.experts = experts
    moe.shared_experts = lambda x: 3 * x
    moe.scoring_func = "sqrtsoftplus"
    moe.n_activated_experts = top_k
    moe.renormalize = True
    moe.routed_scaling_factor = scaling
    moe.image_sentinel_lo = 0

    x = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, vocab, (num_tokens,), device=device)
    out = moe._forward_mega_moe(x, input_ids)

    ids, weights = captured["ids"], captured["weights"]
    assert ids.dtype == torch.int32 and ids.is_contiguous()
    assert weights.dtype == torch.float32 and weights.is_contiguous()

    scores = torch.nn.functional.softplus(x.float() @ moe.gate.weight.t()).sqrt()
    if hash_routing:
        expected_ids = moe.gate.tid2eid[input_ids].long()
    else:
        expected_ids = (scores + moe.gate.e_score_correction_bias).topk(top_k).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights *= scaling / expected_weights.sum(dim=-1, keepdim=True)

    order = ids.long().sort(dim=-1).indices
    expected_order = expected_ids.sort(dim=-1).indices
    torch.testing.assert_close(
        ids.long().gather(1, order), expected_ids.gather(1, expected_order)
    )
    torch.testing.assert_close(
        weights.gather(1, order),
        expected_weights.gather(1, expected_order),
        rtol=1e-3,
        atol=1e-4,
    )
    torch.testing.assert_close(out, 5 * x)


@pytest.mark.cpu_test
@pytest.mark.parametrize("fuse_shared_expert", [False, True])
def test_eplb_expert_weights_are_routed_slot_views(monkeypatch, fuse_shared_expert):
    experts = _make_experts(
        monkeypatch,
        ep_rank=1,
        ep_size=2,
        num_experts=4,
        fuse_shared_expert=fuse_shared_expert,
    )
    params = (
        experts.w13_weight,
        experts.w13_weight_scale,
        experts.w2_weight,
        experts.w2_weight_scale,
    )
    for param in params:
        for slot in range(experts.num_local_experts):
            param.data[slot].fill_(slot + 1)
    # finalize_weights() leaves the scales as 2-D (experts * rows, cols).
    for param in (experts.w13_weight_scale, experts.w2_weight_scale):
        param.data = param.data.view(-1, param.shape[-1])
    experts._runtimes = [(4096, object())]

    views = experts.get_expert_weights()

    assert len(views) == len(params)
    for view, param in zip(views, params):
        # Routed slots only (a fused shared expert is the last local slot), one
        # row per expert, aliasing the weights the MegaMoE kernels read.
        assert view.shape == (2, param.numel() // experts.num_local_experts)
        assert view.data_ptr() == param.data.data_ptr()
        for slot in range(2):
            assert torch.all(view[slot] == slot + 1)


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens_across_dp", [[4, 128], [4, 129]])
def test_eplb_remaps_ids_before_runtime_choice(monkeypatch, tokens_across_dp):
    experts = _make_experts(monkeypatch, ep_size=1, num_experts=4)
    seen = {}

    class Runtime:
        def __init__(self, name):
            self.name = name

        def forward(self, x, topk_weights, topk_ids):
            seen[self.name] = topk_ids
            return x

    experts._runtimes = [(128, Runtime("decode")), (4096, Runtime("prefill"))]
    monkeypatch.setattr(mega_moe_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        mega_moe_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(
                num_tokens_across_dp_cpu=torch.tensor(tokens_across_dp)
            )
        ),
    )
    logical_to_physical = torch.tensor([[[3], [2], [1], [0]]])
    monkeypatch.setattr(
        mega_moe_module,
        "eplb_map_to_physical_and_record",
        lambda topk_ids, logical_to_physical_map, **kwargs: (
            logical_to_physical_map[topk_ids.long(), 0].to(topk_ids.dtype)
        ),
    )
    experts.set_eplb_state(
        0,
        torch.zeros(1, 4, dtype=torch.int32),
        logical_to_physical,
        torch.ones(1, 4, dtype=torch.long),
    )
    experts.eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)

    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    experts(torch.zeros(2, HIDDEN), torch.ones(2, 2), topk_ids)

    # Whichever runtime the DP-wide token count selects gets physical ids.
    (ids,) = seen.values()
    torch.testing.assert_close(ids, torch.tensor([[3, 2], [1, 0]], dtype=torch.int32))


class _MixtureOfExpertsModel(DeepseekV4AiterMegaMoEMixtureOfExperts):
    def __init__(self, layers, use_mega_moe):
        self.config = SimpleNamespace()
        self.model = SimpleNamespace(layers=layers, use_mega_moe=use_mega_moe)

    set_moe_parameters = DeepseekV4ForCausalLM.set_moe_parameters


def _moe_decoder_layer(experts):
    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.experts = experts
    moe.n_routed_experts = 4
    moe.n_shared_experts = 1
    moe.n_logical_experts = 4
    moe.n_physical_experts = 4
    moe.n_redundant_experts = 0
    moe.n_local_physical_experts = 2
    layer = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.ffn = moe
    return layer


@pytest.mark.cpu_test
@pytest.mark.parametrize("use_mega_moe", [False, True])
def test_mixture_of_experts_registers_mega_moe_layers_only(monkeypatch, use_mega_moe):
    experts = [_make_experts(monkeypatch, ep_size=2, num_experts=4) for _ in range(2)]
    layers = [_moe_decoder_layer(e) for e in experts]
    model = _MixtureOfExpertsModel(layers + [torch.nn.Identity()], use_mega_moe)

    model.set_moe_parameters()

    assert is_mixture_of_experts(model) == use_mega_moe
    if use_mega_moe:
        assert model.moe_layers == experts
        assert model.num_logical_experts == model.num_physical_experts == 4
        assert model.num_local_physical_experts == 2
        assert model.num_redundant_experts == 0
        assert model.num_expert_groups == 1
    else:
        assert model.moe_layers == []
