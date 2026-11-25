# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.utils.config_utils import (
    is_shared_expert_quant_fse_compatible,
)
from vllm.models.deepseek_v4.amd import mega_moe as mega_moe_module
from vllm.models.deepseek_v4.amd.mega_moe import (
    DeepseekV4AiterMegaMoEExperts,
)
from vllm.models.deepseek_v4.amd.model import DeepseekV4MoE
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config
from vllm.platforms import current_platform
from vllm.utils.torch_utils import aux_stream

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
    num_logical_experts=None,
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
        num_logical_experts=num_logical_experts,
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
def test_eplb_redundant_slots_load_their_logical_expert(monkeypatch):
    # 4 logical + 2 redundant physical experts over 2 ranks: rank 1 holds
    # physical 3, 4, 5, i.e. logical 3, 0, 1 under EPLB's initial placement.
    experts = _make_experts(
        monkeypatch, ep_rank=1, ep_size=2, num_experts=6, num_logical_experts=4
    )
    param = experts.w2_weight
    for expert in range(4):
        weight = torch.full((HIDDEN, INTER // 2), expert, dtype=torch.uint8)
        loaded = experts.weight_loader(
            param, weight, "w2_weight", "w2", expert, return_success=True
        )
        assert loaded == (expert in (0, 1, 3))
    assert [int(param.data[slot, 0, 0]) for slot in range(3)] == [3, 0, 1]


@pytest.mark.cpu_test
@pytest.mark.parametrize("fuse_shared_expert", [False, True])
@pytest.mark.parametrize("flat_scales", [False, True])
def test_eplb_expert_weights_are_routed_expert_views(
    monkeypatch, fuse_shared_expert, flat_scales
):
    experts = _make_experts(
        monkeypatch, ep_size=2, num_experts=4, fuse_shared_expert=fuse_shared_expert
    )
    if flat_scales:
        # finalize_weights leaves the shuffled scales 2-D, expert-major.
        for name in ("w13_weight_scale", "w2_weight_scale"):
            param = getattr(experts, name)
            param.data = param.data.view(-1, param.shape[-1])
    views = experts.get_expert_weights()
    assert len(views) == 4
    for view, name in zip(
        views, ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    ):
        param = getattr(experts, name)
        assert view.shape == (2, param.numel() // experts.num_local_experts)
        assert view.is_contiguous()
        # EPLB writes through the view into the live parameter.
        view[1].fill_(7)
        slots = param.data.view(experts.num_local_experts, -1)
        assert torch.all(slots[1] == 7)
        assert torch.all(slots[0] == 0)
        if fuse_shared_expert:
            assert torch.all(slots[2] == 0)


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


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "moe_backend, targets, fused",
    [
        ("aiter_mega_moe", None, False),
        ("aiter_mega_moe", {"*shared_experts*": "mxfp8"}, False),
        ("aiter_mega_moe", {"*shared_experts*": "mxfp4"}, True),
        ("aiter", {"*shared_experts*": "mxfp4"}, False),
    ],
)
def test_mega_moe_shared_expert_fusion_requires_mxfp4_target(
    moe_backend, targets, fused
):
    quant_config = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
    )
    quant_config._resolved_expert_dtype = "fp4"
    quant_config.packed_modules_mapping = {"gate_up_proj": ["w1", "w3"]}
    if targets is not None:
        quant_config.online_quantization_config = OnlineQuantizationConfig(
            QuantizationConfigArgs(targets=targets)
        )
    compatible, _ = is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.3.ffn.experts",
        "model.layers.3.ffn.shared_experts",
        moe_backend=moe_backend,
    )
    assert compatible is fused


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


def _eplb_reference(ids, l2p, replica_count, num_unpadded, record, num_physical):
    """Replica choice and load of `eplb_map_and_record` for one token row
    index per row."""
    rows = torch.arange(ids.shape[0], device=ids.device)[:, None].expand_as(ids)
    valid = (ids >= 0) & (ids < l2p.shape[0])
    safe = torch.where(valid, ids, 0).long()
    hashed = (rows.long() * 2654435769) & 0xFFFFFFFF
    replica = hashed % replica_count[safe].long()
    physical = torch.where(valid, l2p[safe, replica], -1)
    load = torch.zeros(num_physical, dtype=torch.int32, device=ids.device)
    if record:
        counted = physical[(rows < num_unpadded) & (physical >= 0)]
        load.index_add_(0, counted, torch.ones_like(counted, dtype=torch.int32))
    return physical, load


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
@pytest.mark.parametrize("fuse_shared_expert", [False, True])
@pytest.mark.parametrize("record", [False, True])
def test_eplb_place_experts_matches_reference(monkeypatch, fuse_shared_expert, record):
    # 384 logical + 16 redundant experts over 8 ranks: 50 routed slots a rank.
    num_logical, num_physical, ep_size, ep_rank = 384, 400, 8, 3
    experts = _make_experts(
        monkeypatch,
        ep_rank=ep_rank,
        ep_size=ep_size,
        num_experts=num_physical,
        num_logical_experts=num_logical,
        fuse_shared_expert=fuse_shared_expert,
    )
    # Logical 0..15 have a second replica in the redundant slots 384..399.
    replica_count = torch.ones(num_logical, dtype=torch.int64, device="cuda")
    replica_count[:16] = 2
    l2p = torch.full((num_logical, 2), -1, dtype=torch.int64, device="cuda")
    l2p[:, 0] = torch.arange(num_logical, device="cuda")
    l2p[:16, 1] = torch.arange(384, 400, device="cuda")
    load = torch.zeros(num_physical, dtype=torch.int32, device="cuda")
    experts.set_eplb_state(0, load[None], l2p[None], replica_count[None])
    num_tokens, num_unpadded = 37, 30
    experts.eplb_state.should_record_tensor = torch.tensor(record, device="cuda")
    experts.eplb_state.num_unpadded_tokens_tensors = [
        torch.tensor(num_unpadded, dtype=torch.int32, device="cuda")
    ]

    ids = torch.randint(0, 16, (num_tokens, 2), device="cuda", dtype=torch.int32)
    ids[::5, 1] = -1
    weights = torch.rand(num_tokens, 2, device="cuda")
    out_weights, out_ids = experts.place_experts(weights, ids)

    physical, expected_load = _eplb_reference(
        ids, l2p, replica_count, num_unpadded, record, num_physical
    )
    torch.testing.assert_close(load, expected_load)
    if not fuse_shared_expert:
        torch.testing.assert_close(out_ids, physical.to(ids.dtype))
        assert out_weights is weights
        return
    routed = 50
    shared = torch.full((num_tokens, 1), ep_rank * (routed + 1) + routed, device="cuda")
    expected_ids = torch.cat(
        [torch.where(physical >= 0, physical + physical // routed, physical), shared],
        1,
    )
    torch.testing.assert_close(out_ids, expected_ids.to(ids.dtype))
    torch.testing.assert_close(
        out_weights, torch.cat([weights, torch.ones(num_tokens, 1, device="cuda")], 1)
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
@pytest.mark.parametrize("hash_routing", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 37])
@pytest.mark.parametrize("shared_stream", [False, True])
def test_mega_moe_routing_matches_reference(
    monkeypatch, hash_routing, num_tokens, shared_stream
):
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
    moe.shared_experts_stream = aux_stream() if shared_stream else None
    moe.shared_experts_events = (
        (torch.cuda.Event(), torch.cuda.Event()) if shared_stream else None
    )
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
