# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import aux_stream

from ..utils import multi_gpu_test

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


def _fake_fused_ep_moe_module(supported=True):
    """Stands in for `aiter.ops.flydsl.fused_ep_moe` (ROCm/aiter#8368)."""
    counts = {"init": 0, "prepare": 0}

    class FusedEPMoE:
        def __init__(self, **kwargs):
            counts["init"] += 1

        def prepare(self, weights):
            counts["prepare"] += 1

        def __call__(self, x, x_scales, weights, topk_weights, topk_ids):
            return "fused"

    module = SimpleNamespace(
        fused_ep_moe_supported=lambda **shape: supported,
        FusedEPMoEWeights=lambda *tensors: tensors,
        FusedEPMoE=FusedEPMoE,
    )
    return module, counts


def _use_fused_ep_moe_module(monkeypatch, module):
    monkeypatch.setattr(mega_moe_module, "_fused_ep_moe_module", lambda: module)
    monkeypatch.setattr(DeepseekV4AiterMegaMoEExperts, "_fused_ep_moe_cache", {})
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "flag, available, supported, instances",
    [
        ("0", True, True, 0),
        ("1", False, True, 0),
        ("1", True, False, 0),
        ("1", True, True, 1),
    ],
)
def test_fused_ep_moe_needs_flag_aiter_op_and_supported_layer(
    monkeypatch, flag, available, supported, instances
):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSED_EP_MOE", flag)
    module, counts = _fake_fused_ep_moe_module(supported)
    _use_fused_ep_moe_module(monkeypatch, module if available else None)
    layers = [_make_experts(monkeypatch) for _ in range(3)]
    for experts in layers:
        experts._runtimes = [(128, "megamoe")]
        experts._init_fused_ep_moe()
        assert experts.uses_fused_ep_moe(1) is bool(instances)
    # Construction and prepare() are collectives: once per process, not per layer.
    assert counts == {"init": instances, "prepare": instances}


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens_across_dp, large_m, fused",
    [
        ([1, 768], False, True),
        ([1, 769], False, False),
        ([1, 512], True, True),
        ([1, 513], True, False),
    ],
)
def test_fused_ep_moe_selected_by_max_tokens_across_dp(
    monkeypatch, tokens_across_dp, large_m, fused
):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSED_EP_MOE", "1")
    _use_fused_ep_moe_module(monkeypatch, _fake_fused_ep_moe_module()[0])
    experts = _make_experts(monkeypatch)
    megamoe = SimpleNamespace(
        large_m_available=large_m, forward=lambda x, w, ids: "megamoe"
    )
    experts._runtimes = [(4096, megamoe)]
    experts._init_fused_ep_moe()
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
    # Both paths are collectives: a rank with one token follows the DP max.
    out = experts(
        torch.zeros(1, HIDDEN), torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.int32)
    )
    assert out == ("fused" if fused else "megamoe")


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


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
@pytest.mark.parametrize("hash_routing", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 37])
@pytest.mark.parametrize(
    "shared_stream, fused_ep_moe", [(False, False), (True, False), (True, True)]
)
def test_mega_moe_routing_matches_reference(
    monkeypatch, hash_routing, num_tokens, shared_stream, fused_ep_moe
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

    class Experts:
        def __call__(self, x, topk_weights, topk_ids):
            captured["weights"], captured["ids"] = topk_weights, topk_ids
            return 2 * x

        def uses_fused_ep_moe(self, num_tokens):
            return fused_ep_moe

    def shared_experts(x):
        captured["shared_stream"] = torch.cuda.current_stream()
        return 3 * x

    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.gate = Gate()
    moe.experts = Experts()
    moe.shared_experts = shared_experts
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
    # FusedEPMoE must not share the GPU, so the shared expert then runs serially.
    on_aux_stream = captured["shared_stream"] == moe.shared_experts_stream
    assert on_aux_stream is (shared_stream and not fused_ep_moe)


def _fused_ep_moe_vs_mega_moe_worker(rank, world_size, port):
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "8G")
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    import mori

    cpu_group = dist.new_group(backend="gloo")
    torch._C._distributed_c10d._register_process_group("mori", cpu_group)
    mori.shmem.shmem_torch_process_group_init("mori")
    mega_moe_module.get_ep_group = lambda: SimpleNamespace(
        rank_in_group=rank, world_size=world_size, cpu_group=cpu_group
    )
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024)
    )
    num_experts, hidden, inter, top_k = 48 * world_size, 5120, 2304, 6
    generator = torch.Generator().manual_seed(rank)
    layers: dict[bool, DeepseekV4AiterMegaMoEExperts] = {}
    for fused in (False, True):
        # finalize_weights() reads the flag.
        os.environ["VLLM_ROCM_USE_AITER_FUSED_EP_MOE"] = str(int(fused))
        experts = DeepseekV4AiterMegaMoEExperts(
            vllm_config,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden,
            intermediate_size=inter,
            swiglu_limit=10.0,
        )
        if not layers:
            for name, param in experts.named_parameters():
                if name.endswith("scale"):
                    # E8M0 exponents 2^-7 .. 2^-4.
                    param.data.random_(120, 124, generator=generator)
                else:
                    param.data.random_(0, 256, generator=generator)
            raw = {k: v.clone() for k, v in experts.state_dict().items()}
        experts.load_state_dict(raw)
        experts.cuda()
        experts.finalize_weights()
        assert experts.uses_fused_ep_moe(1) is fused
        layers[fused] = experts

    for num_tokens in (1, 37, 300):
        x = torch.randn(num_tokens, hidden, generator=generator).bfloat16().cuda()
        topk_ids = (
            torch.rand(num_tokens, num_experts, generator=generator)
            .topk(top_k)
            .indices.int()
            .cuda()
        )
        topk_weights = torch.rand(num_tokens, top_k, generator=generator).cuda()
        expected = layers[False](x, topk_weights, topk_ids)
        out = layers[True](x, topk_weights, topk_ids)
        cosine = torch.nn.functional.cosine_similarity(
            out.float().flatten(), expected.float().flatten(), dim=0
        )
        # Both quantize to MXFP8 and requantize the activation;
        # on random inputs like these FusedEPMoE is within 1e-5 (cosine) of an
        # fp32 reference and MegaMoEV2 within 4e-4.
        assert cosine > 0.999, (num_tokens, cosine)

        static = [t.clone() for t in (x, topk_weights, topk_ids)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = layers[True](*static)
        new_inputs = (x.flip(0), topk_weights.flip(0), topk_ids.flip(0))
        for buffer, value in zip(static, new_inputs):
            buffer.copy_(value)
        graph.replay()
        torch.testing.assert_close(graph_out, layers[True](*new_inputs), rtol=0, atol=0)
    dist.barrier(cpu_group)
    mori.shmem.shmem_finalize()


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
def test_fused_ep_moe_matches_mega_moe():
    """FusedEPMoE on the tensors MegaMoEV2 is built with gives MegaMoEV2's
    output on 2 EP ranks, and graph replays match eager calls bitwise."""
    fused_ep_moe = mega_moe_module._fused_ep_moe_module()
    if fused_ep_moe is None or not fused_ep_moe.fused_ep_moe_supported(
        world_size=2, model_dim=5120, inter_dim=2304, experts=96, topk=6
    ):
        pytest.skip("AITER FusedEPMoE is unavailable on this AITER or GPU")
    mp.spawn(
        _fused_ep_moe_vs_mega_moe_worker,
        args=(2, get_open_port()),
        nprocs=2,
    )
