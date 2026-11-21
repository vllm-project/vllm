# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.config import ParallelConfig, SchedulerConfig, VllmConfig
from vllm.config.quantization import QuantizationConfigArgs
from vllm.forward_context import set_forward_context
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

from ..kernels.moe.modular_kernel_tools.parallel_utils import (
    parallel_launch_with_config,
)
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
        max_tokens_across_dp = staticmethod(
            DeepseekV4AiterMegaMoEExperts.max_tokens_across_dp
        )

        def __call__(self, x, topk_weights, topk_ids, max_tokens_across_dp=None):
            captured["weights"], captured["ids"] = topk_weights, topk_ids
            return 2 * x

        def uses_fused_ep_moe(self, max_tokens_across_dp):
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


def _fused_ep_moe_vs_mega_moe_worker(pgi, vllm_config, cpu_group):
    rank, world_size = pgi.rank, pgi.world_size
    num_experts, hidden, inter, top_k = 48 * world_size, 5120, 2304, 6
    generator = torch.Generator(device=pgi.device).manual_seed(rank)
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
        experts.finalize_weights()
        layers[fused] = experts

    def inputs(num_tokens):
        x = torch.randn(num_tokens, hidden, generator=generator).bfloat16()
        topk_ids = (
            torch.rand(num_tokens, num_experts, generator=generator)
            .topk(top_k)
            .indices.int()
        )
        topk_weights = torch.rand(num_tokens, top_k, generator=generator)
        return x, topk_weights, topk_ids

    def step(tokens_across_dp):
        return set_forward_context(
            None,
            vllm_config,
            num_tokens=tokens_across_dp[rank],
            num_tokens_across_dp=torch.tensor(
                tokens_across_dp, dtype=torch.int32, device="cpu"
            ),
        )

    assert not layers[False].uses_fused_ep_moe(1)
    assert layers[True].uses_fused_ep_moe(768)
    assert not layers[True].uses_fused_ep_moe(769)
    # Tokens per DP rank. Every rank follows the DP max, including one with no
    # tokens: FusedEPMoE up to 768, MegaMoEV2 from 769.
    for tokens_across_dp in ([1, 1], [37, 300], [0, 37], [768, 1], [769, 1]):
        x, topk_weights, topk_ids = inputs(tokens_across_dp[rank])
        with step(tokens_across_dp):
            expected = layers[False](x, topk_weights, topk_ids)
            out = layers[True](x, topk_weights, topk_ids)
        assert out.shape == expected.shape == x.shape
        if max(tokens_across_dp) > 768:
            # Both layers ran the same MegaMoEV2 runtime.
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        elif x.shape[0]:
            cosine = torch.nn.functional.cosine_similarity(
                out.float().flatten(), expected.float().flatten(), dim=0
            )
            # Both quantize to MXFP8 and requantize the activation; on random
            # inputs like these FusedEPMoE is within 1e-5 (cosine) of an fp32
            # reference and MegaMoEV2 within 4e-4.
            assert cosine > 0.999, (tokens_across_dp, cosine)

    # CUDA graphs: DP ranks are padded to the same size.
    for num_tokens in (1, 37, 300):
        x, topk_weights, topk_ids = inputs(num_tokens)
        static = [t.clone() for t in (x, topk_weights, topk_ids)]
        with step([num_tokens] * world_size):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = layers[True](*static)
            new_inputs = (x.flip(0), topk_weights.flip(0), topk_ids.flip(0))
            for buffer, value in zip(static, new_inputs):
                buffer.copy_(value)
            graph.replay()
            torch.testing.assert_close(
                graph_out, layers[True](*new_inputs), rtol=0, atol=0
            )


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
def test_fused_ep_moe_matches_mega_moe():
    """With 2 DP / EP ranks, the FusedEPMoE layer gives the MegaMoEV2 layer's
    output on the same weights, every rank follows the DP-max cutoff, and graph
    replays match eager calls bitwise."""
    fused_ep_moe = mega_moe_module._fused_ep_moe_module()
    if fused_ep_moe is None or not fused_ep_moe.fused_ep_moe_supported(
        world_size=2, model_dim=5120, inter_dim=2304, experts=96, topk=6
    ):
        pytest.skip("AITER FusedEPMoE is unavailable on this AITER or GPU")
    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=2,
            enable_expert_parallel=True,
            all2all_backend="mori",
        ),
        scheduler_config=SchedulerConfig.default_factory(max_num_batched_tokens=1024),
    )
    # MegaMoEV2 needs more than MoRI's default 2 GB symmetric heap.
    env = {"VLLM_ROCM_USE_AITER": "1", "MORI_SHMEM_HEAP_SIZE": "8G"}
    parallel_launch_with_config(2, _fused_ep_moe_vs_mega_moe_worker, vllm_config, env)
