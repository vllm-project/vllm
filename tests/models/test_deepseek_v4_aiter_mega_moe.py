# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm.distributed
import vllm.utils.import_utils
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.kernel import (
    AITER_MEGA_MOE,
    MEGA_MOE_BACKENDS,
    PASSTHROUGH_ALL2ALL_BACKEND,
    bind_passthrough_all2all_backend,
)
from vllm.config.quantization import QuantizationConfigArgs
from vllm.distributed.device_communicators import all2all
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.experts import (
    aiter_mega_moe as mega_moe_module,
)
from vllm.model_executor.layers.fused_moe.experts.aiter_mega_moe import (
    AiterMegaMoEExperts,
    load_aiter_mega_moe_expert,
    shuffle_aiter_mega_moe_weights,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    backend_to_kernel_cls,
    make_mxfp4_moe_quant_config,
    map_mxfp4_backend,
    mxfp4_round_up_hidden_size_and_intermediate_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.utils.config_utils import (
    is_shared_expert_quant_fse_compatible,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
    kMxfp4Static,
)
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config
from vllm.platforms import current_platform

NUM_EXPERTS, TOPK, HIDDEN, INTER = 384, 6, 7168, 3072


def _ep_moe_config(ep_size=8, hidden=HIDDEN, inter=INTER, **kwargs):
    moe = make_dummy_moe_config(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS // ep_size,
        experts_per_token=TOPK,
        hidden_dim=hidden,
        intermediate_size=inter,
        max_num_tokens=3000,
    )
    moe.moe_parallel_config = replace(
        moe.moe_parallel_config,
        **{
            "use_ep": True,
            "ep_size": ep_size,
            "dp_size": ep_size,
            "all2all_backend": PASSTHROUGH_ALL2ALL_BACKEND,
            **kwargs,
        },
    )
    return moe


def _support_reason(moe, activation_key=None) -> str:
    supported, reason = AiterMegaMoEExperts.is_supported_config(
        AiterMegaMoEExperts,
        moe,
        kMxfp4Static,
        activation_key,
        mk.FusedMoEActivationFormat.Standard,
    )
    assert supported == (reason is None)
    return reason or ""


def _make_experts(moe) -> AiterMegaMoEExperts:
    quant_config = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.AITER_MEGA_MOE, w1_scale=torch.ones(1), w2_scale=torch.ones(1)
    )
    assert quant_config is not None
    return AiterMegaMoEExperts(moe, quant_config)


@pytest.fixture
def mega_moe_host(monkeypatch):
    monkeypatch.setattr(
        AiterMegaMoEExperts, "_supports_current_device", staticmethod(lambda: True)
    )
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    monkeypatch.setattr(vllm.utils.import_utils, "has_mori", lambda: True)


def test_aiter_mega_moe_is_an_mxfp4_oracle_backend():
    assert AITER_MEGA_MOE in MEGA_MOE_BACKENDS
    assert map_mxfp4_backend(AITER_MEGA_MOE) == [Mxfp4MoeBackend.AITER_MEGA_MOE]
    assert backend_to_kernel_cls(Mxfp4MoeBackend.AITER_MEGA_MOE) == [
        AiterMegaMoEExperts
    ]
    # MegaMoEV2 is built for the model's own dimensions.
    assert mxfp4_round_up_hidden_size_and_intermediate_size(
        Mxfp4MoeBackend.AITER_MEGA_MOE, 2880, 1440
    ) == (2880, 1440)

    w1_scale, w2_scale = torch.ones(2, 4), torch.ones(2, 2)
    quant_config = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.AITER_MEGA_MOE,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        swiglu_limit=7.0,
    )
    assert quant_config is not None and quant_config.use_mxfp4_w4a16
    assert quant_config.w1_scale is w1_scale and quant_config.w2_scale is w2_scale
    assert quant_config.gemm1_clamp_limit == 7.0


def test_aiter_mega_moe_binds_the_passthrough_all2all_backend():
    parallel = SimpleNamespace(all2all_backend="allgather_reducescatter")
    bind_passthrough_all2all_backend(AITER_MEGA_MOE, parallel)
    assert parallel.all2all_backend == PASSTHROUGH_ALL2ALL_BACKEND

    with pytest.raises(ValueError, match="cannot be paired"):
        bind_passthrough_all2all_backend(
            AITER_MEGA_MOE, SimpleNamespace(all2all_backend="mori_high_throughput")
        )


def test_aiter_mega_moe_reports_its_constraints(mega_moe_host, monkeypatch):
    assert _support_reason(_ep_moe_config()) == ""
    # MegaMoEV2 quantizes activations itself and takes them in bf16.
    assert "quantization scheme" in _support_reason(_ep_moe_config(), kMxfp4Dynamic)

    no_ep = _ep_moe_config()
    no_ep.moe_parallel_config = replace(no_ep.moe_parallel_config, use_ep=False)
    assert "parallel config" in _support_reason(no_ep)

    tp_attention = _ep_moe_config()
    tp_attention.moe_parallel_config = replace(
        tp_attention.moe_parallel_config, dp_size=4
    )
    assert "tensor parallel attention" in _support_reason(tp_attention)

    assert "ep_size=16" in _support_reason(_ep_moe_config(ep_size=16))
    assert "EPLB" in _support_reason(_ep_moe_config(enable_eplb=True))
    assert "mori_high_throughput" in _support_reason(
        _ep_moe_config(all2all_backend="mori_high_throughput")
    )

    too_many_routes = _ep_moe_config()
    too_many_routes.experts_per_token = 17
    assert "top-k 17" in _support_reason(too_many_routes)

    # A fused shared expert is one extra local expert and one extra route.
    fused_shared = _ep_moe_config()
    fused_shared.num_local_experts += 1
    assert _support_reason(fused_shared) == ""
    fused_shared.experts_per_token = 16
    assert "top-k 17" in _support_reason(fused_shared)
    fused_shared.num_local_experts += 1
    assert "2 fused shared experts" in _support_reason(fused_shared)

    monkeypatch.setattr(vllm.utils.import_utils, "has_mori", lambda: False)
    assert "missing MoRI" in _support_reason(_ep_moe_config())


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


def test_layers_share_one_runtime_and_bind_their_own_weights(monkeypatch):
    built: list[dict] = []

    class FakeMegaMoEV2:
        _s1_w1: torch.Tensor
        _s1_w1_scale: torch.Tensor
        w2: torch.Tensor
        w2_scale: torch.Tensor

        def __init__(self, **kwargs):
            built.append(kwargs)

        def forward(self, x, wts, topk_ids):
            assert wts.dtype == torch.float32 and topk_ids.dtype == torch.int32
            # Depends on every bound tensor, so a stale binding changes the result.
            bound = (self._s1_w1, self._s1_w1_scale, self.w2, self.w2_scale)
            return x * sum(t.float().sum() for t in bound)

    mega_moe = ModuleType("aiter.ops.flydsl.kernels.mega_moe")
    mega_moe.MegaMoEV2 = FakeMegaMoEV2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.kernels.mega_moe", mega_moe)
    mega_moe_config = ModuleType("aiter.ops.flydsl.kernels.mega_moe.mega_moe_config")
    mega_moe_config.FIXED_SLOT_MAX_MTPR = 255  # type: ignore[attr-defined]
    mega_moe_config.build_mega_moe_bundle_plan = (  # type: ignore[attr-defined]
        lambda mtpr, **kwargs: SimpleNamespace(fixed_slot_dispatch=mtpr <= 255)
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.kernels.mega_moe.mega_moe_config",
        mega_moe_config,
    )
    monkeypatch.setattr(
        vllm.distributed,
        "get_ep_group",
        lambda: SimpleNamespace(rank_in_group=3, world_size=8, cpu_group="ep"),
    )
    monkeypatch.setattr(AiterMegaMoEExperts, "_runtime_cache", {})
    shmem_groups: list = []
    monkeypatch.setattr(all2all, "init_mori_shmem", shmem_groups.append)
    monkeypatch.setattr(mega_moe_module, "_mori_shmem_initialized", False)
    env = {k: v for k, v in os.environ.items() if k != "MORI_SHMEM_HEAP_SIZE"}
    monkeypatch.setattr(os, "environ", env)

    hidden, inter = 128, 64
    moe = _ep_moe_config(hidden=hidden, inter=inter)
    layers, experts = [], []
    for value in (1, 2):
        layer = SimpleNamespace(
            w13_weight=torch.full((48, 2 * inter, hidden // 2), value),
            w13_weight_scale=torch.full((48 * 2 * inter, hidden // 32), value),
            w2_weight=torch.full((48, hidden, inter // 2), value),
            w2_weight_scale=torch.full((48 * hidden, inter // 32), value),
            apply_router_weight_on_input=False,
        )
        quant_config = make_mxfp4_moe_quant_config(
            Mxfp4MoeBackend.AITER_MEGA_MOE,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            swiglu_limit=10.0,
        )
        assert quant_config is not None
        layer_experts = AiterMegaMoEExperts(moe, quant_config)
        layer_experts.process_weights_after_loading(layer)  # type: ignore[arg-type]
        layers.append(layer)
        experts.append(layer_experts)

    # The heap is set up once, at weight loading, so it is memory profiled.
    assert shmem_groups == ["ep"]
    assert os.environ["MORI_SHMEM_HEAP_SIZE"] == "8G"
    # One fixed-slot runtime for small steps and one for the full batch.
    assert [b["max_tok_per_rank"] for b in built] == [128, 4096]
    weight_args = ("w1", "w1_scale", "w2", "w2_scale", "max_tok_per_rank")
    config = {k: v for k, v in built[0].items() if k not in weight_args}
    assert config == dict(
        rank=3,
        world_size=8,
        model_dim=hidden,
        inter_dim=inter,
        experts=NUM_EXPERTS,
        topk=TOPK,
        quant="a8w4",
        swiglu_limit=10.0,
    )
    assert isinstance(
        experts[0].finalize_weight_and_reduce_impl(), TopKWeightAndReduceNoOP
    )
    assert experts[0].expects_unquantized_inputs

    hidden_states = torch.ones(5, 3)
    for value, layer, layer_experts in zip((1, 2), layers, experts):
        output = torch.empty_like(hidden_states)
        layer_experts.apply(
            output,
            hidden_states,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=torch.ones(5, TOPK, dtype=torch.bfloat16),
            topk_ids=torch.zeros(5, TOPK, dtype=torch.int64),
            activation=moe.activation,
            global_num_experts=NUM_EXPERTS,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=torch.empty(0),
            workspace2=torch.empty(0),
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        numel = sum(
            t.numel()
            for t in (
                layer.w13_weight,
                layer.w13_weight_scale,
                layer.w2_weight,
                layer.w2_weight_scale,
            )
        )
        torch.testing.assert_close(output, hidden_states * value * numel)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens_across_dp, expected",
    [([4, 128], "decode"), ([4, 129], "prefill")],
)
def test_runtime_selected_by_max_tokens_across_dp(
    monkeypatch, tokens_across_dp, expected
):
    experts = _make_experts(_ep_moe_config())
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
    assert experts._select_runtime(4) == expected


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
def test_fused_shared_expert_routes_to_the_local_copy():
    moe = _ep_moe_config(ep_rank=3)
    moe.num_local_experts += 1
    experts = _make_experts(moe)
    assert experts.num_fused_shared_experts == 1

    # The experts take only the routed top-k and append the shared route.
    assert AiterMegaMoEExperts.appends_fused_shared_experts
    ids = torch.randint(0, NUM_EXPERTS, (37, TOPK), device="cuda", dtype=torch.int32)
    ids[::5, 1] = -1
    weights = torch.rand(37, TOPK, device="cuda")
    out_weights, out_ids = experts._append_shared_expert(weights, ids)

    routed = torch.where(ids >= 0, ids + ids // 48, -1)
    shared = torch.full_like(ids[:, :1], 3 * 49 + 48)
    torch.testing.assert_close(out_ids, torch.cat([routed, shared], dim=1))
    torch.testing.assert_close(
        out_weights, torch.cat([weights, torch.ones_like(weights[:, :1])], dim=1)
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")
def test_load_one_expert_matches_shuffling_all_experts():
    num_experts, hidden, inter = 3, 512, 256

    def rand(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")

    raw = (
        rand(num_experts, 2 * inter, hidden // 2),
        rand(num_experts, hidden, inter // 2),
        rand(num_experts, 2 * inter, hidden // 32),
        rand(num_experts, hidden, inter // 32),
    )
    expected = shuffle_aiter_mega_moe_weights(*raw)

    missing_last = [t.clone() for t in raw]
    for t in missing_last:
        t[-1] = 0
    layer = SimpleNamespace(
        **dict(
            zip(
                ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
                shuffle_aiter_mega_moe_weights(*missing_last),
            )
        )
    )
    load_aiter_mega_moe_expert(layer, num_experts - 1, *(t[-1] for t in raw))

    torch.testing.assert_close(layer.w13_weight, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(layer.w2_weight, expected[1], rtol=0, atol=0)
    torch.testing.assert_close(layer.w13_weight_scale, expected[2], rtol=0, atol=0)
    torch.testing.assert_close(layer.w2_weight_scale, expected[3], rtol=0, atol=0)
