# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA (DeepSeek-V3.2, GLM-5.x) weight loading.

`DeepseekV32ForCausalLM` loads through `AutoWeightsLoader`: per-expert and
Transformers v5 stacked routed experts must load alike, also in a load plan.

Reordering weight loaders (e.g. `runai_streamer`) yield tensors in
IO-completion order, so a layer's FP8 `indexer.wk` weight /
`weight_scale_inv` pair can be split across `load_weights()` calls. The
pending pair buffer must therefore survive across calls, or one half of the
pair is silently dropped and the WK shard of the fused `wk_weights_proj`
parameter keeps its initialization.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.distributed.weight_transfer.base import ParamMeta
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.fused_moe import (
    unquantized_fused_moe_method,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.reload import (
    lifecycle,
    plan_loads,
    start_reload,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.platforms import current_platform

if current_platform.is_cuda():
    from vllm.models.deepseek_v32.nvidia import model as deepseek_v32_model
    from vllm.models.deepseek_v32.nvidia import mtp as deepseek_v32_mtp
else:
    # The nvidia module chain is not importable on other platforms; the
    # shared model-executor CI job also collects this file on ROCm mirrors.
    deepseek_v32_model = None
    deepseek_v32_mtp = None

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.skipif(
        not current_platform.is_cuda(), reason="Tests the CUDA deepseek_v32 loader"
    ),
]

WK_ROWS, WP_ROWS, IN_FEATURES = 4, 6, 8

MODEL_WK = "model.layers.1.self_attn.indexer.wk.weight"
MODEL_SCALE = "model.layers.1.self_attn.indexer.wk.weight_scale_inv"
MODEL_WP = "model.layers.1.self_attn.indexer.weights_proj.weight"
MTP_WK = "model.layers.2.self_attn.indexer.wk.weight"
MTP_SCALE = "model.layers.2.self_attn.indexer.wk.weight_scale_inv"
MTP_WP = "model.layers.2.self_attn.indexer.weights_proj.weight"


def _fused_weight_loader(param, loaded_weight, shard_id):
    rows = loaded_weight.shape[0]
    if shard_id == 0:
        param.data[:rows] = loaded_weight
    else:
        param.data[WK_ROWS : WK_ROWS + rows] = loaded_weight


if current_platform.is_cuda():
    # The stubs subclass the CUDA-only modules and must only be defined on
    # CUDA: pytest evaluates class bodies at collection time, before the
    # skipif above can skip the tests, so a plain definition would crash
    # collection on the ROCm mirrors where the modules are None.
    class _StubMTPWeights(deepseek_v32_mtp.DeepseekV32MTP):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = SimpleNamespace(
                n_routed_experts=0, num_hidden_layers=2, num_nextn_predict_layers=1
            )
            self.model = SimpleNamespace(mtp_start_layer_idx=2, num_mtp_layers=1)
            self.weight = nn.Parameter(
                torch.zeros(WK_ROWS + WP_ROWS, IN_FEATURES, dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.weight.weight_loader = _fused_weight_loader

        def named_parameters(self, *, remove_duplicate=True, **kwargs):
            yield (
                "model.layers.2.mtp_block.self_attn.indexer.wk_weights_proj.weight",
                self.weight,
            )


def _wk_pair(scale_layout: str = "scalar"):
    """FP8 wk weight + scale with distinct values per quantization group,
    so a wrong scale broadcast produces a different expected tensor."""
    weight = (
        torch.arange(WK_ROWS * IN_FEATURES, dtype=torch.float32)
        .reshape(WK_ROWS, IN_FEATURES)
        .to(torch.float8_e4m3fn)
    )
    if scale_layout == "scalar":
        scale = torch.full((1, 1), 0.5, dtype=torch.float32)
    elif scale_layout == "block":
        # Same layout as the GLM-5.3 checkpoint: one scale per column block.
        scale = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    elif scale_layout == "per_channel":
        scale = torch.tensor([0.5, 0.25, 0.75, 0.125], dtype=torch.float32)
    else:
        raise ValueError(scale_layout)
    return weight, scale, _dequant_expected(weight, scale)


def _dequant_expected(weight, scale):
    """Dequantized bf16 values computed from first principles (value * scale
    broadcast over its quantization group), independently of the loader."""
    if scale.ndim == 1:
        broadcast = scale.view(-1, 1)
    elif scale.shape == (1, 1):
        broadcast = scale
    else:
        g0 = WK_ROWS // scale.shape[0]
        g1 = IN_FEATURES // scale.shape[1]
        broadcast = scale.repeat_interleave(g0, 0).repeat_interleave(g1, 1)
    return (weight.to(torch.float32) * broadcast).to(torch.bfloat16)


@pytest.mark.parametrize("scale_layout", ["scalar", "block", "per_channel"])
@pytest.mark.parametrize("split_half_first", ["weight", "scale"])
def test_mtp_fuses_fp8_wk_pair_split_across_load_weights_calls(
    split_half_first,
    scale_layout,
):
    weight, scale, expected = _wk_pair(scale_layout)
    wp = torch.full((WP_ROWS, IN_FEATURES), 2.0, dtype=torch.bfloat16)
    model = _StubMTPWeights()

    halves = {
        "weight": (MTP_WK, weight),
        "scale": (MTP_SCALE, scale),
    }
    first_half = halves.pop(split_half_first)
    second_half = halves.popitem()[1]

    model.load_weights(iter([(MTP_WP, wp), first_half]))
    assert torch.equal(
        model.weight.data[:WK_ROWS],
        torch.zeros_like(expected),
    ), "WK shard must not be written before both halves of the pair arrived"

    model.load_weights(iter([second_half]))
    torch.testing.assert_close(model.weight.data[:WK_ROWS], expected)
    torch.testing.assert_close(model.weight.data[WK_ROWS:], wp)


# ------------------------------------------- DeepseekV32ForCausalLM.load_weights

BF16 = torch.bfloat16
H, Q_LORA, KV_A, INTER, E = IN_FEATURES, 4, 6, 6, 4
DENSE, MOE, MTP = "model.layers.0", "model.layers.1", "model.layers.2"


def _container(**children: nn.Module) -> nn.Module:
    module = nn.Module()
    for name, child in children.items():
        setattr(module, name, child)
    return module


def _merged(*sizes: int) -> MergedColumnParallelLinear:
    return MergedColumnParallelLinear(H, list(sizes), bias=False, params_dtype=BF16)


def _mlp() -> nn.Module:
    down_proj = RowParallelLinear(INTER, H, bias=False, params_dtype=BF16)
    return _container(gate_up_proj=_merged(INTER, INTER), down_proj=down_proj)


class _MoERunner(nn.Module):
    load_weights = MoERunner.load_weights


def _routed_experts(prefix: str) -> RoutedExperts:
    moe_config = FusedMoEConfig(
        num_experts=E,
        experts_per_token=2,
        hidden_dim=H,
        intermediate_size=INTER,
        num_local_experts=E,
        num_logical_experts=E,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=BF16,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=16,
    )
    expert_map_manager = SimpleNamespace(
        local_num_experts=E,
        num_fused_shared_experts=0,
        placement_strategy="linear",
        expert_map=None,
        expert_mask=None,
        routing_tables=None,
        map_global_to_local=lambda index: index,
    )
    return RoutedExperts(
        prefix, BF16, moe_config, None, expert_map_manager=expert_map_manager
    )


def _layer(moe: bool) -> nn.Module:
    indexer = _container(wk_weights_proj=_merged(WK_ROWS, WP_ROWS))
    self_attn = _container(fused_qkv_a_proj=_merged(Q_LORA, KV_A), indexer=indexer)
    mlp = _mlp()
    if moe:
        experts = _MoERunner()
        experts.routed_experts = _routed_experts(f"{MOE}.mlp.experts")
        gate = _container()
        gate.weight = nn.Parameter(torch.zeros(E, H, dtype=BF16), requires_grad=False)
        mlp = _container(gate=gate, shared_experts=_mlp(), experts=experts)
    return _container(self_attn=self_attn, mlp=mlp)


@pytest.fixture
def dsa_causal_lm(default_vllm_config, monkeypatch):
    """`DeepseekV32ForCausalLM` over a dense layer, an MoE layer and one MTP
    layer that the checkpoint has but the model leaves to the drafter."""
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        unquantized_fused_moe_method,
        "select_unquantized_moe_backend",
        lambda **kwargs: (None, None),
    )

    def init(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            model_type="deepseek_v32", num_hidden_layers=2, num_nextn_predict_layers=1
        )
        self.model = _container(layers=nn.ModuleList([_layer(False), _layer(True)]))

    monkeypatch.setattr(deepseek_v32_model.DeepseekV2ForCausalLM, "__init__", init)
    return lambda: deepseek_v32_model.DeepseekV32ForCausalLM(vllm_config=None)


def _checkpoint(fp8_wk: bool = True) -> dict[str, torch.Tensor]:
    """Hub names with per-expert routed experts. The MoE layer's indexer `wk`
    is FP8 with block scales, as in GLM-5.x FP8, if `fp8_wk`."""
    gen = torch.Generator().manual_seed(0)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=gen).to(BF16)

    def mlp(prefix: str) -> dict[str, torch.Tensor]:
        return {
            f"{prefix}.gate_proj.weight": rand(INTER, H),
            f"{prefix}.up_proj.weight": rand(INTER, H),
            f"{prefix}.down_proj.weight": rand(H, INTER),
        }

    checkpoint: dict[str, torch.Tensor] = {}
    for layer in (DENSE, MOE, MTP):
        attn = f"{layer}.self_attn"
        checkpoint[f"{attn}.q_a_proj.weight"] = rand(Q_LORA, H)
        checkpoint[f"{attn}.kv_a_proj_with_mqa.weight"] = rand(KV_A, H)
        checkpoint[f"{attn}.indexer.weights_proj.weight"] = rand(WP_ROWS, H)
        if layer == DENSE or not fp8_wk:
            checkpoint[f"{attn}.indexer.wk.weight"] = rand(WK_ROWS, H)
        else:
            weight, scale, _ = _wk_pair("block")
            checkpoint[f"{attn}.indexer.wk.weight"] = weight
            checkpoint[f"{attn}.indexer.wk.weight_scale_inv"] = scale
        if layer == DENSE:
            checkpoint |= mlp(f"{layer}.mlp")
            continue
        checkpoint[f"{layer}.mlp.gate.weight"] = rand(E, H)
        checkpoint |= mlp(f"{layer}.mlp.shared_experts")
        for e in range(E):
            checkpoint |= mlp(f"{layer}.mlp.experts.{e}")
    checkpoint[f"{MTP}.eh_proj.weight"] = rand(H, 2 * H)
    return checkpoint


def _stack_experts(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The Transformers v5 layout RL trainers send: `[E, 2I, H]` gate rows
    before up, and `[E, H, I]`."""

    def stack(*projs: str) -> torch.Tensor:
        experts = f"{MOE}.mlp.experts"
        return torch.stack(
            [
                torch.cat([checkpoint[f"{experts}.{e}.{p}.weight"] for p in projs])
                for e in range(E)
            ]
        )

    stacked = {
        name: tensor
        for name, tensor in checkpoint.items()
        if not name.startswith(f"{MOE}.mlp.experts.")
    }
    stacked[f"{MOE}.mlp.experts.gate_up_proj"] = stack("gate_proj", "up_proj")
    stacked[f"{MOE}.mlp.experts.down_proj"] = stack("down_proj")
    return stacked


def test_per_expert_and_stacked_experts_load_alike(dsa_causal_lm):
    """Hub per-expert names and Transformers v5 stacked experts give the same
    parameters, every parameter is loaded, and MTP tensors are left out."""
    per_expert = _checkpoint()
    models = {}
    for layout, checkpoint in (
        ("per_expert", per_expert),
        ("stacked", _stack_experts(per_expert)),
    ):
        model = dsa_causal_lm()
        loaded = model.load_weights(iter(checkpoint.items()))
        assert loaded == {name for name, _ in model.named_parameters()}, layout
        assert not model._pending_indexer_wk_fp8
        models[layout] = model
    for (name, a), (_, b) in zip(
        models["per_expert"].named_parameters(), models["stacked"].named_parameters()
    ):
        assert torch.equal(a, b), name

    # Equal, and in the right place.
    dense, moe = models["stacked"].model.layers
    gate_up = dense.mlp.gate_up_proj.weight
    assert torch.equal(gate_up[INTER:], per_expert[f"{DENSE}.mlp.up_proj.weight"])
    w13 = moe.mlp.experts.routed_experts.w13_weight
    w2 = moe.mlp.experts.routed_experts.w2_weight
    experts = f"{MOE}.mlp.experts"
    for e in range(E):
        assert torch.equal(
            w13[e, :INTER], per_expert[f"{experts}.{e}.gate_proj.weight"]
        )
        assert torch.equal(w13[e, INTER:], per_expert[f"{experts}.{e}.up_proj.weight"])
        assert torch.equal(w2[e], per_expert[f"{experts}.{e}.down_proj.weight"])
    qkv_a = moe.self_attn.fused_qkv_a_proj.weight
    kv_a = per_expert[f"{MOE}.self_attn.kv_a_proj_with_mqa.weight"]
    assert torch.equal(qkv_a[Q_LORA:], kv_a)
    wk_weights = moe.self_attn.indexer.wk_weights_proj.weight
    torch.testing.assert_close(wk_weights[:WK_ROWS], _wk_pair("block")[2])
    wp = per_expert[f"{MOE}.self_attn.indexer.weights_proj.weight"]
    assert torch.equal(wk_weights[WK_ROWS:], wp)


@pytest.mark.parametrize("split_half_first", ["weight", "scale"])
def test_causal_lm_fuses_fp8_wk_pair_split_across_load_weights_calls(
    dsa_causal_lm, split_half_first
):
    checkpoint = _checkpoint()
    halves = {"weight": MODEL_WK, "scale": MODEL_SCALE}
    halves.pop(split_half_first)
    (second_name,) = halves.values()
    second_half = (second_name, checkpoint.pop(second_name))
    model = dsa_causal_lm()
    wk_weights = model.model.layers[1].self_attn.indexer.wk_weights_proj.weight
    wk_weights.data.zero_()

    model.load_weights(iter(checkpoint.items()))
    assert not wk_weights[:WK_ROWS].any(), (
        "WK shard must not be written before both halves of the pair arrived"
    )
    model.load_weights(iter([second_half]))
    torch.testing.assert_close(wk_weights[:WK_ROWS], _wk_pair("block")[2])
    assert not model._pending_indexer_wk_fp8


def test_load_plan_takes_stacked_experts_directly(dsa_causal_lm, monkeypatch):
    """A direct reload's load plan copies Transformers v5 stacked experts
    straight into the expert weights instead of falling back."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    monkeypatch.setattr(lifecycle, "_weight_update_active", False)
    # RL trainers send the indexer's `wk` in BF16.
    checkpoint = _stack_experts(_checkpoint(fp8_wk=False))
    model = dsa_causal_lm()
    # No MoE kernel on CPU; the plan only needs the expert weights' layout.
    quant_method = model.model.layers[1].mlp.experts.routed_experts.quant_method
    monkeypatch.setattr(quant_method, "process_weights_after_loading", lambda _: None)
    model.load_weights(iter(checkpoint.items()))
    model_config = SimpleNamespace(
        word_embeddings_untied_by_checkpoint=False, quantization=None, dtype=BF16
    )
    process_weights_after_loading(model, model_config, torch.device("cpu"))

    start_reload(model)
    plan = plan_loads(
        model,
        [ParamMeta(name, t.dtype, tuple(t.shape)) for name, t in checkpoint.items()],
    )
    for proj in ("gate_up_proj", "down_proj"):
        assert plan.get_dest(f"{MOE}.mlp.experts.{proj}") is not None, proj
