# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.gpt_oss import (
    GptOssForCausalLM,
    GptOssModel,
    GptOssRoutedExperts,
)


class _TestExperts(GptOssRoutedExperts):
    def __init__(
        self,
        param_name: str,
        shape: tuple[int, ...],
        *,
        tp_rank: int = 0,
        quant_method_name: str = "TestQuantMethod",
        weight_dtype: str = "",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.moe_config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(tp_size=2, tp_rank=tp_rank)
        )
        self.quant_method = type(
            quant_method_name, (), {"weight_dtype": weight_dtype}
        )()

        param = torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        param.weight_loader = self.weight_loader
        self.register_parameter(param_name, param)

    @staticmethod
    def _map_global_expert_id_to_local_expert_id(expert_id: int) -> int:
        return {3: 0, 7: 1}.get(expert_id, -1)


def _load(
    experts: _TestExperts,
    param_name: str,
    loaded_weight: torch.Tensor,
    shard_id: str,
    expert_id: int = 3,
) -> bool:
    param = getattr(experts, param_name)
    return param.weight_loader(
        param,
        loaded_weight,
        weight_name=f"model.layers.0.mlp.experts.routed_experts.{param_name}",
        shard_id=shard_id,
        expert_id=expert_id,
        return_success=True,
    )


@pytest.mark.parametrize("base_layer_prefix", ["", "base_layer."])
def test_streamed_expert_resolves_wrapped_parameter(base_layer_prefix: str) -> None:
    experts = _TestExperts("w13_weight", (2, 4, 3))
    checkpoint_name = "model.layers.2.mlp.experts.7.gate_up_proj"
    mapped_name = GptOssForCausalLM.hf_to_vllm_mapper.apply_list([checkpoint_name])[0]
    fused_name = (
        f"model.layers.2.mlp.experts.{base_layer_prefix}routed_experts.w13_weight"
    )
    params_dict = {fused_name: experts.w13_weight}
    loaded_params: set[str] = set()
    loaded_weight = torch.arange(24, dtype=torch.float32).reshape(3, 8)

    assert GptOssModel._get_streamed_expert_info(mapped_name, params_dict) == (
        7,
        fused_name,
        "gpt_oss_w13",
    )
    assert GptOssModel._try_load_streamed_expert(
        mapped_name, loaded_weight, params_dict, loaded_params
    )
    assert loaded_params == {fused_name}
    assert torch.equal(experts.w13_weight[1], loaded_weight[:, :4].t())
    assert torch.count_nonzero(experts.w13_weight[0]) == 0


def test_streamed_expert_skips_non_local_expert() -> None:
    experts = _TestExperts("w2_bias", (2, 3))

    assert not _load(experts, "w2_bias", torch.ones(3), "gpt_oss_w2", expert_id=6)
    assert torch.count_nonzero(experts.w2_bias) == 0


@pytest.mark.parametrize("param_name", ["w13_weight", "w2_weight"])
def test_streamed_unquantized_expert_tp_slice(param_name: str) -> None:
    if param_name == "w13_weight":
        experts = _TestExperts(param_name, (2, 4, 3), tp_rank=1)
        loaded_weight = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        shard_id = "gpt_oss_w13"
        expected = loaded_weight[:, 4:].t()
    else:
        experts = _TestExperts(param_name, (2, 3, 2), tp_rank=1)
        loaded_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        shard_id = "gpt_oss_w2"
        expected = loaded_weight[2:].t()

    assert _load(experts, param_name, loaded_weight, shard_id)
    assert torch.equal(getattr(experts, param_name)[0], expected)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_streamed_down_bias_owned_by_tp_rank_zero(tp_rank: int) -> None:
    experts = _TestExperts("w2_bias", (2, 3), tp_rank=tp_rank)
    loaded_weight = torch.arange(1, 4, dtype=torch.float32)

    assert _load(experts, "w2_bias", loaded_weight, "gpt_oss_w2")
    expected = loaded_weight if tp_rank == 0 else torch.zeros_like(loaded_weight)
    assert torch.equal(experts.w2_bias[0], expected)


def test_streamed_gate_up_bias_tp_slice() -> None:
    experts = _TestExperts("w13_bias", (2, 4), tp_rank=1)
    loaded_weight = torch.arange(8, dtype=torch.float32)

    assert _load(experts, "w13_bias", loaded_weight, "gpt_oss_w13")
    assert torch.equal(experts.w13_bias[0], loaded_weight[4:])


def _model_with_experts(monkeypatch, tp_rank: int, params: dict, weight_dtype=None):
    """A GptOssModel holding only routed experts with ``params`` (name ->
    shape), on TP rank ``tp_rank`` of 2, whose loaders can be driven."""
    from vllm.model_executor.models import gpt_oss

    group = SimpleNamespace(world_size=1, rank_in_group=0)
    for name in ("get_dp_group", "get_ep_group", "get_pcp_group"):
        monkeypatch.setattr(gpt_oss, name, lambda: group)
    monkeypatch.setattr(gpt_oss, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(gpt_oss, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.config.get_tensor_model_parallel_rank",
        lambda: tp_rank,
    )
    monkeypatch.setattr(gpt_oss, "is_pp_missing_parameter", lambda *args: False)

    (first, first_shape), *rest = params.items()
    experts = _TestExperts(first, first_shape, tp_rank=tp_rank)
    for name, shape in rest:
        param = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
        param.weight_loader = experts.weight_loader
        experts.register_parameter(name, param)

    model = GptOssModel.__new__(GptOssModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_local_experts=2, intermediate_size=4)
    model.parallel_config = SimpleNamespace(enable_expert_parallel=False)
    model.get_expert_mapping = lambda: []  # type: ignore[method-assign]
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = torch.nn.Module()
    layer.mlp.experts.routed_experts = experts
    layer.mlp.experts._quant_method = SimpleNamespace(weight_dtype=weight_dtype)
    model.layers = torch.nn.ModuleList([layer])
    return model, experts


def _reload(model: torch.nn.Module, load, checkpoint: dict) -> None:
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    load(checkpoint)
    finalize_layerwise_reload(model, SimpleNamespace(dtype=torch.float32))


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_fused_bf16_experts_reload(monkeypatch, tp_rank: int) -> None:
    """Fused (BF16) expert tensors are written through the params' loader, so
    a layerwise reload updates them in place, and the down bias zeroed on TP
    ranks != 0 leaves the caller's tensor (a trainer's IPC buffer) untouched."""
    model, experts = _model_with_experts(
        monkeypatch,
        tp_rank,
        {
            "w13_weight": (2, 4, 3),
            "w2_weight": (2, 3, 2),
            "w13_bias": (2, 4),
            "w2_bias": (2, 3),
        },
    )
    prefix = "layers.0.mlp.experts."

    def checkpoint(offset: float) -> dict[str, torch.Tensor]:
        shapes = {
            "w13_weight": (2, 3, 8),
            "w2_weight": (2, 4, 3),
            "w13_bias": (2, 8),
            "w2_bias": (2, 3),
        }
        return {
            name: torch.arange(torch.Size(shape).numel(), dtype=torch.float32)
            .reshape(shape)
            .add(offset)
            for name, shape in shapes.items()
        }

    def load(ckpt: dict[str, torch.Tensor]) -> None:
        weights = [(prefix + name, t) for name, t in ckpt.items()]
        model._load_weights_other(2, 0, weights, [])

    def expected(ckpt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        i = slice(2 * tp_rank, 2 * tp_rank + 2)  # this rank's intermediate rows
        i2 = slice(4 * tp_rank, 4 * tp_rank + 4)  # gate/up interleaved
        return {
            "w13_weight": ckpt["w13_weight"][:, :, i2].permute(0, 2, 1),
            "w2_weight": ckpt["w2_weight"][:, i, :].permute(0, 2, 1),
            "w13_bias": ckpt["w13_bias"][:, i2],
            "w2_bias": ckpt["w2_bias"] * (tp_rank == 0),
        }

    for offset, update in ((0, False), (100, True)):
        ckpt = checkpoint(offset)
        if update:
            _reload(model, load, ckpt)
        else:
            load(ckpt)
            kernel = dict(experts.named_parameters())
        for name, t in ckpt.items():
            assert torch.equal(t, checkpoint(offset)[name]), f"{name} was modified"
        for name, value in expected(ckpt).items():
            assert getattr(experts, name) is kernel[name]
            assert torch.equal(kernel[name], value), name


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_quark_per_expert_reload(monkeypatch, tp_rank: int) -> None:
    """Quark per-expert tensors (FP8 weights, input scales, biases into padded
    params) are written through the params' loader, so a layerwise reload
    updates them in place."""
    model, experts = _model_with_experts(
        monkeypatch,
        tp_rank,
        {
            "w13_weight": (2, 4, 3),
            "w2_weight": (2, 3, 2),
            "w13_input_scale": (2,),
            "w2_bias": (2, 5),  # padded past the checkpoint's 3
        },
        weight_dtype="fp8",
    )

    def checkpoint(offset: float) -> list[tuple[str, torch.Tensor]]:
        weights = []
        for e in range(2):
            prefix = f"layers.0.mlp.experts.{e}."
            base = offset + 10 * e
            weights += [
                (prefix + "w13_weight", torch.arange(24.0).reshape(8, 3) + base),
                (prefix + "w2_weight", torch.arange(12.0).reshape(3, 4) + base),
                (prefix + "w13_input_scale", torch.tensor(base + 0.5)),
                (prefix + "w2_bias", torch.arange(3.0) + base),
            ]
        return weights

    def load(weights: list[tuple[str, torch.Tensor]]) -> None:
        model._load_weights_quark(2, 0, weights, [])

    def check(offset: float) -> None:
        for e in range(2):
            base = offset + 10 * e
            w13 = torch.arange(24.0).reshape(8, 3) + base
            w2 = torch.arange(12.0).reshape(3, 4) + base
            i = slice(2 * tp_rank, 2 * tp_rank + 2)
            i2 = slice(4 * tp_rank, 4 * tp_rank + 4)
            assert torch.equal(experts.w13_weight[e], w13[i2])
            assert torch.equal(experts.w2_weight[e], w2[:, i])
            assert experts.w13_input_scale[e] == base + 0.5
            bias = (torch.arange(3.0) + base) * (tp_rank == 0)
            assert torch.equal(experts.w2_bias[e, :3], bias)

    for offset, update in ((0, False), (100, True)):
        ckpt = checkpoint(offset)
        if update:
            _reload(model, load, ckpt)
        else:
            load(ckpt)
            kernel = dict(experts.named_parameters())
        for (name, t), (_, sent) in zip(ckpt, checkpoint(offset)):
            assert torch.equal(t, sent), f"{name} was modified"
        check(offset)
        for name, param in kernel.items():
            assert getattr(experts, name) is param


@pytest.mark.parametrize("case", ["mxfp4", "nvfp4"])
def test_streamed_quantized_expert_layout(case: str) -> None:
    if case == "mxfp4":
        experts = _TestExperts(
            "w13_weight",
            (2, 4, 4),
            tp_rank=1,
            weight_dtype="gpt_oss_mxfp4",
            dtype=torch.uint8,
        )
        loaded_weight = torch.arange(32, dtype=torch.uint8).reshape(8, 2, 2)
        expected = loaded_weight.reshape(8, 4)[4:]
    else:
        experts = _TestExperts(
            "w13_weight",
            (2, 4, 2),
            tp_rank=1,
            quant_method_name="Nvfp4OnlineMoEMethod",
            dtype=torch.uint8,
        )
        loaded_weight = torch.arange(16, dtype=torch.uint8).reshape(8, 2)
        expected = loaded_weight[4:]

    assert _load(experts, "w13_weight", loaded_weight, "gpt_oss_w13")
    assert torch.equal(experts.w13_weight[0], expected)
