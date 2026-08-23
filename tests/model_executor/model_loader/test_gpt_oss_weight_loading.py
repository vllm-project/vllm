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
