# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.quantization import auto_gptq
from vllm.model_executor.models import qwen3_next


class DummyQuantMethod:
    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **kwargs,
    ):
        qweight = nn.Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("qweight", qweight)

    def apply(self, layer, x, bias=None):
        raise NotImplementedError


class DummyQuantConfig:
    online_quantization_config = None

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_quant_method(self, layer, prefix):
        return DummyQuantMethod()


@pytest.mark.parametrize(
    ("quant_name", "gate_is_quantized"),
    [
        ("inc", True),
        ("modelopt_mixed", True),
        ("modelopt_fp4", False),
        ("auto_gptq", True),
        ("auto_gptq", False),
    ],
)
def test_qwen3_next_moe_gate_quantization(
    monkeypatch,
    dist_init,
    quant_name,
    gate_is_quantized,
):
    class DummyExperts(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(qwen3_next, "FusedMoEFactory", DummyExperts)
    monkeypatch.setattr(
        qwen3_next,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        qwen3_next,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: 1)),
    )

    config = SimpleNamespace(
        hidden_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=0,
    )
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=False,
        enable_expert_parallel=False,
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
    )
    if quant_name == "auto_gptq":

        class DummyGPTQMethod(DummyQuantMethod):
            def __init__(self, quant_config):
                self.quant_config = quant_config

        monkeypatch.setattr(
            auto_gptq,
            "AutoGPTQLinearMethod",
            DummyGPTQMethod,
        )
        quant_config = auto_gptq.AutoGPTQConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "desc_act": False,
                "sym": True,
                "modules_in_block_to_quantize": (
                    ["mlp.gate"] if gate_is_quantized else ["self_attn.q_proj"]
                ),
            }
        )
    else:
        quant_config = DummyQuantConfig(quant_name)

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        parallel_config=parallel_config,
        quant_config=quant_config,
    )

    block = qwen3_next.Qwen3NextSparseMoeBlock(
        vllm_config,
        prefix="model.layers.0.mlp",
    )

    assert isinstance(block.gate, qwen3_next.GateLinear)
    assert block.gate.prefix == "model.layers.0.mlp.gate"

    if quant_name == "modelopt_fp4":
        assert block.gate.quant_config is None
    else:
        assert block.gate.quant_config is quant_config

    if gate_is_quantized:
        assert hasattr(block.gate, "qweight")
        assert not hasattr(block.gate, "weight")
    else:
        assert hasattr(block.gate, "weight")
        assert not hasattr(block.gate, "qweight")
