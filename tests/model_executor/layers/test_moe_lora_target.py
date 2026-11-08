# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config import LoRAConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
    TrtLlmMxfp4LoRAExperts,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    _requested_kernel_cls,
    backend_to_kernel_cls,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.model_executor.model_loader.utils import configure_quant_config

_LAYER_NAME = "model.layers.3.mlp.experts"


def _lora_config(target_modules):
    return LoRAConfig(max_lora_rank=8, target_modules=target_modules)


@pytest.mark.parametrize(
    ("lora_config", "packed_modules_mapping", "expected"),
    [
        pytest.param(None, {}, False, id="lora-disabled"),
        pytest.param(_lora_config(None), {}, True, id="unrestricted"),
        pytest.param(
            _lora_config(["q_proj", "k_proj", "v_proj", "o_proj"]),
            {},
            False,
            id="attention-only",
        ),
        pytest.param(_lora_config(["experts"]), {}, True, id="module-suffix"),
        pytest.param(
            _lora_config(["w1"]),
            {"experts": ["w1", "w2"]},
            True,
            id="packed-parent-child-target",
        ),
        # Dynamic expert mappings require the instantiated model. Until then,
        # targets that can reach their children must retain the LoRA opt-in.
        pytest.param(
            _lora_config(["experts.0.gate_proj"]),
            {},
            True,
            id="potentially-dynamic-expert-target",
        ),
        pytest.param(
            _lora_config(["w1"]),
            {
                "gate_up_proj": ["w1", "w3"],
                "fused_wqa_wkv": ["wq_a", "wkv"],
                "fused_wkv_wgate": ["wkv", "wgate"],
            },
            False,
            id="unrelated-packed-mapping",
        ),
    ],
)
@pytest.mark.skip_global_cleanup
def test_moe_lora_target_selection(lora_config, packed_modules_mapping, expected):
    from vllm.model_executor.layers.fused_moe.layer import _is_moe_lora_target

    model_cls = type(
        "_TestModel", (), {"packed_modules_mapping": packed_modules_mapping}
    )
    quant_config = Mxfp4Config()
    configure_quant_config(quant_config, model_cls)
    moe_config = replace(
        make_dummy_moe_config(),
        is_lora_enabled=_is_moe_lora_target(
            lora_config, _LAYER_NAME, quant_config.packed_modules_mapping
        ),
    )

    assert moe_config.is_lora_enabled is expected

    backend = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16
    candidates = _requested_kernel_cls(backend, moe_config)
    if expected:
        assert candidates[0] is TrtLlmMxfp4LoRAExperts
    else:
        assert candidates == backend_to_kernel_cls(backend)


@pytest.mark.skip_global_cleanup
def test_dynamic_expert_target_is_scoped_to_expert_module():
    from vllm.model_executor.layers.fused_moe.layer import _is_moe_lora_target

    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.Module()
    model.mlp.shared_experts = torch.nn.Module()
    lora_config = _lora_config(["experts.0.gate_proj"])
    decisions = {
        name: _is_moe_lora_target(lora_config, name, {})
        for name, _ in model.named_modules()
    }
    assert decisions["mlp.experts"] is True
    assert decisions["mlp.shared_experts"] is False


def test_attention_only_lora_keeps_native_moe_backend(dist_init):
    vllm_config = VllmConfig()
    vllm_config.lora_config = _lora_config(["q_proj", "k_proj", "v_proj", "o_proj"])

    with set_current_vllm_config(vllm_config):
        moe = FusedMoEFactory(
            num_experts=4,
            top_k=2,
            hidden_size=64,
            intermediate_size=128,
            params_dtype=torch.bfloat16,
            prefix=_LAYER_NAME,
        )

    assert not moe.moe_config.is_lora_enabled
