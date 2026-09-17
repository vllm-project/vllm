# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm.model_executor.models.config import (
    Qwen3_5ForConditionalGenerationConfig,
    Qwen4ExpForConditionalGenerationConfig,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.models.qwen4_exp.cpu import runtime as cpu_runtime
from vllm.models.qwen4_exp.cpu.model import (
    Qwen4ExpSparseMoeBlock,
    _pad_moe_checkpoint_weights,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig


def _text_config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        num_experts=0,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[1],
        mtp_num_hidden_layers=1,
        mtp={"hybrid": True},
    )


def _vllm_config(restriction: str | None = None) -> SimpleNamespace:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=_text_config(),
            hf_config=_text_config(),
            multimodal_config=SimpleNamespace(language_model_only=True),
            enforce_eager=False,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            enable_dbo=False,
            ubatch_size=1,
        ),
        speculative_config=None,
        lora_config=None,
        use_v2_model_runner=restriction != "model_runner",
    )
    if restriction == "multimodal":
        config.model_config.multimodal_config.language_model_only = False
    return config


@pytest.mark.parametrize(
    ("restriction", "error", "message"),
    [
        ("architecture", NotImplementedError, "x86-64"),
        ("multimodal", NotImplementedError, "text-only"),
        ("model_runner", ValueError, "Model Runner V2"),
        ("triton", ValueError, "active CPU backend"),
    ],
)
def test_qwen4_exp_cpu_rejects_unsupported_runtime(
    restriction: str,
    error: type[Exception],
    message: str,
) -> None:
    cpu_arch = CpuArchEnum.ARM if restriction == "architecture" else CpuArchEnum.X86
    with (
        patch.object(
            Qwen3_5ForConditionalGenerationConfig,
            "verify_and_update_config",
        ),
        patch.object(current_platform, "is_cpu", return_value=True),
        patch.object(
            current_platform,
            "get_cpu_architecture",
            return_value=cpu_arch,
        ),
        patch.object(
            cpu_runtime,
            "has_active_triton_cpu_backend",
            return_value=restriction != "triton",
        ),
        pytest.raises(error, match=message),
    ):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
            _vllm_config(restriction)
        )


def _block_fp8_config() -> SimpleNamespace:
    return SimpleNamespace(
        get_name=lambda: "fp8",
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
    )


def _moe_vllm_config(
    tp_size: int,
    *,
    fp8_checkpoint: bool = True,
) -> SimpleNamespace:
    text_config = SimpleNamespace(
        moe_intermediate_size=640,
        shared_expert_intermediate_size=0,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text_config),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            use_sequence_parallel_moe=False,
        ),
        quant_config=_block_fp8_config() if fp8_checkpoint else None,
    )


@pytest.mark.parametrize(
    ("fp8_checkpoint", "expected_intermediate_size"),
    [
        pytest.param(False, 640, id="bf16"),
        pytest.param(True, 768, id="fp8"),
    ],
)
def test_sparse_moe_constructs_with_expected_intermediate_size(
    fp8_checkpoint: bool,
    expected_intermediate_size: int,
) -> None:
    vllm_config = _moe_vllm_config(
        tp_size=2,
        fp8_checkpoint=fp8_checkpoint,
    )
    constructed: dict[str, int] = {}

    def init_base(
        layer: Qwen3NextSparseMoeBlock,
        vllm_config: SimpleNamespace,
        prefix: str,
    ) -> None:
        nn.Module.__init__(layer)
        constructed["intermediate_size"] = (
            vllm_config.model_config.hf_text_config.moe_intermediate_size
        )
        layer.experts = SimpleNamespace(moe_config=SimpleNamespace())

    with patch.object(Qwen3NextSparseMoeBlock, "__init__", init_base):
        layer = Qwen4ExpSparseMoeBlock(vllm_config)

    assert constructed["intermediate_size"] == expected_intermediate_size
    assert vllm_config.model_config.hf_text_config.moe_intermediate_size == 640
    assert layer.original_intermediate_size_per_partition == 320
    if fp8_checkpoint:
        assert layer.experts.moe_config.intermediate_size_per_partition_unpadded == 320
    else:
        assert not hasattr(
            layer.experts.moe_config,
            "intermediate_size_per_partition_unpadded",
        )


def test_block_fp8_moe_checkpoint_padding_is_aligned_and_idempotent() -> None:
    vllm_config = _moe_vllm_config(tp_size=2)
    weights = [
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            torch.ones(640, 2),
        ),
        (
            "model.layers.0.mlp.experts.0.up_proj.weight_scale_inv",
            torch.ones(5, 1),
        ),
        (
            "model.layers.0.mlp.experts.0.down_proj.weight",
            torch.ones(2, 640),
        ),
        (
            "model.layers.0.mlp.experts.0.down_proj.weight_scale_inv",
            torch.ones(1, 5),
        ),
        (
            "model.layers.0.mlp.experts.0.gate_up_proj.weight",
            torch.cat((torch.ones(640, 2), torch.full((640, 2), 2.0))),
        ),
    ]

    padded = dict(_pad_moe_checkpoint_weights(weights, vllm_config))
    gate = padded["model.layers.0.mlp.experts.0.gate_proj.weight"]
    up_scale = padded["model.layers.0.mlp.experts.0.up_proj.weight_scale_inv"]
    down = padded["model.layers.0.mlp.experts.0.down_proj.weight"]
    down_scale = padded["model.layers.0.mlp.experts.0.down_proj.weight_scale_inv"]
    gate_up = padded["model.layers.0.mlp.experts.0.gate_up_proj.weight"]

    assert gate.shape == (768, 2)
    assert up_scale.shape == (6, 1)
    assert down.shape == (2, 768)
    assert down_scale.shape == (1, 6)
    assert gate_up.shape == (1536, 2)
    assert torch.count_nonzero(gate[640:]) == 0
    assert torch.count_nonzero(up_scale[5:]) == 0
    assert torch.count_nonzero(down[:, 640:]) == 0
    assert torch.count_nonzero(down_scale[:, 5:]) == 0
    assert torch.equal(gate_up[:640], torch.ones(640, 2))
    assert torch.count_nonzero(gate_up[640:768]) == 0
    assert torch.equal(gate_up[768:1408], torch.full((640, 2), 2.0))
    assert torch.count_nonzero(gate_up[1408:]) == 0

    padded_twice = dict(_pad_moe_checkpoint_weights(padded.items(), vllm_config))
    for name, tensor in padded.items():
        assert torch.equal(padded_twice[name], tensor)
