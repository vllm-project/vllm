# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Binding behavior for checkpoint-backed Final Norm rotations."""

import pytest
import torch
from transformers import DeepseekV3Config

from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotation.norm import PreRotationRMSNorm
from vllm.model_executor.layers.rotation.replacement import (
    bind_rotation_transforms,
    prepare_rotation_transforms,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.platforms import current_platform

pytestmark = pytest.mark.usefixtures("default_vllm_config")


@pytest.fixture
def rotation_config(tmp_path):
    DeepseekV3Config(
        architectures=["DeepseekV3ForCausalLM"],
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=32,
        num_nextn_predict_layers=1,
        rotation_config={
            "rotations": {
                "final": {
                    "type": "composite_hadamard",
                    "hidden_size": 16,
                    "hadamard_k": 1,
                }
            },
            "apply": [
                {"tag": "final", "targets": ["norm"], "mode": "pre", "inverse": True}
            ],
        },
    ).save_pretrained(tmp_path)
    return VllmConfig(
        model_config=ModelConfig(
            str(tmp_path), skip_tokenizer_init=True, dtype="bfloat16", max_model_len=32
        )
    )


def _prepared_model(
    vllm_config: VllmConfig,
    targets: list[str],
    *,
    draft_dtype: torch.dtype | None = None,
    model: torch.nn.Module | None = None,
    draft_model: torch.nn.Module | None = None,
) -> torch.nn.Module:
    vllm_config.model_config.hf_config.rotation_config["apply"][0]["targets"] = targets
    if draft_dtype is not None:
        vllm_config.speculative_config = SpeculativeConfig(
            method="mtp",
            num_speculative_tokens=1,
            target_model_config=vllm_config.model_config,
            target_parallel_config=vllm_config.parallel_config,
        )
        vllm_config.speculative_config.draft_model_config.dtype = draft_dtype
    if model is None:
        model = torch.nn.Module()
        model.add_module("norm", RMSNorm(16))
    prepare_rotation_transforms(model, vllm_config, vllm_config.model_config)
    if draft_model is not None:
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        prepare_rotation_transforms(
            draft_model, vllm_config, speculative_config.draft_model_config
        )
    with torch.no_grad():
        model.rotations["final"].sign.fill_(1)
        model.rotations["final"].hk.fill_(1)
    return model


def test_submodel_preparation_defers_to_outer_model(rotation_config):
    inner = torch.nn.Module()
    inner.add_module("norm", RMSNorm(16))
    original_weight = inner.norm.weight
    config = rotation_config.with_hf_config(rotation_config.model_config.hf_config)
    prepare_rotation_transforms(inner, config, config.model_config)
    assert type(inner.norm) is RMSNorm
    assert not hasattr(inner, "rotations")

    outer = torch.nn.Module()
    outer.add_module("language_model", inner)
    _prepared_model(rotation_config, ["language_model.norm"], model=outer)
    bind_rotation_transforms(outer)
    assert inner.norm.rotation is outer.rotations["final"]
    assert (
        dict(outer.named_parameters())["language_model.norm.weight"] is original_weight
    )
    assert not hasattr(inner, "rotations")


@pytest.mark.parametrize("draft_dtype", [None, torch.bfloat16, torch.float16])
def test_prepare_registers_mtp_warmup_before_draft_loading(
    rotation_config, draft_dtype
):
    """Prepare covers MTP-only targets and deduplicates equal main/draft keys."""
    registry = JitWarmupRegistry(rotation_config)
    with registry.activate():
        _prepared_model(
            rotation_config,
            ["model.layers.45.shared_head.norm"],
            draft_dtype=draft_dtype,
        )
    dtypes = {torch.bfloat16}
    if draft_dtype is not None:
        dtypes.add(draft_dtype)
    assert len(registry) == (len(dtypes) if current_platform.is_cuda() else 0)


@pytest.mark.usefixtures("disable_log_dedup")
def test_main_norm_binds_without_optional_mtp_model(rotation_config, caplog_vllm):
    """The main norm binds and missing draft targets are visible in the log."""
    model = _prepared_model(
        rotation_config, ["norm", "model.layers.45.shared_head.norm"]
    )
    bind_rotation_transforms(model)
    assert model.norm.rotation is model.rotations["final"]
    assert "Skipping unresolved rotation targets" in caplog_vllm.text
    assert "model.layers.45.shared_head.norm" in caplog_vllm.text


def test_ngram_target_owns_and_binds_rotation(rotation_config):
    rotation_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=2,
        target_model_config=rotation_config.model_config,
        target_parallel_config=rotation_config.parallel_config,
    )
    model = _prepared_model(rotation_config, ["norm"])
    bind_rotation_transforms(model)

    rotation = model.rotations["final"]
    assert model.norm.rotation is rotation
    params = dict(model.named_parameters())
    assert params["rotations.final.sign"] is rotation.sign
    assert params["rotations.final.hk"] is rotation.hk


def test_main_and_mtp_share_one_checkpoint_rotation(rotation_config):
    """Draft preparation reuses the target's parameters without registering copies."""
    draft = torch.nn.Module()
    draft.add_module("mtp_norm", RMSNorm(16))
    model = _prepared_model(
        rotation_config,
        ["norm", "mtp_norm"],
        draft_dtype=torch.bfloat16,
        draft_model=draft,
    )
    bind_rotation_transforms(model, draft)

    assert model.norm.rotation is draft.mtp_norm.rotation is model.rotations["final"]
    assert not hasattr(draft, "rotations")
    assert set(dict(draft.named_parameters())) == {"mtp_norm.weight"}


@torch.inference_mode()
def test_norm_returns_inverse_rotated_residual(rotation_config):
    """Both returned tensors use the restored coordinates of hidden plus residual."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _prepared_model(rotation_config, ["norm"]).to(
        device=device, dtype=torch.float32
    )
    model.rotations["final"].sign[1::2] = -1
    model.norm.weight.copy_(
        torch.linspace(0.5, 1.5, 16, device=device, dtype=torch.float32)
    )
    bind_rotation_transforms(model)

    x = torch.arange(32, device=device, dtype=torch.float32).reshape(2, 16) / 16
    residual = torch.linspace(
        -0.5, 0.5, 32, device=device, dtype=torch.float32
    ).reshape(2, 16)
    restored = model.rotations["final"](x + residual, inverse=True)
    expected = (
        restored
        * torch.rsqrt(
            restored.square().mean(dim=-1, keepdim=True) + model.norm.variance_epsilon
        )
        * model.norm.weight
    )

    actual, actual_residual = model.norm(x, residual)
    torch.testing.assert_close(actual_residual, restored)
    torch.testing.assert_close(actual, expected)


@pytest.mark.usefixtures("disable_log_dedup")
def test_mtp_only_target_warns_without_draft_and_is_checked_with_draft(
    rotation_config, caplog_vllm
):
    """MTP-only targets can be skipped until a draft model is loaded."""
    target = "model.layers.45.shared_head.norm"
    model = _prepared_model(rotation_config, [target])
    bind_rotation_transforms(model)
    assert "Skipping unresolved rotation targets" in caplog_vllm.text
    assert target in caplog_vllm.text
    with pytest.raises(ValueError, match="shared_head.norm"):
        bind_rotation_transforms(model, _prepared_model(rotation_config, [target]))


def test_custom_rmsnorm_is_rejected_without_modification():
    class CustomRMSNorm(RMSNorm):
        def forward(self, x):
            return x + 1

    norm = CustomRMSNorm(16)
    with pytest.raises(TypeError, match="plain RMSNorm"):
        PreRotationRMSNorm.upgrade(norm, tag="final", inverse=True)
    assert type(norm) is CustomRMSNorm
    x = torch.zeros(1, 16)
    torch.testing.assert_close(norm(x), x + 1)
