# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Binding behavior for checkpoint-backed Final Norm rotations."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotation.composite_hadamard import (
    CompositeHadamardRotation,
)
from vllm.model_executor.layers.rotation.norm import PreRotationRMSNorm
from vllm.model_executor.layers.rotation.replacement import (
    bind_rotation_transforms,
    prepare_rotation_transforms,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

pytestmark = pytest.mark.usefixtures("default_vllm_config")


def _prepared_model(
    targets: list[str],
    *,
    draft_dtype: torch.dtype | None = None,
    model: torch.nn.Module | None = None,
    draft_model: torch.nn.Module | None = None,
) -> torch.nn.Module:
    config = SimpleNamespace(
        rotation_config={
            "rotations": {
                "final": {
                    "type": "composite_hadamard",
                    "hidden_size": 16,
                    "hadamard_k": 1,
                }
            },
            "apply": [
                {
                    "tag": "final",
                    "targets": targets,
                    "mode": "pre",
                    "inverse": True,
                }
            ],
        }
    )
    model_config = SimpleNamespace(
        hf_config=config, dtype=torch.bfloat16, is_submodel_config=False
    )
    speculative_config = (
        SimpleNamespace(
            draft_model_config=SimpleNamespace(
                dtype=draft_dtype, is_submodel_config=False
            )
        )
        if draft_dtype is not None
        else None
    )
    vllm_config = cast(
        VllmConfig,
        SimpleNamespace(
            model_config=model_config, speculative_config=speculative_config
        ),
    )
    if model is None:
        model = torch.nn.Module()
        model.add_module("norm", RMSNorm(16))
    prepare_rotation_transforms(model, vllm_config, vllm_config.model_config)
    if draft_model is not None:
        assert speculative_config is not None
        prepare_rotation_transforms(
            draft_model, vllm_config, speculative_config.draft_model_config
        )
    with torch.no_grad():
        model.rotations["final"].sign.fill_(1)
        model.rotations["final"].hk.fill_(1)
    return model


def test_submodel_preparation_defers_to_outer_model():
    inner = torch.nn.Module()
    inner.add_module("norm", RMSNorm(16))
    original_weight = inner.norm.weight
    config = cast(
        VllmConfig,
        SimpleNamespace(model_config=SimpleNamespace(is_submodel_config=True)),
    )
    prepare_rotation_transforms(inner, config, config.model_config)
    assert type(inner.norm) is RMSNorm
    assert not hasattr(inner, "rotations")

    outer = torch.nn.Module()
    outer.add_module("language_model", inner)
    _prepared_model(["language_model.norm"], model=outer)
    bind_rotation_transforms(outer)
    assert inner.norm.rotation is outer.rotations["final"]
    assert (
        dict(outer.named_parameters())["language_model.norm.weight"] is original_weight
    )
    assert not hasattr(inner, "rotations")


@pytest.mark.parametrize("draft_dtype", [None, torch.bfloat16, torch.float16])
def test_prepare_registers_mtp_warmup_before_draft_loading(monkeypatch, draft_dtype):
    """Prepare covers MTP-only targets and deduplicates equal main/draft keys."""
    from vllm.model_executor.kernels.rotation.composite_hadamard.warmup import (
        CompositeHadamardEpilogueKernel,
    )

    kernel = CompositeHadamardEpilogueKernel()
    monkeypatch.setattr(
        CompositeHadamardRotation,
        "register_warmup",
        lambda self, **kwargs: kernel.register_warmup(k=self.k, p=self.p, **kwargs),
    )
    registry = JitWarmupRegistry(None)
    with registry.activate():
        _prepared_model(["model.layers.45.shared_head.norm"], draft_dtype=draft_dtype)
    dtypes = {torch.bfloat16}
    if draft_dtype is not None:
        dtypes.add(draft_dtype)
    assert len(registry) == len(dtypes)
    compiled: list[CompositeHadamardEpilogueKernel.CompileKey] = []
    monkeypatch.setattr(kernel, "compile", compiled.append)
    registry.warmup()
    assert set(compiled) == {kernel.CompileKey(1, 16, True, dtype) for dtype in dtypes}


@pytest.mark.usefixtures("disable_log_dedup")
def test_main_norm_binds_without_optional_mtp_model(caplog_vllm):
    """The main norm binds and missing draft targets are visible in the log."""
    model = _prepared_model(["norm", "model.layers.45.shared_head.norm"])
    bind_rotation_transforms(model)
    assert model.norm.rotation is model.rotations["final"]
    assert "Skipping unresolved rotation targets" in caplog_vllm.text
    assert "model.layers.45.shared_head.norm" in caplog_vllm.text


def test_main_and_mtp_share_one_checkpoint_rotation():
    """Draft preparation reuses the target's parameters without registering copies."""
    draft = torch.nn.Module()
    draft.add_module("mtp_norm", RMSNorm(16))
    model = _prepared_model(
        ["norm", "mtp_norm"], draft_dtype=torch.bfloat16, draft_model=draft
    )
    bind_rotation_transforms(model, draft)

    assert model.norm.rotation is draft.mtp_norm.rotation is model.rotations["final"]
    assert not hasattr(draft, "rotations")
    assert set(dict(draft.named_parameters())) == {"mtp_norm.weight"}


@torch.inference_mode()
def test_norm_returns_inverse_rotated_residual():
    """Both returned tensors use the restored coordinates of hidden plus residual."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _prepared_model(["norm"]).to(device=device, dtype=torch.float32)
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
def test_mtp_only_target_warns_without_draft_and_is_checked_with_draft(caplog_vllm):
    """MTP-only targets can be skipped until a draft model is loaded."""
    target = "model.layers.45.shared_head.norm"
    model = _prepared_model([target])
    bind_rotation_transforms(model)
    assert "Skipping unresolved rotation targets" in caplog_vllm.text
    assert target in caplog_vllm.text
    with pytest.raises(ValueError, match="shared_head.norm"):
        bind_rotation_transforms(model, _prepared_model([target]))


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
