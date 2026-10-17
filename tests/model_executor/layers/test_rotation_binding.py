# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Binding behavior for checkpoint-backed Final Norm rotations."""

from types import SimpleNamespace

import pytest
import torch

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
    targets: list[str], *, draft_dtype: torch.dtype | None = None
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
    model_config = SimpleNamespace(hf_config=config, dtype=torch.bfloat16)
    speculative_config = (
        SimpleNamespace(draft_model_config=SimpleNamespace(dtype=draft_dtype))
        if draft_dtype is not None
        else None
    )
    vllm_config = SimpleNamespace(
        model_config=model_config, speculative_config=speculative_config
    )
    model = torch.nn.Module()
    model.add_module("norm", RMSNorm(16))
    prepare_rotation_transforms(model, vllm_config, model_config)
    with torch.no_grad():
        model.rotations["final"].sign.fill_(1)
        model.rotations["final"].hk.fill_(1)
    return model


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


def test_main_norm_binds_without_optional_mtp_model(caplog_vllm):
    """The main norm binds and missing draft targets are visible in the log."""
    model = _prepared_model(["norm", "model.layers.45.shared_head.norm"])
    bind_rotation_transforms(model)
    assert model.norm.rotation is model.rotations["final"]
    assert "Skipping unresolved rotation targets" in caplog_vllm.text
    assert "model.layers.45.shared_head.norm" in caplog_vllm.text


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
