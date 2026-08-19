# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import torch

from vllm.config import ParallelConfig
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.models.deepseek_v2 import (
    _can_fuse_nvfp4_shared_experts,
    _shared_expert_tensor_chunks,
)


def _nvfp4_config(exclude_modules: list[str] | None = None) -> ModelOptNvFp4Config:
    return ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        exclude_modules=exclude_modules or [],
    )


def test_deepseek_enables_native_nvfp4_shared_experts(monkeypatch) -> None:
    platform = Mock()
    platform.is_cuda.return_value = True
    platform.is_device_capability_family.return_value = True
    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_v2.current_platform",
        platform,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_v2.has_flashinfer_trtllm_fused_moe",
        lambda: True,
    )

    assert _can_fuse_nvfp4_shared_experts(
        _nvfp4_config(),
        ParallelConfig(),
        "model.layers.3.mlp",
    )


def test_deepseek_can_disable_native_nvfp4_shared_experts(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_FLASHINFER_NVFP4_FUSED_SHARED_EXPERTS", "0")
    platform = Mock()
    platform.is_cuda.return_value = True
    platform.is_device_capability_family.return_value = True
    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_v2.current_platform",
        platform,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_v2.has_flashinfer_trtllm_fused_moe",
        lambda: True,
    )

    assert not _can_fuse_nvfp4_shared_experts(
        _nvfp4_config(),
        ParallelConfig(),
        "model.layers.3.mlp",
    )


def test_deepseek_rejects_excluded_nvfp4_shared_weights(monkeypatch) -> None:
    platform = Mock()
    platform.is_cuda.return_value = True
    platform.is_device_capability_family.return_value = True
    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_v2.current_platform",
        platform,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_v2.has_flashinfer_trtllm_fused_moe",
        lambda: True,
    )

    assert not _can_fuse_nvfp4_shared_experts(
        _nvfp4_config(["shared_experts"]),
        ParallelConfig(),
        "model.layers.3.mlp",
    )


def test_deepseek_rejects_native_nvfp4_shared_experts_with_ep(monkeypatch) -> None:
    platform = Mock()
    platform.is_cuda.return_value = True
    platform.is_device_capability_family.return_value = True
    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_v2.current_platform",
        platform,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.deepseek_v2.has_flashinfer_trtllm_fused_moe",
        lambda: True,
    )

    assert not _can_fuse_nvfp4_shared_experts(
        _nvfp4_config(),
        ParallelConfig(enable_expert_parallel=True),
        "model.layers.3.mlp",
    )


def test_deepseek_splits_widened_shared_expert_tensors() -> None:
    gate_weight = torch.arange(24).reshape(6, 4)
    gate_chunks = _shared_expert_tensor_chunks(
        "mlp.shared_experts.gate_proj.weight", gate_weight, 2
    )
    assert [chunk.shape for chunk in gate_chunks] == [(3, 4), (3, 4)]
    torch.testing.assert_close(gate_chunks[0], gate_weight[:3])
    torch.testing.assert_close(gate_chunks[1], gate_weight[3:])

    down_scale = torch.arange(24).reshape(4, 6)
    down_chunks = _shared_expert_tensor_chunks(
        "mlp.shared_experts.down_proj.weight_scale", down_scale, 2
    )
    assert [chunk.shape for chunk in down_chunks] == [(4, 3), (4, 3)]
    torch.testing.assert_close(down_chunks[0], down_scale[:, :3])
    torch.testing.assert_close(down_chunks[1], down_scale[:, 3:])


def test_deepseek_replicates_shared_expert_nvfp4_scalars() -> None:
    scale = torch.tensor(0.125)
    chunks = _shared_expert_tensor_chunks(
        "mlp.shared_experts.gate_proj.weight_scale_2", scale, 2
    )

    assert len(chunks) == 2
    assert all(chunk.ndim == 0 and chunk.item() == scale.item() for chunk in chunks)
