# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.warmup.qwen3_vl_vision_warmup import (
    qwen3_vl_vision_warmup,
)


class Qwen3_VisionTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.hidden_size = 16
        self.num_heads = 2
        self.spatial_merge_size = 2
        self.patch_embed = SimpleNamespace(
            patch_size=4,
            temporal_patch_size=2,
            proj=SimpleNamespace(in_channels=3),
        )
        self.metadata = {"metadata": object()}
        self.metadata_calls: list[list[list[int]]] = []
        self.forward_calls: list[tuple[torch.Tensor, list[list[int]], dict]] = []

    def prepare_encoder_metadata(self, grid_thw):
        self.metadata_calls.append(grid_thw)
        return self.metadata

    def forward(self, pixel_values, grid_thw, *, encoder_metadata=None):
        self.forward_calls.append((pixel_values, grid_thw, encoder_metadata))
        return torch.empty(0)


class OtherVision(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self):  # pragma: no cover - should not be called
        self.calls += 1


def test_qwen3_vl_vision_warmup_runs_minimal_visual_forward() -> None:
    visual = Qwen3_VisionTransformer()
    model = torch.nn.Module()
    model.visual = visual

    qwen3_vl_vision_warmup(model)

    assert visual.metadata_calls == [[[1, 2, 2]]]
    assert len(visual.forward_calls) == 1
    pixel_values, grid_thw, metadata = visual.forward_calls[0]
    assert pixel_values.shape == (4, 96)
    assert pixel_values.device == visual.device
    assert pixel_values.dtype == visual.dtype
    assert grid_thw == [[1, 2, 2]]
    assert metadata is visual.metadata


def test_qwen3_vl_vision_warmup_skips_other_modules() -> None:
    other = OtherVision()
    model = torch.nn.Module()
    model.visual = other

    qwen3_vl_vision_warmup(model)

    assert other.calls == 0


def test_qwen3_vl_vision_warmup_skips_non_module_model() -> None:
    qwen3_vl_vision_warmup(object())
