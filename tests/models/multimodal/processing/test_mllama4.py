# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for mllama's multimodal preprocessing and profiling."""

from types import SimpleNamespace

import pytest
import torch
from torch import prod
from transformers import Llama4Config

from vllm.model_executor.models.mllama4 import Llama4ForConditionalGeneration
from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["meta-llama/Llama-Guard-4-12B"])
@pytest.mark.parametrize("max_model_len", [4096, 8192, 25600, 131072])
def test_profiling(model_id: str, max_model_len: int):
    model_config_kwargs = {
        "max_model_len": max_model_len,
    }
    mm_counts = {"image": 1}
    ctx = build_model_context(
        model_id,
        model_config_kwargs=model_config_kwargs,
        limit_mm_per_prompt=mm_counts,
    )

    mm_inputs = MULTIMODAL_REGISTRY.get_dummy_mm_inputs(
        ctx.model_config,
        mm_counts=mm_counts,
    )

    hf_config = ctx.get_hf_config(Llama4Config)
    image_size = hf_config.vision_config.image_size
    patch_size = hf_config.vision_config.patch_size
    downsample_ratio = int(
        round(1.0 / (hf_config.vision_config.pixel_shuffle_ratio**2))
    )
    tokens_per_patch = ((image_size // patch_size) ** 2) // downsample_ratio

    mm_data = mm_inputs["mm_kwargs"].get_data()
    chunks_per_image = prod(mm_data["patches_per_image"])
    total_num_patches = chunks_per_image * tokens_per_patch
    num_tiles = (
        mm_data["aspect_ratios"][0][0] * mm_data["aspect_ratios"][0][1]
    )  # x-y separator tokens
    total_tokens = (
        total_num_patches.item() + num_tiles.item() + 3
    )  # image start, image, image end

    assert total_num_patches == sum(
        item.get_num_embeds() for item in mm_inputs["mm_placeholders"]["image"]
    )
    assert total_tokens == sum(
        placeholder.length for placeholder in mm_inputs["mm_placeholders"]["image"]
    )


class VisionModelStub(torch.nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values.unsqueeze(-1)


class ProjectorStub(torch.nn.Module):
    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        return vision_embeddings + 1


def make_test_model() -> Llama4ForConditionalGeneration:
    model = object.__new__(Llama4ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(
            image_size=8,
            patch_size=4,
            pixel_shuffle_ratio=1.0,
            num_channels=3,
        ),
        text_config=SimpleNamespace(hidden_size=1),
    )
    model.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=20),
    )
    model.vision_model = VisionModelStub()
    model.multi_modal_projector = ProjectorStub()
    model.use_data_parallel = False
    return model


def test_encoder_cudagraph_metadata():
    model = make_test_model()
    patches_per_chunk = model.get_image_patches_per_chunk()
    mm_kwargs = {
        "patches_per_image": torch.tensor([2, 3], dtype=torch.int32),
    }
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
    )

    config = model.get_encoder_cudagraph_config()

    assert config.modalities == ["image"]
    assert config.input_key_by_modality == {"image": "pixel_values"}
    assert config.buffer_keys == []
    assert config.out_hidden_size == 1
    assert patches_per_chunk == 4
    assert model.get_encoder_cudagraph_budget_range(vllm_config) == (
        patches_per_chunk,
        20,
    )
    item_specs = model.get_encoder_cudagraph_item_specs(mm_kwargs)
    assert [(spec.input_size, spec.output_tokens) for spec in item_specs] == [
        (2, 2 * patches_per_chunk),
        (3, 3 * patches_per_chunk),
    ]


def test_select_encoder_cudagraph_items():
    model = make_test_model()
    mm_kwargs = {
        "pixel_values": torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]]),
        "patches_per_image": torch.tensor([2, 3], dtype=torch.int32),
    }

    selected = model.select_encoder_cudagraph_items(mm_kwargs, [1])
    empty = model.select_encoder_cudagraph_items(mm_kwargs, [])

    assert torch.equal(selected["pixel_values"], torch.tensor([[2.0], [3.0], [4.0]]))
    assert torch.equal(
        selected["patches_per_image"], torch.tensor([3], dtype=torch.int32)
    )
    assert empty["pixel_values"].shape == (0, 1)
    assert empty["patches_per_image"].shape == (0,)


def test_prepare_encoder_cudagraph_capture_inputs_rounds_up():
    model = make_test_model()
    patches_per_chunk = model.get_image_patches_per_chunk()

    capture_inputs = model.prepare_encoder_cudagraph_capture_inputs(
        token_budget=patches_per_chunk + 1,
        max_batch_size=2,
        max_frames_per_batch=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert capture_inputs.mm_kwargs["pixel_values"].shape == (2, 3, 8, 8)
    assert capture_inputs.buffers == {}


def test_encoder_cudagraph_forward_matches_eager():
    model = make_test_model()
    mm_kwargs = {
        "pixel_values": torch.tensor([[1.0], [2.0]]),
    }

    eager = model.encoder_eager_forward(mm_kwargs)
    cg = model.encoder_cudagraph_forward(mm_kwargs, buffers={})

    expected = torch.tensor([[2.0], [3.0]])
    assert torch.equal(eager, expected)
    assert torch.equal(cg, expected)
