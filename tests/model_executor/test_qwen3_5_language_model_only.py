# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models.utils import StageMissingLayer


class _TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.make_empty_intermediate_tensors = Mock()


class _TinyVisionTower(nn.Module):
    def __init__(self, vision_config, **kwargs):
        super().__init__()
        self.out_hidden_size = vision_config.out_hidden_size
        self.spatial_merge_size = vision_config.spatial_merge_size


def _make_vllm_config(
    *,
    language_model_only: bool = False,
    limit_per_prompt: dict[str, int] | None = None,
):
    vision_config = SimpleNamespace(
        deepstack_visual_indexes=[],
        out_hidden_size=8,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    hf_config = SimpleNamespace(
        rms_norm_eps=1e-6,
        vision_config=vision_config,
    )
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_config=hf_config,
        multimodal_config=MultiModalConfig(
            language_model_only=language_model_only,
            limit_per_prompt=limit_per_prompt or {},
        ),
    )
    return SimpleNamespace(
        model_config=model_config,
        quant_config=None,
    )


def _build_model(model_cls, vllm_config, vision_ctor):
    from vllm.model_executor.models import qwen3_5

    language_model_name = (
        "Qwen3_5MoeForCausalLM" if "Moe" in model_cls.__name__ else "Qwen3_5ForCausalLM"
    )
    patches = [
        patch.object(qwen3_5, "Qwen3_VisionTransformer", vision_ctor),
        patch.object(qwen3_5, language_model_name, return_value=_TinyLanguageModel()),
        patch.object(qwen3_5, "cached_tokenizer_from_config", return_value=None),
    ]
    if "Moe" in model_cls.__name__:
        patches.append(
            patch.object(
                qwen3_5.Qwen3_5MoeForConditionalGeneration,
                "set_moe_parameters",
            )
        )

    with patches[0], patches[1], patches[2]:
        if len(patches) == 4:
            with patches[3]:
                return model_cls(vllm_config=vllm_config)
        return model_cls(vllm_config=vllm_config)


MODEL_CLASSES = [
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
]


@pytest.mark.parametrize("model_name", MODEL_CLASSES)
@pytest.mark.parametrize(
    ("language_model_only", "limit_per_prompt"),
    [
        (True, None),
        (False, {"image": 0, "video": 0}),
    ],
)
def test_qwen35_language_model_only_skips_vision_constructor(
    model_name: str,
    language_model_only: bool,
    limit_per_prompt: dict[str, int] | None,
):
    from vllm.model_executor.models import qwen3_5

    vision_ctor = Mock(side_effect=AssertionError("vision tower was constructed"))
    model = _build_model(
        getattr(qwen3_5, model_name),
        _make_vllm_config(
            language_model_only=language_model_only,
            limit_per_prompt=limit_per_prompt,
        ),
        vision_ctor,
    )

    assert vision_ctor.call_count == 0
    assert isinstance(model.visual, StageMissingLayer)
    assert model.visual.stage_name == "vision_tower"
    assert model.visual.out_hidden_size == 8
    assert model.visual.spatial_merge_size == 2
    assert model._tower_model_names == ["visual"]
    assert not list(model.visual.named_parameters())
    assert not list(model.visual.named_buffers())

    loaded = model.load_weights([("model.visual.dummy.weight", torch.empty(1))])
    assert loaded == set()


@pytest.mark.parametrize("model_name", MODEL_CLASSES)
def test_qwen35_constructs_vision_when_a_modality_is_enabled(model_name: str):
    from vllm.model_executor.models import qwen3_5

    vision_ctor = Mock(side_effect=_TinyVisionTower)
    model = _build_model(
        getattr(qwen3_5, model_name),
        _make_vllm_config(limit_per_prompt={"image": 0, "video": 1}),
        vision_ctor,
    )

    vision_ctor.assert_called_once()
    assert not isinstance(model.visual, StageMissingLayer)
