# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from transformers import (
    CLIPVisionConfig,
    LlavaConfig,
    PixtralVisionConfig,
    SiglipVisionConfig,
)

from vllm.model_executor.models.llava import (
    LlavaForConditionalGeneration,
    LlavaMultiModalProjector,
    init_vision_tower_for_llava,
)
from vllm.platforms import current_platform
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

pytestmark = pytest.mark.core_model


def _model(vision_config, strategy):
    model = object.__new__(LlavaForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = LlavaConfig(
        vision_config=vision_config.to_dict(),
        text_config={"model_type": "llama", "hidden_size": 64},
        vision_feature_layer=-1,
        vision_feature_select_strategy=strategy,
    )
    return model


def test_pixtral_encoder_graph_is_rejected():
    model = _model(PixtralVisionConfig(), "full")
    with pytest.raises(NotImplementedError, match="Pixtral"):
        model.get_encoder_cudagraph_config()


def test_chunked_prefill_budget_fits_one_image():
    model = _model(SiglipVisionConfig(image_size=384, patch_size=14), "full")
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
        model_config=SimpleNamespace(max_model_len=4096),
    )
    assert model.get_encoder_cudagraph_budget_range(config) == (729, 729)


@pytest.mark.parametrize(
    "vision_type,strategy,num_tokens",
    [
        (CLIPVisionConfig, "default", 16),
        (CLIPVisionConfig, "full", 17),
        (SiglipVisionConfig, "full", 16),
    ],
)
def test_llava_encoder_graph_input_contract(vision_type, strategy, num_tokens):
    model = _model(vision_type(image_size=32, patch_size=8), strategy)
    pixels = torch.randn(3, 3, 32, 32)
    selected = model.select_encoder_cudagraph_items({"pixel_values": pixels}, [2, 0])
    torch.testing.assert_close(selected["pixel_values"], pixels[[2, 0]])
    assert [
        s.output_tokens for s in model.get_encoder_cudagraph_item_specs(selected)
    ] == [num_tokens, num_tokens]
    empty = model.select_encoder_cudagraph_items({"pixel_values": pixels}, [])
    assert empty["pixel_values"].shape == (0, 3, 32, 32)
    assert model.get_encoder_cudagraph_item_specs(empty) == []
    capture = model.prepare_encoder_cudagraph_capture_inputs(
        8 * num_tokens, 2, 0, torch.device("cpu"), torch.float32
    )
    # Large token budgets must not allocate more images than the batch cap.
    assert capture.values["pixel_values"].shape == (2, 3, 32, 32)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Requires GPU")
@pytest.mark.usefixtures("default_vllm_config", "dist_init")
@pytest.mark.parametrize(
    "vision_type,strategy",
    [
        (CLIPVisionConfig, "default"),
        (CLIPVisionConfig, "full"),
        (SiglipVisionConfig, "full"),
    ],
)
@pytest.mark.parametrize("fallback", [False, True])
@torch.inference_mode()
def test_llava_encoder_graph_matches_eager(vision_type, strategy, fallback):
    torch.manual_seed(0)
    vision = vision_type(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=32,
        patch_size=8,
    )
    model = _model(vision, strategy)
    model.vision_tower = init_vision_tower_for_llava(
        model.config,
        quant_config=None,
        require_post_norm=False,
    )
    model.multi_modal_projector = LlavaMultiModalProjector(
        vision_hidden_size=64,
        text_hidden_size=64,
        projector_hidden_act="gelu",
        multimodal_projector_bias=True,
    )
    model = model.cuda().half().eval()
    # vLLM layers allocate empty weights; initialize explicitly before comparing.
    for name, parameter in model.named_parameters():
        if "norm" in name and name.endswith("weight"):
            parameter.fill_(1)
        else:
            parameter.normal_(std=0.02)
    example = torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.float16)
    num_tokens = model.embed_multimodal(pixel_values=example).shape[1]
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_token_budgets=[
                num_tokens - 1 if fallback else 2 * num_tokens
            ],
            encoder_cudagraph_max_vision_items_per_batch=2,
            encoder_cudagraph_max_frames_per_batch=None,
        ),
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(
                mm_encoder_tp_mode="replicate",
                get_limit_per_prompt=lambda modality: int(modality == "image"),
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    manager = EncoderCudaGraphManager(
        config, torch.device("cuda"), torch.float16, model
    )
    manager.capture(graph_pool=current_platform.graph_pool_handle())
    retained: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch in (1, 3, 2, 1):
        pixels = torch.randn(batch, 3, 32, 32, device="cuda", dtype=torch.float16)
        expected = model.embed_multimodal(pixel_values=pixels)
        actual = manager.execute({"pixel_values": pixels})
        assert len(actual) == batch
        for output, eager in zip(actual, expected):
            assert output.shape == (num_tokens, 64)
            torch.testing.assert_close(output, eager, rtol=1e-3, atol=1e-3)
        for output, snapshot in retained:
            torch.testing.assert_close(output, snapshot, rtol=0, atol=0)
        retained.extend((output, output.clone()) for output in actual)
    assert manager.graph_hits == (0 if fallback else 7)
    assert manager.graph_misses == (7 if fallback else 0)
