# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from vllm.model_executor.models import is_text_generation_model, supports_multimodal
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.models.unlimited_ocr import UnlimitedOCRProcessingInfo
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.unlimited_ocr import UnlimitedOCRConfig
from vllm.transformers_utils.processors.deepseek_ocr import count_tiles


def _write_minimal_config(path):
    config = {
        "architectures": ["UnlimitedOCRForCausalLM"],
        "model_type": "unlimited-ocr",
        "tile_tag": "2D",
        "global_view_pos": "head",
        "projector_config": {
            "input_dim": 2048,
            "model_type": "mlp_projector",
            "n_embed": 1280,
            "projector_type": "linear",
        },
        "vision_config": {
            "image_size": 1024,
            "model_type": "vision",
            "model_name": "deeplip_b_l",
        },
        "language_config": {
            "architectures": ["DeepseekOCRForCausalLM"],
            "model_type": "deepseek_v2",
            "bos_token_id": 0,
            "eos_token_id": 1,
            "hidden_size": 1280,
            "intermediate_size": 6848,
            "max_position_embeddings": 32768,
            "moe_intermediate_size": 896,
            "n_group": 1,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "num_attention_heads": 10,
            "num_experts_per_tok": 6,
            "num_hidden_layers": 12,
            "num_key_value_heads": 10,
            "topk_group": 1,
            "topk_method": "greedy",
            "torch_dtype": "bfloat16",
            "use_mla": False,
            "v_head_dim": 128,
            "vocab_size": 129280,
            "sliding_window_size": 128,
        },
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_unlimited_ocr_config_loads_without_remote_code(tmp_path):
    _write_minimal_config(tmp_path)

    config = get_config(str(tmp_path), trust_remote_code=False)

    assert isinstance(config, UnlimitedOCRConfig)
    assert config.model_type == "unlimited-ocr"
    assert config.architectures == ["UnlimitedOCRForCausalLM"]
    assert config.projector_config.input_dim == 2048
    assert config.text_config.model_type == "deepseek_v2"
    assert config.text_config.architectures == ["DeepseekV2ForCausalLM"]


def test_unlimited_ocr_registry_imports_as_multimodal_generation_model():
    model_cls = ModelRegistry._try_load_model_cls("UnlimitedOCRForCausalLM")

    assert model_cls is not None
    assert model_cls.__name__ == "UnlimitedOCRForCausalLM"
    assert is_text_generation_model(model_cls)
    assert supports_multimodal(model_cls)


def test_unlimited_ocr_image_token_count_matches_v1_layout():
    info = object.__new__(UnlimitedOCRProcessingInfo)

    assert info.get_num_image_tokens(image_width=100, image_height=100) == 273
    assert info.get_num_image_tokens(image_width=640, image_height=640) == 273

    width_tiles, height_tiles = count_tiles(1200, 800, image_size=640)
    assert (width_tiles, height_tiles) == (3, 2)

    global_tokens = 16 * (16 + 1) + 1
    local_tokens = (10 * width_tiles + 1) * 10 * height_tiles
    assert info.get_num_image_tokens(image_width=1200, image_height=800) == (
        global_tokens + local_tokens
    )


def test_unlimited_ocr_tokenizer_uses_bytelevel_decoder():
    model_path = Path("/mnt/weight/Unlimited-OCR")
    if not (model_path / "tokenizer.json").is_file():
        pytest.skip("Unlimited-OCR tokenizer assets are not available")

    tokenizer = get_tokenizer(str(model_path), trust_remote_code=False)

    assert tokenizer.decode([695, 1266, 1080, 201]) == "年月日\n"
