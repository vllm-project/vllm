# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import Qwen2Config

from vllm.transformers_utils.configs.locate_anything import LocateAnythingConfig
from vllm.transformers_utils.configs.moonvit import MoonViTConfig


def _hf_dicts():
    vision = dict(
        model_type="moonvit",
        hidden_size=1152,
        patch_size=14,
        num_hidden_layers=27,
        num_attention_heads=16,
        intermediate_size=4304,
        merge_kernel_size=[2, 2],
        init_pos_emb_height=64,
        init_pos_emb_width=64,
    )
    text = dict(
        model_type="qwen2",
        hidden_size=2048,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=2,
        intermediate_size=11008,
        vocab_size=152681,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        max_position_embeddings=32768,
        null_token_id=152678,
        switch_token_id=152679,
        text_mask_token_id=151676,
    )
    return vision, text


def test_builds_subconfigs_from_dicts():
    vision, text = _hf_dicts()
    cfg = LocateAnythingConfig(
        vision_config=vision,
        text_config=text,
        image_token_index=151665,
        box_start_token_id=151668,
        box_end_token_id=151669,
        ref_start_token_id=151672,
        ref_end_token_id=151673,
        coord_start_token_id=151677,
        coord_end_token_id=152677,
        none_token_id=4064,
        mlp_connector_layers=2,
    )
    assert cfg.model_type == "locateanything"
    assert isinstance(cfg.vision_config, MoonViTConfig)
    assert isinstance(cfg.text_config, Qwen2Config)
    assert cfg.vision_config.hidden_size == 1152
    assert cfg.text_config.vocab_size == 152681
    assert cfg.text_config.tie_word_embeddings is True


def test_special_token_ids_preserved():
    vision, text = _hf_dicts()
    cfg = LocateAnythingConfig(
        vision_config=vision,
        text_config=text,
        image_token_index=151665,
        box_start_token_id=151668,
        box_end_token_id=151669,
        ref_start_token_id=151672,
        ref_end_token_id=151673,
        coord_start_token_id=151677,
        coord_end_token_id=152677,
        none_token_id=4064,
    )
    assert cfg.image_token_index == 151665
    assert cfg.coord_start_token_id == 151677
    assert cfg.coord_end_token_id == 152677
    assert cfg.box_start_token_id == 151668
    assert cfg.box_end_token_id == 151669


def test_defaults_when_subconfigs_none():
    cfg = LocateAnythingConfig()
    assert isinstance(cfg.vision_config, MoonViTConfig)
    assert isinstance(cfg.text_config, Qwen2Config)
