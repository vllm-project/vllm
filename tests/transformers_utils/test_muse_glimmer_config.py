# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MuseGlimmer config normalization: flat (legacy-converter) and nested (canonical).

Guards against the silent-ignore bug where a FLAT MuseGlimmer config.json (older
converter, e.g. rl_v1/hf) deserialized to an all-default text config, silently
dropping every checkpoint value.
"""

from typing import Any

from vllm.transformers_utils.configs.muse_glimmer import MuseGlimmerConfig

# A representative FLAT config (Ruan rl_v1/hf shape), trimmed.
FLAT: dict[str, Any] = {
    "architectures": ["MuseGlimmerForCausalLM"],
    "model_type": "muse_glimmer",
    "has_vision": True,
    "bos_token_id": 200000,
    "eos_token_id": 200001,
    "vocab_size": 202048,
    "hidden_size": 6656,
    "intermediate_size": 19968,
    "num_hidden_layers": 52,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 16384,
    "rms_norm_eps": 1e-5,
    "post_norm_eps": 1e-8,
    "qk_scale_factor": 43.7840518911,
    "use_qk_norm": True,
    "use_attn_output_gate": True,
    "output_multiplier": 0.19611613513818404,
    "output_soft_cap_temp": 20.0,
    "normalize_tok_embeddings": True,
    "rope_theta": 500000.0,
    "sliding_window": 2048,
    "patch_token_id": 200092,
    "vision_latent_dim": 1536,
    "vision_heads": 16,
    "vision_layers": 50,
    "vision_output_dim": 6144,
    "vision_patch_size": 14,
    "vision_patch_temporal": 2,
    "vision_adapter_dim": 4096,
    "vision_pos_emb_grid_h": 32,
    "vision_pos_emb_grid_w": 32,
}

NESTED: dict[str, Any] = {
    "architectures": ["MuseGlimmerForCausalLM"],
    "model_type": "muse_glimmer",
    "image_token_id": 200092,
    "text_config": {
        "model_type": "muse_glimmer_text",
        "vocab_size": 202048,
        "hidden_size": 6656,
        "num_hidden_layers": 52,
        "hidden_activation": "silu",
        "final_logit_softcapping": 20.0,
        "qk_scale_factor": 43.7840518911,
        "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0},
    },
    "vision_config": {
        "model_type": "muse_glimmer_vision",
        "hidden_size": 1536,
        "num_hidden_layers": 50,
    },
}


def test_flat_config_values_respected():
    c = MuseGlimmerConfig(**FLAT)
    t = c.text_config
    assert t.hidden_size == 6656
    assert t.num_hidden_layers == 52
    assert t.vocab_size == 202048
    assert t.head_dim == 128
    assert t.hidden_activation == "silu"  # renamed from hidden_act
    assert t.final_logit_softcapping == 20.0  # renamed from output_soft_cap_temp
    assert abs(t.output_multiplier - 0.19611613513818404) < 1e-12
    assert abs(t.qk_scale_factor - 43.7840518911) < 1e-9
    assert t.rope_parameters["rope_theta"] == 500000.0  # from flat rope_theta
    # vision hoisted + renamed
    assert c.vision_config.hidden_size == 1536
    assert c.vision_config.num_hidden_layers == 50
    assert c.vision_config.output_dim == 6144
    # flat patch_token_id -> image_token_id
    assert c.image_token_id == 200092


def test_flat_config_no_silent_default():
    # The regression: a non-default value MUST be honored, not silently dropped.
    flat = dict(FLAT)
    flat["hidden_size"] = 4096
    flat["num_hidden_layers"] = 40
    c = MuseGlimmerConfig(**flat)
    assert c.text_config.hidden_size == 4096, "flat hidden_size silently ignored!"
    assert c.text_config.num_hidden_layers == 40, "flat num_hidden_layers ignored!"


def test_nested_config_unchanged():
    c = MuseGlimmerConfig(**NESTED)
    assert c.text_config.hidden_size == 6656
    assert c.text_config.hidden_activation == "silu"
    assert c.text_config.final_logit_softcapping == 20.0
    assert c.vision_config.hidden_size == 1536
    assert c.vision_config.num_hidden_layers == 50
    assert c.image_token_id == 200092


if __name__ == "__main__":
    test_flat_config_values_respected()
    test_flat_config_no_silent_default()
    test_nested_config_unchanged()
    print("ALL MUSE_GLIMMER CONFIG TESTS PASSED")
