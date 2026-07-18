from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)


def test_qwen3_omni_hf_to_vllm_mapper_dict():

    mapper = (
        Qwen3OmniMoeThinkerForConditionalGeneration
        .hf_to_vllm_mapper
    )

    weights = {
        # ModelSlim exported checkpoint format
        "model.layers.0.self_attn.q_proj.weight": 1,
        "model.embed_tokens.weight": 2,
        "lm_head.weight": 3,

        # Existing Qwen3 Omni format
        "thinker.model.layers.0.self_attn.q_proj.weight": 4,
        "thinker.lm_head.weight": 5,

        # Should not be modified
        "visual.blocks.0.attn.qkv.weight": 6,
        "audio_tower.layers.0.weight": 7,
    }

    mapped = mapper.apply_dict(weights)

    # New mapping
    assert (
        "language_model.model.layers.0.self_attn.q_proj.weight"
        in mapped
    )

    assert (
        "language_model.model.embed_tokens.weight"
        in mapped
    )

    assert (
        "language_model.lm_head.weight"
        in mapped
    )

    # Existing mapping
    assert (
        "language_model.model.layers.0.self_attn.q_proj.weight"
        in mapped
    )

    assert (
        "language_model.lm_head.weight"
        in mapped
    )

    # Multimodal modules unchanged
    assert (
        "visual.blocks.0.attn.qkv.weight"
        in mapped
    )

    assert (
        "audio_tower.layers.0.weight"
        in mapped
    )