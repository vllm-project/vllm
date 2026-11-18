# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.models.hy_v4.nvidia.attention import (
    HYV4MLAAttention,
    Indexer,
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform


@pytest.fixture
def hy_v4_vl_model():
    from vllm.models.hy_v4.nvidia.vl_model import HYV4VLForConditionalGeneration
    from vllm.transformers_utils.configs.hy_v4 import HYV4VLConfig

    model = HYV4VLForConditionalGeneration.__new__(HYV4VLForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = HYV4VLConfig(
        hidden_size=4, vision_config={"patch_size": 2, "temporal_patch_size": 2}
    )
    model.visual = Mock()
    return model


@pytest.mark.parametrize("video_first", [False, True])
@pytest.mark.parametrize("layout", ["flat", "batched", "list"])
def test_hy_v4_embed_multimodal_preserves_media_order(
    hy_v4_vl_model, video_first, layout
):
    pixels = torch.arange(144, dtype=torch.float32).reshape(12, 12)
    items = [
        (
            "pixel_values",
            "image_grid_thw",
            pixels,
            torch.tensor([[1, 2, 2], [1, 2, 4]]),
        ),
        (
            "pixel_values_videos",
            "video_grid_thw",
            pixels + 144,
            torch.tensor([[3, 2, 2]]),
        ),
    ]
    embeddings = [
        [torch.full((n, 4), float(n)) for n in counts] for counts in ([2, 3], [4])
    ]
    if video_first:
        items.reverse()
        embeddings.reverse()
    kwargs = {}
    for pixel_key, grid_key, values, grid in items:
        if layout == "list":
            values = list(values.split(grid.prod(-1).tolist()))
            grid = list(grid.unbind())
        elif layout == "batched":
            values, grid = values.unsqueeze(0), grid.unsqueeze(0)
        kwargs.update({pixel_key: values, grid_key: grid})
    hy_v4_vl_model.visual.side_effect = embeddings

    actual = hy_v4_vl_model.embed_multimodal(**kwargs)

    expected = [embedding for group in embeddings for embedding in group]
    assert len(actual) == len(expected)
    for output, embedding in zip(actual, expected):
        torch.testing.assert_close(output, embedding)
    for call, (_, _, values, grid) in zip(hy_v4_vl_model.visual.call_args_list, items):
        torch.testing.assert_close(call.args[0], values)
        assert call.args[1] == grid.tolist()


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("invalid", ["patch_width", "grid_width", "missing_grid"])
def test_hy_v4_embed_multimodal_validates_pixel_schema(
    hy_v4_vl_model, modality, invalid
):
    pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
    grid_key = f"{modality}_grid_thw"
    kwargs = {pixel_key: torch.zeros(4, 12), grid_key: torch.tensor([[1, 2, 2]])}
    field = grid_key
    if invalid == "patch_width":
        kwargs[pixel_key] = torch.zeros(4, 13)
        field = pixel_key
    elif invalid == "grid_width":
        kwargs[grid_key] = torch.ones(3, 2, dtype=torch.long)
    else:
        kwargs.pop(grid_key)

    with pytest.raises(ValueError, match=field):
        hy_v4_vl_model.embed_multimodal(**kwargs)
    hy_v4_vl_model.visual.assert_not_called()


@pytest.mark.parametrize(
    "kwargs", [{}, {"pixel_values": None, "pixel_values_videos": None}]
)
def test_hy_v4_embed_multimodal_without_pixels(hy_v4_vl_model, kwargs):
    hy_v4_vl_model.visual = None
    assert hy_v4_vl_model.embed_multimodal(**kwargs) == ()


@pytest.mark.parametrize("key", ["image_embeds", "video_embeds"])
def test_hy_v4_embed_multimodal_rejects_precomputed_embeddings(hy_v4_vl_model, key):
    with pytest.raises(ValueError, match="precomputed"):
        hy_v4_vl_model.embed_multimodal(**{key: torch.zeros(2, 4)})
    hy_v4_vl_model.visual.assert_not_called()


@pytest.mark.parametrize("shape", [(3, 4), (2, 5)])
def test_hy_v4_embed_multimodal_validates_output_shape(hy_v4_vl_model, shape):
    hy_v4_vl_model.visual.return_value = [torch.zeros(shape)]
    with pytest.raises(ValueError, match="visual embeddings"):
        hy_v4_vl_model.embed_multimodal(
            pixel_values=torch.zeros(4, 12), image_grid_thw=torch.tensor([[1, 2, 2]])
        )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires NVIDIA CUDA or AMD ROCm"
)
@pytest.mark.parametrize("role", ["main_attention", "indexer"])
@pytest.mark.parametrize("position_kind", ["text", "mrope_text", "mrope_vision"])
def test_hy_v4_rope_call_sites(default_vllm_config, role: str, position_kind: str):
    """Both HYV4 callers must preserve projection layout and text/MRoPE values."""
    num_tokens, num_heads, rope_dim = 5, 4, 64
    head_dim = 256 if role == "main_attention" else 128
    generator = torch.Generator(device="cuda").manual_seed(0)
    tensor_kwargs = {
        "dtype": torch.bfloat16,
        "device": "cuda",
        "generator": generator,
    }
    query_projection = torch.randn(num_tokens, num_heads * head_dim, **tensor_kwargs)
    key_width = 2048 + 512 + 64 if role == "main_attention" else head_dim + num_heads
    key_projection = torch.randn(num_tokens, key_width, **tensor_kwargs)
    hidden_states = torch.randn(num_tokens, 16, **tensor_kwargs)
    positions = torch.arange(3 * (num_tokens + 1), device="cuda").reshape(
        3, num_tokens + 1
    )[:, :num_tokens]
    rope_parameters = {"rope_theta": 10000000, "rope_type": "default"}
    if position_kind == "text":
        positions = positions[0]
    else:
        rope_parameters.update(mrope_interleaved=True, mrope_section=[12, 10, 10])
        if position_kind == "mrope_text":
            positions[1:] = positions[0]
        assert not positions.is_contiguous()

    rope = get_rope(
        head_size=rope_dim,
        max_position=32,
        is_neox_style=False,
        rope_parameters=rope_parameters,
        dtype=torch.bfloat16,
    ).to(device="cuda")

    def run(rotary_emb):
        if role == "indexer":
            state = SimpleNamespace(
                wq_b=lambda _: (query_projection.clone(), None),
                wk_weights_proj=lambda _: (key_projection.clone(), None),
                k_norm=lambda x: x.contiguous(),
                n_head=num_heads,
                head_dim=head_dim,
                rope_dim=rope_dim,
                quant_block_size=head_dim,
                scale_fmt=None,
                softmax_scale=head_dim**-0.5,
            )
            with patch(
                "vllm.models.hy_v4.nvidia.attention.per_token_group_quant_fp8",
                wraps=per_token_group_quant_fp8,
            ) as quantize:
                _, _, key, weights = Indexer.prepare_inputs(
                    state, hidden_states, hidden_states, positions, rotary_emb
                )
            return quantize.call_args.args[0].clone(), key, weights

        def attention_output(hidden, qr, pos, query, kv, key, output):
            output.copy_(torch.cat((query.flatten(1), key.flatten(1)), dim=-1))

        state = SimpleNamespace(
            q_lora_rank=None,
            q_proj=lambda _: (query_projection.clone(), None),
            kv_a_proj_with_mqa=lambda _: (key_projection.clone()[:, 2048:], None),
            kv_lora_rank=512,
            kv_a_layernorm=lambda x: x,
            num_local_heads=num_heads,
            qk_head_dim=head_dim,
            qk_nope_head_dim=head_dim - rope_dim,
            qk_rope_head_dim=rope_dim,
            rotary_emb=rotary_emb,
            v_head_dim=head_dim + rope_dim // num_heads,
            _indexer_and_attn=attention_output,
            gated_mla=False,
            o_proj=lambda x: (x, None),
        )
        return (HYV4MLAAttention.forward(state, positions, hidden_states),)

    expected = run(rope.forward_native)
    actual = run(rope.forward_cuda)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires NVIDIA CUDA or AMD ROCm"
)
@pytest.mark.parametrize("use_2d_rope", [False, True])
def test_hy_v4_vision_batch_preserves_media_boundaries(dist_init, use_2d_rope):
    from vllm.models.hy_v4.nvidia.vision import HYV4VisionTransformer
    from vllm.transformers_utils.configs.hy_v4 import HYV4VLVisionConfig
    from vllm.transformers_utils.processors.hy_v4 import vision_token_count
    from vllm.utils.torch_utils import set_default_torch_dtype

    config = HYV4VLVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        out_hidden_size=64,
        max_image_size=64,
        projector_type="rmsnorm_mlp_adapter",
        temporal_patch_size=2,
        use_2d_rope=use_2d_rope,
    )
    torch.manual_seed(0)
    with set_default_torch_dtype(torch.bfloat16):
        model = HYV4VisionTransformer(config).to(device="cuda").eval()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(std=0.02)
        pixels = torch.randn(64, 768, device="cuda", dtype=torch.bfloat16)
        grids = [[1, 4, 4], [3, 4, 4]]
        together = model(pixels, grids)
        separate = [model(pixels[:16], grids[:1])[0], model(pixels[16:], grids[1:])[0]]
    for modality, grid, actual, expected in zip(
        ("image", "video"), grids, together, separate
    ):
        assert actual.shape == (vision_token_count(grid, config, modality), 64)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1.6e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires NVIDIA CUDA or AMD ROCm"
)
def test_hy_v4_vision_loads_separate_qkv_shards(dist_init):
    from vllm.models.hy_v4.nvidia.vision import HYV4VisionTransformer
    from vllm.transformers_utils.configs.hy_v4 import HYV4VLVisionConfig

    config = HYV4VLVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        out_hidden_size=64,
        max_image_size=64,
        projector_type="rmsnorm_mlp_adapter",
    )
    model = HYV4VisionTransformer(config)
    shards = [torch.full((128, 128), i + 1.0) for i in range(3)]
    loaded = model.load_weights(
        (f"layers.0.self_attn.{name}_proj.weight", weight)
        for name, weight in zip(("q", "k", "v"), shards)
    )
    assert loaded == {"layers.0.self_attn.qkv.weight"}
    torch.testing.assert_close(model.layers[0].self_attn.qkv.weight, torch.cat(shards))


@pytest.mark.parametrize("modality", ["image", "video", "text"])
def test_hy_v4_mrope_positions_preserve_source_grid_layout(modality):
    from transformers import BatchFeature

    from vllm.models.hy_v4.nvidia.vl_model import HYV4VLForConditionalGeneration
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        MultiModalFieldConfig,
        MultiModalKwargsItems,
        PlaceholderRange,
    )
    from vllm.transformers_utils.configs.hy_v4 import HYV4VLConfig

    config = HYV4VLConfig()
    start, end, pad = (
        config.image_start_token_id,
        config.image_end_token_id,
        config.image_token_id,
    )
    if modality == "text":
        tokens, features = [10, 11, 12], []
        expected = torch.arange(3).repeat(3, 1)
    else:
        if modality == "image":
            grid = [1, 4, 4]
            tokens = [10, start, *([pad] * 6), end, 11]
            expected = torch.tensor(
                [
                    [0, 1, 2, 2, 2, 2, 2, 2, 5, 6],
                    [0, 1, 2, 2, 2, 3, 3, 3, 5, 6],
                    [0, 1, 2, 3, 4, 2, 3, 4, 5, 6],
                ]
            )
        else:
            grid = [2, 4, 4]
            pad = config.video_token_id
            tokens = [
                config.video_start_token_id,
                10,
                start,
                *([pad] * 6),
                end,
                11,
                start,
                *([pad] * 6),
                end,
                config.video_end_token_id,
            ]
            expected = torch.tensor(
                [
                    [0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7, 8, 9, 9, 9, 9, 9, 9, 12, 13],
                    [0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7, 8, 9, 9, 9, 10, 10, 10, 12, 13],
                    [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 9, 10, 11, 12, 13],
                ]
            )
        field = f"{modality}_grid_thw"
        data = MultiModalKwargsItems.from_hf_inputs(
            BatchFeature({field: torch.tensor([grid])}),
            {field: MultiModalFieldConfig.batched(modality)},
        )
        features = [
            MultiModalFeatureSpec(
                data[modality][0], modality, "test", PlaceholderRange(0, len(tokens))
            )
        ]
    actual, delta = HYV4VLForConditionalGeneration.get_mrope_input_positions(
        SimpleNamespace(config=config), tokens, features
    )
    torch.testing.assert_close(actual, expected)
    assert delta == int(expected.max()) + 1 - len(tokens)
    if modality == "image":
        malformed = tokens[:2] + [10] + tokens[2:]
        with pytest.raises(ValueError, match="placeholders"):
            HYV4VLForConditionalGeneration.get_mrope_input_positions(
                SimpleNamespace(config=config), malformed, features
            )


@pytest.mark.parametrize("modality", ["image", "video"])
def test_hy_v4_text_positions_shifted_by_delta_match_after_media(modality):
    """A P/D decode instance without the media (--language-model-only) rebuilds
    positions from the text-only ones plus the delta prefill reports. From the
    last media marker on, those must be the positions prefill computed."""
    from transformers import BatchFeature

    from vllm.models.hy_v4.nvidia.vl_model import HYV4VLForConditionalGeneration
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        MultiModalFieldConfig,
        MultiModalKwargsItems,
        PlaceholderRange,
    )
    from vllm.transformers_utils.configs.hy_v4 import HYV4VLConfig

    config = HYV4VLConfig()
    start, end = config.image_start_token_id, config.image_end_token_id
    if modality == "image":
        grid = [1, 4, 4]
        media = [start, *([config.image_token_id] * 6), end]
    else:
        grid = [2, 4, 4]
        frame = [start, *([config.video_token_id] * 6), end]
        media = [config.video_start_token_id, 10, *frame, 11, *frame]
        media.append(config.video_end_token_id)
    tokens = [20, 21, *media, 30, 31, 32]
    last_media_token = 2 + len(media) - 1

    field = f"{modality}_grid_thw"
    data = MultiModalKwargsItems.from_hf_inputs(
        BatchFeature({field: torch.tensor([grid])}),
        {field: MultiModalFieldConfig.batched(modality)},
    )
    features = [
        MultiModalFeatureSpec(
            data[modality][0], modality, "test", PlaceholderRange(0, len(tokens))
        )
    ]
    model = SimpleNamespace(config=config)
    prefill, delta = HYV4VLForConditionalGeneration.get_mrope_input_positions(
        model, tokens, features
    )
    text, text_delta = HYV4VLForConditionalGeneration.get_mrope_input_positions(
        model, tokens, []
    )

    assert delta < 0
    assert text_delta == 0
    shifted = (text + delta).clamp(min=0)
    torch.testing.assert_close(
        shifted[:, last_media_token:], prefill[:, last_media_token:]
    )
    # The first decoded token continues where the prompt left off.
    assert len(tokens) + delta == int(prefill.max()) + 1


@pytest.mark.parametrize(
    "vision_enabled,weight_name",
    [(False, "visual_extra.weight"), (True, "vit.unknown.weight")],
)
def test_hy_v4_vl_weight_loader_keeps_unexpected_weights_strict(
    hy_v4_vl_model, vision_enabled, weight_name
):
    hy_v4_vl_model.visual = torch.nn.Module() if vision_enabled else None
    with pytest.raises(ValueError, match="There is no module or parameter"):
        hy_v4_vl_model.load_weights([(weight_name, torch.zeros(1))])


@pytest.mark.parametrize("vision_enabled", [False, True])
def test_hy_v4_vl_weight_loader_delegates_text_and_mtp_filtering(vision_enabled):
    from torch import nn

    from vllm.models.hy_v4.nvidia.model import HYV4ForCausalLM
    from vllm.models.hy_v4.nvidia.vl_model import HYV4VLForConditionalGeneration

    text = HYV4ForCausalLM.__new__(HYV4ForCausalLM)
    nn.Module.__init__(text)
    text.config = SimpleNamespace(
        num_hidden_layers=0, num_nextn_predict_layers=1, tie_word_embeddings=False
    )
    text.model = nn.Linear(2, 2, bias=False)
    text.lm_head = nn.Linear(2, 1, bias=False)
    model = HYV4VLForConditionalGeneration.__new__(HYV4VLForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = text
    model.visual = nn.Module() if vision_enabled else None
    if vision_enabled:
        model.visual.embeddings = nn.Linear(2, 2, bias=False)
    loaded = model.load_weights(
        iter(
            [
                ("model.weight", torch.ones(2, 2)),
                ("vit.embeddings.weight", torch.full((2, 2), 2.0)),
                ("model.mtp_layers.0.weight", torch.zeros(2, 2)),
                ("lm_head.weight", torch.full((1, 2), 3.0)),
            ]
        )
    )
    assert loaded == {
        "language_model.model.weight",
        "language_model.lm_head.weight",
    } | ({"visual.embeddings.weight"} if vision_enabled else set())
    torch.testing.assert_close(text.model.weight, torch.ones(2, 2))
    torch.testing.assert_close(text.lm_head.weight, torch.full((1, 2), 3.0))
    assert text.hf_to_vllm_mapper is HYV4ForCausalLM.hf_to_vllm_mapper
    if vision_enabled:
        torch.testing.assert_close(
            model.visual.embeddings.weight, torch.full((2, 2), 2.0)
        )


def _make_vision_model(dist_init, **overrides):
    from vllm.models.hy_v4.nvidia.vision import HYV4VisionTransformer
    from vllm.transformers_utils.configs.hy_v4 import HYV4VLVisionConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    kwargs = dict(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        out_hidden_size=64,
        max_image_size=128,
        projector_type="rmsnorm_mlp_adapter",
        temporal_patch_size=2,
        use_2d_rope=True,
    )
    kwargs.update(overrides)
    config = HYV4VLVisionConfig(**kwargs)
    torch.manual_seed(0)
    with set_default_torch_dtype(torch.bfloat16):
        model = HYV4VisionTransformer(config).to(device="cuda").eval()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(std=0.02)
    return model, config


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires NVIDIA CUDA or AMD ROCm"
)
@pytest.mark.parametrize("use_2d_rope", [False, True])
def test_hy_v4_vision_packed_metadata_matches_eager(dist_init, use_2d_rope):
    import numpy as np

    model, config = _make_vision_model(dist_init, use_2d_rope=use_2d_rope)
    pixels = torch.randn(64, 768, device="cuda", dtype=torch.bfloat16)
    grids = [[1, 4, 4], [3, 4, 4]]

    metadata = model.prepare_encoder_cudagraph_metadata(grids, device=pixels.device)

    # cu_seqlens: one varlen sequence per frame, like the eager path.
    expected_cu = np.cumsum([0, 16, 16, 16, 16], dtype=np.int32)
    torch.testing.assert_close(
        metadata["cu_seqlens"].cpu(), torch.from_numpy(expected_cu)
    )
    assert metadata["max_seqlen"].item() == 16

    # pos_embeds: the eager embedding must decompose into conv + pos_embeds,
    # bitwise (metadata prep reuses the eager per-grid interpolation).
    with torch.no_grad():
        eager = model.embeddings(pixels, grids)
        packed = model.embeddings.forward_with_pos_embeds(
            pixels, metadata["pos_embeds"]
        )
    assert torch.equal(packed, eager)

    # RoPE tables: identical to what the eager encoder computes.
    if use_2d_rope:
        cos, sin = model._rope_cos_sin(grids)
        assert torch.equal(metadata["rope_cos"], cos)
        assert torch.equal(metadata["rope_sin"], sin)
    else:
        assert metadata["rope_cos"] is None and metadata["rope_sin"] is None


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires NVIDIA CUDA or AMD ROCm"
)
@pytest.mark.parametrize("projector_type", ["rmsnorm_mlp_adapter", None])
@pytest.mark.parametrize("use_2d_rope", [False, True])
@pytest.mark.parametrize("temporal_patch_size", [1, 2])
def test_hy_v4_vision_forward_packed_matches_eager(
    dist_init, projector_type, use_2d_rope, temporal_patch_size
):
    """Packed ViT hidden states must be bitwise identical to eager."""
    overrides = {
        "projector_type": projector_type,
        "use_2d_rope": use_2d_rope,
        "temporal_patch_size": temporal_patch_size,
    }
    if projector_type is None:
        overrides["cat_extra_token"] = 1
    model, config = _make_vision_model(dist_init, **overrides)
    # image, temporally padded video (t % temporal_patch_size != 0), even video
    grids = [[1, 4, 4], [3, 4, 4], [2, 8, 4]]
    pixels = torch.randn(
        sum(t * h * w for t, h, w in grids), 768, device="cuda", dtype=torch.bfloat16
    )

    with torch.no_grad():
        metadata = model.prepare_encoder_cudagraph_metadata(grids, device=pixels.device)
        patches = model.embeddings._patch_conv(
            model.input_norm(pixels, model.dtype)
        ).squeeze(0)
        packed_hidden = model.forward_packed(patches, **metadata)
        eager_hidden = model._encode(pixels, grids)
        eager_full = model(pixels, grids)

    assert packed_hidden.shape[1] == eager_hidden.shape[1]
    assert torch.equal(packed_hidden, eager_hidden)

    # The per-item merger on the packed hidden states must reproduce the
    # eager per-item outputs bitwise (same ops, same shapes).
    for expected, states in zip(
        eager_full, model._iter_item_states(packed_hidden, grids)
    ):
        assert torch.equal(model.perceive(states), expected)


def _make_vl_cudagraph_model(visual, vision_config, max_frames):
    from vllm.models.hy_v4.nvidia.vl_model import HYV4VLForConditionalGeneration
    from vllm.transformers_utils.configs.hy_v4 import HYV4VLConfig

    model = HYV4VLForConditionalGeneration.__new__(HYV4VLForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = HYV4VLConfig(
        hidden_size=vision_config.out_hidden_size,
        vision_config=vision_config.to_diff_dict(),
    )
    model.visual = visual
    model.multimodal_config = None
    model.get_max_frames_per_video = lambda: max_frames
    return model


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires NVIDIA CUDA")
@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("use_newline", [True, False])
def test_hy_v4_encoder_cudagraph_capture_replay(dist_init, modality, use_newline):
    """Real torch.cuda.graph capture + replay must be bitwise equal to eager."""
    visual, vision_config = _make_vision_model(
        dist_init, use_image_newline_token=use_newline
    )
    model = _make_vl_cudagraph_model(visual, vision_config, max_frames=4)

    cg_config = model.get_encoder_cudagraph_config()
    assert set(cg_config.modalities) == {"image", "video"}
    assert cg_config.out_hidden_size == 64

    budget, max_batch_size, max_frames_per_batch = 256, 2, 4
    device = torch.device("cuda")
    capture = model.prepare_encoder_cudagraph_capture_inputs(
        budget,
        max_batch_size,
        max_frames_per_batch,
        device,
        torch.bfloat16,
        axis_keys=(modality,),
    )
    values = capture.values

    # Mirror EncoderCudaGraphManager._capture_budget_graph.
    with torch.inference_mode():
        output = model.encoder_cudagraph_forward({**values})
        output_buffer = torch.empty_like(output)
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        output = model.encoder_cudagraph_forward({**values})
        output_buffer.copy_(output)

    def replay(mm_kwargs):
        replay_buffers = model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs, max_batch_size, max_frames_per_batch
        )
        for key, buf in values.items():
            src = replay_buffers.values.get(key)
            if src is None:
                continue
            if src.ndim == 0:
                buf.copy_(src)
                continue
            padding = cg_config.padding_logics.get(key)
            if padding is not None:
                padding(buf, src)
            else:
                buf.zero_()
                buf[: src.shape[0]].copy_(src)
        graph.replay()
        return output_buffer

    grid_key = f"{modality}_grid_thw"
    pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
    # t % temporal_patch_size != 0 exercises the repeat-last-frame pad.
    grids = [[1, 4, 4], [1, 8, 4]] if modality == "image" else [[3, 4, 4]]
    pixels = torch.randn(
        sum(t * h * w for t, h, w in grids), 768, device=device, dtype=torch.bfloat16
    )
    mm_kwargs = {pixel_key: pixels, grid_key: torch.tensor(grids)}

    with torch.inference_mode():
        graphed = replay(mm_kwargs)
        dest: dict[int, torch.Tensor] = {}
        model.postprocess_encoder_output(
            {"default": graphed},
            list(range(len(grids))),
            [],
            dest,
            batch_mm_kwargs=mm_kwargs,
        )
        eager = visual(pixels, grids)

    assert len(dest) == len(grids)
    for i, expected in enumerate(eager):
        assert dest[i].shape == expected.shape
        assert torch.equal(dest[i], expected)

    # Replaying a different, smaller batch into the same graph must not
    # corrupt results (zero padding + trailing cu_seqlens padding sequence).
    small_grids = [grids[0]]
    small_pixels = pixels[: grids[0][0] * grids[0][1] * grids[0][2]]
    small_kwargs = {pixel_key: small_pixels, grid_key: torch.tensor(small_grids)}
    with torch.inference_mode():
        graphed = replay(small_kwargs)
        dest = {}
        model.postprocess_encoder_output(
            {"default": graphed}, [0], [], dest, batch_mm_kwargs=small_kwargs
        )
        expected = visual(small_pixels, small_grids)[0]
    assert torch.equal(dest[0], expected)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires NVIDIA CUDA or AMD ROCm"
)
def test_hy_v4_encoder_eager_forward_matches_visual(dist_init):
    """The eager fallback must return the same hidden states as _encode."""
    visual, vision_config = _make_vision_model(dist_init)
    model = _make_vl_cudagraph_model(visual, vision_config, max_frames=4)
    grids = [[1, 4, 4], [3, 4, 4]]
    pixels = torch.randn(64, 768, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        hidden = model.encoder_eager_forward(
            {"pixel_values": pixels[:16], "image_grid_thw": torch.tensor(grids[:1])}
        )
        reference = visual._encode(pixels[:16], grids[:1])
    assert torch.equal(hidden, reference)


def test_hy_v4_encoder_cudagraph_requires_flash_attn(hy_v4_vl_model):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    hy_v4_vl_model.visual = Mock(
        attn_backend=AttentionBackendEnum.FLASHINFER, rotary_pos_emb=None
    )
    with pytest.raises(ValueError, match="FLASH_ATTN"):
        hy_v4_vl_model.get_encoder_cudagraph_config()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires NVIDIA CUDA")
def test_hy_v4_encoder_cudagraph_manager_end_to_end(dist_init):
    """Full EncoderCudaGraphManager flow: capture, replay, eager fallback."""
    from vllm.platforms import current_platform as platform
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    visual, vision_config = _make_vision_model(dist_init)
    model = _make_vl_cudagraph_model(visual, vision_config, max_frames=4)

    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_token_budgets=[256],
            encoder_cudagraph_max_vision_items_per_batch=2,
            encoder_cudagraph_max_frames_per_batch=4,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        model_config=SimpleNamespace(
            max_model_len=128,
            multimodal_config=SimpleNamespace(
                get_limit_per_prompt=lambda modality: 4,
                mm_encoder_tp_mode="weights",
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    manager = EncoderCudaGraphManager(
        vllm_config, torch.device("cuda"), torch.bfloat16, model
    )
    # The engine captures on a non-default stream; mirror that here.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        manager.capture(platform.graph_pool_handle())
    stream.synchronize()

    cases = [
        # (modality, grids); the last video exceeds the captured budget and
        # must fall back to the eager encoder path.
        ("image", [[1, 4, 4], [1, 8, 4]]),
        ("video", [[3, 4, 4]]),
        ("video", [[9, 8, 8]]),
    ]
    with torch.inference_mode():
        for modality, grids in cases:
            pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
            pixels = torch.randn(
                sum(t * h * w for t, h, w in grids),
                768,
                device="cuda",
                dtype=torch.bfloat16,
            )
            mm_kwargs = {pixel_key: pixels, f"{modality}_grid_thw": torch.tensor(grids)}
            outputs = manager.execute(mm_kwargs)
            expected = visual(pixels, grids)
            assert len(outputs) == len(expected)
            for actual, reference in zip(outputs, expected):
                assert actual.shape == reference.shape
                assert torch.equal(actual, reference)

    stats = manager.get_cumulative_stats()
    assert stats["graph_hits"] == 3  # two image items + one video item
    assert stats["graph_misses"] == 1  # oversized video went eager
