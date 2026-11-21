# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ECTransferConfig,
    ModelConfig,
    VllmConfig,
)
from vllm.distributed.ec_transfer.ec_connector.utils import (
    PlaceholderMetadataResolver,
    collect_ec_item_metadata,
)
from vllm.exceptions import PromptUpdateMismatchError
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.multimodal.utils import encode_image_url, encode_video_url
from vllm.platforms import current_platform
from vllm.renderers import ChatParams, renderer_from_config
from vllm.renderers.chat_utils.item_tracker import _get_embeds_data
from vllm.transformers_utils.processors.kimi_k3 import (
    get_kimi_k3_frame_resize,
    get_kimi_k3_max_video_frames,
    get_kimi_k3_max_video_tokens,
    plan_kimi_k3_video,
)
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    KimiK25FusedVisionProcessor,
)

from ...utils import build_model_context

_MODEL_ID = "moonshotai/Kimi-K3"


@pytest.fixture(scope="module")
def image_processors():
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    k25_fused = getattr(processor.info, "image_processor")  # noqa: B009
    assert isinstance(k25_fused, KimiK25FusedVisionProcessor)
    k3_hf = AutoImageProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
    return k25_fused, k3_hf


def _noise_image(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize("with_embeddings", [False, True])
def test_image_metadata_only_roundtrip(with_embeddings: bool):
    """EC metadata preserves original dimensions and repeated-image placeholders."""
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 4})
    mm_config = ctx.model_config.multimodal_config
    mm_config.enable_mm_embeds = True
    mm_config.allow_missing_mm_embeddings = True
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [_noise_image(w, h) for w, h in [(310, 470), (7300, 30), (13, 9)]]
    images.append(images[0])
    prompt = processor.dummy_inputs.get_dummy_text({"image": len(images)})
    raw = processor(prompt, mm_items=processor.info.parse_mm_data({"image": images}))
    features = [
        MultiModalFeatureSpec(
            data=item,
            modality="image",
            identifier=str(index),
            mm_position=position,
        )
        for index, (item, position) in enumerate(
            zip(raw["mm_kwargs"]["image"], raw["mm_placeholders"]["image"])
        )
    ]
    published = collect_ec_item_metadata(
        features, PlaceholderMetadataResolver(ctx.model_config)
    )
    metadata = json.loads(
        json.dumps([entry["metadata"] for entry in published.values()])
    )
    for image, item in zip(images, metadata):
        assert item == {"image_sizes": list(image.size)}

    if with_embeddings:
        embeddings = [
            torch.full(
                (position.get_num_embeds(), ctx.model_config.get_hidden_size()),
                float(index),
            )
            for index, position in enumerate(raw["mm_placeholders"]["image"])
        ]
        remote_items = [
            {**item, "image_embeds": embedding}
            for item, embedding in zip(metadata, embeddings)
        ]
    else:
        remote_items = metadata
    remote_data = _get_embeds_data("image", remote_items, processor)
    remote = processor(
        prompt, mm_items=processor.info.parse_mm_data({"image": remote_data})
    )
    assert remote["prompt_token_ids"] == raw["prompt_token_ids"]
    assert remote["mm_placeholders"] == raw["mm_placeholders"]
    assert "pixel_values" not in remote["mm_kwargs"].get_data()
    if with_embeddings:
        for item, embedding in zip(remote["mm_kwargs"]["image"], embeddings):
            assert item is not None
            torch.testing.assert_close(item["image_embeds"].data, embedding)

    mm_config.allow_missing_mm_embeddings = False
    ordinary_processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    with pytest.raises(ValueError, match="should contain the fields"):
        ordinary_processor.info.parse_mm_data(
            {"image": {"image_sizes": remote_data["image_sizes"]}}
        )


def test_image_metadata_only_chat():
    """image_embeds inputs preserve K3's native chat tokens and ranges."""
    config = VllmConfig(
        model_config=ModelConfig(
            _MODEL_ID,
            tokenizer_mode="kimi_k3",
            trust_remote_code=True,
            enable_mm_embeds=True,
            limit_mm_per_prompt={"image": 3},
            mm_processor_cache_gb=0,
        ),
        # Rendering needs no KV cache; keep CPU MLA and hybrid-cache defaults aligned.
        cache_config=CacheConfig(enable_prefix_caching=False),
        ec_transfer_config=ECTransferConfig(
            ec_connector="ECExampleConnector", ec_role="ec_consumer"
        ),
        device_config=DeviceConfig(device="cpu"),
    )
    images = [_noise_image(31, 47), _noise_image(73, 30), _noise_image(31, 47)]
    raw_parts = [
        {"type": "image_url", "image_url": {"url": encode_image_url(image)}}
        for image in images
    ]
    metadata_parts = [
        {
            "type": "image_embeds",
            "image_embeds": {
                "image_sizes": list(image.size),
            },
        }
        for image in images
    ]
    renderer = renderer_from_config(config)
    try:
        results = []
        for parts in (raw_parts, metadata_parts):
            _, prompts = renderer.render_chat(
                [[{"role": "user", "content": parts}]], ChatParams()
            )
            results.append(prompts[0])
    finally:
        renderer.shutdown()
    raw, remote = results
    assert remote["prompt_token_ids"] == raw["prompt_token_ids"]
    assert remote["mm_placeholders"] == raw["mm_placeholders"]


_CHAT_IMAGES = [_noise_image(31, 47), _noise_image(73, 30)]


def _image_part(index: int) -> dict:
    return {"type": "image", "image": _CHAT_IMAGES[index]}


def _two_image_config() -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(
            _MODEL_ID,
            tokenizer_mode="kimi_k3",
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 2},
            mm_processor_cache_gb=0,
        ),
        cache_config=CacheConfig(enable_prefix_caching=False),
        device_config=DeviceConfig(device="cpu"),
    )


def _render_chat(messages: list[dict]) -> tuple[dict, int]:
    """Render HF-processor-style messages as a chat request with image URLs."""
    config = _two_image_config()

    def to_request(content):
        if isinstance(content, str):
            return content
        return [
            {"type": "image_url", "image_url": {"url": encode_image_url(p["image"])}}
            if p["type"] == "image"
            else p
            for p in content
        ]

    renderer = renderer_from_config(config)
    try:
        _, prompts = renderer.render_chat(
            [[{**m, "content": to_request(m["content"])} for m in messages]],
            ChatParams(),
        )
    finally:
        renderer.shutdown()
    return prompts[0], config.model_config.hf_config.media_placeholder_token_id


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare "},
                        _image_part(0),
                        {"type": "text", "text": " with "},
                        _image_part(1),
                        {"type": "text", "text": " briefly."},
                        {"type": "text", "text": " Thanks."},
                    ],
                }
            ],
            id="interleaved",
        ),
        pytest.param(
            [
                {"role": "user", "content": "What does <|sep|> mean in XTML?"},
                {"role": "assistant", "content": "It ends a tag."},
                {
                    "role": "user",
                    "content": [
                        _image_part(0),
                        {"type": "text", "text": "Is it in this image?"},
                    ],
                },
            ],
            id="markup_text_in_history",
        ),
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe"},
                        {"type": "text", "text": " nothing."},
                    ],
                },
            ],
            id="text_parts_without_images",
        ),
    ],
)
def test_chat_prompt_matches_hf_processor(messages: list[dict]):
    """A chat request with images gives the model the checkpoint's own prompt:
    parts in request order, no separators, message text tokenized as text."""
    prompt, pad = _render_chat(messages)

    hf = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
    reference_ids = hf(messages=messages, return_tensors="pt")["input_ids"]
    # The HF processor leaves one pad per image for the model to expand.
    num_pads = (
        hf.image_processor.media_tokens_calculator(_image_part(i))
        for i in range(len(_CHAT_IMAGES))
    )
    expected_ids: list[int] = []
    for token_id in reference_ids.flatten().tolist():
        expected_ids += [pad] * next(num_pads) if token_id == pad else [token_id]
    assert prompt["prompt_token_ids"] == expected_ids


def test_chat_placeholder_text_is_not_an_image_slot():
    """Text that spells the image placeholder stays text next to an image."""
    literal, _ = _render_chat(
        [
            {
                "role": "user",
                "content": [
                    _image_part(0),
                    {"type": "text", "text": "Is <|kimi_image_placeholder|> a word?"},
                ],
            }
        ]
    )
    plain, _ = _render_chat(
        [
            {
                "role": "user",
                "content": [_image_part(0), {"type": "text", "text": "Is it a word?"}],
            }
        ]
    )
    assert literal["mm_placeholders"] == plain["mm_placeholders"]
    assert len(literal["prompt_token_ids"]) > len(plain["prompt_token_ids"])


_TEXT_PROMPT = "Compare <|kimi_image_placeholder|><|kimi_image_placeholder|>\nbriefly."


def _render_cmpl(prompt: dict) -> dict:
    """Render a completion-style prompt that carries both chat images."""
    renderer = renderer_from_config(_two_image_config())
    try:
        return renderer.render_cmpl(
            [{**prompt, "multi_modal_data": {"image": _CHAT_IMAGES}}]
        )[0]
    finally:
        renderer.shutdown()


def test_text_prompt_finds_adjacent_image_placeholders():
    """Placeholders that touch each other or other text each get their image."""
    prompt = _render_cmpl({"prompt": _TEXT_PROMPT})
    assert len(prompt["mm_placeholders"]["image"]) == len(_CHAT_IMAGES)


def test_token_prompt_is_not_retokenized_to_find_images():
    """Token ids are used as given: when they hold no image placeholder, the
    request is rejected instead of rebuilding the prompt from its text."""
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, trust_remote_code=True)
    # Encoded in one piece, the placeholders merge with the text around them.
    token_ids = tokenizer.encode(_TEXT_PROMPT, add_special_tokens=False)
    with pytest.raises(PromptUpdateMismatchError):
        _render_cmpl({"prompt_token_ids": token_ids})


@pytest.mark.parametrize(
    ("width", "height"),
    [(310, 470), (7300, 30), (13, 9)],
)
def test_fused_image_processor_matches_hf(image_processors, width: int, height: int):
    fused, hf = image_processors
    image = _noise_image(width, height)
    expected = hf.preprocess([{"type": "image", "image": image}], return_tensors="pt")
    actual = fused.preprocess([{"type": "image", "image": image}], return_tensors="pt")

    torch.testing.assert_close(actual["grid_thws"], expected["grid_thws"])
    torch.testing.assert_close(actual["pixel_values"], expected["pixel_values"])
    assert fused.media_tokens_calculator(
        {"type": "image", "image": image}
    ) == hf.media_tokens_calculator({"type": "image", "image": image})


@pytest.mark.usefixtures("default_vllm_config")
def test_mm_device_do_normalize():
    device = current_platform.device_type
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 2})
    assert ctx.model_config.multimodal_config.mm_device_do_normalize

    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [
        Image.new("RGB", (310, 470), color=(17, 89, 231)),
        Image.new("RGB", (480, 320), color=(201, 13, 127)),
    ]
    prompt = processor.info.get_hf_config().image_placeholder * len(images)
    mm_items = processor.info.parse_mm_data({"image": images})

    normalized_inputs = processor(prompt, mm_items=mm_items)
    normalized_values = normalized_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert isinstance(normalized_values, torch.Tensor)

    ctx.model_config.multimodal_config.mm_device_do_normalize = True
    raw_inputs = processor(prompt, mm_items=mm_items)
    raw_values = raw_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert isinstance(raw_values, torch.Tensor)
    assert raw_values.dtype == torch.uint8

    input_norm = build_mm_input_norm(ctx.model_config).to(device)
    output = input_norm(raw_values.flatten(1).to(device), normalized_values.dtype)
    torch.testing.assert_close(
        output, normalized_values.flatten(1).to(device), rtol=1.6e-2, atol=1e-5
    )


@pytest.fixture(scope="module")
def k3_processor():
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 1, "video": 1})
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


def _video_text(processor, out) -> str:
    """The decoded prompt span of the single video placeholder."""
    [placeholder] = out["mm_placeholders"]["video"]
    start = placeholder.offset
    token_ids = out["prompt_token_ids"][start : start + placeholder.length]
    return processor.info.get_tokenizer().decode(token_ids, skip_special_tokens=False)


@pytest.mark.parametrize(
    ("width", "height", "num_frames", "fps", "expected"),
    [
        pytest.param(640, 480, 60, 30.0, 16, id="sample-fps"),
        pytest.param(320, 240, 50, 5.0, 50, id="slow-source"),
        pytest.param(1805, 1805, 400, 8.0, 36, id="token-budget"),
        pytest.param(28, 28, 1_000_000, 30.0, 768, id="frame-cap"),
    ],
)
def test_video_plan_stays_within_budget(
    k3_processor, width: int, height: int, num_frames: int, fps: float, expected: int
):
    cfg = k3_processor.info.image_processor.media_proc_cfg
    indices = plan_kimi_k3_video(num_frames, fps, width, height, cfg)
    assert indices == (
        np.linspace(0, num_frames - 1, expected).round().astype(int).tolist()
    )

    resize = get_kimi_k3_frame_resize(width, height, cfg)
    patch = cfg["patch_size"]
    frame_patches = ((resize["new_width"] + resize["pad_width"]) // patch) * (
        (resize["new_height"] + resize["pad_height"]) // patch
    )
    num_chunks = -(-expected // cfg["temporal_merge_kernel_size"])
    assert expected * frame_patches <= cfg["in_patch_limit_video"]
    assert num_chunks * resize["num_tokens"] <= get_kimi_k3_max_video_tokens(cfg)


def test_dummy_video_fills_video_budget(k3_processor):
    """Profiling must reach the per-video token budget the encoder cache uses."""
    info = k3_processor.info
    cfg = info.image_processor.media_proc_cfg
    size = info.get_dummy_video_frame_size()
    num_frames = get_kimi_k3_max_video_frames(size.width, size.height, cfg)
    num_chunks = -(-num_frames // cfg["temporal_merge_kernel_size"])
    tokens = (
        num_chunks
        * get_kimi_k3_frame_resize(size.width, size.height, cfg)["num_tokens"]
    )
    max_tokens = info.get_mm_max_tokens_per_item(8192, {"image": 1, "video": 1})
    assert tokens == max_tokens["video"] == get_kimi_k3_max_video_tokens(cfg)


def test_video_prompt_and_inputs(k3_processor):
    """A video becomes one block of timestamped 4-frame chunks."""
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (12, 112, 84, 3), dtype=np.uint8)
    metadata = {"fps": 4.0, "total_num_frames": 12, "frames_indices": list(range(12))}
    prompt = k3_processor.dummy_inputs.get_dummy_text({"video": 1})
    out = k3_processor(
        prompt,
        mm_items=k3_processor.info.parse_mm_data({"video": [(frames, metadata)]}),
    )

    item = out["mm_kwargs"]["video"][0]
    assert item["video_grid_thws"].data.tolist() == [[4, 8, 6]] * 3
    assert item["video_chunk_timestamps"].data.tolist() == [0, 1000, 2000]
    assert item["video_pixel_values"].data.shape[0] == 12 * 8 * 6
    assert item["video_pixel_values"].data.dtype == torch.uint8
    pads = "<|media_pad|>" * 12
    assert _video_text(k3_processor, out) == (
        "<|media_begin|>video 84x112<|media_content|>"
        f"00:00:00.000{pads}00:00:01.000{pads}00:00:02.000{pads}"
        "<|media_end|>"
    )
    assert out["mm_placeholders"]["video"][0].get_num_embeds() == 36


def test_video_subsamples_decoded_frames(k3_processor):
    """Frames decoded beyond the plan (e.g. by another loader) are subsampled."""
    frames = np.zeros((30, 56, 56, 3), dtype=np.uint8)
    metadata = {"fps": 30.0, "total_num_frames": 30, "frames_indices": list(range(30))}
    prompt = k3_processor.dummy_inputs.get_dummy_text({"video": 1})
    out = k3_processor(
        prompt,
        mm_items=k3_processor.info.parse_mm_data({"video": [(frames, metadata)]}),
    )

    # 8 fps keeps 8 of 30 frames, at positions 0, 4, 8, 12, 17, 21, 25, 29.
    item = out["mm_kwargs"]["video"][0]
    assert item["video_grid_thws"].data.tolist() == [[4, 4, 4]] * 2
    assert item["video_chunk_timestamps"].data.tolist() == [0, 566]
    pads = "<|media_pad|>" * 4
    assert _video_text(k3_processor, out) == (
        "<|media_begin|>video 56x56<|media_content|>"
        f"00:00:00.000{pads}00:00:00.566{pads}"
        "<|media_end|>"
    )


def test_video_chat():
    """A `video_url` part renders K3's video block through the chat path."""
    config = VllmConfig(
        model_config=ModelConfig(
            _MODEL_ID,
            tokenizer_mode="kimi_k3",
            trust_remote_code=True,
            limit_mm_per_prompt={"video": 1},
            mm_processor_cache_gb=0,
        ),
        # Rendering needs no KV cache; keep CPU MLA and hybrid-cache defaults aligned.
        cache_config=CacheConfig(enable_prefix_caching=False),
        device_config=DeviceConfig(device="cpu"),
    )
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (6, 56, 56, 3), dtype=np.uint8)
    content = [
        {"type": "video_url", "video_url": {"url": encode_video_url(frames)}},
        {"type": "text", "text": "Describe the video."},
    ]
    renderer = renderer_from_config(config)
    try:
        _, prompts = renderer.render_chat(
            [[{"role": "user", "content": content}]], ChatParams()
        )
        # A JPEG frame sequence carries no frame rate, so it counts as 1 fps.
        pads = "<|media_pad|>" * 4
        assert _video_text(renderer.get_mm_processor(), prompts[0]) == (
            "<|media_begin|>video 56x56<|media_content|>"
            f"00:00:00.000{pads}00:00:04.000{pads}"
            "<|media_end|>"
        )
    finally:
        renderer.shutdown()
