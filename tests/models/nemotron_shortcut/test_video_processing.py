# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import TokenizersBackend
from transformers.models.pixtral import PixtralImageProcessor, PixtralProcessor

from vllm.config import ModelConfig
from vllm.models.nemotron_shortcut.nvidia.vision import (
    NemotronShortcutDummyInputsBuilder,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache.base import MultiModalProcessorOnlyCache

from .configuration_nemotron4 import Nemotron4Config


def processor(
    tmp_path,
    model_patch_size=1,
    *,
    image_edge=56,
    video_target_num_patches=16,
    video_maintain_aspect_ratio=False,
    mm_processor_kwargs=None,
    **video_options,
):
    special_tokens = ["<unk>", "<img>", "<image_break>", "</img>"]
    backend = Tokenizer(BPE(unk_token="<unk>"))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = ByteLevelDecoder()
    backend.train_from_iterator(
        ["\n<video>\nQuestion", "\nThis is a video:\n", "This is a video:\n"],
        BpeTrainer(
            vocab_size=512,
            special_tokens=special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
        ),
    )
    tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        unk_token="<unk>",
        additional_special_tokens=special_tokens[1:],
    )
    PixtralProcessor(
        image_processor=PixtralImageProcessor(
            size={"longest_edge": image_edge}, patch_size={"height": 28, "width": 28}
        ),
        tokenizer=tokenizer,
        patch_size=14,
        spatial_merge_size=2,
        image_token="<img>",
        image_break_token="<image_break>",
        image_end_token="</img>",
    ).save_pretrained(tmp_path)
    Nemotron4Config.register_for_auto_class()
    Nemotron4Config(
        text_config=dict(
            layer_types=["mlp"],
            hidden_size=8,
            vocab_size=len(tokenizer),
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
        ),
        vision_config=dict(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            patch_size=14,
            image_size=56,
        ),
        image_token_id=1,
        spatial_merge_size=2,
        video_temporal_patch_size=model_patch_size,
        video_maintain_aspect_ratio=video_maintain_aspect_ratio,
        **(
            {"video_target_num_patches": video_target_num_patches}
            if video_target_num_patches is not None
            else {}
        ),
    ).save_pretrained(tmp_path)
    return MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(
            str(tmp_path),
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 1, "video": 1},
            mm_processor_kwargs=mm_processor_kwargs,
            media_io_kwargs={"video": video_options},
        )
    )


def process(mm_processor, metadata, cache=None, prompt="<video>"):
    frames = np.zeros((4, 56, 56, 3), dtype=np.uint8)
    return mm_processor(
        prompt,
        mm_items=mm_processor.info.parse_mm_data({"video": [(frames, metadata)]}),
        cache=cache,
    )


def test_t1_timestamps_survive_cache(tmp_path):
    mm_processor = processor(tmp_path)
    cache = MultiModalProcessorOnlyCache(mm_processor.info.ctx.model_config)
    metadata = {"fps": 2.0, "frames_indices": [0, 3, 6, 9], "total_num_frames": 12}
    reference = process(mm_processor, metadata)
    hf_processor = mm_processor.info.get_hf_processor()
    image = hf_processor.replace_image_token({"image_sizes": [(56, 56)]}, 0)
    expected = "This is a video:\n" + "".join(
        f"Frame {i + 1} sampled at {time:.2f} seconds: {image}\n"
        for i, time in enumerate([0.0, 1.5, 3.0, 4.5])
    )
    assert reference["prompt_token_ids"] == hf_processor.tokenizer.encode(
        expected, add_special_tokens=False
    )
    for _ in range(2):
        result = process(mm_processor, metadata, cache)
        assert result["prompt_token_ids"] == reference["prompt_token_ids"]
        item = result["mm_kwargs"]["video"][0]
        assert item["video_timestamps"].data.tolist() == [0.0, 1.5, 3.0, 4.5]
    changed = process(mm_processor, dict(metadata, fps=4.0), cache)
    item = changed["mm_kwargs"]["video"][0]
    assert item["video_timestamps"].data.tolist() == [0.0, 0.75, 1.5, 2.25]


@pytest.mark.parametrize("temporal_patch_size", [1, 2])
def test_loader_labels_and_frame_cap(tmp_path, temporal_patch_size):
    mm_processor = processor(
        tmp_path,
        temporal_patch_size,
        video_backend="nemotron_shortcut_t2",
        temporal_patch_size=temporal_patch_size,
        max_frames=32,
    )
    assert mm_processor.info.get_max_video_frames() == 32
    metadata = dict(
        fps=2.0,
        total_num_frames=4,
        frames_indices=[0, 1, 2, 3],
        frame_source_indices=[0, 1, 2, 3],
        frame_timestamps=[0.25, 0.75, 1.25, 1.75],
        frame_temporal_patch_size=temporal_patch_size,
    )
    result = process(mm_processor, metadata)
    item = result["mm_kwargs"]["video"][0]
    assert item["video_timestamps"].data.tolist() == metadata["frame_timestamps"]
    if temporal_patch_size == 1:
        with pytest.raises(ValueError, match="temporal_patch_size"):
            process(mm_processor, dict(metadata, frame_temporal_patch_size=2))


@pytest.mark.parametrize("temporal_patch_size", [1, 2])
def test_video_header_survives_chat_bpe_boundaries(tmp_path, temporal_patch_size):
    """Replacing a non-special video marker preserves its visual token range."""
    mm_processor = processor(tmp_path, temporal_patch_size)
    cache = MultiModalProcessorOnlyCache(mm_processor.info.ctx.model_config)
    metadata = {"fps": 1.0, "frames_indices": [0, 1, 2, 3], "total_num_frames": 4}
    reference = process(mm_processor, metadata, cache, prompt="\n<video>\nQuestion")
    text = mm_processor.info.get_tokenizer().decode(reference["prompt_token_ids"])
    assert text.startswith("\nThis is a video:\nFrame 1 sampled at 0.00 seconds")
    assert text.endswith("\n\nQuestion")
    assert len(reference["mm_placeholders"]["video"]) == 1
    cached = process(mm_processor, metadata, cache, prompt="\n<video>\nQuestion")
    assert cached["prompt_token_ids"] == reference["prompt_token_ids"]


@pytest.mark.parametrize("temporal_patch_size", [1, 2])
@pytest.mark.parametrize("image_edge", [56, 112])
def test_video_resize_is_independent_of_checkpoint_image_size(
    tmp_path, temporal_patch_size, image_edge
):
    """Absent video geometry defaults to the training 448px square at every T."""
    import cv2

    mm_processor = processor(
        tmp_path,
        temporal_patch_size,
        image_edge=image_edge,
        video_target_num_patches=None,
    )
    frames = np.random.default_rng(0).integers(0, 256, (2, 72, 128, 3), dtype=np.uint8)
    result = mm_processor(
        "<video>", mm_items=mm_processor.info.parse_mm_data({"video": [frames]})
    )
    pixels = result["mm_kwargs"]["video"][0]["pixel_values_videos"].data
    assert [tuple(frame.shape) for frame in pixels] == [(3, 448, 448)] * 2
    assert result["prompt_token_ids"].count(1) == 2 // temporal_patch_size * 256
    image_processor = mm_processor.info.get_hf_processor().image_processor
    raw = frames[0].astype(np.float32) / np.float32(
        1.0 / image_processor.rescale_factor
    )
    reference = torch.from_numpy(
        cv2.resize(raw, (448, 448), interpolation=cv2.INTER_AREA)
    ).permute(2, 0, 1)
    mean = torch.tensor(image_processor.image_mean)[:, None, None]
    std = torch.tensor(image_processor.image_std)[:, None, None]
    torch.testing.assert_close(
        pixels[0], ((reference - mean) / std).to(pixels[0].dtype), rtol=0, atol=0
    )
    builder = NemotronShortcutDummyInputsBuilder(mm_processor.info)
    [video] = builder.get_dummy_mm_data(4096, {"video": 1}, {})["video"]
    assert video[0].shape[:2] == (448, 448)


@pytest.mark.parametrize(
    "image_edge,override,expected,per_request",
    [
        (56, None, 56, False),
        (112, None, 112, False),
        (112, 56, 56, False),
        (112, 56, 56, True),
    ],
)
def test_image_size_uses_checkpoint_and_explicit_override(
    tmp_path, image_edge, override, expected, per_request
):
    kwargs = {"size": {"longest_edge": override}} if override is not None else None
    mm_processor = processor(
        tmp_path,
        image_edge=image_edge,
        mm_processor_kwargs=None if per_request else kwargs,
    )
    result = mm_processor(
        "<img>",
        mm_items=mm_processor.info.parse_mm_data(
            {"image": [Image.new("RGB", (224, 112))]}
        ),
        hf_processor_mm_kwargs=kwargs if per_request else None,
    )
    pixels = result["mm_kwargs"]["image"][0]["pixel_values"].data
    assert pixels.shape[-2:] == (expected // 2, expected)
    assert result["prompt_token_ids"].count(1) == expected // 28 * (expected // 56)


def test_video_aspect_ratio_cannot_exceed_profiled_patch_budget(tmp_path):
    mm_processor = processor(
        tmp_path, video_target_num_patches=1024, video_maintain_aspect_ratio=True
    )
    assert mm_processor.info.get_video_frame_size(1280, 720) == (588, 336)
    with pytest.raises(ValueError, match="profiled frame patch budget"):
        mm_processor(
            "<video>",
            mm_items=mm_processor.info.parse_mm_data(
                {"video": [np.zeros((1, 10, 4000, 3), dtype=np.uint8)]}
            ),
        )


def test_video_accepts_channels_first_tensors(tmp_path):
    mm_processor = processor(tmp_path)
    frames = np.random.default_rng(0).integers(0, 256, (2, 56, 84, 3), dtype=np.uint8)
    outputs = [
        mm_processor(
            "<video>", mm_items=mm_processor.info.parse_mm_data({"video": [video]})
        )
        for video in (frames, torch.from_numpy(frames).permute(0, 3, 1, 2))
    ]
    assert outputs[0]["prompt_token_ids"] == outputs[1]["prompt_token_ids"]
    for hwc, chw in zip(
        outputs[0]["mm_kwargs"]["video"][0]["pixel_values_videos"].data,
        outputs[1]["mm_kwargs"]["video"][0]["pixel_values_videos"].data,
    ):
        torch.testing.assert_close(hwc, chw, rtol=0, atol=0)
