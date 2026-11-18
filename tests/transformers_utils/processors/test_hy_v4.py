# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import TokenizersBackend

from vllm.transformers_utils.configs.hy_v4 import HYV4VLConfig
from vllm.transformers_utils.processor import get_processor
from vllm.transformers_utils.processors.hy_v4 import HYV4VLProcessor, vision_token_count


@pytest.fixture
def hy_v4_processor_dir(tmp_path, request):
    variant = getattr(request, "param", "named")
    special_tokens = {
        "image_start_token": "<｜hy_image▁start:opensource｜>",
        "image_end_token": "<｜hy_image▁end:opensource｜>",
        "image_token": "<｜hy_image▁pad:opensource｜>",
        "image_newline_token": "<｜hy_image▁newline:opensource｜>",
        "video_start_token": "<｜hy_video▁start:opensource｜>",
        "video_end_token": "<｜hy_video▁end:opensource｜>",
        "video_token": "<｜hy_video▁pad:opensource｜>",
    }
    if variant != "named":
        special_tokens = {name: f"<{name}>" for name in special_tokens}
    tokens = ["[UNK]", "[PAD]", *special_tokens.values(), "<image>", "<video>"]
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(
            WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
        extra_special_tokens=(
            list(special_tokens.values()) if variant == "unnamed" else special_tokens
        ),
    )
    tokenizer.add_special_tokens(
        {"extra_special_tokens": ["<image>", "<video>"]},
        replace_extra_special_tokens=False,
    )
    tokenizer.save_pretrained(tmp_path)
    config = HYV4VLConfig(
        image_start_token_id=2,
        image_end_token_id=3,
        image_token_id=4,
        image_newline_token_id=5,
        video_start_token_id=6,
        video_end_token_id=7,
        video_token_id=8,
        vision_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "max_image_size": 64,
            "projector_type": "rmsnorm_mlp_adapter",
        },
    )
    config.save_pretrained(tmp_path)
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "HYV4VLProcessor",
                "image_processor_type": "HYV4VLImageProcessor",
                "min_pixels": 4096,
                "max_pixels": 16384,
                "patch_size": 16,
                "merge_size": 2,
                "temporal_patch_size": 1,
                "spatial_patch_size": 1,
                "image_mean": [0.5] * 3,
                "image_std": [0.5] * 3,
                "min_frames": 2,
                "max_frames": 8,
                "timestamp_token_reserve": 0,
                "timestamp_token_per_frame": 6,
                "video_max_image_size": 64,
            }
        )
    )
    return tmp_path


@pytest.mark.parametrize(
    "hy_v4_processor_dir", ["named", "custom", "unnamed"], indirect=True
)
def test_hy_v4_image_patch_order_and_normalization(hy_v4_processor_dir):
    processor = get_processor(str(hy_v4_processor_dir), trust_remote_code=False)
    assert isinstance(processor, HYV4VLProcessor)
    pixels = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    image = Image.fromarray(pixels)
    result = processor(images=[image], return_tensors="pt")
    expected = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 127.5 - 1
    expected = expected.reshape(3, 4, 16, 4, 16).permute(1, 3, 0, 2, 4).reshape(16, -1)
    torch.testing.assert_close(result["pixel_values"], expected)
    assert result["image_grid_thw"].tolist() == [[1, 4, 4]]
    expanded = processor(
        text=processor.get_placeholder("image"), images=[image], return_tensors="pt"
    )
    assert expanded["input_ids"].tolist() == [[2, *([4] * 6), 3]]


@pytest.mark.parametrize(
    "hy_v4_processor_dir", ["named", "custom", "unnamed"], indirect=True
)
def test_hy_v4_processor_uses_loaded_tokenizer(hy_v4_processor_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hy_v4_processor_dir)
    processor = HYV4VLProcessor.from_pretrained(
        hy_v4_processor_dir, tokenizer=tokenizer
    )
    assert processor.tokenizer is tokenizer
    for name in (
        "image_start_token",
        "image_end_token",
        "image_token",
        "image_newline_token",
        "video_start_token",
        "video_end_token",
        "video_token",
    ):
        token_id = getattr(processor.config, f"{name}_id")
        expected = tokenizer.convert_ids_to_tokens(token_id)
        assert getattr(processor, name) == expected
        assert getattr(processor, f"{name}_id") == token_id


def test_hy_v4_rejects_tokenizer_named_token_id_mismatch(hy_v4_processor_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hy_v4_processor_dir)
    tokenizer.image_start_token = tokenizer.image_end_token
    with pytest.raises(ValueError, match="image_start_token.*does not match"):
        HYV4VLProcessor.from_pretrained(hy_v4_processor_dir, tokenizer=tokenizer)


def test_hy_v4_video_preserves_sampled_indices_and_offset(hy_v4_processor_dir):
    processor = HYV4VLProcessor.from_pretrained(hy_v4_processor_dir)
    frames = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    metadata = dict(
        total_num_frames=10,
        fps=2,
        frames_indices=[2, 8],
        do_sample_frames=False,
        timestamp_offset=3.0,
    )
    output = processor(videos=[(frames, metadata)], return_tensors="pt")
    assert output["video_grid_thw"].tolist() == [[2, 2, 2]]
    assert output["video_timestamps"].tolist() == [4.0, 7.0]
    assert metadata["frames_indices"] == [2, 8]
    count = vision_token_count(
        output["video_grid_thw"][0], processor.config.vision_config, "video"
    )
    replacement = processor.get_replacement(
        "video", output["video_grid_thw"][0], output["video_timestamps"]
    )
    assert replacement.count(processor.video_token_id) == count


@pytest.mark.parametrize("fps", [0, -1])
def test_hy_v4_video_rejects_invalid_fps(hy_v4_processor_dir, fps):
    processor = HYV4VLProcessor.from_pretrained(hy_v4_processor_dir)
    with pytest.raises(ValueError, match="positive fps"):
        processor(videos=[(np.zeros((2, 64, 64, 3), dtype=np.uint8), {"fps": fps})])


def test_hy_v4_image_options_do_not_mutate_other_processors():
    from vllm.transformers_utils.processors.hy_v4 import HYV4VLImageProcessor

    first = HYV4VLImageProcessor(min_pixels=4096, max_pixels=16384)
    second = HYV4VLImageProcessor(min_pixels=1024, max_pixels=4096)
    assert first.size["longest_edge"] == 16384
    assert second.size["longest_edge"] == 4096


def test_hy_v4_image_processor_saves_new_type(hy_v4_processor_dir):
    from vllm.transformers_utils.processors.hy_v4 import HYV4VLImageProcessor

    processor = get_processor(str(hy_v4_processor_dir), trust_remote_code=False)
    image_processor = processor.image_processor
    assert isinstance(image_processor, HYV4VLImageProcessor)
    output_dir = hy_v4_processor_dir / "image_processor"
    image_processor.save_pretrained(output_dir)
    saved_config = json.loads((output_dir / "preprocessor_config.json").read_text())
    assert saved_config["image_processor_type"] == "HYV4VLImageProcessor"
    reloaded = HYV4VLImageProcessor.from_pretrained(output_dir)
    image = Image.new("RGB", (64, 64), "red")
    before = image_processor(images=[image], return_tensors="pt")
    after = reloaded(images=[image], return_tensors="pt")
    torch.testing.assert_close(before["pixel_values"], after["pixel_values"])
    torch.testing.assert_close(before["image_grid_thw"], after["image_grid_thw"])


def test_hy_v4_video_metadata_object_and_explicit_frame_sampling(hy_v4_processor_dir):
    from transformers.video_utils import VideoMetadata

    processor = HYV4VLProcessor.from_pretrained(hy_v4_processor_dir)
    video = np.zeros((8, 64, 64, 3), dtype=np.uint8)
    metadata = VideoMetadata(total_num_frames=8, fps=2)
    result = processor(videos=[video], video_metadata=[metadata], return_tensors="pt")
    assert result["video_timestamps"].tolist() == [i / 2 for i in range(8)]
    sampled = processor(
        videos=[video],
        video_metadata=[metadata],
        do_sample_frames=True,
        num_frames=2,
        return_tensors="pt",
    )
    assert sampled["video_grid_thw"][0, 0].item() == 2
    assert metadata.frames_indices is None


def test_hy_v4_processor_rejects_inputs_exceeding_profile_limits(hy_v4_processor_dir):
    processor = HYV4VLProcessor.from_pretrained(hy_v4_processor_dir)
    with pytest.raises(ValueError, match="max_pixels"):
        processor(images=[Image.new("RGB", (6400, 32))])
    with pytest.raises(ValueError, match="max_frames"):
        processor(
            videos=[np.zeros((32, 32, 32, 3), dtype=np.uint8)],
            video_metadata=[{"total_num_frames": 32, "fps": 2}],
        )


def test_hy_v4_processor_accepts_text_encoding_options(hy_v4_processor_dir):
    processor = HYV4VLProcessor.from_pretrained(hy_v4_processor_dir)
    options = dict(
        add_special_tokens=False,
        truncation=True,
        max_length=2,
        padding="max_length",
        return_tensors="pt",
    )
    actual = processor(text="hello", **options)
    expected = processor.tokenizer(["hello"], **options)
    torch.testing.assert_close(actual["input_ids"], expected["input_ids"])
