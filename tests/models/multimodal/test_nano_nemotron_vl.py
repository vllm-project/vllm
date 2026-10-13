# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from vllm import envs
from vllm.model_executor.models.nano_nemotron_vl import (
    NanoNemotronVLMultiModalProcessor,
    NanoNemotronVLProcessingInfo,
    NemotronH_Nano_VL_V2,
)
from vllm.multimodal.parse import (
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)


@pytest.mark.parametrize("input_key", ["image_embeds", "video_embeds"])
def test_precomputed_multimodal_embeddings(input_key: str):
    model = object.__new__(NemotronH_Nano_VL_V2)
    embeds = torch.randn(2, 4, 8)

    outputs = model.embed_multimodal(**{input_key: embeds})

    assert len(outputs) == len(embeds)
    assert all(torch.equal(output, embed) for output, embed in zip(outputs, embeds))


class _TextOnlyMultiModalConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        return 0


class _ImageOnlyMultiModalConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        return 1 if modality == "image" else 0


class _ModelConfig:
    multimodal_config = _TextOnlyMultiModalConfig()


class _ImageOnlyModelConfig:
    multimodal_config = _ImageOnlyMultiModalConfig()


class _LanguageModel:
    def __init__(self) -> None:
        self.loaded_weights: list[tuple[str, object]] = []

    def load_weights(self, weights):
        self.loaded_weights = list(weights)


class _MissingMultiModalModule:
    def named_parameters(self):
        raise AssertionError("multimodal weights should not be inspected")

    def load_weights(self, weights):
        raise AssertionError("multimodal weights should not be loaded")


class _AdapterModule:
    def named_parameters(self):
        return []


class _VisionModel:
    def __init__(self) -> None:
        self.loaded_weights: list[tuple[str, object]] = []

    def load_weights(self, weights):
        self.loaded_weights = list(weights)


class _RecordingVisionModel:
    def __init__(self) -> None:
        self.input_dtypes: list[torch.dtype] = []
        self.batch_sizes: list[int] = []

    def __call__(self, pixel_values, **kwargs):
        self.input_dtypes.append(pixel_values.dtype)
        self.batch_sizes.append(pixel_values.shape[0])
        if pixel_values.ndim == 3:
            batch, patch_count, _ = pixel_values.shape
        else:
            batch, _, height, width = pixel_values.shape
            patch_count = (height // 2) * (width // 2)

        if (num_frames := kwargs.get("num_frames")) is not None:
            batch = (num_frames + 1) // 2

        return None, torch.zeros(
            batch, patch_count, 2, dtype=torch.float32, device=pixel_values.device
        )


class _RecordingProjector:
    def __init__(self) -> None:
        self.input_dtypes: list[torch.dtype] = []

    def __call__(self, vision_features):
        self.input_dtypes.append(vision_features.dtype)
        return vision_features


class _LanguageModelWithDtype(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = SimpleNamespace(dtype=dtype)


class _DtypeRecordingVisionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.requested_dtype: torch.dtype | None = None

    def to(self, dtype: torch.dtype):
        self.requested_dtype = dtype
        return self


class _Tokenizer:
    def encode(self, *_args, **_kwargs) -> list[int]:
        return []


class _FakeTensor:
    """Sentinel stand-in for torch.Tensor in load_weights tests. Supports the
    .detach().clone() chain used by load_weights for buffered mm weights;
    both methods return self so identity (and the existing equality
    assertions) are preserved through cloning."""

    def detach(self):
        return self

    def clone(self):
        return self


def test_nano_nemotron_vl_skips_multimodal_weights_in_text_only_mode():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    object.__setattr__(model, "model_config", _ModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", _MissingMultiModalModule())
    object.__setattr__(model, "sound_encoder", None)

    language_weight = object()
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("mlp1.0.weight", object()),
            ("vision_model.radio_model.encoder.weight", object()),
            ("sound_encoder.encoder.weight", object()),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]


def test_nano_nemotron_vl_loads_vision_weights_without_sound_encoder():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "sound_encoder", None)

    language_weight = object()
    vision_weight = _FakeTensor()
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("vision_model.radio_model.encoder.weight", vision_weight),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]
    assert vision_model.loaded_weights == [
        ("radio_model.encoder.weight", vision_weight)
    ]


def test_nano_nemotron_vl_requires_sound_encoder_for_sound_weights():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "sound_encoder", None)

    with pytest.raises(AssertionError):
        model.load_weights([("sound_encoder.encoder.weight", object())])


@pytest.mark.parametrize("runtime_dtype", [torch.bfloat16, torch.float16])
def test_nano_nemotron_vl_uses_resolved_runtime_dtype(
    runtime_dtype: torch.dtype,
    default_vllm_config,
):
    checkpoint_dtype = (
        torch.float16 if runtime_dtype is torch.bfloat16 else torch.bfloat16
    )
    text_config = SimpleNamespace(dtype=checkpoint_dtype, hidden_size=2)
    vision_config = SimpleNamespace(args={}, video_temporal_patch_size=1)
    hf_config = SimpleNamespace(
        force_image_size=4,
        patch_size=2,
        template="plain",
        downsample_ratio=1.0,
        ps_version="v2",
        image_tag_type="plain",
        text_config=text_config,
        vision_config=vision_config,
        vit_hidden_size=2,
        projector_hidden_size=2,
        norm_mean=[0.5, 0.5, 0.5],
        norm_std=[0.5, 0.5, 0.5],
    )
    model_config = SimpleNamespace(
        dtype=runtime_dtype,
        hf_config=hf_config,
        multimodal_config=SimpleNamespace(video_pruning_rate=None),
        max_model_len=128,
    )
    vllm_config = SimpleNamespace(model_config=model_config)
    language_model = _LanguageModelWithDtype(checkpoint_dtype)
    vision_model = _DtypeRecordingVisionModel()

    module = "vllm.model_executor.models.nano_nemotron_vl"
    with (
        patch(f"{module}.init_vllm_registered_model", return_value=language_model),
        patch(f"{module}.cached_tokenizer_from_config", return_value=_Tokenizer()),
        patch.object(
            NemotronH_Nano_VL_V2, "_mark_language_model", return_value=nullcontext()
        ),
        patch.object(
            NemotronH_Nano_VL_V2, "_mark_tower_model", return_value=nullcontext()
        ),
        patch.object(
            NemotronH_Nano_VL_V2,
            "get_vit_model_from_radio_config",
            return_value=vision_model,
        ),
    ):
        model = NemotronH_Nano_VL_V2(vllm_config=vllm_config)

    assert model.llm_dtype is runtime_dtype
    assert vision_model.requested_dtype is runtime_dtype
    assert next(model.mlp1.parameters()).dtype is runtime_dtype


@pytest.mark.parametrize("runtime_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dynamic_resolution", [False, True])
def test_nano_nemotron_vl_processor_uses_runtime_dtype(
    runtime_dtype: torch.dtype,
    dynamic_resolution: bool,
):
    vision_args = (
        {"min_num_patches": 4, "max_num_patches": 64} if dynamic_resolution else {}
    )
    config = SimpleNamespace(
        force_image_size=4,
        patch_size=2,
        downsample_ratio=0.5,
        use_thumbnail=False,
        norm_mean=[0.5, 0.5, 0.5],
        norm_std=[0.5, 0.5, 0.5],
        vision_config=SimpleNamespace(
            args=vision_args,
            video_temporal_patch_size=1,
        ),
        dtype=None,
    )
    tokenizer = MagicMock()
    ctx = MagicMock()
    ctx.model_config.dtype = runtime_dtype
    ctx.model_config.max_model_len = 4096
    ctx.get_hf_config.return_value = config
    ctx.get_tokenizer.return_value = tokenizer
    ctx.get_mm_config.return_value = SimpleNamespace(video_pruning_rate=None)
    ctx.init_processor.side_effect = lambda processor_cls, **kwargs: processor_cls(
        **kwargs
    )
    info = NanoNemotronVLProcessingInfo(ctx)

    processor = info.get_hf_processor()

    image = Image.new("RGB", (4, 4))
    if dynamic_resolution:
        assert processor.dynamic_tiler is not None
        image_tensors, _ = processor.dynamic_tiler._images_to_pixel_values_lst(
            text_prompt_length=0,
            images=[image],
            dtype=processor.dtype,
        )
    else:
        image_tensors = processor._images_to_pixel_values_lst([image], max_num_tiles=1)
    video_tensors = processor._videos_to_pixel_values_lst(
        [np.zeros((2, 4, 4, 3), dtype=np.uint8)],
        dtype=processor.dtype,
    )

    assert processor.dtype is runtime_dtype
    assert all(tensor.dtype is runtime_dtype for tensor in image_tensors)
    assert all(tensor.dtype is runtime_dtype for tensor in video_tensors)


@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dynamic_resolution", [False, True])
def test_nano_nemotron_vl_aligns_vision_boundary_dtype(
    compute_dtype: torch.dtype, dynamic_resolution: bool
):
    model = object.__new__(NemotronH_Nano_VL_V2)
    vision_model = _RecordingVisionModel()
    projector = _RecordingProjector()
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "llm_dtype", compute_dtype)
    object.__setattr__(model, "mlp1", projector)
    object.__setattr__(model, "patch_size", 2)
    object.__setattr__(model, "downsample_ratio", 1.0)
    object.__setattr__(model, "ps_version", "v2")

    if dynamic_resolution:
        batch_size = 1
        pixel_values = torch.ones(1, 4, 12, dtype=torch.float32)
        output = model.extract_feature_dynamic(pixel_values, imgs_sizes=[(4, 4)])
    else:
        batch_size = 129
        pixel_values = torch.ones(batch_size, 3, 4, 4, dtype=torch.float32)
        output = model.extract_feature(pixel_values)

    expected_batch_sizes = [1] if dynamic_resolution else [128, 1]
    assert vision_model.batch_sizes == expected_batch_sizes
    assert vision_model.input_dtypes == [compute_dtype] * len(expected_batch_sizes)
    assert projector.input_dtypes == [compute_dtype] * len(expected_batch_sizes)
    assert output.dtype == compute_dtype
    assert output.shape == (batch_size, 4, 2)
    assert pixel_values.dtype == torch.float32


@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float16])
def test_nano_nemotron_vl_aligns_temporal_video_dtype(
    compute_dtype: torch.dtype,
):
    model = object.__new__(NemotronH_Nano_VL_V2)
    vision_model = _RecordingVisionModel()
    projector = _RecordingProjector()
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "llm_dtype", compute_dtype)
    object.__setattr__(model, "mlp1", projector)
    object.__setattr__(model, "patch_size", 2)
    object.__setattr__(model, "downsample_ratio", 1.0)
    object.__setattr__(model, "ps_version", "v2")
    object.__setattr__(model, "video_temporal_patch_size", 2)

    pixel_values = torch.ones(130, 3, 4, 4, dtype=torch.float32)
    output = model.extract_feature(pixel_values, num_frames=130)

    assert vision_model.batch_sizes == [128, 2]
    assert vision_model.input_dtypes == [compute_dtype, compute_dtype]
    assert projector.input_dtypes == [compute_dtype, compute_dtype]
    assert output.dtype == compute_dtype
    assert output.shape == (65, 4, 2)
    assert pixel_values.dtype == torch.float32


def _make_mm_items_with_video_bytes(
    video_bytes: bytes,
) -> MultiModalDataItems:
    """Build a minimal MultiModalDataItems with one video entry."""
    items = MultiModalDataItems()
    items["video"] = VideoProcessorItems(
        data=[None],
        metadata=[{"original_video_bytes": video_bytes}],
    )
    return items


def test_extract_audio_from_videos_passes_max_duration():
    """_extract_audio_from_videos must forward VLLM_MAX_AUDIO_DECODE_DURATION_S
    to load_audio_pyav so decompression-bomb audio is rejected."""
    dummy_audio = (np.zeros(16000, dtype=np.float32), 16000.0)
    mm_items = _make_mm_items_with_video_bytes(b"\x00" * 64)

    processor = object.__new__(NanoNemotronVLMultiModalProcessor)
    processor.data_parser = MultiModalDataParser(
        target_sr=dummy_audio[1], target_channels=1
    )

    target = "vllm.model_executor.models.nano_nemotron_vl.load_audio_pyav"
    with patch(target, return_value=dummy_audio) as mock_load:
        processor._extract_audio_from_videos(mm_items)

    mock_load.assert_called_once()
    _, kwargs = mock_load.call_args
    assert kwargs["max_duration_s"] == envs.VLLM_MAX_AUDIO_DECODE_DURATION_S


def test_extract_audio_from_videos_rejects_oversized_audio():
    """When load_audio_pyav raises due to duration limit the video is
    marked as having no audio instead of crashing the server."""
    mm_items = _make_mm_items_with_video_bytes(b"\x00" * 64)

    processor = object.__new__(NanoNemotronVLMultiModalProcessor)

    target = "vllm.model_executor.models.nano_nemotron_vl.load_audio_pyav"
    with patch(
        target,
        side_effect=ValueError("Audio exceeds maximum allowed duration"),
    ):
        _, audio_items, has_audio = processor._extract_audio_from_videos(mm_items)

    assert audio_items == []
    assert has_audio == [False]
