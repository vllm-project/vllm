# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from transformers import BatchEncoding

from vllm import envs
from vllm.config import MultiModalConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.fusion.mm_input_norm import FusedMMInputNorm
from vllm.model_executor.models.nano_nemotron_vl import (
    NanoNemotronVLDummyInputsBuilder,
    NanoNemotronVLImagePixelInputsDynamic,
    NanoNemotronVLMultiModalProcessor,
    NanoNemotronVLVideoPixelInputs,
    NemotronH_Nano_VL_V2,
)
from vllm.multimodal.parse import (
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.processors import nano_nemotron_vl as nemotron_processors
from vllm.transformers_utils.processors.nano_nemotron_vl import (
    DynamicResolutionImageTiler,
    NanoNemotronVLProcessor,
    _bicubic_resize_and_normalize,
    cvcuda_bicubic_resize,
)

# FusedMMInputNorm runs a Triton kernel on GPU platforms, so its inputs must
# live on the platform's device.
_DEVICE = torch.device(current_platform.device_type)


@pytest.mark.parametrize("do_cpu_normalize", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nemotron_resized_normalization_matches_reference(
    dtype: torch.dtype, do_cpu_normalize: bool, default_vllm_config
):
    pixels = torch.randint(
        0,
        256,
        (1, 19, 23, 3),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(0),
    )
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    output = _bicubic_resize_and_normalize(
        pixels,
        size=(16, 16),
        norm_mean=mean,
        norm_std=std,
        do_cpu_normalize=do_cpu_normalize,
        dtype=dtype,
    )

    if not do_cpu_normalize:
        assert output.dtype == torch.uint8
        model = _uninitialized_model()
        nn.Module.__init__(model)
        model.input_norm = FusedMMInputNorm(
            mean.flatten().tolist(), std.flatten().tolist(), 1.0 / 255.0
        ).to(_DEVICE)
        model.llm_dtype = dtype
        output = model._normalize_pixel_values(output.to(_DEVICE)).cpu()

    resized = torch.nn.functional.interpolate(
        pixels.permute(0, 3, 1, 2).float(),
        size=(16, 16),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    expected = ((resized.float() / 255.0 - mean) / std).to(dtype)
    assert output.dtype == dtype
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.06)
    assert (output.float() - expected.float()).abs().mean() < 0.01


def test_nemotron_dynamic_device_normalization_matches_cpu_reference(
    default_vllm_config,
):
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    shape = (1, 7, 3 * 16 * 16)
    pixels = torch.randint(0, 256, shape, dtype=torch.uint8)

    model = _uninitialized_model()
    nn.Module.__init__(model)
    model.input_norm = FusedMMInputNorm(image_mean, image_std, 1.0 / 255.0).to(_DEVICE)
    model.llm_dtype = torch.bfloat16

    output = model._normalize_pixel_values(pixels.to(_DEVICE)).cpu()
    batch, patches, _ = pixels.shape
    expected = pixels.to(torch.float32).view(batch, patches, 3, -1)
    expected = (
        expected / 255.0 - torch.tensor(image_mean).view(1, 1, 3, 1)
    ) / torch.tensor(image_std).view(1, 1, 3, 1)
    expected = expected.view(shape)

    torch.testing.assert_close(output, expected.to(torch.bfloat16))


def test_nemotron_cpu_normalized_pixels_bypass_input_norm():
    pixels = torch.randn(1, 3, 16, 16)
    model = _uninitialized_model()
    nn.Module.__init__(model)
    model.input_norm = None

    assert model._normalize_pixel_values(pixels) is pixels


@pytest.mark.parametrize("modality", ["image", "video", "dynamic_image"])
def test_nemotron_processor_defers_normalization_to_device(
    modality: str, default_vllm_config
):
    dynamic_image = modality == "dynamic_image"
    config = SimpleNamespace(
        force_image_size=16,
        patch_size=4,
        downsample_ratio=0.5,
        use_thumbnail=False,
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        dtype=torch.bfloat16,
        vision_config=SimpleNamespace(
            args={"min_num_patches": 4, "max_num_patches": 16} if dynamic_image else {}
        ),
        sound_config=None,
    )
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1])
    common_kwargs = {
        "config": config,
        "tokenizer": tokenizer,
        "max_model_len": 1024,
    }
    cpu_processor = NanoNemotronVLProcessor(**common_kwargs)
    device_kwargs = MultiModalConfig(
        mm_device_do_normalize=True
    ).merge_mm_processor_kwargs({})
    device_processor = NanoNemotronVLProcessor(**common_kwargs, **device_kwargs)

    pixels = np.random.default_rng(0).integers(0, 256, (4, 19, 23, 3), dtype=np.uint8)
    if modality == "image":
        image = Image.fromarray(pixels[0])
        cpu_values = cpu_processor._images_to_pixel_values_lst([image], 1)[0]
        raw_values = device_processor._images_to_pixel_values_lst([image], 1)[0]
    elif modality == "video":
        cpu_values = cpu_processor._videos_to_pixel_values_lst(
            [pixels], dtype=torch.bfloat16
        )[0]
        raw_values = device_processor._videos_to_pixel_values_lst([pixels])[0]
    else:
        params = DynamicResolutionImageTiler.DynamicResolutionParams(
            media=Image.fromarray(pixels[0]),
            num_tiles=1,
            num_embeddings=4,
            patch_size=(4, 4),
        )
        assert cpu_processor.dynamic_tiler is not None
        assert device_processor.dynamic_tiler is not None
        cpu_values = cpu_processor.dynamic_tiler.apply_params(
            params, dtype=torch.bfloat16
        )[0]
        raw_values = device_processor.dynamic_tiler.apply_params(params)[0]

    assert raw_values.dtype == torch.uint8
    assert cpu_values.dtype == torch.bfloat16
    assert raw_values.shape == cpu_values.shape

    # Normalizing on the device hands the encoder the same dtype as the CPU
    # path. Values differ only where the uint8 resize rounds or clips the
    # bicubic overshoot: a few elements by several levels, tiny on average.
    model = _uninitialized_model()
    nn.Module.__init__(model)
    model.input_norm = FusedMMInputNorm(
        config.norm_mean, config.norm_std, 1.0 / 255.0
    ).to(_DEVICE)
    model.llm_dtype = config.dtype
    if dynamic_image:
        cpu_values = DynamicResolutionImageTiler.stack([cpu_values], config.patch_size)
        raw_values = DynamicResolutionImageTiler.stack([raw_values], config.patch_size)
    device_values = model._normalize_pixel_values(raw_values.to(_DEVICE)).cpu()
    assert device_values.dtype == cpu_values.dtype
    assert (device_values.float() - cpu_values.float()).abs().mean() < 0.01


def test_nemotron_processor_rejects_partial_normalization():
    with pytest.raises(
        ValueError, match="requires do_rescale and do_normalize to have the same value"
    ):
        NanoNemotronVLProcessor(
            config=None,
            tokenizer=None,
            max_model_len=1024,
            do_rescale=False,
            do_normalize=True,
        )


def test_nemotron_processor_device_resize_requires_device_normalization():
    with pytest.raises(
        ValueError, match="device_resize requires mm_device_do_normalize"
    ):
        NanoNemotronVLProcessor(
            config=None, tokenizer=None, max_model_len=1024, device_resize=True
        )


class _CharTokenizer:
    """One token per character: enough for the prompt-length bookkeeping."""

    def encode(self, text: str, **kwargs) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], **kwargs) -> str:
        return "".join(map(chr, ids))

    def __call__(self, text: str | list[str], **kwargs) -> BatchEncoding:
        if isinstance(text, list):
            return BatchEncoding({"input_ids": [self.encode(t) for t in text]})
        return BatchEncoding({"input_ids": self.encode(text)})


_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]


def _make_processor(**kwargs) -> NanoNemotronVLProcessor:
    config = SimpleNamespace(
        force_image_size=16,
        patch_size=4,
        downsample_ratio=0.5,
        use_thumbnail=False,
        norm_mean=_NORM_MEAN,
        norm_std=_NORM_STD,
        dtype=torch.bfloat16,
        vision_config=SimpleNamespace(
            args={"min_num_patches": 4, "max_num_patches": 64},
            # As in the Nemotron 3 Nano Omni checkpoints.
            video_target_num_patches=16,
            video_maintain_aspect_ratio=True,
        ),
        sound_config=None,
    )
    return NanoNemotronVLProcessor(
        config=config, tokenizer=_CharTokenizer(), max_model_len=1024, **kwargs
    )


@pytest.mark.parametrize(
    ("device_normalize", "cvcuda_available", "device_resize", "expected"),
    [
        (True, True, None, True),  # on by default
        (True, True, False, False),  # opted out
        (True, False, None, False),  # not CUDA, or no CV-CUDA
        (False, True, None, False),  # normalizing on the CPU
    ],
)
def test_nemotron_device_resize_is_on_by_default_where_it_can_run(
    device_normalize: bool,
    cvcuda_available: bool,
    device_resize: bool | None,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        nemotron_processors, "_cvcuda_available", lambda: cvcuda_available
    )
    kwargs = MultiModalConfig(
        mm_device_do_normalize=device_normalize
    ).merge_mm_processor_kwargs({})
    if device_resize is not None:
        kwargs["device_resize"] = device_resize
    assert _make_processor(**kwargs).device_resize is expected


def _preprocess(processor: NanoNemotronVLProcessor, modality: str, sources: list):
    if modality == "image":
        images = [Image.fromarray(source) for source in sources]
        text = ["<image>" * len(images) + " Describe."]
        return processor._preprocess_image(text, images, max_num_tiles=1)
    videos = [
        (source, {"fps": 2.0, "frames_indices": list(range(len(source)))})
        for source in sources
    ]
    return processor._preprocess_video(["<video> Describe."], videos)


def _sources(modality: str) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    shapes = [(37, 53, 3), (90, 20, 3)] if modality == "image" else [(5, 37, 53, 3)]
    return [rng.integers(0, 256, shape, dtype=np.uint8) for shape in shapes]


@pytest.mark.parametrize("modality", ["image", "video"])
def test_nemotron_processor_device_resize_sends_source_pixels(
    modality: str, default_vllm_config
):
    device_kwargs = MultiModalConfig(
        mm_device_do_normalize=True
    ).merge_mm_processor_kwargs({})
    sources = _sources(modality)
    text, inputs = _preprocess(
        _make_processor(**device_kwargs, device_resize=False), modality, sources
    )
    raw_text, raw_inputs = _preprocess(
        _make_processor(**device_kwargs, device_resize=True), modality, sources
    )

    # The prompt (and so every token count) and the target sizes are unchanged;
    # the decoded pixels are sent as they are.
    assert raw_text == text
    if modality == "image":
        resized = inputs.pop("pixel_values_flat")
        raw = raw_inputs.pop("pixel_values_raw")
        sizes = raw_inputs["imgs_sizes"]
        assert raw_inputs == inputs
    else:
        resized = [inputs["pixel_values_flat_video"]]
        raw = raw_inputs["pixel_values_raw_video"]
        sizes = raw_inputs["video_frame_sizes"].tolist()
        torch.testing.assert_close(
            raw_inputs["video_num_patches"], inputs["video_num_patches"]
        )
    assert [tuple(size) for size in sizes] == [tuple(r.shape[-2:]) for r in resized]
    for raw_pixels, source in zip(raw, sources, strict=True):
        assert raw_pixels.dtype == torch.uint8
        assert torch.equal(raw_pixels, torch.from_numpy(source))


def test_nemotron_device_resize_profiles_a_real_video_resize(default_vllm_config):
    device_kwargs = MultiModalConfig(
        mm_device_do_normalize=True
    ).merge_mm_processor_kwargs({})
    results = {}
    for device_resize in (False, True):
        processor = _make_processor(**device_kwargs, device_resize=device_resize)
        info = SimpleNamespace(
            get_dummy_image_size_and_max_tokens=lambda mm_counts: ((16, 16), 4),
            get_hf_processor=lambda processor=processor: processor,
            supports_video=True,
            get_hf_config=lambda: SimpleNamespace(
                force_image_size=16, patch_size=4, downsample_ratio=0.5
            ),
            get_num_frames_with_most_features=lambda seq_len, mm_counts: 4,
            ctx=SimpleNamespace(
                get_mm_config=lambda: SimpleNamespace(media_io_kwargs={})
            ),
            sound_config=None,
        )
        builder = NanoNemotronVLDummyInputsBuilder(info)  # type: ignore[arg-type]
        videos = cast(list, builder.get_dummy_mm_data(64, {"video": 1}, {})["video"])
        results[device_resize] = (
            videos[0][0].shape,
            processor._preprocess_video(["<video>"], videos),
        )

    shape, (text, inputs) = results[False]
    raw_shape, (raw_text, raw_inputs) = results[True]
    # Without device_resize, profiling frames are already at the target size.
    assert tuple(inputs["pixel_values_flat_video"].shape[-2:]) == shape[1:3]
    # With it, they are twice the target, so the profile run includes the
    # resize on the device; the prompt and token counts are unchanged.
    assert raw_text == text
    (frame_size,) = raw_inputs["video_frame_sizes"].tolist()
    assert raw_shape[1:3] == (2 * frame_size[0], 2 * frame_size[1])
    assert tuple(frame_size) == shape[1:3]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CV-CUDA needs CUDA")
@pytest.mark.parametrize("modality", ["image", "video"])
def test_nemotron_device_resize_rounds_fp32_resize_once(
    modality: str, default_vllm_config
):
    pytest.importorskip("cvcuda")
    device_kwargs = MultiModalConfig(
        mm_device_do_normalize=True
    ).merge_mm_processor_kwargs({})
    sources = _sources(modality)
    _, raw_inputs = _preprocess(
        _make_processor(**device_kwargs, device_resize=True), modality, sources
    )

    def fp32_resize_rounded(source: np.ndarray, size) -> torch.Tensor:
        pixels = torch.from_numpy(source).reshape(-1, *source.shape[-3:])
        resized = torch.nn.functional.interpolate(
            pixels.permute(0, 3, 1, 2).float(),
            size=tuple(size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return resized.round().clamp(0, 255).to(torch.uint8)

    model = _uninitialized_model()
    nn.Module.__init__(model)
    model.input_norm = FusedMMInputNorm(_NORM_MEAN, _NORM_STD, 1.0 / 255.0).to(_DEVICE)
    model.patch_size = 4
    model.dynamic_resolution = True
    if modality == "video":
        raw_inputs["frames_indices"] = torch.tensor(raw_inputs["frames_indices"])
    parsed = model._parse_and_validate_multimodal_inputs(**raw_inputs)
    if modality == "image":
        images = parsed["images"]
        assert isinstance(images, NanoNemotronVLImagePixelInputsDynamic)
        pixels = images.pixel_values_flat
        expected = DynamicResolutionImageTiler.stack(
            [
                fp32_resize_rounded(source, size)[0]
                for source, size in zip(sources, raw_inputs["imgs_sizes"])
            ],
            4,
        )
    else:
        videos = parsed["videos"]
        assert isinstance(videos, NanoNemotronVLVideoPixelInputs)
        pixels = videos["pixel_values_flat"]
        expected = fp32_resize_rounded(
            sources[0], raw_inputs["video_frame_sizes"][0].tolist()
        )
        # Chunked uploads give the same pixels as one upload, also without a
        # resize (reformat only).
        frames = raw_inputs["pixel_values_raw_video"][0]
        for size in ((32, 40), tuple(frames.shape[1:3])):
            torch.testing.assert_close(
                cvcuda_bicubic_resize(frames, size, _DEVICE, max_chunk_bytes=1),
                cvcuda_bicubic_resize(frames, size, _DEVICE),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            cvcuda_bicubic_resize(frames, frames.shape[1:3], _DEVICE).cpu(),
            frames.permute(0, 3, 1, 2),
            rtol=0,
            atol=0,
        )

    # uint8 NCHW, like the CPU path, for FusedMMInputNorm to cast and normalize.
    # CV-CUDA rounds and clips the FP32 resize once: at most one level off it.
    # (torch's uint8 CPU resize can be tens of levels off it on upscales.)
    actual = pixels.cpu()
    assert actual.dtype == torch.uint8
    assert actual.shape == expected.shape
    diff = (actual.int() - expected.int()).abs()
    assert diff.max() <= 1
    assert (diff > 0).float().mean() < 1e-3


def _uninitialized_model() -> NemotronH_Nano_VL_V2:
    return object.__new__(NemotronH_Nano_VL_V2)  # type: ignore[type-abstract]


def test_nemotron_device_resize_embeds_videos_of_different_sizes_separately():
    # Videos that keep their aspect ratio can get different frame sizes, and
    # one encoder batch can hold several of them.
    model = _uninitialized_model()
    nn.Module.__init__(model)
    seen: list[dict] = []

    def record(**kwargs):
        seen.append(kwargs)
        return {}

    model._parse_and_validate_multimodal_inputs = record  # type: ignore[method-assign]
    frames = [
        torch.zeros(2, 4, 8, 3, dtype=torch.uint8),
        torch.zeros(2, 6, 6, 3, dtype=torch.uint8),
    ]
    kwargs = dict(
        pixel_values_raw_video=frames,
        video_frame_sizes=torch.tensor([[32, 64], [48, 48]]),
        video_num_patches=torch.tensor([2, 2]),
        frames_indices=torch.tensor([[0, 1], [0, 1]]),
        frame_duration_ms=torch.tensor([500, 500]),
    )
    model.embed_multimodal(**kwargs)
    assert [k["video_frame_sizes"].tolist() for k in seen] == [[[32, 64]], [[48, 48]]]
    assert all(k["pixel_values_raw_video"][0] is f for k, f in zip(seen, frames))
    assert [k["video_num_patches"].tolist() for k in seen] == [[2], [2]]

    seen.clear()
    same_size = torch.tensor([[32, 64], [32, 64]])
    model.embed_multimodal(**{**kwargs, "video_frame_sizes": same_size})
    assert len(seen) == 1


@pytest.mark.parametrize("input_key", ["image_embeds", "video_embeds"])
def test_precomputed_multimodal_embeddings(input_key: str):
    model = _uninitialized_model()
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


def test_nano_nemotron_vl_skips_multimodal_weights_in_text_only_mode():
    model = _uninitialized_model()
    language_model = _LanguageModel()
    object.__setattr__(model, "model_config", _ModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", _MissingMultiModalModule())
    object.__setattr__(model, "vision_final_layernorm", _MissingMultiModalModule())
    object.__setattr__(model, "sound_encoder", None)

    language_weight = torch.tensor(1.0)
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("mlp1.0.weight", torch.empty(0)),
            ("vision_model.radio_model.encoder.weight", torch.empty(0)),
            ("sound_encoder.encoder.weight", torch.empty(0)),
            ("vision_projector.vision_final_layernorm.weight", torch.empty(0)),
            ("vision_projector.vision_final_layernorm.bias", torch.empty(0)),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]


def test_nano_nemotron_vl_loads_vision_weights_without_sound_encoder():
    model = _uninitialized_model()
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "vision_final_layernorm", None)
    object.__setattr__(model, "sound_encoder", None)

    language_weight = torch.tensor(1.0)
    vision_weight = torch.tensor([2.0, 3.0])
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("vision_model.radio_model.encoder.weight", vision_weight),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]
    assert len(vision_model.loaded_weights) == 1
    name, loaded_weight = vision_model.loaded_weights[0]
    assert name == "radio_model.encoder.weight"
    assert isinstance(loaded_weight, torch.Tensor)
    torch.testing.assert_close(loaded_weight, vision_weight)
    assert loaded_weight is not vision_weight


def test_nano_nemotron_vl_requires_sound_encoder_for_sound_weights():
    model = _uninitialized_model()
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "vision_final_layernorm", None)
    object.__setattr__(model, "sound_encoder", None)

    with pytest.raises(AssertionError):
        model.load_weights([("sound_encoder.encoder.weight", torch.empty(0))])


def _vision_norm_model(use_layernorm=True):
    model = _uninitialized_model()
    torch.nn.Module.__init__(model)
    model.model_config = _ImageOnlyModelConfig()
    model.language_model = _LanguageModel()
    model.vision_model = _VisionModel()
    model.sound_encoder = None
    model.input_norm = None
    model.mlp1 = torch.nn.Identity()
    model.vision_final_layernorm = (
        torch.nn.LayerNorm(4, eps=1e-6, dtype=torch.bfloat16) if use_layernorm else None
    )
    model.patch_size = 2
    model.downsample_ratio = 0.5
    model.ps_version = "v2"
    model.video_temporal_patch_size = 2
    return model


@pytest.mark.parametrize("mode", ["image", "dynamic", "video"])
@pytest.mark.parametrize("use_layernorm", [True, False])
def test_vision_layernorm_is_loaded_and_applied_before_pixel_shuffle(
    mode, use_layernorm
):
    model = _vision_norm_model(use_layernorm)
    weight = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.bfloat16)
    bias = torch.tensor([0.5, -0.5, 1.0, -1.0], dtype=torch.bfloat16)
    model.load_weights(
        [
            ("vision_projector.vision_final_layernorm.weight", weight),
            ("vision_projector.vision_final_layernorm.bias", bias),
        ]
        if use_layernorm
        else []
    )
    features = torch.arange(16, dtype=torch.bfloat16).reshape(1, 4, 4)
    expected = (
        torch.nn.functional.layer_norm(features, (4,), weight, bias, eps=1e-6)
        if use_layernorm
        else features
    )
    expected = model.pixel_shuffle(expected.reshape(1, 2, 2, 4)).reshape(1, 1, 16)
    model.vision_model = lambda *args, **kwargs: (None, features)
    pixels = torch.zeros(2 if mode == "video" else 1, 3, 4, 4)
    if mode == "dynamic":
        output = model.extract_feature_dynamic(pixels, imgs_sizes=[(4, 4)])
    else:
        output = model.extract_feature(
            pixels, num_frames=2 if mode == "video" else None
        )
    torch.testing.assert_close(output, expected)


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


@pytest.mark.core_model
@pytest.mark.parametrize(
    "model",
    [
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
    ],
)
@pytest.mark.parametrize(
    ("requested", "expected"), [("auto", "float32"), ("bfloat16", "bfloat16")]
)
def test_omni_mamba_ssm_cache_dtype(model: str, requested: str, expected: str):
    """Omni archs get the NemotronH float32 SSM-cache default; explicit wins."""
    vllm_config = EngineArgs(
        model=model,
        trust_remote_code=True,
        mamba_ssm_cache_dtype=requested,
        max_model_len=4096,
    ).create_engine_config()
    assert vllm_config.cache_config.mamba_ssm_cache_dtype == expected
