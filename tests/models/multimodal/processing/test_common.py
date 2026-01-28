# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Set as AbstractSet
from functools import partial

import numpy as np
import pytest
from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from PIL import Image

from vllm.config import ModelConfig
from vllm.config.multimodal import (
    AudioDummyOptions,
    BaseDummyOptions,
    ImageDummyOptions,
    VideoDummyOptions,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalDataDict
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.inputs import MultiModalInputs, batched_tensors_equal
from vllm.multimodal.processing import BaseMultiModalProcessor, InputProcessingContext
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config
from vllm.tokenizers.mistral import MistralTokenizer

from ....multimodal.utils import random_audio, random_image, random_video
from ...registry import (
    _MULTIMODAL_EXAMPLE_MODELS,
    _TRANSFORMERS_BACKEND_MODELS,
    HF_EXAMPLE_MODELS,
)


def glm4_1v_patch_mm_data(mm_data: MultiModalDataDict) -> MultiModalDataDict:
    """
    Patch the multimodal data for GLM4.1V model.
    """
    # Ensure video metadata is included
    if "video" in mm_data:
        # GLM4.1V doesn't support multiple videos
        video = mm_data["video"]
        num_frames = len(video)
        mm_data["video"] = (
            video,
            {
                "total_num_frames": num_frames,
                "fps": num_frames,
                "duration": 1,
                "frames_indices": [i for i in range(num_frames)],
                "video_backend": "opencv",
                "do_sample_frames": True,
            },
        )
    return mm_data


def qwen3_vl_patch_mm_data(mm_data: MultiModalDataDict) -> MultiModalDataDict:
    """
    Patch the multimodal data for Qwen3-VL model.
    """

    def create_metadata(frames: np.ndarray):
        num_frames = len(frames)
        return {
            "total_num_frames": num_frames,
            "fps": 2.0,
            "duration": num_frames / 2.0,
            "video_backend": "opencv",
            "frames_indices": list(range(num_frames)),
            "do_sample_frames": True,
        }

    # Ensure video metadata is included
    if "video" in mm_data:
        video = mm_data["video"]
        if isinstance(video, list):
            # multiple videos
            mm_data["video"] = [(vid, create_metadata(vid)) for vid in video]
        else:
            # single video
            mm_data["video"] = (video, create_metadata(video))
    return mm_data


def glmasr_patch_mm_data(mm_data: MultiModalDataDict) -> MultiModalDataDict:
    """
    Patch the multimodal data for GLM-ASR model.
    GLM-ASR requires text and audio to match 1:1, so we limit audio to 1.
    """
    if "audio" in mm_data:
        audio = mm_data["audio"]
        if isinstance(audio, list) and len(audio) > 1:
            # Limit to single audio to match text requirement
            mm_data["audio"] = [audio[0]]
    return mm_data


# For some multimodal models, tokenizer will always add bos_token
# at the beginning of prompt by default, causing hf_processor outputs
# incorrect token ids. So we need use `add_special_tokens=False` here
# to leave bos_token to be added by the processor.
_ADD_SPECIAL_TOKENS_OVERRIDES = {
    "nemotron_parse": False,
    "ovis": False,
    "ovis2_5": False,
    "paligemma": False,
    "ultravox": False,
    "whisper": False,
}

_IGNORE_MM_KEYS = {
    # In Ultravox, the audio_features can be different depending on padding
    # The slight difference should not be a problem though, since
    # attention_mask lets us ignore the difference.
    "ultravox": {"audio_features"},
}

MM_DATA_PATCHES = {
    # Ernie4.5-VL, GLM4.1V and Qwen3-VL requires video metadata
    "ernie4_5_moe_vl": qwen3_vl_patch_mm_data,
    "glm4v": glm4_1v_patch_mm_data,
    "glm4v_moe": glm4_1v_patch_mm_data,
    "glm_ocr": glm4_1v_patch_mm_data,
    "glmasr": glmasr_patch_mm_data,
    "molmo2": qwen3_vl_patch_mm_data,
    "qwen3_vl": qwen3_vl_patch_mm_data,
    "qwen3_vl_moe": qwen3_vl_patch_mm_data,
}


def _iter_model_ids_to_test(model_arch_list: AbstractSet[str]):
    for model_arch in model_arch_list:
        model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
        yield model_info.default

        for extra_type, extra_model_id in model_info.extras.items():
            if "fp" in extra_type:
                continue  # Redundant to test quantized models

            yield extra_model_id


def _get_model_ids_to_test(model_arch_list: AbstractSet[str]):
    return list(_iter_model_ids_to_test(model_arch_list))


def get_model_ids_to_test():
    transformers_arch_ids = {
        model_id
        for info in _TRANSFORMERS_BACKEND_MODELS.values()
        for model_id in (info.default, *info.extras.values())
    }
    vllm_only_archs = {
        arch
        for arch, info in _MULTIMODAL_EXAMPLE_MODELS.items()
        if not any(
            model_id in transformers_arch_ids
            for model_id in (info.default, *info.extras.values())
        )
    }

    return _get_model_ids_to_test(vllm_only_archs)


def get_text_token_prompts(
    processor: BaseMultiModalProcessor,
    mm_data: MultiModalDataDict,
):
    dummy_inputs = processor.dummy_inputs
    tokenizer: TokenizerLike = processor.info.get_tokenizer()
    model_config = processor.info.ctx.model_config

    model_type = model_config.hf_config.model_type
    if model_type in MM_DATA_PATCHES:
        mm_data = MM_DATA_PATCHES[model_type](mm_data)

    parsed_data = processor.data_parser.parse_mm_data(mm_data)
    mm_counts = {k: len(vs) for k, vs in parsed_data.items()}

    text_prompt: str | None
    token_prompt: list[int]
    if isinstance(tokenizer, MistralTokenizer):
        images = parsed_data.get("image", [])
        request = ChatCompletionRequest(
            messages=[
                UserMessage(
                    content=[
                        TextChunk(text=""),
                        *(ImageChunk(image=image) for image in images),
                    ]
                ),
            ]
        )
        res = tokenizer.mistral.encode_chat_completion(request)

        # Mistral does not support decode_tokens with skip_special_tokens=False
        text_prompt = None
        token_prompt = res.tokens
    else:
        inputs = dummy_inputs.get_dummy_processor_inputs(
            model_config.max_model_len,
            mm_counts,
        )
        assert isinstance(inputs.prompt, str)

        text_prompt = inputs.prompt
        token_prompt = tokenizer.encode(
            text_prompt,
            add_special_tokens=_ADD_SPECIAL_TOKENS_OVERRIDES.get(model_type, True),
        )

    return text_prompt, token_prompt


def _test_processing_correctness(
    model_id_or_arch: str,
    hit_rate: float,
    num_batches: int,
    simplify_rate: float,
):
    if model_id_or_arch in HF_EXAMPLE_MODELS.get_supported_archs():
        # Use model architecture to get the default model id
        model_info = HF_EXAMPLE_MODELS.get_hf_info(model_id_or_arch)
        model_id = model_info.default
    else:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id_or_arch)
        model_id = model_id_or_arch
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(
        on_fail="skip",
        check_max_version=False,
        check_version_reason="vllm",
    )

    model_config = ModelConfig(
        model_id,
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
        # Ensure that the cache can fit all of the data
        mm_processor_cache_gb=2048,
        skip_tokenizer_init=model_info.require_embed_inputs,
        enable_prompt_embeds=model_info.require_embed_inputs,
        enable_mm_embeds=model_info.require_embed_inputs,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype,
    )

    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)
    factories = model_cls._processor_factory
    ctx = InputProcessingContext(
        model_config,
        tokenizer=cached_tokenizer_from_config(model_config),
    )
    cache = MultiModalProcessorOnlyCache(model_config)

    processing_info = factories.info(ctx)
    supported_mm_limits = processing_info.get_supported_mm_limits()
    # Keep integer limits for local data generation
    limit_mm_per_prompt_ints = {
        modality: 3 if limit is None else limit
        for modality, limit in supported_mm_limits.items()
    }

    def _to_dummy_options(modality: str, count: int) -> BaseDummyOptions:
        if modality == "video":
            return VideoDummyOptions(count=count)
        if modality == "image":
            return ImageDummyOptions(count=count)
        if modality == "audio":
            return AudioDummyOptions(count=count)
        return BaseDummyOptions(count=count)

    # Assign normalized DummyOptions to the model config
    model_config.get_multimodal_config().limit_per_prompt = {
        modality: _to_dummy_options(modality, count)
        for modality, count in limit_mm_per_prompt_ints.items()
    }

    baseline_processor = factories.build_processor(ctx, cache=None)
    cached_processor = factories.build_processor(ctx, cache=cache)

    rng = np.random.RandomState(0)

    input_to_hit = {
        "image": Image.new("RGB", size=(128, 128)),
        "video": np.zeros((4, 128, 128, 3), dtype=np.uint8),
        "audio": (np.zeros((512,)), 16000),
    }
    input_factory = {
        "image": partial(random_image, rng, min_wh=128, max_wh=256),
        "video": partial(
            random_video, rng, min_frames=2, max_frames=16, min_wh=128, max_wh=256
        ),
        "audio": partial(random_audio, rng, min_len=512, max_len=1024, sr=16000),
    }

    for batch_idx in range(num_batches):
        mm_data = {
            k: [
                (input_to_hit[k] if rng.rand() < hit_rate else input_factory[k]())
                for _ in range(rng.randint(limit + 1))
            ]
            for k, limit in limit_mm_per_prompt_ints.items()
        }

        # Drop unnecessary keys and test single -> multi conversion
        if rng.rand() < simplify_rate:
            for k in list(mm_data.keys()):
                if not mm_data[k]:
                    del mm_data[k]
                elif len(mm_data[k]) == 1:
                    mm_data[k] = mm_data[k][0]

        _test_processing_correctness_one(
            model_config,
            mm_data,
            baseline_processor,
            cached_processor,
            batch_idx,
        )


def _test_processing_correctness_one(
    model_config: ModelConfig,
    mm_data: MultiModalDataDict,
    baseline_processor: BaseMultiModalProcessor,
    cached_processor: BaseMultiModalProcessor,
    batch_idx: int,
):
    model_type = model_config.hf_config.model_type

    text_prompt, token_prompt = get_text_token_prompts(baseline_processor, mm_data)
    ignore_mm_keys = _IGNORE_MM_KEYS.get(model_type, set[str]())

    baseline_tokenized_result = baseline_processor.apply(
        token_prompt,
        mm_data=mm_data,
        hf_processor_mm_kwargs={},
    )

    cached_tokenized_result = cached_processor.apply(
        token_prompt,
        mm_data=mm_data,
        hf_processor_mm_kwargs={},
    )

    _assert_inputs_equal(
        baseline_tokenized_result,
        cached_tokenized_result,
        ignore_mm_keys=ignore_mm_keys,
        msg=f"Failed ({batch_idx=}, {token_prompt=}, {mm_data=})",
    )

    if text_prompt is not None:
        baseline_text_result = baseline_processor.apply(
            text_prompt,
            mm_data=mm_data,
            hf_processor_mm_kwargs={},
        )
        cached_text_result = cached_processor.apply(
            text_prompt,
            mm_data=mm_data,
            hf_processor_mm_kwargs={},
        )

        _assert_inputs_equal(
            baseline_text_result,
            cached_text_result,
            ignore_mm_keys=ignore_mm_keys,
            msg=f"Failed ({batch_idx=}, {text_prompt=}, {mm_data=})",
        )

        _assert_inputs_equal(
            baseline_text_result,
            baseline_tokenized_result,
            ignore_mm_keys=ignore_mm_keys,
            msg=f"Failed ({batch_idx=}, {text_prompt=}, {token_prompt=}, {mm_data=})",
        )

        _assert_inputs_equal(
            cached_text_result,
            cached_tokenized_result,
            ignore_mm_keys=ignore_mm_keys,
            msg=f"Failed ({batch_idx=}, {text_prompt=}, {token_prompt=}, {mm_data=})",
        )


@pytest.mark.parametrize("model_id", get_model_ids_to_test())
@pytest.mark.parametrize("hit_rate", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("num_batches", [32])
@pytest.mark.parametrize("simplify_rate", [1.0])
def test_processing_correctness(
    model_id: str,
    hit_rate: float,
    num_batches: int,
    simplify_rate: float,
):
    if model_id == "google/gemma-3n-E2B-it":
        pytest.skip("Fix later")
    if model_id == "OpenGVLab/InternVL2-2B":
        pytest.skip("Fix later")
    if model_id == "jinaai/jina-reranker-m0":
        pytest.skip("Fix later")
    if model_id in {"Qwen/Qwen-VL", "Qwen/Qwen-VL-Chat"}:
        pytest.skip(
            "Qwen-VL tokenizer requires downloading a font file from "
            "servers that often refuse connections in CI"
        )

    _test_processing_correctness(
        model_id,
        hit_rate=hit_rate,
        num_batches=num_batches,
        simplify_rate=simplify_rate,
    )


def _assert_inputs_equal(
    a: MultiModalInputs,
    b: MultiModalInputs,
    *,
    ignore_mm_keys: set[str] | None = None,
    msg: str = "",
):
    if ignore_mm_keys is None:
        ignore_mm_keys = set()

    a_rest = {k: v for k, v in a.items() if k != "mm_kwargs"}
    b_rest = {k: v for k, v in b.items() if k != "mm_kwargs"}

    assert a_rest == b_rest, msg

    a_data = a["mm_kwargs"].get_data()
    b_data = b["mm_kwargs"].get_data()

    for key in ignore_mm_keys:
        a_data.pop(key, None)
        b_data.pop(key, None)

    assert batched_tensors_equal(a_data, b_data), msg
