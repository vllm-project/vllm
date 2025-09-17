# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from typing import Optional, Union

import numpy as np
import pytest
from mistral_common.protocol.instruct.messages import (ImageChunk, TextChunk,
                                                       UserMessage)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from PIL import Image

from vllm.config import ModelConfig
from vllm.inputs import InputProcessingContext
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalDataDict
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.inputs import MultiModalInputs
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.transformers_utils.tokenizer import (AnyTokenizer, MistralTokenizer,
                                               cached_tokenizer_from_config,
                                               encode_tokens)

from ....multimodal.utils import random_audio, random_image, random_video
from ...registry import HF_EXAMPLE_MODELS


def glm4_1v_patch_mm_data(mm_data: MultiModalDataDict) -> MultiModalDataDict:
    """
    Patch the multimodal data for GLM4.1V model.
    """
    # Ensure video metadata is included
    if "video" in mm_data:
        # GLM4.1V doesn't support multiple videos
        video = mm_data["video"]
        num_frames = len(video)
        mm_data["video"] = (video, {
            "total_num_frames": num_frames,
            "fps": num_frames,
            "duration": 1,
            "frames_indices": [i for i in range(num_frames)],
            "video_backend": "opencv",
            "do_sample_frames": True,
        })
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
    model_info.check_transformers_version(on_fail="skip")

    model_config = ModelConfig(
        model_id,
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
        # Ensure that the cache can fit all of the data
        mm_processor_cache_gb=2048,
        skip_tokenizer_init=model_info.skip_tokenizer_init,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype)

    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)
    factories = MULTIMODAL_REGISTRY._processor_factories[model_cls]
    ctx = InputProcessingContext(
        model_config,
        tokenizer=cached_tokenizer_from_config(model_config),
    )
    cache = MultiModalProcessorOnlyCache(model_config)

    processing_info = factories.info(ctx)
    supported_mm_limits = processing_info.get_supported_mm_limits()
    limit_mm_per_prompt = {
        modality: 3 if limit is None else limit
        for modality, limit in supported_mm_limits.items()
    }

    model_config.get_multimodal_config().limit_per_prompt = limit_mm_per_prompt

    baseline_processor = factories.build_processor(ctx, cache=None)
    cached_processor = factories.build_processor(ctx, cache=cache)
    dummy_inputs = baseline_processor.dummy_inputs
    tokenizer = baseline_processor.info.get_tokenizer()

    rng = np.random.RandomState(0)

    input_to_hit = {
        "image": Image.new("RGB", size=(128, 128)),
        "video": np.zeros((4, 128, 128, 3), dtype=np.uint8),
        "audio": (np.zeros((512, )), 16000),
    }
    input_factory = {
        "image":
        partial(random_image, rng, min_wh=128, max_wh=256),
        "video":
        partial(random_video,
                rng,
                min_frames=2,
                max_frames=16,
                min_wh=128,
                max_wh=256),
        "audio":
        partial(random_audio, rng, min_len=512, max_len=1024, sr=16000),
    }

    for batch_idx in range(num_batches):
        mm_data = {
            k:
            [(input_to_hit[k] if rng.rand() < hit_rate else input_factory[k]())
             for _ in range(rng.randint(limit + 1))]
            for k, limit in limit_mm_per_prompt.items()
        }

        mm_counts = {k: len(vs) for k, vs in mm_data.items()}

        # Mistral chat outputs tokens directly, rather than text prompts
        if isinstance(tokenizer, MistralTokenizer):
            images = mm_data.get("image", [])
            request = ChatCompletionRequest(messages=[
                UserMessage(content=[
                    TextChunk(text=""),
                    *(ImageChunk(image=image) for image in images),
                ]),
            ])
            res = tokenizer.mistral.encode_chat_completion(request)
            prompt = res.tokens
        else:
            prompt = dummy_inputs.get_dummy_processor_inputs(
                model_config.max_model_len,
                mm_counts,
            ).prompt

        # Drop unnecessary keys and test single -> multi conversion
        if rng.rand() < simplify_rate:
            for k in list(mm_data.keys()):
                if not mm_data[k]:
                    del mm_data[k]
                elif len(mm_data[k]) == 1:
                    mm_data[k] = mm_data[k][0]

        _test_processing_correctness_one(
            model_config,
            tokenizer,
            prompt,
            mm_data,
            baseline_processor,
            cached_processor,
            batch_idx,
        )


# For some multimodal models, tokenizer will always add bos_token
# at the beginning of prompt by default, causing hf_processor outputs
# incorrect token ids. So we need use `add_special_tokens=False` here
# to leave bos_token to be added by the processor.
_ADD_SPECIAL_TOKENS_OVERRIDES = {
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
    # GLM4.1V and Qwen3-VL requires video metadata to be included in the input
    "glm4v": glm4_1v_patch_mm_data,
    "qwen3_vl": qwen3_vl_patch_mm_data,
    "qwen3_vl_moe": qwen3_vl_patch_mm_data,
}


def _test_processing_correctness_one(
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
    prompt: Union[str, list[int]],
    mm_data: MultiModalDataDict,
    baseline_processor: BaseMultiModalProcessor,
    cached_processor: BaseMultiModalProcessor,
    batch_idx: int,
):
    model_type = model_config.hf_config.model_type
    ignore_mm_keys = _IGNORE_MM_KEYS.get(model_type, set[str]())
    if model_type in MM_DATA_PATCHES:
        mm_data = MM_DATA_PATCHES[model_type](mm_data)

    if isinstance(prompt, str):
        text_prompt = prompt
        token_prompt = encode_tokens(
            tokenizer,
            prompt,
            add_special_tokens=_ADD_SPECIAL_TOKENS_OVERRIDES.get(model_type),
        )
    else:
        # Mistral does not support decode_tokens with skip_special_tokens=False
        text_prompt = None
        token_prompt = prompt

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
            msg=f"Failed ({batch_idx=}, {text_prompt=}, "
            f"{token_prompt=}, {mm_data=})",
        )

        _assert_inputs_equal(
            cached_text_result,
            cached_tokenized_result,
            ignore_mm_keys=ignore_mm_keys,
            msg=f"Failed ({batch_idx=}, {text_prompt=}, "
            f"{token_prompt=}, {mm_data=})",
        )


# yapf: disable
@pytest.mark.parametrize("model_id", [
    "rhymes-ai/Aria",
    "CohereForAI/aya-vision-8b",
    "Salesforce/blip2-opt-2.7b",
    "facebook/chameleon-7b",
    "CohereLabs/command-a-vision-07-2025",
    "deepseek-ai/deepseek-vl2-tiny",
    "baidu/ERNIE-4.5-VL-28B-A3B-PT",
    "adept/fuyu-8b",
    "google/gemma-3-4b-it",
    "google/gemma-3n-E2B-it",
    "zai-org/glm-4v-9b",
    "zai-org/GLM-4.1V-9B-Thinking",
    "zai-org/GLM-4.5V",
    "ibm-granite/granite-speech-3.3-2b",
    "h2oai/h2ovl-mississippi-800m",
    "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
    "HuggingFaceM4/Idefics3-8B-Llama3",
    "internlm/Intern-S1",
    "OpenGVLab/InternVL2-1B",
    "OpenGVLab/InternVL3-1B",
    "OpenGVLab/InternVL3_5-1B",
    "OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview",
    "OpenGVLab/InternVL3_5-30B-A3B",
    "Kwai-Keye/Keye-VL-8B-Preview",
    "Kwai-Keye/Keye-VL-1_5-8B",
    "moonshotai/Kimi-VL-A3B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "TIGER-Lab/Mantis-8B-siglip-llama3",
    "mispeech/midashenglm-7b",
    "openbmb/MiniCPM-Llama3-V-2_5",
    "openbmb/MiniCPM-o-2_6",
    "openbmb/MiniCPM-V-2_6",
    "MiniMaxAI/MiniMax-VL-01",
    "allenai/Molmo-7B-D-0924",
    "allenai/Molmo-7B-O-0924",
    "nvidia/NVLM-D-72B",
    "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
    "AIDC-AI/Ovis1.6-Gemma2-9B",
    "AIDC-AI/Ovis1.6-Llama3.2-3B",
    "AIDC-AI/Ovis2-1B",
    "AIDC-AI/Ovis2.5-2B",
    "google/paligemma-3b-mix-224",
    "google/paligemma2-3b-ft-docci-448",
    "microsoft/Phi-3.5-vision-instruct",
    "microsoft/Phi-4-multimodal-instruct",
    "mistralai/Pixtral-12B-2409",
    "mistral-community/pixtral-12b",
    "Qwen/Qwen-VL-Chat",
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2-Audio-7B-Instruct",
    "Qwen/Qwen2.5-Omni-3B",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "YannQi/R-4B",
    "Skywork/Skywork-R1V-38B",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "stepfun-ai/step3",
    "fixie-ai/ultravox-v0_5-llama-3_2-1b",
    "openai/whisper-large-v3",
    "omni-research/Tarsier-7b",
    "omni-research/Tarsier2-Recap-7b",
    "mistralai/Voxtral-Mini-3B-2507",
])
@pytest.mark.parametrize("hit_rate", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("num_batches", [32])
@pytest.mark.parametrize("simplify_rate", [1.0])
# yapf: enable
def test_processing_correctness(
    model_id: str,
    hit_rate: float,
    num_batches: int,
    simplify_rate: float,
):
    if model_id == "google/gemma-3n-E2B-it":
        pytest.skip("Skipping gemma-3n-E2B-it due to transformers #39911 bug.")
    _test_processing_correctness(
        model_id,
        hit_rate=hit_rate,
        num_batches=num_batches,
        simplify_rate=simplify_rate,
    )


# Phi4MultimodalForCausalLM share same model repo with original format
# Phi4MMForCausalLM, so we add it as a separate test case
# Remove this test after conversion PR merged:
# https://huggingface.co/microsoft/Phi-4-multimodal-instruct/discussions/70
@pytest.mark.parametrize("model_arch", ["Phi4MultimodalForCausalLM"])
@pytest.mark.parametrize("hit_rate", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("num_batches", [32])
@pytest.mark.parametrize("simplify_rate", [1.0])
def test_processing_correctness_phi4_multimodal(
    model_arch: str,
    hit_rate: float,
    num_batches: int,
    simplify_rate: float,
):
    _test_processing_correctness(
        model_arch,
        hit_rate=hit_rate,
        num_batches=num_batches,
        simplify_rate=simplify_rate,
    )


def _assert_inputs_equal(
    a: MultiModalInputs,
    b: MultiModalInputs,
    *,
    ignore_mm_keys: Optional[set[str]] = None,
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

    assert a_data == b_data, msg
