# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Optional, Union

import numpy as np
import pytest
from mistral_common.protocol.instruct.messages import (ImageChunk, TextChunk,
                                                       UserMessage)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from PIL import Image
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.config import ModelConfig
from vllm.inputs import InputProcessingContext
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalDataDict
from vllm.multimodal.inputs import MultiModalInputs
from vllm.multimodal.processing import BaseMultiModalProcessor, ProcessingCache
from vllm.transformers_utils.tokenizer import (MistralTokenizer,
                                               cached_tokenizer_from_config)

from ....multimodal.utils import random_audio, random_image, random_video
from ...registry import HF_EXAMPLE_MODELS


def _test_processing_correctness(
    model_id: str,
    hit_rate: float,
    num_batches: int,
    simplify_rate: float,
    ignore_mm_keys: Optional[set[str]] = None,
):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    model_config = ModelConfig(
        model_id,
        task="auto",
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=model_info.tokenizer_mode,
        trust_remote_code=model_info.trust_remote_code,
        seed=0,
        dtype="float16",
        revision=None,
        hf_overrides=model_info.hf_overrides,
    )

    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)
    factories = MULTIMODAL_REGISTRY._processor_factories[model_cls]
    ctx = InputProcessingContext(
        model_config,
        tokenizer=cached_tokenizer_from_config(model_config),
    )
    # Ensure that it can fit all of the data
    cache = ProcessingCache(capacity_gb=2048)

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
                max_frames=8,
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
        prompt = dummy_inputs.get_dummy_processor_inputs(
            model_config.max_model_len,
            mm_counts,
        ).prompt_text

        # Drop unnecessary keys and test single -> multi conversion
        if rng.rand() < simplify_rate:
            for k in list(mm_data.keys()):
                if not mm_data[k]:
                    del mm_data[k]
                elif len(mm_data[k]) == 1:
                    mm_data[k] = mm_data[k][0]

        if isinstance(tokenizer, MistralTokenizer):
            _test_processing_correctness_mistral(
                model_config,
                tokenizer,
                prompt,
                mm_data,
                baseline_processor,
                cached_processor,
                batch_idx,
                ignore_mm_keys=ignore_mm_keys,
            )
        else:
            _test_processing_correctness_hf(
                model_config,
                tokenizer,
                prompt,
                mm_data,
                baseline_processor,
                cached_processor,
                batch_idx,
                ignore_mm_keys=ignore_mm_keys,
            )


def _test_processing_correctness_hf(
    model_config: ModelConfig,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prompt: str,
    mm_data: MultiModalDataDict,
    baseline_processor: BaseMultiModalProcessor,
    cached_processor: BaseMultiModalProcessor,
    batch_idx: int,
    ignore_mm_keys: Optional[set[str]] = None,
):
    if model_config.hf_config.model_type in ("mllama", "whisper", "ultravox"):
        # For some multimodal models, tokenizer will always add bos_token
        # at the beginning of prompt by default, causing hf_processor outputs
        # incorrect token ids. So we need use `add_special_tokens=False` here
        # to leave bos_token to be added by the processor.
        token_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        token_prompt = tokenizer.encode(prompt)

    baseline_result = baseline_processor.apply(
        prompt,
        mm_data=mm_data,
        hf_processor_mm_kwargs={},
    )
    cached_result = cached_processor.apply(
        prompt,
        mm_data=mm_data,
        hf_processor_mm_kwargs={},
    )

    _assert_inputs_equal(
        baseline_result,
        cached_result,
        ignore_mm_keys=ignore_mm_keys,
        msg=f"Failed ({batch_idx=}, {prompt=}, {mm_data=})",
    )

    baseline_tokenized_result = baseline_processor.apply(
        token_prompt,
        mm_data=mm_data,
        hf_processor_mm_kwargs={},
    )

    _assert_inputs_equal(
        baseline_result,
        baseline_tokenized_result,
        ignore_mm_keys=ignore_mm_keys,
        msg=f"Failed ({batch_idx=}, {prompt=}, {mm_data=})",
    )

    cached_tokenized_result = cached_processor.apply(
        token_prompt,
        mm_data=mm_data,
        hf_processor_mm_kwargs={},
    )

    _assert_inputs_equal(
        cached_result,
        cached_tokenized_result,
        ignore_mm_keys=ignore_mm_keys,
        msg=f"Failed ({batch_idx=}, {prompt=}, {mm_data=})",
    )


def _test_processing_correctness_mistral(
    model_config: ModelConfig,
    tokenizer: MistralTokenizer,
    prompt: str,
    mm_data: MultiModalDataDict,
    baseline_processor: BaseMultiModalProcessor,
    cached_processor: BaseMultiModalProcessor,
    batch_idx: int,
    ignore_mm_keys: Optional[set[str]] = None,
):
    images = mm_data.get("image", [])
    if not isinstance(images, list):
        images = [images]

    request = ChatCompletionRequest(messages=[
        UserMessage(content=[
            TextChunk(text=prompt),
            *(ImageChunk(image=image) for image in images),
        ]),
    ])
    res = tokenizer.mistral.encode_chat_completion(request)
    token_prompt = res.tokens

    # Mistral chat outputs tokens directly, rather than text prompts
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
        msg=f"Failed ({batch_idx=}, {prompt=}, {mm_data=})",
    )


# yapf: disable
@pytest.mark.parametrize("model_id", [
    "rhymes-ai/Aria",
    "CohereForAI/aya-vision-8b",
    "Salesforce/blip2-opt-2.7b",
    "facebook/chameleon-7b",
    "deepseek-ai/deepseek-vl2-tiny",
    "microsoft/Florence-2-base",
    "adept/fuyu-8b",
    "google/gemma-3-4b-it",
    "THUDM/glm-4v-9b",
    "h2oai/h2ovl-mississippi-800m",
    "OpenGVLab/InternVL2-1B",
    "HuggingFaceM4/Idefics3-8B-Llama3",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "moonshotai/Kimi-VL-A3B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "TIGER-Lab/Mantis-8B-siglip-llama3",
    "openbmb/MiniCPM-Llama3-V-2_5",
    "openbmb/MiniCPM-o-2_6",
    "openbmb/MiniCPM-V-2_6",
    "allenai/Molmo-7B-D-0924",
    "allenai/Molmo-7B-O-0924",
    "nvidia/NVLM-D-72B",
    "google/paligemma-3b-mix-224",
    "google/paligemma2-3b-ft-docci-448",
    "microsoft/Phi-4-multimodal-instruct",
    "mistralai/Pixtral-12B-2409",
    "mistral-community/pixtral-12b",
    "Qwen/Qwen-VL-Chat",
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2-Audio-7B-Instruct",
    "Qwen/Qwen2.5-Omni-7B",
    "Skywork/Skywork-R1V-38B",
    "fixie-ai/ultravox-v0_5-llama-3_2-1b",
    "openai/whisper-large-v3",
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
    ignore_mm_keys = None
    if 'ultravox' in model_id:
        # In Ultravox, the audio_features can be different depending on padding
        # The slight difference should not be a problem though, since
        # attention_mask lets us ignore the difference.
        ignore_mm_keys = {"audio_features"}

    _test_processing_correctness(
        model_id,
        hit_rate=hit_rate,
        num_batches=num_batches,
        simplify_rate=simplify_rate,
        ignore_mm_keys=ignore_mm_keys,
    )


# yapf: disable
@pytest.mark.parametrize("model_id", ["microsoft/Phi-3.5-vision-instruct"])
@pytest.mark.parametrize("hit_rate", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("num_batches", [32])
@pytest.mark.parametrize("simplify_rate", [1.0])
# yapf: enable
def test_processing_correctness_phi3v(
    model_id: str,
    hit_rate: float,
    num_batches: int,
    simplify_rate: float,
):
    # HACK - this is an attempted workaround for the following bug
    # https://github.com/huggingface/transformers/issues/34307
    from transformers import AutoImageProcessor  # noqa: F401
    from transformers import AutoProcessor  # noqa: F401

    AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)

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
    ignore_mm_keys: Optional[set[str]] = None,
    msg: str = "",
):
    if ignore_mm_keys is None:
        ignore_mm_keys = set()

    if msg is None:
        assert "mm_kwargs" in a and "mm_kwargs" in b
    else:
        assert "mm_kwargs" in a and "mm_kwargs" in b, msg

    for key in ignore_mm_keys:
        a["mm_kwargs"].pop(key, None)
        b["mm_kwargs"].pop(key, None)

    if msg is None:
        assert a == b
    else:
        assert a == b, msg
