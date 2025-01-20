from functools import partial

import numpy as np
import pytest
from PIL import Image

from vllm.config import ModelConfig
from vllm.inputs import InputProcessingContext
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import ProcessingCache
from vllm.multimodal.utils import cached_get_tokenizer

from ....multimodal.utils import random_audio, random_image, random_video
from ...registry import HF_EXAMPLE_MODELS


def _test_processing_correctness(
    model_id: str,
    modalities: dict[str, bool],
    hit_rate: float,
    num_batches: int,
    simplify_rate: float,
):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    limit_mm_per_prompt = {
        modality: 3 if supports_multi else 1
        for modality, supports_multi in modalities.items()
    }

    model_config = ModelConfig(
        model_id,
        task="auto",
        tokenizer=model_id,
        tokenizer_mode="auto",
        trust_remote_code=True,
        seed=0,
        dtype="float16",
        revision=None,
        hf_overrides=model_info.hf_overrides,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )

    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)
    factories = MULTIMODAL_REGISTRY._processor_factories[model_cls]
    ctx = InputProcessingContext(
        model_config,
        tokenizer=cached_get_tokenizer(model_config.tokenizer),
    )
    # Ensure that it can fit all of the data
    cache = ProcessingCache(capacity=1 << 30)

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
             for _ in range(rng.randint(limit_mm_per_prompt[k]))]
            for k in modalities
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

        assert baseline_result == cached_result, (
            f"Failed ({batch_idx=}, {prompt=}, {mm_data=})")

        baseline_tokenized_result = baseline_processor.apply(
            tokenizer.encode(prompt),
            mm_data=mm_data,
            hf_processor_mm_kwargs={},
        )

        assert baseline_result == baseline_tokenized_result, (
            f"Failed ({batch_idx=}, {prompt=}, {mm_data=})")

        cached_tokenized_result = cached_processor.apply(
            tokenizer.encode(prompt),
            mm_data=mm_data,
            hf_processor_mm_kwargs={},
        )

        assert cached_result == cached_tokenized_result, (
            f"Failed ({batch_idx=}, {prompt=}, {mm_data=})")


# yapf: disable
# True if the model supports multiple data items of the modality per request
@pytest.mark.parametrize(("model_id", "modalities"), [
    ("rhymes-ai/Aria", {"image": True}),
    ("Salesforce/blip2-opt-2.7b", {"image": False}),
    ("facebook/chameleon-7b", {"image": False}),
    ("deepseek-ai/deepseek-vl2-tiny", {"image": True}),
    ("adept/fuyu-8b", {"image": False}),
    ("llava-hf/llava-1.5-7b-hf", {"image": True}),
    ("llava-hf/llava-v1.6-mistral-7b-hf", {"image": True}),
    ("llava-hf/LLaVA-NeXT-Video-7B-hf", {"video": False}),
    ("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", {"image": True, "video": True}),  # noqa: E501
    ("TIGER-Lab/Mantis-8B-siglip-llama3", {"image": True}),
    ("mistral-community/pixtral-12b", {"image": True}),
    ("Qwen/Qwen2-VL-2B-Instruct", {"image": True, "video": True}),
    ("Qwen/Qwen2-Audio-7B-Instruct", {"audio": True}),
    ("fixie-ai/ultravox-v0_3", {"audio": True}),
])
@pytest.mark.parametrize("hit_rate", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("num_batches", [32])
@pytest.mark.parametrize("simplify_rate", [1.0])
# yapf: enable
def test_processing_correctness(
    model_id: str,
    modalities: dict[str, bool],
    hit_rate: float,
    num_batches: int,
    simplify_rate: float,
):
    _test_processing_correctness(
        model_id,
        modalities,
        hit_rate=hit_rate,
        num_batches=num_batches,
        simplify_rate=simplify_rate,
    )


# yapf: disable
@pytest.mark.parametrize(("model_id", "modalities"), [
    ("microsoft/Phi-3-vision-128k-instruct", {"image": True}),
])
@pytest.mark.parametrize("hit_rate", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("num_batches", [32])
@pytest.mark.parametrize("simplify_rate", [1.0])
# yapf: enable
def test_processing_correctness_phi3v(
    model_id: str,
    modalities: dict[str, bool],
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
        modalities,
        hit_rate=hit_rate,
        num_batches=num_batches,
        simplify_rate=simplify_rate,
    )
