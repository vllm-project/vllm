# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from functools import partial
from typing import Any, Union
from unittest.mock import patch

import numpy as np
import pytest
from mistral_common.protocol.instruct.messages import (ImageChunk, TextChunk,
                                                       UserMessage)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from PIL import Image

from vllm.config import ModelConfig
from vllm.engine.llm_engine import LLMEngine as V0LLMEngine
from vllm.inputs import InputProcessingContext
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs)
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config
from vllm.utils import GiB_bytes, is_list_of, set_default_torch_num_threads
from vllm.v1.core.kv_cache_utils import get_kv_cache_config
from vllm.v1.engine.core import EngineCore as V1EngineCore

from ...conftest import VllmRunner
from ..registry import _MULTIMODAL_EXAMPLE_MODELS, HF_EXAMPLE_MODELS
from ..utils import dummy_hf_overrides

ARCH_TO_SKIP = {
    "MolmoForCausalLM": "incompatible requirements",
    "MiniMaxVL01ForConditionalGeneration": "broken model",
}
ARCH_NEEDS_EXTRAS = [
    "InternVLChatModel",
    "Idefics3ForConditionalGeneration",
    "LlavaForConditionalGeneration",
    "MiniCPMV",
    "PaliGemmaForConditionalGeneration",
]
REPO_ID_TO_SKIP = {"nm-testing/pixtral-12b-FP8-dynamic": "duplicated test"}

ImageInput = list[Image.Image]
VideoInput = Union[list[Image.Image], list[np.ndarray],
                   list[tuple[np.ndarray, dict[str, Any]]]]
AudioInput = list[tuple[np.ndarray, int]]


def _resize_data(_data: Union[Image.Image, np.ndarray],
                 size_factor: float) -> Union[Image.Image, np.ndarray]:
    assert size_factor <= 1, "Size factor must be less than 1"
    # Image input
    if isinstance(_data, Image.Image):
        W, H = _data.width, _data.height
        W, H = map(lambda x: int(x * size_factor), (W, H))
        return _data.resize((W, H))
    # Video input with PIL Images
    elif is_list_of(_data, Image.Image):
        W, H = next(iter(_data)).width, next(iter(_data)).height
        T = len(_data)
        T, W, H = map(lambda x: max(int(x * size_factor), 1), (T, W, H))
        return [d.resize((W, H)) for d in _data[:T]]
    # Video input with numpy arrays
    elif isinstance(_data, np.ndarray) and _data.ndim >= 4:
        T, H, W, C = _data.shape[-4:]
        T, H, W = map(lambda x: max(int(x * size_factor), 1), (T, H, W))
        return _data[..., :T, :H, :W, :C]
    # Audio input
    elif isinstance(_data, np.ndarray) and _data.ndim == 1:
        return _data[:int(len(_data) * size_factor)]
    raise AssertionError("This line should be unreachable.")


def resize_mm_data(
    data: Union[ImageInput, VideoInput, AudioInput],
    size_factors: tuple[float,
                        ...]) -> Union[ImageInput, VideoInput, AudioInput]:
    size_factors = size_factors[:len(data)]
    if is_list_of(data, (Image.Image, np.ndarray, list)):
        return [_resize_data(d, s) for d, s in zip(data, size_factors)]
    elif is_list_of(data, tuple):
        return [(_resize_data(d, s), meta)
                for (d, meta), s in zip(data, size_factors)]
    raise ValueError("Unsupported multimodal data type.")


def create_batched_mm_kwargs(
    model_config: ModelConfig,
    processor: BaseMultiModalProcessor,
    size_factors: tuple[float, ...] = (1.0, 0.5, 0.25),
) -> Iterable[tuple[str, int, BatchedTensorInputs]]:
    processing_info = processor.info
    dummy_inputs = processor.dummy_inputs
    supported_mm_limits = processing_info.get_supported_mm_limits()
    mm_counts = {
        modality: 3 if limit is None else limit
        for modality, limit in supported_mm_limits.items()
    }
    processor_inputs = dummy_inputs.get_dummy_processor_inputs(
        seq_len=model_config.max_model_len,
        mm_counts=mm_counts,
    )
    mm_data = processor_inputs.mm_data
    resized_mm_data = {
        modality: resize_mm_data(data, size_factors)
        for modality, data in mm_data.items()
    }
    # Mistral chat outputs tokens directly, rather than text prompts
    if model_config.tokenizer_mode == "mistral":
        images = resized_mm_data.get("image", [])
        request = ChatCompletionRequest(messages=[
            UserMessage(content=[
                TextChunk(text=""),
                *(ImageChunk(image=image) for image in images),
            ]),
        ])
        tokenizer = processing_info.get_tokenizer()
        res = tokenizer.mistral.encode_chat_completion(request)
        prompt = res.tokens
    else:
        prompt = processor_inputs.prompt
    mm_kwargs = processor.apply(
        prompt=prompt,
        mm_data=resized_mm_data,
        hf_processor_mm_kwargs=processor_inputs.hf_processor_mm_kwargs,
        tokenization_kwargs=processor_inputs.tokenization_kwargs,
    )["mm_kwargs"]
    items = [
        item for modality in supported_mm_limits
        for item in mm_kwargs[modality]
    ]
    return group_mm_kwargs_by_modality(items)


def get_model_id_to_test(
        model_arch_list: Iterable[str]) -> list[tuple[str, str]]:
    filtered_results = []
    for model_arch in model_arch_list:
        model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
        if model_info.extras and model_arch in ARCH_NEEDS_EXTRAS:
            available_repos = list(
                map(lambda model_id: (model_arch, model_id),
                    [model_info.default, *model_info.extras.values()]))
            filtered_results.extend(available_repos)
        else:
            filtered_results.append((model_arch, model_info.default))
    return filtered_results


@pytest.mark.core_model
@pytest.mark.parametrize(
    "model_arch, model_id",
    get_model_id_to_test(_MULTIMODAL_EXAMPLE_MODELS.keys()))
def test_model_tensor_schema(model_arch: str, model_id: str,
                             vllm_runner: type[VllmRunner], monkeypatch):
    if model_arch in ARCH_TO_SKIP:
        pytest.skip(f"Skipping {model_arch} due to {ARCH_TO_SKIP[model_arch]}")
    if model_id in REPO_ID_TO_SKIP:
        pytest.skip(f"Skipping {model_id} due to {REPO_ID_TO_SKIP[model_id]}")

    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip",
                                          check_max_version=False)

    hf_overrides_fn = partial(dummy_hf_overrides,
                              model_arch=model_arch,
                              exist_overrides=model_info.hf_overrides)

    model_config = ModelConfig(
        model_id,
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
    )
    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)
    factories = MULTIMODAL_REGISTRY._processor_factories[model_cls]

    if not any(
            hasattr(model_cls, f"_parse_and_validate_{m}_input")
            for m in ["image", "video", "audio"]):
        pytest.skip(f"{model_arch} does not support tensor schema validation.")

    ctx = InputProcessingContext(
        model_config,
        tokenizer=cached_tokenizer_from_config(model_config),
    )
    processing_info = factories.info(ctx)
    supported_mm_limits = processing_info.get_supported_mm_limits()
    limit_mm_per_prompt = {
        modality: 3 if limit is None else limit
        for modality, limit in supported_mm_limits.items()
    }

    # Avoid calling model.forward()
    def _initialize_kv_caches_v0(self) -> None:
        self.cache_config.num_gpu_blocks = 0
        self.cache_config.num_cpu_blocks = 0

    def _initialize_kv_caches_v1(self, vllm_config):
        kv_cache_specs = self.model_executor.get_kv_cache_specs()
        scheduler_kv_cache_config = get_kv_cache_config(
            vllm_config,
            kv_cache_specs[0],
            10 * GiB_bytes,
        )

        # gpu_blocks (> 0), cpu_blocks, scheduler_kv_cache_config
        return 1, 0, scheduler_kv_cache_config

    with (patch.object(V0LLMEngine, "_initialize_kv_caches",
                       _initialize_kv_caches_v0),
          patch.object(V1EngineCore, "_initialize_kv_caches",
                       _initialize_kv_caches_v1), monkeypatch.context() as m):
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        if model_info.v0_only:
            m.setenv("VLLM_USE_V1", "0")

        # TODO(Isotr0py): Can we avoid initializing engine?
        with (
                set_default_torch_num_threads(1),
                vllm_runner(
                    model_id,
                    tokenizer_name=model_info.tokenizer,
                    tokenizer_mode=model_info.tokenizer_mode,
                    revision=model_info.revision,
                    trust_remote_code=model_info.trust_remote_code,
                    max_model_len=model_info.max_model_len,
                    load_format="dummy",
                    hf_overrides=hf_overrides_fn,
                    limit_mm_per_prompt=limit_mm_per_prompt,
                    enforce_eager=True,
                ) as vllm_model,
        ):
            model_config = vllm_model.llm.llm_engine.model_config
            llm_engine = vllm_model.llm.llm_engine

            if hasattr(llm_engine, "processor"):
                # v1 processor
                mm_registry = llm_engine.processor.mm_registry
            else:
                # v0 input_preprocessor
                mm_registry = llm_engine.input_preprocessor.mm_registry

            processor = mm_registry.create_processor(model_config)

            def validate_model_input(model, modality: str,
                                     mm_kwargs: MultiModalKwargs):
                method_name = f"_parse_and_validate_{modality}_input"
                if hasattr(model, method_name):
                    getattr(model, method_name)(**mm_kwargs)

            for modality, _, mm_kwargs in create_batched_mm_kwargs(
                    model_config, processor):
                valid_func = partial(validate_model_input,
                                     modality=modality,
                                     mm_kwargs=mm_kwargs)
                vllm_model.apply_model(valid_func)
