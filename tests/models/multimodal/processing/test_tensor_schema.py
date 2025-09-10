# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from functools import partial
from typing import Any, Union

import numpy as np
import pytest
import torch.nn as nn
from mistral_common.protocol.instruct.messages import (ImageChunk, TextChunk,
                                                       UserMessage)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from PIL import Image

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.inputs import InputProcessingContext
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.multimodal import MULTIMODAL_REGISTRY, BatchedTensorInputs
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config
from vllm.utils import is_list_of

from ...registry import _MULTIMODAL_EXAMPLE_MODELS, HF_EXAMPLE_MODELS
from ...utils import dummy_hf_overrides

ARCH_TO_SKIP = {
    "MolmoForCausalLM": "incompatible requirements",
}
ARCH_NEEDS_EXTRAS = [
    "InternVLChatModel",
    "Idefics3ForConditionalGeneration",
    "LlavaForConditionalGeneration",
    "MiniCPMV",
    "PaliGemmaForConditionalGeneration",
]
REPO_ID_TO_SKIP = {
    "nm-testing/pixtral-12b-FP8-dynamic": "duplicated test",
}

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


@contextmanager
def initialize_dummy_model(model_cls: nn.Module, model_config: ModelConfig):
    temp_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    vllm_config = VllmConfig(model_config=model_config)
    with set_current_vllm_config(vllm_config=vllm_config):
        with set_default_torch_dtype(model_config.dtype):
            model = model_cls(vllm_config=vllm_config)
        yield model

    del model
    cleanup_dist_env_and_memory()


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


@pytest.mark.parametrize(
    "model_arch, model_id",
    get_model_id_to_test(_MULTIMODAL_EXAMPLE_MODELS.keys()))
def test_model_tensor_schema(model_arch: str, model_id: str):
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
        hf_overrides=hf_overrides_fn,
        skip_tokenizer_init=model_info.skip_tokenizer_init,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype)
    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)
    factories = MULTIMODAL_REGISTRY._processor_factories[model_cls]

    inputs_parse_methods = []
    for attr_name in dir(model_cls):
        attr = getattr(model_cls, attr_name)
        if hasattr(attr, "__annotations__"):
            return_type = attr.__annotations__.get("return", None)
            if return_type is not None and "Input" in str(return_type):
                inputs_parse_methods.append(attr_name)

    if not any(inputs_parse_methods):
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
    model_config.get_multimodal_config().limit_per_prompt = limit_mm_per_prompt
    processor = factories.build_processor(ctx, cache=None)

    with initialize_dummy_model(model_cls, model_config) as model:
        for modality, _, mm_kwargs in create_batched_mm_kwargs(
                model_config, processor):
            for method_name in inputs_parse_methods:
                print(f"Testing `{method_name}` with modality={modality} "
                      f"and mm_kwargs{list(mm_kwargs.keys())}")
                getattr(model, method_name)(modality=modality, **mm_kwargs)
