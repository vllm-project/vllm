# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from functools import partial
from typing import Any, TypeAlias

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.config.multimodal import (
    AudioDummyOptions,
    BaseDummyOptions,
    ImageDummyOptions,
    VideoDummyOptions,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    supports_multimodal,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, BatchedTensorInputs
from vllm.multimodal.processing import BaseMultiModalProcessor, InputProcessingContext
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.platforms import current_platform
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.collection_utils import is_list_of
from vllm.utils.torch_utils import set_default_torch_dtype

from ....utils import create_new_process_for_each_test
from ...registry import HF_EXAMPLE_MODELS
from ...utils import dummy_hf_overrides
from .test_common import get_model_ids_to_test, get_text_token_prompts

ImageInput = list[Image.Image]
VideoInput: TypeAlias = (
    list[Image.Image] | list[np.ndarray] | list[tuple[np.ndarray, dict[str, Any]]]
)
AudioInput = list[tuple[np.ndarray, int]]


def _resize_data(
    _data: Image.Image | np.ndarray, size_factor: float
) -> Image.Image | np.ndarray:
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
        T, W, H = map(lambda x: max(int(x * size_factor), 2), (T, W, H))
        return [d.resize((W, H)) for d in _data[:T]]
    # Video input with numpy arrays
    elif isinstance(_data, np.ndarray) and _data.ndim >= 4:
        T, H, W, C = _data.shape[-4:]
        T, H, W = map(lambda x: max(int(x * size_factor), 2), (T, H, W))
        return _data[..., :T, :H, :W, :C]
    # Audio input
    elif isinstance(_data, np.ndarray) and _data.ndim == 1:
        return _data[: int(len(_data) * size_factor)]
    raise AssertionError("This line should be unreachable.")


def resize_mm_data(
    data: ImageInput | VideoInput | AudioInput, size_factors: tuple[float, ...]
) -> ImageInput | VideoInput | AudioInput:
    size_factors = size_factors[: len(data)]
    if is_list_of(data, (Image.Image, np.ndarray, list)):
        return [_resize_data(d, s) for d, s in zip(data, size_factors)]
    elif is_list_of(data, tuple):
        return [_resize_data(d, s) for (d, _), s in zip(data, size_factors)]
    raise ValueError("Unsupported multimodal data type.")


def create_batched_mm_kwargs(
    model_cls: type[SupportsMultiModal],
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

    # video metadata will be added back to the resized video data here.
    text_prompt, token_prompt = get_text_token_prompts(processor, resized_mm_data)

    mm_kwargs = processor.apply(
        prompt=token_prompt if text_prompt is None else text_prompt,
        mm_data=resized_mm_data,
        hf_processor_mm_kwargs=processor_inputs.hf_processor_mm_kwargs,
        tokenization_kwargs=processor_inputs.tokenization_kwargs,
    )["mm_kwargs"].require_data()

    return group_mm_kwargs_by_modality(
        [
            (item, modality)
            for modality in supported_mm_limits
            for item in mm_kwargs[modality]
        ]
    )


# TODO(Isotr0py): Don't initialize model during test
@contextmanager
def initialize_dummy_model(
    model_cls: type[nn.Module],
    model_config: ModelConfig,
):
    temp_file = tempfile.mkstemp()[1]
    current_device = torch.get_default_device()
    vllm_config = VllmConfig(model_config=model_config)
    with set_current_vllm_config(vllm_config=vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)

        with set_default_torch_dtype(model_config.dtype):
            torch.set_default_device(current_platform.device_type)
            model = model_cls(vllm_config=vllm_config)
            torch.set_default_device(current_device)
        yield model

    del model
    cleanup_dist_env_and_memory()


@create_new_process_for_each_test()
@pytest.mark.parametrize("model_id", get_model_ids_to_test())
def test_model_tensor_schema(model_id: str):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(
        on_fail="skip",
        check_max_version=False,
        check_version_reason="vllm",
    )

    model_arch = next(
        arch for arch, info in HF_EXAMPLE_MODELS.hf_models.items() if info == model_info
    )

    hf_overrides_fn = partial(
        dummy_hf_overrides,
        model_arch=model_arch,
        exist_overrides=model_info.hf_overrides,
    )

    # ROCm: Detect if model uses AWQ quantization and set appropriate dtype
    if "awq" in model_id.lower() and current_platform.is_rocm():
        dtype = "float16"
    else:
        dtype = model_info.dtype

    model_config = ModelConfig(
        model_id,
        tokenizer=model_info.tokenizer or model_id,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=hf_overrides_fn,
        skip_tokenizer_init=model_info.require_embed_inputs,
        enable_prompt_embeds=model_info.require_embed_inputs,
        enable_mm_embeds=model_info.require_embed_inputs,
        enforce_eager=model_info.enforce_eager,
        dtype=dtype,
    )

    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)
    assert supports_multimodal(model_cls)

    factories = model_cls._processor_factory

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

    def _to_dummy_options(modality: str, count: int) -> BaseDummyOptions:
        if modality == "video":
            return VideoDummyOptions(count=count)
        if modality == "image":
            return ImageDummyOptions(count=count)
        if modality == "audio":
            return AudioDummyOptions(count=count)
        return BaseDummyOptions(count=count)

    model_config.get_multimodal_config().limit_per_prompt = {
        modality: _to_dummy_options(modality, count)
        for modality, count in limit_mm_per_prompt.items()
    }
    processor = factories.build_processor(ctx, cache=None)

    with initialize_dummy_model(model_cls, model_config) as model:
        for modality, _, mm_kwargs in create_batched_mm_kwargs(
            model_cls, model_config, processor
        ):
            for method_name in inputs_parse_methods:
                print(
                    f"Testing `{method_name}` with modality={modality} "
                    f"and mm_kwargs{list(mm_kwargs.keys())}"
                )
                getattr(model, method_name)(modality=modality, **mm_kwargs)
