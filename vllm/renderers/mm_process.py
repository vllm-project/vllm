# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker entry points for CPU-only multimodal preprocessing."""

from multiprocessing.synchronize import Barrier
from typing import TYPE_CHECKING

import torch
from torch.utils._pytree import tree_leaves

from vllm.config.multimodal import MultiModalConfig
from vllm.inputs import MultiModalInput
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    ProcessorInputs,
    TimingContext,
)
from vllm.plugins import load_general_plugins
from vllm.tokenizers import TokenizerLike
from vllm.utils.system_utils import set_process_title
from vllm.utils.torch_utils import set_default_torch_num_threads

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_processor: BaseMultiModalProcessor | None = None
_warmup_barrier: Barrier | None = None


def initialize_mm_process(
    config: "VllmConfig", tokenizer: TokenizerLike | None, warmup_barrier: Barrier
) -> None:
    global _processor, _warmup_barrier

    set_process_title("MMProcessor")
    load_general_plugins()
    with set_default_torch_num_threads():
        _processor = MULTIMODAL_REGISTRY.create_processor(
            config.model_config, tokenizer=tokenizer
        )

    with set_default_torch_num_threads(1):
        mm_counts = {
            modality: 1
            for modality, limit in _processor.info.allowed_mm_limits.items()
            if limit > 0
        }
        _processor.get_dummy_mm_inputs(
            mm_counts, cache=None, scheduler_config=config.scheduler_config
        )
    _warmup_barrier = warmup_barrier


def ensure_mm_process_ready(*, synchronize: bool = True) -> None:
    assert _processor is not None, "Multimodal worker has not been initialized"
    assert _warmup_barrier is not None
    if synchronize:
        _warmup_barrier.wait()


def validate_mm_process_inputs(inputs: ProcessorInputs) -> None:
    if inputs.cache is not None:
        raise ValueError("Multimodal process workers require the processor cache off.")
    MultiModalConfig.validate_cpu_mm_processor_kwargs(inputs.hf_processor_mm_kwargs)
    for items in inputs.mm_data_items.values():
        for leaf in tree_leaves(items.get_all()):
            if isinstance(leaf, torch.Tensor) and leaf.device.type != "cpu":
                raise ValueError(
                    "Multimodal process workers require CPU input tensors; "
                    f"received a tensor on {leaf.device}."
                )


def apply_mm_processor(
    inputs: ProcessorInputs, timing: TimingContext
) -> tuple[MultiModalInput, TimingContext]:
    assert _processor is not None

    with set_default_torch_num_threads():
        result = _processor.apply(inputs, timing)
    for items in result["mm_kwargs"].values():
        for item in items:
            if item is not None:
                for elem in item.values():
                    if any(
                        isinstance(leaf, torch.Tensor) and leaf.device.type != "cpu"
                        for leaf in tree_leaves(elem.data)
                    ):
                        raise ValueError(
                            "Multimodal process workers require CPU output tensors."
                        )
    return result, timing
