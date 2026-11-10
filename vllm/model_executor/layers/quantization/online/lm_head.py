# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online-quantized copies of an already-loaded lm_head."""

from itertools import chain
from typing import Literal
from weakref import WeakKeyDictionary

import torch
from torch.nn import Module

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
    OnlineLinearBase,
)
from vllm.model_executor.layers.quantization.online.nvfp4 import (
    Nvfp4WeightOnlyOnlineLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


_LM_HEAD_METHODS: dict[str, type[OnlineLinearBase]] = {
    "fp8": Fp8PtpcOnlineLinearMethod,
    "nvfp4": Nvfp4WeightOnlyOnlineLinearMethod,
}


def _quantize_copy(lm_head: ParallelLMHead, quantization: str) -> ParallelLMHead:
    if (
        not isinstance(
            lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        )
        or getattr(lm_head, "bias", None) is not None
        or lm_head.weight.dtype not in (torch.float16, torch.bfloat16)
    ):
        raise ValueError(
            "Quantizing an lm_head copy needs an unquantized fp16/bf16 lm_head "
            "without bias."
        )
    # Online methods take their output dtype from the default dtype, which is
    # only the model dtype while the model itself is being built.
    dtype = lm_head.weight.dtype
    with set_default_torch_dtype(dtype):
        head = ParallelLMHead(
            lm_head.num_embeddings,
            lm_head.embedding_dim,
            params_dtype=dtype,
            org_num_embeddings=lm_head.org_vocab_size,
            padding_size=lm_head.padding_size,
            prefix=lm_head.prefix,
            disable_tp=lm_head.disable_tp,
            quant_method=_LM_HEAD_METHODS[quantization](),
        )
    # Never loaded: a new param sharing the source storage, never a copy of it.
    head.weight = torch.nn.Parameter(lm_head.weight.data, requires_grad=False)
    head.quant_method.process_weights_after_loading(head)
    # The copy is never loaded. The online loader wrapper captures the layer,
    # so keeping it would leave each copy to the cycle collector.
    for param in head.parameters(recurse=False):
        if hasattr(param, "weight_loader"):
            del param.weight_loader
    return head


# Each quantized copy maps to the head it is derived from. Weak keys, and no
# module attribute, so the copy never owns the source as a submodule.
_SOURCES: WeakKeyDictionary[Module, tuple[ParallelLMHead, str]] = WeakKeyDictionary()


def quantized_lm_head_copy(
    lm_head: ParallelLMHead, quantization: Literal["fp8", "nvfp4"]
) -> ParallelLMHead:
    """A ParallelLMHead holding this rank's shard of `lm_head`, quantized.

    "fp8" uses per-row weight and per-token activation scales on the platform's
    FP8 linear kernel; "nvfp4" is weight-only on the
    platform's W4A16 kernel. `lm_head` is left
    unchanged and no bf16 copy of its weight is made. The copy is derived state:
    `refresh_quantized_lm_heads` re-derives it after `lm_head` changes.
    """
    head = _quantize_copy(lm_head, quantization)
    # Keep it out of layerwise reload, which would re-run
    # process_weights_after_loading on the already-quantized weight.
    get_layerwise_info(head).reset()
    _SOURCES[head] = (lm_head, quantization)
    logger.info(
        "Quantized a %s copy of the lm_head (%d x %d per rank).",
        quantization,
        head.output_size_per_partition,
        head.input_size_per_partition,
    )
    return head


@torch.no_grad()
def refresh_quantized_lm_heads(model: Module) -> None:
    """Re-derive the quantized lm_head copies in `model` from their source heads.

    Copies in place, so CUDA graphs keep pointing at valid tensors.
    """
    for module in model.modules():
        if module not in _SOURCES:
            continue
        fresh = _quantize_copy(*_SOURCES[module])
        for name, tensor in chain(
            fresh.named_parameters(recurse=False), fresh.named_buffers(recurse=False)
        ):
            getattr(module, name).copy_(tensor)


# (parent, attribute, copy) for each copy swapped out by share_source_lm_heads.
_SWAPPED: dict[Module, list[tuple[Module, str, Module]]] = {}


def share_source_lm_heads(model: Module) -> None:
    """Replace each quantized copy in `model` with its source head.

    `model` then loads weights like a drafter sharing the unquantized head,
    until `restore_quantized_lm_heads` swaps the copies back in.
    """
    swapped = [
        (module, name, child)
        for module in model.modules()
        for name, child in module.named_children()
        if child in _SOURCES
    ]
    for module, name, child in swapped:
        setattr(module, name, _SOURCES[child][0])
    _SWAPPED[model] = swapped


def restore_quantized_lm_heads(model: Module, refresh: bool = True) -> None:
    """Undo `share_source_lm_heads`, then re-derive the copies in `model`.

    Args:
        model: The model passed to `share_source_lm_heads`, if any.
        refresh: Whether to re-derive the copies; False when the source heads
            may be mid-reload.

    """
    for module, name, head in _SWAPPED.pop(model, ()):
        setattr(module, name, head)
    if refresh:
        refresh_quantized_lm_heads(model)
