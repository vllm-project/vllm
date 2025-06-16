# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Optional

import torch

from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.sample.logits_processor import (LogitBiasLogitsProcessor,
                                             LogitsProcessor,
                                             MinPLogitsProcessor,
                                             MinTokensLogitsProcessor)

# Logits processor id strs
STR_NO_LOGITPROC = "none"
STR_MIN_P_LOGITPROC_ID = "min_p"
STR_MIN_TOKENS_LOGITPROC_ID = "min_tokens"
STR_LOGITS_BIAS_LOGITPROC_ID = "logit_bias"


def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings,
    expected_num_items: int,
) -> None:
    """
    Perform sanity checks for the result of
    [`vllm.model_executor.models.SupportsMultiModal.get_multimodal_embeddings`][].
    """
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")

    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")

    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")


def scatter_mm_placeholders(
    embeds: torch.Tensor,
    is_embed: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Scatter the multimodal embeddings into a contiguous tensor that represents
    the placeholder tokens.

    [`vllm.multimodal.processing.PromptUpdateDetails.is_embed`][].

    Args:
        embeds: The multimodal embeddings.
          Shape: `(num_embeds, embed_dim)`
        is_embed: A boolean mask indicating which positions in the placeholder
          tokens need to be filled with multimodal embeddings.
          Shape: `(num_placeholders, num_embeds)`
    """
    if is_embed is None:
        return embeds

    placeholders = embeds.new_full(
        (is_embed.shape[0], embeds.shape[-1]),
        fill_value=torch.nan,
    )
    placeholders[is_embed] = embeds
    return placeholders


def gather_mm_placeholders(
    placeholders: torch.Tensor,
    is_embed: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Reconstructs the embeddings from the placeholder tokens.

    This is the operation of [scatter_mm_placeholders][].
    """
    if is_embed is None:
        return placeholders

    return placeholders[is_embed]


def initialize_kv_cache_for_kv_sharing(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list[KVCacheGroupSpec],
    kv_caches: dict[str, torch.Tensor],
) -> None:
    """
    Sets up KV cache sharing by reusing the allocated KV caches in `kv_caches`
    for layers that do not allocate its own KV cache, based on the mapping in
    `shared_kv_cache_layers`. Adds these layers to the corresponding KV cache
    group, which is needed to ensure that attention metadata is assigned later.

    Args:
        shared_kv_cache_layers: Layer pairings for cross-layer KV sharing.
            If an Attention layer `layer_name` is in the keys of this dict, it
            means this layer will perform attention using the keys and values
            from the KV cache of `shared_kv_cache_layers[layer_name]`.
        kv_cache_groups: The KV cache groups of the model.
        kv_caches: The allocated kv_caches with layer names as keys.
            Note that layers in shared_kv_cache_layers.keys() are not
            originally included as it only contains layers which have its own
            KV cache allocation.
    """
    # Record index of KV cache group for each layer that allocates a KV cache.
    layer_to_kv_cache_group_idx: dict[str, int] = {}
    for i, kv_cache_group in enumerate(kv_cache_groups):
        for layer_name in kv_cache_group.layer_names:
            layer_to_kv_cache_group_idx[layer_name] = i

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]
        group_idx = layer_to_kv_cache_group_idx[target_layer_name]
        kv_cache_groups[group_idx].layer_names.append(layer_name)


@dataclass
class LogitsProcessorObjects:
    """Encapsulates initialized logitsproc objects.
    
    Each logits processor has a unique id.
    """
    nongreedy: dict[str, LogitsProcessor] = field(
        default_factory=dict)  # id -> nongreedy-sampling-only logitsproc
    greedy: dict[str, LogitsProcessor] = field(
        default_factory=dict)  # id -> greedy-sampling compatible logitsproc

    def __post_init__(self):
        """Guarantee unique ids"""
        if (self.nongreedy.keys() & self.greedy.keys()):
            raise ValueError("Greedy and non-greedy logits "
                             "processors must not share ids")

    def get_logitsproc_by_id(self, id: str) -> Optional[LogitsProcessor]:
        """Find logits processor by id, if it exists"""
        return self.all.get(id, None)

    @property
    def all(self) -> dict[str, LogitsProcessor]:
        """All logits processors"""
        return self.greedy | self.nongreedy

    @property
    def nongreedy_list(self) -> list[LogitsProcessor]:
        return list(self.nongreedy.values())

    @property
    def greedy_list(self) -> list[LogitsProcessor]:
        return list(self.greedy.values())

    @property
    def all_list(self) -> list[LogitsProcessor]:
        """List of all logits processors"""
        return self.nongreedy_list + self.greedy_list


def init_hard_coded_logitsprocs(
        pin_memory_available: bool, max_num_reqs: int,
        device: torch.device) -> LogitsProcessorObjects:
    min_tokens_logitproc = MinTokensLogitsProcessor(
        pin_memory=pin_memory_available, device=device)
    logit_bias_logitproc = LogitBiasLogitsProcessor(
        pin_memory=pin_memory_available, device=device)
    min_p_logitproc = MinPLogitsProcessor(
        pin_memory=pin_memory_available,
        device=device,
        # +1 for temporary swap space
        max_num_reqs=max_num_reqs + 1)
    return LogitsProcessorObjects(
        greedy={
            STR_MIN_TOKENS_LOGITPROC_ID: min_tokens_logitproc,
            STR_LOGITS_BIAS_LOGITPROC_ID: logit_bias_logitproc
        },
        nongreedy={STR_MIN_P_LOGITPROC_ID: min_p_logitproc},
    )
