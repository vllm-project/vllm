# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import ModelConfig, SchedulerConfig
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import extract_layer_index
from vllm.multimodal.registry import MultiModalRegistry
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.kv_cache_interface import KVCacheGroupSpec

if TYPE_CHECKING:
    from vllm.attention.layer import Attention


class MultiModalBudget:
    """Helper class to calculate budget information for multi-modal models."""

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
        mm_registry: MultiModalRegistry,
        *,
        max_model_len: int,
        max_num_reqs: int,
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.mm_registry = mm_registry

        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
            mm_registry=mm_registry,
        )

        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size
        self.max_model_len = max_model_len
        self.max_num_reqs = max_num_reqs

        self.mm_limits = mm_registry.get_mm_limits_per_prompt(model_config)

        max_items_per_prompt_by_modality = dict[str, int]()
        max_items_per_batch_by_modality = dict[str, int]()

        max_tokens_by_modality = mm_registry \
            .get_max_tokens_per_item_by_nonzero_modality(model_config)

        for modality, max_tokens in max_tokens_by_modality.items():
            (
                max_items_per_prompt,
                max_items_per_batch,
            ) = self.get_max_items(modality, max_tokens)

            max_items_per_prompt_by_modality[modality] = max_items_per_prompt
            max_items_per_batch_by_modality[modality] = max_items_per_batch

        self.max_tokens_by_modality = max_tokens_by_modality
        self.max_items_per_prompt_by_modality = max_items_per_prompt_by_modality
        self.max_items_per_batch_by_modality = max_items_per_batch_by_modality

    def get_modality_with_max_tokens(self) -> tuple[str, int]:
        max_tokens_by_modality = self.max_tokens_by_modality
        modality, max_tokens = max(max_tokens_by_modality.items(),
                                   key=lambda item: item[1])

        return modality, max_tokens

    def get_encoder_budget(self) -> int:
        return min(self.max_num_encoder_input_tokens, self.encoder_cache_size)

    def get_max_items(
        self,
        modality: str,
        max_tokens_per_item: int,
    ) -> tuple[int, int]:
        if max_tokens_per_item == 0:
            return 0, 0

        # Check how many items of this modality can be supported by
        # the encoder budget.
        encoder_budget = self.get_encoder_budget()

        # TODO: handle encoder-decoder models once we support them.
        if encoder_budget == 0:
            return 0, 0

        max_encoder_items_per_batch = encoder_budget // max_tokens_per_item

        # Check how many items of this modality can be supported by
        # the decoder budget.
        mm_limit = self.mm_limits[modality]

        max_items_per_prompt = max(
            1,
            min(mm_limit, self.max_model_len // max_tokens_per_item),
        )

        scheduler_config = self.scheduler_config
        max_num_reqs = self.max_num_reqs

        if not scheduler_config.enable_chunked_prefill:
            max_num_reqs = min(
                max_num_reqs,
                scheduler_config.max_num_batched_tokens // max_tokens_per_item,
            )

        max_decoder_items_per_batch = max_num_reqs * max_items_per_prompt

        max_items_per_batch = max(
            1,
            min(max_encoder_items_per_batch, max_decoder_items_per_batch),
        )

        return max_items_per_prompt, max_items_per_batch


@dataclass
class AttentionGroup:
    backend: type[AttentionBackend]
    metadata_builder: AttentionMetadataBuilder
    layer_names: list[str]


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
    # Optional for now to avoid breaking TPU
    attn_groups: Optional[list[list[AttentionGroup]]] = None,
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

        if attn_groups is not None:
            assert len(attn_groups[group_idx]) == 1, (
                "Only one attention group per KV cache group is supported "
                "for KV-cache sharing for now.")
            # TODO(lucas): I think in the future the layers that re-use a
            # KV cache will be in a different attention group so we can
            # remove this code from here.
            attn_groups[group_idx][0].layer_names.append(layer_name)


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, "Attention"],
    runner_kv_caches: list[torch.Tensor],
) -> None:
    """
    Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        forward_context: The global forward context containing all Attention
        layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
    """
    # Bind kv_caches to ModelRunner
    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.
            raise NotImplementedError
        layer_name = layer_names[0]
        runner_kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, kv_cache in kv_caches.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].kv_cache = [kv_cache]
