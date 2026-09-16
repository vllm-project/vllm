# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import WorkerLoRAManager
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.v1.worker.gpu.lora_utils import LoraState
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


@dataclass(frozen=True)
class MMEncoderLoraInput:
    """LoRA and shape metadata for one scheduled multimodal item."""

    lora_id: int
    lora_request: LoRARequest | None
    modality: str
    mm_kwargs: MultiModalKwargsItem
    num_mm_embeds: int


@dataclass(frozen=True)
class _MMEncoderLoraMappingItem:
    lora_id: int
    lora_request: LoRARequest | None
    tower_tokens: int | Mapping[str, int]
    connector_tokens: int | Mapping[str, int] | None


class MMEncoderLoraActivation:
    """Build and activate tower/connector mappings for encoder items."""

    def __init__(
        self,
        lora_manager: WorkerLoRAManager,
        items: list[_MMEncoderLoraMappingItem],
        *,
        has_connector: bool,
        requires_per_item: bool,
    ) -> None:
        self._lora_manager = lora_manager
        self._items = items
        self._has_connector = has_connector
        self.requires_per_item = requires_per_item
        # Per-item mapping changes must not evict adapters needed by later
        # encoder items in the same scheduler step.
        self._lora_requests = {
            item.lora_request
            for item in items
            if item.lora_id > 0 and item.lora_request is not None
        }

    @property
    def num_items(self) -> int:
        return len(self._items)

    def activate(self, item_indices: tuple[int, ...] | None = None) -> None:
        """Activate all items, or only the selected encoder item indices."""

        items = (
            self._items
            if item_indices is None
            else [self._items[item_idx] for item_idx in item_indices]
        )
        if not items:
            return

        self._activate_component(items, "tower_tokens", LoRAMappingType.TOWER)

        if not self._has_connector:
            return

        self._activate_component(items, "connector_tokens", LoRAMappingType.CONNECTOR)

    def _activate_component(
        self,
        items: list[_MMEncoderLoraMappingItem],
        count_attr: str,
        mapping_type: LoRAMappingType,
    ) -> None:
        counts = [getattr(item, count_attr) for item in items]
        if any(count is None for count in counts):
            if not all(count is None for count in counts):
                raise ValueError(
                    "MM LoRA token counts must either be present for every "
                    "selected item or absent for all of them"
                )
            return

        prompt_mapping = tuple(item.lora_id for item in items)
        if all(isinstance(count, int) for count in counts):
            mappings: list[tuple[str | None, list[int]]] = [
                (None, [int(count) for count in counts])
            ]
        elif all(isinstance(count, Mapping) for count in counts):
            prefix_sets = [set(count) for count in counts]
            if any(prefixes != prefix_sets[0] for prefixes in prefix_sets[1:]):
                raise ValueError(
                    "Per-module MM LoRA token counts must use the same prefixes "
                    "for every selected item"
                )
            mappings = [
                (prefix, [int(count[prefix]) for count in counts])
                for prefix in counts[0]
            ]
        else:
            raise TypeError(
                "MM LoRA token counts must be all integers or all prefix mappings"
            )

        for target_prefix, token_counts in mappings:
            if any(count < 0 for count in token_counts):
                raise ValueError("MM LoRA token counts must be non-negative")
            token_mapping = tuple(
                item.lora_id
                for item, token_count in zip(items, token_counts)
                for _ in range(token_count)
            )
            self._lora_manager.set_active_adapters(
                self._lora_requests,
                LoRAMapping(
                    index_mapping=token_mapping,
                    prompt_mapping=prompt_mapping,
                    is_prefill=True,
                    type=mapping_type,
                    target_prefix=target_prefix,
                ),
            )


def prepare_mm_lora_activation(
    model: Any,
    lora_manager: WorkerLoRAManager,
    inputs: list[MMEncoderLoraInput],
) -> MMEncoderLoraActivation | None:
    """Create an activation and eagerly apply it unless per-item is required."""

    if not inputs:
        return None

    items = []
    for item in inputs:
        tower_tokens, connector_tokens = model.get_mm_lora_token_counts(
            modality=item.modality,
            mm_kwargs=item.mm_kwargs,
            num_mm_embeds=item.num_mm_embeds,
        )
        items.append(
            _MMEncoderLoraMappingItem(
                lora_id=item.lora_id,
                lora_request=item.lora_request,
                tower_tokens=tower_tokens,
                connector_tokens=connector_tokens,
            )
        )

    mm_mapping = model.get_mm_mapping() if hasattr(model, "get_mm_mapping") else None
    activation = MMEncoderLoraActivation(
        lora_manager,
        items,
        has_connector=bool(mm_mapping is not None and mm_mapping.connector),
        requires_per_item=(
            getattr(model, "requires_mm_lora_per_item_mapping", False) is True
        ),
    )
    if not activation.requires_per_item:
        activation.activate()
    return activation


def set_active_mm_loras(
    model: Any,
    lora_manager: WorkerLoRAManager,
    encoder_cache: EncoderCache | None,
    req_id_to_index: dict[str, int],
    lora_state: LoraState,
    scheduled_encoder_inputs: dict[str, list[int]],
) -> MMEncoderLoraActivation | None:
    if (
        not scheduled_encoder_inputs
        or encoder_cache is None
        or not lora_manager.supports_tower_connector_lora()
    ):
        return None

    inputs: list[MMEncoderLoraInput] = []

    # Keep the same item order and filtering as EncoderRunner.prepare_mm_inputs.
    for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
        req_idx = req_id_to_index.get(req_id)
        if req_idx is None:
            continue

        mm_features = encoder_cache.mm_features[req_id]

        for mm_input_id in encoder_input_ids:
            mm_feature = mm_features[mm_input_id]
            if (
                mm_feature.data is None
                or mm_feature.modality == "prompt_embeds"
                or mm_feature.identifier in encoder_cache.encoder_outputs
            ):
                continue

            lora_id = int(lora_state.lora_ids[req_idx])
            inputs.append(
                MMEncoderLoraInput(
                    lora_id=lora_id,
                    lora_request=lora_state.lora_requests.get(req_id),
                    modality=mm_feature.modality,
                    mm_kwargs=mm_feature.data,
                    num_mm_embeds=mm_feature.mm_position.get_num_embeds(),
                )
            )

    return prepare_mm_lora_activation(model, lora_manager, inputs)
