# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.lora.worker_manager import WorkerLoRAManager
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem
from vllm.v1.worker.gpu.lora_utils import LoraState
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner


def _collect_mm_items(
    lora_manager: WorkerLoRAManager,
    encoder_cache: EncoderCache | None,
    req_id_to_index: dict[str, int],
    lora_state: LoraState,
    scheduled_encoder_inputs: dict[str, list[int]],
) -> list[tuple[int, Any, MultiModalFeatureSpec]]:
    if (
        not scheduled_encoder_inputs
        or encoder_cache is None
        or not lora_manager.supports_tower_connector_lora()
    ):
        return []

    mm_items: list[tuple[int, Any, MultiModalFeatureSpec]] = []
    for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
        req_idx = req_id_to_index.get(req_id)
        if req_idx is None:
            continue

        lora_id = int(lora_state.lora_ids[req_idx])
        lora_request = lora_state.lora_requests.get(req_id) if lora_id > 0 else None
        mm_features = encoder_cache.mm_features[req_id]
        # "prompt_embeds" is a passthrough modality: the tensor is already in
        # the model's embedding space, so no encoder runs and there is no
        # tower or connector output to apply LoRA to.
        mm_items.extend(
            (lora_id, lora_request, mm_features[i])
            for i in encoder_input_ids
            if mm_features[i].modality != "prompt_embeds"
        )
    return mm_items


def _apply_mm_loras_for_items(
    model: Any,
    lora_manager: WorkerLoRAManager,
    mm_items: list[tuple[int, Any, MultiModalFeatureSpec]],
) -> None:
    if not mm_items:
        return

    prompt_lora_mapping: list[int] = []
    token_lora_mapping: list[int] = []
    connector_token_counts: list[int | None] = []
    lora_requests = set()

    for lora_id, lora_request, mm_feature in mm_items:
        if lora_id > 0 and lora_request is not None:
            lora_requests.add(lora_request)

        pos_info = mm_feature.mm_position
        tower_tokens, connector_tokens = model.get_mm_lora_token_counts(
            modality=mm_feature.modality,
            mm_kwargs=mm_feature.data,
            num_mm_embeds=pos_info.get_num_embeds(),
        )

        prompt_lora_mapping.append(lora_id)
        token_lora_mapping.extend([lora_id] * tower_tokens)
        connector_token_counts.append(connector_tokens)

    if not prompt_lora_mapping:
        return

    lora_manager.set_active_adapters(
        lora_requests,
        LoRAMapping(
            tuple(token_lora_mapping),
            tuple(prompt_lora_mapping),
            is_prefill=True,
            type=LoRAMappingType.TOWER,
        ),
    )

    mm_mapping = model.get_mm_mapping() if hasattr(model, "get_mm_mapping") else None
    if (
        mm_mapping is None
        or not mm_mapping.connector
        or not all(count is not None for count in connector_token_counts)
    ):
        return

    connector_token_mapping = np.repeat(
        np.array(prompt_lora_mapping, dtype=np.int32),
        np.array(connector_token_counts, dtype=np.int32),
    )
    lora_manager.set_active_adapters(
        lora_requests,
        LoRAMapping(
            index_mapping=tuple(connector_token_mapping.tolist()),
            prompt_mapping=tuple(prompt_lora_mapping),
            is_prefill=True,
            type=LoRAMappingType.CONNECTOR,
        ),
    )


def set_active_mm_loras(
    model: Any,
    lora_manager: WorkerLoRAManager,
    encoder_cache: EncoderCache | None,
    req_id_to_index: dict[str, int],
    lora_state: LoraState,
    scheduled_encoder_inputs: dict[str, list[int]],
) -> None:
    mm_items = _collect_mm_items(
        lora_manager,
        encoder_cache,
        req_id_to_index,
        lora_state,
        scheduled_encoder_inputs,
    )
    if not mm_items:
        return

    # Runs items sorted by modality.
    mm_items.sort(key=lambda x: x[2].modality)
    _apply_mm_loras_for_items(model, lora_manager, mm_items)


@contextmanager
def activate_mm_loras(
    model: Any,
    lora_manager: WorkerLoRAManager,
    encoder_runner: "EncoderRunner",
    req_id_to_index: dict[str, int],
    lora_state: LoraState,
    scheduled_encoder_inputs: dict[str, list[int]],
):
    """Apply tower/connector LoRA mappings per encoder batch.

    When one step encodes multiple modality groups or split batches, each
    `embed_multimodal` call reads LoRA mappings from index 0, so mappings must
    be scoped to the items in that batch rather than built once for the step.
    """
    mm_items = _collect_mm_items(
        lora_manager,
        encoder_runner.encoder_cache,
        req_id_to_index,
        lora_state,
        scheduled_encoder_inputs,
    )
    if not mm_items:
        yield
        return

    items_by_data_id: dict[int, deque[tuple[int, Any, MultiModalFeatureSpec]]] = (
        defaultdict(deque)
    )
    for item in mm_items:
        items_by_data_id[id(item[2].data)].append(item)

    def _on_encoder_batch(
        batch_items: list[tuple[str, MultiModalKwargsItem]],
    ) -> None:
        batch_mm_items: list[tuple[int, Any, MultiModalFeatureSpec]] = []
        for _, mm_data in batch_items:
            queue = items_by_data_id.get(id(mm_data))
            if not queue:
                continue
            entry = queue.popleft() if len(queue) > 1 else queue[0]
            batch_mm_items.append(entry)
        _apply_mm_loras_for_items(model, lora_manager, batch_mm_items)

    callback: Callable[[list[tuple[str, MultiModalKwargsItem]]], None] = (
        _on_encoder_batch
    )
    with encoder_runner.mm_lora_batch_callback(callback):
        yield
