# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np

from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.lora.worker_manager import WorkerLoRAManager
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.v1.worker.gpu.lora_utils import LoraState
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


def plan_mm_lora_batches(
    model: Any,
    lora_manager: WorkerLoRAManager,
    encoder_cache: EncoderCache | None,
    lora_state: LoraState,
    scheduled: dict[str, list[int]],
    budget: int,
) -> tuple[list[dict[str, list[int]]], int | None]:
    get_stride = getattr(model, "get_mm_lora_padded_stride", None)
    if (
        get_stride is None
        or encoder_cache is None
        or not lora_manager.supports_tower_connector_lora()
        or not any(req_id in lora_state.lora_requests for req_id in scheduled)
    ):
        return [scheduled], None

    remaining: dict[str, list[int]] = {}
    by_stride: dict[int | None, dict[str, list[int]]] = {}
    for req_id, indices in scheduled.items():
        for idx in indices:
            feature = encoder_cache.mm_features[req_id][idx]
            if (
                feature.data is None
                or feature.identifier in encoder_cache.encoder_outputs
            ):
                continue
            stride = get_stride(feature.modality, feature.data)
            remaining.setdefault(req_id, []).append(idx)
            by_stride.setdefault(stride, {}).setdefault(req_id, []).append(idx)

    # Keep an audio batch intact when the rectangular projections fit in
    # both LoRA buffers. Otherwise equal-stride groups need no extra rows.
    if by_stride and None not in by_stride:
        padded_stride = max(s for s in by_stride if s is not None)
        if padded_stride * sum(map(len, remaining.values())) <= budget:
            return [remaining], padded_stride
    return list(by_stride.values()) or [{}], None


def set_active_mm_loras(
    model: Any,
    lora_manager: WorkerLoRAManager,
    encoder_cache: EncoderCache | None,
    req_id_to_index: dict[str, int],
    lora_state: LoraState,
    scheduled_encoder_inputs: dict[str, list[int]],
    padded_rows_per_item: int | None = None,
) -> None:
    if (
        not scheduled_encoder_inputs
        or encoder_cache is None
        or not lora_manager.supports_tower_connector_lora()
    ):
        return

    prompt_lora_mapping: list[int] = []
    token_lora_mapping: list[int] = []
    lora_requests = set()
    encoder_token_counts: list[int] = []
    connector_token_counts: list[int] = []

    mm_items: list[tuple[int, MultiModalFeatureSpec]] = []
    for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
        req_idx = req_id_to_index.get(req_id)
        if req_idx is None:
            continue

        lora_id = int(lora_state.lora_ids[req_idx])
        mm_features = encoder_cache.mm_features[req_id]
        # "prompt_embeds" is a passthrough modality: the tensor is already in
        # the model's embedding space, so no encoder runs and there is no
        # tower or connector output to apply LoRA to.
        mm_items.extend(
            (lora_id, mm_features[i])
            for i in encoder_input_ids
            if mm_features[i].modality != "prompt_embeds"
        )

        if lora_id > 0:
            lora_request = lora_state.lora_requests.get(req_id)
            if lora_request is not None:
                lora_requests.add(lora_request)

    # Runs items sorted by modality.
    mm_items.sort(key=lambda x: x[1].modality)
    for lora_id, mm_feature in mm_items:
        pos_info = mm_feature.mm_position

        tower_tokens, connector_tokens = model.get_mm_lora_token_counts(
            modality=mm_feature.modality,
            mm_kwargs=mm_feature.data,
            num_mm_embeds=pos_info.get_num_embeds(),
        )

        if padded_rows_per_item is not None:
            tower_tokens = connector_tokens = padded_rows_per_item

        prompt_lora_mapping.append(lora_id)
        token_lora_mapping.extend([lora_id] * tower_tokens)
        encoder_token_counts.append(tower_tokens)
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
