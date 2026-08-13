# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np

from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.lora.worker_manager import WorkerLoRAManager
from vllm.v1.worker.gpu.lora_utils import LoraState
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


def set_active_mm_loras(
    model: Any,
    lora_manager: WorkerLoRAManager,
    encoder_cache: EncoderCache | None,
    req_id_to_index: dict[str, int],
    lora_state: LoraState,
    scheduled_encoder_inputs: dict[str, list[int]],
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

    # iterate through images
    for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
        req_idx = req_id_to_index.get(req_id)
        if req_idx is None:
            continue

        lora_id = int(lora_state.lora_ids[req_idx])
        mm_features = encoder_cache.mm_features[req_id]

        # iterate through visual tokens
        for mm_input_id in encoder_input_ids:
            pos_info = mm_features[mm_input_id].mm_position
            num_tokens = model.get_num_mm_encoder_tokens(pos_info.get_num_embeds())
            prompt_lora_mapping.append(lora_id)
            token_lora_mapping.extend([lora_id] * num_tokens)
            encoder_token_counts.append(num_tokens)

        if lora_id > 0:
            lora_request = lora_state.lora_requests.get(req_id)
            if lora_request is not None:
                lora_requests.add(lora_request)

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
        or not hasattr(model, "get_num_mm_connector_tokens")
    ):
        return

    connector_token_mapping = np.repeat(
        np.array(prompt_lora_mapping, dtype=np.int32),
        np.array(
            [
                model.get_num_mm_connector_tokens(num_tokens)
                for num_tokens in encoder_token_counts
            ],
            dtype=np.int32,
        ),
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
