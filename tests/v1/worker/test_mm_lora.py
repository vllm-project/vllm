# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np

from vllm.v1.worker.gpu.mm.lora import set_active_mm_loras


class _FakeModel:
    """Records calls to get_mm_lora_token_counts so the test can assert it
    is never invoked for the "prompt_embeds" passthrough modality."""

    def __init__(self):
        self.calls: list[str] = []

    def get_mm_lora_token_counts(self, *, modality, mm_kwargs, num_mm_embeds):
        del mm_kwargs
        self.calls.append(modality)
        return num_mm_embeds, num_mm_embeds

    def get_mm_mapping(self):
        return SimpleNamespace(connector=None)


class _FakeLoRAManager:
    def __init__(self):
        self.set_active_adapters_calls: list[tuple] = []

    def supports_tower_connector_lora(self) -> bool:
        return True

    def set_active_adapters(self, requests, mapping) -> None:
        self.set_active_adapters_calls.append((requests, mapping))


def _mm_feature(modality: str, num_embeds: int):
    return SimpleNamespace(
        modality=modality,
        data=None,
        mm_position=SimpleNamespace(get_num_embeds=lambda: num_embeds),
    )


def test_set_active_mm_loras_skips_prompt_embeds_passthrough():
    """`prompt_embeds` is a passthrough modality: the tensor is already in
    the model's embedding space, no encoder runs, and there is nothing to
    apply tower/connector LoRA to. set_active_mm_loras must skip it instead
    of forwarding it to the model's get_mm_lora_token_counts (which doesn't
    know about it and would raise "Unsupported modality")."""
    req_id = "req-0"
    model = _FakeModel()
    lora_manager = _FakeLoRAManager()
    encoder_cache = SimpleNamespace(
        mm_features={
            req_id: [
                _mm_feature("prompt_embeds", num_embeds=32),
                _mm_feature("image", num_embeds=4),
            ]
        }
    )
    lora_state = SimpleNamespace(
        lora_ids=np.array([7], dtype=np.int32),
        lora_requests={req_id: object()},
    )

    set_active_mm_loras(
        model=model,
        lora_manager=lora_manager,
        encoder_cache=encoder_cache,
        req_id_to_index={req_id: 0},
        lora_state=lora_state,
        scheduled_encoder_inputs={req_id: [0, 1]},
    )

    # Only the real "image" item should ever reach the model.
    assert model.calls == ["image"]
    assert len(lora_manager.set_active_adapters_calls) == 1
    _, mapping = lora_manager.set_active_adapters_calls[0]
    assert mapping.index_mapping == (7, 7, 7, 7)
