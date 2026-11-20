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


def _mm_feature(
    modality: str,
    num_embeds: int,
    identifier: str | None = None,
    data: object | None = "kwargs",
):
    return SimpleNamespace(
        modality=modality,
        identifier=identifier or f"{modality}-{num_embeds}",
        data=data,
        mm_position=SimpleNamespace(get_num_embeds=lambda: num_embeds),
    )


def _run(features, encoder_outputs=None):
    """Runs set_active_mm_loras for one request (LoRA id 7) scheduling every
    feature in `features`, and returns (model, lora_manager)."""
    req_id = "req-0"
    model = _FakeModel()
    lora_manager = _FakeLoRAManager()
    encoder_cache = SimpleNamespace(
        mm_features={req_id: features},
        encoder_outputs=encoder_outputs or {},
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
        scheduled_encoder_inputs={req_id: list(range(len(features)))},
    )
    return model, lora_manager


def test_set_active_mm_loras_skips_prompt_embeds_passthrough():
    """`prompt_embeds` is a passthrough modality: the tensor is already in
    the model's embedding space, no encoder runs, and there is nothing to
    apply tower/connector LoRA to. set_active_mm_loras must skip it instead
    of forwarding it to the model's get_mm_lora_token_counts (which doesn't
    know about it and would raise "Unsupported modality")."""
    model, lora_manager = _run(
        [_mm_feature("prompt_embeds", num_embeds=32), _mm_feature("image", 4)]
    )

    # Only the real "image" item should ever reach the model.
    assert model.calls == ["image"]
    assert len(lora_manager.set_active_adapters_calls) == 1
    _, mapping = lora_manager.set_active_adapters_calls[0]
    assert mapping.index_mapping == (7, 7, 7, 7)


def test_set_active_mm_loras_skips_items_already_encoded():
    """The scheduler can evict an encoder-cache entry and re-allocate it in
    the same step, keeping the worker-side tensor: the item is scheduled but
    EncoderRunner.prepare_mm_inputs doesn't re-encode it. Mapping it anyway
    makes the tower mapping longer than the encoder's output, and the LoRA
    kernels then read and write past their buffers."""
    cached = _mm_feature("image", num_embeds=8, identifier="cached-image")
    fresh = _mm_feature("image", num_embeds=4, identifier="fresh-image")
    model, lora_manager = _run(
        [cached, fresh], encoder_outputs={"cached-image": object()}
    )

    assert model.calls == ["image"]
    assert len(lora_manager.set_active_adapters_calls) == 1
    _, mapping = lora_manager.set_active_adapters_calls[0]
    assert mapping.index_mapping == (7, 7, 7, 7)


def test_set_active_mm_loras_skips_items_without_data():
    """prepare_mm_inputs doesn't encode an item whose data is None, so it must
    not be mapped either."""
    model, lora_manager = _run(
        [_mm_feature("image", num_embeds=8, data=None), _mm_feature("image", 4)]
    )

    assert model.calls == ["image"]
    _, mapping = lora_manager.set_active_adapters_calls[0]
    assert mapping.index_mapping == (7, 7, 7, 7)


def test_set_active_mm_loras_noop_when_every_item_is_cached():
    """When nothing will be encoded this step, no tower mapping is set."""
    item = _mm_feature("image", num_embeds=4, identifier="cached-image")
    model, lora_manager = _run([item], encoder_outputs={"cached-image": object()})

    assert model.calls == []
    assert lora_manager.set_active_adapters_calls == []
