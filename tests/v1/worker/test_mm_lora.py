# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import torch

from vllm.lora.layers import LoRAMappingType
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.mm.lora import activate_mm_loras, set_active_mm_loras


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


class _BatchRecordingModel:
    def __init__(self, lora_manager: _FakeLoRAManager):
        self._lora_manager = lora_manager
        self.recorded_batches: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def get_mm_lora_token_counts(self, *, modality, mm_kwargs, num_mm_embeds):
        del modality, mm_kwargs
        return num_mm_embeds, num_mm_embeds

    def get_mm_mapping(self):
        return SimpleNamespace(connector=["visual.merger"])

    def embed_multimodal(self, **mm_kwargs_batch):
        tower = next(
            m
            for _, m in reversed(self._lora_manager.set_active_adapters_calls)
            if m.type == LoRAMappingType.TOWER
        )
        connector = next(
            m
            for _, m in reversed(self._lora_manager.set_active_adapters_calls)
            if m.type == LoRAMappingType.CONNECTOR
        )
        self.recorded_batches.append((tower.index_mapping, connector.index_mapping))
        batch_len = mm_kwargs_batch["pixel_values"].shape[0]
        return [torch.zeros(1, 4) for _ in range(batch_len)]


def _real_mm_feature(
    modality: str, identifier: str, num_embeds: int, field_name: str, shared_val: int
) -> MultiModalFeatureSpec:
    item = MultiModalKwargsItem(
        {
            "pixel_values": MultiModalFieldElem(
                data=torch.zeros(1, 2),
                field=MultiModalSharedField(batch_size=1),
            ),
            field_name: MultiModalFieldElem(
                data=torch.tensor([shared_val]),
                field=MultiModalSharedField(batch_size=1),
            ),
        }
    )
    return MultiModalFeatureSpec(
        data=item,
        modality=modality,
        identifier=identifier,
        mm_position=PlaceholderRange(offset=0, length=num_embeds),
    )


def test_activate_mm_loras_scopes_mapping_per_encoder_batch():
    """Each embed_multimodal batch receives only its own items' LoRA mapping,
    both across modalities and when one modality splits across batches."""
    encoder_cache = EncoderCache()
    # Trigger 3 batches:
    # 1) "image" item (shared field mismatch splits batch_a -> 2 batches)
    # 2) "image" item
    # 3) "video" item (first in scheduled_encoder_inputs, sorted after image)
    feat_vid = _real_mm_feature("video", "h_vid", 5, "tag", shared_val=1)
    feat_img1 = _real_mm_feature("image", "h_img1", 2, "tag", shared_val=1)
    feat_img2 = _real_mm_feature("image", "h_img2", 3, "tag", shared_val=2)

    encoder_cache.mm_features["req-vid"] = [feat_vid]
    encoder_cache.mm_features["req-img1"] = [feat_img1]
    encoder_cache.mm_features["req-img2"] = [feat_img2]

    lora_manager = _FakeLoRAManager()
    model = _BatchRecordingModel(lora_manager)
    runner = EncoderRunner(
        model=model,
        max_num_tokens=16,
        hidden_size=4,
        encoder_cache=encoder_cache,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    req_v, req_1, req_2 = object(), object(), object()
    lora_state = SimpleNamespace(
        lora_ids=np.array([2, 1, 3], dtype=np.int32),
        lora_requests={"req-vid": req_v, "req-img1": req_1, "req-img2": req_2},
    )
    scheduled = {"req-vid": [0], "req-img1": [0], "req-img2": [0]}

    with activate_mm_loras(
        model=model,
        lora_manager=lora_manager,
        encoder_runner=runner,
        req_id_to_index={"req-vid": 0, "req-img1": 1, "req-img2": 2},
        lora_state=lora_state,
        scheduled_encoder_inputs=scheduled,
    ):
        _, mm_kwargs = runner.prepare_mm_inputs(scheduled)
        outputs = runner.execute_mm_encoder(mm_kwargs)

    assert len(outputs) == 3
    assert model.recorded_batches == [
        ((1, 1), (1, 1)),
        ((3, 3, 3), (3, 3, 3)),
        ((2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
    ]
    # Callback should be cleared once the context exits.
    assert runner._mm_lora_batch_callback is None
