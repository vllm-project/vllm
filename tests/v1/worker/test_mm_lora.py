# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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


@pytest.mark.parametrize(("active", "budget"), [(True, 150), (True, 100), (False, 150)])
def test_gemma4_audio_lora_padded_stride(active, budget):
    from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    # 97 and 100 frames both project to 25 rows. They can stay batched
    # even with the 50-row clip scheduled between them.
    features = []
    for frames in (97, 200, 100):
        item = _mm_feature("audio", (frames + 3) // 4)
        item.data = {
            "input_features_padded": SimpleNamespace(data=torch.empty(1, frames, 8))
        }
        item.identifier = str(frames)
        features.append(item)

    class Model(_FakeModel):
        get_mm_lora_padded_stride = (
            Gemma4ForConditionalGeneration.get_mm_lora_padded_stride
        )

    encoded = []
    manager = _FakeLoRAManager()
    runner = SimpleNamespace(
        model=Model(),
        lora_manager=manager,
        encoder_cache=SimpleNamespace(
            mm_features={"a": features[:2], "b": features[2:]}, encoder_outputs={}
        ),
        req_states=SimpleNamespace(req_id_to_index={"a": 0, "b": 1}),
        lora_state=SimpleNamespace(
            lora_requests={"a": object(), "b": object()} if active else {},
            lora_ids=np.array([3, 7] if active else [0, 0]),
        ),
        model_state=SimpleNamespace(execute_mm_encoder=encoded.append),
        scheduler_config=SimpleNamespace(
            max_num_encoder_input_tokens=budget, encoder_cache_size=budget
        ),
    )
    scheduled = {"a": [0, 1], "b": [0]}
    last = GPUModelRunner._prepare_mm_lora_batch(runner, scheduled)
    if active and budget == 150:
        assert not encoded
        assert last == scheduled
        assert (
            manager.set_active_adapters_calls[0][1].index_mapping
            == (3,) * 100 + (7,) * 50
        )
    elif active:
        assert encoded == [{"a": [0], "b": [0]}]
        assert last == {"a": [1]}
        mappings = [m.index_mapping for _, m in manager.set_active_adapters_calls]
        assert mappings == [(3,) * 25 + (7,) * 25, (3,) * 50]
    else:
        assert not encoded
        assert last is scheduled
