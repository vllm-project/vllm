# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest

from vllm.model_executor.models.midashenglm import (
    _MAX_AUDIO_LENGTH,
    MiDashengLMModel,
    calculate_mel_frames_dasheng,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from ..utils import build_model_context


class _StubModel:
    """Carries only the attributes `get_mm_lora_token_counts` reads from
    `self`, so the real method can be exercised without constructing the
    full model (Dasheng tower, language model, etc.)."""

    get_mm_lora_token_counts = MiDashengLMModel.get_mm_lora_token_counts
    _mm_lora_counts_per_item = MiDashengLMModel._mm_lora_counts_per_item


@pytest.mark.parametrize("model_id", ["mispeech/midashenglm-7b"])
def test_mm_lora_token_counts_match_padded_tower_input(model_id):
    """The LoRA token counts must match what the Dasheng tower and projector
    actually process: with `pad_audio_to_max_length` (enforced when
    tower/connector LoRA is enabled) every audio is padded to
    `_MAX_AUDIO_LENGTH` samples, so the per-item counts are constants that
    cannot be derived by scaling an unpadded audio's placeholder count —
    in particular for lengths whose patch count is not a multiple of
    `subsample_factor`, where the projector's floor discards the remainder."""
    ctx = build_model_context(model_id, limit_mm_per_prompt={"audio": 3})
    model_config = ctx.model_config
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    feature_extractor = processor.info.get_feature_extractor()
    sr = feature_extractor.sampling_rate

    stub = _StubModel()
    stub.config = model_config.hf_config

    # The tower runs on the mel patch count of the padded input; the
    # projector folds those patches by `subsample_factor` before its linears.
    # `calculate_mel_frames_dasheng` is the same mel/patch math the processor
    # uses for placeholders.
    tower_per_item = calculate_mel_frames_dasheng(
        _MAX_AUDIO_LENGTH, model_subsampling=1
    )
    connector_per_item = tower_per_item // model_config.hf_config.subsample_factor
    assert (tower_per_item, connector_per_item) == (250, 50)

    # Awkward lengths: the first two have patch counts not divisible by
    # `subsample_factor`; the last fills `_MAX_AUDIO_LENGTH` exactly.
    rng = np.random.RandomState(0)
    durations_s = [0.37, 3.3, 10.0]
    audios = [rng.rand(int(d * sr)).astype(np.float32) for d in durations_s]
    assert any(
        calculate_mel_frames_dasheng(len(audio), model_subsampling=1) % 5 != 0
        for audio in audios
    )
    mm_data = {"audio": [(audio, sr) for audio in audios]}

    prompt = processor.dummy_inputs.get_dummy_text({"audio": len(audios)})
    mm_input = processor(prompt, mm_items=processor.info.parse_mm_data(mm_data))

    mm_items = mm_input["mm_kwargs"]["audio"]
    placeholders = mm_input["mm_placeholders"]["audio"]
    assert len(mm_items) == len(placeholders) == len(audios)

    runtime_counts = []
    for mm_item, placeholder, audio in zip(mm_items, placeholders, audios):
        num_embeds = placeholder.get_num_embeds()
        assert num_embeds == calculate_mel_frames_dasheng(len(audio))

        # Every valid audio fits within the fixed padding and trims at most
        # the projector's row count.
        stored_len = len(mm_item.get_data()["input_values"])
        assert len(audio) <= stored_len <= _MAX_AUDIO_LENGTH
        assert 1 <= num_embeds <= connector_per_item

        counts = stub.get_mm_lora_token_counts(
            modality="audio", mm_kwargs=mm_item, num_mm_embeds=num_embeds
        )
        assert counts == (tower_per_item, connector_per_item)
        runtime_counts.append(counts)

    # A full-length audio fills the projector's rows exactly, so shorter
    # items only ever trim rows.
    assert placeholders[-1].get_num_embeds() == connector_per_item

    # Wrapper sizing (LoRAModelManager passes the encoder budget with
    # mm_kwargs=None) must bound the runtime counts of any batch within that
    # budget. Placeholders are not padded per item, so the budget bounds the
    # item count only through one-embed items.
    budget = sum(placeholder.get_num_embeds() for placeholder in placeholders)
    sizing = stub.get_mm_lora_token_counts(
        modality="audio", mm_kwargs=None, num_mm_embeds=budget
    )
    assert sizing == (budget * tower_per_item, budget * connector_per_item)
    assert sizing[0] >= sum(tower for tower, _ in runtime_counts)
    assert sizing[1] >= sum(conn for _, conn in runtime_counts)
