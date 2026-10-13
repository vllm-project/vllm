# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import math
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.multimodal import MultiModalDummyOptions
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.voxtral import (
    AudioLanguageAdapter,
    VoxtralForConditionalGeneration,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import ensure_current_vllm_config
from ..utils import build_model_context


class _StubModel:
    """Carries only the attributes `get_mm_lora_token_counts` reads from
    `self`, so the real method can be exercised without constructing the
    full model (whisper tower, language model, etc.)."""

    get_mm_lora_token_counts = VoxtralForConditionalGeneration.get_mm_lora_token_counts


@pytest.mark.parametrize("model_id", ["mistralai/Voxtral-Mini-3B-2507"])
def test_mm_lora_token_counts_match_processor_output(model_id):
    """The LoRA token counts must match what the whisper tower and adapter
    actually process: the real processor pads each audio independently to
    whole 30-second chunks, so the tower token count cannot be derived by
    scaling the placeholder count of an unpadded audio."""
    ctx = build_model_context(model_id, limit_mm_per_prompt={"audio": 2})
    model_config = ctx.model_config
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    feature_extractor = processor.info.get_feature_extractor()
    sr = feature_extractor.sampling_rate

    mistral_audio_config = feature_extractor.audio_encoder.audio_config
    chunk_samples = int(mistral_audio_config.chunk_length_s * sr)

    audio_config = model_config.hf_config.audio_config
    tokens_per_chunk = math.ceil(
        audio_config.max_source_positions / audio_config.downsample_factor
    )

    stub = _StubModel()
    stub.config = model_config.hf_config
    stub.downsample_factor = audio_config.downsample_factor
    # Mel frames per 30-second chunk, as `VoxtralEncoderModel.chunk_size`
    # derives from its conv strides.
    stub.whisper_encoder = SimpleNamespace(
        chunk_size=chunk_samples // audio_config.hop_length
    )

    # Awkward lengths: not aligned to the 30-second chunk or the token frame;
    # the second audio spans multiple chunks.
    rng = np.random.RandomState(0)
    audios = [
        rng.rand(int(7.3 * sr)).astype(np.float32),
        rng.rand(int(41.7 * sr)).astype(np.float32),
    ]
    mm_data = {"audio": [(audio, sr) for audio in audios]}

    token_prompt = processor.get_dummy_inputs(
        model_config.max_model_len,
        {"audio": len(audios)},
        mm_options=MultiModalDummyOptions(),
        mm_data=mm_data,
    ).prompt
    mm_input = processor(
        token_prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    mm_items = mm_input["mm_kwargs"]["audio"]
    placeholders = mm_input["mm_placeholders"]["audio"]
    assert len(mm_items) == len(placeholders) == len(audios)

    runtime_counts = []
    for mm_item, placeholder, audio in zip(mm_items, placeholders, audios):
        num_embeds = placeholder.get_num_embeds()
        padded_len = len(mm_item.get_data()["audio_arrays"])
        assert padded_len >= len(audio)
        assert padded_len % chunk_samples == 0
        num_chunks = padded_len // chunk_samples

        tower_tokens, connector_tokens = stub.get_mm_lora_token_counts(
            modality="audio", mm_kwargs=mm_item, num_mm_embeds=num_embeds
        )
        # The whisper tower runs every padded chunk at full context; the
        # adapter consumes one row per LM audio embed.
        assert tower_tokens == num_chunks * audio_config.max_source_positions
        assert connector_tokens == num_embeds
        assert num_embeds == num_chunks * tokens_per_chunk

        # An item served from the encoder cache (mm_kwargs=None) must get
        # the same exact counts.
        assert stub.get_mm_lora_token_counts(
            modality="audio", mm_kwargs=None, num_mm_embeds=num_embeds
        ) == (tower_tokens, connector_tokens)

        runtime_counts.append((tower_tokens, connector_tokens))

    # Wrapper sizing (LoRAModelManager passes the encoder budget with
    # mm_kwargs=None) must bound the runtime counts of any batch within
    # that budget.
    budget = sum(placeholder.get_num_embeds() for placeholder in placeholders)
    sizing_tower, sizing_connector = stub.get_mm_lora_token_counts(
        modality="audio", mm_kwargs=None, num_mm_embeds=budget
    )
    assert sizing_tower >= sum(tower for tower, _ in runtime_counts)
    assert sizing_connector >= sum(conn for _, conn in runtime_counts)

    # The real encoder budget is not a multiple of tokens_per_chunk (375
    # here), so both bounds must round up to whole chunks: 16384 embeds
    # span 44 chunks of 1500 tower tokens and 44 * 375 = 16500 connector
    # rows. Scaling the tower by downsample_factor instead would
    # under-allocate (16384 * 4 = 65536 < 66000), and returning the raw
    # budget for the connector would too: encoder profiling batches
    # maximum-size items whose embeds are chunk-padded, so an 8192-token
    # budget profiled with two 4096-embed items runs 2 * 4125 = 8250
    # connector rows.
    assert tokens_per_chunk == 375
    assert stub.get_mm_lora_token_counts(
        modality="audio", mm_kwargs=None, num_mm_embeds=16384
    ) == (66000, 16500)


@pytest.fixture
def dist_init():
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        with ensure_current_vllm_config():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1)
            yield
        cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_file)


def test_adapter_linears_are_lora_wrappable(dist_init) -> None:
    # vllm.lora.utils.from_layer only wraps vLLM-native linear layers, so the
    # connector must use ReplicatedLinear for LoRA adapters targeting
    # audio_language_adapter.{w_in,w_out} to load.
    adapter = AudioLanguageAdapter(hidden_size=8, dim=4)
    assert isinstance(adapter.w_in, ReplicatedLinear)
    assert isinstance(adapter.w_out, ReplicatedLinear)

    # Checkpoint remapping in load_weights relies on these parameter names.
    assert {name for name, _ in adapter.named_parameters()} == {
        "w_in.weight",
        "w_out.weight",
    }

    for layer in (adapter.w_in, adapter.w_out):
        layer.quant_method.process_weights_after_loading(layer)
    out = adapter(torch.randn(3, 8))
    assert out.shape == (3, 4)
