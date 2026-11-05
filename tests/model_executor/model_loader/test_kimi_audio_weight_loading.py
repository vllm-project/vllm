# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-Audio's Whisper tower must load from a checkpoint without k_proj.bias.

Whisper has no k_proj bias, but the tower fuses q/k/v into one biased
`qkv_proj`, so the k slice of that bias never comes from the checkpoint.
"""

import pytest
import torch
from safetensors.torch import save_file
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoder as HFWhisperEncoder,
)

from vllm.config import LoadConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.kimi_audio import (
    KIMIA_WHISPER_SUBFOLDER,
    KimiAudioForConditionalGeneration,
    KimiAudioWhisperEncoder,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.usefixtures("dist_init")]

D_MODEL = 64


def test_whisper_tower_loads_without_k_proj_bias(tmp_path):
    config = WhisperConfig(
        d_model=D_MODEL,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_layers=1,
        decoder_attention_heads=4,
        decoder_ffn_dim=128,
        num_mel_bins=8,
        max_source_positions=16,
        max_target_positions=16,
    )
    config.save_pretrained(tmp_path)
    config.save_pretrained(tmp_path / KIMIA_WHISPER_SUBFOLDER)
    model_config = ModelConfig(
        model=str(tmp_path), dtype="float32", skip_tokenizer_init=True
    )
    vllm_config = VllmConfig(model_config=model_config)

    torch.manual_seed(0)
    hf_state = HFWhisperEncoder(config).state_dict()
    assert not any(name.endswith("k_proj.bias") for name in hf_state)
    # Store the names the tower receives from the model's own mapper, as a full
    # load of the `whisper-large-v3` subfolder routes them.
    mapper = KimiAudioForConditionalGeneration.hf_to_vllm_mapper
    save_file(
        {
            name.removeprefix("audio_tower."): tensor.contiguous()
            for name, tensor in mapper.apply(
                (f"model.encoder.{name}", tensor) for name, tensor in hf_state.items()
            )
        },
        tmp_path / "model.safetensors",
    )

    with set_current_vllm_config(vllm_config):
        tower = KimiAudioWhisperEncoder(vllm_config=vllm_config)
    DefaultModelLoader(LoadConfig()).load_weights(tower, model_config)

    q_bias, k_bias, v_bias = tower.layers[0].self_attn.qkv_proj.bias.split(D_MODEL)
    torch.testing.assert_close(q_bias, hf_state["layers.0.self_attn.q_proj.bias"])
    torch.testing.assert_close(k_bias, torch.zeros_like(k_bias))
    torch.testing.assert_close(v_bias, hf_state["layers.0.self_attn.v_proj.bias"])
