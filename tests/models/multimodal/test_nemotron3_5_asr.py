# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import Nemotron3_5AsrConfig, NemotronAsrStreamingEncoderConfig

from vllm.model_executor.models.nemotron3_5_asr import (
    Nemotron3_5AsrAudioEncoder,
)


def _get_tiny_config() -> Nemotron3_5AsrConfig:
    encoder_config = NemotronAsrStreamingEncoderConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        attention_bias=False,
        convolution_bias=False,
        conv_kernel_size=3,
        subsampling_factor=8,
        subsampling_conv_channels=4,
        num_mel_bins=8,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=32,
        scale_input=False,
        sliding_window=9,
        default_num_lookahead_tokens=3,
    )
    return Nemotron3_5AsrConfig(
        encoder_config=encoder_config,
        decoder_hidden_size=8,
        num_prompts=8,
        prompt_intermediate_size=16,
        default_prompt_id=3,
    )


@pytest.mark.parametrize(
    ("num_mel_frames", "physical_frames", "valid_frames"),
    [(26, 5, 4), (128, 17, 17)],
)
def test_nemotron_audio_encoder_preserves_batch_and_valid_lengths(
    num_mel_frames: int,
    physical_frames: int,
    valid_frames: int,
) -> None:
    torch.manual_seed(0)
    config = _get_tiny_config()
    config.encoder_config.num_hidden_layers = 2
    model = Nemotron3_5AsrAudioEncoder(config).eval()
    input_features = torch.randn(2, num_mel_frames, 8)
    attention_mask = torch.zeros(2, num_mel_frames, dtype=torch.bool)
    attention_mask[0, : num_mel_frames - 1] = True
    attention_mask[1, :17] = True

    with torch.inference_mode():
        output, output_mask = model(
            input_features,
            attention_mask,
            prompt_ids=torch.tensor([2, 3]),
        )
        unpadded_output, unpadded_mask = model(
            input_features[1:2, :17],
            torch.ones(1, 17, dtype=torch.bool),
            prompt_ids=torch.tensor([3]),
        )

    assert output.shape == (2, physical_frames, 8)
    assert output_mask is not None
    assert output_mask.sum(-1).tolist() == [valid_frames, 3]
    assert torch.isfinite(output).all()
    assert unpadded_mask is not None
    assert unpadded_mask.sum().item() == 3
    torch.testing.assert_close(output[1, :3], unpadded_output[0, :3])
