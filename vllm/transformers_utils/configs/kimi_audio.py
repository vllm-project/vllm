# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct/blob/main/configuration_moonshot_kimia.py
from transformers.models.whisper.configuration_whisper import WhisperConfig


class WhisperVQConfig(WhisperConfig):
    model_type = "whisper_vq"
    
    def __init__(self,
                 pooling_kernel_size=None,
                 pooling_type="max",
                 pooling_position=0,
                 quantize_vocab_size=None,
                 quantize_position=16,
                 quantize_commit_coefficient=0.25,
                 quantize_loss_scale=1.0,
                 quantize_ema_decay=None,
                 quantize_restart_interval=None,
                 quantize_encoder_only=False,
                 quantize_causal_encoder=False,
                 quantize_causal_block_size=None,
                 skip_language_detection=False,
                 encoder_causal_attention=False,
                 encoder_causal_convolution=False,
                 **kwargs):
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_type = pooling_type
        self.pooling_position = pooling_position
        self.quantize_vocab_size = quantize_vocab_size
        self.quantize_position = quantize_position
        self.quantize_commit_coefficient = quantize_commit_coefficient
        self.quantize_loss_scale = quantize_loss_scale
        self.quantize_ema_decay = quantize_ema_decay
        self.quantize_restart_interval = quantize_restart_interval
        self.quantize_encoder_only = quantize_encoder_only
        self.quantize_causal_encoder = quantize_causal_encoder
        self.quantize_causal_block_size = quantize_causal_block_size
        self.skip_language_detection = skip_language_detection
        self.encoder_causal_attention = encoder_causal_attention
        self.encoder_causal_convolution = encoder_causal_convolution
        super().__init__(**kwargs)