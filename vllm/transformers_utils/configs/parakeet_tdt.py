# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import ParakeetEncoderConfig, PretrainedConfig

PARAKEET_TDT_EOS_TOKEN_ID = 3


class ParakeetTDTConfig(PretrainedConfig):
    """Configuration for NVIDIA Parakeet TDT checkpoints.

    Transformers versions that include the Parakeet encoder do not yet expose
    the TDT wrapper used by ``nvidia/parakeet-tdt-0.6b-v3``.
    """

    model_type = "parakeet_tdt"
    sub_configs = {"encoder_config": ParakeetEncoderConfig}

    def __init__(
        self,
        *,
        encoder_config: dict | ParakeetEncoderConfig | None = None,
        vocab_size: int = 8193,
        blank_token_id: int = 8192,
        decoder_hidden_size: int = 640,
        durations: list[int] | None = None,
        num_decoder_layers: int = 2,
        max_symbols_per_step: int = 10,
        sample_rate: int = 16000,
        hidden_act: str = "relu",
        pad_token_id: int = 2,
        eos_token_id: int = PARAKEET_TDT_EOS_TOKEN_ID,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=blank_token_id,
            decoder_start_token_id=blank_token_id,
            **kwargs,
        )

        if isinstance(encoder_config, dict):
            encoder_config = ParakeetEncoderConfig(**encoder_config)
        elif encoder_config is None:
            encoder_config = ParakeetEncoderConfig()

        # The HF checkpoint stores the audio sample rate in processor_config,
        # while vLLM needs it on the model config for ASR preprocessing.
        encoder_config.sampling_rate = sample_rate

        self.encoder_config = encoder_config
        self.vocab_size = vocab_size
        self.blank_token_id = blank_token_id
        self.decoder_hidden_size = decoder_hidden_size
        self.hidden_size = decoder_hidden_size
        self.durations = durations or [0, 1, 2, 3, 4]
        self.num_decoder_layers = num_decoder_layers
        self.num_hidden_layers = 0
        self.num_attention_heads = 0
        self.max_symbols_per_step = max_symbols_per_step
        self.sample_rate = sample_rate
        self.hidden_act = hidden_act
        self.is_encoder_decoder = True


__all__ = ["PARAKEET_TDT_EOS_TOKEN_ID", "ParakeetTDTConfig"]
