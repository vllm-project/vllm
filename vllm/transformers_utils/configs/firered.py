# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig, Qwen2Config


class FireRedAudioEncoderConfig(PretrainedConfig):
    model_type = "firered_audio_encoder"

    def __init__(
        self,
        blank="<blank>",
        pad="<pad>",
        sos="<sos>",
        eos="<eos>",
        unk="<unk>",
        input_length_max=60.0,
        input_length_min=0.1,
        output_length_max=250,
        output_length_min=1,
        idim=80,
        odim=7832,
        encoder_layers=16,
        encoder_attention_heads=20,
        d_model=1280,
        d_inner=5120,
        residual_dropout=0.1,
        pe_maxlen=5000,
        dropout_rate=0.1,
        kernel_size=33,
        blank_id=0,
        sos_id=3,
        eos_id=4,
        pad_id=2,
        **kwargs,
    ):
        self.blank = blank
        self.pad = pad
        self.sos = sos
        self.eos = eos
        self.unk = unk
        self.input_length_max = input_length_max
        self.input_length_min = input_length_min
        self.output_length_max = output_length_max
        self.output_length_min = output_length_min
        self.idim = idim
        self.odim = odim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.d_model = d_model
        self.d_inner = d_inner
        self.residual_dropout = residual_dropout
        self.dropout_rate = dropout_rate
        self.pe_maxlen = pe_maxlen
        self.kernel_size = kernel_size
        self.blank_id = blank_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        super().__init__(**kwargs)


class FireRedAudioConfig(PretrainedConfig):
    model_type = "firered_audio"
    sub_configs = {
        "audio_config": FireRedAudioEncoderConfig,
        "text_config": Qwen2Config,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        encoder_downsample_rate=2,
        default_speech_token="<speech>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_downsample_rate = encoder_downsample_rate
        self.default_speech_token = default_speech_token

        if isinstance(audio_config, dict):
            audio_config = FireRedAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = FireRedAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen2Config(**text_config)
        elif text_config is None:
            text_config = Qwen2Config()
        self.text_config = text_config
