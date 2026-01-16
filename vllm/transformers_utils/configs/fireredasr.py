import os
from typing import Optional

from transformers import PretrainedConfig
from transformers import Qwen2Config


class FireRedAsrEncoderConfig(PretrainedConfig):
    model_type = "fireredasr_audio_encoder"

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
        n_layers_enc=16,
        n_head=20,
        d_model=1280,
        d_inner=5120,
        residual_dropout=0.1,
        pe_maxlen=5000,
        dropout_rate=0.1,
        kernel_size=33,
        n_layers_dec=16,
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
        self.n_layers_enc = n_layers_enc
        self.n_head = n_head
        self.d_model = d_model
        self.d_inner = d_inner
        self.residual_dropout = residual_dropout
        self.dropout_rate = dropout_rate
        self.pe_maxlen = pe_maxlen
        self.kernel_size = kernel_size
        self.n_layers_dec = n_layers_dec
        self.blank_id = blank_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        super().__init__(**kwargs)


class FireRedAsrConfig(PretrainedConfig):
    model_type = "fireredasr"
    sub_configs = {
        "encode_config": FireRedAsrEncoderConfig,
        "text_config": Qwen2Config,
    }

    def __init__(
        self,
        text_config=None,
        encode_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(encode_config, dict):
            encode_config = FireRedAsrEncoderConfig(**encode_config)
        elif encode_config is None:
            encode_config = FireRedAsrEncoderConfig()
        self.encode_config = encode_config

        if isinstance(text_config, dict):
            text_config = Qwen2Config(**text_config)
        elif text_config is None:
            text_config = Qwen2Config()
        self.text_config = text_config


    def get_text_config(self, **kwargs) -> PretrainedConfig:        
        return self.text_config.get_text_config()