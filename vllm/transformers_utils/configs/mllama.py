from transformers.models.mllama.configuration_mllama import (
    MllamaConfig as MllamaConfigHf)
from transformers.models.mllama.configuration_mllama import (
    MllamaTextConfig as MllamaTextConfigHf)


class MllamaTextConfig(MllamaTextConfigHf):
    '''
    Use this class to override is_encoder_decoder:
    - transformers regards mllama as is_encoder_decoder=False
    - vllm needs is_encoder_decoder=True to enable cross-attention
    '''

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_act = self.hidden_activation
        self.is_encoder_decoder = True


class MllamaConfig(MllamaConfigHf):

    def __init__(
        self,
        text_config=None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = MllamaTextConfig(**text_config)
        super().__init__(text_config=text_config, **kwargs)
