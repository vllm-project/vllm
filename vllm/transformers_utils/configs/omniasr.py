# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import LlamaConfig, PretrainedConfig


class OmniASRConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`OmniAsrForConditionalGeneration`]. It is used to instantiate an
    OmniASR LLM model according to the specified arguments, defining the model
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to
    control the model outputs. Read the documentation from [`PretrainedConfig`]
    for more information.

    Default values correspond to the 300M encoder variant
    (facebook/omniASR-LLM-300M). The LLaMA decoder is identical across all
    variants (300M, 1B, 3B, 7B).

    Args:
        sampling_rate (`int`, *optional*, defaults to 16000):
            Sampling rate of input audio waveforms in Hz.
        subsampling_factor (`int`, *optional*, defaults to 320):
            Total downsampling factor of the CNN feature extractor.
        feature_dim (`int`, *optional*, defaults to 512):
            Dimensionality of extracted features from the CNN frontend.
        feature_extractor_layer_descs (`list[tuple[int, int, int]]`, *optional*):
            A list of (out_channels, kernel_size, stride) for each conv layer
            in the feature extractor. Defaults to the standard wav2vec 2.0
            7-layer configuration.
        feature_extractor_bias (`bool`, *optional*, defaults to `True`):
            If `True`, convolutions in feature extraction layers learn an
            additive bias.
        feature_extractor_layer_norm_convs (`bool`, *optional*, defaults to `True`):
            If `True`, applies Layer Normalization to outputs of convolutions
            in feature extraction layers.
        encoder_embed_dim (`int`, *optional*, defaults to 1024):
            Dimensionality of the wav2vec 2.0 transformer encoder.
        encoder_num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in each encoder layer.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Inner dimensionality of feed-forward networks in the encoder.
        encoder_num_layers (`int`, *optional*, defaults to 24):
            Number of transformer layers in the encoder.
        pos_encoder_kernel_size (`int`, *optional*, defaults to 128):
            Kernel size of the convolutional position encoder.
        pos_encoder_groups (`int`, *optional*, defaults to 16):
            Number of groups in the convolutional position encoder.
        projection_dim (`int`, *optional*):
            Dimensionality of the linear projection from encoder to decoder.
            Defaults to `text_config.hidden_size` if not specified.
        text_config (`dict`, *optional*):
            Configuration dict for the LLaMA decoder. Passed to
            `LlamaConfig`. Defaults to a 12-layer, 8-head, 4096-dim decoder.
        lang_embeddings_p (`float`, *optional*, defaults to 0.5):
            Probability of including language embeddings during training.
        n_special_tokens (`int`, *optional*, defaults to 1):
            Number of special tokens prepended to the decoder input.
        target_vocab_size (`int`, *optional*, defaults to 9812):
            Size of the target vocabulary for the decoder output projection.
        num_languages (`int`, *optional*, defaults to 1694):
            Number of supported languages for language conditioning.
        bos_token_id (`int`, *optional*, defaults to 0):
            Token id for beginning-of-sequence.
        eos_token_id (`int`, *optional*, defaults to 2):
            Token id for end-of-sequence.
        pad_token_id (`int`, *optional*, defaults to 1):
            Token id for padding.
        lid_marker_token_id (`int`, *optional*, defaults to 9812):
            Token id for lid marker

    Example:

    ```python
    >>> from vllm.transformers_utils.configs.omniasr import OmniASRConfig

    >>> # Initializing a default OmniASR config (300M encoder)
    >>> configuration = OmniASRConfig()

    >>> # Accessing encoder dimension
    >>> configuration.encoder_embed_dim
    1024
    ```
    """

    model_type = "omniasr_llm"

    def __init__(
        self,
        sampling_rate: int = 16000,
        subsampling_factor: int = 320,
        feature_dim: int = 512,
        feature_extractor_layer_descs: list[tuple[int, int, int]] | None = None,
        feature_extractor_bias: bool = True,
        feature_extractor_layer_norm_convs: bool = True,
        encoder_embed_dim: int = 1024,
        encoder_num_heads: int = 16,
        encoder_ffn_dim: int = 4096,
        encoder_num_layers: int = 24,
        pos_encoder_kernel_size: int = 128,
        pos_encoder_groups: int = 16,
        projection_dim: int | None = None,
        text_config: dict | LlamaConfig | None = None,
        lang_embeddings_p: float = 0.5,
        n_special_tokens: int = 1,
        target_vocab_size: int = 9812,
        num_languages: int = 1694,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pad_token_id: int = 1,
        lid_marker_token_id: int = 9812,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.sampling_rate = sampling_rate
        self.subsampling_factor = subsampling_factor
        self.feature_dim = feature_dim
        self.feature_extractor_layer_descs = feature_extractor_layer_descs or [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ]
        self.feature_extractor_bias = feature_extractor_bias
        self.feature_extractor_layer_norm_convs = feature_extractor_layer_norm_convs

        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_num_heads = encoder_num_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_num_layers = encoder_num_layers

        self.pos_encoder_kernel_size = pos_encoder_kernel_size
        self.pos_encoder_groups = pos_encoder_groups

        if text_config is None:
            text_config = {
                "model_type": "llama",
                "hidden_size": 4096,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "num_hidden_layers": 12,
                "intermediate_size": 2816,
                "max_position_embeddings": 8192,
                "rope_theta": 10000.0,
                "vocab_size": 9813,
                "rms_norm_eps": 1e-05,
            }
        if isinstance(text_config, dict):
            text_config = text_config.copy()
            text_config.setdefault("architectures", ["LlamaForCausalLM"])
            text_config = LlamaConfig(**text_config)
        assert isinstance(text_config, LlamaConfig)
        self.text_config = text_config
        self.projection_dim = projection_dim or self.text_config.hidden_size
        self.lang_embeddings_p = lang_embeddings_p
        self.n_special_tokens = n_special_tokens
        self.target_vocab_size = target_vocab_size
        self.num_languages = num_languages
        self.lid_marker_token_id = lid_marker_token_id
