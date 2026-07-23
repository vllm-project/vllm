# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://github.com/ManaEstras/transformers/blob/v4.57.1.hyvl/src/transformers/models/hunyuan_vl/configuration_hunyuan_vl.py

from transformers import PretrainedConfig


class HunYuanVLVisionConfig(PretrainedConfig):
    model_type = "hunyuan_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_act="gelu",
        hidden_size=1152,
        intermediate_size=4304,
        interpolate_mode="bilinear",
        rms_norm_eps=1e-05,
        learnable_mlp_pooling_size=0,
        num_attention_heads=16,
        num_key_value_heads=None,
        num_channels=3,
        num_hidden_layers=27,
        out_hidden_size=4096,
        patch_size=16,
        remove_prenorm=True,
        spatial_merge_size=2,
        temporal_patch_size=1,
        resize_resolution=2048,
        img_max_token_num=4096,
        max_image_size=2048,
        video_max_image_size=768,
        video_min_image_size=256,
        min_image_size=512,
        anyres_vit_max_image_size=2048,
        max_vit_seq_len=16384,
        text_hidden_size=3072,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.interpolate_mode = interpolate_mode
        self.learnable_mlp_pooling_size = learnable_mlp_pooling_size
        self.num_attention_heads = num_attention_heads
        if not num_key_value_heads:
            self.num_key_value_heads = num_attention_heads
        else:
            self.num_key_value_heads = num_key_value_heads
        self.num_channels = num_channels
        self.num_hidden_layers = num_hidden_layers
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.remove_prenorm = remove_prenorm
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps

        self.resize_resolution = resize_resolution
        self.img_max_token_num = img_max_token_num
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.video_max_image_size = video_max_image_size
        self.video_min_image_size = video_min_image_size
        self.anyres_vit_max_image_size = anyres_vit_max_image_size
        self.max_vit_seq_len = max_vit_seq_len
        self.text_hidden_size = text_hidden_size


class HunYuanVLTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HunYuanVLTextConfig`]. It is used to instantiate an
    HunYuan model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the HunYuan-7B.
    Hunyuan-7B-Instruct [tencent/Hunyuan-7B-Instruct](https://huggingface.co/tencent/Hunyuan-7B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 290943):
            Vocabulary size of the HunYuan model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HunYuanVLTextConfig`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations or shared MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        eod_token_id (int, *optional*, defaults to 3):
            Token ID representing the end-of-document marker. Used to indicate the termination of a text sequence.
            Example: In multi-document processing, this token helps the model distinguish between separate documents.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
    """  # noqa: E501

    model_type = "hunyuan_vl_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=290943,
        hidden_size=4096,
        intermediate_size: int = 11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        eod_token_id=3,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # self._rope_scaling_validation()   # TODO: Need validation?
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and "
                f"`factor` or `type` and `alpha`, got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        rope_scaling_alpha = self.rope_scaling.get("alpha", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                "`rope_scaling`'s type field must be one of ['linear', 'dynamic'], "
                f"got {rope_scaling_type}"
            )
        if rope_scaling_factor is None and rope_scaling_alpha is None:
            raise ValueError(
                "`rope_scaling`'s factor or alpha field must be have one, "
                "got both of none"
            )
        if rope_scaling_factor is not None and (
            not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0
        ):
            raise ValueError(
                "`rope_scaling`'s factor field must be a float > 1.0, "
                f"got {rope_scaling_factor}"
            )
        if rope_scaling_alpha is not None and (
            not isinstance(rope_scaling_alpha, float) or rope_scaling_alpha <= 1.0
        ):
            raise ValueError(
                "`rope_scaling`'s alpha field must be a float > 1.0, "
                f"got {rope_scaling_alpha}"
            )


class HunYuanVLConfig(PretrainedConfig):
    model_type = "hunyuan_vl"
    sub_configs = {
        "vision_config": HunYuanVLVisionConfig,
        "text_config": HunYuanVLTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        im_start_id=120118,
        im_end_id=120119,
        image_token_id=120120,
        im_newline_id=120121,
        video_start_id=120122,
        video_end_id=120123,
        **kwargs,
    ):
        # We need to init super() here so that it does not reset values
        # that are in text config to the BaseClass defaults. The Base
        # config has many text related defaults and not all defaults are
        # same as for `HunYuanVLTextConfig`.
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.im_newline_id = im_newline_id
        self.video_start_id = video_start_id
        self.video_end_id = video_end_id

        self.vision_config.text_hidden_size = self.text_config.hidden_size

        # Attention implementation to use. It sets it recursively on sub-configs
        # so we call it again in the end.
        self._attn_implementation = kwargs.pop("attn_implementation", None)

    def __setattr__(self, key, value):
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config"))
            is not None
            and key not in ["dtype", "_attn_implementation_internal"]
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        if "text_config" in super().__getattribute__("__dict__") and key not in [
            "_name_or_path",
            "model_type",
            "dtype",
            "_attn_implementation_internal",
        ]:
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                return getattr(text_config, key)

        return super().__getattribute__(key)
