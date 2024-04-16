""" Jamba model configuration"""
import math
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig


class JambaConfig(PretrainedConfig):
    r"""
    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the Jurassic model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`JurassicModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        # max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
        #     The maximum sequence length that this model might ever be used with. Jurassic's sliding window attention
        #     allows sequence of up to 4096*32 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        use_positional_embeddings (`bool`, *optional, default False)
            flag indicating whether to use positional embeddings or not
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If not specified, will default to `4096`.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            The number of experts to root per-token, can be also interpreted as the `top-p` routing
            parameter
        num_experts (`int`, *optional*, defaults to 16):
            Number of experts per Sparse MLP layer.
        expert_layer_period (`int`, *optional*, defaults to 2)
            Once in this many layers, we will have an expert layer
        expert_layer_offset(`int`, *optional*, defaults to 1)
            The first layer index that contains an expert mlp layer
        attn_layer_period (`int`, *optional*, defaults to 8)
            Once in this many layers, we will have a vanilla attention layer
        attn_layer_offset(`int`, *optional*, defaults to 4)
            The first layer index that contains a vanilla attention mlp layer
    """

    model_type = "jamba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=65536,
        tie_word_embeddings=False,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_experts=16,
        expert_layer_offset=1,
        expert_layer_period=2,
        attn_layer_period=8,
        attn_layer_offset=4,
        use_mamba_kernels=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank="auto",
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_inner_layernorms=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.expert_layer_period = expert_layer_period
        self.expert_layer_offset = expert_layer_offset
        self.attn_layer_period = attn_layer_period
        self.attn_layer_offset = attn_layer_offset

        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = math.ceil(
            self.hidden_size /
            16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_inner_layernorms = mamba_inner_layernorms

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


AutoConfig.register('jamba', JambaConfig)
