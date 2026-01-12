from transformers import PretrainedConfig
from typing import Any, Dict, Optional
from vllm.transformers_utils.config import patch_rope_parameters

class AXK1Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AXK1Model`]. It is used to instantiate an A.X
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the A.X K1.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 163840):
            Vocabulary size of the A.X K1 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`AXK1Model`]
        hidden_size (`int`, *optional*, defaults to 7168):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18432):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 61):
            Number of hidden layers in the Transformer decoder.
        num_nextn_predict_layers (`int`, *optional*, defaults to 1):
            Number of nextn predict layers in the AXK1 Model.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer decoder.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts, None means dense model.
        n_routed_experts (`int`, *optional*, defaults to 192):
            Number of routed experts, None means dense model.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor or routed experts.
        topk_method (`str`, *optional*, defaults to `noaux_tc`):
            Topk method used in routed gate.
        n_group (`int`, *optional*, defaults to 8):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 4):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts, None means dense model.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
        first_k_dense_replace (`int`, *optional*, defaults to 1):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                            \--k dense layers--/
        norm_topk_prob (`bool`, *optional*, defaults to True):
            Whether to normalize the weights of the routed experts.
        scoring_func (`str`, *optional*, defaults to 'sigmoid'):
            Method of computing expert weights.
        aux_loss_alpha (`float`, *optional*, defaults to 0.0001):
            Auxiliary loss weight coefficient.
        seq_aux = (`bool`, *optional*, defaults to True):
            Whether to compute the auxiliary loss for each individual sample.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 163691):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 163691):
            End of stream token id.
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
            `max_position_embeddings` to the expected new maximum.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    """

    model_type = "AXK1"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 163840,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 61,
        num_nextn_predict_layers: Optional[int] = 1,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        n_shared_experts: Optional[int] = 1,
        n_routed_experts: Optional[int] = 192,
        ep_size: Optional[int] = 8,                     ## Ignored - Expert parallel size
        routed_scaling_factor: Optional[float] = 2.5,
        kv_lora_rank: Optional[int] = 512,
        q_lora_rank: Optional[int] = 1536,
        qk_rope_head_dim: Optional[int] = 64,
        v_head_dim: Optional[int] = 128,
        qk_nope_head_dim: Optional[int] = 128,
        topk_method: Optional[str] = 'noaux_tc',
        n_group: Optional[int] = 8,
        topk_group: Optional[int] = 4,
        num_experts_per_tok: Optional[int] = 8,
        moe_layer_freq: Optional[int] = 1,
        first_k_dense_replace: int = 1,
        norm_topk_prob: bool = True,
        scoring_func: Optional[str] = 'sigmoid',
        aux_loss_alpha: Optional[float] = 0.0001,
        seq_aux: Optional[float] = True,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 131072,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = 163691,
        eos_token_id: Optional[int] = 163691,
        pretraining_tp: Optional[int] = 1,
        tie_word_embeddings: Optional[bool] = False,
        rope_theta: Optional[float] = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_parameters: Optional[Dict[str, Any]] = None,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
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
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
