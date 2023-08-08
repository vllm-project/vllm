from transformers.configuration_utils import PretrainedConfig


class SkyWorkConfig(PretrainedConfig):
    model_type = "skywork"

    def __init__(
        self,
        activation_function="gelu",
        vocab_size=63174,
        hidden_size=8192,
        max_position_embeddings=16384,
        n_ctx=16384,
        num_hidden_layers=96,
        num_attention_heads=64,
        layer_norm_epsilon=1e-05,
        n_inner=32768,
        bos_token_id=6,
        eos_token_id=1,
        pad_token_id=0,
        sep_token_id=2,
        reorder_and_upcast_attn=False,
        scale_attn_by_inverse_layer_idx=False,
        scale_attn_weights=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_type="cls_index",
        summary_use_proj=True,
        use_cache=True,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.activation_function = activation_function
        self.layer_norm_epsilon = layer_norm_epsilon
        self.n_ctx = n_ctx
        self.n_inner = n_inner
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.scale_attn_weights = scale_attn_weights
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.use_cache = use_cache
        self.architectures = ["SkyWorkLMHeadModel"]
        

class SkyWork_2_6BConfig(PretrainedConfig):
    model_type = "skywork2.6B"

    def __init__(
        self,
        activation_function="gelu",
        vocab_size=63174,
        hidden_size=2560,
        max_position_embeddings=16384,
        n_ctx=16384,
        num_hidden_layers=32,
        num_attention_heads=32,
        layer_norm_epsilon=1e-05,
        n_inner=None,
        bos_token_id=6,
        eos_token_id=1,
        pad_token_id=0,
        sep_token_id=2,
        reorder_and_upcast_attn=False,
        scale_attn_by_inverse_layer_idx=False,
        scale_attn_weights=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_type="cls_index",
        summary_use_proj=True,
        use_cache=True,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.activation_function = activation_function
        self.layer_norm_epsilon = layer_norm_epsilon
        self.n_ctx = n_ctx
        self.n_inner = n_inner
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.scale_attn_weights = scale_attn_weights
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.use_cache = use_cache
        self.architectures = ["SkyWorkLMHeadModel"]