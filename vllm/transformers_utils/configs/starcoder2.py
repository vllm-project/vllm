from transformers import PretrainedConfig


class Starcoder2Config(PretrainedConfig):
    model_type = "starcoder2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=49152,
        hidden_size=3072,
        intermediate_size=12288,
        num_hidden_layers=30,
        num_attention_heads=24,
        num_key_value_heads=2,
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=4096,
        initializer_range=0.018042,
        norm_epsilon=1e-5,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        rope_theta=10000.0,
        sliding_window=None,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        use_bias=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.use_bias = use_bias
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.norm_epsilon = norm_epsilon
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        if self.architectures is None:
            self.architectures = ['Starcoder2ForCausalLM']
