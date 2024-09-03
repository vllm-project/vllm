from transformers import PretrainedConfig
from typing import Optional


class LlamaVLConfig(PretrainedConfig):
    model_type = "llamavl"

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False

    max_batch_size: int = 32
    max_seq_len: int = 2048

    # vision model params
    vision_chunk_size: int = -1  # image resolution for image models
    vision_max_num_chunks: int = 4
    vision_num_cross_attention_layers: int = -1

    model_type: str = "llamavl"
    architectures: list[str] = ["LlamaVLForCausalLM"]

    torch_dtype: str = "bfloat16"

    attribute_map = {
        "num_hidden_layers": "n_layers",
        "hidden_size": "dim",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads",
    }

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0

        super().__init__(**kwargs)
