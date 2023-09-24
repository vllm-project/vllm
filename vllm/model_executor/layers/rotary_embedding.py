import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):

    def __init__(self,
                 rotary_dim: int,
                 max_position_embeddings: int = 2048,
                 base: int = 10000) -> None:
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rotary_dim = rotary_dim

        # Create the cos and sin cache.
        inv_freq = 1.0 / (base**(torch.arange(
            0, rotary_dim, 2, dtype=torch.float, device="cuda") / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cache(self, t: torch.Tensor):
        freqs = torch.einsum("i,j -> ij", t, self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        # FIXME(woosuk): This assumes that we configure the default dtype when
        # initializing the model.
        # TODO(woosuk): Make it more robust.
        torch_dtype = torch.get_default_dtype()
        cache = cache.to(torch_dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         dtype=torch.float,
                         device="cuda")
        self._set_cache(t)

    # Unlink HF or TGI, we do not dynamically set the cos_sin_cache
    # because vLLM needs to statically allocate memory in the beginning
    # Therefore, we don't need to know the current seq_len
    def forward(self):
        return self.cos_sin_cache


class LinearScalingRotaryEmbedding(RotaryEmbedding):

    def __init__(self,
                 rotary_dim: int,
                 max_position_embeddings: int = 2048,
                 base: int = 10000,
                 scaling_factor: float = 1.0):
        self.scaling_factor = scaling_factor
        super().__init__(rotary_dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len * self.scaling_factor
        t = torch.arange(self.max_seq_len_cached,
                         dtype=torch.float,
                         device="cuda") / self.scaling_factor
        self._set_cache(t)
