import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, LinearScalingRotaryEmbedding, DynamicNTKScalingRotaryEmbedding


class RefLlamaRotaryEmbedding(torch.nn.Module):

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base**(
            torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device=self.inv_freq.device,
                                dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos()[None, None, :, :].to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin()[None, None, :, :].to(dtype),
                             persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len,
                                    device=x.device,
                                    dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class RefLlamaLinearScalingRotaryEmbedding(RefLlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None,
                 scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos()[None, None, :, :].to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin()[None, None, :, :].to(dtype),
                             persistent=False)


class RefLlamaDynamicNTKScalingRotaryEmbedding(RefLlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None,
                 scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * ((self.scaling_factor * seq_len /
                                 self.max_position_embeddings) -
                                (self.scaling_factor - 1))**(self.dim /
                                                             (self.dim - 2))
            inv_freq = 1.0 / (base**(
                torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos()[None, None, :, :].to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin()[None, None, :, :].to(dtype),
                             persistent=False)


def test_rope():

    def ref_rope(dim, max_position_embeddings):
        emb = RefLlamaRotaryEmbedding(dim, max_position_embeddings)
        return torch.cat((emb.cos_cached.squeeze()[:, :dim // 2],
                          emb.sin_cached.squeeze()[:, :dim // 2]),
                         dim=-1).cuda()

    def vllm_rope(dim, max_position_embeddings):
        emb = RotaryEmbedding(dim, max_position_embeddings)
        return emb.cos_sin_cache

    dim, max_position_embeddings = 128, 2048
    assert torch.allclose(ref_rope(dim, max_position_embeddings),
                          vllm_rope(dim, max_position_embeddings))


def test_linear_rope():

    def ref_rope(dim, max_position_embeddings, scaling_factor):
        emb = RefLlamaLinearScalingRotaryEmbedding(
            dim, max_position_embeddings, scaling_factor=scaling_factor)
        return torch.cat((emb.cos_cached.squeeze()[:, :dim // 2],
                          emb.sin_cached.squeeze()[:, :dim // 2]),
                         dim=-1).cuda()

    def vllm_rope(dim, max_position_embeddings, scaling_factor):
        emb = LinearScalingRotaryEmbedding(dim,
                                                max_position_embeddings,
                                                scaling_factor=scaling_factor)
        return emb.cos_sin_cache

    dim, max_position_embeddings, scaling_factor = 128, 2048, 8.0
    assert torch.allclose(
        ref_rope(dim, max_position_embeddings, scaling_factor),
        vllm_rope(dim, max_position_embeddings, scaling_factor))


def test_ntk_rope():

    def ref_rope(dim, max_position_embeddings, scaling_factor):
        emb = RefLlamaDynamicNTKScalingRotaryEmbedding(
            dim, max_position_embeddings, scaling_factor=scaling_factor)
        return torch.cat((emb.cos_cached.squeeze()[:, :dim // 2],
                          emb.sin_cached.squeeze()[:, :dim // 2]),
                         dim=-1).cuda()

    def vllm_rope(dim, max_position_embeddings, scaling_factor):
        emb = DynamicNTKScalingRotaryEmbedding(
            dim, max_position_embeddings, scaling_factor=scaling_factor)
        return emb.cos_sin_cache

    dim, max_position_embeddings, scaling_factor = 128, 2048, 8.0
    assert torch.allclose(
        ref_rope(dim, max_position_embeddings, scaling_factor),
        vllm_rope(dim, max_position_embeddings, scaling_factor))


test_rope()
test_linear_rope()
test_ntk_rope()
