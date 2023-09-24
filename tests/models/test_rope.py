import torch
from typing import Optional
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, LinearScalingRotaryEmbedding


class RefRotaryEmbedding(torch.nn.Module):

    def __init__(self,
                 dim: int,
                 max_position_embeddings: int = 2048,
                 base: int = 10000,
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

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device,
                           dtype: torch.dtype):
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

    def forward(self, x: torch.tensor, seq_len: Optional[int] = None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len,
                                    device=x.device,
                                    dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class RefLinearScalingRotaryEmbedding(RefRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self,
                 dim: int,
                 max_position_embeddings: int = 2048,
                 base: int = 10000,
                 device: Optional[torch.device] = None,
                 scaling_factor: Optional[float] = 1.0):
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


def test_rope():

    def ref_rope(dim: int, max_position_embeddings: int):
        emb = RefRotaryEmbedding(dim, max_position_embeddings, device='cuda')
        return torch.cat((emb.cos_cached.squeeze()[:, :dim // 2],
                          emb.sin_cached.squeeze()[:, :dim // 2]),
                         dim=-1)

    def vllm_rope(dim: int, max_position_embeddings: int):
        emb = RotaryEmbedding(dim, max_position_embeddings)
        return emb.cos_sin_cache

    dim, max_position_embeddings = 128, 2048
    assert torch.allclose(ref_rope(dim, max_position_embeddings),
                          vllm_rope(dim, max_position_embeddings))


def test_linear_rope():

    def ref_rope(dim: int, max_position_embeddings: int,
                 scaling_factor: float):
        emb = RefLinearScalingRotaryEmbedding(dim,
                                              max_position_embeddings,
                                              scaling_factor=scaling_factor,
                                              device='cuda')
        seq_len = int(max_position_embeddings * scaling_factor)
        x = torch.rand((1, 1, seq_len, 1), dtype=torch.float, device='cuda')
        emb(x, seq_len)
        return torch.cat((emb.cos_cached.squeeze()[:, :dim // 2],
                          emb.sin_cached.squeeze()[:, :dim // 2]),
                         dim=-1)

    def vllm_rope(dim: int, max_position_embeddings: int,
                  scaling_factor: float):
        emb = LinearScalingRotaryEmbedding(dim,
                                           max_position_embeddings,
                                           scaling_factor=scaling_factor)
        return emb.cos_sin_cache

    dim, max_position_embeddings, scaling_factor = 128, 2048, 8.0
    assert torch.allclose(
        ref_rope(dim, max_position_embeddings, scaling_factor),
        vllm_rope(dim, max_position_embeddings, scaling_factor))


test_rope()
test_linear_rope()
