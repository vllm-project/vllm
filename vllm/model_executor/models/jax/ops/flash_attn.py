import jax
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes, flash_attention

_DEFAULT_BLOCK_SIZES = {
    "block_q": 512,
    "block_k_major": 512,
    "block_k": 512,
    "block_b": 2,
}

def flash_attn(
    q: jax.Array,  # [batch, seq_len, num_heads, head_size]
    k: jax.Array,  # [batch, seq_len, num_heads, head_size]
    v: jax.Array,  # [batch, seq_len, num_heads, head_size]
    sm_scale: float,
) -> jax.Array:    # [batch, seq_len, num_heads, head_size]
    return flash_attention(
        q.transpose(0, 2, 1, 3),
        k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3),
        causal=True,
        sm_scale=sm_scale,
        block_sizes=BlockSizes(
        min(_DEFAULT_BLOCK_SIZES["block_q"], q.shape[1]),
        min(_DEFAULT_BLOCK_SIZES["block_k_major"], k.shape[1]),
        min(_DEFAULT_BLOCK_SIZES["block_k"], k.shape[1]),
        min(_DEFAULT_BLOCK_SIZES["block_b"], q.shape[0])),
    ).transpose(0, 2, 1, 3)
