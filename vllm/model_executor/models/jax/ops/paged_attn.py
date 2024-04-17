import jax
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention


def paged_attn(
    q: jax.Array,               # [batch, 1, num_heads, head_size]
    k_cache: jax.Array,         # [num_kv_heads, num_blocks, block_size, head_size]
    v_cache: jax.Array,         # [num_kv_heads, num_blocks, block_size, head_size]
    block_tables: jax.Array,    # [batch, max_num_blocks_per_batch]
    context_lens: jax.Array,    # [batch]
) -> jax.Array:                 # [batch, 1, num_heads, head_size]
    q = q.squeeze(1)
    output = paged_attention(
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        pages_per_compute_block=4,
    )
    return output.reshape(q.shape[0], 1, q.shape[1], q.shape[2])
