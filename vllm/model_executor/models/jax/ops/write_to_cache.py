import jax


def write_to_cache(
    x: jax.Array,
    cache: jax.Array,
    slot_mapping: jax.Array,
) -> jax.Array:
    num_heads, num_blocks, block_size, head_size = cache.shape
    cache = cache.reshape(num_heads, num_blocks * block_size, head_size)
    x = x.reshape(-1, x.shape[-2], x.shape[-1])
    slot_mapping = slot_mapping.reshape(-1)
    cache = cache.at[:, slot_mapping, :].set(x.transpose(1, 0, 2))
    return cache.reshape(num_heads, num_blocks, block_size, head_size)
