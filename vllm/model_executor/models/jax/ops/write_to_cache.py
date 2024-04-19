import chex
import jax
import jax.numpy as jnp

_PAD_SLOT_ID = -1


def write_to_kv_cache(
    key: jax.Array,             # [batch_size, seq_len, num_heads, head_size]
    value: jax.Array,           # [batch_size, seq_len, num_heads, head_size]
    kv_cache: jax.Array,        # [2, num_heads, num_blocks, block_size, head_size]
    slot_mapping: jax.Array,    # [batch_size, seq_len]
) -> jax.Array:
    f = _write_to_kv_cache
    return f(key, value, kv_cache, slot_mapping)
 

def _write_to_kv_cache(
    key: jax.Array,             # [batch_size, seq_len, num_heads, head_size]
    value: jax.Array,           # [batch_size, seq_len, num_heads, head_size]
    kv_cache: jax.Array,        # [2, num_heads, num_blocks, block_size, head_size]
    slot_mapping: jax.Array,    # [batch_size, seq_len]
) -> jax.Array:
    """Out-of-place write to KV cache."""
    num_heads, num_blocks, block_size, head_size = kv_cache.shape[1:]
    key_value = jnp.stack([key, value])  # [2, batch_size, seq_len, num_heads, head_size]
    key_value = key_value.reshape(2, -1, num_heads, head_size)
    key_value = key_value.transpose((0, 2, 1, 3))

    kv_cache = kv_cache.reshape(2, num_heads, num_blocks * block_size, head_size)
    kv_cache = kv_cache.at[:, :, slot_mapping.reshape(-1), :].set(key_value)
    kv_cache = kv_cache.reshape(2, num_heads, num_blocks, block_size, head_size)
    return kv_cache


def _write_to_kv_cache_in_place(
    key: jax.Array,             # [batch_size, seq_len, num_heads, head_size]
    value: jax.Array,           # [batch_size, seq_len, num_heads, head_size]
    kv_cache: jax.Array,        # [2, num_heads, num_blocks, block_size, head_size]
    slot_mapping: jax.Array,    # [batch_size, seq_len]
) -> jax.Array:
    """In-place write to KV cache."""
    batch_size = slot_mapping.shape[0]
    key_value = jnp.stack([key, value], axis=2)  # [batch_size, seq_len, 2, num_heads, head_size]

    def cond(val: _IteratorState):
        return val.idx < batch_size

    def body(val: _IteratorState):
        val.kv_cache = _write_seq_to_kv_cache(
            key_value[val.idx],
            val.kv_cache,
            slot_mapping[val.idx],
        )
        val.idx += 1
        return val

    iterator = _IteratorState(idx=0, kv_cache=kv_cache)
    iterator = jax.lax.while_loop(cond, body, iterator)
    return iterator.kv_cache


def _write_seq_to_kv_cache(
    key_value: jax.Array,       # [seq_len, 2, num_heads, head_size]
    kv_cache: jax.Array,        # [2, num_heads, num_blocks, block_size, head_size]
    slot_mapping: jax.Array,    # [seq_len]
) -> jax.Array:
    seq_len = slot_mapping.shape[0]
    num_heads, _, block_size, head_size = kv_cache.shape[1:]
    # Reshape to match the rank of kv_cache.
    key_value = key_value.reshape(seq_len, 2, num_heads, 1, 1, head_size)

    def cond(val: _IteratorState):
        return jnp.logical_and(
            val.idx < seq_len, slot_mapping[val.idx] != _PAD_SLOT_ID)

    def body(val: _IteratorState):
        slot_idx = slot_mapping[val.idx]
        val.kv_cache = jax.lax.dynamic_update_slice(
            val.kv_cache,
            key_value[val.idx],
            (0, 0, slot_idx // block_size, slot_idx % block_size, 0),
        )
        val.idx += 1
        return val

    iterator = _IteratorState(idx=0, kv_cache=kv_cache)
    iterator = jax.lax.while_loop(cond, body, iterator)
    return iterator.kv_cache


@chex.dataclass
class _IteratorState:

    idx: jnp.int32
    kv_cache: jnp.ndarray       # [2, num_heads, num_blocks, block_size, head_size]
