import functools
import time
from typing import Tuple

import chex
import jax
import jax.numpy as jnp

_PAD_SLOT_ID = -1


@jax.jit
def write_to_kv_cache1(
    key: jax.Array,             # [batch_size, seq_len, num_heads, head_size]
    value: jax.Array,           # [batch_size, seq_len, num_heads, head_size]
    k_cache: jax.Array,         # [num_heads, num_blocks * block_size, head_size]
    v_cache: jax.Array,         # [num_heads, num_blocks * block_size, head_size]
    slot_mapping: jax.Array,    # [batch_size, seq_len]
) -> Tuple[jax.Array, jax.Array]:
    num_heads = key.shape[-2]
    head_size = key.shape[-1]

    key = key.reshape(-1, num_heads, head_size)
    key = key.transpose((1, 0, 2))
    value = value.reshape(-1, num_heads, head_size)
    value = value.transpose((1, 0, 2))

    k_cache = k_cache.at[:, slot_mapping.reshape(-1), :].set(key)
    v_cache = v_cache.at[:, slot_mapping.reshape(-1), :].set(value)
    return k_cache, v_cache


@functools.partial(jax.jit, donate_argnums=(2, 3))
def write_to_kv_cache2(
    key: jax.Array,             # [batch_size, seq_len, num_heads, head_size]
    value: jax.Array,           # [batch_size, seq_len, num_heads, head_size]
    k_cache: jax.Array,         # [num_heads, num_blocks * block_size, head_size]
    v_cache: jax.Array,         # [num_heads, num_blocks * block_size, head_size]
    slot_mapping: jax.Array,    # [batch_size, seq_len]
) -> Tuple[jax.Array, jax.Array]:
    batch_size = slot_mapping.shape[0]

    def cond(val: _IteratorState):
        return val.idx < batch_size

    def body(val: _IteratorState):
        k_cache, v_cache = _write_seq_to_kv_cache(
            key[val.idx],
            value[val.idx],
            val.k_cache,
            val.v_cache,
            slot_mapping[val.idx],
        )
        val.k_cache = k_cache
        val.v_cache = v_cache
        val.idx += 1
        return val

    iterator = _IteratorState(idx=0, k_cache=k_cache, v_cache=v_cache)
    iterator = jax.lax.while_loop(cond, body, iterator)
    return iterator.k_cache, iterator.v_cache


@functools.partial(jax.jit, donate_argnums=(2, 3))
def _write_seq_to_kv_cache(
    key: jax.Array,             # [seq_len, num_heads, head_size]
    value: jax.Array,           # [seq_len, num_heads, head_size]
    k_cache: jax.Array,         # [num_heads, num_blocks * block_size, head_size]
    v_cache: jax.Array,         # [num_heads, num_blocks * block_size, head_size]
    slot_mapping: jax.Array,    # [seq_len]
) -> Tuple[jax.Array, jax.Array]:
    seq_len = slot_mapping.shape[0]
    num_heads, _, head_size = k_cache.shape
    # Reshape to match the rank of kv_cache.
    key = key.reshape(seq_len, num_heads, 1, head_size)
    value = value.reshape(seq_len, num_heads, 1, head_size)

    def cond(val: _IteratorState):
        return jnp.logical_and(
            val.idx < seq_len, slot_mapping[val.idx] != _PAD_SLOT_ID)

    def body(val: _IteratorState):
        slot_idx = slot_mapping[val.idx]
        val.k_cache = jax.lax.dynamic_update_slice(
            val.k_cache,
            key[val.idx],
            (0, slot_idx, 0),
        )
        val.v_cache = jax.lax.dynamic_update_slice(
            val.v_cache,
            value[val.idx],
            (0, slot_idx, 0),
        )
        val.idx += 1
        return val

    iterator = _IteratorState(idx=0, k_cache=k_cache, v_cache=v_cache)
    iterator = jax.lax.while_loop(cond, body, iterator)
    return iterator.k_cache, iterator.v_cache


@chex.dataclass
class _IteratorState:

    idx: jnp.int32
    k_cache: jnp.ndarray       # [num_heads, num_blocks, block_size, head_size]
    v_cache: jnp.ndarray       # [num_heads, num_blocks, block_size, head_size]


def benchmark_write_to_kv_cache(
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_size: int,
    num_blocks: int,
    block_size: int,
    version: int = 1,
):
    if version == 1:
        f = write_to_kv_cache1
    elif version == 2:
        f = write_to_kv_cache2
    else:
        raise ValueError(f"Invalid version: {version}")

    rng_key = jax.random.PRNGKey(0)
    key = jax.random.normal(rng_key, (batch_size, seq_len, num_kv_heads, head_size), dtype=jnp.bfloat16)
    value = jax.random.normal(rng_key, (batch_size, seq_len, num_kv_heads, head_size), dtype=jnp.bfloat16)
    k_cache = jax.random.normal(rng_key, (num_kv_heads, num_blocks * block_size, head_size), dtype=jnp.bfloat16)
    v_cache = jax.random.normal(rng_key, (num_kv_heads, num_blocks * block_size, head_size), dtype=jnp.bfloat16)
    slot_mapping = jax.random.randint(rng_key, (batch_size, seq_len), 0, num_blocks * block_size, dtype=jnp.int32)

    # For JIT compilation.
    k_cache, v_cache = f(key, value, k_cache, v_cache, slot_mapping)
    k_cache.block_until_ready()

    start = time.time()
    for _ in range(100):
        k_cache, v_cache = f(key, value, k_cache, v_cache, slot_mapping)
    k_cache.block_until_ready()
    end = time.time()
    print(f"Time taken: {(end - start) * 10:.2f} ms")


if __name__ == "__main__":
    for num_blocks in [16, 256, 512, 1024, 2048, 8192, 16384]:
        print(f"Benchmarking Write to KV Cache w/ {num_blocks} blocks")
        benchmark_write_to_kv_cache(16, 256, 16, 256, num_blocks, 16, version=1)
