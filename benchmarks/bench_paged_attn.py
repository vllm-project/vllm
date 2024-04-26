import argparse
import functools
import time

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention

BLOCK_SIZE = 16
MAX_NUM_BLOCKS_PER_SEQ = 512


@functools.partial(jax.jit, static_argnums=(6,))
def paged_attn(
    q: jax.Array,               # [batch, 1, num_heads, head_size]
    k_cache: jax.Array,         # [num_kv_heads, num_blocks * block_size, head_size]
    v_cache: jax.Array,         # [num_kv_heads, num_blocks * block_size, head_size]
    sm_scale: float,
    block_tables: jax.Array,    # [batch, max_num_blocks_per_batch]
    context_lens: jax.Array,    # [batch]
    pages_per_compute_block: int,
) -> jax.Array:                 # [batch, 1, num_heads, head_size]
    q = q.squeeze(1)
    q = q * sm_scale

    head_size = q.shape[-1]
    num_slots = k_cache.shape[-2]
    k_cache = k_cache.reshape(-1, num_slots // BLOCK_SIZE, BLOCK_SIZE, head_size)
    v_cache = v_cache.reshape(-1, num_slots // BLOCK_SIZE, BLOCK_SIZE, head_size)

    output = paged_attention(
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        pages_per_compute_block=pages_per_compute_block,
    )
    return output.reshape(q.shape[0], 1, q.shape[1], q.shape[2])


def benchmark_paged_attn(
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    context_len: int,
    num_blocks: int,
    pages_per_compute_block: int,
):
    rng_key = jax.random.PRNGKey(0)
    query = jax.random.normal(rng_key, (batch_size, 1, num_heads, head_size), dtype=jnp.bfloat16)
    k_cache = jax.random.normal(rng_key, (num_kv_heads, num_blocks * BLOCK_SIZE, head_size), dtype=jnp.bfloat16)
    v_cache = jax.random.normal(rng_key, (num_kv_heads, num_blocks * BLOCK_SIZE, head_size), dtype=jnp.bfloat16)
    sm_scale = BLOCK_SIZE ** -0.5
    block_tables = jax.random.randint(rng_key, (batch_size, MAX_NUM_BLOCKS_PER_SEQ), 0, num_blocks, dtype=jnp.int32)
    context_lens = jnp.array([context_len] * batch_size, dtype=jnp.int32)

    # For JIT compilation.
    output = paged_attn(query, k_cache, v_cache, sm_scale, block_tables, context_lens, pages_per_compute_block)
    output.block_until_ready()

    start = time.time()
    for _ in range(100):
        output = paged_attn(query, k_cache, v_cache, sm_scale, block_tables, context_lens, pages_per_compute_block)
    output.block_until_ready()
    end = time.time()

    print(f"Time taken: {(end - start) * 10000:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=16)
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--context-len", type=int, default=512)
    args = parser.parse_args()
    print(args)

    for num_blocks in [2048]:
        for pages_per_compute_block in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            if pages_per_compute_block > MAX_NUM_BLOCKS_PER_SEQ:
                continue
            print(f"num_blocks: {num_blocks}, pages_per_compute_block: {pages_per_compute_block}")
            benchmark_paged_attn(
                args.batch_size,
                args.num_heads,
                args.num_kv_heads,
                args.head_size,
                args.context_len,
                num_blocks,
                pages_per_compute_block,
            )
