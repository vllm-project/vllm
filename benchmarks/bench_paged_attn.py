import argparse
import functools
import time

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention

BLOCK_SIZE = 16
MAX_NUM_BLOCKS_PER_SEQ = 512


@functools.partial(jax.jit, static_argnums=(6, 7))
def paged_attn(
    q: jax.Array,               # [batch, 1, num_heads, head_size]
    k_cache: jax.Array,         # [num_kv_heads, num_blocks * block_size, head_size]
    v_cache: jax.Array,         # [num_kv_heads, num_blocks * block_size, head_size]
    sm_scale: float,
    block_tables: jax.Array,    # [batch, max_num_blocks_per_batch]
    context_lens: jax.Array,    # [batch]
    block_size: int,
    pages_per_compute_block: int,
) -> jax.Array:                 # [batch, 1, num_heads, head_size]
    q = q.squeeze(1)
    q = q * sm_scale

    head_size = q.shape[-1]
    num_slots = k_cache.shape[-2]
    k_cache = k_cache.reshape(-1, num_slots // block_size, block_size, head_size)
    v_cache = v_cache.reshape(-1, num_slots // block_size, block_size, head_size)

    output = paged_attention(
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        pages_per_compute_block=pages_per_compute_block,
        megacore_mode="kv_head",
    )
    return output.reshape(q.shape[0], 1, q.shape[1], q.shape[2])


def benchmark_paged_attn(
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    context_len: int,
    num_blocks: int,
    block_size: int,
    pages_per_compute_block: int,
):
    rng_key = jax.random.PRNGKey(0)
    query = jax.random.normal(rng_key, (batch_size, 1, num_heads, head_size), dtype=jnp.bfloat16)
    k_cache = jax.random.normal(rng_key, (num_kv_heads, num_blocks * block_size, head_size), dtype=jnp.bfloat16)
    v_cache = jax.random.normal(rng_key, (num_kv_heads, num_blocks * block_size, head_size), dtype=jnp.bfloat16)
    sm_scale = head_size ** -0.5
    block_tables = jax.random.randint(rng_key, (batch_size, MAX_NUM_BLOCKS_PER_SEQ), 0, num_blocks, dtype=jnp.int32)
    context_lens = jnp.array([context_len] * batch_size, dtype=jnp.int32)

    # For JIT compilation.
    output = paged_attn(query, k_cache, v_cache, sm_scale, block_tables, context_lens, block_size, pages_per_compute_block)
    output.block_until_ready()

    start = time.time()
    for _ in range(100):
        output = paged_attn(query, k_cache, v_cache, sm_scale, block_tables, context_lens, block_size, pages_per_compute_block)
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
    parser.add_argument("--num-blocks", type=int, default=2048)
    args = parser.parse_args()
    print(args)

    for block_size in [16, 32, 64, 128]:
        for pages_per_compute_block in [1, 2, 4, 8, 16, 32, 64, 128]:
            if pages_per_compute_block > MAX_NUM_BLOCKS_PER_SEQ:
                continue
            if block_size * pages_per_compute_block > 1024:
                continue
            print(f"block_size {block_size}, pages_per_compute_block: {pages_per_compute_block}")
            benchmark_paged_attn(
                args.batch_size,
                args.num_heads,
                args.num_kv_heads,
                args.head_size,
                args.context_len,
                args.num_blocks,
                block_size,
                pages_per_compute_block,
            )
