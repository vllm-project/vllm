import argparse
import time

import jax
import jax.numpy as jnp


@jax.jit
def attn(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    sm_scale: float,
) -> jax.Array:
    seq_len = k_cache.shape[1]
    attn_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

    q = q * sm_scale
    logits = jnp.einsum('BTNH,BSNH->BTNS', q, k_cache)
    masked_logits = jnp.where((jnp.expand_dims(attn_mask, -2)),
                                logits, -2.3819763e38)
    probs = jax.nn.softmax(masked_logits, axis=-1).astype(k_cache.dtype)
    output = jnp.einsum('BTNS,BSNH->BTNH', probs, v_cache)
    return output


def benchmark_attn(
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    context_len: int,
):
    rng_key = jax.random.PRNGKey(0)
    query = jax.random.normal(rng_key, (batch_size, 1, num_heads, head_size), dtype=jnp.bfloat16)
    k_cache = jax.random.normal(rng_key, (batch_size, context_len, num_kv_heads, head_size), dtype=jnp.bfloat16)
    v_cache = jax.random.normal(rng_key, (batch_size, context_len, num_kv_heads, head_size), dtype=jnp.bfloat16)
    sm_scale = head_size ** -0.5

    # For JIT compilation.
    output = attn(query, k_cache, v_cache, sm_scale)
    output.block_until_ready()

    start = time.time()
    for _ in range(100):
        output = attn(query, k_cache, v_cache, sm_scale)
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

    benchmark_attn(
        args.batch_size,
        args.num_heads,
        args.num_kv_heads,
        args.head_size,
        args.context_len,
    )
