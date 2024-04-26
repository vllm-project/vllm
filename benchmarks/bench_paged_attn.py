import time
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention

BLOCK_SIZE = 16

@jax.jit
def paged_attn(
    q: jax.Array,               # [batch, 1, num_heads, head_size]
    k_cache: jax.Array,         # [num_kv_heads, num_blocks * block_size, head_size]
    v_cache: jax.Array,         # [num_kv_heads, num_blocks * block_size, head_size]
    sm_scale: float,
    block_tables: jax.Array,    # [batch, max_num_blocks_per_batch]
    context_lens: jax.Array,    # [batch]
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
        pages_per_compute_block=4,  # TODO(woosuk): Tune this value.
    )
    return output.reshape(q.shape[0], 1, q.shape[1], q.shape[2])


def benchmark_paged_attn(
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    context_len: int,
    num_blocks: int,
):
    rng_key = jax.random.PRNGKey(0)
    query = jax.random.normal(rng_key, (batch_size, 1, num_heads, head_size), dtype=jnp.bfloat16)
    k_cache = jax.random.normal(rng_key, (num_kv_heads, num_blocks * BLOCK_SIZE, head_size), dtype=jnp.bfloat16)
    v_cache = jax.random.normal(rng_key, (num_kv_heads, num_blocks * BLOCK_SIZE, head_size), dtype=jnp.bfloat16)
    sm_scale = BLOCK_SIZE ** -0.5
    block_tables = jax.random.randint(rng_key, (batch_size, context_len // BLOCK_SIZE), 0, num_blocks, dtype=jnp.int32)
    context_lens = jnp.array([context_len] * batch_size, dtype=jnp.int32)

    # For JIT compilation.
    output = paged_attn(query, k_cache, v_cache, sm_scale, block_tables, context_lens)
    output.block_until_ready()

    start = time.time()
    for _ in range(100):
        output = paged_attn(query, k_cache, v_cache, sm_scale, block_tables, context_lens)
    output.block_until_ready()
    end = time.time()

    print(f"Time taken: {(end - start) * 10:.2f} ms")


if __name__ == "__main__":

    for num_blocks in [16, 256, 512, 2048]:
        print(f"Benchmarking Paged Attention w/ {num_blocks} blocks")
        benchmark_paged_attn(1, 16, 16, 256, 128, num_blocks)

    # BUG: This will raise the following error:
    # jaxlib.xla_extension.XlaRuntimeError: INTERNAL: Program or fatal error occurred; computation may be invalid:
    # INTERNAL: Accelerator device halted prematurely, perhaps due to an on-device check-failure.
    # Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:1:0xad3 (from TensorCoreSequencer:1:0xad4):
    # no debugging message found for this tag:pc. HLO: custom-call.2; HLO computation: main.55
    num_blocks = 1024
    print(f"Benchmarking Paged Attention w/ {num_blocks} blocks")
    benchmark_paged_attn(1, 16, 16, 256, 128, num_blocks)
