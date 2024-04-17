# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Gemma transformer."""
import jax
import jax.numpy as jnp
from flax import linen as nn
from transformers import GemmaConfig

from vllm.model_executor.models.jax.ops.flash_attn import flash_attn
from vllm.model_executor.models.jax.ops.paged_attn import paged_attn
from vllm.model_executor.models.jax.ops.write_to_cache import write_to_cache

K_MASK = -2.3819763e38  # Set to a large negative number.


class Einsum(nn.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""
  shape: tuple[int, ...]

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    w = self.param('w', nn.initializers.normal(), self.shape)
    return jnp.einsum(eqn, x, w)


class RMSNorm(nn.Module):
  """RMSNorm layer."""

  @nn.compact
  def __call__(self, x):
    scale = self.param('scale', nn.initializers.zeros_init(), (x.shape[-1]))
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
    # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
    # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
    # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
    scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
    normed_inputs = normed_inputs * (1 + scale)
    return normed_inputs


def apply_rope(
    inputs: jax.Array,    # [B, L]
    positions: jax.Array, # [B, L]
    head_dim: int,
    max_wavelength: int = 10_000,
) -> jax.Array:
  """Applies RoPE."""
  fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
  timescale = max_wavelength**fraction

  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)


class Embedder(nn.Module):
  """Embedder module."""

  vocab_size: int
  embed_dim: int

  def setup(self):
    self.input_embedding_table = self.param(
        'input_embedding',
        nn.initializers.normal(),
        (self.vocab_size, self.embed_dim),
    )

  def encode(self, x: jax.Array) -> jax.Array:
    x = self.input_embedding_table[(x,)]
    x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
    return x

  def decode(self, x: jax.Array) -> jax.Array:
    return jnp.dot(x, self.input_embedding_table.T)


class Attention(nn.Module):
  """Attention module."""

  num_heads: int
  num_kv_heads: int
  features: int
  head_dim: int

  @property
  def use_qkv_einsum(self):
    return self.num_kv_heads == self.num_heads

  def setup(self):
    self.attn_vec_einsum = Einsum(
        shape=(self.num_heads, self.head_dim, self.features),
    )

    if self.use_qkv_einsum:
      self.qkv_einsum = Einsum(
          shape=(3, self.num_heads, self.features, self.head_dim),
      )
    else:
      self.q_einsum = Einsum(
          shape=(self.num_heads, self.features, self.head_dim),
      )
      self.kv_einsum = Einsum(
          shape=(2, self.num_kv_heads, self.features, self.head_dim),
      )
    self.sm_scale = self.head_dim**-0.5

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      slot_mapping: jax.Array,
      block_tables: jax.Array | None,
      context_lens: jax.Array | None,
      cache: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    if self.use_qkv_einsum:
      query_proj, key_proj, value_proj = self.qkv_einsum('BTD,SNDH->SBTNH', x)
    else:
      query_proj = self.q_einsum('BTD,NDH->BTNH', x)
      key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)

    query_proj = apply_rope(
        query_proj,
        segment_pos,
        head_dim=self.head_dim,
    )
    key_proj = apply_rope(
        key_proj,
        segment_pos,
        head_dim=self.head_dim,
    )

    # Write the incoming keys and values to KV cache.
    key_cache = cache[0]
    value_cache = cache[1]
    key_cache = write_to_cache(key_proj, key_cache, slot_mapping)
    value_cache = write_to_cache(value_proj, value_cache, slot_mapping)
    cache = jnp.stack([key_cache, value_cache])

    if block_tables is None:
      # Prompt attention.
      if not self.use_qkv_einsum:
        # MQA/GQA.
        value_proj = jnp.repeat(value_proj, self.num_heads, axis=-2)
        key_proj = jnp.repeat(key_proj, self.num_heads, axis=-2)

      if False:
        # FIXME(woosuk)
        output = flash_attn(
            query_proj,
            key_proj,
            value_proj,
            self.sm_scale,
        )
      else:
        seq_len = query_proj.shape[1]
        attn_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

        query_scaled = query_proj * self.sm_scale
        logits = jnp.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)
        masked_logits = jnp.where(
            (jnp.expand_dims(attn_mask, -2)), logits, K_MASK
        )
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(key_proj.dtype)
        output = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)
    else:
      # Decode attention.
      output = paged_attn(
          query_proj,
          cache[0],
          cache[1],
          block_tables,
          context_lens,
      )

    attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', output)
    return cache, attn_output


class FeedForward(nn.Module):
  """Feed forward module."""

  features: int
  hidden_dim: int

  @nn.compact
  def __call__(self, x):
    w_gating = self.param(
        'gating_einsum',
        nn.initializers.zeros_init(),
        ((2, self.features, self.hidden_dim)),
    )
    ff_gate = jnp.dot(x, w_gating[0])
    gate_value = nn.gelu(ff_gate)

    ff1 = jnp.dot(x, w_gating[1])
    activations = gate_value * ff1

    w_linear = self.param(
        'linear',
        nn.initializers.zeros_init(),
        (self.hidden_dim, self.features),
    )
    outputs = jnp.dot(activations, w_linear)
    return outputs


class Block(nn.Module):
  """Transformer block."""

  num_heads: int
  num_kv_heads: int
  embed_dim: int
  head_dim: int
  hidden_dim: int

  def setup(self):
    self.pre_attention_norm = RMSNorm()
    self.attn = Attention(
        num_heads=self.num_heads,
        features=self.embed_dim,
        head_dim=self.head_dim,
        num_kv_heads=self.num_kv_heads,
    )
    self.pre_ffw_norm = RMSNorm()
    self.mlp = FeedForward(features=self.embed_dim, hidden_dim=self.hidden_dim)

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      slot_mapping: jax.Array,
      block_tables: jax.Array | None,
      context_lens: jax.Array | None,
      cache: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    inputs_normalized = self.pre_attention_norm(x)
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        slot_mapping,
        block_tables,
        context_lens,
        cache,
    )
    attn_output += x
    residual = attn_output
    attn_output = self.pre_ffw_norm(attn_output)
    outputs = self.mlp(attn_output)
    outputs = residual + outputs
    return outputs, cache


class Transformer(nn.Module):
  """Gemma transformer."""

  config: GemmaConfig

  def setup(self):
    self.embedder = Embedder(
        vocab_size=256128,  # != self.config.vocab_size
        embed_dim=self.config.hidden_size,
    )
    self.blocks = [
        Block(
            name=f'layer_{i}',
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            embed_dim=self.config.hidden_size,
            head_dim=self.config.head_dim,
            hidden_dim=self.config.intermediate_size,
        )
        for i in range(self.config.num_hidden_layers)
    ]
    self.final_norm = RMSNorm()

  def __call__(
      self,
      token_ids: jax.Array,
      positions: jax.Array,
      slot_mapping: jax.Array,
      block_tables: jax.Array | None,
      context_lens: jax.Array | None,
      kv_caches: list[jax.Array],
      logits_indices: jax.Array,
  ) -> tuple[jax.Array, list[jax.Array]]:
    x = self.embedder.encode(token_ids)
    for i, block in enumerate(self.blocks):
      layer_cache = kv_caches[i]
      x, layer_cache = block(
          x,
          positions,
          slot_mapping,
          block_tables,
          context_lens,
          layer_cache,
      )
      kv_caches[i] = layer_cache
    x = self.final_norm(x)
    x = x.reshape(-1, x.shape[-1])
    hidden_states = x[logits_indices]
    logits = self.embedder.decode(hidden_states)
    return logits, kv_caches
