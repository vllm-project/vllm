# MLA Architecture in vLLM

This document describes how Multi-Head Latent Attention (MLA) is implemented in vLLM, specifically how `DeepseekV2MLAAttention` uses `MultiHeadLatentAttentionWrapper` and how `MultiHeadLatentAttentionWrapper` uses `MLACommonImpl`.

## 1. DeepseekV2MLAAttention → MultiHeadLatentAttentionWrapper

**Location:** `vllm/model_executor/models/deepseek_v2.py:924-1106`

`DeepseekV2MLAAttention` creates and uses `MultiHeadLatentAttentionWrapper` as follows:

### Initialization (lines 933-1098)

- Creates all the projection layers (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`, etc.)
- Creates rotary embeddings
- For DeepSeek V3.2 sparse attention (`is_v32`), also creates an `Indexer` for sparse attention
- Bundles all modules into an `MLAModules` dataclass (lines 1065-1083)
- Creates `MultiHeadLatentAttentionWrapper` with these modules (lines 1085-1098):

```python
self.mla_attn = MultiHeadLatentAttentionWrapper(
    self.hidden_size,
    self.num_local_heads,
    self.scaling,
    self.qk_nope_head_dim,
    self.qk_rope_head_dim,
    self.v_head_dim,
    self.q_lora_rank,
    self.kv_lora_rank,
    mla_modules,  # Contains all the projections and modules
    cache_config,
    quant_config,
    prefix,
)
```

### Forward pass (lines 1100-1106)

Simply delegates to `mla_attn`:

```python
def forward(self, positions, hidden_states, llama_4_scaling):
    return self.mla_attn(positions, hidden_states, llama_4_scaling)
```

## 2. MultiHeadLatentAttentionWrapper → MLAAttention → MLACommonImpl

**Location:** `vllm/model_executor/layers/mla.py:32-177`

`MultiHeadLatentAttentionWrapper` is a `CustomOp` that:

### Stores all MLA modules (lines 74-90)

- `fused_qkv_a_proj`, `kv_a_proj_with_mqa`, `q_a_layernorm`, `q_b_proj`, `q_proj`
- `kv_a_layernorm`, `kv_b_proj`, `rotary_emb`, `o_proj`
- `indexer` (for sparse attention)

### Creates MLAAttention (lines 92-106)

```python
self.mla_attn = MLAAttention(
    num_heads=self.num_heads,
    scale=scale,
    qk_nope_head_dim=self.qk_nope_head_dim,
    qk_rope_head_dim=self.qk_rope_head_dim,
    v_head_dim=self.v_head_dim,
    q_lora_rank=self.q_lora_rank,
    kv_lora_rank=self.kv_lora_rank,
    cache_config=cache_config,
    quant_config=quant_config,
    prefix=f"{prefix}.attn",
    kv_b_proj=self.kv_b_proj,
    use_sparse=self.is_sparse,
    indexer=self.indexer,
)
```

### forward_native method (lines 110-173)

Does the MLA preprocessing:

1. Computes Q via `fused_qkv_a_proj` → `q_a_layernorm` → `q_b_proj` (if `q_lora_rank` is not None)
2. Splits `kv_lora` into `kv_c` and `k_pe`
3. Normalizes `kv_c` with `kv_a_layernorm`
4. Applies rotary embeddings to Q and K position embeddings
5. Runs the indexer for sparse attention (if enabled)
6. Calls `mla_attn.forward()` with preprocessed tensors
7. Applies output projection with `o_proj`

```python
def forward_native(self, positions, hidden_states, llama_4_scaling=None):
    # 1. Project hidden states to Q and KV latents
    if self.q_lora_rank is not None:
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_lora = qkv_lora.split([self.q_lora_rank, ...], dim=-1)
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0]
    else:
        kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        q = self.q_proj(hidden_states)[0]

    # 2. Split kv_lora into compressed KV and position embeddings
    kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_c_normed = self.kv_a_layernorm(kv_c)

    # 3. Apply RoPE to Q and K
    q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(positions, q[..., self.qk_nope_head_dim:], k_pe)

    # 4. Run indexer for sparse attention (if V3.2)
    if self.indexer and self.is_sparse:
        _topk_indices = self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

    # 5. Call the attention layer
    attn_out = self.mla_attn(q, kv_c_normed, k_pe, output_shape=(...))

    # 6. Output projection
    return self.o_proj(attn_out)[0]
```

## 3. MLAAttention → MLACommonImpl

**Location:** `vllm/attention/layer.py:409-624`

`MLAAttention` is an attention layer that:

### Gets the MLA backend (lines 463-470)

Via `get_attn_backend()`.

### Creates MLACommonImpl as self.impl (lines 488-510)

```python
impl_cls = cast(type[MLAAttentionImpl], self.attn_backend.get_impl_cls())
self.impl = impl_cls(
    num_heads=self.num_heads,
    head_size=self.head_size,  # = kv_lora_rank + qk_rope_head_dim
    scale=self.scale,
    num_kv_heads=1,
    # MLA-specific args
    q_lora_rank=self.q_lora_rank,
    kv_lora_rank=self.kv_lora_rank,
    qk_nope_head_dim=self.qk_nope_head_dim,
    qk_rope_head_dim=self.qk_rope_head_dim,
    qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
    v_head_dim=self.v_head_dim,
    kv_b_proj=kv_b_proj,
    indexer=indexer,
)
```

### forward method (lines 533-583)

Calls `self.impl.forward()`:

```python
def forward(self, q, kv_c_normed, k_pe, output_shape=None):
    self.impl.forward(
        self,           # attention layer (for accessing scales, kv_cache, etc.)
        q,              # query tensor [B, N, qk_head_dim]
        kv_c_normed,    # normalized compressed KV [B, kv_lora_rank]
        k_pe,           # position embeddings for K [B, 1, qk_rope_head_dim]
        self_kv_cache,  # KV cache
        attn_metadata,  # metadata for the attention
        output=output,  # pre-allocated output buffer
    )
    return output
```

## 4. MLACommonImpl Forward Logic

**Location:** `vllm/v1/attention/backends/mla/common.py:1282-2126`

`MLACommonImpl.forward()` (lines 1938-2125) handles both prefill and decode:

### Cache the latent KV (lines 2007-2015)

```python
ops.concat_and_cache_mla(k_c_normed, k_pe.squeeze(1), kv_cache, slot_mapping, ...)
```

### Prefill path - `_forward_prefill()` (lines 1864-1927)

- Projects `kv_c_normed` through `kv_b_proj` to get `k_nope` and `v`
- Concatenates `k_nope` with `k_pe` to form full K
- Runs attention using FlashAttention/FlashInfer/cuDNN (compute-friendly approach)
- Handles chunked context for long sequences

### Decode path - `_forward_decode()` (abstract, implemented by subclasses)

- Uses data-movement friendly approach (MQA in latent space)
- Projects Q through `W_UK_T` to get `ql_nope`
- Runs decode attention on latent KV cache
- Projects attention output through `W_UV` (via `_v_up_proj`)

## Data Flow Summary

```
hidden_states
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ DeepseekV2MLAAttention                                      │
│   ├─ Creates projection layers (q_a_proj, kv_b_proj, etc.) │
│   └─ Creates MultiHeadLatentAttentionWrapper                │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ MultiHeadLatentAttentionWrapper.forward_native()            │
│   ├─ Q projection: h → q_c → q (via q_a_proj, q_b_proj)    │
│   ├─ KV projection: h → (kv_c, k_pe)                        │
│   ├─ RoPE on q_pe and k_pe                                  │
│   ├─ Sparse indexer (if V3.2)                               │
│   ├─ MLAAttention.forward(q, kv_c_normed, k_pe)            │
│   └─ Output projection: o_proj(attn_out)                    │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ MLAAttention.forward()                                      │
│   └─ impl.forward(layer, q, kv_c_normed, k_pe, cache, ...)  │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ MLACommonImpl.forward()                                     │
│   ├─ concat_and_cache_mla() - store to KV cache            │
│   ├─ Prefill: kv_b_proj → (k_nope, v), FlashAttn           │
│   └─ Decode: q @ W_UK_T → ql_nope, MQA, v_up_proj          │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
output
```

## MLA Dimension Reference (from DeepSeek V3)

| Symbol | Description | Value in DSV3 |
|--------|-------------|---------------|
| H | Hidden size | - |
| N | Number of attention heads | - |
| Lq | Latent dimension for Q | 1536 |
| Lkv | Latent dimension for K/V | 512 |
| P | Nope dimension (no rope) | 128 |
| R | Rope dimension | 64 |
| V | V head dim | 128 |

## Key Weight Matrices

| Matrix | Shape | Description |
|--------|-------|-------------|
| W_DQ | [H, Lq] | Project h_t to q_c |
| W_UQ | [Lq, N * P] | Project q_c to q_nope |
| W_QR | [Lq, N * R] | Project q_c to q_pe |
| W_DKV | [H, Lkv] | Project h_t to kv_c |
| W_UK | [Lkv, N, P] | Project kv_c to k_nope |
| W_KR | [H, R] | Project h_t to k_pe |
| W_UV | [Lkv, N, V] | Project kv_c to v |
| W_O | [N * V, H] | Project v to h_t |

## Two Computational Approaches

### Compute-Friendly (Prefill)

Used when Sq/Skv ratio is small (near 1):

```
q_nope = (q_c @ W_UQ).view(Sq, N, P)
q_pe = RoPE(q_c @ W_QR).view(Sq, N, R)
k_nope = (kv_c @ W_UK).view(Skv, N, P)
v = (kv_c @ W_UV).view(Skv, N, V)

# MHA with QK headdim = P + R, V headdim = V
spda_o = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([k_nope, k_pe.expand(-1, N, -1)], dim=-1),
    v
)
```

### Data-Movement Friendly (Decode)

Used when Sq/Skv ratio is large:

```
ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)
q_pe = RoPE(q_c @ W_QR).view(Sq, N, R)

# MQA with QK headdim = Lkv + R, V headdim = Lkv
spda_o = scaled_dot_product_attention(
    torch.cat([ql_nope, q_pe], dim=-1),
    torch.cat([kv_c, k_pe], dim=-1),
    kv_c
)

o = einsum("snl,lnv->snv", spda_o, W_UV)
```

The decode approach is less compute-friendly (Lkv > P) but more data-movement friendly (MQA vs MHA).
