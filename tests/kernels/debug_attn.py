"""Test: encode K/V → decode → run attention → compare with baseline.

Runs the attention kernel on BOTH:
A) Standard bf16 cache (ground truth)
B) TurboQuant encode→decode→bf16 cache

If attention outputs match, the algorithm + integration are correct.
"""
import math
import torch
from vllm.model_executor.layers.quantization.turboquant import (
    TurboQuantConfig, TurboQuantState,
)
from vllm.v1.attention.ops.triton_hadamard_turboquant import (
    hadamard_turboquant_encode, hadamard_turboquant_decode,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

device = torch.device("cuda")
NUM_TOKENS = 7  # prefill
NUM_Q_HEADS = 28
NUM_KV_HEADS = 4
HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 4  # enough for 7 tokens

torch.manual_seed(42)

# Simulate model outputs
query = torch.randn(NUM_TOKENS, NUM_Q_HEADS, HEAD_SIZE,
                     device=device, dtype=torch.bfloat16)
key = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE,
                   device=device, dtype=torch.bfloat16)
# Test with both extreme and mild RoPE simulation
import sys
if "--mild" in sys.argv:
    # Mild: 3x inflation (realistic RoPE ratio)
    key[..., 0:10] *= 3.0
    key[..., 64:74] *= 2.5
    print("Using MILD RoPE inflation (3x)")
else:
    # Extreme: 20x inflation (worst case)
    key[..., 0:10] *= 20.0
    key[..., 64:74] *= 15.0
    print("Using EXTREME RoPE inflation (20x)")
value = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE,
                     device=device, dtype=torch.bfloat16)

# Attention metadata
cu_seqlens_q = torch.tensor([0, NUM_TOKENS], device=device, dtype=torch.int32)
seq_lens = torch.tensor([NUM_TOKENS], device=device, dtype=torch.int32)
block_table = torch.arange(NUM_BLOCKS, device=device,
                            dtype=torch.int32).unsqueeze(0)

# ======== Path A: Standard bf16 cache (ground truth) ========
bf16_key_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS,
                              HEAD_SIZE, device=device, dtype=torch.bfloat16)
bf16_val_cache = torch.zeros_like(bf16_key_cache)

# Write K/V to cache
for i in range(NUM_TOKENS):
    bi = i // BLOCK_SIZE
    bo = i % BLOCK_SIZE
    bf16_key_cache[bi, bo] = key[i]
    bf16_val_cache[bi, bo] = value[i]

output_a = torch.empty(NUM_TOKENS, NUM_Q_HEADS, HEAD_SIZE,
                         device=device, dtype=torch.bfloat16)
k_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
descale = k_scale.expand(1, NUM_KV_HEADS)

unified_attention(
    q=query, k=bf16_key_cache, v=bf16_val_cache, out=output_a,
    cu_seqlens_q=cu_seqlens_q, max_seqlen_q=NUM_TOKENS,
    seqused_k=seq_lens, max_seqlen_k=NUM_TOKENS,
    softmax_scale=1.0 / math.sqrt(HEAD_SIZE),
    causal=True, window_size=(-1, -1),
    block_table=block_table, softcap=0,
    q_descale=None, k_descale=descale, v_descale=descale,
)
print(f"Path A (bf16): output norm = {output_a.float().norm():.2f}")

# ======== Path B: TurboQuant encode→decode cache ========
cfg = TurboQuantConfig(bit_width=4, outlier_fraction=0.15)
k_state = TurboQuantState(cfg, HEAD_SIZE, layer_idx=0, device=device)
v_state = TurboQuantState(cfg, HEAD_SIZE, layer_idx=10000, device=device)

# Calibrate on actual data
k_state.calibrate_outliers(key.reshape(-1, HEAD_SIZE))
v_state.calibrate_outliers(value.reshape(-1, HEAD_SIZE))

normal_size = k_state.normal_size
n_outliers = HEAD_SIZE - normal_size
packed_bytes = math.ceil(normal_size * 4 / 8)
slot_bytes = n_outliers * 2 + packed_bytes + 2

# Create uint8 cache with K/V separation (dim 1 = 2)
uint8_cache = torch.zeros(NUM_BLOCKS, 2, BLOCK_SIZE, NUM_KV_HEADS,
                           slot_bytes, device=device, dtype=torch.uint8)

# Encode K/V to uint8 cache
for kv_idx, (tensor, state) in enumerate([(key, k_state), (value, v_state)]):
    normal_x = tensor[..., state.normal_idx].contiguous()
    outlier_x = tensor[..., state.outlier_idx]

    indices, norms = hadamard_turboquant_encode(
        normal_x, state.sign_flips, state.codebook, state.boundaries)

    flat_idx = indices.reshape(-1, normal_size)
    N = flat_idx.shape[0]
    if normal_size % 2 != 0:
        flat_idx = torch.nn.functional.pad(flat_idx, (0, 1), value=0)
    packed = flat_idx[:, 0::2] | (flat_idx[:, 1::2] << 4)
    packed = packed[:, :packed_bytes]

    parts = []
    ob = outlier_x.reshape(N, n_outliers).to(torch.bfloat16).view(
        torch.uint8).reshape(N, n_outliers * 2)
    parts.append(ob)
    parts.append(packed)
    norm_bytes = norms.reshape(N).to(torch.float16).view(
        torch.uint8).reshape(N, 2)
    parts.append(norm_bytes)
    slot_data = torch.cat(parts, dim=-1)

    # Write to cache (kv_idx separates K and V)
    cache_kv = uint8_cache[:, kv_idx]
    for i in range(NUM_TOKENS):
        bi = i // BLOCK_SIZE
        bo = i % BLOCK_SIZE
        for h in range(NUM_KV_HEADS):
            cache_kv[bi, bo, h] = slot_data[i * NUM_KV_HEADS + h]

# Decode from uint8 cache
decoded_caches = []
for kv_idx, state in enumerate([k_state, v_state]):
    cache_kv = uint8_cache[:, kv_idx]
    N = NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS
    flat = cache_kv.reshape(N, slot_bytes)

    pos = 0
    outlier_vals = flat[:, :n_outliers*2].clone().view(
        torch.bfloat16).reshape(N, n_outliers)
    pos = n_outliers * 2
    flat_packed = flat[:, pos:pos+packed_bytes]
    pos += packed_bytes
    norms_dec = flat[:, pos:pos+2].clone().view(torch.float16).reshape(N)

    low = flat_packed & 0x0F
    high = (flat_packed >> 4) & 0x0F
    dec_indices = torch.stack([low, high], dim=-1).reshape(
        N, -1)[:, :normal_size]

    normal_dec = hadamard_turboquant_decode(
        dec_indices.reshape(N, 1, normal_size).to(torch.uint8),
        norms_dec.reshape(N, 1),
        state.sign_flips, state.codebook,
        output_dtype=torch.bfloat16,
    ).reshape(N, normal_size)

    full = torch.empty(N, HEAD_SIZE, dtype=torch.bfloat16, device=device)
    full[:, state.normal_idx] = normal_dec
    full[:, state.outlier_idx] = outlier_vals
    decoded_caches.append(full.reshape(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE))

tq_key_cache, tq_val_cache = decoded_caches

# Compare decoded vs original
cos_k = torch.nn.functional.cosine_similarity(
    bf16_key_cache[:1, :NUM_TOKENS].reshape(-1, HEAD_SIZE).float(),
    tq_key_cache[:1, :NUM_TOKENS].reshape(-1, HEAD_SIZE).float(),
    dim=1,
).mean().item()
cos_v = torch.nn.functional.cosine_similarity(
    bf16_val_cache[:1, :NUM_TOKENS].reshape(-1, HEAD_SIZE).float(),
    tq_val_cache[:1, :NUM_TOKENS].reshape(-1, HEAD_SIZE).float(),
    dim=1,
).mean().item()
print(f"Key cosine (original vs decoded):   {cos_k:.6f}")
print(f"Value cosine (original vs decoded): {cos_v:.6f}")

# Run attention on decoded cache
output_b = torch.empty_like(output_a)
unified_attention(
    q=query, k=tq_key_cache, v=tq_val_cache, out=output_b,
    cu_seqlens_q=cu_seqlens_q, max_seqlen_q=NUM_TOKENS,
    seqused_k=seq_lens, max_seqlen_k=NUM_TOKENS,
    softmax_scale=1.0 / math.sqrt(HEAD_SIZE),
    causal=True, window_size=(-1, -1),
    block_table=block_table, softcap=0,
    q_descale=None, k_descale=descale, v_descale=descale,
)
print(f"Path B (TQ):   output norm = {output_b.float().norm():.2f}")

# Compare attention outputs
cos_out = torch.nn.functional.cosine_similarity(
    output_a.reshape(-1, HEAD_SIZE).float(),
    output_b.reshape(-1, HEAD_SIZE).float(),
    dim=1,
).mean().item()
max_diff = (output_a.float() - output_b.float()).abs().max().item()
print(f"\nAttention output cosine: {cos_out:.6f}")
print(f"Attention output max diff: {max_diff:.4f}")
print(f"VERDICT: {'PASS' if cos_out > 0.95 else 'FAIL'}")
