"""
Minimal repro: FlashInfer MLA MQA decode NaN for padding tokens (seqlen=0).

Reproduces fwd_mqa_nan=65536 seen in decode pod logs:
  - 7 tokens total: 3 real (seqlen > 0), 4 padding (seqlen = 0)
  - 4 padding tokens × 128 heads × 128 v_dim = 65536 NaN

The FlashInfer trtllm_batch_decode_with_kv_cache_mla kernel computes
softmax over empty sequences for padding tokens → 0/0 = NaN.
"""

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

# DeepSeek R1 MLA parameters
num_heads = 128
kv_lora_rank = 512
qk_rope_head_dim = 64
qk_nope_head_dim = 128
# Absorbed query dim: kv_lora_rank + qk_rope_head_dim = 576
absorbed_head_dim = kv_lora_rank + qk_rope_head_dim  # 576
# KV cache dim: kv_lora_rank + qk_rope_head_dim = 576
kv_dim = kv_lora_rank + qk_rope_head_dim  # 576

block_size = 64  # FlashInfer MLA requires 32 or 64

# Batch: 7 tokens, 3 real + 4 padding
num_tokens = 7
num_real = 3
num_padding = 4

# Sequence lengths: real tokens have seqlen > 0, padding have seqlen = 0
seq_lens = torch.tensor([100, 50, 200, 0, 0, 0, 0], dtype=torch.int32, device="cuda")
max_seq_len = 200

# Block table: each request needs ceil(max_seq_len / block_size) entries
num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size  # 4
total_blocks = num_tokens * num_blocks_per_seq

# Allocate KV cache: [num_blocks, 1, block_size, kv_dim] (HND layout)
kv_cache = torch.randn(
    total_blocks, 1, block_size, kv_dim,
    dtype=torch.bfloat16, device="cuda"
)

# Block table: [num_tokens, num_blocks_per_seq]
block_table = torch.arange(
    total_blocks, dtype=torch.int32, device="cuda"
).reshape(num_tokens, num_blocks_per_seq)

# Query: [num_tokens, 1 (q_len_per_request), num_heads, absorbed_head_dim]
# In MLA decode, query has been absorbed: q = cat(q_nope @ W_UK_T, q_pe)
q = torch.randn(
    num_tokens, 1, num_heads, absorbed_head_dim,
    dtype=torch.bfloat16, device="cuda"
)

# Workspace buffer
workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

# Scale factors (no FP8 quantization in this test)
bmm1_scale = 1.0 / (absorbed_head_dim ** 0.5)  # standard attention scale
bmm2_scale = 1.0

print(f"q shape: {q.shape}")
print(f"kv_cache shape: {kv_cache.shape}")
print(f"block_table shape: {block_table.shape}")
print(f"seq_lens: {seq_lens}")
print(f"bmm1_scale: {bmm1_scale}")
print(f"bmm2_scale: {bmm2_scale}")
print(f"qk_nope_head_dim: {qk_nope_head_dim}")
print(f"kv_lora_rank: {kv_lora_rank}")
print(f"qk_rope_head_dim: {qk_rope_head_dim}")
print()

# Run FlashInfer MLA decode kernel
o = trtllm_batch_decode_with_kv_cache_mla(
    query=q,
    kv_cache=kv_cache.unsqueeze(1),  # add page dimension
    workspace_buffer=workspace,
    qk_nope_head_dim=qk_nope_head_dim,
    kv_lora_rank=kv_lora_rank,
    qk_rope_head_dim=qk_rope_head_dim,
    block_tables=block_table,
    seq_lens=seq_lens,
    max_seq_len=max_seq_len,
    bmm1_scale=bmm1_scale,
    bmm2_scale=bmm2_scale,
)

print(f"Output shape: {o.shape}")

# Check for NaN
o_flat = o.view(-1, o.shape[-2], o.shape[-1])
nan_total = torch.isnan(o_flat).sum().item()
print(f"Total NaN in output: {nan_total}")

# Per-token NaN count
for i in range(num_tokens):
    token_nan = torch.isnan(o_flat[i]).sum().item()
    label = "REAL" if seq_lens[i] > 0 else "PAD"
    print(f"  token {i} (seqlen={seq_lens[i].item()}, {label}): {token_nan} NaN")

# Expected: padding tokens produce NaN (softmax over empty seq → 0/0)
expected_nan = num_padding * num_heads * kv_lora_rank  # 4 * 128 * 512 = 262144
# Actually output dim is v_head_dim which for absorbed MLA = kv_lora_rank
print(f"\nExpected ~{num_padding} * {num_heads} * v_dim NaN from padding tokens")
print(f"Observed: {nan_total} NaN")

if nan_total > 0:
    print("\n*** REPRODUCED: FlashInfer MLA kernel produces NaN for seqlen=0 ***")

    # Now test the fix: set seqlen=1 for padding tokens
    print("\n--- Testing workaround: seqlen=1 for padding tokens ---")
    seq_lens_fixed = seq_lens.clone()
    seq_lens_fixed[seq_lens_fixed == 0] = 1

    o_fixed = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_table,
        seq_lens=seq_lens_fixed,
        max_seq_len=max_seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )

    nan_fixed = torch.isnan(o_fixed).sum().item()
    print(f"NaN with seqlen=1 fix: {nan_fixed}")
    if nan_fixed == 0:
        print("*** FIX WORKS: seqlen=1 for padding eliminates NaN ***")
    else:
        print(f"*** FIX PARTIAL: still {nan_fixed} NaN ***")
else:
    print("\nNo NaN observed - kernel handles seqlen=0 correctly")
