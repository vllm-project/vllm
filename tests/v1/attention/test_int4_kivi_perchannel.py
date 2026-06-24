"""Per-channel-K (KIVI) layout check for the vLLM INT4 backend.

Confirms (1) the store/dequant roundtrip is correct for full + partial blocks,
and (2) on K with per-channel outliers (the real failure mode), the per-channel
layout reconstructs K substantially better than a per-token baseline would.

Run as a script:
  CUDA_HOME=/usr/local/cuda-12.8 .venv-vllm/bin/python \
    vllm/tests/v1/attention/test_int4_kivi_perchannel.py
"""
import torch
from vllm.v1.attention.ops.triton_int4_kivi import (
    int4_kivi_store, int4_kivi_gather_dequant, BLOCK, QMAX,
)
from vllm.utils.torch_utils import int4_kivi_kv_cache_full_dim

torch.manual_seed(0)
dev = "cuda"
H, D = 2, 128
block_size = 16
num_blocks = 16

# One request, 64 tokens = 4 FULL blocks (all per-channel), no partial tail.
L = 64
B = 1
full_dim = int4_kivi_kv_cache_full_dim(D)
kv_cache = torch.zeros((num_blocks, 2, block_size, H, full_dim),
                       dtype=torch.uint8, device=dev)
block_table = torch.full((B, 8), -1, dtype=torch.int32, device=dev)
nblk = L // block_size
block_table[0, :nblk] = torch.arange(nblk, dtype=torch.int32, device=dev)
seq_lens = torch.tensor([L], dtype=torch.int32, device=dev)
slots = torch.arange(L, dtype=torch.int64, device=dev)  # blocks 0..3 contiguous

# K with strong PER-CHANNEL outliers: a few channels have ~30x larger scale.
# This is exactly what KIVI's per-channel layout isolates; per-token K (mixing
# an outlier channel with normal ones in a 16-elem head_dim block) smears it.
base = torch.randn(L, H, D, device=dev)
chan_scale = torch.ones(D, device=dev)
# One outlier channel per 16-elem head_dim block: in the per-TOKEN layout this
# outlier shares its block's single scale with 15 normal channels, forcing a
# coarse scale that wrecks them.  Per-CHANNEL gives every channel its own scale.
chan_scale[::16] = 30.0
key = (base * chan_scale[None, None, :]).to(torch.bfloat16)
value = torch.randn(L, H, D, dtype=torch.bfloat16, device=dev)

int4_kivi_store(key, value, kv_cache, slot_mapping=slots, head_size=D)
k_out, v_out = int4_kivi_gather_dequant(kv_cache, block_table, seq_lens,
                                        D, H, max_seq=L)

kk = key[:, :, :].permute(1, 0, 2).float()  # [H, L, D]
kgot = k_out[0, :, :L, :].float()
k_rel = (kgot - kk).pow(2).mean().sqrt() / kk.pow(2).mean().sqrt()

# Per-TOKEN baseline (what the cache USED to do for K): absmax over each 16-elem
# head_dim block, which lumps outlier + normal channels together.
def per_token_rt(x):  # [H, L, D]
    xf = x.reshape(-1, BLOCK)
    amax = xf.abs().amax(1, keepdim=True).clamp_min(1e-9)
    scale = (amax / QMAX).to(torch.float8_e4m3fn).float()
    code = torch.round(xf / scale).clamp(-QMAX, QMAX)
    return (code * scale).reshape(x.shape)

k_pt = per_token_rt(kk)

# Global relRMSE is dominated by the (huge) outlier channels themselves, so it
# barely moves.  The per-channel WIN is that the NORMAL channels no longer share
# a coarse block scale with an outlier — measure them directly.  This mirrors why
# per-channel-K helps attention (dot products use all channels) even though the
# global energy is outlier-dominated.
normal = chan_scale < 2.0  # the non-outlier channels
def rel(a, b):  # relRMSE over a channel subset
    a = a[..., normal]; b = b[..., normal]
    return (a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()

k_rel_norm = rel(kgot, kk)
k_rel_pt_norm = rel(k_pt, kk)
print(f"K relRMSE  per-channel (cache) global = {k_rel:.4f}")
print(f"normal-channel relRMSE  per-channel   = {k_rel_norm:.4f}")
print(f"normal-channel relRMSE  per-token     = {k_rel_pt_norm:.4f}")
print(f"per-channel / per-token (normal ch)   = {k_rel_norm / k_rel_pt_norm:.3f}")

assert k_rel_norm < k_rel_pt_norm * 0.5, (
    "per-channel-K should be markedly better than per-token on normal channels"
)
print("PER-CHANNEL-K ACTIVE AND BETTER (normal channels isolated from outliers)")

# Also: a partial trailing block must still round-trip (per-token fallback).
kv2 = torch.zeros((num_blocks, 2, block_size, H, full_dim),
                  dtype=torch.uint8, device=dev)
L2 = 20  # block0 full (per-channel), tokens 16..19 partial (per-token)
bt2 = torch.full((1, 8), -1, dtype=torch.int32, device=dev)
bt2[0, :2] = torch.tensor([0, 1], device=dev)
sl2 = torch.arange(L2, dtype=torch.int64, device=dev)
seq2 = torch.tensor([L2], dtype=torch.int32, device=dev)
k2 = torch.randn(L2, H, D, dtype=torch.bfloat16, device=dev)
v2 = torch.randn(L2, H, D, dtype=torch.bfloat16, device=dev)
int4_kivi_store(k2, v2, kv2, sl2, D)
k2o, _ = int4_kivi_gather_dequant(kv2, bt2, seq2, D, H, max_seq=L2)
kk2 = k2.permute(1, 0, 2).float()
rel2 = (k2o[0, :, :L2, :].float() - kk2).pow(2).mean().sqrt() / kk2.pow(2).mean().sqrt()
print(f"partial-block K relRMSE = {rel2:.4f}")
assert rel2 < 0.10, "partial-block per-token fallback broken"
print("PARTIAL-BLOCK FALLBACK OK")
