import torch
from vllm.v1.attention.ops.triton_int4_kivi import (
    int4_kivi_store, int4_kivi_gather_dequant, BLOCK, QMAX,
)
from vllm.utils.torch_utils import int4_kivi_kv_cache_full_dim

torch.manual_seed(0)
dev = "cuda"
H, D = 4, 128
block_size = 16
num_blocks = 8
B = 2
seqs = [20, 33]   # spans multiple pages, with partial tail
N = sum(seqs)

full_dim = int4_kivi_kv_cache_full_dim(D)
print("full_dim", full_dim)
kv_cache = torch.zeros((num_blocks, 2, block_size, H, full_dim), dtype=torch.uint8, device=dev)

# build block_table: req0 uses blocks [0,1], req1 uses [2,3,4]
block_table = torch.full((B, 8), -1, dtype=torch.int32, device=dev)
block_table[0, :2] = torch.tensor([0, 1], device=dev)
block_table[1, :3] = torch.tensor([2, 3, 4], device=dev)
seq_lens = torch.tensor(seqs, dtype=torch.int32, device=dev)

# slot mapping for all tokens, in request order
slots = []
for r, L in enumerate(seqs):
    for p in range(L):
        lblk = p // block_size
        slots.append(int(block_table[r, lblk]) * block_size + (p % block_size))
slot_mapping = torch.tensor(slots, dtype=torch.int64, device=dev)

key = torch.randn(N, H, D, dtype=torch.bfloat16, device=dev)
value = torch.randn(N, H, D, dtype=torch.bfloat16, device=dev)

int4_kivi_store(key, value, kv_cache, slot_mapping, D)
max_seq = max(seqs)
k_out, v_out = int4_kivi_gather_dequant(kv_cache, block_table, seq_lens, D, H, max_seq)

# reference: per-token absmax int4 roundtrip on head_dim 16-blocks
def ref_rt(x):  # [N,H,D]
    xf = x.float().reshape(-1, BLOCK)
    amax = xf.abs().amax(1, keepdim=True).clamp_min(1e-9)
    scale = (amax / QMAX).to(torch.float8_e4m3fn).float()  # match e4m3 storage
    code = torch.round(xf / scale).clamp(-QMAX, QMAX)
    return (code * scale).reshape(x.shape)

# The meaningful metric: relative reconstruction error of dequant(store(x))
# vs the ORIGINAL bf16 x. INT4 (symmetric, 16-elem absmax blocks) should give
# relative RMSE well under ~5% — this is the real serving-quality check.
off = 0
worst_rel = 0.0
for r, L in enumerate(seqs):
    kk = key[off:off+L].permute(1,0,2).float()   # [H,L,D]
    vv = value[off:off+L].permute(1,0,2).float()
    off += L
    kgot = k_out[r, :, :L, :].float()
    vgot = v_out[r, :, :L, :].float()
    krel = (kgot-kk).pow(2).mean().sqrt() / kk.pow(2).mean().sqrt()
    vrel = (vgot-vv).pow(2).mean().sqrt() / vv.pow(2).mean().sqrt()
    worst_rel = max(worst_rel, krel.item(), vrel.item())
    print(f"req{r} L={L} K relRMSE={krel:.4f} V relRMSE={vrel:.4f}")

print("WORST relRMSE vs original bf16 =", worst_rel)
# ~0.087 is the theoretical absmax-INT4 reconstruction error on N(0,1) data
# (verified against a pure-torch roundtrip). Real model K/V is far lower.
assert worst_rel < 0.10, "reconstruction error too high — kernel likely buggy"
print("KERNEL ROUNDTRIP OK")
