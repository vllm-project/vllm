"""Grid-dimension regression for int4_kivi_gather_dequant.

The gather/dequant grid carries the sequence-position axis.  CUDA limits grid
dimensions y and z to 65535, but x to ~2**31.  The position axis must therefore
ride on grid.x — otherwise any gather with ``max_seq > 65535`` (i.e.
``max_model_len`` past 64k) fails to launch with CUDA "invalid argument".

This test forces ``max_seq`` past 65535 with a tiny real ``seq_len`` (so only a
handful of positions do work) and checks that the launch succeeds and the live
positions round-trip.

Run as a script:
  CUDA_HOME=/usr/local/cuda-12.8 .venv-vllm/bin/python \
    vllm/tests/v1/attention/test_int4_kivi_grid.py
"""
import torch
from vllm.v1.attention.ops.triton_int4_kivi import (
    int4_kivi_store, int4_kivi_gather_dequant, BLOCK,
)

dev = "cuda"
torch.manual_seed(0)
H, D = 8, 128
block_size = 16
full_dim = D // 2 + D // BLOCK

L = 64  # real tokens: 4 full blocks
MAX_SEQ = 70000  # > 65535 -> would overflow grid.y under the old (B, max_seq, H)
assert MAX_SEQ > 65535

nblk = L // block_size
num_blocks = nblk + 2
kv_cache = torch.zeros((num_blocks, 2, block_size, H, full_dim),
                       dtype=torch.uint8, device=dev)
block_table = torch.full((1, nblk + 2), -1, dtype=torch.int32, device=dev)
block_table[0, :nblk] = torch.arange(nblk, dtype=torch.int32, device=dev)
seq_lens = torch.tensor([L], dtype=torch.int32, device=dev)
slots = torch.arange(L, dtype=torch.int64, device=dev)
key = torch.randn(L, H, D, dtype=torch.bfloat16, device=dev)
value = torch.randn(L, H, D, dtype=torch.bfloat16, device=dev)

int4_kivi_store(key, value, kv_cache, slot_mapping=slots, head_size=D)
k_out, v_out = int4_kivi_gather_dequant(kv_cache, block_table, seq_lens,
                                        D, H, max_seq=MAX_SEQ)
torch.cuda.synchronize()  # surface any async launch failure here


def relrmse(got, ref):
    got = got.float(); ref = ref.float()
    return (got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()


assert k_out.shape[2] == MAX_SEQ
k_rel = relrmse(k_out[0, :, :L], key.permute(1, 0, 2))
v_rel = relrmse(v_out[0, :, :L], value.permute(1, 0, 2))
print(f"max_seq={MAX_SEQ}  K relRMSE={k_rel:.4f}  V relRMSE={v_rel:.4f}")
assert k_rel < 0.10 and v_rel < 0.10, "roundtrip broke at large max_seq"
# positions past seq_len must stay zero (kernel early-returns)
assert k_out[0, :, L:].abs().sum().item() == 0.0
assert v_out[0, :, L:].abs().sum().item() == 0.0
print("GRID OK: position axis on grid.x; gather correct beyond the 65535 limit")
