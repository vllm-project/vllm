"""Exercise the rebuilt QuickReduce on gfx1100 in isolation, before letting it
anywhere near the model.

Checks the result against a reference all-reduce, and a CTRL case that must
fail, so a harness that cannot fail is visible as one that never fails.

Run: torchrun --nproc_per_node=2 test_qr.py
"""

import os

import torch
import torch.distributed as dist

torch.ops.load_library("/tmp/qr_build/libqr_rdna3.so")
qr = torch.ops._qr_rdna3

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank)
dev = f"cuda:{rank}"

MAX = 16 * 1024 * 1024
ptr = qr.init_custom_qr(rank, world, MAX)
handle = qr.qr_get_handle(ptr)
handles = [None] * world
dist.all_gather_object(handles, handle)
qr.qr_open_handles(ptr, handles)
if rank == 0:
    print(f"QuickReduce inicializado, world={world}, max={MAX >> 20} MiB")

# FP=0 sin cuantizar; el resto cuantizan el payload del enlace. Los codecs
# cuantizados usan __shfl sobre grupos de 8 hilos, asi que hay que comprobar
# que no asumen wave64.
REGIMES = [(0, "FP  "), (1, "INT8"), (2, "INT6"), (3, "INT4")]
SIZES = [32 * 1024, 512 * 1024]
ok_all = True

for FP, rname in REGIMES:
  for dtype, name in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
    for nbytes in SIZES:
          n = nbytes // dtype.itemsize
          torch.manual_seed(1234 + rank)
          inp = (torch.randn(n, device=dev, dtype=dtype) * 0.1).contiguous()

          ref = inp.clone()
          dist.all_reduce(ref)  # RCCL como referencia

          out = torch.empty_like(inp)
          qr.qr_all_reduce(ptr, inp, out, FP, False)
          torch.cuda.synchronize()

          diff = (out.float() - ref.float()).abs()
          tol = (2e-2 if dtype is torch.bfloat16 else 5e-3) if FP == 0 else 0.30
          bad = int((diff > tol).sum())
          ok = bad == 0
          ok_all &= ok
          if rank == 0:
              print(
                  f"  {rname} {name} {nbytes // 1024:5d} KiB  {'ok' if ok else 'ROTO'}"
                  f"   max|dif|={diff.max().item():.3e}  fuera_de_tol={bad}/{n}"
              )

# CTRL: el arnes tiene que saber fallar
out = torch.empty(4096, device=dev, dtype=torch.float16)
inp = torch.ones(4096, device=dev, dtype=torch.float16)
qr.qr_all_reduce(ptr, inp, out, FP, False)
torch.cuda.synchronize()
wrong = float(world) + 1.0
if rank == 0:
    print(
        f"  CTRL suma de unos: obtenido={out[0].item()} esperado={float(world)}"
        f"  (un arnes roto diria ok contra {wrong})"
    )
    print("VEREDICTO:", "TODO OK" if ok_all else "HAY FALLOS")

qr.qr_destroy(ptr)
dist.destroy_process_group()
