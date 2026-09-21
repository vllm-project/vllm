# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""⛔ RETIRADO el 21-sep-2026: MIDE BIEN EN MICROBANCO Y CUESTA CARO DESPLEGADO.

Las cifras de abajo (3,7x y 2,0x) son microbancos en UNA GPU con las formas de
produccion, y son ciertas. Medido e2e a TP4 con concurrencia real, este parche junto
con el override del rocBLAS cuesta:

    6 sesiones (prompts de 4,3k):  44,3 ms sin ellos  ->  56,7 ms con ellos   (-22%)
    4 sesiones:                    34,4 ms            ->  47,1 ms             (-27%)

y ademas deja UNA GPU al 100% de uso y 120-123 W CON EL SERVIDOR EN REPOSO (tres
arranques seguidos; al quitarlos, las cuatro a 0% y 6-13 W). Eso explica ademas por que
"el paquete completo hunde el decode en ainode1 y en ainode2 no": cuesta en las dos
cajas, y con UNA sola sesion no se ve.

No esta desplegado en ninguna caja. Si alguien lo vuelve a encender, que mida el PASO
con 4 y 6 sesiones, no un kernel suelto.

"""

"""Triton GEMV for the fp16 linears rocBLAS serves badly on gfx1100.

The Qwen3.5 recipe leaves two linears unquantized, and both land on
``UnquantizedLinearMethod.apply`` -> ``F.linear`` -> rocBLAS:

  * ``in_proj_ba`` (GDN gating), N=24 per rank at TP4, K=5120.  rocBLAS takes
    20.6 us to move 0.24 MiB -- its fixed floor, flat for every N up to ~512 --
    and there are 48 of them in a decode step.
  * ``lm_head``, N=62080 per rank, K=5120.  0.59 GiB at ~390 GB/s, 41% of the
    bus, the single most expensive GEMM of the step.

Measured on one 7900 XTX with the production shapes: 3.7x on the first, 2.0x on
the second, together ~1.5 ms off an 8.7 ms per-step GEMM budget.

Enable with ``VLLM_RDNA3_TINY_GEMV=1``.  Anything outside the two measured
bands, or a shape the kernels do not cover, falls through to ``F.linear``.
"""

import os

import torch
import triton
import triton.language as tl

# ROCm serves these through wvSplitK (`rocm_unquantized_gemm`), not rocBLAS, and
# its fast path only covers M <= 5.  From M = 6 it falls back and the cost jumps
# ~4x on the small shape and ~2.4x on the vocabulary one, measured on gfx1100:
#
#   in_proj_ba N=24     M<=5: 4.3-5.5 us   M>=6: 20.6-20.9 us   (this kernel: 7.8-9.4)
#   lm_head    N=62080  M<=5:  691-703 us  M>=6: 1654-2199 us   (this kernel:  819-909)
#
# M is num_seqs x (1 + num_speculative_tokens), so with MTP k=2 the SECOND
# concurrent sequence walks off that cliff: ~1.6 ms of a 20.3 ms decode step.
# Below the cliff wvSplitK wins and is left alone.
MIN_M = 6
SMALL_N = 512
LARGE_N = 16384
# The small-N kernel keeps the weight row in registers for every M; past 32 rows
# BLOCK_M rounds to 64, the x tile spills and it drops to 0.81x.
MAX_M_SMALL = 32
# One pass over the weight covers this many rows; above it the tl.dot tile stops
# paying for itself (measured 0.91x at M=36).
MAX_M_LARGE = 32


@triton.jit
def _gemv_small_n(X, W, Y, B, M, K, sxm, swn, sym,
                  HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    """One program per output column: the weight row is read once for all M."""
    n = tl.program_id(0)
    ms = tl.arange(0, BLOCK_M)
    mmask = ms < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        kmask = ks < K
        w = tl.load(W + n * swn + ks, mask=kmask, other=0.0).to(tl.float32)
        x = tl.load(X + ms[:, None] * sxm + ks[None, :],
                    mask=mmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(x * w[None, :], axis=1)
    if HAS_BIAS:
        acc += tl.load(B + n).to(tl.float32)
    tl.store(Y + ms * sym + n, acc.to(Y.dtype.element_ty), mask=mmask)


@triton.jit
def _gemm_large_n(X, W, Y, B, M, N, K, sxm, swn, sym,
                  HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Column block per program; fp16 operands so tl.dot maps onto WMMA.

    BLOCK_M covers every row in one pass on purpose: tiling M would re-read the
    whole 0.59 GiB weight per tile, which at M=18 cost all of the 2x.
    """
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = cols < N
    ms = tl.arange(0, BLOCK_M)
    mmask = ms < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        kmask = ks < K
        x = tl.load(X + ms[:, None] * sxm + ks[None, :],
                    mask=mmask[:, None] & kmask[None, :], other=0.0)
        w = tl.load(W + cols[None, :] * swn + ks[:, None],
                    mask=kmask[:, None] & cmask[None, :], other=0.0)
        acc += tl.dot(x, w, out_dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(B + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
    tl.store(Y + ms[:, None] * sym + cols[None, :], acc.to(Y.dtype.element_ty),
             mask=mmask[:, None] & cmask[None, :])


def _eligible(x: torch.Tensor, w: torch.Tensor) -> bool:
    if x.dim() != 2 or w.dim() != 2 or x.dtype != w.dtype:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if x.size(1) != w.size(1):
        return False
    if w.stride(1) != 1 or x.stride(1) != 1:
        return False
    n, m = w.size(0), x.size(0)
    if m < MIN_M:  # below the cliff wvSplitK is faster than anything here
        return False
    if n <= SMALL_N:
        return m <= MAX_M_SMALL
    return n >= LARGE_N and m <= MAX_M_LARGE


def tiny_gemv(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None):
    m, k = x.shape
    n = w.size(0)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if n <= SMALL_N:
        _gemv_small_n[(n,)](
            x, w, y, bias, m, k, x.stride(0), w.stride(0), y.stride(0),
            HAS_BIAS=bias is not None, BLOCK_M=max(16, triton.next_power_of_2(m)),
            BLOCK_K=512,
        )
    else:
        _gemm_large_n[(triton.cdiv(n, 16),)](
            x, w, y, bias, m, n, k, x.stride(0), w.stride(0), y.stride(0),
            HAS_BIAS=bias is not None, BLOCK_M=max(16, triton.next_power_of_2(m)),
            BLOCK_N=16, BLOCK_K=128,
        )
    return y


_TARGET = "vllm.model_executor.layers.linear"


def apply_when_imported() -> None:
    """Patch as soon as the linear module is loaded, not before.

    Importing vLLM from ``sitecustomize`` would pull the whole framework in at
    interpreter start, before the environment the workers set up; this waits for
    the real import instead and patches on the way out of it.
    """
    import sys

    if _TARGET in sys.modules:
        apply()
        return

    import importlib.abc
    import importlib.machinery

    class _PatchAfterLoad(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                return None
            inner = spec.loader.exec_module

            def exec_module(module):
                inner(module)
                try:
                    apply()
                except Exception as e:  # noqa: BLE001
                    print("[tiny-gemv] no aplicado:", e, file=sys.stderr)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PatchAfterLoad())


def _wrap(cls) -> bool:
    if getattr(cls, "_tiny_gemv_patched", False):
        return False
    original = cls.apply

    def patched(self, layer, x, bias=None):
        w = getattr(layer, "weight", None)
        if w is not None and _eligible(x, w):
            try:
                return tiny_gemv(x, w, bias)
            except Exception:  # noqa: BLE001  a bad shape must not kill a request
                pass
        return original(self, layer, x, bias)

    cls.apply = patched
    cls._tiny_gemv_patched = True
    return True


def apply() -> None:
    """Patch both unquantized apply paths. Never raises on import.

    ``UnquantizedEmbeddingMethod`` does NOT inherit from
    ``UnquantizedLinearMethod`` and has its own ``apply``, so patching only the
    linear one would silently miss an lm_head that got the embedding method.
    With this recipe the excluded lm_head gets the linear one, but that is a
    property of the checkpoint, not something to rely on.
    """
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return

    hechos = []
    if _wrap(UnquantizedLinearMethod):
        hechos.append("UnquantizedLinearMethod")
    try:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        if _wrap(UnquantizedEmbeddingMethod):
            hechos.append("UnquantizedEmbeddingMethod")
    except Exception:  # noqa: BLE001
        pass
    if hechos:
        print(f"[tiny-gemv] parcheado {' + '.join(hechos)} (N<={SMALL_N} con "
              f"M<={MAX_M_SMALL}, N>={LARGE_N}; sólo desde M>={MIN_M})")


if os.environ.get("VLLM_RDNA3_TINY_GEMV") == "1":
    apply_when_imported()
