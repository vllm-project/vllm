# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route the INT8 per-token-head decode attention to the GQA-batched kernel.

The shipped v3 kernel gives every (query row, query head, split) its own wave,
so with GQA the same KV bytes are read once per query head -- six times per rank
at TP4 on Qwen3.5 -- and each read pays its own wave reduction.  Measured in
production (ainode2, TP4, 167k context) it is 38.6% of the decode step: 18 calls
of 704 us, moving 85 MiB of unique KV at 46 GB/s, i.e. bound by instruction
issue, not by memory.

``pth_decode_int8_rdna3_gqa.cu`` batches the heads of one KV head into a wave and
skips the softmax rescale when the running max does not grow (a wave-uniform
branch, so no divergence).  Measured on gfx1100 against the shipped kernel:

    ctx    q=3     q=18
    32k    1.56x   1.65x
    167k   1.37x   1.82x

Numerics: identical maths in a different association order, max relative
difference 6.3e-04 on the fp16 output; with one head per wave the output is
bit-identical, which is the control that the batching is not dropping work.

Enable with ``VLLM_RDNA3_GQA_DECODE=1``.  The op is replaced by name on
``torch.ops._C``, so nothing in vLLM needs editing; on any failure the original
op stays in place.
"""

import os
import sys

# Reusing K/V across heads beats having more waves, so the rule is: take the
# LARGEST head group that still leaves enough waves to fill the machine. The
# floor is what a joint (heads-per-wave x splits) sweep measured on gfx1100, at
# the production ns=256 with 6 q-heads per rank:
#
#   ctx    M    hpw=1   hpw=2   hpw=3   hpw=6      waves at hpw=6
#   167k   4    963     761     720     685.5 us   1024
#   167k  12   2612    1929    1687    1525        3072
#    32k   4    274     235     230     192.6      1024
#
# hpw=6 wins in all three. It only starves below ~1024 waves (at ns=128 it drops
# to 960/210 us and hpw=3 takes over), so that is the floor. An earlier version
# of this file used 8*192 = 1536 and therefore REJECTED hpw=6 at M=4, costing 5%
# of the kernel at 167k and 16% at 32k.
_WAVE_FLOOR = 1024

_ext = None


def _load():
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load

    src = os.environ.get(
        "VLLM_RDNA3_GQA_SRC", "/app/gqa/pth_decode_int8_rdna3_gqa.cu"
    )
    build_dir = os.environ.get("VLLM_RDNA3_GQA_BUILD", "/app/gqa/build")
    os.makedirs(build_dir, exist_ok=True)
    _ext = load(
        name="pth_gqa",
        sources=[src],
        extra_cuda_cflags=["-DUSE_ROCM", "-O3", "--offload-arch=gfx1100"],
        build_directory=build_dir,
        verbose=False,
    )
    return _ext


def _heads_per_wave(num_q: int, num_q_heads: int, num_kv_heads: int, splits: int) -> int:
    """Largest head group that still leaves _WAVE_FLOOR waves."""
    ratio = num_q_heads // max(num_kv_heads, 1)
    best = 1
    for hpw in (2, 3, 4, 6, 8):
        if ratio % hpw or num_q_heads % hpw:
            continue
        if num_q * (num_q_heads // hpw) * splits >= _WAVE_FLOOR:
            best = hpw
    return best


def apply() -> None:
    import torch

    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return
    ns = torch.ops._C
    original = ns.pth_decode_int8_rdna3
    if getattr(original, "_gqa_wrapped", False):
        return
    ext = _load()

    def wrapper(out, query, key_cache, value_cache, k_scale_cache, v_scale_cache,
                block_table, q_to_req, q_to_klen, mid_o_buf, sm_scale,
                num_kv_splits):
        try:
            hpw = _heads_per_wave(
                query.size(0), query.size(1), k_scale_cache.size(2),
                int(num_kv_splits),
            )
            if hpw > 1:
                # El kernel codifica el prefetch en el propio parametro: 100+hpw
                # lo activa (asi se pueden medir las dos variantes con el mismo
                # binario). Pasar hpw a secas deja el prefetch APAGADO, que es
                # exactamente el fallo que hizo que su +12% de banco no
                # apareciera e2e: el paso no se movio ni un 0,1%.
                return ext.pth_decode_int8_gqa(
                    out, query, key_cache, value_cache, k_scale_cache,
                    v_scale_cache, block_table, q_to_req, q_to_klen, mid_o_buf,
                    sm_scale, num_kv_splits, 100 + hpw,
                )
        except Exception as e:  # noqa: BLE001  never lose a request over this
            print("[gqa-decode] fallback:", e, file=sys.stderr)
        return original(out, query, key_cache, value_cache, k_scale_cache,
                        v_scale_cache, block_table, q_to_req, q_to_klen,
                        mid_o_buf, sm_scale, num_kv_splits)

    wrapper._gqa_wrapped = True
    # _OpNamespace caches resolved ops in its own __dict__, so setting the
    # attribute is enough: every later `torch.ops._C.pth_decode_int8_rdna3`
    # finds this wrapper instead of going back to the dispatcher.
    setattr(ns, "pth_decode_int8_rdna3", wrapper)
    print("[gqa-decode] pth_decode_int8_rdna3 -> kernel GQA agrupado")


_TARGET = "vllm.v1.attention.backends.triton_attn"


def apply_when_imported() -> None:
    """Defer until the attention backend is imported (and torch.ops._C exists)."""
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
                    print("[gqa-decode] no aplicado:", e, file=sys.stderr)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PatchAfterLoad())


if os.environ.get("VLLM_RDNA3_GQA_DECODE") == "1":
    apply_when_imported()
