# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from triton.testing import do_bench_cudagraph

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
from vllm.model_executor.layers.fused_moe.router.ll_a_gemm_tma import ll_a_gemm_tma

q = [0.5, 0.2, 0.8]
_HAS_DSV3 = hasattr(ops, 'dsv3_fused_a_gemm')

try:
    from flashinfer.gemm import tgv_gemm_sm100
    from flashinfer import autotune
    _HAS_TGV = True
except ImportError:
    _HAS_TGV = False

try:
    from flashinfer.gemm import tinygemm_bf16
    _HAS_TINY = True
except ImportError:
    _HAS_TINY = False

print(f'Device: {torch.cuda.get_device_name()}')
print(f'DSV3-A: {_HAS_DSV3} | TGV: {_HAS_TGV} | tinygemm: {_HAS_TINY}')
print()

SHAPES = [
    (7168, 2112,  "a_proj combined"),
    (7168, 576,   "kv_a_proj"),
    (7168, 1536,  "q_a_proj"),
    (1536, 24576, "q_b_proj TP1"),
    (1536, 3072,  "q_b_proj TP8"),
    (512, 32768,  "kv_b_proj TP1"),
    (512, 4096,   "kv_b_proj TP8"),
]


def _bench(fn, q=q):
    return do_bench_cudagraph(fn, rep=200, quantiles=q)[0] * 1000


def bench_one(M, K, N):
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    a8 = a.to(torch.float8_e4m3fn).view(torch.bfloat16)
    b8 = b.to(torch.float8_e4m3fn).view(torch.bfloat16)

    r = {}

    # Peeled cp.async
    r['p-bf16'] = _bench(lambda: ll_a_gemm(a, b))
    r['p-fp8'] = _bench(lambda: ll_a_gemm(a8, b8, is_fp8=True))

    # TMA pipeline
    try:
        ll_a_gemm_tma(a, b); torch.cuda.synchronize()
        r['t-bf16'] = _bench(lambda: ll_a_gemm_tma(a, b))
    except Exception:
        r['t-bf16'] = float('nan')

    try:
        ll_a_gemm_tma(a8, b8, is_fp8=True); torch.cuda.synchronize()
        r['t-fp8'] = _bench(lambda: ll_a_gemm_tma(a8, b8, is_fp8=True))
    except Exception:
        r['t-fp8'] = float('nan')

    # DSV3 fused A GEMM (C++) — only K=7168, N=2112, M<=16
    if _HAS_DSV3 and K == 7168 and N == 2112 and M <= 16:
        o = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        r['DSV3'] = _bench(lambda: ops.dsv3_fused_a_gemm(o, a, b.T))
    else:
        r['DSV3'] = float('nan')

    # TGV-sm100 (FlashInfer) — requires N%16==0
    if _HAS_TGV and N % 16 == 0:
        bias = torch.zeros(N, dtype=torch.bfloat16, device='cuda')
        out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        with autotune(True):
            tgv_gemm_sm100(a, b.T, bias, out=out)
        torch.cuda.synchronize()
        r['TGV'] = _bench(lambda: tgv_gemm_sm100(a, b.T, bias, out=out))
    else:
        r['TGV'] = float('nan')

    # tinygemm_bf16 (FlashInfer tinygemm2) — requires N%16==0
    if _HAS_TINY and N % 16 == 0:
        bias_t = torch.zeros(N, dtype=torch.bfloat16, device='cuda')
        out_t = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        try:
            with autotune(True):                
                tinygemm_bf16(a, b, out_t, bias=bias_t)
            torch.cuda.synchronize()
            r['tiny2'] = _bench(lambda: tinygemm_bf16(a, b, out_t, bias=bias_t))
        except Exception:
            r['tiny2'] = float('nan')
    else:
        r['tiny2'] = float('nan')

    # cuBLAS
    omm = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    r['cuBLAS'] = _bench(lambda: torch.mm(a, b.T, out=omm))

    return r


cols = ['p-bf16', 'p-fp8', 't-bf16', 't-fp8', 'DSV3', 'TGV', 'tiny2', 'cuBLAS']

for K, N, label in SHAPES:
    print(f'=== {label}: K={K}, N={N} ===')
    hdr = f"{'M':>3} |" + "".join(f" {c:>8}" for c in cols)
    print(hdr)
    print('-' * len(hdr))

    for M in [1, 4, 16]:
        r = bench_one(M, K, N)
        bf16_cols = ['p-bf16', 't-bf16', 'DSV3', 'TGV', 'tiny2', 'cuBLAS']
        fp8_cols = ['p-fp8', 't-fp8']
        best_bf16 = min((k for k in bf16_cols if r[k] == r[k]),
                         key=lambda k: r[k])
        best_fp8 = min((k for k in fp8_cols if r[k] == r[k]),
                        key=lambda k: r[k])
        vals = "".join(f" {r[c]:7.2f}us" if r[c] == r[c] else "      N/A"
                       for c in cols)
        print(f' {M:2d} |{vals}  bf16:{best_bf16} fp8:{best_fp8}')
    print()
