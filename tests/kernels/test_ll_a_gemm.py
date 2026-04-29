# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
from vllm.model_executor.layers.fused_moe.router.ll_a_gemm_tma import ll_a_gemm_tma

SHAPES = [
    (16, 512, 16),
    (16, 512, 32),
    (1, 512, 16),
    (1, 2048, 512),
    (4, 2048, 512),
    (16, 2048, 512),
    (1, 7168, 2112),
    (4, 7168, 2112),
    (8, 7168, 2112),
    (16, 7168, 2112),
]

KERNELS = {
    'peeled': ll_a_gemm,
    'tma':    ll_a_gemm_tma,
}


def test_shape(kernel_name, kernel_fn, M, K, N, is_fp8=False):
    if is_fp8:
        a_raw = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn)
        b_raw = torch.randn(N, K, device='cuda').to(torch.float8_e4m3fn)
        a = a_raw.view(torch.bfloat16)
        b = b_raw.view(torch.bfloat16)
    else:
        a_raw = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        b_raw = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
        a, b = a_raw, b_raw

    try:
        c = kernel_fn(a, b, is_fp8=is_fp8)
        torch.cuda.synchronize()
    except Exception as e:
        print(f'  {kernel_name:6s} M={M:2d} K={K:4d} N={N:4d} '
              f'{"fp8" if is_fp8 else "bf16"}: ERROR {e}')
        return False

    ref = torch.mm(a_raw.float(), b_raw.float().T).to(torch.bfloat16)
    err = (c - ref).abs().max().item()
    rel_err = err / max(ref.abs().max().item(), 1e-6)
    dtype_str = 'fp8' if is_fp8 else 'bf16'
    status = 'PASS' if rel_err < 1e-2 else 'FAIL'
    print(f'  {kernel_name:6s} M={M:2d} K={K:4d} N={N:4d} {dtype_str}: '
          f'abs={err:.2e} rel={rel_err:.2e} [{status}]')
    return rel_err < 1e-2


def main():
    print('cuteDSL A GEMM test — all implementations')
    print(f'Device: {torch.cuda.get_device_name()}')
    print()

    all_pass = True
    for kernel_name, kernel_fn in KERNELS.items():
        for is_fp8 in [False, True]:
            dtype_str = 'fp8' if is_fp8 else 'bf16'
            print(f'--- {kernel_name} {dtype_str} ---')
            for M, K, N in SHAPES:
                all_pass &= test_shape(kernel_name, kernel_fn, M, K, N,
                                        is_fp8=is_fp8)
            print()

    print('ALL PASS' if all_pass else 'SOME FAILED')


if __name__ == '__main__':
    main()
