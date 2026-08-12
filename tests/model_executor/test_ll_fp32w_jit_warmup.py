# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def test_warmup_keys_match_legacy_expansion() -> None:
    from vllm.model_executor.kernels.linear.cute_dsl import ll_fp32w

    kernel = ll_fp32w.LLFp32WGemm()
    shapes = ((6144, 128), (6144, 256), (2048, 64))
    m_values = range(1, 33)
    a_dtypes = (torch.bfloat16, torch.float16, torch.float32)

    expected = []
    for K, N in shapes:
        for M in m_values:
            default_config = (
                ll_fp32w._DEFAULT_DOTPROD_GROUPED_CONFIG
                if N <= 128 and M >= 12 and M % 2 == 0
                else ll_fp32w._DEFAULT_DOTPROD_CONFIG
            )
            bs, token_groups, epb = ll_fp32w._TUNED_DOTPROD_CONFIGS.get((K, N), {}).get(
                M, default_config
            )
            for a_dtype in a_dtypes:
                expected.append(
                    kernel.CompileKey(
                        m=M,
                        k=K,
                        bs=bs,
                        a_dtype=a_dtype,
                        token_groups=token_groups,
                        epb=epb,
                    )
                )

    assert kernel.get_warmup_keys(
        shapes=shapes,
        m_values=m_values,
        a_dtypes=a_dtypes,
    ) == list(dict.fromkeys(expected))
