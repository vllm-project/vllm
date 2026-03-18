# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Per-matmul decision tree for async TP fusion gating.

Ported from the AutoHeuristic output trained on H100 data (D176943).
The tree predicts whether fusing GEMM+reduce_scatter or all_gather+GEMM
via symmetric-memory ops is faster than the unfused baseline for a given
(M, K, N) shape.
"""

import os


def should_fuse_async_tp(M: int, K: int, N: int) -> bool:
    """Return True if the fused async TP implementation is predicted faster."""
    if os.environ.get("ASYNC_TP_ALWAYS_FUSE", "0") == "1":
        return True
    m_times_n = M * N
    m_times_k = M * K
    arith_intensity = M * K * N / (m_times_k + K * N + m_times_n)

    if m_times_n <= 2_457_600:
        if K <= 9_600:
            return True
        else:
            if N <= 34_816:
                return False
            else:
                return True
    else:
        if m_times_k <= 24_903_680:
            if N <= 6_144:
                return False
            else:
                if M <= 3_072:
                    if arith_intensity <= 248.49:
                        if m_times_n <= 3_538_944:
                            return True
                        else:
                            return True
                    else:
                        return True
                else:
                    return False
        else:
            if N <= 40_960:
                return True
            else:
                if M <= 12_288:
                    return False
                else:
                    return True
