# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand-written custom baselines for benchmarking Helion kernels.

Each entry in ``CUSTOM_BASELINE_FNS`` maps a Helion kernel name to a reference
callable sharing that kernel's argument interface, so the kernel's input tuple
can be forwarded verbatim. These are used as the ``custom`` performance baseline
(e.g. by ``scripts/benchmark_helion_kernels.py``) to compare a Helion kernel
against a production vLLM op rather than the native-torch autotune reference.

Add a new baseline by defining a function with the kernel's argument interface
and registering it in ``CUSTOM_BASELINE_FNS``.
"""

from collections.abc import Callable

import torch


def _block_scaled_mm(
    out: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    a_scales: torch.Tensor,  # [num_group_m, num_group_k]
    b_scales: torch.Tensor,  # [num_group_k, num_group_n]
) -> None:
    from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import fp8_gemm_nt

    fp8_gemm_nt((a, a_scales), (b.T, b_scales.T), out, is_deep_gemm_e8m0_used=True)


# Maps a Helion kernel name to a hand-written reference used as the ``custom``
# baseline. Each function shares the kernel's argument interface, so the
# kernel's input tuple is forwarded verbatim.
CUSTOM_BASELINE_FNS: dict[str, Callable] = {
    "block_scaled_mm": _block_scaled_mm,
}
