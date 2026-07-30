# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def cpu_thread_choices() -> list[int]:
    """Candidate ``num_cpu_threads`` values for CPU autotuning.

    Probe points scale with the compute-unit count; ``0`` means all cores.
    Empty off-CPU, where it is unused.
    """
    if not current_platform.is_cpu():
        return []
    n = current_platform.num_compute_units() or 1
    return sorted({max(1, n // 4), max(1, 3 * n // 4), 0})


@triton.jit
def fast_exp(x):
    """Faster alternative to tl.exp() using the hardware exp2 instruction.

    tl.math.exp2 maps directly to a single ex2.approx.f32 PTX instruction,
    while tl.exp goes through libdevice __nv_expf which adds function call
    overhead and extra range checking.
    """
    # exp(x) = exp2(x * log2(e)), where log2(e) = 1/ln(2) = 1.4426950408889634
    LOG2E = tl.constexpr(1.4426950408889634)
    return tl.math.exp2(LOG2E * x)
