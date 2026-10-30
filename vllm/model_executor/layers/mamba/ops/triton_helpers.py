# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.triton_utils import tl, triton

_autotuned_configs: dict[tuple, Any] = {}


def launch_autotuned(kernel: Any, grid: Any, **kwargs) -> None:
    """Launch a ``@triton.autotune`` kernel with the config it already picked.

    ``Autotuner.run`` rebuilds its cache key on every launch, matching each
    argument name against a list and formatting the dtype of every tensor
    argument, several microseconds of host time per launch. This builds an
    equivalent key directly from ``kwargs``: the values of the kernel's
    autotune ``key`` arguments plus the dtype (or ``None``) of every argument.
    """
    key = (
        kernel,
        *[kwargs.get(name) for name in kernel.keys],
        *[getattr(arg, "dtype", None) for arg in kwargs.values()],
    )
    config = _autotuned_configs.get(key)
    if config is None:
        # Autotunes on the first launch for this key.
        kernel[grid](**kwargs)
        if kernel.best_config.pre_hook is None:
            _autotuned_configs[key] = kernel.best_config
        return
    kernel.fn[grid](**kwargs, **config.all_kwargs())


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
