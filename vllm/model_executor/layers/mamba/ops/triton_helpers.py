# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.triton_utils import tl, triton

_autotuned_configs: dict[tuple, Any] = {}


def launch_autotuned(kernel: Any, grid: Any, config_key: tuple, **kwargs) -> None:
    """Launch a ``@triton.autotune`` kernel with the config it picked for
    ``config_key``.

    ``Autotuner.run`` rebuilds its cache key from every argument on each
    launch, several microseconds of host time per launch. ``config_key`` must
    determine that key: the values of the kernel's autotune ``key`` arguments
    and the dtype (or absence) of every tensor argument.
    """
    cache_key = (kernel, config_key)
    config = _autotuned_configs.get(cache_key)
    if config is None:
        # Autotunes on the first launch for this key.
        kernel[grid](**kwargs)
        if kernel.best_config.pre_hook is None:
            _autotuned_configs[cache_key] = kernel.best_config
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
