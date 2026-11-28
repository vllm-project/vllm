# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.triton_utils import tl, triton

# The keyword arguments of the config each launch key picked.
_autotuned_configs: dict[tuple, dict[str, Any]] = {}


def launch_autotuned(
    kernel: Any, grid: Any, variant: dict[str, Any] | None = None, **kwargs
) -> None:
    """Launch a `@triton.autotune` kernel with the config it already picked.

    `Autotuner.run` rebuilds its cache key on every launch, matching each
    argument name against a list and formatting the dtype of every tensor
    argument, several microseconds of host time per launch. This builds an
    equivalent key directly from `kwargs`: the values of the kernel's
    autotune `key` arguments plus the dtype (or `None`) of every argument.
    The picked config's keyword arguments are kept, not rebuilt per launch.

    `variant` holds extra kernel arguments that make the kernel skip or
    redirect work. The variant runs with the config picked for `kwargs`
    alone, so it never autotunes, leaves that pick alone and computes every
    value the same way. The first launch for a key runs the plain launch
    (to autotune) and then the variant.
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
        best = kernel.best_config
        if best.pre_hook is not None:
            assert variant is None, "variants need configs without pre_hook"
            return
        config = _autotuned_configs[key] = best.all_kwargs()
        if variant is None:
            return
    if variant is not None:
        kwargs.update(variant)
    kernel.fn[grid](**kwargs, **config)


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
