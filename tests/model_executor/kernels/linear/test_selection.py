# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.kernels.linear import choose_w8a8_linear_kernel
from vllm.model_executor.kernels.linear.base import w8a8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape, QuantKey, ScaleDesc,
)

HIGH_CAP = 80
LOW_CAP = 70

class AlwaysSelectsMixin:
    @classmethod
    def is_supported(cls, compute_capability=None): return True, None
    @classmethod
    def can_implement(cls, config): return True, None

class NeverSelectsUnsupportedMixin:
    @classmethod
    def is_supported(cls, compute_capability=None): return False, "unsupported platform"
    @classmethod
    def can_implement(cls, config): return True, None

class NeverSelectsCannotImplementMixin:
    @classmethod
    def is_supported(cls, compute_capability=None): return True, None
    @classmethod
    def can_implement(cls, config): return False, "cannot implement this config"

class RequiresHighCapabilityMixin:
    @classmethod
    def is_supported(cls, compute_capability=None):
        if compute_capability is not None and compute_capability < HIGH_CAP:
            return False, f"requires compute capability >= {HIGH_CAP}"
        return True, None
    @classmethod
    def can_implement(cls, config): return True, None




def _mock(base, mixin):
    """Combine a kernel base class with a behavior mixin to create a mock kernel."""
    return type(f"{mixin.__name__}_{base.__name__}", (mixin, base), {})


def _fp_config():
    qk = QuantKey(
        dtype=torch.float8_e4m3fn,
        scale=ScaleDesc(dtype=torch.float32, static=True, group_shape=GroupShape.PER_TENSOR),
        symmetric=True,
    )
    return w8a8.FpKernelConfig(weight_quant_key=qk, activation_quant_key=qk, out_dtype=torch.float16)


def _int_config():
    return w8a8.IntKernelConfig(is_channelwise=True, is_static_input_scheme=True, input_symmetric=True)


@pytest.fixture(params=[
    (w8a8.FpKernel,  _fp_config),
    (w8a8.IntKernel, _int_config),
])
def ctx(request):
    base, make_config = request.param
    return (
        make_config(),
        _mock(base, AlwaysSelectsMixin),
        _mock(base, AlwaysSelectsMixin),   # always_b — distinct class from always_a
        _mock(base, NeverSelectsUnsupportedMixin),
        _mock(base, NeverSelectsCannotImplementMixin),
        _mock(base, RequiresHighCapabilityMixin),
    )

def test_selects_first_supported_kernel(ctx):
    config, always_a, always_b, *_ = ctx
    assert choose_w8a8_linear_kernel(config, [always_a, always_b]) == always_a


def test_skips_unsupported_kernels(ctx):
    config, always_a, _, unsupported, *_ = ctx
    assert choose_w8a8_linear_kernel(config, [unsupported, always_a]) == always_a


def test_skips_kernels_that_cannot_implement(ctx):
    config, always_a, _, _, cannot_implement, _ = ctx
    assert choose_w8a8_linear_kernel(config, [cannot_implement, always_a]) == always_a


def test_raises_when_no_kernel_available(ctx):
    config, _, _, unsupported, cannot_implement, _ = ctx
    with pytest.raises(ValueError, match="Failed to find a kernel"):
        choose_w8a8_linear_kernel(config, [unsupported, cannot_implement], compute_capability=HIGH_CAP)


def test_forced_kernel_is_selected_when_available(ctx):
    config, always_a, always_b, *_ = ctx
    assert choose_w8a8_linear_kernel(config, [always_a, always_b], force_kernel=always_b) == always_b


def test_forced_kernel_fallback_when_unavailable(ctx):
    config, always_a, _, unsupported, *_ = ctx
    assert choose_w8a8_linear_kernel(config, [always_a], force_kernel=unsupported) == always_a


def test_compute_capability_filtering(ctx):
    config, always_a, _, _, _, requires_high_cap = ctx
    assert choose_w8a8_linear_kernel(config, [requires_high_cap, always_a], compute_capability=LOW_CAP) == always_a
    assert choose_w8a8_linear_kernel(config, [requires_high_cap, always_a], compute_capability=HIGH_CAP) == requires_high_cap
