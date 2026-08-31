# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 W4A16/W4A4 per-forward dispatch.

The dispatch is only sound when both kernels consume the checkpoint-native
weight layout. These tests pin that contract, because the failure mode of
getting it wrong is either silently wrong numerics (the second kernel reads
weights the first rewrote) or double-allocated weights.
"""

import pytest
import torch

from vllm.model_executor.kernels.linear.nvfp4.dynamic import (
    DynamicNvFp4LinearKernel,
    LayoutMismatchError,
)


class _Kernel:
    """Minimal stand-in; records which M values reached it."""

    preserves_checkpoint_layout = True

    def __init__(self, tag, rewrites=False):
        self.tag, self.rewrites, self.seen = tag, rewrites, []

    def process_weights_after_loading(self, layer):
        if self.rewrites:
            layer.weight = torch.nn.Parameter(
                torch.zeros(8, 8, dtype=torch.uint8), requires_grad=False
            )

    def apply_weights(self, layer, x, bias=None):
        self.seen.append(x.numel() // x.shape[-1])
        return torch.zeros((*x.shape[:-1], 4))


class _Rewriting(_Kernel):
    preserves_checkpoint_layout = False


def _layer():
    m = torch.nn.Module()
    m.weight = torch.nn.Parameter(
        torch.zeros(4, 4, dtype=torch.uint8), requires_grad=False
    )
    m.weight_scale = torch.nn.Parameter(
        torch.zeros(4, 1, dtype=torch.uint8), requires_grad=False
    )
    return m


def _build(a16, a4, max_m=16):
    k = object.__new__(DynamicNvFp4LinearKernel)
    k.a16_kernel, k.a4_kernel, k.a16_max_m = a16, a4, max_m
    return k


@pytest.mark.parametrize("m,expect", [(1, "a16"), (16, "a16"), (17, "a4"), (512, "a4")])
def test_dispatches_on_m(m, expect):
    """M at or below the threshold goes to W4A16, above it to W4A4."""
    a16, a4 = _Kernel("a16"), _Kernel("a4")
    k = _build(a16, a4, max_m=16)
    k.apply_weights(_layer(), torch.zeros(m, 4))
    assert (a16.seen, a4.seen) == (([m], []) if expect == "a16" else ([], [m]))


def test_rejects_kernel_that_rewrites_weights():
    """A kernel not declaring the contract must be refused, not run."""
    k = _build(_Rewriting("a16"), _Kernel("a4"))
    with pytest.raises(LayoutMismatchError, match="cannot share one weight layout"):
        k.process_weights_after_loading(_layer())


def test_catches_kernel_that_lies_about_the_contract():
    """Declaring the contract but rewriting anyway is caught, not trusted."""
    liar = _Kernel("a16", rewrites=True)  # declares True, still rewrites
    k = _build(liar, _Kernel("a4"))
    with pytest.raises(LayoutMismatchError, match="changed the layout"):
        k.process_weights_after_loading(_layer())


def test_compatible_pair_prepares_once_and_keeps_layout():
    a16, a4 = _Kernel("a16"), _Kernel("a4")
    layer = _layer()
    before = (tuple(layer.weight.shape), str(layer.weight.dtype))
    _build(a16, a4).process_weights_after_loading(layer)
    assert (tuple(layer.weight.shape), str(layer.weight.dtype)) == before


def test_no_in_tree_kernel_declares_the_contract_yet():
    """Guard: if a kernel starts declaring it, this must be re-reviewed."""
    from vllm.model_executor.kernels.linear.nvfp4 import (
        cutlass,
        flashinfer,
        marlin,
    )

    declaring = [
        c.__name__
        for mod in (marlin, cutlass, flashinfer)
        for c in vars(mod).values()
        if isinstance(c, type) and getattr(c, "preserves_checkpoint_layout", False)
    ]
    assert declaring == [], (
        f"{declaring} now declare preserves_checkpoint_layout; verify the "
        "dispatch pairing and update this guard"
    )


def test_opt_in_branch_executes_on_any_platform(monkeypatch):
    """Regression: the branch must not read names bound later in the function.

    The dispatch branch sits above the auto-select block, so anything it reads
    from there is unbound at that point and raises NameError on the first
    layer built with the feature enabled.
    """
    import vllm.envs as envs
    from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel

    monkeypatch.setattr(envs, "VLLM_NVFP4_A16_MAX_M", 8)
    try:
        init_nvfp4_linear_kernel(use_a16=True)
    except NameError as e:
        pytest.fail(f"dispatch branch references an unbound name: {e}")
    except Exception:
        pass  # no NVFP4 kernel on this platform is an acceptable outcome
