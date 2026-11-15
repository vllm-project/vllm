# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for autotuning Helion kernels, including disabled kernels with no configs."""

import pytest
import torch

from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

import helion
import helion.language as hl
from helion.autotuner.base_search import BaseSearch

from tests.kernels.helion.helpers import dummy_kernel_registry
from vllm.kernels.helion.register import create_helion_decorated_kernel


def _add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


class NoCompileSearch(BaseSearch):
    """Autotuner that returns the default config without GPU compilation.

    Modeled after helion's test BasicSearch (pytorch/helion#1649).
    """

    def autotune(self, *, skip_cache: bool = False):
        return self.config_spec.default_config()


def _no_compile_autotuner_fn(bound_kernel, args, **kwargs):
    return NoCompileSearch(bound_kernel, args, **kwargs)


class TestAutotuneDisabledKernel:
    """Test autotuning flow on disabled kernels (no platform configs)."""

    def setup_method(self):
        from vllm.kernels.helion.register import _REGISTERED_KERNELS

        self._saved_registry = dict(_REGISTERED_KERNELS)
        _REGISTERED_KERNELS.clear()

    def teardown_method(self):
        from vllm.kernels.helion.register import _REGISTERED_KERNELS

        _REGISTERED_KERNELS.clear()
        _REGISTERED_KERNELS.update(self._saved_registry)

    def test_autotune_disabled_kernel_produces_valid_config(self):
        """Register a kernel with no configs (disabled), run autotune,
        verify it produces a valid helion.Config."""
        with dummy_kernel_registry(configs={}) as register:
            wrapper = register(
                "autotune_test_kernel",
                config_picker=lambda args, keys: None,
                fake_impl=lambda *a, **kw: None,
                input_generator=lambda: {
                    "small": (
                        torch.randn(4, 4, device="cuda"),
                        torch.randn(4, 4, device="cuda"),
                    ),
                },
            )(_add_kernel)

        assert wrapper._disabled is True

        inputs = wrapper.get_inputs()
        assert "small" in inputs

        settings = helion.Settings()
        settings.autotuner_fn = _no_compile_autotuner_fn
        wrapper.helion_settings = settings

        config = wrapper.run_autotune(inputs["small"])
        expected_default = (
            create_helion_decorated_kernel(_add_kernel, helion_settings=settings)
            .bind(inputs["small"])
            .config_spec.default_config()
        )
        assert config == expected_default
