# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vision-tower CPU offloading in Qwen3.5 models.

The vision tower is constructed directly instead of via ``make_layers``, so
without explicit wiring it never reaches ``get_offloader().wrap_modules`` and
``--cpu-offload-params visual`` silently matches nothing. These tests verify
that ``_maybe_offload_visual_tower`` routes the tower through the UVA
offloader, and leaves it untouched for the noop/prefetch backends.
"""

import torch.nn as nn

from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.offloader import (
    NoopOffloader,
    UVAOffloader,
    get_offloader,
    set_offloader,
)


class _FakeVisualTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.merger = nn.Linear(4, 4)


def test_visual_tower_untouched_with_noop_offloader():
    """With the default (noop) offloader the tower must pass through unchanged."""
    original_offloader = get_offloader()
    try:
        set_offloader(NoopOffloader())

        tower = _FakeVisualTower()
        tower_param = next(tower.parameters())

        model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
        result = model._maybe_offload_visual_tower(tower)

        assert result is tower
        assert next(result.parameters()) is tower_param
    finally:
        set_offloader(original_offloader)


def test_visual_tower_routed_through_uva_offloader():
    """With the UVA offloader active, the tower must be passed to wrap_modules.

    We use a recording subclass so the test does not require real UVA
    hardware; the point is to verify the wiring, not the offload itself.
    """

    class _RecordingUVAOffloader(UVAOffloader):
        def __init__(self):
            super().__init__(
                cpu_offload_max_bytes=1024**3,
                cpu_offload_params={"visual"},
            )
            self.wrapped_modules: list[nn.Module] = []

        def wrap_modules(self, modules_generator):
            modules = list(modules_generator)
            self.wrapped_modules.extend(modules)
            return modules

    original_offloader = get_offloader()
    try:
        offloader = _RecordingUVAOffloader()
        set_offloader(offloader)

        tower = _FakeVisualTower()

        model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
        result = model._maybe_offload_visual_tower(tower)

        # The tower is handed to the offloader inside a container so the
        # parameters carry the "visual." segment (segment matching works on
        # the wrapped module's relative names), and the original tower comes
        # back unchanged.
        assert len(offloader.wrapped_modules) == 1
        wrapped = offloader.wrapped_modules[0]
        assert wrapped.visual is tower
        param_names = [n for n, _ in wrapped.named_parameters()]
        assert all(n.startswith("visual.") for n in param_names)
        assert result is tower
    finally:
        set_offloader(original_offloader)


def test_visual_tower_untouched_with_prefetch_offloader():
    """PrefetchOffloader.wrap_modules must only be called once (by make_layers).

    The vision tower must therefore not be routed through it, mirroring the
    layer-stack scope of the prefetch backend.
    """
    from vllm.model_executor.offloader import PrefetchOffloader

    original_offloader = get_offloader()
    try:
        # __init__ allocates a CUDA stream; skip it since we only verify the
        # wiring decision (prefetch must not touch the tower).
        offloader = PrefetchOffloader.__new__(PrefetchOffloader)
        set_offloader(offloader)

        tower = _FakeVisualTower()
        tower_param = next(tower.parameters())

        model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
        result = model._maybe_offload_visual_tower(tower)

        assert result is tower
        assert next(result.parameters()) is tower_param
    finally:
        set_offloader(original_offloader)


def test_visual_tower_segment_matching():
    """The "visual" segment must match the tower parameter names.

    This mirrors the check UVAOffloader performs per parameter
    (f".{param}." in f".{name}.") so we can assert the user-facing contract
    ``--cpu-offload-params visual`` targets exactly the tower weights.
    """
    tower = _FakeVisualTower()

    for name, _ in tower.named_parameters(prefix="visual"):
        assert any(f".{p}." in f".{name}." for p in {"visual"}), name
