# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MonoKernel conformance kit (RFC #60904).

Every model that ships a ``MonoSpec`` subclasses ``MonoConformance`` and fills
in the five hooks, which gives it the contract the library relies on: refusals
are complete, the spec module is importable without a kernel backend, the
capture sizes are the spec's widths, one forward decides once and alike on
every rank, caches re-bind when they move, attach does not move the model's
weights, cache checks report every mismatch, served steps are counted, and
every width is warmed up before capture.

The kit never launches a kernel: it drives the library through the spec's own
callbacks, so a spec whose kernels need a GPU still conforms on CPU.
"""

import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.models.common.mono import runtime
from vllm.models.common.mono.guards import (
    _load_specs,
    _refuse_config,
    _select_spec,
    _set_capture_sizes,
    configure,
)
from vllm.models.common.mono.layer import attach, forward_layer
from vllm.models.common.mono.spec import Feature, MonoUnsupported

KERNEL_BACKENDS = ("aiter", "flydsl")
"""Backends a spec module must not pull in: it is imported at config time, on
every platform, before anything has decided the spec can run."""


@contextmanager
def forward(config, mode: CUDAGraphMode, attn_metadata: Any) -> Iterator[None]:
    """One forward pass's context, as a runner would set it."""
    context = ForwardContext(
        no_compile_layers=config.compilation_config.static_forward_context,
        attn_metadata=attn_metadata,
        slot_mapping={},
        cudagraph_runtime_mode=mode,
    )
    with override_forward_context(context):
        yield


def _spy(spec, field: str, calls: list):
    """``spec`` with one callback recording its calls."""
    inner = getattr(spec, field)

    def recorded(*args):
        calls.append(args)
        return inner(*args)

    return replace(spec, **{field: recorded})


class MonoConformance:
    """What every ``MonoSpec`` must satisfy. Subclass it in the model's test
    module and fill in the hooks below."""

    specs_path: str
    """``"package.module:MONO_SPECS"``, as the model passes it to ``attach``."""

    def make_config(self, **overrides) -> Any:
        """A config this spec serves, with ``overrides`` applied on top."""
        raise NotImplementedError

    def make_model(self, config) -> Any:
        """A loaded model this spec serves, with its KV caches bound into
        ``config.compilation_config.static_forward_context``."""
        raise NotImplementedError

    def decode_metadata(self, config, model) -> Any:
        """Attention metadata for a decode step the kernels take."""
        raise NotImplementedError

    def skipped_metadata(self, config, model) -> Any:
        """Attention metadata for a step ``skip_step`` sends to vLLM's path."""
        raise NotImplementedError

    def bad_caches(self, config, model) -> dict[str, Any]:
        """Caches that break at least two of ``bind``'s checks at once."""
        raise NotImplementedError

    @pytest.fixture
    def config(self):
        return self.make_config()

    @pytest.fixture
    def model(self, config):
        return self.make_model(config)

    @pytest.fixture
    def mono(self, config, model):
        return attach(model, config, self.specs_path)

    @pytest.fixture(autouse=True)
    def _counters(self):
        runtime.STEPS.clear()
        yield
        runtime.STEPS.clear()

    def spec(self, config):
        spec, refusals = _select_spec(_load_specs(self.specs_path), config)
        assert spec is not None, refusals
        return spec

    def test_refusals_report_every_reason(self, config):
        config.lora_config = object()
        config.speculative_config = object()
        config.use_v2_model_runner = False
        for spec in _load_specs(self.specs_path):
            reasons = _refuse_config(spec, config)
            assert sum("Model Runner V2" in r for r in reasons) == 1, reasons
            for feature in (Feature.LORA, Feature.SPECULATIVE_DECODING):
                want = 0 if feature in spec.supports else 1
                assert sum(feature.value in r for r in reasons) == want, reasons

    def test_refusal_names_every_variant(self, config):
        config.use_v2_model_runner = False
        with pytest.raises(MonoUnsupported) as excinfo:
            configure(config, self.specs_path)
        for spec in _load_specs(self.specs_path):
            assert spec.name in str(excinfo.value)

    def test_spec_module_imports_no_kernel_backend(self):
        module = self.specs_path.split(":")[0]
        source = (
            "import sys; sys.path.insert(0, '.'); import importlib;"
            f"importlib.import_module({module!r});"
            "print('\\n'.join(sys.modules))"
        )
        loaded = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        pulled = {n.split(".")[0] for n in loaded} & set(KERNEL_BACKENDS)
        assert not pulled, f"{module} imports {sorted(pulled)}"

    def test_capture_sizes_are_the_widths(self, config):
        spec = self.spec(config)
        widths = spec.widths(config)
        config.compilation_config.cudagraph_capture_sizes = list(
            range(1, max(widths) * 2 + 1)
        )
        _set_capture_sizes(spec, config)
        sizes = config.compilation_config.cudagraph_capture_sizes
        assert [s for s in sizes if s <= max(widths)] == sorted(widths)

    def test_decided_once_a_forward(self, config, model, mono):
        asked: list = []
        mono.spec = _spy(mono.spec, "skip_step", asked)
        with forward(config, CUDAGraphMode.NONE, self.decode_metadata(config, model)):
            self._run_layers(mono, model)
        assert len(asked) == 1
        assert len(mono.kernels) > 1

    def test_every_rank_decides_alike(self, config, model):
        decisions = []
        for rank in range(2):
            other = attach(self.make_model(config), config, self.specs_path)
            other.ctx = replace(other.ctx, rank=rank)
            with forward(
                config, CUDAGraphMode.NONE, self.decode_metadata(config, model)
            ):
                decisions.append(other.begin_step(self._positions(other)))
        assert decisions[0] == decisions[1] is None

    def test_skipped_step_runs_vllm_layers(self, config, model, mono):
        with forward(config, CUDAGraphMode.NONE, self.skipped_metadata(config, model)):
            mono.begin_once(self._positions(mono))
            assert mono.step is None
            assert mono.last is not None

    def test_width_off_the_capture_sizes_runs_vllm_layers(self, config, model, mono):
        off = max(mono.widths) + 1
        with forward(config, CUDAGraphMode.NONE, self.decode_metadata(config, model)):
            assert mono.begin_step(torch.zeros(off, dtype=torch.long)) == "width"

    def test_piecewise_runs_vllm_layers(self, config, model, mono):
        meta = self.decode_metadata(config, model)
        with forward(config, CUDAGraphMode.PIECEWISE, meta):
            assert mono.begin_step(self._positions(mono)) == "piecewise"

    def test_caches_rebind_when_they_move(self, config, model, mono):
        binds: list = []
        mono.spec = _spy(mono.spec, "bind", binds)
        self._decode(config, model, mono)
        assert len(binds) == 1
        self._decode(config, model, mono)
        assert len(binds) == 1
        sfc = config.compilation_config.static_forward_context
        for name in mono.cache_layers:
            sfc[name].kv_cache = sfc[name].kv_cache.clone()
        self._decode(config, model, mono)
        assert len(binds) == 2

    def test_bind_reports_every_mismatch(self, config, model, mono):
        reasons = mono.spec.bind(mono.ctx, self.bad_caches(config, model))
        assert len(reasons) >= 2, reasons

    def test_mismatched_caches_refuse_the_step(self, config, model, mono):
        sfc = config.compilation_config.static_forward_context
        bad = self.bad_caches(config, model)
        for name, cache in zip(mono.cache_layers, bad.values(), strict=False):
            sfc[name].kv_cache = cache
        with pytest.raises(MonoUnsupported):
            self._decode(config, model, mono)

    def test_attach_does_not_move_weights(self, config):
        model = self.make_model(config)
        before = {n: p.data_ptr() for n, p in model.named_parameters()}
        attach(model, config, self.specs_path)
        after = {n: p.data_ptr() for n, p in model.named_parameters()}
        assert before == after

    def test_counters_follow_full_replay(self, config, model, mono):
        width = min(mono.widths)
        with forward(config, CUDAGraphMode.FULL, self.decode_metadata(config, model)):
            mono.begin_step(torch.zeros(width, dtype=torch.long))
        for _ in range(3):
            mono.account(width, CUDAGraphMode.FULL)
        assert runtime.drain() == {f"steps:{mono.spec.name}:mono": 3}

    def test_counters_name_the_skipped_path(self, config, model, mono):
        width = min(mono.widths)
        with forward(config, CUDAGraphMode.FULL, self.skipped_metadata(config, model)):
            why = mono.begin_step(torch.zeros(width, dtype=torch.long))
        mono.account(width, CUDAGraphMode.FULL)
        assert runtime.drain() == {f"steps:{mono.spec.name}:{why}": 1}

    def test_warmup_registered_for_every_width(self, config):
        registry = JitWarmupRegistry(config)
        with registry.activate():
            mono = attach(self.make_model(config), config, self.specs_path)
        assert len(registry) >= 1
        by_width: dict[int, set] = defaultdict(set)
        for (_, width), kernel in mono.kernels.items():
            by_width[width] |= set(kernel.warmup_keys())
        assert set(by_width) == set(mono.widths)
        assert all(by_width.values())
        registry.warmup()

    def _positions(self, mono) -> torch.Tensor:
        return torch.zeros(min(mono.widths), dtype=torch.long)

    def _decode(self, config, model, mono) -> None:
        with forward(config, CUDAGraphMode.NONE, self.decode_metadata(config, model)):
            mono.begin_step(self._positions(mono))

    def _run_layers(self, mono, model) -> None:
        hidden = torch.zeros(min(mono.widths), model.config.hidden_size)
        positions = self._positions(mono)
        for i, layer in enumerate(model.layers):
            hidden = forward_layer(mono, i, layer, positions, hidden)
