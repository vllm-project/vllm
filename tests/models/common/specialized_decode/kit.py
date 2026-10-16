# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode conformance kit (RFC #60904): what a ``DecodeSpec``
must satisfy."""

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.models.common.specialized_decode.dispatch import SpecializedDecode
from vllm.models.common.specialized_decode.guards import (
    _refuse_config,
    _select_spec,
    _set_capture_sizes,
    select_decode_spec,
)
from vllm.models.common.specialized_decode.spec import DecodeUnsupported, Feature

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


def _record(obj, method: str) -> list:
    """``obj.method``'s calls from here on, as they are made."""
    calls: list = []
    inner = getattr(obj, method)

    def recorded(*args, **kwargs):
        calls.append(args)
        return inner(*args, **kwargs)

    setattr(obj, method, recorded)
    return calls


class DecodeConformance:
    """What every ``DecodeSpec`` must satisfy. Subclass it in the model's test
    module and fill in the hooks below."""

    specs: tuple[Any, ...]
    """The model's variants, as it passes them to ``select_decode_spec``."""

    specs_module: str
    """Where ``specs`` is defined, for the import test to load on its own."""

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
        """Caches that break at least two of ``refuse_caches``'s checks."""
        raise NotImplementedError

    @pytest.fixture
    def config(self):
        return self.make_config()

    @pytest.fixture
    def model(self, config):
        return self.make_model(config)

    @pytest.fixture
    def decode(self, config, model):
        return self.make_decode(config, model)

    def make_decode(self, config, model=None, bind: bool = True):
        """This rank's ``SpecializedDecode``, built the way a model builds it:
        the variant picked once by ``select_decode_spec``, then the dispatch
        over the loaded model, then the caches the runner binds."""
        model = self.make_model(config) if model is None else model
        spec = select_decode_spec(config, self.specs)
        assert spec is not None
        decode = SpecializedDecode(model, config, spec)
        if bind:
            decode.bind_caches()
        return decode

    def spec(self, config):
        spec, refusals = _select_spec(self.specs, config)
        assert spec is not None, refusals
        return spec

    def test_refusals_report_every_reason(self, config):
        config.lora_config = object()
        config.speculative_config = object()
        config.use_v2_model_runner = False
        for spec in self.specs:
            reasons = _refuse_config(spec, config)
            assert sum("Model Runner V2" in r for r in reasons) == 1, reasons
            for feature in (Feature.LORA, Feature.SPECULATIVE_DECODING):
                want = 0 if feature in spec.supports else 1
                assert sum(feature.value in r for r in reasons) == want, reasons

    def test_refusal_names_every_variant(self, config):
        config.use_v2_model_runner = False
        with pytest.raises(DecodeUnsupported) as excinfo:
            select_decode_spec(config, self.specs)
        for spec in self.specs:
            assert spec.name in str(excinfo.value)

    def test_spec_module_imports_no_kernel_backend(self):
        module = self.specs_module
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

    def test_decided_once_a_step(self, config, model, decode):
        asked: list = []
        decode.spec = _spy(decode.spec, "skip_step", asked)
        step = self._plan(config, model, decode)
        assert step is not None and step.width == min(decode.widths)
        assert len(asked) == 1

    def test_every_rank_decides_alike(self, config, model):
        decisions = []
        for _ in range(2):
            other = self.make_decode(config)
            decisions.append(self._plan(config, model, other))
        assert decisions[0] is not None and decisions[1] is not None

    def test_skipped_step_runs_vllm_layers(self, config, model, decode):
        with forward(config, CUDAGraphMode.NONE, self.skipped_metadata(config, model)):
            assert decode.plan(self._positions(decode)) is None

    def test_pipeline_stage_input_runs_vllm_layers(self, config, model, decode):
        with forward(config, CUDAGraphMode.NONE, self.decode_metadata(config, model)):
            assert decode.plan(self._positions(decode), object()) is None

    def test_width_off_the_capture_sizes_runs_vllm_layers(self, config, model, decode):
        off = torch.zeros(max(decode.widths) + 1, dtype=torch.long)
        with forward(config, CUDAGraphMode.NONE, self.decode_metadata(config, model)):
            assert decode.plan(off) is None

    def test_piecewise_runs_vllm_layers(self, config, model, decode):
        meta = self.decode_metadata(config, model)
        with forward(config, CUDAGraphMode.PIECEWISE, meta):
            assert decode.plan(self._positions(decode)) is None

    def test_unbound_caches_run_vllm_layers(self, config, model):
        decode = self.make_decode(config, model, bind=False)
        assert self._plan(config, model, decode) is None

    def test_caches_checked_and_handed_over_on_every_bind(self, config, model, decode):
        checks: list = []
        taken = _record(decode.kernel, "bind_caches")
        decode.spec = _spy(decode.spec, "refuse_caches", checks)
        decode.bind_caches()
        assert len(checks) == len(taken) == 1
        assert set(taken[0][0]) == set(decode.kernel.cache_layers)
        sfc = config.compilation_config.static_forward_context
        for name in decode.kernel.cache_layers:
            sfc[name].kv_cache = sfc[name].kv_cache.clone()
        decode.bind_caches()
        assert len(checks) == len(taken) == 2

    def test_refuse_caches_reports_every_reason(self, config, model, decode):
        reasons = decode.spec.refuse_caches(
            decode.vllm_config, decode.kernel, self.bad_caches(config, model)
        )
        assert len(reasons) >= 2, reasons

    def test_mismatched_caches_refuse_the_bind(self, config, model, decode):
        sfc = config.compilation_config.static_forward_context
        taken = _record(decode.kernel, "bind_caches")
        bad = self.bad_caches(config, model)
        for name, cache in zip(decode.kernel.cache_layers, bad.values(), strict=False):
            sfc[name].kv_cache = cache
        with pytest.raises(DecodeUnsupported):
            decode.bind_caches()
        assert not taken

    def test_build_does_not_move_weights(self, config):
        model = self.make_model(config)
        before = {n: p.data_ptr() for n, p in model.named_parameters()}
        self.make_decode(config, model)
        after = {n: p.data_ptr() for n, p in model.named_parameters()}
        assert before == after

    def test_build_runs_once_over_every_width(self, config, model):
        calls: list = []
        spec = _spy(select_decode_spec(config, self.specs), "build", calls)
        decode = SpecializedDecode(model, config, spec)
        assert [args[1] for args in calls] == [model]
        assert set(calls[0][2]) == set(decode.widths)

    def test_warmup_registered_for_every_key(self, config):
        registry = JitWarmupRegistry(config)
        with registry.activate():
            decode = self.make_decode(config)
        keys = decode.kernel.warmup_keys()
        assert len(keys) == len(set(keys)) >= 1
        assert len(registry) == len(keys)
        registry.warmup()

    def _positions(self, decode) -> torch.Tensor:
        return torch.zeros(min(decode.widths), dtype=torch.long)

    def _plan(self, config, model, decode):
        with forward(config, CUDAGraphMode.NONE, self.decode_metadata(config, model)):
            return decode.plan(self._positions(decode))
