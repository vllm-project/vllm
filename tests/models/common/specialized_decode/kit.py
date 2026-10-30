# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What a model's ``DecodeSpec`` must satisfy."""

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.models.common.specialized_decode.dispatch import SpecializedDecode
from vllm.models.common.specialized_decode.guards import select_decode_spec

KERNEL_BACKENDS = ("aiter", "flydsl")
"""Backends a spec module must not pull in of its own: it is imported at config
time, before anything has decided the spec can run. The package holding it is
the model's own hardware entry point, which already pulls its platform's
backend, so only what the spec module adds on top of that package counts."""


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


class DecodeConformance:
    """What every model's ``DecodeSpec`` must satisfy. Subclass it in the
    model's test module and fill in the hooks below."""

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

    def test_spec_module_imports_no_kernel_backend(self):
        module = self.specs_module
        package = module.rpartition(".")[0]
        source = (
            "import sys; sys.path.insert(0, '.'); import importlib;"
            f"importlib.import_module({package!r});"
            "loaded = set(sys.modules);"
            f"importlib.import_module({module!r});"
            "print('\\n'.join(set(sys.modules) - loaded))"
        )
        loaded = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        pulled = {n.split(".")[0] for n in loaded} & set(KERNEL_BACKENDS)
        assert not pulled, f"{module} imports {sorted(pulled)}"

    def test_refuse_caches_reports_every_reason(self, config, model, decode):
        reasons = decode.spec.refuse_caches(
            decode.vllm_config, decode.kernel, self.bad_caches(config, model)
        )
        assert len(reasons) >= 2, reasons

    def test_build_does_not_move_weights(self, config):
        model = self.make_model(config)
        before = {n: p.data_ptr() for n, p in model.named_parameters()}
        self.make_decode(config, model)
        after = {n: p.data_ptr() for n, p in model.named_parameters()}
        assert before == after
