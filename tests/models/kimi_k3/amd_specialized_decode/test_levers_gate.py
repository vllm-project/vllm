# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The levers' gate (``specialized_decode.levers``): every lever flag defaults
to 1 but reads as off until ``decode_backend=latency_specialized`` activates
the gate (``SpecializedDecodeMixin.init_specialized_decode``); the stock decode
neither turns it on nor installs the v1 patches, and turns it back off."""

import pytest

from vllm.models.kimi_k3.amd.specialized_decode import levers
from vllm.models.kimi_k3.amd.specialized_decode.hooks import SpecializedDecodeMixin


@pytest.fixture
def gate(monkeypatch):
    """The gate off and nothing installed, restored after the test."""
    monkeypatch.setattr(levers, "_active", False)
    monkeypatch.setattr(levers, "_installed", False)
    for name in levers.LEVERS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_levers_off_without_the_gate(gate):
    assert not levers.is_active()
    assert not any(levers.lever_on(name) for name in levers.LEVERS)
    for name in levers.LEVERS:
        gate.setenv(name, "1")
    assert not any(levers.lever_on(name) for name in levers.LEVERS)


def test_levers_default_on_under_the_gate(gate):
    levers.activate(install_patches=False)
    assert levers.is_active() and not levers.installed()
    assert all(levers.lever_on(name) for name in levers.LEVERS)
    for name in levers.LEVERS:
        gate.setenv(name, "0")
        assert not levers.lever_on(name), name
        gate.delenv(name)
    levers.deactivate()
    assert not any(levers.lever_on(name) for name in levers.LEVERS)


def test_stock_decode_leaves_the_gate_off(gate):
    installs = []
    gate.setattr(levers, "install", lambda: installs.append(True))
    model = object.__new__(SpecializedDecodeMixin)
    model.init_specialized_decode(None)
    assert not levers.is_active() and installs == []
    # Any DecodeConfig (only latency_specialized makes one) turns it on.
    model.init_specialized_decode(object())
    assert levers.is_active() and installs == [True]
    # A stock model built afterwards in the same process turns it back off.
    model.init_specialized_decode(None)
    assert not levers.is_active()
    assert not any(levers.lever_on(name) for name in levers.LEVERS)


def test_failed_install_leaves_the_gate_off(gate):
    def install():
        raise RuntimeError("renamed v1 method")

    gate.setattr(levers, "install", install)
    with pytest.raises(RuntimeError):
        levers.activate()
    assert not levers.is_active()


def test_wrap_once():
    class Owner:
        def f(self):
            return 0

    def make(stock):
        return lambda self: stock(self) + 1

    levers.wrap(Owner, "f", make)
    levers.wrap(Owner, "f", make)
    assert levers.is_wrapped(Owner.f) and Owner().f() == 1


# Run in a fresh interpreter: another test in the session may have built a
# latency_specialized model.
_UNPATCHED = """
import importlib

import vllm.models.kimi_k3.amd.kda_metadata  # noqa: F401
import vllm.models.kimi_k3.amd.specialized_decode.hooks  # noqa: F401
import vllm.models.kimi_k3.amd.specialized_decode.model_state  # noqa: F401
from vllm.models.kimi_k3.amd.specialized_decode import levers

assert not levers.is_active() and not levers.installed()
assert not any(levers.lever_on(name) for name in levers.LEVERS)
for module, owner, name in (
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", "execute_model"),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", "prepare_inputs"),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", "sample"),
    ("vllm.v1.worker.gpu.spec_decode.dspark.speculator", "DSparkSpeculator",
     "_sample_sequential"),
    ("vllm.v1.worker.gpu.spec_decode.dflash.speculator", "DFlashSpeculator",
     "set_attn"),
    ("vllm.v1.worker.gpu.spec_decode.dflash.speculator", "DFlashSpeculator",
     "load_draft_model"),
    ("vllm.model_executor.layers.logits_processor", "LogitsProcessor",
     "_gather_logits"),
    ("vllm.model_executor.models.qwen3_dflash", "DFlashQwen3Model",
     "precompute_and_store_context_kv"),
    ("vllm.v1.attention.backends.mla.rocm_aiter_mla", "AiterMLAMetadataBuilder",
     "_build_decode"),
    ("vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse",
     "ROCMAiterMLASparseMetadataBuilder", "build"),
):
    cls = getattr(importlib.import_module(module), owner)
    assert not levers.is_wrapped(getattr(cls, name)), (owner, name)
try:
    import aiter
except ImportError:
    aiter = None
if aiter is not None and hasattr(aiter, "get_mla_metadata_v1"):
    assert not hasattr(aiter.get_mla_metadata_v1, "aiter_planner")
"""


def test_nothing_patched_without_the_gate():
    """The Kimi-K3 modules import (as a stock c16 / c48 deployment does) with
    every lever flag at its default 1, and no v1 method is patched."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in levers.LEVERS}
    subprocess.run([sys.executable, "-c", _UNPATCHED], env=env, check=True)
