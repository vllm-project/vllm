# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``VLLM_K31_STOCK_LEVERS``: the levers' gate for the stock decode (no
``latency_specialized``). Unset (the default) the stock decode keeps every
lever off and installs nothing, as before; set to 1 it installs the v1 patches
and turns on only the audited ``levers.STOCK_SAFE`` levers."""

import pytest

from vllm.models.kimi_k3.amd.specialized_decode import levers
from vllm.models.kimi_k3.amd.specialized_decode.hooks import SpecializedDecodeMixin

# The audited allow-list (analysis/impl/C.md). Changing it needs a new audit.
AUDITED = {
    "VLLM_K31_BEAT_MARKOV",
    "VLLM_K31_BEAT_DRAFTMETA",
    "VLLM_K31_BEAT_EAGERPREP",
    "VLLM_K31_BEAT_DRAFTFC",
    "VLLM_K31_BEAT_LOGITSAG",
    "VLLM_K31_BEAT_MARKOVTP",
    "VLLM_K31_BEAT_CTXKV",
}


@pytest.fixture
def gate(monkeypatch):
    """The gate off, nothing installed and every flag at its default,
    restored after the test."""
    monkeypatch.setattr(levers, "_active", False)
    monkeypatch.setattr(levers, "_stock", False)
    monkeypatch.setattr(levers, "_installed", False)
    for name in (*levers.LEVERS, levers.STOCK_LEVERS):
        monkeypatch.delenv(name, raising=False)
    installs: list[bool] = []
    monkeypatch.setattr(levers, "install", lambda: installs.append(True))
    monkeypatch.installs = installs
    return monkeypatch


def _build(config):
    model = object.__new__(SpecializedDecodeMixin)
    model.init_specialized_decode(config)
    return model


def test_allow_list_is_the_audited_one():
    assert set(levers.STOCK_SAFE) == AUDITED
    assert set(levers.STOCK_SAFE) <= set(levers.LEVERS)
    # Read only by the specialized runtime, or not audited.
    for name in (
        levers.K2SYNC,
        levers.KDAK1,
        levers.PROLOGUE_INDEXER,
        levers.SAMPLED_TOPK,
    ):
        assert name not in levers.STOCK_SAFE


@pytest.mark.parametrize("value", [None, "0", "", "true", "2"])
def test_stock_decode_unchanged_unless_set_to_1(gate, value):
    if value is not None:
        gate.setenv(levers.STOCK_LEVERS, value)
    _build(None)
    assert not levers.is_active() and not levers.is_stock()
    assert gate.installs == []
    assert not any(levers.lever_on(name) for name in levers.LEVERS)


def test_stock_levers_allow_list_only(gate):
    gate.setenv(levers.STOCK_LEVERS, "1")
    _build(None)
    assert levers.is_active() and levers.is_stock()
    assert gate.installs == [True]
    for name in levers.LEVERS:
        assert levers.lever_on(name) == (name in levers.STOCK_SAFE), name
    # Explicitly set flags do not reach past the allow-list.
    for name in levers.LEVERS:
        gate.setenv(name, "1")
    for name in levers.LEVERS:
        assert levers.lever_on(name) == (name in levers.STOCK_SAFE), name


def test_stock_levers_keep_their_own_flags(gate):
    gate.setenv(levers.STOCK_LEVERS, "1")
    _build(None)
    for name in levers.STOCK_SAFE:
        gate.setenv(name, "0")
        assert not levers.lever_on(name), name
        assert all(levers.lever_on(n) for n in levers.STOCK_SAFE - {name})
        gate.delenv(name)


def test_specialized_decode_ignores_stock_mode(gate):
    """latency_specialized turns every lever on, with or without the env."""
    gate.setenv(levers.STOCK_LEVERS, "1")
    _build(object())
    assert levers.is_active() and not levers.is_stock()
    assert all(levers.lever_on(name) for name in levers.LEVERS)


def test_rebuilds_switch_modes(gate):
    gate.setenv(levers.STOCK_LEVERS, "1")
    _build(None)
    assert levers.is_stock()
    _build(object())
    assert levers.is_active() and not levers.is_stock()
    gate.delenv(levers.STOCK_LEVERS)
    _build(None)
    assert not levers.is_active() and not levers.is_stock()
    assert not any(levers.lever_on(name) for name in levers.LEVERS)


def test_stock_levers_logged(gate):
    """The mode logs the levers it turned on (most of them log nothing)."""
    logged: list[tuple] = []
    gate.setattr(levers.logger, "info_once", lambda *args: logged.append(args))
    gate.setenv(levers.STOCK_LEVERS, "1")
    gate.setenv(levers.MARKOVTP, "0")
    _build(None)
    gate.delenv(levers.STOCK_LEVERS, raising=False)
    _build(None)
    assert len(logged) == 1
    fmt, *args = logged[0]
    names = (fmt % tuple(args)).split(": ", 2)[-1].split(", ")
    assert names == [n for n in levers.LEVERS if n in AUDITED - {levers.MARKOVTP}]


def test_activate_stock_without_patches(gate):
    levers.activate_stock(install_patches=False)
    assert gate.installs == []
    assert levers.is_active() and levers.is_stock()
    levers.deactivate()
    assert not levers.is_active() and not levers.is_stock()


def test_failed_install_leaves_the_stock_gate_off(gate):
    def install():
        raise RuntimeError("renamed v1 method")

    gate.setattr(levers, "install", install)
    gate.setenv(levers.STOCK_LEVERS, "1")
    with pytest.raises(RuntimeError):
        _build(None)
    assert not levers.is_active() and not levers.is_stock()


# A fresh interpreter: the real install, with VLLM_K31_STOCK_LEVERS=1.
_STOCK_INSTALL = """
import importlib

from vllm.models.kimi_k3.amd.specialized_decode import levers
from vllm.models.kimi_k3.amd.specialized_decode.hooks import SpecializedDecodeMixin

assert not levers.is_active() and not levers.installed()
model = object.__new__(SpecializedDecodeMixin)
model.init_specialized_decode(None)
assert levers.is_active() and levers.is_stock() and levers.installed()
for module, owner, name in (
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", "execute_model"),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", "sample"),
    ("vllm.v1.worker.gpu.spec_decode.dspark.speculator", "DSparkSpeculator",
     "_sample_sequential"),
    ("vllm.model_executor.layers.logits_processor", "LogitsProcessor",
     "_gather_logits"),
):
    cls = getattr(importlib.import_module(module), owner)
    assert levers.is_wrapped(getattr(cls, name)), (owner, name)
assert not levers.lever_on(levers.K2SYNC)
assert levers.lever_on(levers.DRAFTFC)
"""


def test_stock_levers_install_the_patches():
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in levers.LEVERS}
    env[levers.STOCK_LEVERS] = "1"
    subprocess.run([sys.executable, "-c", _STOCK_INSTALL], env=env, check=True)
