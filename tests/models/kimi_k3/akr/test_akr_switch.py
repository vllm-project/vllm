# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""kdamoe sub-switches (CPU only): route gating, VLLM_KIMI_AKR_<NAME> env, defaults."""

import pytest

from vllm.models.kimi_k3.amd import akr_config, akr_switch


@pytest.fixture(autouse=True)
def _routes(monkeypatch):
    monkeypatch.setattr(akr_config, "_routes", frozenset(akr_config.ROUTES))
    for n in akr_switch.SWITCHES:
        monkeypatch.delenv(f"VLLM_KIMI_AKR_{n}", raising=False)


def test_defaults():
    for name in akr_switch.SWITCHES:
        assert akr_switch.on(name) == (name not in akr_switch.DEFAULT_OFF), name


def test_env_off_and_on(monkeypatch):
    monkeypatch.setenv("VLLM_KIMI_AKR_ROUTER", "0")
    assert not akr_switch.on("ROUTER")
    monkeypatch.setenv("VLLM_KIMI_AKR_K1", "1")
    assert akr_switch.on("K1")


def test_route_off_wins(monkeypatch):
    monkeypatch.setattr(akr_config, "_routes", frozenset(akr_config.ROUTES) - {"moe"})
    assert not akr_switch.on("EXPERTS") and not akr_switch.on("LATENT_AR")
    assert akr_switch.on("KDA_DECODE")
    monkeypatch.setenv("VLLM_KIMI_AKR_K2", "1")
    assert not akr_switch.on("K2")  # K2 is on the moe route
