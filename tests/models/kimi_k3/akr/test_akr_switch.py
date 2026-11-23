# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""kdamoe sub-switches (CPU only): route gating, VLLM_KIMI_AKR_<NAME> env, defaults;
the KDA prefill hook's switch and mixed-step gate."""

from types import SimpleNamespace

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


def test_kda_prefill_default_on(monkeypatch):
    from vllm.models.kimi_k3.amd import akr_kda_prefill

    monkeypatch.delenv("VLLM_KIMI_AKR_KDA_PREFILL", raising=False)
    assert akr_kda_prefill._env_on()
    for v in ("0", "false", "off", "no"):
        monkeypatch.setenv("VLLM_KIMI_AKR_KDA_PREFILL", v)
        assert not akr_kda_prefill._env_on(), v
    monkeypatch.setenv("VLLM_KIMI_AKR_KDA_PREFILL", "1")
    assert akr_kda_prefill._env_on()


@pytest.mark.parametrize(
    "num_decodes,num_prefill_tokens,spec,expected",
    [
        (0, 64, False, None),  # pure prefill: always the kernel
        (0, 16384, False, None),
        (1, 2048, False, "short mixed"),  # decodes + a short prefill: stock
        (40, 10239, False, "short mixed"),
        (1, 10240, False, None),  # decodes + a long prefill: the kernel
        (12, 16000, False, None),
        (0, 4096, True, "spec"),  # spec-verify rows: stock
    ],
)
def test_kda_prefill_mixed_step_gate(
    monkeypatch, num_decodes, num_prefill_tokens, spec, expected
):
    """The hook declines decode + prefill steps below MIXED_MIN_PREFILL_TOKENS prefill
    tokens (measured slower than stock there) and keeps pure prefill steps."""
    from vllm.models.kimi_k3.amd import akr_kda_prefill as ak

    monkeypatch.delenv("VLLM_KIMI_AKR_KDA_PREFILL", raising=False)
    assert ak.MIXED_MIN_PREFILL_TOKENS == 10240
    m = SimpleNamespace(
        num_decodes=num_decodes,
        num_prefill_tokens=num_prefill_tokens,
        spec_sequence_masks=object() if spec else None,
        akr_chunk_counts=[1],
        chunk_indices=object(),
    )
    assert ak._unsupported(m) == expected
    monkeypatch.setattr(akr_config, "_routes", frozenset(akr_config.ROUTES) - {"kda"})
    assert ak._unsupported(m) == "off"
