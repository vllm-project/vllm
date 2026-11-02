# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deployment checks of the MiniMax-M3.1 DSpark drafter kernels that need no GPU.

The kernels are built for 7 query rows a request at TP4; any other deployment
must keep the stock draft layers, as must a ROCm install without FlyDSL. These
tests run in the ROCm images (where FlyDSL is installed); elsewhere they are
skipped.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("flydsl")

from vllm.models.minimax_m3.amd.mono.drafter import dispatch  # noqa: E402


def _vllm_config(tp=4, dp=1, pp=1, method="dspark", k=7):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, data_parallel_size=dp, pipeline_parallel_size=pp
        ),
        speculative_config=SimpleNamespace(method=method, num_speculative_tokens=k),
    )


@pytest.fixture(autouse=True)
def _gfx950(monkeypatch):
    monkeypatch.setattr(dispatch, "on_gfx950", lambda: True)
    monkeypatch.setattr(dispatch, "_flydsl_available", lambda: True)


def test_supported_deployment():
    assert dispatch._config_refusal(_vllm_config()) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"tp": 8},
        {"dp": 2},
        {"pp": 2},
        {"method": "eagle3"},
        {"k": 3},
    ],
)
def test_other_deployments_keep_the_stock_layers(overrides):
    assert dispatch._config_refusal(_vllm_config(**overrides)) is not None


def test_no_speculative_config():
    config = _vllm_config()
    config.speculative_config = None
    assert dispatch._config_refusal(config) is not None


def test_not_gfx950(monkeypatch):
    monkeypatch.setattr(dispatch, "on_gfx950", lambda: False)
    assert dispatch._config_refusal(_vllm_config()) is not None


def test_no_flydsl(monkeypatch):
    monkeypatch.setattr(dispatch, "_flydsl_available", lambda: False)
    assert dispatch._config_refusal(_vllm_config()) is not None
