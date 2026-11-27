# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HY4-Flash DecodeBenchConnector and DFly synthetic acceptance."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.hy_v4.common import dfly_speculator
from vllm.v1.worker.gpu.spec_decode.dfly.config import DFlySpeculatorConfig


@pytest.mark.parametrize("method", ["standard", "synthetic"])
def test_dfly_speculator_config_allows_standard_and_synthetic(method: str):
    spec = SimpleNamespace(
        enable_adaptive_verification=False,
        rejection_sample_method=method,
    )
    DFlySpeculatorConfig.apply_options(spec)


def test_dfly_speculator_config_rejects_unknown_rejection_method():
    spec = SimpleNamespace(
        enable_adaptive_verification=False,
        rejection_sample_method="typical_acceptance",
    )
    with pytest.raises(
        ValueError, match="DFly requires standard or synthetic target verification"
    ):
        DFlySpeculatorConfig.apply_options(spec)


@pytest.mark.parametrize(
    ("connector", "allowed"),
    [
        ("DecodeBenchConnector", True),
        ("NixlConnector", True),
        ("MooncakeConnector", True),
        ("LMCacheConnectorV1", False),
    ],
)
def test_dfly_speculator_kv_connector_guard(monkeypatch, connector: str, allowed: bool):
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"),
        speculative_config=SimpleNamespace(kv_cache_dtype=None),
        kv_transfer_config=SimpleNamespace(kv_connector=connector),
        lora_config=None,
        parallel_config=SimpleNamespace(use_ubatching=False),
    )

    # Reaching DraftModelSpeculator.__init__ proves the guard passed.
    def _stop_init(self, vllm_config, device):
        raise RuntimeError("sentinel")

    monkeypatch.setattr(dfly_speculator.DraftModelSpeculator, "__init__", _stop_init)
    match = "sentinel" if allowed else "only with NixlConnector or"
    with pytest.raises((RuntimeError, ValueError), match=match):
        dfly_speculator.HYV4DFlySpeculator(
            cfg,  # type: ignore[arg-type]
            torch.device("cpu"),
        )
