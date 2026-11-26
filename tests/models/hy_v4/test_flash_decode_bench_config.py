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
    DFlySpeculatorConfig.apply_options(spec)  # type: ignore[arg-type]


def test_dfly_speculator_config_rejects_unknown_rejection_method():
    spec = SimpleNamespace(
        enable_adaptive_verification=False,
        rejection_sample_method="typical_acceptance",
    )
    with pytest.raises(
        ValueError, match="DFly requires standard or synthetic target verification"
    ):
        DFlySpeculatorConfig.apply_options(spec)  # type: ignore[arg-type]


def test_dfly_speculator_allows_decode_bench_connector_and_rejects_pd(monkeypatch):
    def make_cfg(connector: str) -> SimpleNamespace:
        return SimpleNamespace(
            cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"),
            speculative_config=SimpleNamespace(kv_cache_dtype=None),
            kv_transfer_config=SimpleNamespace(kv_connector=connector),
            lora_config=None,
            parallel_config=SimpleNamespace(use_ubatching=False),
        )

    with pytest.raises(ValueError, match="DFly does not support PD disaggregation"):
        dfly_speculator.HYV4DFlySpeculator(
            make_cfg("NixlConnector"),  # type: ignore[arg-type]
            torch.device("cpu"),
        )

    # Reaching DraftModelSpeculator.__init__ proves the guard passed for
    # DecodeBenchConnector.
    reached = {"called": False}

    def _stop_init(self, vllm_config, device):
        reached["called"] = True
        raise RuntimeError("sentinel")

    monkeypatch.setattr(dfly_speculator.DraftModelSpeculator, "__init__", _stop_init)
    with pytest.raises(RuntimeError, match="sentinel"):
        dfly_speculator.HYV4DFlySpeculator(
            make_cfg("DecodeBenchConnector"),  # type: ignore[arg-type]
            torch.device("cpu"),
        )
    assert reached["called"]
