# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 oracle selection between the FlashInfer CuTe-DSL W4A4 and W4A16
experts. The SM10x device check is patched, so this covers selection only."""

import dataclasses

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLExperts,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_w4a16_moe import (  # noqa: E501
    FlashInferCuteDSLW4A16Experts,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)

pytest.importorskip("flashinfer.fused_moe.cute_dsl.tuner")


@pytest.fixture(autouse=True)
def _cutedsl_device(monkeypatch):
    monkeypatch.setattr(
        FlashInferCuteDSLExperts, "_supports_current_device", staticmethod(lambda: True)
    )


def _config(moe_backend="flashinfer_cutedsl", hidden_dim=2688):
    config = make_dummy_moe_config(
        num_experts=128,
        experts_per_token=6,
        hidden_dim=hidden_dim,
        intermediate_size=1856,
        activation=MoEActivation.RELU2_NO_MUL,
    )
    return dataclasses.replace(config, moe_backend=moe_backend)


def test_explicit_cutedsl_picks_w4a16_experts_without_activation_scales():
    assert select_nvfp4_moe_backend(_config(), kNvfp4Static, None) == (
        NvFp4MoeBackend.FLASHINFER_CUTEDSL,
        FlashInferCuteDSLW4A16Experts,
    )


def test_explicit_cutedsl_keeps_w4a4_experts_for_nvfp4_activations():
    assert select_nvfp4_moe_backend(_config(), kNvfp4Static, kNvfp4Dynamic) == (
        NvFp4MoeBackend.FLASHINFER_CUTEDSL,
        FlashInferCuteDSLExperts,
    )


def test_auto_selection_never_picks_w4a16_experts():
    try:
        _, experts_cls = select_nvfp4_moe_backend(
            _config(moe_backend="auto"), kNvfp4Static, None
        )
    except NotImplementedError:
        return
    assert experts_cls is not FlashInferCuteDSLW4A16Experts


def test_explicit_cutedsl_error_lists_every_candidate():
    with pytest.raises(ValueError) as excinfo:
        select_nvfp4_moe_backend(_config(hidden_dim=4096), kNvfp4Static, None)
    message = str(excinfo.value)
    assert "requires Lightning H2688/I1856/E128/top6" in message
    assert "; " in message
