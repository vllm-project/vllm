# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DSA_REPLICATE_ATTN_MIN_TOKENS routes small steps head-sharded only where the
sliced F.linear projections and the backend's head count are both valid."""

from types import SimpleNamespace

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.deepseek_v32.attention import tp_routing_blockers
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
)


class _FixedHeadsImpl:
    """A backend sized for the head count it was built with (e.g. SM90 FlashInfer)."""


def _proj(quantized: bool = False):
    method = object() if quantized else UnquantizedLinearMethod()
    return SimpleNamespace(quant_method=method)


def test_sm120_with_bf16_projections_may_route():
    impl = object.__new__(FlashInferMLASparseSM120Impl)
    assert tp_routing_blockers(impl, (_proj(), _proj())) == []


def test_quantized_projection_blocks_routing():
    impl = object.__new__(FlashInferMLASparseSM120Impl)
    blockers = tp_routing_blockers(impl, (_proj(), _proj(quantized=True)))
    assert blockers == ["quantized q_b_proj/o_proj"]


def test_fixed_head_backend_blocks_routing():
    blockers = tp_routing_blockers(_FixedHeadsImpl(), (_proj(), _proj()))
    assert len(blockers) == 1 and "_FixedHeadsImpl" in blockers[0]
