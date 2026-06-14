# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP8 MoE backend selection (oracle/fp8.py).

Regression coverage for https://github.com/vllm-project/vllm/issues/45101:
with LoRA enabled, w8a8 backends (e.g. Triton) feed fp8-quantized activations
into the Triton MoE-LoRA shrink kernel, which crashes on the mixed
``fp8 x bf16`` ``tl.dot``. The selection oracle must route FP8 MoE + LoRA to
the Marlin (W8A16) backend, whose activations stay in the original dtype.
"""

import dataclasses

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

skipif_not_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Only supported on CUDA platforms.",
)


@skipif_not_cuda
@pytest.mark.parametrize(
    "weight_key,activation_key",
    [
        # Block-quantized FP8 checkpoint (e.g. Qwen3.5/3.6-MoE-FP8).
        (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        # Per-tensor FP8 checkpoint.
        (kFp8StaticTensorSym, kFp8DynamicTokenSym),
    ],
)
def test_fp8_moe_with_lora_selects_marlin(weight_key, activation_key):
    """FP8 MoE + LoRA must select the W8A16 Marlin backend so the MoE-LoRA
    kernels receive unquantized (bf16/fp16) activations."""
    moe_config = dataclasses.replace(
        make_dummy_moe_config(
            num_experts=8,
            experts_per_token=2,
            hidden_dim=2048,
            intermediate_size=512,
        ),
        is_lora_enabled=True,
    )

    backend, experts_cls = select_fp8_moe_backend(
        config=moe_config,
        weight_key=weight_key,
        activation_key=activation_key,
        allow_vllm_cutlass=False,
    )

    assert backend == Fp8MoeBackend.MARLIN
    assert experts_cls is MarlinExperts
    assert experts_cls.supports_lora()
