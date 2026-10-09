# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parameter dtype contract of the Qwen3.5 GDN linear attention layer.

The fused CUDA GDN decoder accepts only an FP32 A_log and does not fall back
(csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu), so the A_log dtype must
not follow the checkpoint series. Upcasting the BF16 A_log of a Qwen3.6
checkpoint to FP32 keeps those values exact, so nothing is lost by pinning it.

The norm weight is different: Qwen3.5 stores it in FP32 and Qwen3.6 in BF16.
Both series ship configs that are identical in every other field, so the layer
can only learn the series from the optional real_model_type override. Without
it a Qwen3.5 norm weight is truncated to the model dtype on load.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.utils.torch_utils import set_default_torch_dtype

PREFIX = "model.layers.0.linear_attn"
MODEL_DTYPE = torch.bfloat16


def _build_layer(real_model_type: str | None) -> QwenGatedDeltaNetAttention:
    config = Qwen3_5TextConfig(real_model_type=real_model_type)
    vllm_config = VllmConfig()
    # Construction only reads the dtype and the text config.
    vllm_config.model_config = SimpleNamespace(
        dtype=MODEL_DTYPE,
        hf_config=config,
        hf_text_config=config,
    )
    # Model loading allocates parameters under the model dtype, so mirror that
    # rather than the torch default of float32.
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(MODEL_DTYPE):
        return QwenGatedDeltaNetAttention(
            config=config,
            vllm_config=vllm_config,
            prefix=PREFIX,
        )


@pytest.mark.parametrize("real_model_type", [None, "qwen3_5", "qwen3_6"])
def test_a_log_dtype_is_always_fp32(dist_init, real_model_type: str | None) -> None:
    # Deriving this dtype from the checkpoint series used to hand a BF16 A_log
    # to the fused CUDA decoder, which asserts FP32 at call time.
    layer = _build_layer(real_model_type)
    assert layer.A_log.dtype == torch.float32


@pytest.mark.parametrize(
    ("real_model_type", "expected_norm_dtype"),
    [
        (None, MODEL_DTYPE),
        ("qwen3_5", torch.float32),
        ("qwen3_6", MODEL_DTYPE),
    ],
)
def test_norm_dtype_matches_checkpoint_series(
    dist_init, real_model_type: str | None, expected_norm_dtype: torch.dtype
) -> None:
    layer = _build_layer(real_model_type)
    assert layer.norm.weight.dtype == expected_norm_dtype
