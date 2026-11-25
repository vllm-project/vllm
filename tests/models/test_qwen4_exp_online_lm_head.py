# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp builds its lm_head with the quant config only for an online
`quantization_config.targets` entry naming it. The ModelOpt checkpoint
excludes the head, so without a target it stays unquantized and takes the
low-latency GEMM.

Run `pytest tests/models/test_qwen4_exp_online_lm_head.py`.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.model_executor.layers.quantization.online import fp8 as online_fp8
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.models.qwen4_exp.nvidia.model import _online_lm_head_quant_config

# The shape of the Qwen3.8-Flash-Next-NVFP4 hf_quant_config: NVFP4 experts,
# the lm_head excluded.
CHECKPOINT = {
    "quantization": {
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": None,
        "group_size": 16,
        "exclude_modules": ["lm_head", "model.language_model.embed_tokens"],
        "quantized_layers": {
            "model.language_model.layers.0.mlp.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            }
        },
    }
}


def _checkpoint_config(targets):
    args = None if targets is None else QuantizationConfigArgs(targets=targets)
    config = ModelOptMixedPrecisionConfig.from_config(
        {**CHECKPOINT, "_online_quantization_args": args}
    )
    if args is not None:
        config.online_quantization_config = OnlineQuantizationConfig(args)
    return config


def test_no_target_leaves_the_head_without_quant_config():
    assert _online_lm_head_quant_config(_checkpoint_config(None), "lm_head") is None
    assert _online_lm_head_quant_config(None, "lm_head") is None


def test_target_on_another_layer_leaves_the_head_alone():
    config = _checkpoint_config(
        {"re:.*o_proj$": QuantSpec(weight="fp8_per_channel_static")}
    )
    assert _online_lm_head_quant_config(config, "lm_head") is None


@pytest.fixture
def bf16_model(dist_init, monkeypatch):
    # Online linear methods read the model dtype from the current config.
    monkeypatch.setattr(
        online_fp8,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    with set_current_vllm_config(VllmConfig()), torch.device("meta"):
        yield


def _head(quant_config):
    return ParallelLMHead(
        248320,
        2560,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix="lm_head",
        disable_tp=True,
    )


def test_target_quantizes_the_excluded_head(bf16_model):
    config = _checkpoint_config(
        {"re:.*lm_head$": QuantSpec(weight="fp8_per_channel_static")}
    )
    head_config = _online_lm_head_quant_config(config, "lm_head")
    assert head_config is config
    assert isinstance(_head(head_config).quant_method, Fp8PtpcOnlineLinearMethod)


def test_head_without_target_is_unquantized(bf16_model):
    config = _checkpoint_config(None)
    head = _head(_online_lm_head_quant_config(config, "lm_head"))
    # The type the low-latency GEMM swap requires.
    assert type(head.quant_method) is UnquantizedEmbeddingMethod
