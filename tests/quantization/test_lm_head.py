# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether gptq models with quantized lm_head can be loaded.

Run `pytest tests/quantization/test_quant_lm_head_true.py --forked`.
"""

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import preset_name_to_scheme

from tests.quantization.utils import load_model_without_vllm_runner
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

PROMPT = "On the surface of Mars, we found"

MODELS_QUANT = [
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit", False),
]


@pytest.mark.parametrize("model_id, lm_head_quantized", MODELS_QUANT)
def test_lm_head(
    model_id: str,
    lm_head_quantized: bool,
    monkeypatch,
    dist_init,
    workspace_init,
) -> None:
    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    model, _ = load_model_without_vllm_runner(
        model_id,
        dtype=torch.float16,
        model_config_kwargs={
            "max_model_len": 2048,
            "hf_overrides": {"num_hidden_layers": 3},
        },
    )
    lm_head_layer = model.lm_head
    if lm_head_quantized:
        assert isinstance(lm_head_layer.quant_method, AutoGPTQLinearMethod)
    else:
        assert isinstance(lm_head_layer.quant_method, UnquantizedEmbeddingMethod)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_humming(), reason="requires Humming/CUDA"
)
@pytest.mark.parametrize(
    "preset,quant_format",
    [
        ("FP8_DYNAMIC", "float-quantized"),
        ("NVFP4A16", "nvfp4-pack-quantized"),
        ("W4A16", "pack-quantized"),
        ("W4A16_NVFP4", None),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_quantized_lm_head_matches_linear(
    dist_init, default_vllm_config, preset, quant_format, bias
):
    """An explicitly quantized head must produce the same logits as a linear layer."""
    default_vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16, head_dtype=None
    )
    default_vllm_config.kernel_config.linear_backend = "humming"
    if quant_format is None:
        quant_config = ModelOptNvFp4Config(
            quant_method=preset,
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        )
    else:
        scheme = preset_name_to_scheme(preset, targets=["Linear", "lm_head"])
        quant_config = CompressedTensorsConfig.from_config(
            {"config_groups": {"group_0": scheme.model_dump()}, "format": quant_format}
        )
    with torch.device("cuda"):
        kwargs = dict(
            bias=bias,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            disable_tp=True,
        )
        head = ParallelLMHead(500, 256, prefix="lm_head", **kwargs)
        linear = ColumnParallelLinear(256, 512, prefix="proj", **kwargs)
        for name, param in linear.named_parameters():
            if "scale" in name:
                param.fill_(1.0)
            elif name == "weight_shape":
                param.copy_(torch.tensor([512, 256]))
            elif param.is_floating_point():
                param.copy_(torch.randn(param.shape, dtype=torch.float32))
            else:
                param.random_(0, 127)
        head.load_state_dict(linear.state_dict())
        for layer in (head, linear):
            layer.quant_method.process_weights_after_loading(layer)
        x = torch.randn(8, 256, dtype=torch.bfloat16)
        expected, _ = linear(x)
        actual = LogitsProcessor(500)(head, x, head.bias)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected[:, :500], rtol=0, atol=0)
