# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for online weight-only NVFP4 lm_head quantization (a
`quantization_config.targets` pattern naming the lm_head, with the
`"nvfp4_weight_only"` shorthand).

Run `pytest tests/quantization/test_online_nvfp4_lm_head.py`.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.quantization.online import fp8 as online_fp8
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.lm_head import (
    quantized_lm_head_copy,
)
from vllm.model_executor.layers.quantization.online.nvfp4 import (
    Nvfp4WeightOnlyOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kNvfp4Static,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_processing,
)
from vllm.platforms import current_platform

SCHEME = "nvfp4_weight_only"
LM_HEAD_TARGET = {"lm_head": SCHEME}
# Not a multiple of the vocab padding, so the head has zero-filled rows.
VOCAB, HIDDEN = 4000, 2048

requires_blackwell = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(100),
    reason="The NVFP4 W4A16 linear kernel needs a Blackwell GPU",
)


def _resolve(spec):
    config = OnlineQuantizationConfig(QuantizationConfigArgs(targets={"lm_head": spec}))
    return config.resolve_quant_method_cls(ParallelLMHead, "lm_head")


def test_target_naming_the_head_selects_weight_only_nvfp4():
    resolved = _resolve(SCHEME)
    assert resolved is not None
    assert resolved[0].value == "targets" and resolved[2] == "lm_head"
    assert resolved[-1] is Nvfp4WeightOnlyOnlineLinearMethod
    assert resolved[3].activation is None  # W4A16


def test_activation_quantization_is_rejected():
    with pytest.raises(NotImplementedError, match="activation override"):
        _resolve(QuantSpec(weight=kNvfp4Static, activation=kFp8DynamicTokenSym))


@pytest.fixture
def online_head_config(dist_init, monkeypatch):
    # Online linear methods read the model dtype from the current config.
    monkeypatch.setattr(
        online_fp8,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    config = OnlineQuantizationConfig(QuantizationConfigArgs(targets=LM_HEAD_TARGET))
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        yield config


@requires_blackwell
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_head_matches_the_drafter_copy(online_head_config, num_tokens):
    """The quantized head computes the same logits as the NVFP4 copy that
    `draft_lm_head_quantization` derives from a bf16 head."""
    torch.manual_seed(0)
    loaded = (torch.randn(VOCAB, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    head = ParallelLMHead(
        VOCAB,
        HIDDEN,
        params_dtype=torch.bfloat16,
        quant_config=online_head_config,
        prefix="lm_head",
        disable_tp=True,
    )
    assert isinstance(head.quant_method, Nvfp4WeightOnlyOnlineLinearMethod)
    head.weight.weight_loader(head.weight, loaded)
    finalize_layerwise_processing(head, model_config=None)
    assert head.weight.dtype != torch.bfloat16  # quantized, in the kernel layout

    source = ParallelLMHead(
        VOCAB, HIDDEN, params_dtype=torch.bfloat16, prefix="lm_head", disable_tp=True
    )
    source.weight.data.zero_()
    source.weight.data[:VOCAB] = loaded
    copy = quantized_lm_head_copy(source, "nvfp4")

    x = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    out = head.quant_method.apply(head, x)
    assert out.shape == (num_tokens, head.num_embeddings_per_partition)
    torch.testing.assert_close(out, copy.quant_method.apply(copy, x), rtol=0, atol=0)
    ref = x.float() @ loaded.float().t()
    err = (out[:, :VOCAB].float() - ref).norm() / ref.norm()
    assert err < 0.15, err
    # The padded rows quantize from zeros to zero logits.
    assert torch.all(out[:, VOCAB:] == 0)
