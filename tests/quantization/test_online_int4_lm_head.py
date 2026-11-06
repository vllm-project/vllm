# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for online int4 lm_head quantization (a `quantization_config.targets`
pattern naming the lm_head, with `{"weight": "int4_per_group_32_asym"}`).

Run `pytest tests/quantization/test_online_int4_lm_head.py`.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.quantization import (
    QuantizationConfigArgs,
    QuantSpec,
    resolve_quantization_config,
)
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import resolve_quant_method
from vllm.model_executor.layers.quantization.online import fp8 as online_fp8
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.int4 import (
    Int4PerGroup32AsymLMHeadMethod,
    quantize_int4_per_group_asym,
    quantize_int4_to_mp_layout,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt4Static32Asym,
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_processing,
)
from vllm.scalar_type import scalar_types

SCHEME = "int4_per_group_32_asym"
GROUP = 32
LM_HEAD_TARGET = {"lm_head": QuantSpec(weight=SCHEME)}
# Not a multiple of the vocab padding, so the head has zero-filled rows.
VOCAB, HIDDEN = 4000, 2048


def _dequantize(codes, scales, zero_points, group_size=GROUP):
    scale = scales.float().repeat_interleave(group_size, dim=1)
    zero_point = zero_points.float().repeat_interleave(group_size, dim=1)
    return (codes.float() - zero_point) * scale


def _kernel_available() -> bool:
    config = MPLinearLayerConfig(
        full_weight_shape=(HIDDEN, VOCAB),
        partition_weight_shape=(HIDDEN, 4032),
        weight_type=scalar_types.uint4,
        act_type=torch.bfloat16,
        group_size=GROUP,
        zero_points=True,
    )
    try:
        choose_mp_linear_kernel(config)
    except (ValueError, RuntimeError):
        return False
    return torch.cuda.is_available()


requires_kernel = pytest.mark.skipif(
    not _kernel_available(),
    reason="No mixed-precision linear kernel for uint4 g32 with zero points",
)


# ---------------------------------------------------------------- config


def test_name_maps_to_asymmetric_key():
    assert QuantSpec(weight=SCHEME).weight == kInt4Static32Asym
    args = resolve_quantization_config(
        "online", {"targets": {"lm_head": {"weight": SCHEME}}}
    )
    assert args is not None and args.targets == LM_HEAD_TARGET


def test_target_naming_the_head_selects_it():
    config = OnlineQuantizationConfig(QuantizationConfigArgs(targets=LM_HEAD_TARGET))
    resolved = config.resolve_quant_method_cls(ParallelLMHead, "lm_head")
    assert resolved is not None
    assert resolved[0].value == "targets" and resolved[2] == "lm_head"
    assert resolved[-1] is Int4PerGroup32AsymLMHeadMethod
    assert resolved[3].activation is None  # weight-only
    assert config.resolve_quant_method_cls(VocabParallelEmbedding, "embed") is None
    assert config.resolve_quant_method_cls(ReplicatedLinear, "proj") is None


def test_broad_targets_do_not_select_the_head():
    # A catch-all pattern keeps meaning the linear and MoE layers.
    args = QuantizationConfigArgs(targets={"re:.*": "fp8_per_tensor"})
    config = OnlineQuantizationConfig(args)
    assert config.resolve_quant_method_cls(ParallelLMHead, "lm_head") is None
    # It composes with a pattern naming the head.
    args = QuantizationConfigArgs(
        targets={"re:.*proj$": "fp8_per_tensor", r"re:.*lm_head$": {"weight": SCHEME}}
    )
    config = OnlineQuantizationConfig(args)
    resolved = config.resolve_quant_method_cls(ParallelLMHead, "lm_head")
    assert resolved is not None and resolved[-1] is Int4PerGroup32AsymLMHeadMethod
    resolved = config.resolve_quant_method_cls(ReplicatedLinear, "o_proj")
    assert resolved is not None and resolved[-1] is not Int4PerGroup32AsymLMHeadMethod


def test_head_in_targets_and_ignore_is_rejected():
    args = QuantizationConfigArgs(targets=LM_HEAD_TARGET, ignore=["lm_head"])
    config = OnlineQuantizationConfig(args)
    with pytest.raises(ValueError, match="matches both"):
        config.resolve_quant_method_cls(ParallelLMHead, "lm_head")


@pytest.mark.parametrize(
    "args",
    [
        QuantizationConfigArgs(linear=QuantSpec(weight=SCHEME)),
        QuantizationConfigArgs(targets={"re:.*": {"weight": SCHEME}}),
    ],
    ids=["linear", "broad_target"],
)
def test_int4_asym_is_rejected_on_linear_layers(args):
    config = OnlineQuantizationConfig(args)
    with pytest.raises(ValueError, match="quantizes only the lm_head"):
        config.resolve_quant_method_cls(ReplicatedLinear, "model.layers.0.o_proj")


def test_unsupported_lm_head_key_is_rejected():
    config = OnlineQuantizationConfig(
        QuantizationConfigArgs(
            targets={"lm_head": QuantSpec(weight="int8_per_channel_static")}
        )
    )
    with pytest.raises(ValueError, match="not supported"):
        config.resolve_quant_method_cls(ParallelLMHead, "lm_head")


@pytest.mark.parametrize("checkpoint_quantized", [False, True])
def test_checkpoint_lm_head_method(default_vllm_config, checkpoint_quantized):
    # Online methods read the model dtype from the current config.
    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    checkpoint_method = object() if checkpoint_quantized else None
    quant_config = SimpleNamespace(
        online_quantization_config=OnlineQuantizationConfig(
            QuantizationConfigArgs(targets=LM_HEAD_TARGET)
        ),
        packed_modules_mapping={},
        get_quant_method=lambda layer, prefix: checkpoint_method,
    )
    # Only the layer type is used, so skip building the head.
    head = ParallelLMHead.__new__(ParallelLMHead)
    if checkpoint_quantized:
        with pytest.raises(NotImplementedError, match="already quantized"):
            resolve_quant_method(quant_config, head, "lm_head")
    else:
        method = resolve_quant_method(quant_config, head, "lm_head")
        assert isinstance(method, Int4PerGroup32AsymLMHeadMethod)


# ------------------------------------------------------------- quantizer


def test_quantizer_error_bound():
    torch.manual_seed(0)
    weight = (torch.randn(64, 256) * 0.05).to(torch.bfloat16)
    codes, scales, zero_points = quantize_int4_per_group_asym(weight, GROUP)
    assert codes.max() <= 15 and zero_points.max() <= 15
    deq = _dequantize(codes, scales, zero_points)
    # Half a step of rounding, plus up to half a step at the group's ends from
    # rounding the zero point.
    step = scales.float().repeat_interleave(GROUP, dim=1)
    assert torch.all((deq - weight.float()).abs() <= step + 1e-6)


def test_quantizer_keeps_exact_zero():
    weight = torch.zeros(8, 64, dtype=torch.float16)
    weight[0, 0] = 1.0
    codes, scales, zero_points = quantize_int4_per_group_asym(weight, GROUP)
    deq = _dequantize(codes, scales, zero_points)
    assert torch.all(deq[weight == 0] == 0)


def test_mp_layout_round_trip():
    torch.manual_seed(0)
    weight = torch.randn(64, 256, dtype=torch.bfloat16)
    codes, scales, zero_points = quantize_int4_per_group_asym(weight, GROUP)
    packed, packed_scales, packed_zp = quantize_int4_to_mp_layout(
        weight, GROUP, chunk_rows=24
    )
    assert packed.shape == (64, 256 // 8) and packed.dtype == torch.int32
    assert packed_zp.shape == (64 // 8, 256 // GROUP)
    unpacked = unpack_quantized_values_into_int32(
        packed, scalar_types.uint4, packed_dim=1
    )
    unpacked_zp = unpack_quantized_values_into_int32(
        packed_zp, scalar_types.uint4, packed_dim=0
    )
    torch.testing.assert_close(unpacked, codes.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(unpacked_zp, zero_points.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(packed_scales, scales, rtol=0, atol=0)


# ----------------------------------------------------------------- layer


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


def _reference_logits(x, weight):
    codes, scales, zero_points = quantize_int4_per_group_asym(weight, GROUP)
    return x.float() @ _dequantize(codes, scales, zero_points).t()


@requires_kernel
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_untied_head_matches_reference(online_head_config, num_tokens):
    torch.manual_seed(0)
    head = ParallelLMHead(
        VOCAB,
        HIDDEN,
        params_dtype=torch.bfloat16,
        quant_config=online_head_config,
        prefix="lm_head",
        disable_tp=True,
    )
    assert isinstance(head.quant_method, Int4PerGroup32AsymLMHeadMethod)
    assert head.weight.is_meta  # materialized while loading
    loaded = (torch.randn(VOCAB, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    head.weight.weight_loader(head.weight, loaded)
    # The padded vocab rows are never loaded: processing waits for finalize.
    finalize_layerwise_processing(head, model_config=None)
    assert head.weight.dtype == torch.int8 or head.weight.dtype == torch.int32
    x = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    out = head.quant_method.apply(head, x)
    assert out.shape == (num_tokens, head.num_embeddings_per_partition)
    ref = _reference_logits(x, loaded)
    err = (out[:, :VOCAB].float() - ref).norm() / ref.norm()
    assert err < 1e-2, err
    # The padded rows quantize from zeros to zero logits.
    assert torch.all(out[:, VOCAB:] == 0)


@requires_kernel
def test_tied_head_keeps_the_embedding(online_head_config):
    torch.manual_seed(0)
    embed = VocabParallelEmbedding(
        VOCAB, HIDDEN, params_dtype=torch.bfloat16, disable_tp=True
    )
    assert isinstance(embed.quant_method, UnquantizedEmbeddingMethod)
    embed.weight.data.normal_(std=0.02)
    original = embed.weight.data.clone()
    head = ParallelLMHead(
        VOCAB,
        HIDDEN,
        params_dtype=torch.bfloat16,
        quant_config=online_head_config,
        prefix="lm_head",
        disable_tp=True,
    )
    head.tie_weights(embed)
    # The model loader's process_weights_after_loading pass.
    head.quant_method.process_weights_after_loading(head)
    assert head.weight is embed.weight
    torch.testing.assert_close(embed.weight.data, original, rtol=0, atol=0)
    tokens = torch.arange(16, device="cuda")
    torch.testing.assert_close(embed(tokens), original[tokens], rtol=0, atol=0)
    x = torch.randn(4, HIDDEN, device="cuda", dtype=torch.bfloat16)
    out = head.quant_method.apply(head, x)
    ref = _reference_logits(x, original[:VOCAB])
    err = (out[:, :VOCAB].float() - ref).norm() / ref.norm()
    assert err < 1e-2, err
