# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the online weight-only lm_head formats (a
`quantization_config.targets` pattern naming the lm_head, with
`{"weight": "int4_per_group_32_asym"}`, `{"weight": "int8_per_channel_static"}`
or `{"weight": "int8_per_group_32"}`).

Run `pytest tests/quantization/test_online_lm_head.py`.
"""

from collections.abc import Callable
from dataclasses import dataclass
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
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8PerChannelLMHeadMethod,
    Int8PerGroup32LMHeadMethod,
    quantize_int8_per_group_sym,
    quantize_int8_to_mp_layout,
)
from vllm.model_executor.layers.quantization.online.lm_head import (
    OnlineLMHeadMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static32Asym,
    kInt8Static32Sym,
    kInt8StaticChannelSym,
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

# Not a multiple of the vocab padding, so the head has zero-filled rows.
VOCAB, HIDDEN = 4000, 2048


@dataclass(frozen=True)
class Scheme:
    name: str
    key: QuantKey
    method: type[OnlineLMHeadMethodBase]
    quantize: Callable
    """`weight` [N, K] -> codes [N, K], scales and zero points (or None)."""
    to_mp_layout: Callable
    """`(weight, group_size, chunk_rows)` -> the `MPLinearKernel` layout."""
    max_error_steps: float
    """Quantizer error bound, in quantization steps."""
    full_precision_tol: float
    """Bound on the logit error against the full-precision lm_head."""

    @property
    def target(self) -> dict[str, QuantSpec]:
        return {"lm_head": QuantSpec(weight=self.name)}


SCHEMES = [
    Scheme(
        "int4_per_group_32_asym",
        kInt4Static32Asym,
        Int4PerGroup32AsymLMHeadMethod,
        lambda w: quantize_int4_per_group_asym(w, 32),
        quantize_int4_to_mp_layout,
        # Half a step of rounding, plus up to half a step at the group's ends
        # from rounding the zero point.
        max_error_steps=1.0,
        full_precision_tol=0.1,
    ),
    Scheme(
        "int8_per_channel_static",
        kInt8StaticChannelSym,
        Int8PerChannelLMHeadMethod,
        lambda w: (*quantize_int8_per_group_sym(w, -1), None),
        quantize_int8_to_mp_layout,
        max_error_steps=0.5,
        full_precision_tol=0.015,
    ),
    Scheme(
        "int8_per_group_32",
        kInt8Static32Sym,
        Int8PerGroup32LMHeadMethod,
        lambda w: (*quantize_int8_per_group_sym(w, 32), None),
        quantize_int8_to_mp_layout,
        max_error_steps=0.5,
        # Tighter than per-channel scales meet.
        full_precision_tol=0.0075,
    ),
]
SCHEME_IDS = [scheme.name for scheme in SCHEMES]
parametrize_schemes = pytest.mark.parametrize("scheme", SCHEMES, ids=SCHEME_IDS)


def _dequantize(codes, scales, zero_points):
    group_size = codes.shape[1] // scales.shape[1]
    scale = scales.float().repeat_interleave(group_size, dim=1)
    if zero_points is None:
        return codes.float() * scale
    zero_point = zero_points.float().repeat_interleave(group_size, dim=1)
    return (codes.float() - zero_point) * scale


def _kernel_available(scheme: Scheme) -> bool:
    config = MPLinearLayerConfig(
        full_weight_shape=(HIDDEN, VOCAB),
        partition_weight_shape=(HIDDEN, 4032),
        weight_type=scheme.method.weight_type,
        act_type=torch.bfloat16,
        group_size=scheme.method.group_size,
        zero_points=scheme.method.zero_points,
    )
    try:
        choose_mp_linear_kernel(config)
    except (ValueError, RuntimeError):
        return False
    return torch.cuda.is_available()


parametrize_kernel_schemes = pytest.mark.parametrize(
    "scheme",
    [
        pytest.param(
            scheme,
            marks=pytest.mark.skipif(
                not _kernel_available(scheme),
                reason=f"No mixed-precision linear kernel for {scheme.name}",
            ),
        )
        for scheme in SCHEMES
    ],
    ids=SCHEME_IDS,
)


# ---------------------------------------------------------------- config


@parametrize_schemes
def test_name_maps_to_weight_key(scheme):
    assert QuantSpec(weight=scheme.name).weight == scheme.key
    args = resolve_quantization_config(
        "online", {"targets": {"lm_head": {"weight": scheme.name}}}
    )
    assert args is not None and args.targets == scheme.target


@parametrize_schemes
def test_target_naming_the_head_selects_it(scheme):
    config = OnlineQuantizationConfig(QuantizationConfigArgs(targets=scheme.target))
    resolved = config.resolve_quant_method_cls(ParallelLMHead, "lm_head")
    assert resolved is not None
    assert resolved[0].value == "targets" and resolved[2] == "lm_head"
    assert resolved[-1] is scheme.method
    assert resolved[3].activation is None  # weight-only
    assert config.resolve_quant_method_cls(VocabParallelEmbedding, "embed") is None
    assert config.resolve_quant_method_cls(ReplicatedLinear, "proj") is None


@parametrize_schemes
def test_broad_targets_do_not_select_the_head(scheme):
    # A catch-all pattern keeps meaning the linear and MoE layers.
    args = QuantizationConfigArgs(targets={"re:.*": "fp8_per_tensor"})
    config = OnlineQuantizationConfig(args)
    assert config.resolve_quant_method_cls(ParallelLMHead, "lm_head") is None
    # It composes with a pattern naming the head.
    args = QuantizationConfigArgs(
        targets={
            "re:.*proj$": "fp8_per_tensor",
            r"re:.*lm_head$": QuantSpec(weight=scheme.name),
        }
    )
    config = OnlineQuantizationConfig(args)
    resolved = config.resolve_quant_method_cls(ParallelLMHead, "lm_head")
    assert resolved is not None and resolved[-1] is scheme.method
    resolved = config.resolve_quant_method_cls(ReplicatedLinear, "o_proj")
    assert resolved is not None and resolved[-1] is not scheme.method


def test_head_in_targets_and_ignore_is_rejected():
    args = QuantizationConfigArgs(targets=SCHEMES[0].target, ignore=["lm_head"])
    config = OnlineQuantizationConfig(args)
    with pytest.raises(ValueError, match="matches both"):
        config.resolve_quant_method_cls(ParallelLMHead, "lm_head")


@parametrize_schemes
@pytest.mark.parametrize("source", ["linear", "broad_target"])
def test_lm_head_only_weight_is_rejected_on_linear_layers(scheme, source):
    spec = QuantSpec(weight=scheme.name)
    if source == "linear":
        args = QuantizationConfigArgs(linear=spec)
    else:
        args = QuantizationConfigArgs(targets={"re:.*": spec})
    config = OnlineQuantizationConfig(args)
    with pytest.raises(ValueError, match="quantizes only the lm_head"):
        config.resolve_quant_method_cls(ReplicatedLinear, "model.layers.0.o_proj")


def test_unsupported_lm_head_key_is_rejected():
    config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={"lm_head": QuantSpec(weight="lut_b")})
    )
    with pytest.raises(ValueError, match="not supported"):
        config.resolve_quant_method_cls(ParallelLMHead, "lm_head")


@parametrize_schemes
@pytest.mark.parametrize("checkpoint_quantized", [False, True])
def test_checkpoint_lm_head_method(default_vllm_config, scheme, checkpoint_quantized):
    # Online methods read the model dtype from the current config.
    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    checkpoint_method = object() if checkpoint_quantized else None
    quant_config = SimpleNamespace(
        online_quantization_config=OnlineQuantizationConfig(
            QuantizationConfigArgs(targets=scheme.target)
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
        assert isinstance(method, scheme.method)


# ------------------------------------------------------------- quantizer


@parametrize_schemes
def test_quantizer_error_bound(scheme):
    torch.manual_seed(0)
    weight = (torch.randn(64, 256) * 0.05).to(torch.bfloat16)
    codes, scales, zero_points = scheme.quantize(weight)
    assert scheme.method.weight_type.min() <= codes.min()
    assert codes.max() <= scheme.method.weight_type.max()
    if zero_points is not None:
        assert zero_points.max() <= scheme.method.weight_type.max()
    deq = _dequantize(codes, scales, zero_points)
    step = scales.float().repeat_interleave(weight.shape[1] // scales.shape[1], 1)
    assert torch.all(
        (deq - weight.float()).abs() <= scheme.max_error_steps * step + 1e-6
    )


@parametrize_schemes
def test_quantizer_keeps_exact_zero(scheme):
    weight = torch.zeros(8, 64, dtype=torch.float16)
    weight[0, 0] = 1.0
    deq = _dequantize(*scheme.quantize(weight))
    assert torch.all(deq[weight == 0] == 0)


@parametrize_schemes
def test_mp_layout_round_trip(scheme):
    torch.manual_seed(0)
    weight = torch.randn(64, 256, dtype=torch.bfloat16)
    codes, scales, zero_points = scheme.quantize(weight)
    packed, packed_scales, packed_zp = scheme.to_mp_layout(
        weight, scheme.method.group_size, chunk_rows=24
    )
    pack_factor = 32 // scheme.method.weight_type.size_bits
    assert packed.shape == (64, 256 // pack_factor) and packed.dtype == torch.int32
    unpacked = unpack_quantized_values_into_int32(
        packed, scheme.method.weight_type, packed_dim=1
    )
    stored = codes.to(torch.int32) + scheme.method.weight_type.bias
    torch.testing.assert_close(unpacked, stored, rtol=0, atol=0)
    torch.testing.assert_close(packed_scales, scales, rtol=0, atol=0)
    if zero_points is None:
        assert packed_zp is None
        return
    assert packed_zp.shape == (64 // pack_factor, scales.shape[1])
    unpacked_zp = unpack_quantized_values_into_int32(
        packed_zp, scheme.method.weight_type, packed_dim=0
    )
    torch.testing.assert_close(unpacked_zp, zero_points.to(torch.int32), rtol=0, atol=0)


# ----------------------------------------------------------------- layer


@pytest.fixture
def online_head_config(dist_init, monkeypatch, scheme):
    # Online linear methods read the model dtype from the current config.
    monkeypatch.setattr(
        online_fp8,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    config = OnlineQuantizationConfig(QuantizationConfigArgs(targets=scheme.target))
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        yield config


def _rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def _reference_logits(scheme, x, weight):
    return x.float() @ _dequantize(*scheme.quantize(weight)).t()


@parametrize_kernel_schemes
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_untied_head_matches_reference(online_head_config, scheme, num_tokens):
    torch.manual_seed(0)
    head = ParallelLMHead(
        VOCAB,
        HIDDEN,
        params_dtype=torch.bfloat16,
        quant_config=online_head_config,
        prefix="lm_head",
        disable_tp=True,
    )
    assert isinstance(head.quant_method, scheme.method)
    assert head.weight.is_meta  # materialized while loading
    loaded = (torch.randn(VOCAB, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    head.weight.weight_loader(head.weight, loaded)
    # The padded vocab rows are never loaded: processing waits for finalize.
    finalize_layerwise_processing(head, model_config=None)
    assert head.weight.dtype == torch.int8 or head.weight.dtype == torch.int32
    x = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    out = head.quant_method.apply(head, x)
    assert out.shape == (num_tokens, head.num_embeddings_per_partition)
    err = _rel_err(out[:, :VOCAB], _reference_logits(scheme, x, loaded))
    assert err < 1e-2, err
    full_err = _rel_err(out[:, :VOCAB], x.float() @ loaded.float().t())
    assert full_err < scheme.full_precision_tol, full_err
    # The padded rows quantize from zeros to zero logits.
    assert torch.all(out[:, VOCAB:] == 0)


@parametrize_kernel_schemes
def test_tied_head_keeps_the_embedding(online_head_config, scheme):
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
    err = _rel_err(out[:, :VOCAB], _reference_logits(scheme, x, original[:VOCAB]))
    assert err < 1e-2, err
