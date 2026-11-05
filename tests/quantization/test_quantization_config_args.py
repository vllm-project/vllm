# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for QuantizationConfigArgs parsing."""

import pytest

from tests.quantization.utils import quant_config_args, quant_spec
from vllm.config import ModelConfig
from vllm.config.quantization import (
    QUANT_KEY_NAMES,
    QuantizationConfigArgs,
    QuantSpec,
    resolve_quantization_config,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
    _find_matching_targets,
)
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kInt4Static32,
    kInt8StaticChannelSym,
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4DynamicTensor,
    kNvfp4DynamicToken,
    kNvfp4Static,
)

# ---- QuantSpec ------------------------------------------------------------


def test_quant_spec_resolves_string_to_quant_key():
    spec = quant_spec(weight="mxfp8", activation="fp8_per_token")
    assert spec.weight == kMxfp8Dynamic
    assert spec.activation == kFp8DynamicTokenSym


def test_quant_spec_accepts_quant_key_directly():
    spec = QuantSpec(weight=kFp8StaticTensorSym)
    assert spec.weight is kFp8StaticTensorSym
    assert spec.activation is None


def test_quant_spec_string_representation_uses_quantization_name():
    assert str(quant_spec(weight="mxfp4")) == "mxfp4"


def test_quant_spec_resolves_groupwise_int4_weight():
    spec = QuantSpec(weight="int4_per_group_32")
    assert spec.weight == kInt4Static32
    assert spec.activation is None


def test_quant_spec_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown quantization name"):
        quant_spec(weight="not_a_real_format")


# ---- QuantizationConfigArgs string shorthand on linear/moe ----------------


def test_args_linear_string_resolves_via_quant_key_names():
    # A bare QUANT_KEY_NAMES entry desugars to QuantSpec(weight=<key>).
    args = quant_config_args(linear="fp8_per_block_static")
    assert args.linear == QuantSpec(weight=kFp8Static128BlockSym)
    assert args.moe is None


@pytest.mark.parametrize(
    "shorthand,weight",
    [("fp8_per_block", kFp8Static128BlockSym), ("nvfp4_per_token", kNvfp4Static)],
)
def test_args_moe_string_resolves_via_online_shorthand(shorthand, weight):
    # An online-shorthand name pulls the matching slot from _ONLINE_SHORTHANDS
    # (so `linear: "fp8_per_block"` and `moe: "fp8_per_block"` produce the
    # same per-layer-kind spec the `--quantization fp8_per_block` shorthand
    # would).
    args = quant_config_args(moe=shorthand)
    assert args.moe == QuantSpec(weight=weight)


def test_args_string_shorthand_missing_slot_raises():
    # int8_per_channel_weight_only sets only `moe`; using it on `linear`
    # has no defined spec and should raise rather than silently no-op.
    with pytest.raises(ValueError, match="does not define a linear spec"):
        quant_config_args(linear="int8_per_channel_weight_only")


@pytest.mark.parametrize(
    "activation,expected",
    [("mxfp8", kMxfp8Dynamic), ("nvfp4_per_token", kNvfp4DynamicToken)],
)
def test_args_accepts_dict_form(activation, expected):
    args = quant_config_args(moe={"activation": activation})
    assert args.moe == QuantSpec(weight=None, activation=expected)
    assert args.linear is None


def test_targets_reject_non_string_keys():
    with pytest.raises(ValueError, match="targets keys must be strings"):
        QuantizationConfigArgs._validate_targets({123: "mxfp8"})


# ---- resolve_quantization_config -----------------------------------------


def test_resolve_shorthand_only_populates_both_slots():
    args = resolve_quantization_config("fp8_per_block", None)
    assert args is not None
    assert args.linear == QuantSpec(weight=kFp8Static128BlockSym)
    assert args.moe == QuantSpec(weight=kFp8Static128BlockSym)


@pytest.mark.parametrize("quantization", ["mxfp4", "mxfp8"])
def test_resolve_colliding_shorthand_is_deferred(quantization: str):
    """Checkpoint metadata determines whether an MXFP shorthand is online."""
    assert resolve_quantization_config(quantization, None) is None


@pytest.mark.parametrize(
    "shorthand,weight",
    [
        ("int8_per_channel_weight_only", kInt8StaticChannelSym),
        ("nvfp4_per_token", kNvfp4Static),
    ],
)
def test_resolve_moe_only_shorthand_leaves_linear_unset(shorthand, weight):
    args = resolve_quantization_config(shorthand, None)
    assert args is not None
    assert args.linear is None
    assert args.moe == QuantSpec(weight=weight)


@pytest.mark.parametrize("quantization", [None, "modelopt_fp4"])
@pytest.mark.parametrize(
    "activation,expected",
    [("mxfp8", kMxfp8Dynamic), ("nvfp4_per_token", kNvfp4DynamicToken)],
)
def test_resolve_activation_override(quantization, activation, expected):
    # When only `quantization_config` is given (e.g. for an already-quantized
    # checkpoint that needs an activation override), it's returned as-is.
    args = resolve_quantization_config(
        quantization, {"moe": {"activation": activation}}
    )
    assert args is not None
    assert args.linear is None
    assert args.moe == QuantSpec(weight=None, activation=expected)


def test_resolve_merges_explicit_over_shorthand():
    # Explicit linear in quantization_config wins; moe falls back to the
    # shorthand's slot.
    args = resolve_quantization_config(
        "fp8_per_tensor",
        {"linear": "fp8_per_block"},
    )
    assert args is not None
    assert args.linear == QuantSpec(weight=kFp8Static128BlockSym)
    assert args.moe == QuantSpec(weight=kFp8StaticTensorSym)


def test_resolve_quantization_config_with_checkpoint_quantization():
    args = resolve_quantization_config("gptq", {"linear": "fp8_per_block"})
    assert args == quant_config_args(linear="fp8_per_block")


# ---- QuantSpec.phase -----------------------------------------------------

_PHASE = {"decode": "unquantized", "prefill": "nvfp4", "max_decode_tokens": 1024}

_U, _NVFP4, _MXFP8 = None, kNvfp4DynamicTensor, kMxfp8Dynamic


@pytest.mark.parametrize(
    "decode,prefill,modes,scale",
    [
        ("unquantized", "nvfp4", (_U, _NVFP4), "dynamic_tensor"),
        ("nvfp4", "nvfp4", (_NVFP4, _NVFP4), "dynamic_tensor"),
        ("nvfp4", "unquantized", (_NVFP4, _U), "dynamic_tensor"),
        ("unquantized", "unquantized", (_U, _U), None),
        ("unquantized", "mxfp8", (_U, _MXFP8), None),
        ("mxfp8", "mxfp8", (_MXFP8, _MXFP8), None),
        ("mxfp8", "unquantized", (_MXFP8, _U), None),
    ],
)
def test_phase_resolves_activation_modes(decode, prefill, modes, scale):
    phase = quant_spec(phase={**_PHASE, "decode": decode, "prefill": prefill}).phase
    assert phase is not None
    assert phase.activation_modes() == modes
    assert phase.scale == scale
    assert phase.quantized_format == next(
        (f for f in (decode, prefill) if f != "unquantized"), None
    )


def test_phase_defaults_and_optional_cutoff():
    phase = quant_spec(phase={**_PHASE, "scale": "dynamic_tensor"}).phase
    assert (
        phase == quant_spec(phase={"prefill": "nvfp4", "max_decode_tokens": 1024}).phase
    )
    # One format for both phases needs no cutoff.
    for fmt in ("nvfp4", "mxfp8", "unquantized"):
        phase = quant_spec(phase={"decode": fmt, "prefill": fmt}).phase
        assert phase is not None and phase.max_decode_tokens is None


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"max_decode_tokens": 0}, "greater than or equal to 1"),
        ({"max_decode_tokens": None}, "max_decode_tokens is required"),
        (
            {"decode": "mxfp8", "max_decode_tokens": None},
            "max_decode_tokens is required",
        ),
        ({"chunk": 1}, "Unexpected keyword argument"),
        ({"decode": "nvfp4", "prefill": "mxfp8"}, "need two weight copies"),
        ({"decode": "mxfp8", "prefill": "nvfp4"}, "need two weight copies"),
        (
            {"decode": "fp8_per_token"},
            "phase.decode\n.*'unquantized', 'nvfp4' or 'mxfp8'",
        ),
        (
            {"prefill": "nvfp4_per_token"},
            "phase.prefill\n.*'unquantized', 'nvfp4' or 'mxfp8'",
        ),
        ({"scale": "dynamic_token"}, "phase.scale\n.*'dynamic_tensor'"),
        (
            {"prefill": "mxfp8", "scale": "dynamic_tensor"},
            "scale only applies to nvfp4",
        ),
        (
            {"prefill": "unquantized", "scale": "dynamic_tensor"},
            "scale only applies to nvfp4",
        ),
    ],
)
def test_phase_rejects_unserved_specs(overrides, match):
    with pytest.raises(ValueError, match=match):
        quant_spec(phase={**_PHASE, **overrides})


@pytest.mark.parametrize("field", ["weight", "activation"])
def test_nvfp4_is_only_a_phase_name(field):
    with pytest.raises(ValueError, match="unknown quantization name 'nvfp4'"):
        quant_spec(**{field: "nvfp4"})


def test_phase_is_exclusive_with_activation_and_linear_only():
    with pytest.raises(ValueError, match="mutually exclusive"):
        quant_spec(activation="nvfp4_per_token", phase=_PHASE)
    with pytest.raises(ValueError, match="moe.phase is not supported"):
        quant_config_args(moe={"phase": _PHASE})


def test_online_quantization_rejects_phase():
    config = OnlineQuantizationConfig(
        quant_config_args(linear={"weight": "fp8_per_channel_static", "phase": _PHASE})
    )
    with pytest.raises(
        NotImplementedError, match="phase .* is not supported for online"
    ):
        config.resolve_quant_method_cls(LinearBase, "model.layers.0.self_attn.o_proj")


def test_online_mxfp8_accepts_phase():
    """Online MXFP8 linears select their kernel like checkpoint MXFP8 ones, so
    the phase applies to them."""
    phase = {"decode": "unquantized", "prefill": "mxfp8", "max_decode_tokens": 64}
    config = OnlineQuantizationConfig(
        quant_config_args(linear={"weight": "mxfp8", "phase": phase})
    )
    resolved = config.resolve_quant_method_cls(
        LinearBase, "model.layers.0.self_attn.o_proj"
    )
    assert resolved is not None
    assert resolved[-1] is Mxfp8OnlineLinearMethod
    assert resolved[-2].phase == quant_spec(phase=phase).phase


@pytest.mark.parametrize(
    "shorthand,linear",
    [
        ("mxfp4", {"activation": None, "weight": None}),
        ("fp8_per_tensor", {}),
        ("fp8_per_tensor", {"weight": None}),
    ],
)
def test_shorthand_respects_explicitly_unquantized_linear(shorthand, linear):
    """An explicit linear spec without a phase keeps linears unquantized, as in
    the MoE-only recipes; only a phase-only spec inherits the weight."""
    args = resolve_quantization_config(shorthand, {"linear": linear})
    assert args is not None and args.linear is not None
    assert args.linear.weight is None


def test_phase_requires_prefill():
    with pytest.raises(ValueError, match="phase.prefill is required"):
        quant_spec(phase={"decode": "mxfp8", "max_decode_tokens": 8})


def test_mxfp8_shorthand_keeps_weight_with_phase():
    """`--quantization mxfp8` with only `linear.phase` set must still quantize
    the linears to MXFP8."""
    phase = {"decode": "unquantized", "prefill": "mxfp8", "max_decode_tokens": 64}
    args = resolve_quantization_config("mxfp8", {"linear": {"phase": phase}})
    assert args is not None and args.linear is not None
    assert args.linear.weight == kMxfp8Dynamic
    assert args.linear.phase == quant_spec(phase=phase).phase
    resolved = OnlineQuantizationConfig(args).resolve_quant_method_cls(
        LinearBase, "model.layers.0.self_attn.o_proj"
    )
    assert resolved is not None
    assert resolved[-1] is Mxfp8OnlineLinearMethod
    # Other linear fields survive the inherited weight.
    args = resolve_quantization_config(
        "mxfp8", {"linear": {"phase": phase, "hadamard_size": 32}}
    )
    assert args is not None and args.linear is not None
    assert (args.linear.weight, args.linear.hadamard_size) == (kMxfp8Dynamic, 32)
    args = resolve_quantization_config("mxfp4_hadamard32", {"linear": {"phase": phase}})
    assert args is not None and args.linear is not None
    assert args.linear.hadamard_size == 32


def test_phase_enters_model_config_hash():
    """ModelConfig.compute_hash keys the compile cache, so it must change with
    the phase."""

    def config_hash(linear):
        args = None if linear is None else quant_config_args(linear=linear)
        return ModelConfig("facebook/opt-125m", quantization_config=args).compute_hash()

    base = config_hash({"phase": _PHASE})
    assert base == config_hash({"phase": dict(_PHASE)})
    assert base != config_hash(None)
    assert base != config_hash({"phase": {**_PHASE, "max_decode_tokens": 2048}})


# ---- QUANT_KEY_NAMES coverage --------------------------------------------


def test_quant_key_names_round_trip():
    # Every advertised name should round-trip through QuantSpec without error
    # and produce the same QuantKey it maps to.
    for name, expected in QUANT_KEY_NAMES.items():
        assert quant_spec(weight=name).weight == expected, name
        assert quant_spec(activation=name).activation == expected, name


def test_static_block_weight_paired_with_dynamic_block_activation():
    # The block-FP8 shorthand pair: 128x128 static weights + 1x128 dynamic
    # activations. Pinning this so renames in QUANT_KEY_NAMES don't quietly
    # rewire the kernel dispatch.
    spec = quant_spec(weight="fp8_per_block_static", activation="fp8_per_block_dynamic")
    assert spec.weight == kFp8Static128BlockSym
    assert spec.activation == kFp8Dynamic128Sym


def test_targets_allow_distinct_patterns_with_the_same_shorthand():
    layer_name = "model.layers.0.self_attn.qkv_proj"
    fused_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    targets = {
        r"re:.*\.q_proj$": "mxfp8",
        r"re:.*\.k_proj$": "mxfp8",
        r"re:.*\.v_proj$": "mxfp8",
    }

    matches = _find_matching_targets(layer_name, targets, fused_mapping)
    assert len(matches) == 1
    assert targets[matches[0]] == "mxfp8"


def test_targets_reject_overlapping_patterns():
    targets = {
        r"re:.*o_proj": "fp8_per_tensor",
        "model.layers.0.self_attn.o_proj": "fp8_per_block",
    }

    with pytest.raises(ValueError, match="multiple quantization_config.targets"):
        _find_matching_targets("model.layers.0.self_attn.o_proj", targets)


def test_targets_reject_partially_matched_fused_layer():
    targets = {r"re:.*q_proj": "fp8_per_tensor"}
    fused_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    with pytest.raises(ValueError, match="unmatched shards"):
        _find_matching_targets(
            "model.layers.0.self_attn.qkv_proj", targets, fused_mapping
        )


def test_targets_reject_fused_shards_with_different_schemes():
    targets = {
        r"re:.*\.q_proj$": "fp8_per_tensor",
        r"re:.*\.k_proj$": "fp8_per_block",
        r"re:.*\.v_proj$": "fp8_per_tensor",
    }
    fused_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    with pytest.raises(ValueError, match="different quantization_config.targets"):
        _find_matching_targets(
            "model.layers.0.self_attn.qkv_proj", targets, fused_mapping
        )


def test_targets_reject_moe_only_shorthand_for_linear_layer():
    config = OnlineQuantizationConfig(
        QuantizationConfigArgs(
            targets={"model.layers.0.self_attn.o_proj": "nvfp4_per_token"}
        )
    )

    with pytest.raises(ValueError, match="does not define a QuantSpec"):
        config.resolve_quant_method_cls(LinearBase, "model.layers.0.self_attn.o_proj")


# ---- hadamard_size ---------------------------------------------------------


def test_mxfp4_hadamard32_target_resolves_to_rotated_mxfp4_linear():
    from vllm.model_executor.layers.quantization.online.mxfp4 import (
        Mxfp4OnlineLinearMethod,
    )

    config = OnlineQuantizationConfig(
        QuantizationConfigArgs(
            targets={"re:.*linear_attn\\.out_proj": "mxfp4_hadamard32"}
        )
    )
    resolved = config.resolve_quant_method_cls(
        LinearBase, "model.layers.0.linear_attn.out_proj"
    )
    assert resolved is not None
    _, quant_key_str, _, spec, method_cls = resolved
    assert quant_key_str == "mxfp4_hadamard32"
    assert spec.hadamard_size == 32
    assert method_cls is Mxfp4OnlineLinearMethod


def test_rocm_platform_supports_mxfp4_hadamard32():
    """`--quantization mxfp4_hadamard32` reaches the ROCm allowlist unchanged."""
    try:
        from vllm.platforms.rocm import RocmPlatform
    except ImportError:
        pytest.skip("vllm.platforms.rocm not importable outside a ROCm stack")

    RocmPlatform.verify_quantization("mxfp4_hadamard32")


@pytest.mark.parametrize("size", [0, 24, 33])
def test_hadamard_size_must_be_power_of_two(size):
    with pytest.raises(ValueError, match="power of two"):
        quant_config_args(linear={"weight": kMxfp4Static, "hadamard_size": size})


@pytest.mark.parametrize(
    "linear,moe",
    [
        ({"weight": "fp8_per_block_static", "hadamard_size": 32}, None),
        (None, {"weight": kMxfp4Static, "hadamard_size": 32}),
    ],
)
def test_hadamard_size_not_implemented_for_unvalidated_methods(linear, moe):
    config = OnlineQuantizationConfig(quant_config_args(linear=linear, moe=moe))
    layer_cls = LinearBase if linear is not None else RoutedExperts
    with pytest.raises(NotImplementedError, match="hadamard_size"):
        config.resolve_quant_method_cls(layer_cls, "model.layers.0.mlp.down_proj")
