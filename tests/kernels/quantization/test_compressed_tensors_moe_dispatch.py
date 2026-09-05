# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch-layer tests for CompressedTensorsMoEMethod.get_moe_method.

get_moe_method is pure routing logic: given a quantization scheme (plus
platform/config state for the ROCm branches), it picks one of ~9 backend
MoE method classes.

Every backend class is mocked, so these tests assert routing only -- which
class gets selected, and with what arguments -- not what that class does
internally. They run on CPU with no GPU/model dependency.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy, QuantizationType

from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa: E501
    WNA16_SUPPORTED_BITS,
)

MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe"
)
PKG = (
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe"
)

ALL_PREDICATES = [
    "_is_mxfp4",
    "_is_mxfp8",
    "_is_wNa16_group_channel",
    "_is_nvfp4_format",
    "_is_fp8_w8a8_sm90",
    "_is_fp8_w8a8_sm100",
    "_is_fp8_w8a8",
    "_is_dynamic_token_w8a8",
    "_is_fp8_w4a8_sm90",
    "_is_dynamic_token_w4a8_int",
]

LAYER_NAME = "layer.0"


def _make_quant_config(scheme_dict, **predicate_overrides):
    """CompressedTensorsConfig stand-in: every `_is_*` predicate is False
    unless overridden, and every unfused projection name resolves to the
    same scheme_dict (the normal, non-mismatch path)."""
    quant_config = MagicMock()
    for name in ALL_PREDICATES:
        getattr(quant_config, name).return_value = predicate_overrides.get(name, False)
    quant_config.get_scheme_dict.return_value = scheme_dict
    return quant_config


def _weight_quant(**kwargs):
    weight_quant = MagicMock()
    for key, value in kwargs.items():
        setattr(weight_quant, key, value)
    return weight_quant


def _call(quant_config):
    layer = MagicMock()
    result = CompressedTensorsMoEMethod.get_moe_method(quant_config, layer, LAYER_NAME)
    return result, layer


class TestSchemeResolution:
    def test_ignored_layer_returns_unquantized(self, default_vllm_config):
        quant_config = _make_quant_config(scheme_dict=None)
        result, _ = _call(quant_config)
        assert isinstance(result, UnquantizedFusedMoEMethod)

    def test_mismatched_scheme_across_projections_raises(self):
        quant_config = _make_quant_config(scheme_dict=None)
        quant_config.get_scheme_dict.side_effect = [{"a": 1}, {"a": 2}, {"a": 1}]
        with pytest.raises(ValueError, match="same"):
            _call(quant_config)

    def test_unsupported_scheme_raises(self):
        quant_config = _make_quant_config(
            scheme_dict={
                "weights": _weight_quant(),
                "input_activations": None,
                "format": None,
            }
        )
        with pytest.raises(RuntimeError, match="Unsupported FusedMoe scheme"):
            _call(quant_config)


@patch(f"{MODULE}.current_platform")
class TestSimpleBackendRouting:
    """Branches with a `Cls(layer.moe_config)` constructor -- no
    weight_quant/input_quant threaded through."""

    @pytest.mark.parametrize(
        "predicate,submodule,cls_name",
        [
            (
                "_is_mxfp4",
                "compressed_tensors_moe_w4a4_mxfp4",
                "CompressedTensorsW4A4Mxfp4MoEMethod",
            ),
            (
                "_is_mxfp8",
                "compressed_tensors_moe_w8a8_mxfp8",
                "CompressedTensorsW8A8Mxfp8MoEMethod",
            ),
        ],
    )
    def test_routes_to_expected_class(
        self, mock_platform, predicate, submodule, cls_name
    ):
        quant_config = _make_quant_config(
            scheme_dict={
                "weights": _weight_quant(),
                "input_activations": None,
                "format": None,
            },
            **{predicate: True},
        )
        with patch(f"{PKG}.{submodule}.{cls_name}") as mock_cls:
            result, layer = _call(quant_config)
        mock_cls.assert_called_once_with(layer.moe_config)
        assert result is mock_cls.return_value


@patch(f"{MODULE}.current_platform")
class TestWeightInputBackendRouting:
    """Branches with a `Cls(weight_quant, input_quant, layer.moe_config)`
    constructor -- the majority of the non-ROCm-specific paths."""

    @pytest.mark.parametrize(
        "predicate,submodule,cls_name",
        [
            (
                "_is_fp8_w8a8",
                "compressed_tensors_moe_w8a8_fp8",
                "CompressedTensorsW8A8Fp8MoEMethod",
            ),
            (
                "_is_fp8_w8a8_sm90",
                "compressed_tensors_moe_w8a8_fp8",
                "CompressedTensorsW8A8Fp8MoEMethod",
            ),
            (
                "_is_fp8_w8a8_sm100",
                "compressed_tensors_moe_w8a8_fp8",
                "CompressedTensorsW8A8Fp8MoEMethod",
            ),
            (
                "_is_dynamic_token_w8a8",
                "compressed_tensors_moe_w8a8_int8",
                "CompressedTensorsW8A8Int8MoEMethod",
            ),
            (
                "_is_fp8_w4a8_sm90",
                "compressed_tensors_moe_w4a8_fp8",
                "CompressedTensorsW4A8Fp8MoEMethod",
            ),
            (
                "_is_dynamic_token_w4a8_int",
                "compressed_tensors_moe_w4a8_int8",
                "CompressedTensorsW4A8Int8MoEMethod",
            ),
        ],
    )
    def test_routes_to_expected_class(
        self, mock_platform, predicate, submodule, cls_name
    ):
        weight_quant = _weight_quant()
        quant_config = _make_quant_config(
            scheme_dict={
                "weights": weight_quant,
                "input_activations": None,
                "format": None,
            },
            **{predicate: True},
        )
        with patch(f"{PKG}.{submodule}.{cls_name}") as mock_cls:
            result, layer = _call(quant_config)
        mock_cls.assert_called_once_with(weight_quant, None, layer.moe_config)
        assert result is mock_cls.return_value

    def test_wna16_default_non_rocm(self, mock_platform):
        mock_platform.is_rocm.return_value = False
        weight_quant = _weight_quant(num_bits=WNA16_SUPPORTED_BITS[0])
        quant_config = _make_quant_config(
            scheme_dict={
                "weights": weight_quant,
                "input_activations": None,
                "format": CompressionFormat.pack_quantized.value,
            },
            _is_wNa16_group_channel=True,
        )
        with patch(
            f"{PKG}.compressed_tensors_moe_wna16.CompressedTensorsWNA16MoEMethod"
        ) as mock_cls:
            result, layer = _call(quant_config)
        mock_cls.assert_called_once_with(weight_quant, None, layer.moe_config)
        assert result is mock_cls.return_value

    def test_wna16_invalid_bits_or_format_raises(self, mock_platform):
        mock_platform.is_rocm.return_value = False
        weight_quant = _weight_quant(num_bits=99)  # not in WNA16_SUPPORTED_BITS
        quant_config = _make_quant_config(
            scheme_dict={
                "weights": weight_quant,
                "input_activations": None,
                "format": CompressionFormat.pack_quantized.value,
            },
            _is_wNa16_group_channel=True,
        )
        with pytest.raises(ValueError):
            _call(quant_config)


class TestNvfp4Routing:
    def test_valid_nvfp4_routes_with_use_a16_true_when_input_quant_none(self):
        weight_quant = _weight_quant()
        quant_config = _make_quant_config(
            scheme_dict={
                "weights": weight_quant,
                "input_activations": None,
                "format": None,
            },
        )
        quant_config._is_nvfp4_format.side_effect = lambda q: q is weight_quant

        with patch(
            f"{PKG}.compressed_tensors_moe_w4a4_nvfp4.CompressedTensorsW4A4Nvfp4MoEMethod"
        ) as mock_cls:
            result, layer = _call(quant_config)
        mock_cls.assert_called_once_with(layer.moe_config, LAYER_NAME, use_a16=True)
        assert result is mock_cls.return_value

    def test_valid_nvfp4_routes_with_use_a16_false_when_input_quant_also_nvfp4(self):
        weight_quant = _weight_quant()
        input_quant = _weight_quant()
        quant_config = _make_quant_config(
            scheme_dict={
                "weights": weight_quant,
                "input_activations": input_quant,
                "format": None,
            },
            _is_nvfp4_format=True,
        )

        with patch(
            f"{PKG}.compressed_tensors_moe_w4a4_nvfp4.CompressedTensorsW4A4Nvfp4MoEMethod"
        ) as mock_cls:
            result, layer = _call(quant_config)
        mock_cls.assert_called_once_with(layer.moe_config, LAYER_NAME, use_a16=False)
        assert result is mock_cls.return_value

    def test_nvfp4_weights_with_non_nvfp4_input_quant_raises(self):
        weight_quant = _weight_quant()
        input_quant = _weight_quant()
        quant_config = _make_quant_config(
            scheme_dict={
                "weights": weight_quant,
                "input_activations": input_quant,
                "format": None,
            },
        )
        quant_config._is_nvfp4_format.side_effect = lambda q: q is weight_quant

        with pytest.raises(ValueError, match="NVFP4"):
            _call(quant_config)


class TestRocmWNA16Routing:
    """The three ROCm-specific sub-branches inside the wNa16 path: native
    RDNA HIP kernel, gfx950 FlyDSL, and the emulation-backend log-only path
    that still falls through to the default WNA16 method."""

    def test_native_rdna_kernel_used_when_supported(self):
        weight_quant = _weight_quant(num_bits=WNA16_SUPPORTED_BITS[0])
        quant_config = _make_quant_config(
            scheme_dict={
                "weights": weight_quant,
                "input_activations": None,
                "format": CompressionFormat.pack_quantized.value,
            },
            _is_wNa16_group_channel=True,
        )
        with (
            patch(f"{MODULE}.current_platform") as mock_platform,
            patch(
                f"{PKG}.rocm_moe_rdna.is_supported", return_value=True
            ) as mock_is_supported,
            patch(f"{PKG}.rocm_moe_rdna.make_method") as mock_make_method,
        ):
            mock_platform.is_rocm.return_value = True
            result, layer = _call(quant_config)

        mock_is_supported.assert_called_once_with(weight_quant)
        mock_make_method.assert_called_once_with(weight_quant, None, layer.moe_config)
        assert result is mock_make_method.return_value

    def test_gfx950_flydsl_routes_correctly_when_all_conditions_met(self):
        """Confirms the gfx950/FlyDSL branch is reachable under its own,
        distinct condition set -- separate from the rocm_moe_rdna guard
        above it in the dispatch chain.

        The real module this branch imports (`compressed_tensors_moe_w4a16_flydsl`)
        pulls in `aiter`, a gfx950-only dependency not present on most test
        runners. A fake module is injected into `sys.modules` so the routing
        logic under test never touches `aiter`.
        """
        weight_quant = _weight_quant(
            num_bits=4,
            group_size=32,
            strategy=QuantizationStrategy.GROUP,
            type=QuantizationType.INT,
        )
        quant_config = _make_quant_config(
            scheme_dict={
                "weights": weight_quant,
                "input_activations": None,
                "format": CompressionFormat.pack_quantized.value,
            },
            _is_wNa16_group_channel=True,
        )
        vllm_config = MagicMock()
        vllm_config.lora_config = None
        vllm_config.kernel_config.moe_backend = "flydsl"

        flydsl_cls_mock = MagicMock()
        fake_module = MagicMock(CompressedTensorsW4A16FlydslMoEMethod=flydsl_cls_mock)

        with (
            patch(f"{MODULE}.current_platform") as mock_platform,
            patch(f"{PKG}.rocm_moe_rdna.is_supported", return_value=False),
            patch(f"{MODULE}.get_current_vllm_config", return_value=vllm_config),
            patch("vllm.platforms.rocm.on_gfx950", return_value=True),
            patch.dict(
                sys.modules,
                {f"{PKG}.compressed_tensors_moe_w4a16_flydsl": fake_module},
            ),
        ):
            mock_platform.is_rocm.return_value = True
            result, layer = _call(quant_config)

        flydsl_cls_mock.assert_called_once_with(weight_quant, None, layer.moe_config)
        assert result is flydsl_cls_mock.return_value

    def test_emulation_backend_falls_through_to_default_wna16(self):
        """`moe_backend == "emulation"` only logs -- it does not change which
        class gets returned. Guards against someone "completing" that branch
        with an early return that would change behavior."""
        weight_quant = _weight_quant(
            num_bits=4,
            group_size=32,
            strategy=QuantizationStrategy.GROUP,
            type=QuantizationType.INT,
        )
        quant_config = _make_quant_config(
            scheme_dict={
                "weights": weight_quant,
                "input_activations": None,
                "format": CompressionFormat.pack_quantized.value,
            },
            _is_wNa16_group_channel=True,
        )
        vllm_config = MagicMock()
        vllm_config.lora_config = None
        vllm_config.kernel_config.moe_backend = "emulation"

        with (
            patch(f"{MODULE}.current_platform") as mock_platform,
            patch(f"{PKG}.rocm_moe_rdna.is_supported", return_value=False),
            patch(f"{MODULE}.get_current_vllm_config", return_value=vllm_config),
            patch("vllm.platforms.rocm.on_gfx950", return_value=False),
            patch(
                f"{PKG}.compressed_tensors_moe_wna16.CompressedTensorsWNA16MoEMethod"
            ) as mock_cls,
        ):
            mock_platform.is_rocm.return_value = True
            result, layer = _call(quant_config)

        mock_cls.assert_called_once_with(weight_quant, None, layer.moe_config)
        assert result is mock_cls.return_value


class TestRocmFlydslConditionsAllRequired:
    """The gfx950/FlyDSL route is gated by a 7-term AND. A single positive
    test (all conditions True) can't tell a correct AND from one that's lost
    a term or had a term swapped for an OR. Each row here starts from an
    otherwise-all-true baseline and breaks exactly one term, asserting the
    route falls through to the default WNA16 method instead."""

    def _baseline_kwargs(self):
        return dict(
            num_bits=4,
            group_size=32,
            strategy=QuantizationStrategy.GROUP,
            type=QuantizationType.INT,
        )

    @pytest.mark.parametrize(
        "weight_quant_overrides,lora_config,moe_backend,on_gfx950_return",
        [
            pytest.param(
                {"strategy": QuantizationStrategy.CHANNEL},
                None,
                "flydsl",
                True,
                id="wrong_strategy",
            ),
            pytest.param(
                {"type": QuantizationType.FLOAT},
                None,
                "flydsl",
                True,
                id="wrong_type",
            ),
            pytest.param(
                {"group_size": 16},
                None,
                "flydsl",
                True,
                id="wrong_group_size",
            ),
            pytest.param(
                {"num_bits": 8},
                None,
                "flydsl",
                True,
                id="wrong_num_bits",
            ),
            pytest.param(
                {},
                MagicMock(),
                "flydsl",
                True,
                id="lora_enabled",
            ),
            pytest.param(
                {},
                None,
                "flydsl",
                False,
                id="not_on_gfx950",
            ),
            pytest.param(
                {},
                None,
                "triton",
                True,
                id="wrong_moe_backend",
            ),
        ],
    )
    def test_single_broken_condition_prevents_flydsl_routing(
        self, weight_quant_overrides, lora_config, moe_backend, on_gfx950_return
    ):
        kwargs = self._baseline_kwargs()
        kwargs.update(weight_quant_overrides)
        weight_quant = _weight_quant(**kwargs)
        quant_config = _make_quant_config(
            scheme_dict={
                "weights": weight_quant,
                "input_activations": None,
                "format": CompressionFormat.pack_quantized.value,
            },
            _is_wNa16_group_channel=True,
        )
        vllm_config = MagicMock()
        vllm_config.lora_config = lora_config
        vllm_config.kernel_config.moe_backend = moe_backend

        with (
            patch(f"{MODULE}.current_platform") as mock_platform,
            patch(f"{PKG}.rocm_moe_rdna.is_supported", return_value=False),
            patch(f"{MODULE}.get_current_vllm_config", return_value=vllm_config),
            patch("vllm.platforms.rocm.on_gfx950", return_value=on_gfx950_return),
            patch(
                f"{PKG}.compressed_tensors_moe_wna16.CompressedTensorsWNA16MoEMethod"
            ) as mock_wna16_cls,
        ):
            mock_platform.is_rocm.return_value = True
            result, layer = _call(quant_config)

        # The flydsl submodule imports `aiter` (gfx950-only); it must never
        # get imported when the AND-chain isn't fully satisfied.
        assert f"{PKG}.compressed_tensors_moe_w4a16_flydsl" not in sys.modules
        mock_wna16_cls.assert_called_once_with(weight_quant, None, layer.moe_config)
        assert result is mock_wna16_cls.return_value
