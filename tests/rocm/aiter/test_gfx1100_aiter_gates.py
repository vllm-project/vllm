# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hermetic tests for the narrowly scoped gfx1100 AITER W8A8 gate."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import vllm._aiter_ops as aiter_ops_module
from vllm._aiter_ops import (
    _rocm_aiter_w8a8_gemm_impl,
    is_aiter_found_and_supported,
    is_aiter_found_and_supported_on_rdna4,
    rocm_aiter_ops,
)
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterInt8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.fused_moe.oracle import fp8 as fp8_moe
from vllm.model_executor.layers.fused_moe.oracle import (
    unquantized as unquantized_moe,
)
from vllm.platforms.rocm import _get_backend_priorities


@pytest.fixture(autouse=True)
def _rocm_platform():
    with patch("vllm._aiter_ops.current_platform") as platform:
        platform.is_rocm.return_value = True
        yield platform


@pytest.fixture(autouse=True)
def _arch_predicates():
    with (
        patch("vllm.platforms.rocm.on_gfx1100", return_value=False),
        patch("vllm.platforms.rocm.on_rdna4", return_value=False),
        patch("vllm.platforms.rocm.on_gfx950", return_value=False),
        patch("vllm.platforms.rocm.get_cdna_version", return_value=0),
    ):
        yield


@pytest.fixture(autouse=True)
def _aiter_found_and_enabled():
    with (
        patch("vllm._aiter_ops.IS_AITER_FOUND", True),
        patch.object(rocm_aiter_ops, "_AITER_ENABLED", True),
        patch.object(rocm_aiter_ops, "_LINEAR_ENABLED", True),
        patch.object(rocm_aiter_ops, "_FMOE_ENABLED", True),
        patch.object(rocm_aiter_ops, "_MLA_ENABLED", True),
        patch.object(rocm_aiter_ops, "_MHA_ENABLED", True),
        patch.object(rocm_aiter_ops, "_CUSTOM_ALL_REDUCE_ENABLED", True),
    ):
        yield


def _fake_aiter(
    tmp_path: Path,
    *,
    public_export: bool = True,
    packaged_config: bool = True,
    triton_only: bool = False,
) -> ModuleType:
    package_dir = tmp_path / "aiter"
    package_dir.mkdir()
    init_file = package_dir / "__init__.py"
    init_file.touch()

    if packaged_config:
        config = package_dir / aiter_ops_module._AITER_GFX1100_W8A8_CONFIG
        config.parent.mkdir(parents=True)
        config.write_text("{}")

    module = ModuleType("aiter")
    module.__file__ = str(init_file)
    module.AITER_TRITON_ONLY = triton_only
    if public_export:
        module.gemm_a8w8 = Mock(name="gemm_a8w8")
    return module


class TestGfx1100Capability:
    def test_complete_package_enables_only_gfx1100_w8a8(self, tmp_path):
        fake_aiter = _fake_aiter(tmp_path)
        with (
            patch.dict("sys.modules", {"aiter": fake_aiter}),
            patch("vllm.platforms.rocm.on_gfx1100", return_value=True),
        ):
            assert rocm_aiter_ops.is_gfx1100_aiter_enabled() is True
            assert rocm_aiter_ops.is_gfx1100_linear_enabled() is True

            supported, reason = AiterInt8ScaledMMLinearKernel.is_supported()
            assert supported, reason

    @pytest.mark.parametrize(
        ("public_export", "packaged_config", "triton_only"),
        [
            (False, True, False),
            (True, False, False),
            (True, True, True),
        ],
        ids=["missing-public-export", "missing-config", "triton-only"],
    )
    def test_incompatible_packages_fail_closed(
        self, tmp_path, public_export, packaged_config, triton_only
    ):
        fake_aiter = _fake_aiter(
            tmp_path,
            public_export=public_export,
            packaged_config=packaged_config,
            triton_only=triton_only,
        )
        with (
            patch.dict("sys.modules", {"aiter": fake_aiter}),
            patch("vllm.platforms.rocm.on_gfx1100", return_value=True),
        ):
            assert rocm_aiter_ops.is_gfx1100_aiter_enabled() is False
            assert rocm_aiter_ops.is_gfx1100_linear_enabled() is False

    def test_main_aiter_toggle_is_required(self, tmp_path):
        fake_aiter = _fake_aiter(tmp_path)
        with (
            patch.dict("sys.modules", {"aiter": fake_aiter}),
            patch("vllm.platforms.rocm.on_gfx1100", return_value=True),
            patch.object(rocm_aiter_ops, "_AITER_ENABLED", False),
        ):
            assert rocm_aiter_ops.is_gfx1100_aiter_enabled() is False


class TestFeatureAdmission:
    def test_gfx1100_does_not_enter_broad_or_rdna4_gates(self):
        with (
            patch("vllm.platforms.rocm.on_gfx1100", return_value=True),
            patch.object(rocm_aiter_ops, "is_gfx1100_aiter_enabled", return_value=True),
        ):
            assert is_aiter_found_and_supported() is False
            assert not rocm_aiter_ops.is_enabled()
            assert not rocm_aiter_ops.is_rdna_aiter_enabled()
            assert not rocm_aiter_ops.is_rdna_linear_enabled()
            assert not rocm_aiter_ops.is_linear_enabled()
            assert not rocm_aiter_ops.is_linear_fp8_enabled()
            assert not rocm_aiter_ops.is_fused_moe_enabled()
            assert not rocm_aiter_ops.is_mla_enabled()
            assert not rocm_aiter_ops.is_mha_enabled()
            assert not rocm_aiter_ops.is_custom_all_reduce_enabled()
            assert not rocm_aiter_ops.is_triton_unified_attn_enabled()
            assert not rocm_aiter_ops.is_fp8bmm_enabled()
            assert not rocm_aiter_ops.is_triton_rotary_embed_enabled()

    def test_gfx1100_does_not_add_aiter_attention_backend(self):
        with (
            patch("vllm.platforms.rocm.on_gfx1100", return_value=True),
            patch(
                "vllm._aiter_ops.is_aiter_found_and_supported",
                return_value=False,
            ),
            patch.object(rocm_aiter_ops, "is_mha_enabled", return_value=False),
            patch.object(rocm_aiter_ops, "is_rdna_aiter_enabled", return_value=False),
        ):
            backends = _get_backend_priorities(use_mla=False, use_sparse=False)

        assert backends[0].name == "ROCM_ATTN"
        assert all("AITER" not in backend.name for backend in backends)

    @pytest.mark.parametrize("importable", [False, True])
    def test_gfx1100_gdn_requires_importable_kernels(self, importable):
        with (
            patch.object(rocm_aiter_ops, "is_gfx1100_aiter_enabled", return_value=True),
            patch.object(
                rocm_aiter_ops,
                "_gdn_triton_kernels_importable",
                return_value=importable,
            ),
        ):
            assert (
                rocm_aiter_ops.is_gfx1100_gdn_triton_kernels_available() is importable
            )


class TestNonGfx1100Preservation:
    def test_cdna3_keeps_original_ck_gates(self):
        with patch("vllm.platforms.rocm.get_cdna_version", return_value=3):
            assert is_aiter_found_and_supported() is True
            assert rocm_aiter_ops.is_enabled() is True
            assert rocm_aiter_ops.is_linear_enabled() is True
            assert rocm_aiter_ops.is_linear_fp8_enabled() is True
            assert rocm_aiter_ops.is_fused_moe_enabled() is True
            assert rocm_aiter_ops.is_mla_enabled() is True
            assert rocm_aiter_ops.is_mha_enabled() is True
            assert not rocm_aiter_ops.is_rdna_aiter_enabled()
            assert rocm_aiter_ops.is_gfx1100_aiter_enabled() is False

    def test_rdna4_keeps_original_rdna_gates_and_priority(self):
        with patch("vllm.platforms.rocm.on_rdna4", return_value=True):
            assert is_aiter_found_and_supported() is False
            assert is_aiter_found_and_supported_on_rdna4() is True
            assert rocm_aiter_ops.is_rdna_aiter_enabled() is True
            assert rocm_aiter_ops.is_rdna_linear_enabled() is True
            assert not rocm_aiter_ops.is_linear_enabled()
            assert not rocm_aiter_ops.is_linear_fp8_enabled()
            assert rocm_aiter_ops.is_gfx1100_aiter_enabled() is False

            with (
                patch(
                    "vllm._aiter_ops.is_aiter_found_and_supported",
                    return_value=False,
                ),
                patch.object(rocm_aiter_ops, "is_mha_enabled", return_value=False),
            ):
                backends = _get_backend_priorities(use_mla=False, use_sparse=False)

        assert backends[0].name == "ROCM_AITER_UNIFIED_ATTN"
        assert backends[1].name == "ROCM_ATTN"


class TestW8A8DispatchAndLayouts:
    def test_w8a8_uses_public_dispatcher(self):
        fake_aiter = ModuleType("aiter")
        gemm_a8w8 = Mock(return_value="dispatched")
        fake_aiter.gemm_a8w8 = gemm_a8w8

        with (
            patch.dict("sys.modules", {"aiter": fake_aiter}),
            patch.object(rocm_aiter_ops, "is_gfx1100", return_value=True),
        ):
            result = _rocm_aiter_w8a8_gemm_impl(
                "A", "B", "As", "Bs", "bias", torch.float16
            )

        assert result == "dispatched"
        gemm_a8w8.assert_called_once_with("A", "B", "As", "Bs", "bias", torch.float16)

    def test_cdna_w8a8_keeps_ck_dispatch(self):
        fake_aiter = ModuleType("aiter")
        gemm_a8w8_ck = Mock(return_value="ck")
        fake_aiter.gemm_a8w8_CK = gemm_a8w8_ck

        with (
            patch.dict("sys.modules", {"aiter": fake_aiter}),
            patch.object(rocm_aiter_ops, "is_gfx1100", return_value=False),
        ):
            result = _rocm_aiter_w8a8_gemm_impl(
                "A", "B", "As", "Bs", "bias", torch.float16
            )

        assert result == "ck"
        gemm_a8w8_ck.assert_called_once_with(
            "A", "B", "As", "Bs", "bias", torch.float16
        )

    @staticmethod
    def _config(static: bool, channelwise: bool):
        from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
            Int8ScaledMMLinearLayerConfig,
        )

        return Int8ScaledMMLinearLayerConfig(
            is_static_input_scheme=static,
            is_channelwise=channelwise,
            input_symmetric=True,
        )

    @pytest.mark.parametrize(
        ("static", "channelwise", "expected"),
        [
            (False, False, False),
            (False, True, True),
            (True, False, False),
            (True, True, False),
        ],
    )
    def test_gfx1100_accepts_exactly_dynamic_channelwise(
        self, static, channelwise, expected
    ):
        with patch("vllm.platforms.rocm.on_gfx1100", return_value=True):
            supported, reason = AiterInt8ScaledMMLinearKernel.can_implement(
                self._config(static, channelwise)
            )

        assert supported is expected
        assert (reason is None) is expected

    @pytest.mark.parametrize(
        ("static", "channelwise"),
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_cdna_layout_behavior_is_unchanged(self, static, channelwise):
        supported, reason = AiterInt8ScaledMMLinearKernel.can_implement(
            self._config(static, channelwise)
        )
        assert supported, reason


class TestRegistration:
    def test_gfx1100_registers_exactly_w8a8(self):
        saved = aiter_ops_module._OPS_REGISTERED
        try:
            aiter_ops_module._OPS_REGISTERED = False
            with (
                patch("vllm._aiter_ops.direct_register_custom_op") as register,
                patch(
                    "vllm._aiter_ops.is_aiter_found_and_supported",
                    return_value=False,
                ),
                patch(
                    "vllm._aiter_ops.is_aiter_found_and_supported_on_rdna4",
                    return_value=False,
                ),
                patch.object(
                    rocm_aiter_ops,
                    "is_gfx1100_aiter_enabled",
                    return_value=True,
                ),
            ):
                rocm_aiter_ops.register_ops_once()

            assert [call.kwargs["op_name"] for call in register.call_args_list] == [
                "rocm_aiter_w8a8_gemm"
            ]
            assert aiter_ops_module._OPS_REGISTERED is True
        finally:
            aiter_ops_module._OPS_REGISTERED = saved

    @pytest.mark.parametrize(
        ("cdna", "rdna4"), [(True, False), (False, True)], ids=["cdna", "rdna4"]
    )
    def test_existing_targets_still_register_full_op_set(self, cdna, rdna4):
        saved = aiter_ops_module._OPS_REGISTERED
        try:
            aiter_ops_module._OPS_REGISTERED = False
            with (
                patch("vllm._aiter_ops.direct_register_custom_op") as register,
                patch(
                    "vllm._aiter_ops.is_aiter_found_and_supported",
                    return_value=cdna,
                ),
                patch(
                    "vllm._aiter_ops.is_aiter_found_and_supported_on_rdna4",
                    return_value=rdna4,
                ),
                patch.object(
                    rocm_aiter_ops,
                    "is_gfx1100_aiter_enabled",
                    return_value=False,
                ),
            ):
                rocm_aiter_ops.register_ops_once()

            names = [call.kwargs["op_name"] for call in register.call_args_list]
            assert "rocm_aiter_w8a8_gemm" in names
            assert "rocm_aiter_fused_moe" in names
            assert len(names) > 2
        finally:
            aiter_ops_module._OPS_REGISTERED = saved


class _SupportedExperts:
    @staticmethod
    def is_supported_config(*args, **kwargs):
        return True, None


def _auto_moe_config():
    return SimpleNamespace(
        moe_backend="auto",
        is_lora_enabled=False,
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
    )


def _assert_fp8_moe_skips_aiter(*, capable: bool, gfx1100: bool):
    fp8 = fp8_moe

    with (
        patch.object(
            fp8,
            "_get_priority_backends",
            return_value=[fp8.Fp8MoeBackend.AITER, fp8.Fp8MoeBackend.TRITON],
        ),
        patch.object(fp8, "backend_to_kernel_cls", return_value=[_SupportedExperts]),
        patch.object(
            fp8.envs,
            "is_set",
            side_effect=lambda name: (
                name in {"VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE"}
            ),
        ),
        patch.object(fp8.envs, "VLLM_ROCM_USE_AITER", True),
        patch.object(fp8.envs, "VLLM_ROCM_USE_AITER_MOE", True),
        patch.object(fp8.envs, "VLLM_TEST_FORCE_FP8_MARLIN", False),
        patch.object(rocm_aiter_ops, "is_rdna_aiter_enabled", return_value=False),
        patch.object(
            rocm_aiter_ops,
            "is_gfx1100_aiter_enabled",
            return_value=capable,
        ),
        patch.object(rocm_aiter_ops, "is_gfx1100", return_value=gfx1100),
    ):
        backend, _ = fp8.select_fp8_moe_backend(
            _auto_moe_config(), None, None, allow_vllm_cutlass=True
        )

    assert backend is fp8.Fp8MoeBackend.TRITON


def _assert_unquantized_moe_skips_aiter(*, capable: bool, gfx1100: bool):
    unquantized = unquantized_moe

    with (
        patch.object(
            unquantized,
            "_get_priority_backends",
            return_value=[
                unquantized.UnquantizedMoeBackend.AITER,
                unquantized.UnquantizedMoeBackend.TRITON,
            ],
        ),
        patch.object(
            unquantized, "backend_to_kernel_cls", return_value=[_SupportedExperts]
        ),
        patch.object(
            unquantized.envs,
            "is_set",
            side_effect=lambda name: (
                name in {"VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE"}
            ),
        ),
        patch.object(unquantized.envs, "VLLM_ROCM_USE_AITER", True),
        patch.object(unquantized.envs, "VLLM_ROCM_USE_AITER_MOE", True),
        patch.object(rocm_aiter_ops, "is_rdna_aiter_enabled", return_value=False),
        patch.object(
            rocm_aiter_ops,
            "is_gfx1100_aiter_enabled",
            return_value=capable,
        ),
        patch.object(rocm_aiter_ops, "is_gfx1100", return_value=gfx1100),
    ):
        backend, _ = unquantized.select_unquantized_moe_backend(_auto_moe_config())

    assert backend is unquantized.UnquantizedMoeBackend.TRITON


@pytest.mark.parametrize(
    ("capable", "gfx1100"),
    [(True, False), (False, True)],
    ids=["capability-predicate", "incompatible-install-fail-closed"],
)
def test_gfx1100_skips_ck_aiter_moe_oracles(capable, gfx1100):
    _assert_fp8_moe_skips_aiter(capable=capable, gfx1100=gfx1100)
    _assert_unquantized_moe_skips_aiter(capable=capable, gfx1100=gfx1100)
