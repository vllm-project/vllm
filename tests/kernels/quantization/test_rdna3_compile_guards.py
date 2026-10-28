#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-guard tests for the ROCm RDNA3 W4A16 kernels (dense + MoE).

Verifies that the gfx1100 compilation and dispatch guards are hermetic:
  - On gfx1100: all ops exist, dispatch selects RDNA3 kernels.
  - On CDNA (gfx942/gfx950) or other non-gfx1100: ops must NOT exist,
    dispatch must fall through to Triton/Marlin, and no RDNA3 code
    path is reachable.

The negative (non-gfx1100) tests verify at three layers:
  1. Compile-level: on non-gfx1100 hardware, the RDNA3 ops are absent
     from the compiled _rocm_C extension — real binary verification.
  2. Static source analysis: parses CMakeLists.txt and torch_bindings.cpp
     to verify that all RDNA3 .cu files and op registrations are inside
     gfx1100-only guards.
  3. Runtime mock: patches on_gfx1100() to False and verifies that the
     Python dispatch chain rejects the RDNA3 path.

The W4A8 MoE decode kernel (GFX11_OPS) supports every gfx11 GPU, so it has its
own guard (VLLM_ROCM_HAS_GFX11 / VLLM_ROCM_GFX11) and is checked separately.

Run `pytest tests/kernels/quantization/test_rdna3_compile_guards.py`.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import regex as re
import torch

import vllm
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA3 compile-guard tests are ROCm-only", allow_module_level=True)

from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    kInt4Static,
    kInt4Static32,
    kInt4Static32Asym,
    kInt4StaticAsym,
    kInt8Static,
)
from vllm.platforms.rocm import on_gfx1100  # noqa: E402

gfx1100_only = pytest.mark.skipif(
    not on_gfx1100(),
    reason="Requires gfx1100 hardware",
)

not_gfx1100 = pytest.mark.skipif(
    on_gfx1100(),
    reason="This test verifies non-gfx1100 builds — skip on gfx1100",
)

RDNA3_OPS = [
    "gptq_gemm_rdna3",
    "gptq_gemm_rdna3_wmma",
    "moe_gptq_gemm_rdna3",
    "mxfp4_w4a8_gemv_rdna3",
]
RDNA3_CU_FILES = [
    "q_gemm_rdna3.cu",
    "q_gemm_rdna3_wmma.cu",
    "moe_q_gemm_rdna3.cu",
    "mxfp4_w4a8_gemv_rdna3.cu",
]
# Built whenever any gfx11 target is requested, not only gfx1100.
GFX11_OPS = ["moe_w4a8_decode_rdna3"]
GFX11_CU_FILES = ["moe_w4a8_decode_rdna3.cu"]
# torch_bindings.cpp guards; an RDNA3 op may sit inside either.
BINDINGS_GUARDS = ("#ifdef VLLM_ROCM_GFX1100", "#ifdef VLLM_ROCM_GFX11")


def _on_gfx11() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx11")
    )


def _find_repo_root() -> Path | None:
    """Walk up from this file to find the repo root (has CMakeLists.txt)."""
    for parent in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (parent / "CMakeLists.txt").exists() and (parent / "csrc").is_dir():
            return parent
    return None


REPO_ROOT = _find_repo_root()

# Directory of the *installed* vllm python package. The .py guard checks read
# from here so they verify the code that is actually imported at runtime — this
# works even on CI images that ship the wheel instead of the python source tree
# (where only csrc/ + CMakeLists.txt are checked out for building).
VLLM_PKG_DIR: Path | None = (
    Path(vllm.__file__).parent if getattr(vllm, "__file__", None) else None
)

needs_source = pytest.mark.skipif(
    REPO_ROOT is None,
    reason="C/CMake source tree not available (installed package only)",
)


def _read_source_or_skip(*relparts: str) -> str:
    """Read a C/CMake source file from the repo tree, or skip if absent.

    Used for csrc/ and CMakeLists.txt — these only exist in a source checkout,
    not in the installed wheel.
    """
    assert REPO_ROOT is not None  # callers are gated by @needs_source
    path = REPO_ROOT.joinpath(*relparts)
    if not path.exists():
        pytest.skip(f"{path} not present in this source tree")
    return path.read_text()


def _read_pkg_source_or_skip(*relparts: str) -> str:
    """Read a python source file from the installed vllm package.

    Reflects the code actually loaded at runtime, so these guard checks run in
    CI against the wheel — no source checkout required. Only skips for an
    exotic install layout (namespace/zipimport) where __file__ is unavailable.
    """
    if VLLM_PKG_DIR is None:
        pytest.skip("vllm package directory not resolvable (zip/namespace?)")
    assert VLLM_PKG_DIR is not None  # narrow for mypy (skip above is NoReturn)
    path = VLLM_PKG_DIR.joinpath(*relparts)
    if not path.exists():
        pytest.skip(f"{path} not present in installed vllm package")
    return path.read_text()


# ============================================================================
# Part A: POSITIVE — on gfx1100, ops exist and dispatch works
# ============================================================================


@gfx1100_only
@pytest.mark.parametrize("op_name", RDNA3_OPS)
def test_op_registered_on_gfx1100(op_name):
    """On gfx1100, all RDNA3 ops must be registered in _rocm_C."""
    assert hasattr(torch.ops, "_rocm_C"), "_rocm_C module not loaded"
    assert hasattr(torch.ops._rocm_C, op_name), (
        f"_rocm_C.{op_name} not registered — "
        "check CMakeLists.txt VLLM_ROCM_HAS_GFX1100 "
        "and torch_bindings.cpp #ifdef VLLM_ROCM_GFX1100"
    )


@gfx1100_only
def test_all_ops_present_or_all_absent():
    """The 3 RDNA3 ops are behind the same #ifdef — all present or all absent.

    Catches someone accidentally moving an op outside the guard.
    """
    has_rocm_c = hasattr(torch.ops, "_rocm_C")
    if not has_rocm_c:
        pytest.skip("_rocm_C not loaded")

    present = {op: hasattr(torch.ops._rocm_C, op) for op in RDNA3_OPS}
    values = set(present.values())
    assert len(values) == 1, (
        f"Guard inconsistency — some RDNA3 ops registered, others not: "
        f"{present}. Check torch_bindings.cpp #ifdef VLLM_ROCM_GFX1100 block."
    )


@pytest.mark.skipif(not _on_gfx11(), reason="Requires gfx11 hardware")
@pytest.mark.parametrize("op_name", GFX11_OPS)
def test_gfx11_op_registered_on_gfx11(op_name):
    """On any gfx11 GPU (not only gfx1100) the gfx11 ops must be registered."""
    assert hasattr(torch.ops, "_rocm_C"), "_rocm_C module not loaded"
    assert hasattr(torch.ops._rocm_C, op_name), (
        f"_rocm_C.{op_name} not registered — check CMakeLists.txt "
        "VLLM_ROCM_HAS_GFX11 and torch_bindings.cpp #ifdef VLLM_ROCM_GFX11"
    )


# ============================================================================
# Part B: NEGATIVE — compile-level verification on non-gfx1100
# ============================================================================


@not_gfx1100
@pytest.mark.parametrize("op_name", RDNA3_OPS)
def test_op_absent_on_non_gfx1100(op_name):
    """On non-gfx1100 (CDNA), RDNA3 ops must NOT exist in _rocm_C.

    This is the real compile-level check: the binary was built without
    gfx1100 support, so the ops should not have been compiled or registered.
    """
    if not hasattr(torch.ops, "_rocm_C"):
        return
    assert not hasattr(torch.ops._rocm_C, op_name), (
        f"_rocm_C.{op_name} is registered on non-gfx1100 hardware — "
        "compile guard is broken: check CMakeLists.txt "
        "VLLM_ROCM_HAS_GFX1100 and torch_bindings.cpp #ifdef"
    )


@not_gfx1100
def test_rocm_moe_not_supported_on_non_gfx1100():
    """The RDNA3 MoE experts must not be selectable on non-gfx1100 hardware."""
    from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import (
        Rdna3WNA16Experts,
    )

    assert Rdna3WNA16Experts._supports_current_device() is False, (
        "Rdna3WNA16Experts reported support on non-gfx1100 — dispatch guard is broken"
    )


@not_gfx1100
def test_dense_kernel_rejects_on_non_gfx1100():
    """RDNA3W4A16LinearKernel.can_implement must reject on non-gfx1100."""
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E501
        MPLinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 import (  # noqa: E501
        RDNA3W4A16LinearKernel,
    )
    from vllm.scalar_type import scalar_types

    config = MPLinearLayerConfig(
        full_weight_shape=(1024, 256),
        partition_weight_shape=(1024, 256),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=128,
        zero_points=False,
    )
    ok, reason = RDNA3W4A16LinearKernel.can_implement(config)
    assert ok is False, f"RDNA3 dense kernel accepted on non-gfx1100: {reason}"


# ============================================================================
# Part C: Static source analysis (build-level guards)
# ============================================================================


@needs_source
class TestCMakeGuards:
    """Verify CMakeLists.txt only compiles RDNA3 .cu files for gfx1100."""

    @staticmethod
    def _read_cmake():
        return _read_source_or_skip("CMakeLists.txt")

    def test_rdna3_cu_files_inside_gfx1100_conditional(self):
        """All RDNA3 .cu files must be listed inside the
        ``if(VLLM_GPU_ARCHES MATCHES "gfx1100")`` block, not unconditionally.
        """
        cmake = self._read_cmake()
        for cu_file in RDNA3_CU_FILES:
            assert cu_file in cmake, f"{cu_file} not found in CMakeLists.txt"

            lines = cmake.splitlines()
            in_gfx1100_block = False
            for line in lines:
                if 'VLLM_GPU_ARCHES MATCHES "gfx1100"' in line:
                    in_gfx1100_block = True
                if in_gfx1100_block and "endif()" in line:
                    in_gfx1100_block = False
                if cu_file in line:
                    assert in_gfx1100_block, (
                        f"{cu_file} is listed OUTSIDE the gfx1100 "
                        f"conditional in CMakeLists.txt — CDNA builds "
                        f"would compile RDNA3 code. Line: {line.strip()}"
                    )

    def test_gfx11_cu_files_inside_gfx11_conditional(self):
        """The gfx11 .cu files must be listed inside the
        ``if(VLLM_GPU_ARCHES MATCHES "gfx11[0-9]+")`` block, so a build that
        targets only gfx1151 still compiles them and CDNA builds do not.
        """
        cmake = self._read_cmake()
        for cu_file in GFX11_CU_FILES:
            assert cu_file in cmake, f"{cu_file} not found in CMakeLists.txt"
            in_gfx11_block = False
            for line in cmake.splitlines():
                if 'VLLM_GPU_ARCHES MATCHES "gfx11[0-9]+"' in line:
                    in_gfx11_block = True
                if in_gfx11_block and "endif()" in line:
                    in_gfx11_block = False
                if cu_file in line:
                    assert in_gfx11_block, (
                        f"{cu_file} is listed outside the gfx11 conditional "
                        f"in CMakeLists.txt. Line: {line.strip()}"
                    )

    def test_gfx11_compile_definition_only_for_gfx11(self):
        """VLLM_ROCM_GFX11 compile definition must be conditional."""
        cmake = self._read_cmake()
        in_gfx11_block = False
        for line in cmake.splitlines():
            if "VLLM_ROCM_HAS_GFX11)" in line:
                in_gfx11_block = True
            if in_gfx11_block and "endif()" in line:
                in_gfx11_block = False
            if (
                re.search(r"\bVLLM_ROCM_GFX11\b", line)
                and "target_compile_definitions" in line
            ):
                assert in_gfx11_block, (
                    "VLLM_ROCM_GFX11 compile definition is set outside the "
                    f"VLLM_ROCM_HAS_GFX11 conditional. Line: {line.strip()}"
                )

    def test_compile_definition_only_for_gfx1100(self):
        """VLLM_ROCM_GFX1100 compile definition must be conditional."""
        cmake = self._read_cmake()
        lines = cmake.splitlines()
        in_gfx1100_block = False
        for line in lines:
            if "VLLM_ROCM_HAS_GFX1100)" in line:
                in_gfx1100_block = True
            if in_gfx1100_block and "endif()" in line:
                in_gfx1100_block = False
            if "VLLM_ROCM_GFX1100" in line and "target_compile_definitions" in line:
                assert in_gfx1100_block, (
                    "VLLM_ROCM_GFX1100 compile definition is set outside "
                    "the VLLM_ROCM_HAS_GFX1100 conditional — CDNA builds "
                    f"would define it. Line: {line.strip()}"
                )


@needs_source
class TestTorchBindingsGuards:
    """Verify torch_bindings.cpp gates all RDNA3 ops behind #ifdef."""

    @staticmethod
    def _read_bindings():
        return _read_source_or_skip("csrc", "rocm", "torch_bindings.cpp")

    def test_all_rdna3_ops_inside_ifdef(self):
        """Every rdna3 op def/impl must be between #ifdef VLLM_ROCM_GFX1100
        and #endif. If any is outside, a CDNA build would try to register
        the op and link a symbol that doesn't exist.
        """
        src = self._read_bindings()
        lines = src.splitlines()

        inside_guard = False
        rdna3_lines_outside = []

        for i, line in enumerate(lines, 1):
            if line.strip() in BINDINGS_GUARDS:
                inside_guard = True
            elif line.strip() == "#endif" and inside_guard:
                inside_guard = False

            if (
                "rdna3" in line.lower()
                and not line.strip().startswith("//")
                and not inside_guard
            ):
                rdna3_lines_outside.append((i, line.strip()))

        assert not rdna3_lines_outside, (
            "RDNA3 op references found OUTSIDE #ifdef VLLM_ROCM_GFX1100 "
            "in torch_bindings.cpp — these would break CDNA builds:\n"
            + "\n".join(f"  L{n}: {s}" for n, s in rdna3_lines_outside)
        )

    def test_gfx11_ops_inside_gfx11_ifdef(self):
        """The gfx11 ops must be registered under #ifdef VLLM_ROCM_GFX11, not
        the gfx1100 guard, so gfx1150/gfx1151-only builds register them."""
        src = self._read_bindings()
        guard = None
        for i, line in enumerate(src.splitlines(), 1):
            if line.strip() in BINDINGS_GUARDS:
                guard = line.strip()
            elif line.strip() == "#endif":
                guard = None
            if any(f'"{op}' in line for op in GFX11_OPS):
                assert guard == "#ifdef VLLM_ROCM_GFX11", (
                    f"L{i}: gfx11 op registered under {guard}: {line.strip()}"
                )

    def test_no_unconditional_rdna3_includes(self):
        """No #include of RDNA3-specific headers outside the guard."""
        src = self._read_bindings()
        lines = src.splitlines()

        inside_guard = False
        for i, line in enumerate(lines, 1):
            if line.strip() in BINDINGS_GUARDS:
                inside_guard = True
            elif line.strip() == "#endif" and inside_guard:
                inside_guard = False

            if "#include" in line and "rdna3" in line.lower():
                assert inside_guard, (
                    f"L{i}: RDNA3 include outside gfx1100 guard: {line.strip()}"
                )


class TestCustomOpsGuards:
    """Verify _custom_ops.py gates register_fake behind hasattr checks."""

    @staticmethod
    def _read_custom_ops():
        return _read_pkg_source_or_skip("_custom_ops.py")

    def test_register_fake_guarded_by_hasattr(self):
        """Every register_fake for an RDNA3 op must be preceded by a hasattr
        check — otherwise it would crash on import on CDNA where the ops
        don't exist.
        """
        src = self._read_custom_ops()
        for op in RDNA3_OPS:
            pattern = rf'register_fake\(\s*"_rocm_C::{op}"\s*\)'
            match = re.search(pattern, src)
            if match is None:
                continue
            preceding = src[: match.start()]
            guards = list(
                re.finditer(
                    rf'hasattr\(\s*torch\.ops\._rocm_C,\s*"{op}"\s*\)', preceding
                )
            )
            assert guards, (
                f'register_fake("_rocm_C::{op}") is not preceded by a '
                f"hasattr check — would crash on CDNA import"
            )
            last_hasattr = guards[-1].start()
            gap = preceding[last_hasattr:].count("\n")
            assert gap <= 5, (
                f"hasattr guard for {op} is {gap} lines before "
                f"register_fake — suspiciously far; verify it's the "
                f"actual guard and not a coincidence"
            )

    def test_no_toplevel_rocm_c_import(self):
        """No top-level ``from vllm._rocm_C import`` — would crash on CDNA."""
        src = self._read_custom_ops()
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            assert "from vllm._rocm_C import" not in stripped, (
                f"Top-level import of _rocm_C in _custom_ops.py would "
                f"crash on CDNA: {stripped}"
            )


# ============================================================================
# Part D: Runtime mock (simulate CDNA on gfx1100 hardware)
# ============================================================================


class TestMoEDispatchMocked:
    """Mock on_gfx1100() to False and verify RDNA3 MoE is unreachable."""

    def test_kernel_unavailable_when_mocked_cdna(self):
        """The device gate must reject when not on gfx1100."""
        from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import (
            rdna3_moe_kernel_available,
        )

        with patch("vllm.platforms.rocm.on_gfx1100", return_value=False):
            assert rdna3_moe_kernel_available() is False

    def test_w4a8_kernel_unavailable_when_mocked_non_gfx11(self):
        from vllm.model_executor.layers.fused_moe.experts.rdna3_w4a8_moe import (
            rdna3_w4a8_moe_kernel_available,
        )

        with patch("vllm.platforms.rocm.on_gfx11", return_value=False):
            assert rdna3_w4a8_moe_kernel_available() is False

    @pytest.mark.parametrize(
        "weight_key",
        [kInt8Static, kInt4StaticAsym, kInt4Static32Asym, None],
    )
    def test_quant_scheme_rejects_non_symmetric_int4(self, weight_key):
        """Only symmetric int4 weight-only schemes reach the kernel."""
        from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import (
            Rdna3WNA16Experts,
        )

        assert Rdna3WNA16Experts._supports_quant_scheme(weight_key, None) is False

    @pytest.mark.parametrize("weight_key", [kInt4Static, kInt4Static32])
    def test_quant_scheme_accepts_symmetric_int4(self, weight_key):
        from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import (
            Rdna3WNA16Experts,
        )

        assert Rdna3WNA16Experts._supports_quant_scheme(weight_key, None) is True

    def test_kernel_unavailable_when_op_missing(self):
        """The device gate returns False when the C++ op doesn't exist."""
        from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import (
            rdna3_moe_kernel_available,
        )

        fake_rocm_c = type("FakeRocmC", (), {"gptq_gemm_rdna3": None})()
        with patch.object(torch, "ops", create=True) as mock_ops:
            mock_ops._rocm_C = fake_rocm_c
            assert rdna3_moe_kernel_available() is False

    def test_kernel_unavailable_when_rocm_c_absent(self):
        """The device gate returns False when _rocm_C doesn't exist at all."""
        from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import (
            rdna3_moe_kernel_available,
        )

        fake_ops = type("FakeOps", (), {})()
        with patch.object(torch, "ops", fake_ops):
            assert rdna3_moe_kernel_available() is False


class TestDenseKernelSelectionMocked:
    """Mock on_gfx1100() and verify dense RDNA3 kernel is not selected."""

    @gfx1100_only
    def test_can_implement_rejects_when_mocked_cdna(self):
        """RDNA3W4A16LinearKernel.can_implement must reject on mocked CDNA."""
        from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E501
            MPLinearLayerConfig,
        )
        from vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 import (  # noqa: E501
            RDNA3W4A16LinearKernel,
        )
        from vllm.scalar_type import scalar_types

        config = MPLinearLayerConfig(
            full_weight_shape=(1024, 256),
            partition_weight_shape=(1024, 256),
            weight_type=scalar_types.uint4b8,
            act_type=torch.float16,
            group_size=128,
            zero_points=False,
        )
        ok, _ = RDNA3W4A16LinearKernel.can_implement(config)
        assert ok is True

        with (
            patch("vllm.platforms.rocm.on_gfx1100", return_value=False),
            patch("vllm.platforms.rocm._ON_GFX1100", False),
        ):
            ok, reason = RDNA3W4A16LinearKernel.can_implement(config)
            assert ok is False, f"RDNA3 kernel accepted on simulated CDNA: {reason}"

    @gfx1100_only
    def test_chooser_skips_rdna3_when_mocked_cdna(self):
        """choose_mp_linear_kernel must NOT return RDNA3 on mocked CDNA."""
        from vllm.model_executor.kernels.linear import (
            choose_mp_linear_kernel,
        )
        from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E501
            MPLinearLayerConfig,
        )
        from vllm.scalar_type import scalar_types

        config = MPLinearLayerConfig(
            full_weight_shape=(1024, 256),
            partition_weight_shape=(1024, 256),
            weight_type=scalar_types.uint4b8,
            act_type=torch.float16,
            group_size=128,
            zero_points=False,
        )
        with (
            patch("vllm.platforms.rocm.on_gfx1100", return_value=False),
            patch("vllm.platforms.rocm._ON_GFX1100", False),
        ):
            chosen = choose_mp_linear_kernel(config)
            assert chosen.__name__ != "RDNA3W4A16LinearKernel", (
                "RDNA3 kernel was selected on simulated CDNA — "
                "choose_mp_linear_kernel guard is broken"
            )


class TestWNA16OracleWiring:
    """The RDNA3 backend must reach the kernel only through the oracle."""

    def test_backend_maps_to_rdna3_experts(self):
        from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import (
            Rdna3WNA16Experts,
        )
        from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
            WNA16MoEBackend,
            backend_to_kernel_cls,
            map_wna16_backend,
        )

        assert backend_to_kernel_cls(WNA16MoEBackend.RDNA3) == [Rdna3WNA16Experts]
        assert map_wna16_backend("rdna3") == WNA16MoEBackend.RDNA3

    def test_backend_is_offered_before_the_triton_fallback(self):
        """Priority order: the native kernel outranks Triton when supported."""
        from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
            WNA16MoEBackend,
            _get_priority_backends,
        )

        backends = _get_priority_backends()
        assert backends.index(WNA16MoEBackend.RDNA3) < backends.index(
            WNA16MoEBackend.TRITON
        )

    def test_w4a8_backend_maps_to_w4a8_experts(self):
        from vllm.model_executor.layers.fused_moe.experts.rdna3_w4a8_moe import (
            Rdna3W4A8Experts,
        )
        from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
            WNA16MoEBackend,
            backend_to_kernel_cls,
            map_wna16_backend,
        )

        assert backend_to_kernel_cls(WNA16MoEBackend.RDNA3_W4A8) == [Rdna3W4A8Experts]
        assert map_wna16_backend("rdna3_w4a8") == WNA16MoEBackend.RDNA3_W4A8

    def test_w4a8_backend_is_opt_in(self):
        """W4A8 changes activation numerics, so auto selection never picks it."""
        from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
            WNA16MoEBackend,
            _get_priority_backends,
        )

        assert WNA16MoEBackend.RDNA3_W4A8 not in _get_priority_backends()
