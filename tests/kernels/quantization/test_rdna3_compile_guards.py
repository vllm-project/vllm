#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-guard tests for the ROCm RDNA3 W4A16 kernels (dense + MoE).

Verifies that the RDNA3 compilation and dispatch guards are hermetic:
  - On gfx1100/gfx1151: scalar dense + MoE ops exist and dispatch selects
    RDNA3 kernels.
  - On gfx1100: the WMMA prefill op also exists.
  - On CDNA (gfx942/gfx950) or other non-RDNA3: ops must NOT exist,
    dispatch must fall through to Triton/Marlin, and no RDNA3 code
    path is reachable.

The negative (non-RDNA3) tests verify at three layers:
  1. Compile-level: on non-RDNA3 hardware, the RDNA3 ops are absent
     from the compiled _rocm_C extension — real binary verification.
  2. Static source analysis: parses CMakeLists.txt and torch_bindings.cpp
     to verify that all RDNA3 .cu files and op registrations are inside
     RDNA3-only guards.
  3. Runtime mock: patches on_gfx1100()/on_gfx1151() to False and verifies
     that the Python dispatch chain rejects the RDNA3 path.

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

from vllm.platforms.rocm import on_gfx1100, on_gfx1151  # noqa: E402

rdna3_only = pytest.mark.skipif(
    not (on_gfx1100() or on_gfx1151()),
    reason="Requires gfx1100 or gfx1151 hardware",
)

gfx1100_only = pytest.mark.skipif(
    not on_gfx1100(),
    reason="Requires gfx1100 hardware",
)

not_rdna3 = pytest.mark.skipif(
    on_gfx1100() or on_gfx1151(),
    reason="This test verifies non-RDNA3 builds — skip on RDNA3",
)

RDNA3_SCALAR_OPS = ["gptq_gemm_rdna3", "moe_gptq_gemm_rdna3"]
RDNA3_WMMA_OPS = ["gptq_gemm_rdna3_wmma"]
RDNA3_OPS = RDNA3_SCALAR_OPS + RDNA3_WMMA_OPS
RDNA3_SCALAR_CU_FILES = ["q_gemm_rdna3.cu", "moe_q_gemm_rdna3.cu"]
RDNA3_WMMA_CU_FILES = ["q_gemm_rdna3_wmma.cu"]
RDNA3_CU_FILES = RDNA3_SCALAR_CU_FILES + RDNA3_WMMA_CU_FILES


def _line_mentions_symbol(line: str, symbol: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])",
        line,
    ) is not None


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


@rdna3_only
@pytest.mark.parametrize("op_name", RDNA3_SCALAR_OPS)
def test_scalar_op_registered_on_rdna3(op_name):
    """On gfx1100/gfx1151, scalar RDNA3 ops must be registered."""
    assert hasattr(torch.ops, "_rocm_C"), "_rocm_C module not loaded"
    assert hasattr(torch.ops._rocm_C, op_name), (
        f"_rocm_C.{op_name} not registered — "
        "check CMakeLists.txt VLLM_ROCM_HAS_RDNA3_GEMM "
        "and torch_bindings.cpp #ifdef VLLM_ROCM_RDNA3_GEMM"
    )


@gfx1100_only
@pytest.mark.parametrize("op_name", RDNA3_WMMA_OPS)
def test_wmma_op_registered_on_gfx1100(op_name):
    """The WMMA prefill op is gfx1100-only."""
    assert hasattr(torch.ops, "_rocm_C"), "_rocm_C module not loaded"
    assert hasattr(torch.ops._rocm_C, op_name), (
        f"_rocm_C.{op_name} not registered — "
        "check CMakeLists.txt VLLM_ROCM_HAS_RDNA3_WMMA "
        "and torch_bindings.cpp #ifdef VLLM_ROCM_RDNA3_WMMA"
    )


@rdna3_only
def test_scalar_ops_present_or_absent_together():
    """The scalar dense and MoE ops are behind the same #ifdef."""
    if not hasattr(torch.ops, "_rocm_C"):
        pytest.skip("_rocm_C not loaded")

    present = {op: hasattr(torch.ops._rocm_C, op) for op in RDNA3_SCALAR_OPS}
    values = set(present.values())
    assert len(values) == 1, (
        f"Guard inconsistency — some scalar RDNA3 ops registered, others not: "
        f"{present}. Check torch_bindings.cpp #ifdef VLLM_ROCM_RDNA3_GEMM block."
    )


# ============================================================================
# Part B: NEGATIVE — compile-level verification on non-gfx1100
# ============================================================================


@not_rdna3
@pytest.mark.parametrize("op_name", RDNA3_OPS)
def test_op_absent_on_non_rdna3(op_name):
    """On non-RDNA3 (CDNA), RDNA3 ops must NOT exist in _rocm_C.

    This is the real compile-level check: the binary was built without
    RDNA3 support, so the ops should not have been compiled or registered.
    """
    if not hasattr(torch.ops, "_rocm_C"):
        return
    assert not hasattr(torch.ops._rocm_C, op_name), (
        f"_rocm_C.{op_name} is registered on non-RDNA3 hardware — "
        "compile guard is broken: check CMakeLists.txt "
        "VLLM_ROCM_HAS_RDNA3_GEMM and torch_bindings.cpp #ifdef"
    )


@not_rdna3
def test_rocm_moe_not_supported_on_non_rdna3():
    """rocm_moe_rdna.is_supported() must return False on non-RDNA3 hardware."""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        rocm_moe_rdna,
    )

    wq = type("WQ", (), {"num_bits": 4})()
    assert rocm_moe_rdna.is_supported(wq) is False, (
        "rocm_moe_rdna.is_supported() returned True on non-RDNA3 — "
        "dispatch guard is broken"
    )


@not_rdna3
def test_dense_kernel_rejects_on_non_rdna3():
    """RDNA3W4A16LinearKernel.can_implement must reject on non-RDNA3."""
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
        has_g_idx=False,
    )
    ok, reason = RDNA3W4A16LinearKernel.can_implement(config)
    assert ok is False, f"RDNA3 dense kernel accepted on non-RDNA3: {reason}"


# ============================================================================
# Part C: Static source analysis (build-level guards)
# ============================================================================


@needs_source
class TestCMakeGuards:
    """Verify CMakeLists.txt gates RDNA3 scalar and WMMA sources."""

    @staticmethod
    def _read_cmake():
        return _read_source_or_skip("CMakeLists.txt")

    def test_scalar_rdna3_cu_files_inside_rdna3_conditional(self):
        """Scalar RDNA3 .cu files are inside the gfx1100/gfx1151 block."""
        cmake = self._read_cmake()
        for cu_file in RDNA3_SCALAR_CU_FILES:
            assert cu_file in cmake, f"{cu_file} not found in CMakeLists.txt"

            lines = cmake.splitlines()
            in_rdna3_block = False
            for line in lines:
                if 'VLLM_GPU_ARCHES MATCHES "gfx1100|gfx1151"' in line:
                    in_rdna3_block = True
                if in_rdna3_block and "endif()" in line:
                    in_rdna3_block = False
                if cu_file in line:
                    assert in_rdna3_block, (
                        f"{cu_file} is listed OUTSIDE the RDNA3 "
                        f"conditional in CMakeLists.txt — CDNA builds "
                        f"would compile RDNA3 code. Line: {line.strip()}"
                    )

    def test_wmma_cu_file_inside_gfx1100_conditional(self):
        """The WMMA .cu file is inside the gfx1100-only block."""
        cmake = self._read_cmake()
        for cu_file in RDNA3_WMMA_CU_FILES:
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
                        f"{cu_file} is listed OUTSIDE the gfx1100-only "
                        f"conditional in CMakeLists.txt. Line: {line.strip()}"
                    )

    @pytest.mark.parametrize(
        ("has_var", "definition"),
        [
            ("VLLM_ROCM_HAS_RDNA3_GEMM", "VLLM_ROCM_RDNA3_GEMM"),
            ("VLLM_ROCM_HAS_RDNA3_WMMA", "VLLM_ROCM_RDNA3_WMMA"),
        ],
    )
    def test_compile_definition_only_inside_matching_conditional(
        self,
        has_var,
        definition,
    ):
        """RDNA3 compile definitions must be conditional."""
        cmake = self._read_cmake()
        lines = cmake.splitlines()
        inside_block = False
        for line in lines:
            if f"{has_var})" in line:
                inside_block = True
            if inside_block and "endif()" in line:
                inside_block = False
            if definition in line and "target_compile_definitions" in line:
                assert inside_block, (
                    f"{definition} compile definition is set outside "
                    f"the {has_var} conditional. Line: {line.strip()}"
                )


@needs_source
class TestTorchBindingsGuards:
    """Verify torch_bindings.cpp gates RDNA3 ops behind #ifdefs."""

    @staticmethod
    def _read_bindings():
        return _read_source_or_skip("csrc", "rocm", "torch_bindings.cpp")

    @pytest.mark.parametrize(
        ("ops", "guard"),
        [
            (RDNA3_SCALAR_OPS, "VLLM_ROCM_RDNA3_GEMM"),
            (RDNA3_WMMA_OPS, "VLLM_ROCM_RDNA3_WMMA"),
        ],
    )
    def test_rdna3_ops_inside_matching_ifdef(self, ops, guard):
        """Every RDNA3 op def/impl must be inside its matching guard."""
        src = self._read_bindings()
        lines = src.splitlines()

        inside_guard = False
        lines_outside = []

        for i, line in enumerate(lines, 1):
            if f"#ifdef {guard}" in line:
                inside_guard = True
            elif line.strip() == "#endif" and inside_guard:
                inside_guard = False

            if (
                any(_line_mentions_symbol(line, op) for op in ops)
                and not line.strip().startswith("//")
                and not inside_guard
            ):
                lines_outside.append((i, line.strip()))

        assert not lines_outside, (
            f"RDNA3 op references found OUTSIDE #ifdef {guard} "
            "in torch_bindings.cpp:\n"
            + "\n".join(f"  L{n}: {s}" for n, s in lines_outside)
        )

    def test_no_unconditional_rdna3_includes(self):
        """No #include of RDNA3-specific headers outside the guard."""
        src = self._read_bindings()
        lines = src.splitlines()

        inside_guard = False
        for i, line in enumerate(lines, 1):
            if "#ifdef VLLM_ROCM_RDNA3_GEMM" in line:
                inside_guard = True
            elif line.strip() == "#endif" and inside_guard:
                inside_guard = False

            if "#include" in line and "rdna3" in line.lower():
                assert inside_guard, (
                    f"L{i}: RDNA3 include outside RDNA3 guard: {line.strip()}"
                )


@needs_source
class TestQGemmRdna3DispatchGuards:
    """Verify dense RDNA3 dispatch keeps WMMA gfx1100-only at runtime."""

    @staticmethod
    def _read_q_gemm_rdna3():
        return _read_source_or_skip("csrc", "rocm", "q_gemm_rdna3.cu")

    def test_wmma_dispatch_checks_runtime_arch(self):
        """Fat binaries may contain both gfx1100 and gfx1151 code objects.

        In that case VLLM_ROCM_RDNA3_WMMA is defined, but gfx1151 must still
        use the scalar path because q_gemm_rdna3_wmma.cu is gfx1100-only.
        """
        src = self._read_q_gemm_rdna3()
        assert "current_device_supports_rdna3_wmma" in src
        assert "gcnArchName" in src
        assert 'device_arch.rfind("gfx1100", 0) == 0' in src
        assert "current_device_supports_rdna3_wmma() &&" in src
        assert "return gptq_gemm_rdna3_wmma" in src


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
            last_hasattr = preceding.rfind(f'hasattr(torch.ops._rocm_C, "{op}")')
            assert last_hasattr != -1, (
                f'register_fake("_rocm_C::{op}") is not preceded by a '
                f"hasattr check — would crash on CDNA import"
            )
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


class _FakeWeightQuant:
    """Minimal stand-in for a weight quantization config."""

    def __init__(self, num_bits):
        self.num_bits = num_bits


class TestMoEDispatchMocked:
    """Mock on_gfx1100()/on_gfx1151() to False and verify RDNA3 MoE is unreachable."""

    def test_is_supported_false_when_mocked_cdna(self):
        """rocm_moe_rdna.is_supported() must return False when not on RDNA3."""
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
            rocm_moe_rdna,
        )

        with (
            patch("vllm.platforms.rocm.on_gfx1100", return_value=False),
            patch("vllm.platforms.rocm.on_gfx1151", return_value=False),
        ):
            assert rocm_moe_rdna.is_supported(_FakeWeightQuant(num_bits=4)) is False

    @pytest.mark.parametrize("num_bits", [2, 3, 8, 16])
    def test_is_supported_rejects_non_w4(self, num_bits):
        """is_supported() rejects non-4-bit even before checking arch."""
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
            rocm_moe_rdna,
        )

        assert rocm_moe_rdna.is_supported(_FakeWeightQuant(num_bits=num_bits)) is False

    def test_is_supported_false_when_op_missing(self):
        """is_supported() returns False when the C++ op doesn't exist."""
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
            rocm_moe_rdna,
        )

        fake_rocm_c = type("FakeRocmC", (), {"gptq_gemm_rdna3": None})()
        with patch.object(torch, "ops", create=True) as mock_ops:
            mock_ops._rocm_C = fake_rocm_c
            assert rocm_moe_rdna.is_supported(_FakeWeightQuant(num_bits=4)) is False

    def test_is_supported_false_when_rocm_c_absent(self):
        """is_supported() returns False when _rocm_C doesn't exist at all."""
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
            rocm_moe_rdna,
        )

        fake_ops = type("FakeOps", (), {})()
        with patch.object(torch, "ops", fake_ops):
            assert rocm_moe_rdna.is_supported(_FakeWeightQuant(num_bits=4)) is False


class TestDenseKernelSelectionMocked:
    """Mock on_gfx1100()/on_gfx1151() and verify dense RDNA3 kernel is not selected."""

    @rdna3_only
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
            has_g_idx=False,
        )
        ok, _ = RDNA3W4A16LinearKernel.can_implement(config)
        assert ok is True

        with (
            patch("vllm.platforms.rocm.on_gfx1100", return_value=False),
            patch("vllm.platforms.rocm._ON_GFX1100", False),
            patch("vllm.platforms.rocm.on_gfx1151", return_value=False),
            patch("vllm.platforms.rocm._ON_GFX1151", False),
        ):
            ok, reason = RDNA3W4A16LinearKernel.can_implement(config)
            assert ok is False, f"RDNA3 kernel accepted on simulated CDNA: {reason}"

    @rdna3_only
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
            has_g_idx=False,
        )
        with (
            patch("vllm.platforms.rocm.on_gfx1100", return_value=False),
            patch("vllm.platforms.rocm._ON_GFX1100", False),
            patch("vllm.platforms.rocm.on_gfx1151", return_value=False),
            patch("vllm.platforms.rocm._ON_GFX1151", False),
        ):
            chosen = choose_mp_linear_kernel(config)
            assert chosen.__name__ != "RDNA3W4A16LinearKernel", (
                "RDNA3 kernel was selected on simulated CDNA — "
                "choose_mp_linear_kernel guard is broken"
            )


class TestCompressedTensorsMoEDispatchGuard:
    """Verify compressed_tensors_moe.py only enters rocm_moe_rdna under is_rocm()."""

    def test_rocm_guard_in_dispatch_source(self):
        """The rocm_moe_rdna import and call must be inside an is_rocm() check."""
        src = _read_pkg_source_or_skip(
            "model_executor",
            "layers",
            "quantization",
            "compressed_tensors",
            "compressed_tensors_moe",
            "compressed_tensors_moe.py",
        )
        lines = src.splitlines()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "rocm_moe" in stripped and not stripped.startswith("#"):
                found_guard = False
                for j in range(i - 1, max(0, i - 15), -1):
                    if "is_rocm()" in lines[j - 1]:
                        found_guard = True
                        break
                assert found_guard, (
                    f"L{i}: rocm_moe_rdna reference not protected by "
                    f"is_rocm() guard: {stripped}"
                )
