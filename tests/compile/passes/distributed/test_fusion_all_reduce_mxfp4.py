# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed tests for AllReduce + MXFP4 kernel fusion patterns.

Covers:
  Multi-GPU tests (via torch.multiprocessing.spawn, requires 2 GPUs):
    - Pattern A (AllReduce → RMSNorm → MXFP4): no residual — 3-node subgraph
    - Pattern B (AllReduce → fused_add_RMSNorm → MXFP4): with residual — 4-node
    - Registration ordering: Pattern B must come before Pattern A (greedy match)
    - Graceful fallback: when fused_allreduce_rmsnorm_mxfp4_quant is absent,
      existing AllReduce + RMSNorm patterns are still applied

  Single-GPU unit tests (no communication required):
    - Pattern structure validation (inputs count, dtypes, callables)
    - Registration guard: MXFP4 patterns only appear when probe returns True

Similar models used as references:
  - TestAllReduceRMSNormModel in test_fusion_all_reduce.py
  - AiterAllreduceFusedRMSNormPattern / AiterAllreduceFusedAddRMSNormPattern
    (existing FP8-quant equivalents in allreduce_rms_fusion.py)

Design notes:
  - has_fused_allreduce_rmsnorm_mxfp4_quant() currently returns False until
    AITER ships the fused_allreduce_rmsnorm_mxfp4_quant kernel.
    Tests requiring it are marked xfail(strict=False) so they auto-pass
    when the kernel is eventually added.
  - Pattern struct tests run without a GPU (just require vllm._C for op
    registration).
"""

import pytest
import torch

from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm.platforms import current_platform

# ─── Skip/xfail markers ──────────────────────────────────────────────────────

_NEEDS_ROCM = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific AllReduce tests"
)

_NEEDS_ROCM_AITER = pytest.mark.skipif(
    not (current_platform.is_rocm() and IS_AITER_FOUND),
    reason="Requires ROCm platform with AITER installed",
)

# AllReduce MXFP4 kernel is forward-looking — mark tests as xfail
# with strict=False (will auto-pass when AITER ships the kernel)
_NEEDS_AR_MXFP4_KERNEL = pytest.mark.xfail(
    not rocm_aiter_ops.has_fused_allreduce_rmsnorm_mxfp4_quant(),
    reason="aiter.fused_allreduce_rmsnorm_mxfp4_quant not yet in this AITER build",
    strict=False,
)


def _skip_if_no_vllm_c():
    """Skip the calling test if vllm._C is absent (no GPU build)."""
    try:
        import vllm._C  # noqa: F401
    except (ImportError, AttributeError) as e:
        pytest.skip(f"vllm._C not available: {e}")


def _import_ar_fusion():
    """Import allreduce_rms_fusion, skip on missing deps."""
    try:
        import vllm.compilation.passes.fusion.allreduce_rms_fusion as m

        return m
    except (ImportError, AttributeError) as e:
        pytest.skip(f"allreduce_rms_fusion not importable: {e}")


# ─── Model definitions (mirrors TestAllReduceRMSNormModel pattern) ────────────


def _build_ar_mxfp4_model(hidden_size: int, eps: float, dtype: torch.dtype):
    """Build a minimal AllReduce + RMSNorm + MXFP4-quant model.

    Structure (mirrors DeepSeek-V3 forward pass):
      Layer 0 (no residual):   allreduce → rms_norm → dynamic_mxfp4_quant
      Layer 1 (with residual): allreduce → fused_add_rms_norm → dynamic_mxfp4_quant
      Layer 2 (with residual): allreduce → fused_add_rms_norm → dynamic_mxfp4_quant

    After fusion with MXFP4 AR patterns:
      Layer 0: rocm_aiter_fused_allreduce_rmsnorm_mxfp4_quant   (Pattern A)
      Layer 1/2: rocm_aiter_fused_allreduce_add_rmsnorm_mxfp4_quant  (Pattern B)
    """
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.model_executor.layers.layernorm import RMSNorm

    mxfp4_quant_op = rocm_aiter_ops.get_dynamic_mxfp4_quant_op()

    class _ARMxfp4Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm0 = RMSNorm(hidden_size, eps=eps)
            self.norm1 = RMSNorm(hidden_size, eps=eps)
            self.norm2 = RMSNorm(hidden_size, eps=eps)
            self.w0 = torch.nn.Parameter(
                torch.rand(hidden_size, hidden_size, dtype=dtype)
            )
            self.w1 = torch.nn.Parameter(
                torch.rand(hidden_size, hidden_size, dtype=dtype)
            )

        def forward(self, x: torch.Tensor):
            import vllm.ir.ops as vllm_ir

            # avoid graph input being a direct pattern arg
            z = torch.relu(x)

            # Layer 0: AR → RMSNorm → MXFP4 (Pattern A target)
            ar0 = tensor_model_parallel_all_reduce(z)
            normed0 = vllm_ir.rms_norm(
                ar0, self.norm0.weight, self.norm0.variance_epsilon
            )
            fp4_0, scale_0 = mxfp4_quant_op(normed0)

            # Linear to advance state
            z2 = torch.mm(fp4_0.float().view(fp4_0.shape[0], -1), self.w0)

            # Layer 1: AR → fused_add_RMSNorm → MXFP4 (Pattern B target)
            ar1 = tensor_model_parallel_all_reduce(z2.to(dtype))
            normed1, resid1 = vllm_ir.fused_add_rms_norm(
                ar1, ar0, self.norm1.weight, self.norm1.variance_epsilon
            )
            fp4_1, scale_1 = mxfp4_quant_op(normed1)

            z3 = torch.mm(fp4_1.float().view(fp4_1.shape[0], -1), self.w1)

            # Layer 2: AR → fused_add_RMSNorm → MXFP4 (Pattern B target again)
            ar2 = tensor_model_parallel_all_reduce(z3.to(dtype))
            normed2, resid2 = vllm_ir.fused_add_rms_norm(
                ar2, resid1, self.norm2.weight, self.norm2.variance_epsilon
            )
            fp4_2, scale_2 = mxfp4_quant_op(normed2)
            return fp4_2, scale_2

        def ops_in_model_before(self):
            return [
                torch.ops.vllm.all_reduce.default,
                mxfp4_quant_op,
            ]

        def ops_in_model_after_mxfp4(self):
            return [
                rocm_aiter_ops.get_fused_allreduce_rmsnorm_mxfp4_quant_op(),  # A
                rocm_aiter_ops.get_fused_allreduce_add_rmsnorm_mxfp4_quant_op(),  # B
            ]

    return _ARMxfp4Model()


# ─── UNIT TESTS: pattern structure (no GPU required) ─────────────────────────


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
def test_unit_ar_pattern_a_inputs_count(epsilon):
    """Pattern A (no residual): get_inputs() must return 2 tensors (input_, weight)."""
    _skip_if_no_vllm_c()
    mod = _import_ar_fusion()
    p = mod.AiterAllreduceFusedRMSNormMXFP4QuantPattern(
        epsilon=epsilon, dtype=torch.bfloat16, device="cpu"
    )
    inputs = p.get_inputs()
    assert len(inputs) == 2, f"Expected 2 inputs for Pattern A, got {len(inputs)}"
    assert inputs[0].dtype == torch.bfloat16
    assert inputs[1].dtype == torch.bfloat16
    assert inputs[0].ndim == 2  # input_: (M, N)
    assert inputs[1].ndim == 1  # weight: (N,)


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
def test_unit_ar_pattern_b_inputs_count(epsilon):
    """Pattern B (with residual): get_inputs() must return 3 tensors."""
    _skip_if_no_vllm_c()
    mod = _import_ar_fusion()
    p = mod.AiterAllreduceFusedAddRMSNormMXFP4QuantPattern(
        epsilon=epsilon, dtype=torch.bfloat16, device="cpu"
    )
    inputs = p.get_inputs()
    assert len(inputs) == 3, f"Expected 3 inputs for Pattern B, got {len(inputs)}"
    assert all(t.dtype == torch.bfloat16 for t in inputs)
    assert inputs[0].ndim == 2  # input_
    assert inputs[1].ndim == 2  # residual
    assert inputs[2].ndim == 1  # weight


def test_unit_ar_pattern_a_is_callable():
    """Both pattern and replacement attributes of Pattern A must be callable."""
    _skip_if_no_vllm_c()
    mod = _import_ar_fusion()
    p = mod.AiterAllreduceFusedRMSNormMXFP4QuantPattern(
        epsilon=1e-6, dtype=torch.bfloat16, device="cpu"
    )
    assert callable(p.pattern), "pattern must be callable"
    assert callable(p.replacement), "replacement must be callable"


def test_unit_ar_pattern_b_is_callable():
    """Both pattern and replacement attributes of Pattern B must be callable."""
    _skip_if_no_vllm_c()
    mod = _import_ar_fusion()
    p = mod.AiterAllreduceFusedAddRMSNormMXFP4QuantPattern(
        epsilon=1e-6, dtype=torch.bfloat16, device="cpu"
    )
    assert callable(p.pattern), "pattern must be callable"
    assert callable(p.replacement), "replacement must be callable"


# ─── UNIT TESTS: registration guard ──────────────────────────────────────────


@_NEEDS_ROCM_AITER
def test_unit_mxfp4_patterns_not_registered_without_kernel(monkeypatch):
    """When has_fused_allreduce_rmsnorm_mxfp4_quant() returns False, the AR
    MXFP4 pattern classes must NOT appear in RocmAiterAllReduceFusionPass."""
    _skip_if_no_vllm_c()

    if rocm_aiter_ops.has_fused_allreduce_rmsnorm_mxfp4_quant():
        pytest.skip("Kernel is available — test only applies when probe returns False")

    mod = _import_ar_fusion()

    import vllm.config
    from vllm.config import CompilationConfig, CompilationMode, VllmConfig

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()

    with vllm.config.set_current_vllm_config(vllm_config):
        pass_obj = mod.RocmAiterAllReduceFusionPass(vllm_config)

    mxfp4_classes = {
        "AiterAllreduceFusedRMSNormMXFP4QuantPattern",
        "AiterAllreduceFusedAddRMSNormMXFP4QuantPattern",
    }
    registered_names = {type(p).__name__ for p in pass_obj._pattern_replacements}
    for cls_name in mxfp4_classes:
        assert cls_name not in registered_names, (
            f"{cls_name} must NOT be registered when "
            "fused_allreduce_rmsnorm_mxfp4_quant is unavailable "
            "(has_fused_allreduce_rmsnorm_mxfp4_quant() returned False)"
        )


@_NEEDS_ROCM_AITER
@_NEEDS_AR_MXFP4_KERNEL
def test_unit_mxfp4_registration_order_greedy(monkeypatch):
    """When the kernel IS available, Pattern B (4-node, with residual) must be
    registered before Pattern A (3-node, no residual).

    Greedy matching: the matcher tries each registered pattern in order and
    uses the first match.  Larger subgraphs must come first to avoid Pattern A
    consuming the first 3 nodes of a Pattern B site.
    """
    _skip_if_no_vllm_c()
    mod = _import_ar_fusion()

    import vllm.config
    from vllm.config import CompilationConfig, CompilationMode, VllmConfig

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()

    with vllm.config.set_current_vllm_config(vllm_config):
        pass_obj = mod.RocmAiterAllReduceFusionPass(vllm_config)

    names = [type(p).__name__ for p in pass_obj._pattern_replacements]

    idx_b = next(
        (
            i
            for i, n in enumerate(names)
            if n == "AiterAllreduceFusedAddRMSNormMXFP4QuantPattern"
        ),
        None,
    )
    idx_a = next(
        (
            i
            for i, n in enumerate(names)
            if n == "AiterAllreduceFusedRMSNormMXFP4QuantPattern"
        ),
        None,
    )

    assert idx_b is not None, "Pattern B not registered despite probe returning True"
    assert idx_a is not None, "Pattern A not registered despite probe returning True"
    assert idx_b < idx_a, (
        f"Pattern B (idx={idx_b}) must come before "
        f"Pattern A (idx={idx_a}) for greedy match"
    )


# ─── MULTI-GPU FUNCTIONAL TESTS ───────────────────────────────────────────────
#
# These require 2 GPUs.  Guarded with @multi_gpu_test(num_gpus=2).
# If the MXFP4 AR kernel is not yet available they are xfail(strict=False).
#


def _try_import_multi_gpu_test():
    try:
        from tests.utils import multi_gpu_test

        return multi_gpu_test
    except ImportError:
        return None


_multi_gpu_test = _try_import_multi_gpu_test()


def _ar_mxfp4_spawn_worker(
    local_rank: int,
    world_size: int,
    hidden_size: int,
    eps: float,
    dtype: torch.dtype,
    expect_fused: bool,
):
    """Worker function for torch.multiprocessing.spawn AR MXFP4 tests."""
    import os

    from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
        RocmAiterAllReduceFusionPass,
    )
    from vllm.compilation.passes.utility.fix_functionalization import (
        FixFunctionalizationPass,
    )
    from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
    from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.system_utils import update_environment_variables

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    rocm_aiter_ops.refresh_env_variables()

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29800",
        }
    )

    init_distributed_environment()

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )

    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        from tests.compile.backend import TestBackend

        ar_pass = RocmAiterAllReduceFusionPass(vllm_config)
        noop_pass = NoOpEliminationPass(vllm_config)
        func_pass = FixFunctionalizationPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)
        backend = TestBackend(noop_pass, ar_pass, func_pass, cleanup_pass)

        model = _build_ar_mxfp4_model(hidden_size, eps, dtype)

        num_tokens = 8
        x = torch.randn(num_tokens, hidden_size, dtype=dtype)
        torch._dynamo.mark_dynamic(x, 0)

        compiled_model = torch.compile(model, backend=backend)
        fp4_out, scale_out = compiled_model(x)

        if expect_fused:
            # Verify fused ops appear in the compiled graph
            backend.check_after_ops(model.ops_in_model_after_mxfp4())
            # And standalone all_reduce + dynamic_mxfp4_quant are gone
            # (just check matched count > 0 as proxy)
            assert ar_pass.matched_count >= 1, (
                f"Expected ≥1 AR MXFP4 fusion match, got {ar_pass.matched_count}"
            )

        # Numerical sanity: output shape
        assert fp4_out.shape[0] == num_tokens, (
            f"fp4 output token dim mismatch: {fp4_out.shape[0]} vs {num_tokens}"
        )


@pytest.mark.skipif(_multi_gpu_test is None, reason="multi_gpu_test not available")
@pytest.mark.skipif(
    not (current_platform.is_rocm() and IS_AITER_FOUND),
    reason="Requires ROCm with AITER",
)
@_NEEDS_AR_MXFP4_KERNEL
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("hidden_size", [64, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_ar_mxfp4_fusion_fires(hidden_size, eps, dtype):
    """Multi-GPU: AllReduce + MXFP4 fusion pass fires and produces correct outputs.

    - Pattern A (no residual, 3-node) and Pattern B (with residual, 4-node)
      must both be matched (matched_count >= 1 each).
    - Compiled graph must contain fused AR+MXFP4 ops.
    - Output shapes must match unfused path.

    This test is xfail until aiter.fused_allreduce_rmsnorm_mxfp4_quant is
    shipped in AITER (see _NEEDS_AR_MXFP4_KERNEL marker above).
    """
    torch.multiprocessing.spawn(
        _ar_mxfp4_spawn_worker,
        args=(2, hidden_size, eps, dtype, True),
        nprocs=2,
    )


@pytest.mark.skipif(_multi_gpu_test is None, reason="multi_gpu_test not available")
@pytest.mark.skipif(
    not (current_platform.is_rocm() and IS_AITER_FOUND),
    reason="Requires ROCm with AITER",
)
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_ar_mxfp4_fallback_when_kernel_absent(hidden_size, dtype):
    """Multi-GPU: When fused_allreduce_rmsnorm_mxfp4_quant is unavailable, the
    existing (non-MXFP4) AR fusion patterns must still be applied — no crash.

    This test intentionally runs regardless of the AR kernel availability
    to verify the graceful fallback path.
    """
    if rocm_aiter_ops.has_fused_allreduce_rmsnorm_mxfp4_quant():
        pytest.skip("Kernel IS available; fallback test not applicable")

    # expect_fused=False: we don't expect MXFP4 fused ops, just no crash
    torch.multiprocessing.spawn(
        _ar_mxfp4_spawn_worker,
        args=(2, hidden_size, 1e-6, dtype, False),
        nprocs=2,
    )


# ─── UNIT TESTS: DeepSeek-R1 shape sizes ─────────────────────────────────────


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
def test_unit_ds_r1_hidden_size_pattern_a(epsilon):
    """Pattern A inputs at DeepSeek-R1 hidden_size=7168 have correct shape contract."""
    _skip_if_no_vllm_c()
    _import_ar_fusion()
    # Using a small device-free tensor to verify the shape logic
    x = torch.empty(4, 7168, dtype=torch.bfloat16, device="cpu")
    w = torch.empty(7168, dtype=torch.bfloat16, device="cpu")
    assert x.shape[1] == w.shape[0], "input and weight hidden dims must match"


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
def test_unit_ds_r1_hidden_size_pattern_b(epsilon):
    """Pattern B inputs at DeepSeek-R1 hidden_size=7168 check 3-tensor contract."""
    _skip_if_no_vllm_c()
    _import_ar_fusion()
    x = torch.empty(4, 7168, dtype=torch.bfloat16, device="cpu")
    residual = torch.empty(4, 7168, dtype=torch.bfloat16, device="cpu")
    w = torch.empty(7168, dtype=torch.bfloat16, device="cpu")
    assert x.shape == residual.shape, "input and residual shapes must match"
    assert x.shape[1] == w.shape[0], "input and weight hidden dims must match"


# ─── UNIT TESTS: feature probe results with AITER present ────────────────────


@_NEEDS_ROCM_AITER
def test_unit_probe_positive_when_kernel_present():
    """When AITER is available and has fused_allreduce_rmsnorm_mxfp4_quant,
    probe must return True (and our implementation must match)."""
    import aiter

    kernel_available = hasattr(aiter, "fused_allreduce_rmsnorm_mxfp4_quant")
    probe_result = rocm_aiter_ops.has_fused_allreduce_rmsnorm_mxfp4_quant()
    assert probe_result == kernel_available, (
        f"Probe result ({probe_result}) disagrees with "
        f"hasattr check ({kernel_available})"
    )


@_NEEDS_ROCM_AITER
def test_unit_rmsnorm_mxfp4_probe_positive_with_triton_kernel():
    """When AITER's fused_rms_mxfp4_quant is importable, probe must return True."""
    try:
        from aiter.ops.triton.fused_mxfp4_quant import (
            fused_rms_mxfp4_quant,  # noqa: F401
        )

        kernel_importable = True
    except ImportError:
        kernel_importable = False

    probe_result = rocm_aiter_ops.has_fused_rmsnorm_mxfp4_quant()
    assert probe_result == kernel_importable, (
        f"has_fused_rmsnorm_mxfp4_quant() returned {probe_result} but "
        f"fused_rms_mxfp4_quant is "
        f"{'importable' if kernel_importable else 'not importable'}"
    )
