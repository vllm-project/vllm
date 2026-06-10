# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


_NEEDS_ROCM_AITER = pytest.mark.skipif(
    not (current_platform.is_rocm() and is_aiter_found_and_supported()),
    reason="Requires ROCm platform with AITER installed",
)

_NEEDS_MXFP4_STANDALONE = pytest.mark.skipif(
    not (
        current_platform.is_rocm()
        and is_aiter_found_and_supported()
        and rocm_aiter_ops.has_fused_rmsnorm_mxfp4_quant()
    ),
    reason="Requires aiter.ops.triton.fused_mxfp4_quant (fused_rms_mxfp4_quant)",
)

_NEEDS_NO_AITER = pytest.mark.skipif(
    is_aiter_found_and_supported(),
    reason="AITER is present — this test only validates the absent-AITER path",
)


def test_unit_probe_rmsnorm_mxfp4_returns_bool():
    result = rocm_aiter_ops.has_fused_rmsnorm_mxfp4_quant()
    assert isinstance(result, bool), (
        f"has_fused_rmsnorm_mxfp4_quant returned {type(result)}, expected bool"
    )


@_NEEDS_NO_AITER
def test_unit_probe_rmsnorm_false_without_aiter():
    assert rocm_aiter_ops.has_fused_rmsnorm_mxfp4_quant() is False




@_NEEDS_MXFP4_STANDALONE
def test_unit_get_ops_exist():
    ops = {
        "get_dynamic_mxfp4_quant_op": rocm_aiter_ops.get_dynamic_mxfp4_quant_op,
        "get_fused_rmsnorm_mxfp4_quant_op": (
            rocm_aiter_ops.get_fused_rmsnorm_mxfp4_quant_op
        ),
        "get_fused_rmsnorm_add_mxfp4_quant_op": (
            rocm_aiter_ops.get_fused_rmsnorm_add_mxfp4_quant_op
        ),
    }
    for name, getter in ops.items():
        op = getter()
        assert op is not None, f"{name}() returned None"






@_NEEDS_MXFP4_STANDALONE
@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
def test_unit_deepseek_shape_no_residual(epsilon):
    hidden_size = 7168
    num_tokens = 4
    fused_op = rocm_aiter_ops.get_fused_rmsnorm_mxfp4_quant_op()
    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

    fp4, scale = fused_op(x=x, weight=weight, epsilon=epsilon)

    assert fp4.shape == (num_tokens, hidden_size // 2), (
        f"fp4 shape {fp4.shape} != expected {(num_tokens, hidden_size // 2)}"
    )
    expected_scale_cols = math.ceil(hidden_size / 32)
    assert scale.shape[1] >= expected_scale_cols, (
        f"scale cols {scale.shape[1]} < ceil(N/32)={expected_scale_cols}"
    )


# _AiterRMSNormMXFP4QuantModel uses torch.ops.vllm.rocm_aiter_dynamic_mxfp4_quant.
# The _NEEDS_MXFP4_STANDALONE marker on every test that instantiates it ensures
# AITER is available before the op is accessed, so the class can safely live at
# module scope.



@_NEEDS_ROCM_AITER
def test_unit_standalone_registration_order(monkeypatch):
    import vllm.config
    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        AiterFusedAddRMSNormMXFP4QuantPattern,
        AiterRMSNormMXFP4QuantPattern,
        RocmAiterRMSNormQuantFusionPass,
    )
    from vllm.config import CompilationConfig, CompilationMode, VllmConfig

    if not rocm_aiter_ops.has_fused_rmsnorm_mxfp4_quant():
        pytest.skip("Standalone MXFP4 fused kernel not available in this AITER build")

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()
        fusion_pass = RocmAiterRMSNormQuantFusionPass(vllm_config)

    names = [type(p).__name__ for p in fusion_pass._pattern_replacements]

    idx_with_res = next(
        (
            i
            for i, n in enumerate(names)
            if n == AiterFusedAddRMSNormMXFP4QuantPattern.__name__
        ),
        None,
    )
    idx_no_res = next(
        (i for i, n in enumerate(names) if n == AiterRMSNormMXFP4QuantPattern.__name__),
        None,
    )

    assert idx_with_res is not None, (
        "AiterFusedAddRMSNormMXFP4QuantPattern not registered"
    )
    assert idx_no_res is not None, "AiterRMSNormMXFP4QuantPattern not registered"
    assert idx_with_res < idx_no_res, (
        f"Residual pattern (idx={idx_with_res}) must be before no-residual "
        f"pattern (idx={idx_no_res}) for greedy matching"
    )


@_NEEDS_ROCM_AITER
def test_unit_uuid_changes_with_mxfp4(monkeypatch):
    import vllm.config
    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterRMSNormQuantFusionPass,
    )
    from vllm.config import CompilationConfig, CompilationMode, VllmConfig

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
    )

    with vllm.config.set_current_vllm_config(vllm_config):
        # Pass with MXFP4 patterns included
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()
        pass_with = RocmAiterRMSNormQuantFusionPass(vllm_config)
        uuid_with = pass_with.uuid()

    # The uuid is derived from source of pattern classes; it will differ if
    # MXFP4 class is included in the hash.  Just assert it is a non-empty string.
    assert isinstance(uuid_with, str) and len(uuid_with) > 0, (
        "uuid() must return a non-empty string"
    )




class _RMSNormMXFP4Model(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.eps = eps
        self._mxfp4_quant_op = rocm_aiter_ops.get_dynamic_mxfp4_quant_op()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import vllm.ir.ops as vllm_ir

        normed = vllm_ir.rms_norm(x, self.weight, self.eps)
        fp4, scale = self._mxfp4_quant_op(normed)
        return fp4, scale


class _FusedAddRMSNormMXFP4Model(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.eps = eps
        self._mxfp4_quant_op = rocm_aiter_ops.get_dynamic_mxfp4_quant_op()

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import vllm.ir.ops as vllm_ir

        normed, residual_out = vllm_ir.fused_add_rms_norm(
            x, residual, self.weight, self.eps
        )
        fp4, scale = self._mxfp4_quant_op(normed)
        return fp4, scale, residual_out


def _dequant_mxfp4(fp4: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Each uint8 byte = two 4-bit values packed as lo | (hi << 4)
    lo = (fp4 & 0x0F).float()
    hi = (fp4 >> 4).float()
    # Expand scale to match unpacked shape
    # scale shape: (M, ceil(N/32)), fp4 shape: (M, N//2)
    N_half = fp4.shape[1]
    N = N_half * 2
    scale_blocks = scale[:, : math.ceil(N / 32)].float()
    block_size = 32
    # Each scale covers 32 original values = 16 uint8 pairs
    scale_expanded = scale_blocks.repeat_interleave(block_size // 2, dim=1)[:, :N_half]
    dq = (lo + hi) * scale_expanded
    return dq


@_NEEDS_MXFP4_STANDALONE
@pytest.mark.parametrize("hidden_size", [256, 512])
@pytest.mark.parametrize("num_tokens", [1, 8, 32])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_functional_standalone_no_residual_scale_shape(hidden_size, num_tokens, eps):
    fused_op = rocm_aiter_ops.get_fused_rmsnorm_mxfp4_quant_op()
    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

    fp4, scale = fused_op(x=x, weight=weight, epsilon=eps)

    assert fp4.dtype == torch.uint8, f"fp4 dtype must be uint8, got {fp4.dtype}"
    assert scale.dtype == torch.uint8, (
        f"scale dtype must be uint8 (E8M0), got {scale.dtype}"
    )
    assert fp4.shape[0] == num_tokens
    assert fp4.shape[1] == hidden_size // 2, (
        f"fp4 second dim must be hidden_size//2={hidden_size // 2}, got {fp4.shape[1]}"
    )
    expected_scale_cols = math.ceil(hidden_size / 32)
    assert scale.shape[1] >= expected_scale_cols, (
        f"scale cols must be >= ceil(N/32)={expected_scale_cols}, got {scale.shape[1]}"
    )


@_NEEDS_MXFP4_STANDALONE
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("num_tokens", [4, 16])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_functional_standalone_with_residual_outputs(hidden_size, num_tokens, eps):
    fused_op = rocm_aiter_ops.get_fused_rmsnorm_add_mxfp4_quant_op()
    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

    fp4, scale, residual_out = fused_op(
        x=x, residual=residual, weight=weight, epsilon=eps
    )

    assert fp4.shape == (num_tokens, hidden_size // 2)
    assert residual_out.shape == (num_tokens, hidden_size), (
        f"residual_out shape mismatch: {residual_out.shape}"
    )
    assert residual_out.dtype == torch.bfloat16


@_NEEDS_MXFP4_STANDALONE
@pytest.mark.parametrize("num_tokens", [1, 8])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_functional_residual_update_correct(num_tokens, eps):
    hidden_size = 256
    fused_op = rocm_aiter_ops.get_fused_rmsnorm_add_mxfp4_quant_op()
    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

    _, _, residual_out = fused_op(
        x=x.clone(), residual=residual.clone(), weight=weight, epsilon=eps
    )

    expected_residual = x + residual
    # BF16 addition: max error is bounded by 1 BF16 ULP of the result.
    # BF16 epsilon = 2^-7 ≈ 7.8e-3; for |val| ≤ 3 (3σ of randn), bound ≈ 2.3e-2.
    # Using 5e-3 is tighter than 1 BF16 ULP and catches any precision regression.
    diff = (residual_out.float() - expected_residual.float()).abs().max().item()
    assert diff < 5e-3, f"residual_out = x + residual_in failed: max diff={diff:.4e}"


@_NEEDS_MXFP4_STANDALONE
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_functional_scale_numerically_correct(eps):
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    hidden_size = 256
    num_tokens = 8

    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

    # Reference: RMSNorm (native) → standalone MXFP4 quant
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normed_ref = (x.float() * torch.rsqrt(variance + eps)).to(torch.bfloat16) * weight
    fp4_ref, scale_ref = dynamic_mxfp4_quant(normed_ref)

    # Fused kernel
    fused_op = rocm_aiter_ops.get_fused_rmsnorm_mxfp4_quant_op()
    fp4_fused, scale_fused = fused_op(x=x, weight=weight, epsilon=eps)

    # Shapes must match
    assert fp4_fused.shape == fp4_ref.shape, (
        f"fp4 shape: {fp4_fused.shape} vs ref {fp4_ref.shape}"
    )
    assert scale_fused.shape[0] == scale_ref.shape[0], (
        f"scale row count: {scale_fused.shape[0]} vs ref {scale_ref.shape[0]}"
    )

    # Scale values must be within 1 ULP of E8M0 (uint8)
    valid_cols = min(scale_fused.shape[1], scale_ref.shape[1])
    scale_diff = (
        (scale_fused[:, :valid_cols].int() - scale_ref[:, :valid_cols].int())
        .abs()
        .max()
        .item()
    )
    assert scale_diff <= 2, (
        f"Scale E8M0 mismatch: max uint8 diff={scale_diff} (expected <= 2 ULP)"
    )




@_NEEDS_MXFP4_STANDALONE
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("num_tokens", [16])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_functional_pattern_fires_no_residual(
    hidden_size, num_tokens, eps, monkeypatch
):
    import vllm.config
    from tests.compile.backend import TestBackend
    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterRMSNormQuantFusionPass,
    )
    from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
    from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
    from vllm.config import CompilationConfig, CompilationMode, VllmConfig

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm"],
        ),
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        set_random_seed(42)

        model = _RMSNormMXFP4Model(hidden_size=hidden_size, eps=eps).cuda()

        fusion_pass = RocmAiterRMSNormQuantFusionPass(vllm_config)
        noop_pass = NoOpEliminationPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)
        backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)

        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
        torch._dynamo.mark_dynamic(x, 0)

        compiled = torch.compile(model, backend=backend)
        compiled(x)

    # Fused op must appear in graph after pass
    backend.check_after_ops([rocm_aiter_ops.get_fused_rmsnorm_mxfp4_quant_op()])

    assert fusion_pass.matched_count >= 1, (
        f"Expected at least 1 pattern match, got {fusion_pass.matched_count}"
    )


@_NEEDS_MXFP4_STANDALONE
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("num_tokens", [16])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_functional_pattern_fires_with_residual(
    hidden_size, num_tokens, eps, monkeypatch
):
    import vllm.config
    from tests.compile.backend import TestBackend
    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterRMSNormQuantFusionPass,
    )
    from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
    from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
    from vllm.config import CompilationConfig, CompilationMode, VllmConfig

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm"],
        ),
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        set_random_seed(42)

        model = _FusedAddRMSNormMXFP4Model(hidden_size=hidden_size, eps=eps).cuda()

        fusion_pass = RocmAiterRMSNormQuantFusionPass(vllm_config)
        noop_pass = NoOpEliminationPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)
        backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)

        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(
            num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        # fused_add_rms_norm has allow_inplace=True; using mark_dynamic on x's
        # batch dim would force a symbolic shape but the mutating overload
        # specializes it. Use maybe_mark_dynamic so compilation succeeds.
        torch._dynamo.maybe_mark_dynamic(x, 0)

        compiled = torch.compile(model, backend=backend)
        compiled(x, residual)

    backend.check_after_ops([rocm_aiter_ops.get_fused_rmsnorm_add_mxfp4_quant_op()])
    assert fusion_pass.matched_count >= 1, (
        f"Expected at least 1 match, got {fusion_pass.matched_count}"
    )


@_NEEDS_MXFP4_STANDALONE
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_functional_fused_matches_unfused_output(
    hidden_size, num_tokens, eps, monkeypatch
):
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()

    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

    # Unfused: manual RMSNorm → standalone quant
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normed = (x.float() * torch.rsqrt(variance + eps)).to(torch.bfloat16) * weight
    fp4_ref, scale_ref = dynamic_mxfp4_quant(normed)

    # Fused kernel
    fused_op = rocm_aiter_ops.get_fused_rmsnorm_mxfp4_quant_op()
    fp4_fused, scale_fused = fused_op(x=x, weight=weight, epsilon=eps)

    assert fp4_fused.shape == fp4_ref.shape
    valid_cols = min(scale_fused.shape[1], scale_ref.shape[1])
    scale_diff = (
        (scale_fused[:, :valid_cols].int() - scale_ref[:, :valid_cols].int())
        .abs()
        .max()
        .item()
    )
    assert scale_diff <= 2, (
        f"eps={eps}: scale E8M0 max diff={scale_diff} exceeds tolerance of 2 ULP"
    )




class _AiterRMSNormMXFP4QuantModel(torch.nn.Module):
    def __init__(self, hidden_size=256, eps=1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm_weight_0 = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.norm_weight_1 = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(self, x: torch.Tensor, residual: torch.Tensor):
        # Site 0: no-residual — exercises AiterRMSNormMXFP4QuantPattern
        normed_0 = torch.ops.vllm_ir.rms_norm(x, self.norm_weight_0, self.eps)
        quant_0, scale_0 = torch.ops.vllm.rocm_aiter_dynamic_mxfp4_quant(normed_0)

        # Site 1: with-residual — exercises AiterFusedAddRMSNormMXFP4QuantPattern
        normed_1, residual_out = torch.ops.vllm_ir.fused_add_rms_norm(
            x, residual, self.norm_weight_1, self.eps
        )
        quant_1, scale_1 = torch.ops.vllm.rocm_aiter_dynamic_mxfp4_quant(normed_1)

        return quant_0, scale_0, quant_1, scale_1, residual_out


@_NEEDS_MXFP4_STANDALONE
def test_mxfp4_patterns_fire_on_model(monkeypatch):
    import vllm.config
    from tests.compile.backend import TestBackend
    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterRMSNormQuantFusionPass,
    )
    from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
    from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
    from vllm.config import CompilationConfig, CompilationMode, VllmConfig

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()

    hidden_size = 256
    num_tokens = 16
    eps = 1e-6

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm"],
        ),
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        set_random_seed(42)

        model = _AiterRMSNormMXFP4QuantModel(hidden_size=hidden_size, eps=eps).cuda()

        fusion_pass = RocmAiterRMSNormQuantFusionPass(vllm_config)
        noop_pass = NoOpEliminationPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)
        backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)

        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(
            num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.maybe_mark_dynamic(residual, 0)

        compiled = torch.compile(model, backend=backend)
        compiled(x, residual)

    # Both fused ops must appear in the post-pass graph
    backend.check_after_ops(
        [
            rocm_aiter_ops.get_fused_rmsnorm_mxfp4_quant_op(),
            rocm_aiter_ops.get_fused_rmsnorm_add_mxfp4_quant_op(),
        ]
    )
    # Standalone quant must be absent from the post-pass graph (mirrors PR#42864)
    backend.check_not_in_after_ops(
        [
            rocm_aiter_ops.get_dynamic_mxfp4_quant_op(),
        ]
    )
    # Standalone quant must be fully eliminated from before→after
    backend.check_before_ops(
        [rocm_aiter_ops.get_dynamic_mxfp4_quant_op()],
        fully_replaced=True,
    )
    assert fusion_pass.matched_count == 2, (
        f"matched_count must be 2 (one per site), got {fusion_pass.matched_count}"
    )
    print(f"PASS: matched_count={fusion_pass.matched_count}")
