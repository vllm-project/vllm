# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for IR inplace functionalization pass integration.

This test suite verifies that the inplace functionalization pass, lowering pass,
and clone cleanup pass work together correctly with donated buffer tracking.
"""

from collections.abc import Callable

import pytest
import torch
import torch._dynamo.exc
from torch import nn

import vllm.kernels  # noqa: F401 to register kernels
from vllm.compilation.passes.inductor_pass import InductorPass, get_pass_context
from vllm.compilation.passes.ir.clone_elimination import (
    UnsafeCloneEliminationPass,
)
from vllm.compilation.passes.ir.inplace_functionalization import (
    VllmIRInplaceFunctionalizationPass,
)
from vllm.compilation.passes.ir.lowering_pass import VllmIRLoweringPass
from vllm.config import VllmConfig
from vllm.ir import ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

from ...backend import TestBackend


class StoreDonationInfoPass(InductorPass):
    def __init__(self):
        self.donated_input_ids_sets: list[set[int]] = []

    def __call__(self, *args, **kwargs):
        ctx = get_pass_context()
        self.donated_input_ids_sets += [ctx.donated_input_ids]


class MaybeInplaceModel(nn.Module):
    """Model using only maybe_inplace variants."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.weight1 = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.weight2 = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(
        self, x: torch.Tensor, residual1: torch.Tensor, residual2: torch.Tensor
    ):
        # First maybe_inplace - x & residual1 are donated
        x_normed1, residual_out1 = ops.fused_add_rms_norm.maybe_inplace(
            x, residual1, self.weight1, 1e-5
        )
        # Second maybe_inplace - residual2 is donated
        x_normed2, residual_out2 = ops.fused_add_rms_norm.maybe_inplace(
            x_normed1, residual2, self.weight2, 1e-5
        )
        return x_normed2, residual_out1, residual_out2


class FunctionalModel(nn.Module):
    """Model using only functional (default) variants."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.weight1 = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.weight2 = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(
        self, x: torch.Tensor, residual1: torch.Tensor, residual2: torch.Tensor
    ):
        # First functional - no donation
        x_normed1, residual_out1 = ops.fused_add_rms_norm(
            x, residual1, self.weight1, 1e-5
        )
        # Second functional - no donation
        x_normed2, residual_out2 = ops.fused_add_rms_norm(
            x_normed1, residual2, self.weight2, 1e-5
        )
        return x_normed2, residual_out1, residual_out2


class MixedModel(nn.Module):
    """Model mixing maybe_inplace and functional variants."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.weight1 = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.weight2 = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(
        self, x: torch.Tensor, residual1: torch.Tensor, residual2: torch.Tensor
    ):
        # First maybe_inplace - x & residual1 are donated
        x_normed1, residual_out1 = ops.fused_add_rms_norm.maybe_inplace(
            x, residual1, self.weight1, 1e-5
        )
        # Second functional - no donation, x_normed1 must be preserved as it's returned
        x_normed2, residual_out2 = ops.fused_add_rms_norm(
            x_normed1, residual2, self.weight2, 1e-5
        )
        # Return both to prevent x_normed1 from being optimized away
        return x_normed1, x_normed2, residual_out1, residual_out2


class ModelWithTritonAfterMaybeInplace(nn.Module):
    """
    Model using maybe_inplace followed by a Triton kernel.
    Test clone elimination can handle Triton in the graph
    """

    def __init__(self, hidden_size=16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

        @triton.jit
        def _triton_add_kernel(
            x_ptr,
            y_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = x + 0.1
            tl.store(y_ptr + offsets, y, mask=mask)

        def triton_add(x: torch.Tensor) -> torch.Tensor:
            """Simple Triton add kernel."""
            y = torch.empty_like(x)
            n_elements = x.numel()
            grid = (triton.cdiv(n_elements, 256),)
            _triton_add_kernel[grid](x, y, n_elements, BLOCK_SIZE=256)
            return y

        self.triton_add = triton_add

    def forward(self, x: torch.Tensor, residual: torch.Tensor, residual2: torch.Tensor):
        x_normed, residual_out = ops.fused_add_rms_norm.maybe_inplace(
            x, residual, self.weight, 1e-5
        )

        x_processed = self.triton_add(x_normed)

        # x_processed does not need to be cloned, residual2 does
        x_normed2, residual_out2 = ops.fused_add_rms_norm(
            x_processed, residual2, self.weight, 1e-5
        )
        return x_normed2, residual_out2


skipif_no_triton = pytest.mark.skipif(not HAS_TRITON, reason="Requires Triton")


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Only test on cuda and rocm platform",
)
@pytest.mark.parametrize(
    "model_class,expected_functionalized,expected_donated,expected_clones",
    [
        # 2 inplace calls, all activations donated, all clones eliminated
        (MaybeInplaceModel, 2, 3, 0),
        # No inplace calls, no donations, 3 clones (one eliminated)
        (FunctionalModel, 0, 0, 3),
        # One inplace call, two donated activations, 2 clones
        (MixedModel, 1, 2, 2),
        # One inplace call, two donated, 1 clone remaining
        pytest.param(ModelWithTritonAfterMaybeInplace, 1, 2, 1, marks=skipif_no_triton),
    ],
)
def test_inplace_functionalization(
    default_vllm_config: VllmConfig,
    model_class,
    expected_functionalized: int,
    expected_clones: int,
    expected_donated: int,
):
    """Test inplace functionalization, lowering, and clone cleanup."""
    torch.set_default_device(current_platform.device_type)

    # Use vllm_c so inplace path is triggered
    default_vllm_config.kernel_config.ir_op_priority.fused_add_rms_norm = [
        "vllm_c",
        "native",
    ]

    # Create passes in order they run during compilation
    functionalization_pass = VllmIRInplaceFunctionalizationPass(default_vllm_config)
    lowering_pass = VllmIRLoweringPass(default_vllm_config)
    donated_info_pass = StoreDonationInfoPass()
    cleanup_pass = UnsafeCloneEliminationPass(default_vllm_config)

    # Set up backend with pre-grad pass
    backend = TestBackend(lowering_pass, donated_info_pass, cleanup_pass)
    backend.inductor_config["pre_grad_custom_pass"] = functionalization_pass

    model = model_class()
    x = torch.randn(8, 16, dtype=torch.bfloat16)
    residual1 = torch.randn(8, 16, dtype=torch.bfloat16)
    residual2 = torch.randn(8, 16, dtype=torch.bfloat16)

    with default_vllm_config.kernel_config.ir_op_priority.set_priority():
        # Reference output without optimization
        ref_output = model(x.clone(), residual1.clone(), residual2.clone())

        # Compile with inplace optimization
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        output = compiled_model(x.clone(), residual1.clone(), residual2.clone())

    # Verify correctness (relaxed tolerance for bfloat16)
    for i in range(len(ref_output)):
        torch.testing.assert_close(output[i], ref_output[i], rtol=1e-2, atol=1e-2)

    # Verify expected number of ops were functionalized
    func_ops = functionalization_pass.functionalized_ops
    assert len(func_ops) == int(bool(expected_functionalized))
    if expected_functionalized > 0:
        assert "fused_add_rms_norm" in func_ops
        assert func_ops["fused_add_rms_norm"] == expected_functionalized

    # Verify lowering happened (2 ops in all cases)
    assert "fused_add_rms_norm" in lowering_pass.selected_impls
    assert len(lowering_pass.selected_impls["fused_add_rms_norm"]) == 2
    assert all(
        provider == "vllm_c"
        for node, provider in lowering_pass.selected_impls["fused_add_rms_norm"].items()
    ), lowering_pass.selected_impls

    # Verify correct number of donated IDs
    assert len(donated_info_pass.donated_input_ids_sets) == 1
    assert len(donated_info_pass.donated_input_ids_sets[0]) == expected_donated

    # Verify expected number of clones after cleanup
    actual_clones = backend.op_count(torch.ops.aten.clone.default, before=False)
    assert actual_clones == expected_clones, (
        f"Expected {expected_clones} clones, got {actual_clones}:"
        f"{backend.print_graphs()}"
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Only test on cuda and rocm platform",
)
def test_donated_buffer_context_propagation(default_vllm_config):
    """Test that donated_input_ids propagates correctly through pass_context."""
    torch.set_default_device(current_platform.device_type)

    # Create a custom backend that inspects pass_context in cleanup pass
    functionalization_pass = VllmIRInplaceFunctionalizationPass(default_vllm_config)
    lowering_pass = VllmIRLoweringPass(default_vllm_config)

    donation_info_pass = StoreDonationInfoPass()
    cleanup_pass = UnsafeCloneEliminationPass(default_vllm_config)

    backend = TestBackend(lowering_pass, donation_info_pass, cleanup_pass)
    backend.inductor_config["pre_grad_custom_pass"] = functionalization_pass

    model = MaybeInplaceModel()
    x = torch.randn(8, 16, dtype=torch.bfloat16)
    residual1 = torch.randn(8, 16, dtype=torch.bfloat16)
    residual2 = torch.randn(8, 16, dtype=torch.bfloat16)

    compiled_model = torch.compile(model, backend=backend, fullgraph=True)
    compiled_model(x.clone(), residual1.clone(), residual2.clone())

    donated_ids_seen = donation_info_pass.donated_input_ids_sets
    # Verify donated_input_ids was set and propagated
    assert len(donated_ids_seen) == 1
    # Should have donated inputs (exact indices depend on AOTAutograd)
    assert len(donated_ids_seen[0]) == 3
    # All donated ids should be valid non-negative integers
    for idx in donated_ids_seen[0]:
        assert isinstance(idx, int) and idx >= 0, f"Invalid donated index: {idx}"


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Only test on cuda and rocm platform",
)
def test_maybe_inplace_reuse_error(default_vllm_config):
    """Test that reusing a donated activation input raises ValueError."""
    torch.set_default_device(current_platform.device_type)

    class ReuseModel(nn.Module):
        """Model that incorrectly reuses a donated activation input."""

        def __init__(self, hidden_size=16):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

        def forward(self, x: torch.Tensor, residual: torch.Tensor):
            # x is donated to maybe_inplace
            x_normed, residual_out = ops.fused_add_rms_norm.maybe_inplace(
                x, residual, self.weight, 1e-5
            )
            # ERROR: x is used again after being donated
            return x_normed + x  # This should raise ValueError

    functionalization_pass = VllmIRInplaceFunctionalizationPass(default_vllm_config)
    lowering_pass = VllmIRLoweringPass(default_vllm_config)
    cleanup_pass = UnsafeCloneEliminationPass(default_vllm_config)

    backend = TestBackend(lowering_pass, cleanup_pass)
    backend.inductor_config["pre_grad_custom_pass"] = functionalization_pass

    model = ReuseModel()
    x = torch.randn(8, 16, dtype=torch.bfloat16)
    residual = torch.randn(8, 16, dtype=torch.bfloat16)

    # Compilation should raise BackendCompilerFailed wrapping ValueError
    with pytest.raises(
        torch._dynamo.exc.BackendCompilerFailed,
        match="is used again after the node",
    ):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        compiled_model(x.clone(), residual.clone())


# Piecewise compilation tests with graph splitting


@torch.library.custom_op("vllm::test_split_marker", mutates_args=())
def test_split_marker(x: torch.Tensor) -> torch.Tensor:
    """Identity op that marks a split point for piecewise compilation."""
    return x.clone()


@test_split_marker.register_fake
def _fake_split_marker(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class TransformerBlockWithSplits(nn.Module):
    """Transformer block with explicit split points for piecewise compilation."""

    def __init__(self, hidden_size=32, intermediate_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Attention-like projection
        self.attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=False, dtype=torch.bfloat16
        )

        # Post-attention norm
        self.post_attn_norm = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16)
        )

        # MLP
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=torch.bfloat16
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=torch.bfloat16
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, dtype=torch.bfloat16
        )

        # Post-MLP norm
        self.post_mlp_norm = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor):
        # Attention block with residual
        residual1 = x
        attn_out = self.attn_proj(x)

        # Fused add + norm (maybe_inplace: residual1 is donated)
        normed1, residual1 = ops.fused_add_rms_norm.maybe_inplace(
            attn_out, residual1, self.post_attn_norm, 1e-5
        )

        # Force a graph split here
        normed1 = torch.ops.vllm.test_split_marker(normed1)

        # MLP block
        gate = self.gate_proj(normed1)
        up = self.up_proj(normed1)
        mlp_out = self.down_proj(gate * torch.nn.functional.silu(up))

        # Fused add + norm (maybe_inplace: residual1 is donated)
        normed2, residual2 = ops.fused_add_rms_norm.maybe_inplace(
            mlp_out, residual1, self.post_mlp_norm, 1e-5
        )

        return normed2, residual2


def with_dyn_arg(fn: Callable, arg_index: int, dim_index: int):
    def inner(*args):
        torch._dynamo.mark_dynamic(args[arg_index], dim_index)
        return fn(*args)

    return inner


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Only test on cuda and rocm platform",
)
def test_piecewise_compilation_with_donated_buffers(monkeypatch, fresh_vllm_cache):
    """
    Test piecewise compilation with donated buffers across graph splits.
    Utilizes a custom splitting op. Uses fresh cache to avoid compilation caching.
    """
    torch.set_default_device(current_platform.device_type)

    # Disable compilation cache to avoid serialization issues
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    from vllm.compilation.backends import VllmBackend
    from vllm.config import CompilationConfig, VllmConfig

    # Create config with custom splitting op
    store_donation_info = StoreDonationInfoPass()
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            custom_ops=["all"],
            splitting_ops=["vllm::test_split_marker"],
            inductor_compile_config={"post_grad_custom_post_pass": store_donation_info},
        )
    )

    backend = VllmBackend(vllm_config)

    model = TransformerBlockWithSplits()
    x = torch.randn(8, 32, dtype=torch.bfloat16)

    # Reference output
    ref_output = with_dyn_arg(model, 0, 0)(x.clone())

    # Compile with piecewise compilation (graph will split at split_marker)
    compiled_model = torch.compile(model, backend=backend, fullgraph=False)
    output = with_dyn_arg(compiled_model, 0, 0)(x.clone())

    # Verify correctness (relaxed tolerance for bfloat16)
    torch.testing.assert_close(output[0], ref_output[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(output[1], ref_output[1], rtol=1e-2, atol=1e-2)

    # Verify the model was split into multiple submodules
    assert hasattr(backend, "split_gm"), "Backend should have split graph module"

    # Should have at least 2 submodules (split by test_split_marker op)
    submodules = list(backend.split_gm.named_children())
    num_submodules = len(submodules)
    assert num_submodules >= 2, (
        f"Expected at least 2 submodules (split), got {num_submodules}"
    )

    # Check that donation info was propagated correctly
    donated_inputs_sets = store_donation_info.donated_input_ids_sets
    assert len(donated_inputs_sets) == 2
    assert len(donated_inputs_sets[0]) == 1
    assert len(donated_inputs_sets[1]) == 1
