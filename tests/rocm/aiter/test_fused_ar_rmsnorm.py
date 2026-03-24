# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused allreduce+RMSNorm feature (MI355X).

Tests cover:
- Custom op registration and fake implementation
- Auto-detection gating (proves non-regression for non-MI355X / non-AITER)
- Graph-level decomposition of fused_allreduce_rmsnorm when followed by FP8 quant
- Preservation of fused_allreduce_rmsnorm when NOT followed by FP8 quant (FP4 path)
- Full-pass pipeline: decompose → pattern-match → fuse (FP8) vs preserve (FP4)
"""

import operator
from unittest.mock import patch

import pytest
import torch
from torch import fx

from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not IS_AITER_FOUND,
    reason="ROCm with AITER required",
)


# ---------------------------------------------------------------------------
# Custom-op registration / fake-impl tests
# ---------------------------------------------------------------------------


def test_fused_allreduce_rmsnorm_op_registered():
    """Verify the custom op is registered and callable."""
    assert hasattr(torch.ops.vllm, "fused_allreduce_rmsnorm")
    op = torch.ops.vllm.fused_allreduce_rmsnorm.default
    assert op is not None


def test_fused_allreduce_rmsnorm_fake_shapes():
    """Verify the fake implementation returns tensors of correct shape."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    hidden = 128
    tokens = 32

    with FakeTensorMode():
        input_ = torch.randn(tokens, hidden, device="cuda")
        residual = torch.randn(tokens, hidden, device="cuda")
        weight = torch.randn(hidden, device="cuda")

        out, resid_out = torch.ops.vllm.fused_allreduce_rmsnorm(
            input_, residual, weight, 1e-5, "fake_group"
        )

        assert out.shape == (tokens, hidden)
        assert resid_out.shape == (tokens, hidden)


# ---------------------------------------------------------------------------
# Decomposition-pass graph-level tests
# ---------------------------------------------------------------------------


def _build_fused_ar_rmsnorm_graph(
    hidden_size: int,
    eps: float,
    fp8_quant_op,
    add_fp8_consumer: bool = True,
) -> tuple[fx.Graph, fx.Node]:
    """Build a minimal FX graph with fused_allreduce_rmsnorm.

    Returns (graph, fused_ar_rms_node).
    If add_fp8_consumer is True, the normed output (getitem 0) feeds
    into an FP8 quant op.
    """
    graph = fx.Graph()

    input_ = graph.placeholder("input_")
    residual = graph.placeholder("residual")
    weight = graph.placeholder("weight")

    fused_node = graph.call_function(
        torch.ops.vllm.fused_allreduce_rmsnorm.default,
        args=(input_, residual, weight, eps, "test_group"),
    )

    normed = graph.call_function(operator.getitem, args=(fused_node, 0))
    resid_out = graph.call_function(operator.getitem, args=(fused_node, 1))

    fake_input = torch.randn(4, hidden_size)
    fake_residual = torch.randn(4, hidden_size)
    input_.meta["val"] = fake_input
    residual.meta["val"] = fake_residual
    weight.meta["val"] = torch.randn(hidden_size)
    fused_node.meta["val"] = (fake_residual.clone(), fake_residual.clone())
    normed.meta["val"] = fake_residual.clone()
    resid_out.meta["val"] = fake_residual.clone()

    if add_fp8_consumer:
        quant_node = graph.call_function(fp8_quant_op, args=(normed,))
        quant_out = graph.call_function(operator.getitem, args=(quant_node, 0))
        quant_scale = graph.call_function(
            operator.getitem, args=(quant_node, 1)
        )

        quant_node.meta["val"] = (
            fake_residual.to(torch.float8_e4m3fnuz),
            torch.randn(4, 1),
        )
        quant_out.meta["val"] = fake_residual.to(torch.float8_e4m3fnuz)
        quant_scale.meta["val"] = torch.randn(4, 1)

        graph.output((quant_out, resid_out, quant_scale))
    else:
        graph.output((normed, resid_out))

    return graph, fused_node


def test_decompose_fused_ar_rmsnorm_with_fp8(monkeypatch: pytest.MonkeyPatch):
    """When fused_allreduce_rmsnorm feeds into FP8 quant, it should be
    decomposed into all_reduce + rmsnorm_with_add."""
    import vllm.config

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()

    try:
        fp8_quant_op = rocm_aiter_ops.get_per_token_quant_op()
    except Exception:
        pytest.skip("AITER per_token_quant op not available")

    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterRMSNormQuantFusionPass,
    )
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        ModelConfig,
        PassConfig,
        VllmConfig,
    )

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+quant_fp8"],
            pass_config=PassConfig(
                fuse_norm_quant=True, eliminate_noops=True
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = RocmAiterRMSNormQuantFusionPass(vllm_config)

    graph, _ = _build_fused_ar_rmsnorm_graph(
        hidden_size=256,
        eps=1e-5,
        fp8_quant_op=fp8_quant_op,
        add_fp8_consumer=True,
    )

    fused_ar_op = torch.ops.vllm.fused_allreduce_rmsnorm.default
    all_reduce_op = torch.ops.vllm.all_reduce.default
    rmsnorm_add_op = rocm_aiter_ops.get_rmsnorm_fused_add_op()

    fused_before = sum(
        1 for n in graph.nodes if n.target == fused_ar_op
    )
    assert fused_before == 1, "Expected 1 fused_allreduce_rmsnorm node"

    count = fusion_pass._decompose_fused_allreduce_rmsnorm(graph)

    assert count == 1, f"Expected 1 decomposition, got {count}"

    fused_after = sum(
        1 for n in graph.nodes if n.target == fused_ar_op
    )
    assert fused_after == 0, "fused_allreduce_rmsnorm should be removed"

    ar_nodes = sum(
        1 for n in graph.nodes if n.target == all_reduce_op
    )
    assert ar_nodes == 1, f"Expected 1 all_reduce node, got {ar_nodes}"

    rms_nodes = sum(
        1 for n in graph.nodes if n.target == rmsnorm_add_op
    )
    assert rms_nodes == 1, f"Expected 1 rmsnorm_with_add node, got {rms_nodes}"


def test_preserve_fused_ar_rmsnorm_without_fp8(
    monkeypatch: pytest.MonkeyPatch,
):
    """When fused_allreduce_rmsnorm does NOT feed into FP8 quant (e.g. FP4),
    the node should be preserved."""
    import vllm.config

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()

    try:
        fp8_quant_op = rocm_aiter_ops.get_per_token_quant_op()
    except Exception:
        pytest.skip("AITER per_token_quant op not available")

    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterRMSNormQuantFusionPass,
    )
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        ModelConfig,
        PassConfig,
        VllmConfig,
    )

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+quant_fp8"],
            pass_config=PassConfig(
                fuse_norm_quant=True, eliminate_noops=True
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = RocmAiterRMSNormQuantFusionPass(vllm_config)

    graph, _ = _build_fused_ar_rmsnorm_graph(
        hidden_size=256,
        eps=1e-5,
        fp8_quant_op=fp8_quant_op,
        add_fp8_consumer=False,
    )

    fused_ar_op = torch.ops.vllm.fused_allreduce_rmsnorm.default

    fused_before = sum(
        1 for n in graph.nodes if n.target == fused_ar_op
    )
    assert fused_before == 1

    count = fusion_pass._decompose_fused_allreduce_rmsnorm(graph)

    assert count == 0, "Should not decompose when no FP8 consumer"

    fused_after = sum(
        1 for n in graph.nodes if n.target == fused_ar_op
    )
    assert fused_after == 1, "fused_allreduce_rmsnorm should be preserved"


def test_decompose_no_op_without_fused_nodes(
    monkeypatch: pytest.MonkeyPatch,
):
    """Decomposition should be a no-op when graph has no fused_allreduce_rmsnorm."""
    import vllm.config

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()

    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterRMSNormQuantFusionPass,
    )
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        ModelConfig,
        PassConfig,
        VllmConfig,
    )

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+quant_fp8"],
            pass_config=PassConfig(
                fuse_norm_quant=True, eliminate_noops=True
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = RocmAiterRMSNormQuantFusionPass(vllm_config)

    graph = fx.Graph()
    x = graph.placeholder("x")
    y = graph.call_function(torch.relu, args=(x,))
    graph.output(y)

    count = fusion_pass._decompose_fused_allreduce_rmsnorm(graph)
    assert count == 0


def test_is_fused_allreduce_rmsnorm_supported():
    """Verify auto-detection method exists and returns a bool."""
    result = rocm_aiter_ops.is_fused_allreduce_rmsnorm_supported()
    assert isinstance(result, (bool, type(None)))


# ---------------------------------------------------------------------------
# Auto-detection gating tests — proves non-regression for other platforms
# ---------------------------------------------------------------------------


def test_auto_detection_disabled_without_gfx950():
    """Feature must be disabled when not on gfx950 (MI355X).

    This proves non-MI355X ROCm GPUs are unaffected by the optimization.
    """
    with patch("vllm._aiter_ops.rocm_aiter_ops._AITER_ENABLED", True), \
         patch("vllm._aiter_ops.rocm_aiter_ops._RMSNORM_ENABLED", True), \
         patch("vllm.platforms.rocm.on_gfx950", return_value=False):
        result = rocm_aiter_ops.is_fused_allreduce_rmsnorm_supported()
    assert not result, (
        "Fused AR+RMSNorm should be disabled when not on gfx950"
    )


def test_auto_detection_disabled_without_aiter():
    """Feature must be disabled when AITER kernels are not available.

    This proves non-AITER environments (e.g. CUDA, older ROCm) are
    unaffected.
    """
    saved = rocm_aiter_ops._AITER_ENABLED
    try:
        rocm_aiter_ops._AITER_ENABLED = False
        result = rocm_aiter_ops.is_fused_allreduce_rmsnorm_supported()
        assert not result, (
            "Fused AR+RMSNorm should be disabled when AITER is not enabled"
        )
    finally:
        rocm_aiter_ops._AITER_ENABLED = saved


def test_auto_detection_disabled_without_rmsnorm():
    """Feature must be disabled when AITER RMSNorm is not available."""
    saved = rocm_aiter_ops._RMSNORM_ENABLED
    try:
        rocm_aiter_ops._RMSNORM_ENABLED = False
        result = rocm_aiter_ops.is_fused_allreduce_rmsnorm_supported()
        assert not result, (
            "Fused AR+RMSNorm should be disabled when RMSNorm is not enabled"
        )
    finally:
        rocm_aiter_ops._RMSNORM_ENABLED = saved


# ---------------------------------------------------------------------------
# Full-pass pipeline tests (decompose + pattern-match fusion)
# ---------------------------------------------------------------------------


def _make_fusion_pass(monkeypatch):
    """Create a RocmAiterRMSNormQuantFusionPass with standard config."""
    import vllm.config

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()

    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterRMSNormQuantFusionPass,
    )
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        ModelConfig,
        PassConfig,
        VllmConfig,
    )

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+quant_fp8"],
            pass_config=PassConfig(
                fuse_norm_quant=True, eliminate_noops=True
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config):
        return RocmAiterRMSNormQuantFusionPass(vllm_config)


def _wrap_graph_in_module(graph: fx.Graph) -> fx.GraphModule:
    """Wrap a bare fx.Graph in a GraphModule.

    The Inductor PatternMatcherPass requires graph.owning_module to be
    a GraphModule. Standalone graphs don't have one, so we wrap them.
    """
    return fx.GraphModule(torch.nn.Module(), graph)


def test_full_pass_fp8_decompose_and_fuse(monkeypatch: pytest.MonkeyPatch):
    """Full pass on FP8 graph: verify the complete pass runs and decomposes
    fused_allreduce_rmsnorm when FP8 quant consumers are present.

    Decomposition is verified here at graph level. The subsequent
    rmsnorm+quant pattern-match fusion requires a properly traced graph
    (via torch.compile) and is covered by the multi-GPU integration test
    test_rocm_fused_ar_rmsnorm_fp8_decomposed.
    """
    try:
        fp8_quant_op = rocm_aiter_ops.get_per_token_quant_op()
    except Exception:
        pytest.skip("AITER per_token_quant op not available")

    fusion_pass = _make_fusion_pass(monkeypatch)

    graph, _ = _build_fused_ar_rmsnorm_graph(
        hidden_size=256,
        eps=1e-5,
        fp8_quant_op=fp8_quant_op,
        add_fp8_consumer=True,
    )
    gm = _wrap_graph_in_module(graph)
    graph = gm.graph

    fused_ar_op = torch.ops.vllm.fused_allreduce_rmsnorm.default
    all_reduce_op = torch.ops.vllm.all_reduce.default
    rmsnorm_add_op = rocm_aiter_ops.get_rmsnorm_fused_add_op()

    assert sum(1 for n in graph.nodes if n.target == fused_ar_op) == 1
    assert sum(1 for n in graph.nodes if n.target == all_reduce_op) == 0

    fusion_pass(graph)

    assert sum(1 for n in graph.nodes if n.target == fused_ar_op) == 0, (
        "fused_allreduce_rmsnorm should be decomposed in FP8 path"
    )
    assert sum(1 for n in graph.nodes if n.target == all_reduce_op) == 1, (
        "all_reduce should be present after decomposition"
    )
    assert sum(1 for n in graph.nodes if n.target == rmsnorm_add_op) >= 1, (
        "rmsnorm_with_add should be present after decomposition"
    )
    assert sum(1 for n in graph.nodes if n.target == fp8_quant_op) >= 1, (
        "fp8_quant should still be in graph (pattern matching requires "
        "torch.compile traced graphs for full fusion)"
    )


def test_full_pass_fp4_preserves_fused(monkeypatch: pytest.MonkeyPatch):
    """Full pass on FP4/BF16 graph: fused_allreduce_rmsnorm has no FP8
    consumer, so it should survive the entire pass untouched.

    This verifies that FP4 models keep the AITER fused AR+RMSNorm kernel
    and the pass does not alter the graph.
    """
    try:
        fp8_quant_op = rocm_aiter_ops.get_per_token_quant_op()
    except Exception:
        pytest.skip("AITER per_token_quant op not available")

    fusion_pass = _make_fusion_pass(monkeypatch)

    graph, _ = _build_fused_ar_rmsnorm_graph(
        hidden_size=256,
        eps=1e-5,
        fp8_quant_op=fp8_quant_op,
        add_fp8_consumer=False,
    )
    gm = _wrap_graph_in_module(graph)
    graph = gm.graph

    fused_ar_op = torch.ops.vllm.fused_allreduce_rmsnorm.default

    assert sum(1 for n in graph.nodes if n.target == fused_ar_op) == 1

    fusion_pass(graph)

    assert sum(1 for n in graph.nodes if n.target == fused_ar_op) == 1, (
        "fused_allreduce_rmsnorm must be preserved for FP4/BF16 models"
    )
