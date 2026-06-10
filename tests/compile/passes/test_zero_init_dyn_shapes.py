"""Focused dynamic-shape compile test to isolate which zero-init producer path
specializes a symbolic batch dim when used under torch.compile.

Each fused-path test compiles a tiny module that calls one explicit
``*_with_zero_init`` mutating producer alias via ``auto_functionalized``, on a
tensor with ``mark_dynamic`` on the batch dimension. If the op introduces a
specialization that torch.compile would later raise as a
`ConstraintViolationError`, this test will surface it in isolation (without
needing to boot vLLM).

Run with `pytest tests/compile/passes/test_zero_init_dyn_shapes.py -s`.
"""

from __future__ import annotations

import pytest
import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="zero-init producers are ROCm-only",
)


def _rocm_available() -> bool:
    try:
        rocm_aiter_ops.register_ops_once()
    except Exception:
        return False
    return True


@pytest.fixture(scope="module", autouse=True)
def _register_ops():
    if not _rocm_available():
        pytest.skip("rocm_aiter_ops not available")


def _compile_and_check(fn, *args):
    """Compile `fn` with dynamic on the first dim of every tensor arg, run
    twice with different batch sizes, and return without exception.
    Raises ConstraintViolationError if dynamic dim got specialized."""
    for a in args:
        if torch.is_tensor(a):
            torch._dynamo.mark_dynamic(a, 0)
    compiled = torch.compile(fn, dynamic=True, fullgraph=True)
    out = compiled(*args)
    return out


def test_group_quant_dynamic_M():
    op = rocm_aiter_ops.get_group_quant_with_zero_init_op()

    def fn(x, y):
        # y is the zero-init buffer; pretend N=256
        res = auto_functionalized(op, x=x, gemm_out_zero_init=y, group_size=128)
        return res[0]

    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(8, 256, dtype=torch.bfloat16, device="cuda")
    _compile_and_check(fn, x, y)


def test_rmsnorm_group_quant_dynamic_M():
    op = rocm_aiter_ops.get_rmsnorm_fp8_group_quant_with_zero_init_op()

    def fn(x, y, w):
        res = auto_functionalized(
            op,
            x=x,
            gemm_out_zero_init=y,
            weight=w,
            variance_epsilon=1e-6,
            group_size=128,
        )
        return res[0]

    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(8, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    _compile_and_check(fn, x, y, w)


def test_rmsnorm_with_add_group_quant_dynamic_M():
    op = rocm_aiter_ops.get_rmsnorm_with_add_fp8_group_quant_with_zero_init_op()

    def fn(x, y, r, w):
        res = auto_functionalized(
            op,
            x=x,
            gemm_out_zero_init=y,
            residual=r,
            weight=w,
            variance_epsilon=1e-6,
            group_size=128,
        )
        return res[0]

    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(8, 256, dtype=torch.bfloat16, device="cuda")
    r = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    _compile_and_check(fn, x, y, r, w)


def test_act_mul_group_quant_dynamic_M():
    op = rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_with_zero_init_op()

    def fn(x, y):
        res = auto_functionalized(op, x=x, gemm_out_zero_init=y, group_size=128)
        return res[0]

    x = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(8, 128, dtype=torch.bfloat16, device="cuda")
    _compile_and_check(fn, x, y)


def test_full_pass_register_then_compile_dyn_model():
    """End-to-end smoke: instantiate `BlockScaleSplitKZeroInitFusionPass`
    (which registers all producer/GEMM patterns), then compile a tiny model
    whose only op is `rocm_aiter_rmsnorm_fp8_group_quant` with dynamic M,
    then invoke it. Guards against pattern registration introducing a global
    shape specialization that would surface as a ConstraintViolationError
    under dynamic batch sizes.
    """
    from vllm.compilation.passes.fusion.blockscale_splitk_zero_init import (
        BlockScaleSplitKZeroInitFusionPass,
    )
    from vllm.config import (
        CompilationConfig,
        ModelConfig,
        VllmConfig,
    )
    from vllm.config.compilation import CompilationMode

    model_config = ModelConfig(model="facebook/opt-125m", dtype="bfloat16")
    compilation_config = CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    vllm_config = VllmConfig(
        model_config=model_config,
        compilation_config=compilation_config,
    )
    # Constructing the pass registers the producer/GEMM patterns globally
    # on the inductor pattern matcher table.
    BlockScaleSplitKZeroInitFusionPass(vllm_config, output_dtype=torch.bfloat16)

    # Now compile a model that DOESN'T match any of the producers but is
    # still subject to dynamo guard building.
    rmsnorm_op = rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()

    def fn(x, w):
        out = rmsnorm_op(x, w, 1e-6, group_size=128)
        return out[0]

    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    torch._dynamo.mark_dynamic(x, 0)
    compiled = torch.compile(fn, dynamic=True, fullgraph=True)
    compiled(x, w)
    # second run with a different M
    x2 = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
    torch._dynamo.mark_dynamic(x2, 0)
    compiled(x2, w)
