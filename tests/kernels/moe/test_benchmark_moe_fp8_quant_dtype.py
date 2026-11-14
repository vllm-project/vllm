# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""benchmark_moe.py must pass the platform FP8 dtype into the quant config.

FusedMoEQuantConfig.use_fp8_w8a8 is `quant_dtype == current_platform.fp8_dtype()`.
The tuner casts expert weights with FP8_DTYPE (= that platform dtype) but used
to hardcode torch.float8_e4m3fn in the quant config. On FNUZ (gfx942) those
disagree, use_fp8_w8a8 is False, and --dtype fp8_w8a8 tuning dies in the
unquantized path.
"""

import ast
from pathlib import Path
from types import SimpleNamespace


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "benchmarks" / "kernels" / "benchmark_moe.py").is_file():
            return parent
    raise FileNotFoundError("benchmarks/kernels/benchmark_moe.py")


def _benchmark_moe_tree() -> ast.Module:
    path = _repo_root() / "benchmarks" / "kernels" / "benchmark_moe.py"
    return ast.parse(path.read_text(), filename=str(path))


def _quant_dtype_if_in_benchmark_config(tree: ast.Module) -> ast.If:
    bench = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "benchmark_config"
    )
    run = next(
        n for n in bench.body if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    for stmt in run.body:
        if not isinstance(stmt, ast.If):
            continue
        if isinstance(stmt.test, ast.Name) and stmt.test.id == "use_fp8_w8a8":
            assigns_quant = any(
                isinstance(s, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "quant_dtype" for t in s.targets
                )
                for s in stmt.body
            )
            if assigns_quant:
                return stmt
    raise AssertionError("no use_fp8_w8a8 quant_dtype branch in benchmark_config.run")


def _eval_quant_dtype(
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    *,
    fp8_dtype: object,
    ocp_fp8: object,
    int8: object = "int8",
) -> object:
    if_node = _quant_dtype_if_in_benchmark_config(_benchmark_moe_tree())
    ns: dict = {
        "use_fp8_w8a8": use_fp8_w8a8,
        "use_int8_w8a16": use_int8_w8a16,
        "FP8_DTYPE": fp8_dtype,
        "torch": SimpleNamespace(float8_e4m3fn=ocp_fp8, int8=int8),
    }
    module = ast.Module(body=[if_node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "benchmark_moe.py", "exec"), ns)
    return ns["quant_dtype"]


def test_fp8_w8a8_quant_dtype_matches_weight_fp8_dtype():
    """On FNUZ, quant_dtype must be FP8_DTYPE, not hardcoded e4m3fn."""
    fnuz = object()
    ocp = object()
    quant_dtype = _eval_quant_dtype(True, False, fp8_dtype=fnuz, ocp_fp8=ocp)
    assert quant_dtype is fnuz, (
        "fp8_w8a8 quant_dtype must be FP8_DTYPE (same dtype used to cast w1/w2); "
        "hardcoding torch.float8_e4m3fn disagrees on FNUZ platforms"
    )
    assert quant_dtype is not ocp


def test_int8_and_unquantized_quant_dtype_branches():
    fnuz = object()
    ocp = object()
    int8 = object()
    assert (
        _eval_quant_dtype(False, True, fp8_dtype=fnuz, ocp_fp8=ocp, int8=int8) is int8
    )
    assert (
        _eval_quant_dtype(False, False, fp8_dtype=fnuz, ocp_fp8=ocp, int8=int8) is None
    )
