# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inventory operator row dispatch without importing CUDA or optional backends."""

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = {
    "M",
    "m",
    "rows",
    "tokens",
    "num_tokens",
    "num_rows",
    "batch_size",
    "n_tokens",
    "tokens_per_expert",
    "base_programs",
    "effective_batch",
}
SCOPES = (
    "vllm/_custom_ops.py",
    "vllm/models/common",
    "vllm/compilation/passes/fusion",
    "vllm/model_executor/kernels",
    "vllm/model_executor/layers",
    "vllm/models/qwen4_exp",
    "vllm/models/deepseek_v41",
    "vllm/v1/attention",
    "vllm/utils",
    "vllm/v1/sample/ops",
    "vllm/distributed/device_communicators",
)
EXCLUDED = (
    "/amd/",
    "rocm",
    "rdna",
    "cpu",
    "xpu",
    "flydsl",
    "/helion/",
    "/minimax",
    "/bailing",
)


def row_operand(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in ROWS
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "shape"
        and "weight" not in ast.unparse(node.value.value)
        and "query_start_loc" not in ast.unparse(node.value.value)
        and "cache" not in ast.unparse(node.value.value)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == 0
    )


def fixed_limit(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and node.value > 1
    if isinstance(node, ast.Name):
        return node.id.isupper() and any(x in node.id for x in ("TOKENS", "ROWS", "_M"))
    if isinstance(node, ast.Attribute):
        return any(
            x in node.attr.upper() for x in ("MAX_TOKENS", "MAX_ROWS", "MIN_ROWS")
        )
    if isinstance(node, ast.Call):
        name = ast.unparse(node.func)
        return "resident_clusters" in name or name.endswith(("max_tokens", "max_rows"))
    if isinstance(node, ast.BinOp):
        return fixed_limit(node.left) or fixed_limit(node.right)
    return False


class Visitor(ast.NodeVisitor):
    def __init__(self, relative: str, records: list[dict]) -> None:
        self.relative = relative
        self.records = records
        self.owners: list[str] = []

    def record(self, node: ast.AST, kind: str) -> None:
        self.records.append(
            {
                "file": self.relative,
                "operator": ".".join(self.owners),
                "line": node.lineno,
                "kind": kind,
                "expression": ast.unparse(node),
            }
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.owners.append(node.name)
        self.generic_visit(node)
        self.owners.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        decorators = " ".join(ast.unparse(d) for d in node.decorator_list)
        if any(x in decorators for x in (".jit", ".kernel", "T.prim_func")):
            return
        if any(x in node.name for x in ("rocm", "aiter")):
            return
        self.owners.append(node.name)
        self.generic_visit(node)
        self.owners.pop()

    def visit_Assert(self, node: ast.Assert) -> None:
        # Buffer/shape validation is not a performance bucket.
        return

    def visit_Compare(self, node: ast.Compare) -> None:
        if not any(
            isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops
        ):
            return
        operands = (node.left, *node.comparators)
        if any(row_operand(n) for n in operands) and any(
            fixed_limit(n) for n in operands
        ):
            self.record(node, "raw_host_threshold")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = ast.unparse(node.func).rsplit(".", 1)[-1]
        if name in ("m_in_range", "select_nearest_m"):
            self.record(node, "range" if name == "m_in_range" else "nearest")
        elif name in ("MBucketTable", "PhaseBucketTable"):
            self.record(node, "table")
        self.generic_visit(node)


def inventory(root: Path = ROOT) -> list[dict]:
    records = []
    for scope in SCOPES:
        source = root / scope
        paths = [source] if source.is_file() else sorted(source.rglob("*.py"))
        for path in paths:
            relative = str(path.relative_to(root))
            if any(part in relative for part in EXCLUDED):
                continue
            tree = ast.parse(path.read_text())

            Visitor(relative, records).visit(tree)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the complete JSON inventory")
    parser.add_argument(
        "--check", action="store_true", help="Reject unconverted host row thresholds"
    )
    args = parser.parse_args()
    records = inventory()
    if args.output:
        args.output.write_text(json.dumps(records, indent=2) + "\n")
    counts = {
        kind: sum(r["kind"] == kind for r in records)
        for kind in sorted({r["kind"] for r in records})
    }
    print(json.dumps(counts, sort_keys=True))
    for record in records:
        if record["kind"] == "raw_host_threshold":
            print(f"{record['file']}:{record['line']} {record['expression']}")
    if args.check and any(r["kind"] == "raw_host_threshold" for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
