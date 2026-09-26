# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generated_content import fill_markers  # noqa: E402

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).parent.parent.parent.parent

# Files to scan for metric definitions - each fills a `gen:` marker in
# docs/usage/metrics.md with its table (the section heading and any preamble
# live in the tracked page next to the marker).
METRIC_SOURCE_FILES = [
    {"path": "vllm/v1/metrics/loggers.py", "key": "metrics-general"},
    {"path": "vllm/v1/spec_decode/metrics.py", "key": "metrics-spec-decode"},
    {
        "path": "vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py",
        "key": "metrics-nixl",
    },
    {"path": "vllm/v1/metrics/perf.py", "key": "metrics-mfu"},
]


def is_metric_name(value: str) -> bool:
    return value.startswith("vllm:") and value.removeprefix("vllm:").isidentifier()


def collect_string_bindings(func: ast.AST) -> dict[str, list[str]]:
    """Map local names in `func` to the string values they can hold.

    Covers `name = "vllm:..."` and `for name, doc in rows` where `rows` is
    assigned literal lists of string tuples, so that `name=name` resolves.
    """
    strings: dict[str, list[str]] = {}
    rows: dict[str, list[tuple[str, ...]]] = {}
    for node in ast.walk(func):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if not isinstance(target, ast.Name):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            strings.setdefault(target.id, []).append(value.value)
        elif isinstance(value, ast.List | ast.Tuple):
            try:
                table = ast.literal_eval(value)
            except ValueError:
                continue
            rows.setdefault(target.id, []).extend(
                row
                for row in table
                if isinstance(row, tuple) and all(isinstance(v, str) for v in row)
            )

    for node in ast.walk(func):
        if not isinstance(node, ast.For | ast.comprehension):
            continue
        target, source = node.target, node.iter
        if not (isinstance(target, ast.Tuple) and isinstance(source, ast.Name)):
            continue
        for i, elt in enumerate(target.elts):
            if isinstance(elt, ast.Name):
                strings.setdefault(elt.id, []).extend(
                    row[i] for row in rows.get(source.id, []) if i < len(row)
                )
    return strings


class MetricExtractor(ast.NodeVisitor):
    """AST visitor to extract metric definitions."""

    def __init__(self):
        self.metrics: list[dict[str, str]] = []
        self._bindings: dict[str, list[str]] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        outer = self._bindings
        self._bindings = collect_string_bindings(node)
        self.generic_visit(node)
        self._bindings = outer

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls to find metric class instantiations."""
        metric_type = self._get_metric_type(node)
        if metric_type:
            names = self._extract_kwarg(node, "name")
            docs = self._extract_kwarg(node, "documentation")
            if len(docs) != len(names):
                docs = [docs[0] if len(docs) == 1 else ""] * len(names)

            for name, documentation in zip(names, docs):
                self.metrics.append(
                    {
                        "name": name,
                        "type": metric_type,
                        "documentation": documentation,
                    }
                )

        self.generic_visit(node)

    def _get_metric_type(self, node: ast.Call) -> str | None:
        """Determine if this call creates a metric and return its type."""
        metric_type_map = {
            "_gauge_cls": "gauge",
            "_counter_cls": "counter",
            "_histogram_cls": "histogram",
        }
        if isinstance(node.func, ast.Attribute):
            return metric_type_map.get(node.func.attr)
        return None

    def _extract_kwarg(self, node: ast.Call, key: str) -> list[str]:
        """Extract the possible values of a keyword argument of a call."""
        for keyword in node.keywords:
            if keyword.arg == key:
                if isinstance(keyword.value, ast.Name):
                    return self._bindings.get(keyword.value.id, [])
                value = self._get_string_value(keyword.value)
                return [value] if value is not None else []
        return []

    def _get_string_value(self, node: ast.AST) -> str | None:
        """Extract string value from an AST node."""
        if isinstance(node, ast.Constant):
            return str(node.value) if node.value is not None else None
        return None


def extract_metrics_from_file(filepath: Path) -> list[dict[str, str]]:
    """Parse a Python file and extract all metric definitions."""
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(filepath))
        extractor = MetricExtractor()
        extractor.visit(tree)
        return extractor.metrics
    except Exception as e:
        raise RuntimeError(f"Failed to parse {filepath}: {e}") from e


def generate_markdown_table(metrics: list[dict[str, str]]) -> str:
    """Generate a markdown table from extracted metrics."""
    if not metrics:
        return "No metrics found.\n"

    # Sort by type, then by name
    metrics_sorted = sorted(metrics, key=lambda m: (m["type"], m["name"]))

    lines = []
    lines.append("| Metric Name | Type | Description |")
    lines.append("|-------------|------|-------------|")

    for metric in metrics_sorted:
        name = metric["name"]
        metric_type = metric["type"].capitalize()
        doc = metric["documentation"].replace("\n", " ").strip()
        lines.append(f"| `{name}` | {metric_type} | {doc} |")

    return "\n".join(lines) + "\n"


logger.info("Generating metrics documentation")

blocks = {}
total_metrics = 0
for source_config in METRIC_SOURCE_FILES:
    source_path = source_config["path"]

    filepath = ROOT_DIR / source_path
    if not filepath.exists():
        raise FileNotFoundError(f"Metrics source file not found: {filepath}")

    logger.debug("Extracting metrics from: %s", source_path)
    metrics = extract_metrics_from_file(filepath)
    logger.debug("Found %d metrics in %s", len(metrics), source_path)

    tree = ast.parse(filepath.read_text(encoding="utf-8"))
    literal_names = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and is_metric_name(node.value)
    }
    if missed := literal_names - {m["name"] for m in metrics}:
        raise ValueError(
            f"{source_path}: MetricExtractor missed {sorted(missed)}; "
            "teach it how these metrics are defined so they appear in the docs"
        )

    blocks[source_config["key"]] = generate_markdown_table(metrics).strip()
    total_metrics += len(metrics)

fill_markers("usage/metrics.md", blocks)
logger.info(
    "Total metrics generated: %d across %d files",
    total_metrics,
    len(METRIC_SOURCE_FILES),
)
