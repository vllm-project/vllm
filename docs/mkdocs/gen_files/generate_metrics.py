# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generated_content import append_to_page  # noqa: E402

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).parent.parent.parent.parent

# Files to scan for metric definitions - each will generate a separate table
METRIC_SOURCE_FILES = [
    {"path": "vllm/v1/metrics/loggers.py", "title": "General Metrics"},
    {
        "path": "vllm/v1/spec_decode/metrics.py",
        "title": "Speculative Decoding Metrics",
    },
    {
        "path": "vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py",
        "title": "NIXL KV Connector Metrics",
    },
    {
        "path": "vllm/v1/metrics/perf.py",
        "title": "Model Flops Utilization (MFU) Performance Metrics",
        "preamble": "These metrics are available via `--enable-mfu-metrics`:",
    },
]


class MetricExtractor(ast.NodeVisitor):
    """AST visitor to extract metric definitions."""

    def __init__(self):
        self.metrics: list[dict[str, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls to find metric class instantiations."""
        metric_type = self._get_metric_type(node)
        if metric_type:
            name = self._extract_kwarg(node, "name")
            documentation = self._extract_kwarg(node, "documentation")

            if name:
                self.metrics.append(
                    {
                        "name": name,
                        "type": metric_type,
                        "documentation": documentation or "",
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

    def _extract_kwarg(self, node: ast.Call, key: str) -> str | None:
        """Extract a keyword argument value from a function call."""
        for keyword in node.keywords:
            if keyword.arg == key:
                return self._get_string_value(keyword.value)
        return None

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

content = ""
total_metrics = 0
for source_config in METRIC_SOURCE_FILES:
    source_path = source_config["path"]

    filepath = ROOT_DIR / source_path
    if not filepath.exists():
        raise FileNotFoundError(f"Metrics source file not found: {filepath}")

    logger.debug("Extracting metrics from: %s", source_path)
    metrics = extract_metrics_from_file(filepath)
    logger.debug("Found %d metrics in %s", len(metrics), source_path)

    content += f"## {source_config['title']}\n\n"
    if preamble := source_config.get("preamble"):
        content += f"{preamble}\n\n"
    content += f"{generate_markdown_table(metrics)}\n"
    total_metrics += len(metrics)

append_to_page("usage/metrics.md", content)
logger.info(
    "Total metrics generated: %d across %d files",
    total_metrics,
    len(METRIC_SOURCE_FILES),
)
