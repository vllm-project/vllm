# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generates documentation table for attention backends showing feature support.

This script parses all registered attention backends using AST (no imports needed)
and generates a markdown table showing what features each backend supports,
based on the checks in AttentionBackend.validate_configuration().

This approach avoids requiring CUDA/ROCm/GPU libraries to be installed.
"""

import argparse
import ast
import contextlib
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent.parent
BACKENDS_DIR = REPO_ROOT / "vllm" / "v1" / "attention" / "backends"
MLA_COMMON_FILE = (
    REPO_ROOT / "vllm" / "model_executor" / "layers" / "attention" / "mla_attention.py"
)
REGISTRY_FILE = BACKENDS_DIR / "registry.py"


def parse_registry() -> tuple[dict[str, str], dict[str, str]]:
    """Parse the registry.py file to get backend names and their class paths."""
    source = REGISTRY_FILE.read_text()
    tree = ast.parse(source)

    attention_backends: dict[str, str] = {}
    mamba_backends: dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name == "AttentionBackendEnum":
            attention_backends = _extract_enum_values(node)
        elif node.name == "MambaAttentionBackendEnum":
            mamba_backends = _extract_enum_values(node)

    return attention_backends, mamba_backends


def _extract_enum_values(node: ast.ClassDef) -> dict[str, str]:
    """Extract enum name -> value mapping from a class definition."""
    result: dict[str, str] = {}
    for item in node.body:
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if not isinstance(target, ast.Name):
                continue
            if isinstance(item.value, ast.Constant) and item.value.value:
                result[target.id] = item.value.value
    return result


def get_file_from_class_path(class_path: str) -> Path | None:
    """Convert a class path to a file path."""
    if not class_path:
        return None
    module_path = class_path.rsplit(".", 1)[0]
    file_path = REPO_ROOT / module_path.replace(".", "/")
    py_file = Path(str(file_path) + ".py")
    return py_file if py_file.exists() else None


def find_class_in_ast(tree: ast.AST, class_name: str) -> ast.ClassDef | None:
    """Find a class definition in an AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def find_method(node: ast.ClassDef, method_name: str) -> ast.FunctionDef | None:
    """Find a method in a class definition."""
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == method_name:
            return item
    return None


def method_returns_true(method: ast.FunctionDef | None) -> bool:
    """Check if a method simply returns True."""
    if method is None:
        return False
    for node in ast.walk(method):
        if not isinstance(node, ast.Return):
            continue
        if isinstance(node.value, ast.Constant) and node.value.value is True:
            return True
    return False


def parse_supported_dtypes(node: ast.ClassDef) -> str:
    """Parse supported_dtypes class variable."""
    for item in node.body:
        if not isinstance(item, ast.AnnAssign):
            continue
        if not isinstance(item.target, ast.Name):
            continue
        if item.target.id != "supported_dtypes":
            continue
        if not (item.value and isinstance(item.value, ast.List)):
            continue
        dtypes = []
        for elt in item.value.elts:
            if isinstance(elt, ast.Attribute):
                dtype_map = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}
                dtypes.append(dtype_map.get(elt.attr, elt.attr))
        return ", ".join(dtypes)
    return "fp16, bf16"  # Default


def parse_kv_cache_dtypes(node: ast.ClassDef) -> str:
    """Parse supported_kv_cache_dtypes class variable."""
    for item in node.body:
        if not isinstance(item, ast.AnnAssign):
            continue
        if not isinstance(item.target, ast.Name):
            continue
        if item.target.id != "supported_kv_cache_dtypes":
            continue
        if not (item.value and isinstance(item.value, ast.List)):
            continue
        dtypes = [
            str(elt.value) for elt in item.value.elts if isinstance(elt, ast.Constant)
        ]
        return ", ".join(dtypes)
    return "auto"  # Default


def parse_block_sizes(node: ast.ClassDef) -> str:
    """Parse get_supported_kernel_block_sizes method."""
    method = find_method(node, "get_supported_kernel_block_sizes")
    if method is None:
        return "×1"  # Default is MultipleOf(1)

    for stmt in ast.walk(method):
        if not isinstance(stmt, ast.Return):
            continue
        if not isinstance(stmt.value, ast.List):
            continue
        sizes = []
        for elt in stmt.value.elts:
            if isinstance(elt, ast.Constant):
                sizes.append(str(elt.value))
            elif isinstance(elt, ast.Call):
                is_multiple_of = (
                    isinstance(elt.func, ast.Name)
                    and elt.func.id == "MultipleOf"
                    and elt.args
                    and isinstance(elt.args[0], ast.Constant)
                )
                if is_multiple_of:
                    sizes.append(f"×{elt.args[0].value}")
        if sizes:
            return ", ".join(sizes)
    return "×1"


def parse_head_sizes(node: ast.ClassDef) -> str:
    """Parse get_supported_head_sizes method."""
    method = find_method(node, "get_supported_head_sizes")
    if method is None:
        return "Any"

    for stmt in ast.walk(method):
        if not isinstance(stmt, ast.Return):
            continue
        if not isinstance(stmt.value, ast.List):
            continue
        sizes = [
            str(elt.value) for elt in stmt.value.elts if isinstance(elt, ast.Constant)
        ]
        if sizes:
            return ", ".join(sizes)
    return "Any"


def parse_compute_capability(node: ast.ClassDef) -> str:
    """Parse supports_compute_capability method."""
    method = find_method(node, "supports_compute_capability")
    if method is None:
        return "Any"

    source = ast.unparse(method)
    if "8, 0" in source or "(8," in source:
        return "≥8.0"
    if "9, 0" in source or "9," in source:
        return "9.x-10.x" if "10" in source else "≥9.0"
    if "7, 5" in source or "7," in source:
        return "≥7.5"
    if ".major in" in source:
        return "Specific"
    return "Any"


def parse_attention_types(node: ast.ClassDef) -> str:
    """Parse supports_attn_type method."""
    method = find_method(node, "supports_attn_type")
    if method is None:
        return "Decoder"  # Default

    source = ast.unparse(method)
    types = []
    if "DECODER" in source:
        types.append("Decoder")
    if "ENCODER_ONLY" in source:
        types.append("Encoder Only")
    if "ENCODER_DECODER" in source:
        types.append("Enc-Dec")
    # Check for ENCODER without the others
    enc_only = "ENCODER_ONLY" not in source and "ENCODER_DECODER" not in source
    if "ENCODER" in source and enc_only:
        types.append("Encoder")

    if not types:
        return "Decoder"
    if len(types) >= 3:
        return "All"
    return ", ".join(types)


def check_method_overrides(node: ast.ClassDef, method_name: str) -> bool:
    """Check if a method is overridden and returns True."""
    method = find_method(node, method_name)
    return method_returns_true(method)


def get_parent_class_name(node: ast.ClassDef) -> str | None:
    """Get the name of the first parent class."""
    if not node.bases:
        return None
    first_base = node.bases[0]
    if isinstance(first_base, ast.Name):
        return first_base.id
    if isinstance(first_base, ast.Attribute):
        return first_base.attr
    return None


def analyze_backend(backend_name: str, class_path: str) -> dict[str, Any] | None:
    """Analyze a backend class and extract feature information."""
    file_path = get_file_from_class_path(class_path)
    if file_path is None:
        return None

    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except Exception as e:
        print(f"  Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return None

    class_name = class_path.rsplit(".", 1)[1]
    class_node = find_class_in_ast(tree, class_name)
    if class_node is None:
        return None

    # Check if this is an MLA backend
    parent = get_parent_class_name(class_node)
    mla_parents = ("MLACommonBackend", "FlashMLABackend", "FlashMLASparseBackend")
    is_mla_backend = (
        parent in mla_parents
        or ".mla." in class_path.lower()
        or "_mla" in backend_name.lower()
    )

    info = {
        "name": backend_name,
        "dtypes": parse_supported_dtypes(class_node),
        "kv_cache_dtypes": parse_kv_cache_dtypes(class_node),
        "block_sizes": parse_block_sizes(class_node),
        "head_sizes": parse_head_sizes(class_node),
        "attn_types": parse_attention_types(class_node),
        "compute_capability": parse_compute_capability(class_node),
        "is_mla": is_mla_backend or check_method_overrides(class_node, "is_mla"),
        "supports_sink": check_method_overrides(class_node, "supports_sink"),
        "is_sparse": check_method_overrides(class_node, "is_sparse"),
        "supports_mm_prefix": check_method_overrides(class_node, "supports_mm_prefix"),
    }

    # Ensure MLA backends are correctly marked
    if is_mla_backend:
        info["is_mla"] = True

    return info


def bool_to_emoji(value: bool) -> str:
    """Convert a boolean to a checkmark or X emoji."""
    return "✓" if value else "✗"


def generate_markdown_table(backends: list[dict[str, Any]], title: str) -> str:
    """Generate a markdown table from backend info."""
    if not backends:
        return f"## {title}\n\nNo backends found.\n"

    header = (
        "| Backend | Dtypes | KV Cache Dtypes | Block Sizes | Head Sizes "
        "| MLA | Sink | Sparse | MM Prefix | Attention Types | Compute Cap. |"
    )
    separator = (
        "|---------|--------|-----------------|-------------|------------"
        "|-----|------|--------|-----------|-----------------|--------------|"
    )
    lines = [f"## {title}", "", header, separator]

    for info in sorted(backends, key=lambda x: x["name"]):
        row = "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            info["name"],
            info["dtypes"],
            info["kv_cache_dtypes"],
            info["block_sizes"],
            info["head_sizes"],
            bool_to_emoji(info["is_mla"]),
            bool_to_emoji(info["supports_sink"]),
            bool_to_emoji(info["is_sparse"]),
            bool_to_emoji(info["supports_mm_prefix"]),
            info["attn_types"],
            info["compute_capability"],
        )
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def generate_legend() -> str:
    """Generate a legend explaining the table columns."""
    return """## Legend

| Column | Description |
|--------|-------------|
| **Dtypes** | Supported model data types (fp16, bf16, fp32) |
| **KV Cache Dtypes** | Supported KV cache data types (auto, fp8, fp8_e4m3, etc.) |
| **Block Sizes** | Supported KV cache block sizes (×N means multiples of N) |
| **Head Sizes** | Supported attention head sizes |
| **MLA** | Multi-head Latent Attention support (for DeepSeek-style models) |
| **Sink** | Attention sink support (for StreamingLLM) |
| **Sparse** | Sparse attention support |
| **MM Prefix** | Multimodal prefix full attention support |
| **Attention Types** | Supported attention patterns (Decoder, Encoder, Enc-Dec) |
| **Compute Cap.** | Required CUDA compute capability |

**Symbols:** ✓ = Supported, ✗ = Not supported

"""


def generate_docs() -> str:
    """Generate the complete documentation."""
    attention_backends_map, _ = parse_registry()

    # Parse the MLA common backend for default values
    mla_common_tree = None
    if MLA_COMMON_FILE.exists():
        with contextlib.suppress(Exception):
            mla_common_tree = ast.parse(MLA_COMMON_FILE.read_text())

    # Collect backend info
    all_backends = []
    for backend_name, class_path in attention_backends_map.items():
        if backend_name in ("CUSTOM", "TORCH_SDPA"):
            continue
        info = analyze_backend(backend_name, class_path)
        if info:
            all_backends.append(info)

    # Split into MLA and non-MLA
    mla_backends = [b for b in all_backends if b["is_mla"]]
    non_mla_backends = [b for b in all_backends if not b["is_mla"]]

    # Generate documentation
    script_path = "tools/pre_commit/generate_attention_backend_docs.py"
    doc_lines = [
        "# Attention Backend Feature Support",
        "",
        f"This document is auto-generated by `{script_path}`.",
        "It shows the feature support for each registered attention backend",
        "based on the checks in `AttentionBackend.validate_configuration()`.",
        "",
        "**Do not edit this file manually.** Run the following command to",
        "regenerate it:",
        "",
        "```bash",
        f"python {script_path}",
        "```",
        "",
    ]

    doc_lines.append(generate_legend())
    doc_lines.append(
        generate_markdown_table(non_mla_backends, "Standard Attention Backends")
    )
    mla_title = "MLA (Multi-head Latent Attention) Backends"
    doc_lines.append(generate_markdown_table(mla_backends, mla_title))

    # Suppress unused variable warning - mla_common_tree reserved for future use
    _ = mla_common_tree

    return "\n".join(doc_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate attention backend documentation table"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(REPO_ROOT / "docs" / "design" / "attention_backends.md"),
        help="Output file path (default: docs/design/attention_backends.md)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if the documentation is up to date (for pre-commit)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    # Generate the documentation
    new_content = generate_docs()

    if args.check:
        # Check mode: compare with existing file
        if not output_path.exists():
            print(f"❌ Documentation file does not exist: {output_path}")
            print(
                "Run 'python tools/pre_commit/generate_attention_backend_docs.py'"
                " to generate it."
            )
            sys.exit(1)

        existing_content = output_path.read_text()
        if existing_content != new_content:
            print(f"❌ Documentation is out of date: {output_path}")
            print(
                "Run 'python tools/pre_commit/generate_attention_backend_docs.py'"
                " to update it."
            )
            sys.exit(1)

        print(f"✅ Documentation is up to date: {output_path}")
        sys.exit(0)

    # Write mode: generate the file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(new_content)
    print(f"Generated documentation: {output_path}")


if __name__ == "__main__":
    main()
