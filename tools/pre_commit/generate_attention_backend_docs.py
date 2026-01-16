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
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent.parent
BACKENDS_DIR = REPO_ROOT / "vllm" / "v1" / "attention" / "backends"
MLA_COMMON_FILE = (
    REPO_ROOT / "vllm" / "model_executor" / "layers" / "attention" / "mla_attention.py"
)
REGISTRY_FILE = BACKENDS_DIR / "registry.py"
CUDA_PLATFORM_FILE = REPO_ROOT / "vllm" / "platforms" / "cuda.py"


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


def parse_cuda_priority_lists() -> dict[str, list[str]]:
    """Parse priority lists from cuda.py using AST.

    The structure of _get_backend_priorities is:
        if use_mla:
            if device_capability.major == 10:
                return [MLA list for SM100]
            else:
                return [MLA list for default]
        else:
            if device_capability.major == 10:
                return [Standard list for SM100]
            else:
                return [Standard list for default]
    """
    if not CUDA_PLATFORM_FILE.exists():
        return {}

    try:
        source = CUDA_PLATFORM_FILE.read_text()
        tree = ast.parse(source)
    except Exception:
        return {}

    priorities: dict[str, list[str]] = {}

    # Find the _get_backend_priorities function
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "_get_backend_priorities":
            continue

        # Process the function body directly
        for stmt in node.body:
            if not isinstance(stmt, ast.If):
                continue

            # Check if this is the "if use_mla:" branch
            is_mla_branch = (
                isinstance(stmt.test, ast.Name) and stmt.test.id == "use_mla"
            )

            if is_mla_branch:
                # Process MLA branches
                _extract_priority_from_if(stmt.body, priorities, "mla")
                if stmt.orelse:
                    _extract_priority_from_else(stmt.orelse, priorities, "standard")
            else:
                # Process standard branches directly
                _extract_priority_from_if([stmt], priorities, "standard")

    return priorities


def _extract_priority_from_if(
    body: list, priorities: dict[str, list[str]], prefix: str
) -> None:
    """Extract priority list from if statement body."""
    for stmt in body:
        if isinstance(stmt, ast.If):
            # Check for device_capability.major == 10
            test_src = ast.unparse(stmt.test)
            is_sm100 = "major == 10" in test_src or ".major == 10" in test_src

            # Get the return list from the if body
            for sub in stmt.body:
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.List):
                    backends = _extract_backend_names(sub.value)
                    if is_sm100:
                        priorities[f"{prefix}_sm100"] = backends
                    else:
                        priorities[f"{prefix}_default"] = backends

            # Get the return list from the else body
            for sub in stmt.orelse:
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.List):
                    backends = _extract_backend_names(sub.value)
                    if is_sm100:
                        priorities[f"{prefix}_default"] = backends
                    else:
                        priorities[f"{prefix}_sm100"] = backends

        elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.List):
            backends = _extract_backend_names(stmt.value)
            priorities[f"{prefix}_default"] = backends


def _extract_priority_from_else(
    orelse: list, priorities: dict[str, list[str]], prefix: str
) -> None:
    """Extract priority list from else clause."""
    for stmt in orelse:
        if isinstance(stmt, ast.If):
            _extract_priority_from_if([stmt], priorities, prefix)
        elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.List):
            backends = _extract_backend_names(stmt.value)
            priorities[f"{prefix}_default"] = backends


def _extract_backend_names(list_node: ast.List) -> list[str]:
    """Extract backend enum names from an AST List node."""
    backends = []
    for elt in list_node.elts:
        if isinstance(elt, ast.Attribute):
            backends.append(elt.attr)
    return backends


def generate_usage_section() -> str:
    """Generate the usage documentation section."""
    return """## Setting the Attention Backend

### Command Line

There are two ways to specify the backend from the command line:

**Option 1: Using `--attention-backend` (simple)**

```bash
vllm serve <model> --attention-backend FLASH_ATTN
```

**Option 2: Using `--attention-config.backend` (structured config)**

```bash
# Dot notation
vllm serve <model> --attention-config.backend FLASH_ATTN

# JSON format with -ac shorthand
vllm serve <model> -ac '{"backend": "FLASH_ATTN"}'
```

> **Note:** `--attention-backend` and `--attention-config.backend` are mutually
> exclusive. Use one or the other, not both.

### Python API

Use `AttentionConfig` with the `LLM` class:

```python
from vllm import LLM
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# Method 1: Using AttentionConfig with enum
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN),
)

# Method 2: Using attention_backend parameter with string
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    attention_backend="FLASH_ATTN",
)
```

## Backend Selection Behavior

### Manual Selection

When you explicitly set a backend via `--attention-backend` or `AttentionConfig`:

1. The backend is **validated** against your configuration (model dtype, head
   size, compute capability, etc.)
2. If the backend **doesn't support** your configuration, an error is raised
   with the specific reason
3. If valid, the backend is used

Example error when selecting an incompatible backend:

```text
ValueError: Selected backend FLASHMLA is not valid for this configuration.
Reason: ['compute capability not supported']
```

### Automatic Selection

When no backend is specified (the default):

1. vLLM iterates through backends in **priority order** (see tables below)
2. Each backend is validated against your configuration
3. The **first compatible backend** is selected
4. If no backend is compatible, an error is raised listing all backends and
   their incompatibility reasons

"""


def generate_priority_section(priorities: dict[str, list[str]]) -> str:
    """Generate the priority ranking section."""
    lines = [
        "## Backend Priority (CUDA)",
        "",
        "When no backend is explicitly selected, vLLM chooses the first",
        "compatible backend from these priority-ordered lists.",
        "",
        "Priority is **1 = highest** (tried first).",
        "",
    ]

    # Standard backends
    lines.append("### Standard Attention (non-MLA)")
    lines.append("")

    if "standard_sm100" in priorities:
        lines.append("**Blackwell (SM 10.x):**")
        lines.append("")
        lines.append("| Priority | Backend |")
        lines.append("|----------|---------|")
        for i, backend in enumerate(priorities["standard_sm100"], 1):
            lines.append(f"| {i} | {backend} |")
        lines.append("")

    if "standard_default" in priorities:
        lines.append("**Ampere/Hopper (SM 8.x-9.x):**")
        lines.append("")
        lines.append("| Priority | Backend |")
        lines.append("|----------|---------|")
        for i, backend in enumerate(priorities["standard_default"], 1):
            lines.append(f"| {i} | {backend} |")
        lines.append("")

    # MLA backends
    lines.append("### MLA Attention (DeepSeek-style)")
    lines.append("")

    if "mla_sm100" in priorities:
        lines.append("**Blackwell (SM 10.x):**")
        lines.append("")
        lines.append("| Priority | Backend |")
        lines.append("|----------|---------|")
        for i, backend in enumerate(priorities["mla_sm100"], 1):
            lines.append(f"| {i} | {backend} |")
        lines.append("")

    if "mla_default" in priorities:
        lines.append("**Ampere/Hopper (SM 8.x-9.x):**")
        lines.append("")
        lines.append("| Priority | Backend |")
        lines.append("|----------|---------|")
        for i, backend in enumerate(priorities["mla_default"], 1):
            lines.append(f"| {i} | {backend} |")
        lines.append("")

    lines.append(
        "> **Note:** ROCm and CPU platforms have their own selection logic. "
        "See the platform-specific documentation for details."
    )
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

    # Parse priority lists from cuda.py
    priorities = parse_cuda_priority_lists()

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

    # Add usage documentation
    doc_lines.append(generate_usage_section())

    # Add priority section
    doc_lines.append(generate_priority_section(priorities))

    # Add legend and feature tables
    doc_lines.append(generate_legend())
    doc_lines.append(
        generate_markdown_table(non_mla_backends, "Standard Attention Backends")
    )
    mla_title = "MLA (Multi-head Latent Attention) Backends"
    doc_lines.append(generate_markdown_table(mla_backends, mla_title))

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
