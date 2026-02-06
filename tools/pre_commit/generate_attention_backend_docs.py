# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generates documentation table for attention backends showing feature support.

This script parses all registered attention backends using AST (no imports needed)
and generates a markdown table showing what features each backend supports,
based on the checks in AttentionBackend.validate_configuration().

This approach avoids requiring CUDA/ROCm/GPU libraries to be installed.

When used as a pre-commit hook, this script receives filenames as arguments
and only runs the check if any of the relevant files were modified.
"""

import argparse
import ast
import fnmatch
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent.parent

RELEVANT_PATTERNS = [
    "vllm/v1/attention/backends/*.py",
    "vllm/v1/attention/backends/**/*.py",
    "vllm/v1/attention/backends/fa_utils.py",
    "vllm/model_executor/layers/attention/mla_attention.py",
    "vllm/platforms/cuda.py",
    "tools/pre_commit/generate_attention_backend_docs.py",
    "docs/design/attention_backends.md",
]


def is_relevant_file(filepath: str) -> bool:
    """Check if a file matches any of the relevant patterns."""
    path = Path(filepath)
    if path.is_absolute():
        try:
            path = path.relative_to(REPO_ROOT)
        except ValueError:
            return False
    path_str = str(path)

    return any(fnmatch.fnmatch(path_str, pattern) for pattern in RELEVANT_PATTERNS)


BACKENDS_DIR = REPO_ROOT / "vllm" / "v1" / "attention" / "backends"
REGISTRY_FILE = BACKENDS_DIR / "registry.py"
CUDA_PLATFORM_FILE = REPO_ROOT / "vllm" / "platforms" / "cuda.py"
FA_UTILS_FILE = BACKENDS_DIR / "fa_utils.py"
FLASHINFER_UTILS_FILE = REPO_ROOT / "vllm" / "utils" / "flashinfer.py"
MLA_ATTENTION_FILE = (
    REPO_ROOT / "vllm" / "model_executor" / "layers" / "attention" / "mla_attention.py"
)


def parse_registry() -> dict[str, str]:
    """Parse the registry.py file to get backend names and their class paths."""
    tree = ast.parse(REGISTRY_FILE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AttentionBackendEnum":
            return _extract_enum_values(node)
    return {}


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
    module_path = class_path.rsplit(".", 1)[0].replace(".", "/")
    py_file = REPO_ROOT / f"{module_path}.py"
    return py_file if py_file.exists() else None


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    """Find a function definition by name in an AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _function_checks_fa_version(func: ast.FunctionDef | None) -> bool:
    """Check if a function compares get_flash_attn_version() to a value."""
    if func is None:
        return False
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Name)
            and node.left.func.id == "get_flash_attn_version"
        ):
            return True
    return False


def _extract_major_check(compare: ast.Compare) -> tuple[str, int] | None:
    """Extract (op, value) from `device_capability.major <op> <value>`.

    Returns ("==", 9) for `major == 9`, (">=", 10) for `major >= 10`, etc.
    """
    if not (
        isinstance(compare.left, ast.Attribute)
        and compare.left.attr == "major"
        and compare.comparators
        and isinstance(compare.comparators[0], ast.Constant)
    ):
        return None

    op = compare.ops[0]
    val = compare.comparators[0].value
    if not isinstance(val, int):
        return None
    if isinstance(op, ast.Eq):
        return ("==", val)
    if isinstance(op, ast.GtE):
        return (">=", val)
    return None


def _parse_fa_compute_caps(func: ast.FunctionDef | None) -> dict[str, str]:
    """Parse get_flash_attn_version() to find FA3/FA4 compute capability checks.

    Looks for patterns like:
      - `if device_capability.major == 9 and ...:`  -> FA3
      - `elif device_capability.major >= 10 and ...:`  -> FA4
    """
    result: dict[str, str] = {}
    if func is None:
        return result

    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue

        # Extract comparisons from either BoolOp (and/or) or direct Compare
        test = node.test
        comparisons = (
            [v for v in test.values if isinstance(v, ast.Compare)]
            if isinstance(test, ast.BoolOp)
            else [test]
            if isinstance(test, ast.Compare)
            else []
        )

        for comp in comparisons:
            check = _extract_major_check(comp)
            if check is None:
                continue
            op, val = check
            if op == "==" and "fa3" not in result:
                result["fa3"] = f"{val}.x"
            elif op == ">=" and "fa4" not in result:
                result["fa4"] = f"≥{val}.0"

    return result


def _parse_fa4_supported_caps() -> str | None:
    """Parse flash_attn_interface.py for FA4 supported compute capabilities.

    Looks for `cc not in [9, 10, 11]` pattern in _is_fa4_supported().
    """
    fa_interface_file = (
        REPO_ROOT / "vllm" / "vllm_flash_attn" / "flash_attn_interface.py"
    )
    if not fa_interface_file.exists():
        return None

    try:
        tree = ast.parse(fa_interface_file.read_text())
    except Exception:
        return None

    func = _find_function(tree, "_is_fa4_supported")
    if func is None:
        return None

    for node in ast.walk(func):
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotIn)
            and isinstance(node.comparators[0], ast.List)
        ):
            continue

        caps: list[int] = [
            e.value
            for e in node.comparators[0].elts
            if isinstance(e, ast.Constant) and isinstance(e.value, int)
        ]
        if caps:
            caps.sort()
            return f"{caps[0]}.x-{caps[-1]}.x"

    return None


def parse_flash_attn_features() -> dict[str, dict[str, Any]]:
    """Parse fa_utils.py to detect FA2 vs FA3 vs FA4 feature differences.

    Returns a dict with 'fa2', 'fa3', and 'fa4' keys containing their respective
    feature overrides for compute capability, KV cache dtypes, and sink support.
    """
    if not FA_UTILS_FILE.exists():
        return {}

    try:
        tree = ast.parse(FA_UTILS_FILE.read_text())
    except Exception:
        return {}

    # Check which features are gated by FA version checks
    fp8_func = _find_function(tree, "flash_attn_supports_fp8")
    sinks_func = _find_function(tree, "flash_attn_supports_sinks")
    fa3_supports_fp8 = _function_checks_fa_version(fp8_func)
    fa3_supports_sinks = _function_checks_fa_version(sinks_func)

    # Parse compute capability requirements from get_flash_attn_version()
    version_func = _find_function(tree, "get_flash_attn_version")
    compute_caps = _parse_fa_compute_caps(version_func)
    fa3_compute_cap = compute_caps.get("fa3")
    fa4_compute_cap = compute_caps.get("fa4") or _parse_fa4_supported_caps()

    return {
        "fa2": {"supports_fp8": False, "supports_sink": False},
        "fa3": {
            "compute_capability": fa3_compute_cap,
            "supports_fp8": fa3_supports_fp8,
            "supports_sink": fa3_supports_sinks,
        },
        "fa4": {
            "compute_capability": fa4_compute_cap,
            "supports_fp8": False,
            "supports_sink": False,
        },
    }


def parse_flashinfer_trtllm_features() -> dict[str, dict[str, Any]]:
    """Parse flashinfer.py to detect TRTLLM-specific features.

    FLASHINFER uses TRTLLM attention on SM100 (Blackwell), which has different
    capabilities (e.g., sink support) than native FlashInfer on earlier GPUs.
    """
    if not FLASHINFER_UTILS_FILE.exists():
        return {}

    try:
        tree = ast.parse(FLASHINFER_UTILS_FILE.read_text())
    except Exception:
        return {}

    trtllm_compute_cap: str | None = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        # Parse supports_trtllm_attention for compute capability
        # Look for: current_platform.is_device_capability_family(100)
        if node.name == "supports_trtllm_attention":
            for n in ast.walk(node):
                if (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "is_device_capability_family"
                    and n.args
                    and isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, int)
                ):
                    cap = n.args[0].value
                    # Convert 100 -> "10.x"
                    trtllm_compute_cap = f"{cap // 10}.x"
                    break

    if not trtllm_compute_cap:
        return {}

    return {
        "native": {
            # Native FlashInfer: everything except SM100
            "supports_sink": False,
        },
        "trtllm": {
            # TRTLLM pathway on Blackwell
            "compute_capability": trtllm_compute_cap,
            "supports_sink": True,
        },
    }


def parse_mla_prefill_backends() -> list[dict[str, Any]]:
    """Parse MLA prefill backend options from mla_attention.py.

    MLA uses different backends for prefill vs decode. The decode backends are
    registered in the registry, but prefill backends are selected at runtime
    based on conditions in MLACommonImpl.__init__.

    Returns a list of prefill backend info dicts with their requirements.
    """
    if not MLA_ATTENTION_FILE.exists():
        return []

    try:
        tree = ast.parse(MLA_ATTENTION_FILE.read_text())
    except Exception:
        return []

    # Find compute capability requirements by parsing use_* functions
    flashinfer_cc: str | None = None
    cudnn_cc: str | None = None
    trtllm_cc: str | None = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        # Parse use_flashinfer_prefill for compute capability (SM100)
        if node.name == "use_flashinfer_prefill":
            for n in ast.walk(node):
                if (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "is_device_capability_family"
                    and n.args
                    and isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, int)
                ):
                    flashinfer_cc = f"{n.args[0].value // 10}.x"

        # Parse use_cudnn_prefill for compute capability (SM100)
        if node.name == "use_cudnn_prefill":
            for n in ast.walk(node):
                if (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "is_device_capability_family"
                    and n.args
                    and isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, int)
                ):
                    cudnn_cc = f"{n.args[0].value // 10}.x"

        # Parse use_trtllm_ragged_deepseek_prefill for compute capability
        if node.name == "use_trtllm_ragged_deepseek_prefill":
            for n in ast.walk(node):
                if (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "is_device_capability_family"
                    and n.args
                    and isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, int)
                ):
                    trtllm_cc = f"{n.args[0].value // 10}.x"

    # Build prefill backend list based on what we found
    # Order matches the priority in MLACommonImpl.__init__
    prefill_backends: list[dict[str, Any]] = []

    # TRT-LLM Ragged (highest priority if available)
    if trtllm_cc:
        prefill_backends.append(
            {
                "name": "TRT-LLM Ragged‡",
                "description": "TensorRT-LLM ragged attention",
                "compute_capability": trtllm_cc,
                "enable": "Default on SM100",
                "disable": "`-ac.use_trtllm_ragged_deepseek_prefill=0`",
                "notes": "DeepSeek R1 dims only",
            }
        )

    # FlashInfer prefill
    if flashinfer_cc:
        prefill_backends.append(
            {
                "name": "FlashInfer",
                "description": "FlashInfer CUTLASS backend",
                "compute_capability": flashinfer_cc,
                "enable": "`-ac.disable_flashinfer_prefill=0`",
                "disable": "`-ac.disable_flashinfer_prefill=1`",
                "notes": "DeepSeek R1 dims only",
            }
        )

    # cuDNN prefill
    if cudnn_cc:
        prefill_backends.append(
            {
                "name": "cuDNN",
                "description": "cuDNN-based attention",
                "compute_capability": cudnn_cc,
                "enable": "`-ac.use_cudnn_prefill=1`",
                "disable": "`-ac.use_cudnn_prefill=0`",
                "notes": "",
            }
        )

    # FlashAttention is always available as fallback
    prefill_backends.append(
        {
            "name": "FlashAttention",
            "description": "FlashAttention varlen (FA2/FA3)",
            "compute_capability": "Any",
            "enable": "Default fallback",
            "disable": "Use other backends",
            "notes": "FA3 on SM90, FA2 otherwise",
        }
    )

    return prefill_backends


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


def _parse_list_class_var(node: ast.ClassDef, var_name: str) -> list[str] | None:
    """Parse a list-type class variable, returning None if not found."""
    for item in node.body:
        if not isinstance(item, ast.AnnAssign):
            continue
        if not isinstance(item.target, ast.Name):
            continue
        if item.target.id != var_name:
            continue
        if not (item.value and isinstance(item.value, ast.List)):
            continue
        result = []
        for elt in item.value.elts:
            if isinstance(elt, ast.Attribute):
                result.append(elt.attr)
            elif isinstance(elt, ast.Constant):
                result.append(str(elt.value))
        return result
    return None


def parse_supported_dtypes(node: ast.ClassDef) -> str:
    """Parse supported_dtypes class variable."""
    dtype_map = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}
    dtypes = _parse_list_class_var(node, "supported_dtypes")
    if dtypes is None:
        return "fp16, bf16"
    return ", ".join(dtype_map.get(d, d) for d in dtypes)


def parse_kv_cache_dtypes(node: ast.ClassDef) -> str:
    """Parse supported_kv_cache_dtypes class var or supports_kv_cache_dtype method."""
    # First try the class variable
    dtypes = _parse_list_class_var(node, "supported_kv_cache_dtypes")
    if dtypes:
        return ", ".join(dtypes)

    # Fall back to parsing the supports_kv_cache_dtype method
    # Look for `kv_cache_dtype in ["auto", "bfloat16"]` pattern
    method = find_method(node, "supports_kv_cache_dtype")
    if method:
        for n in ast.walk(method):
            if (
                isinstance(n, ast.Compare)
                and len(n.ops) == 1
                and isinstance(n.ops[0], ast.In)
                and len(n.comparators) == 1
                and isinstance(n.comparators[0], ast.List)
            ):
                dtypes = [
                    e.value
                    for e in n.comparators[0].elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
                if dtypes:
                    return ", ".join(dtypes)

    return "auto"


def _parse_return_list(
    method: ast.FunctionDef | None, handle_multiple_of: bool = False
) -> list[str]:
    """Extract list items from a method's return statement."""
    if method is None:
        return []
    for stmt in ast.walk(method):
        if not isinstance(stmt, ast.Return):
            continue
        if not isinstance(stmt.value, ast.List):
            continue
        sizes = []
        for elt in stmt.value.elts:
            if isinstance(elt, ast.Constant):
                sizes.append(str(elt.value))
            elif (
                handle_multiple_of
                and isinstance(elt, ast.Call)
                and isinstance(elt.func, ast.Name)
                and elt.func.id == "MultipleOf"
                and elt.args
                and isinstance(elt.args[0], ast.Constant)
            ):
                sizes.append(f"%{elt.args[0].value}")
        if sizes:
            return sizes
    return []


def parse_block_sizes(node: ast.ClassDef) -> str:
    """Parse get_supported_kernel_block_sizes method."""
    method = find_method(node, "get_supported_kernel_block_sizes")
    sizes = _parse_return_list(method, handle_multiple_of=True)
    return ", ".join(sizes) if sizes else "Any"


def parse_head_sizes(node: ast.ClassDef) -> str:
    """Parse get_supported_head_sizes method."""
    method = find_method(node, "get_supported_head_sizes")
    sizes = _parse_return_list(method)
    return ", ".join(sizes) if sizes else "Any"


def parse_compute_capability(node: ast.ClassDef) -> str:
    """Parse supports_compute_capability method."""
    method = find_method(node, "supports_compute_capability")
    if method is None:
        return "Any"

    min_cap: tuple[int, int] | None = None
    max_cap: tuple[int, int] | None = None
    major_list: list[int] = []

    for n in ast.walk(method):
        if not isinstance(n, ast.Compare):
            continue

        # Handle `capability >= DeviceCapability(...)` or `capability <= ...`
        for op, comp in zip(n.ops, n.comparators):
            if not (
                isinstance(comp, ast.Call)
                and isinstance(comp.func, ast.Name)
                and comp.func.id == "DeviceCapability"
                and comp.args
                and isinstance(comp.args[0], ast.Constant)
            ):
                continue
            major = comp.args[0].value
            minor = 0
            if len(comp.args) > 1 and isinstance(comp.args[1], ast.Constant):
                minor = comp.args[1].value
            if isinstance(op, ast.GtE):
                min_cap = (major, minor)
            elif isinstance(op, ast.LtE):
                max_cap = (major, minor)

        # Handle `capability.major == N` or `capability.major in [N, M]`
        if (
            isinstance(n.left, ast.Attribute)
            and n.left.attr == "major"
            and len(n.ops) == 1
            and len(n.comparators) == 1
        ):
            comp = n.comparators[0]
            if isinstance(n.ops[0], ast.Eq) and isinstance(comp, ast.Constant):
                major_list.append(comp.value)
            elif isinstance(n.ops[0], ast.In) and isinstance(comp, ast.List):
                major_list.extend(
                    e.value
                    for e in comp.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, int)
                )

    if major_list:
        major_list.sort()
        if len(major_list) == 1:
            return f"{major_list[0]}.x"
        return f"{major_list[0]}.x-{major_list[-1]}.x"

    if min_cap:
        if max_cap:
            return f"{min_cap[0]}.x-{max_cap[0]}.x"
        return f"≥{min_cap[0]}.{min_cap[1]}"

    return "Any"


def parse_attention_types(node: ast.ClassDef) -> str:
    """Parse supports_attn_type method."""
    method = find_method(node, "supports_attn_type")
    if method is None:
        return "Decoder"

    type_map = {
        "DECODER": "Decoder",
        "ENCODER": "Encoder",
        "ENCODER_ONLY": "Encoder Only",
        "ENCODER_DECODER": "Enc-Dec",
    }
    types: set[str] = set()

    for n in ast.walk(method):
        # Handle `attn_type in (AttentionType.DECODER, ...)`
        if not (
            isinstance(n, ast.Compare)
            and len(n.ops) == 1
            and isinstance(n.ops[0], ast.In)
            and len(n.comparators) == 1
            and isinstance(n.comparators[0], ast.Tuple | ast.Set)
        ):
            continue

        for elt in n.comparators[0].elts:
            if isinstance(elt, ast.Attribute) and elt.attr in type_map:
                types.add(type_map[elt.attr])

    if not types:
        return "Decoder"
    return "All" if len(types) >= 3 else ", ".join(sorted(types))


def check_method_overrides(node: ast.ClassDef, method_name: str) -> bool:
    """Check if a method is overridden and returns True."""
    method = find_method(node, method_name)
    return method_returns_true(method)


def analyze_backend(backend_name: str, class_path: str) -> dict[str, Any] | None:
    """Analyze a backend class and extract feature information."""
    file_path = get_file_from_class_path(class_path)
    if file_path is None:
        return None

    try:
        tree = ast.parse(file_path.read_text())
    except Exception as e:
        print(f"  Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return None

    class_name = class_path.rsplit(".", 1)[1]
    class_node = find_class_in_ast(tree, class_name)
    if class_node is None:
        return None

    # Check if this is an MLA backend by parent class or naming
    parent = None
    if class_node.bases:
        base = class_node.bases[0]
        parent = base.id if isinstance(base, ast.Name) else None
    mla_parents = {"MLACommonBackend", "FlashMLABackend", "FlashMLASparseBackend"}
    is_mla_backend = (
        parent in mla_parents
        or ".mla." in class_path.lower()
        or "_mla" in backend_name.lower()
    )

    # Determine compute capability - use N/A for non-CUDA backends
    is_non_cuda = backend_name.startswith(("CPU_", "ROCM_"))
    compute_cap = "N/A" if is_non_cuda else parse_compute_capability(class_node)

    return {
        "name": backend_name,
        "dtypes": parse_supported_dtypes(class_node),
        "kv_cache_dtypes": parse_kv_cache_dtypes(class_node),
        "block_sizes": parse_block_sizes(class_node),
        "head_sizes": parse_head_sizes(class_node),
        "attn_types": parse_attention_types(class_node),
        "compute_capability": compute_cap,
        "is_mla": is_mla_backend or check_method_overrides(class_node, "is_mla"),
        "supports_sink": check_method_overrides(class_node, "supports_sink"),
        "is_sparse": check_method_overrides(class_node, "is_sparse"),
        "supports_mm_prefix": check_method_overrides(class_node, "supports_mm_prefix"),
    }


def add_literal_quotes(value: str) -> str:
    """Add literal backticks around all comma-separated items in a string."""
    items = [item.strip() for item in value.split(",")]
    quoted_items = [f"`{item}`" for item in items]
    return ", ".join(quoted_items)


def bool_to_emoji(value: bool) -> str:
    """Convert a boolean to a checkmark or X emoji."""
    return "✅" if value else "❌"


def generate_markdown_table(
    backends: list[dict[str, Any]], title: str, is_mla_table: bool = False
) -> str:
    """Generate a markdown table from backend info.

    Args:
        backends: List of backend info dictionaries.
        title: Table title.
        is_mla_table: If True, include MLA and Sparse columns (for MLA table).
                      If False, exclude them (for standard attention table).
    """
    if not backends:
        return f"## {title}\n\nNo backends found.\n"

    # Check if any backend has a version (for FA2/FA3 split)
    has_versions = any(b.get("version") for b in backends)

    if is_mla_table:
        header = (
            "| Backend | Dtypes | KV Dtypes | Block Sizes | Head Sizes "
            "| Sink | Sparse | MM Prefix | Attention Types | Compute Cap. |"
        )
        separator = (
            "|---------|--------|-----------|-------------|------------"
            "|------|--------|-----------|-----------------|--------------|"
        )
    elif has_versions:
        header = (
            "| Backend | Version | Dtypes | KV Dtypes | Block Sizes "
            "| Head Sizes | Sink | MM Prefix | Attention Types | Compute Cap. |"
        )
        separator = (
            "|---------|---------|--------|-----------|-------------"
            "|------------|------|-----------|-----------------|--------------|"
        )
    else:
        header = (
            "| Backend | Dtypes | KV Dtypes | Block Sizes | Head Sizes "
            "| Sink | MM Prefix | Attention Types | Compute Cap. |"
        )
        separator = (
            "|---------|--------|-----------|-------------|------------"
            "|------|-----------|-----------------|--------------|"
        )
    lines = [f"## {title}", "", header, separator]

    def sort_key(x: dict[str, Any]) -> tuple[str, int]:
        """Sort key that keeps parent/child rows together in order."""
        return (x.get("_sort_key", x["name"]), x.get("_sort_order", 0))

    for info in sorted(backends, key=sort_key):
        if is_mla_table:
            row = "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                info["name"],
                info["dtypes"],
                add_literal_quotes(info["kv_cache_dtypes"]),
                info["block_sizes"],
                info["head_sizes"],
                bool_to_emoji(info["supports_sink"]),
                bool_to_emoji(info["is_sparse"]),
                bool_to_emoji(info["supports_mm_prefix"]),
                info["attn_types"],
                info["compute_capability"],
            )
        elif has_versions:
            row = "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                info["name"],
                info.get("version", ""),
                info["dtypes"],
                add_literal_quotes(info["kv_cache_dtypes"]),
                info["block_sizes"],
                info["head_sizes"],
                bool_to_emoji(info["supports_sink"]),
                bool_to_emoji(info["supports_mm_prefix"]),
                info["attn_types"],
                info["compute_capability"],
            )
        else:
            row = "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                info["name"],
                info["dtypes"],
                add_literal_quotes(info["kv_cache_dtypes"]),
                info["block_sizes"],
                info["head_sizes"],
                bool_to_emoji(info["supports_sink"]),
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
                _extract_priorities(stmt.body, priorities, "mla")
                if stmt.orelse:
                    _extract_priorities(stmt.orelse, priorities, "standard")
            else:
                _extract_priorities([stmt], priorities, "standard")

    return priorities


def _get_backends_from_return(stmts: list) -> list[str]:
    """Extract backend names from return statements in a list of statements."""
    for stmt in stmts:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.List):
            return [e.attr for e in stmt.value.elts if isinstance(e, ast.Attribute)]
    return []


def _is_sm100_check(test: ast.expr) -> bool:
    """Check if test is `something.major == 10`."""
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Attribute)
        and test.left.attr == "major"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == 10
    )


def _extract_priorities(body: list, priorities: dict[str, list[str]], prefix: str):
    """Extract priority lists from if/else statement body."""
    for stmt in body:
        if isinstance(stmt, ast.If):
            is_sm100 = _is_sm100_check(stmt.test)
            if_key = f"{prefix}_sm100" if is_sm100 else f"{prefix}_default"
            else_key = f"{prefix}_default" if is_sm100 else f"{prefix}_sm100"

            if backends := _get_backends_from_return(stmt.body):
                priorities[if_key] = backends
            if backends := _get_backends_from_return(stmt.orelse):
                priorities[else_key] = backends

        elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.List):
            backends = [e.attr for e in stmt.value.elts if isinstance(e, ast.Attribute)]
            priorities[f"{prefix}_default"] = backends


def generate_usage_section() -> str:
    """Generate the usage documentation section."""
    return """## Setting the Attention Backend

### Command Line

There are two ways to specify the backend from the command line:

**Option 1: Using `--attention-backend` (simple)**

```bash
vllm serve <model> --attention-backend FLASH_ATTN
```

**Option 2: Using `--attention-config.backend` / `-ac.backend` (structured config)**

```bash
# Dot notation
vllm serve <model> --attention-config.backend FLASH_ATTN
vllm serve <model> -ac.backend FLASH_ATTN

# JSON format
vllm serve <model> --attention-config '{"backend": "FLASH_ATTN"}'
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
    model="Qwen/Qwen3-0.6B",
    attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN),
)

# Method 2: Using attention_backend parameter with string
llm = LLM(
    model="Qwen/Qwen3-0.6B",
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


def _priority_table(title: str, backends: list[str]) -> list[str]:
    """Generate a priority table for a list of backends."""
    return [
        f"**{title}:**",
        "",
        "| Priority | Backend |",
        "|----------|---------|",
        *[f"| {i} | `{b}` |" for i, b in enumerate(backends, 1)],
        "",
    ]


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
        "### Standard Attention (MHA, MQA, GQA)",
        "",
    ]

    sm100 = "Blackwell (SM 10.x)"
    ampere = "Ampere/Hopper (SM 8.x-9.x)"

    if "standard_sm100" in priorities:
        lines.extend(_priority_table(sm100, priorities["standard_sm100"]))
    if "standard_default" in priorities:
        lines.extend(_priority_table(ampere, priorities["standard_default"]))

    lines.extend(["### MLA Attention (DeepSeek-style)", ""])

    if "mla_sm100" in priorities:
        lines.extend(_priority_table(sm100, priorities["mla_sm100"]))
    if "mla_default" in priorities:
        lines.extend(_priority_table(ampere, priorities["mla_default"]))

    lines.append(
        "> **Note:** ROCm and CPU platforms have their own selection logic. "
        "See the platform-specific documentation for details."
    )
    lines.append("")

    return "\n".join(lines)


def generate_mla_section(
    prefill_backends: list[dict[str, Any]], decode_backends: list[dict[str, Any]]
) -> str:
    """Generate the complete MLA section with prefill and decode tables."""
    lines = [
        "## MLA (Multi-head Latent Attention) Backends",
        "",
        "MLA uses separate backends for prefill and decode phases.",
        "",
        "### Prefill Backends",
        "",
        "The prefill backend is selected at runtime based on hardware and",
        "configuration.",
        "",
        "| Backend | Description | Compute Cap. | Enable | Disable | Notes |",
        "|---------|-------------|--------------|--------|---------|-------|",
    ]

    for backend in prefill_backends:
        row = "| {} | {} | {} | {} | {} | {} |".format(
            backend["name"],
            backend["description"],
            backend["compute_capability"],
            backend["enable"],
            backend["disable"],
            backend.get("notes", ""),
        )
        lines.append(row)

    lines.extend(
        [
            "",
            "> **‡** TRT-LLM Ragged is the default on Blackwell (SM100).",
            "> On other GPUs, FlashAttention is used as the default.",
            "",
            "### Decode Backends",
            "",
        ]
    )

    # Generate decode backends table
    header = (
        "| Backend | Dtypes | KV Dtypes | Block Sizes | Head Sizes "
        "| Sink | Sparse | MM Prefix | Attention Types | Compute Cap. |"
    )
    separator = (
        "|---------|--------|-----------|-------------|------------"
        "|------|--------|-----------|-----------------|--------------|"
    )
    lines.extend([header, separator])

    def sort_key(x: dict[str, Any]) -> tuple[str, int]:
        return (x.get("_sort_key", x["name"]), x.get("_sort_order", 0))

    for info in sorted(decode_backends, key=sort_key):
        row = "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            info["name"],
            info["dtypes"],
            add_literal_quotes(info["kv_cache_dtypes"]),
            info["block_sizes"],
            info["head_sizes"],
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
| **KV Dtypes** | Supported KV cache data types (`auto`, `fp8`, `fp8_e4m3`, etc.) |
| **Block Sizes** | Supported KV cache block sizes (%N means multiples of N) |
| **Head Sizes** | Supported attention head sizes |
| **Sink** | Attention sink support (for StreamingLLM) |
| **Sparse** | Sparse attention support (MLA only) |
| **MM Prefix** | Multimodal prefix full attention support |
| **Attention Types** | Supported attention patterns (Decoder, Encoder, Enc-Dec) |
| **Compute Cap.** | Required CUDA compute capability (N/A for non-CUDA backends) |

**Symbols:** ✅ = Supported, ❌ = Not supported
"""


def generate_docs() -> str:
    """Generate the complete documentation."""
    attention_backends_map = parse_registry()

    # Parse priority lists from cuda.py
    priorities = parse_cuda_priority_lists()

    # Parse FlashAttention FA2/FA3 feature differences
    fa_features = parse_flash_attn_features()

    # Parse FlashInfer TRTLLM feature differences (native vs TRTLLM on Blackwell)
    fi_features = parse_flashinfer_trtllm_features()

    # Parse MLA prefill backends
    mla_prefill_backends = parse_mla_prefill_backends()

    # Collect backend info
    all_backends = []
    for backend_name, class_path in attention_backends_map.items():
        if backend_name in ("CUSTOM", "TORCH_SDPA"):
            continue
        info = analyze_backend(backend_name, class_path)
        if info:
            all_backends.append(info)

    # Expand FLASH_ATTN into FA2, FA3, and FA4 variants with different capabilities
    if fa_features:
        expanded_backends = []
        for backend in all_backends:
            if backend["name"] == "FLASH_ATTN":
                # Create FA2 entry (keeps base backend's compute_capability)
                fa2 = backend.copy()
                fa2["name"] = "FLASH_ATTN"
                fa2["version"] = "FA2*"
                fa2["_sort_key"] = "FLASH_ATTN"
                fa2["_sort_order"] = 0
                fa2["supports_sink"] = fa_features["fa2"]["supports_sink"]

                # Create FA3 entry (uses parsed compute_capability from fa_utils)
                fa3 = backend.copy()
                fa3["name"] = "FLASH_ATTN"
                fa3["version"] = "FA3*"
                fa3["_sort_key"] = "FLASH_ATTN"
                fa3["_sort_order"] = 1
                if fa_features["fa3"]["compute_capability"]:
                    fa3["compute_capability"] = fa_features["fa3"]["compute_capability"]
                fa3["supports_sink"] = fa_features["fa3"]["supports_sink"]
                if fa_features["fa3"]["supports_fp8"]:
                    # Add fp8 dtypes to the base backend's kv_cache_dtypes
                    base_dtypes = backend["kv_cache_dtypes"].split(", ")
                    fp8_dtypes = ["fp8", "fp8_e4m3", "fp8_e5m2"]
                    new_dtypes = [d for d in fp8_dtypes if d not in base_dtypes]
                    fa3["kv_cache_dtypes"] = ", ".join(base_dtypes + new_dtypes)

                # Create FA4 entry
                fa4 = backend.copy()
                fa4["name"] = "FLASH_ATTN"
                fa4["version"] = "FA4*"
                fa4["_sort_key"] = "FLASH_ATTN"
                fa4["_sort_order"] = 2
                if fa_features["fa4"]["compute_capability"]:
                    fa4["compute_capability"] = fa_features["fa4"]["compute_capability"]
                fa4["supports_sink"] = fa_features["fa4"]["supports_sink"]
                # FA4 does not support FP8 KV cache (same as FA2)

                # Add FA2, FA3, then FA4
                expanded_backends.append(fa2)
                expanded_backends.append(fa3)
                expanded_backends.append(fa4)
            else:
                backend["_sort_key"] = backend["name"]
                backend["_sort_order"] = 0
                backend["version"] = ""  # No version for other backends
                expanded_backends.append(backend)
        all_backends = expanded_backends

    # Expand FLASHINFER into native and TRTLLM variants
    if fi_features:
        expanded_backends = []
        for backend in all_backends:
            if backend["name"] == "FLASHINFER":
                # Parse original compute capability to get min CC
                orig_cap = backend["compute_capability"]
                parts = orig_cap.replace(".x", "").split("-")
                min_cc = parts[0] if parts else "7"
                trtllm_cc = fi_features["trtllm"]["compute_capability"]

                # Create native entry (pre-Blackwell GPUs)
                native = backend.copy()
                native["name"] = "FLASHINFER"
                native["version"] = "Native†"
                native["_sort_key"] = "FLASHINFER"
                native["_sort_order"] = 0
                native["supports_sink"] = fi_features["native"]["supports_sink"]
                # Native FlashInfer is used on GPUs before SM100 (Blackwell)
                native["compute_capability"] = f"{min_cc}.x-9.x"

                # Create TRTLLM entry
                trtllm = backend.copy()
                trtllm["name"] = "FLASHINFER"
                trtllm["version"] = "TRTLLM†"
                trtllm["_sort_key"] = "FLASHINFER"
                trtllm["_sort_order"] = 1
                trtllm["compute_capability"] = trtllm_cc
                trtllm["supports_sink"] = fi_features["trtllm"]["supports_sink"]

                expanded_backends.append(native)
                expanded_backends.append(trtllm)
            else:
                expanded_backends.append(backend)
        all_backends = expanded_backends

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
    standard_title = "Standard Attention (MHA, MQA, GQA) Backends"
    doc_lines.append(
        generate_markdown_table(non_mla_backends, standard_title, is_mla_table=False)
    )
    # Add footnotes for version/variant distinctions (in table order)
    footnotes = []
    if fi_features:
        footnotes.append(
            "> **†** FlashInfer uses TRTLLM attention on Blackwell (SM100), which "
            "supports sinks. Disable via `--attention-config.use_trtllm_attention=0`."
        )
    if fa_features:
        footnotes.append(
            "> **\\*** Specify the FlashAttention version via "
            "`--attention-config.flash_attn_version=2`, `3`, or `4`. "
            "Default is FA4 on SM100+ (Blackwell), FA3 on SM90 (Hopper), "
            "FA2 otherwise."
        )
    if footnotes:
        doc_lines.append("\n>\n".join(footnotes) + "\n")

    # Add MLA section with prefill and decode backends
    doc_lines.append(generate_mla_section(mla_prefill_backends, mla_backends))

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
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to check (passed by pre-commit). If none are relevant, skip.",
    )
    args = parser.parse_args()

    if args.files and not any(is_relevant_file(f) for f in args.files):
        sys.exit(0)

    output_path = Path(args.output)
    new_content = generate_docs()

    if args.check:
        needs_update = (
            not output_path.exists() or output_path.read_text() != new_content
        )
        if needs_update:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(new_content)
            print(f"🔄 Regenerated: {output_path}")
            sys.exit(1)
        print(f"✅ Up to date: {output_path}")
        sys.exit(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(new_content)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
