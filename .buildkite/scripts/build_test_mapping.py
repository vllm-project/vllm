#!/usr/bin/env python3
"""Build source→test mapping from static import analysis.

Resolves three layers of dependencies:
  1. Direct vllm imports in test files
  2. Conftest fixtures (pytest scoping: directory + parents)
  3. Transitive imports through tests/ helper modules
     (e.g., test_chat.py → tests.utils → vllm.entrypoints)

Usage:
    # Full directory-level mapping (all source dirs → all test dirs)
    python build_test_mapping.py
    python build_test_mapping.py --detail       # per-file breakdown
    python build_test_mapping.py --output map.md # write to file

    # Pre-filtered file-level lookup (only candidates for changed files)
    python build_test_mapping.py \&&
        --files "vllm/config/model_config.py,vllm/utils/misc.py"
"""

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

TESTS_ROOT = Path("tests")

# Maximum depth for transitive import resolution to avoid cycles
MAX_TRANSITIVE_DEPTH = 10


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------


def get_all_imports(filepath: Path) -> tuple[set[str], set[str]]:
    """Extract all imports from a Python file.

    Returns:
        (vllm_imports, other_imports) where other_imports includes
        tests.* and relative imports resolved to absolute paths.
    """
    try:
        tree = ast.parse(filepath.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return set(), set()

    vllm_imports: set[str] = set()
    other_imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("vllm"):
                    vllm_imports.add(alias.name)
                else:
                    other_imports.add(alias.name)

        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import(node, filepath)
            if module is None:
                continue
            if module.startswith("vllm"):
                vllm_imports.add(module)
            else:
                other_imports.add(module)

    return vllm_imports, other_imports


def _resolve_import(node: ast.ImportFrom, filepath: Path) -> str | None:
    """Resolve an ImportFrom node to an absolute module path.

    Handles:
      - from vllm.foo import bar      → "vllm.foo"
      - from tests.utils import X     → "tests.utils"
      - from . import utils           → relative to filepath's package
      - from ..utils import X         → relative to filepath's parent package
    """
    if node.module and node.level == 0:
        # Absolute import
        return node.module

    if node.level and node.level > 0:
        # Relative import — resolve against file's location
        parts = list(filepath.parent.parts)

        # Go up `level - 1` directories (level=1 means current package)
        levels_up = node.level - 1
        if levels_up > 0:
            if levels_up >= len(parts):
                return None  # relative import goes above root
            parts = parts[:-levels_up]

        base_module = ".".join(parts)
        if node.module:
            return f"{base_module}.{node.module}"
        else:
            # `from . import something` — the module is the package itself
            return base_module

    return node.module


def get_vllm_imports(filepath: Path) -> set[str]:
    """Extract only vllm.* imports from a file (convenience wrapper)."""
    vllm, _ = get_all_imports(filepath)
    return vllm


# ---------------------------------------------------------------------------
# Transitive import resolution through tests/ helper modules
# ---------------------------------------------------------------------------


def build_helper_module_index() -> dict[str, Path]:
    """Build an index of module_name → file_path for all non-test Python
    files under tests/ (helper modules, utils, etc.).

    Maps dotted module names like "tests.utils" → tests/utils.py
    and "tests.entrypoints.openai.utils" → tests/entrypoints/openai/utils.py
    """
    index: dict[str, Path] = {}

    for pyfile in TESTS_ROOT.rglob("*.py"):
        # Build dotted module name from path
        # tests/utils.py → tests.utils
        # tests/entrypoints/openai/utils.py → tests.entrypoints.openai.utils
        parts = list(pyfile.parts)
        if pyfile.name == "__init__.py":
            parts = parts[:-1]  # package init
        else:
            parts[-1] = pyfile.stem  # strip .py

        module_name = ".".join(parts)
        index[module_name] = pyfile

    return index


def resolve_transitive_vllm_imports(
    filepath: Path,
    helper_index: dict[str, Path],
    cache: dict[str, set[str]],
    depth: int = 0,
) -> set[str]:
    """Recursively resolve vllm imports through tests/ helper modules.

    For a file that imports from tests.utils, this follows the chain:
      test_file.py → tests.utils → tests.conftest → vllm.entrypoints

    Uses a cache to avoid re-processing files and a depth limit
    to prevent infinite loops from circular imports.
    """
    cache_key = str(filepath)
    if cache_key in cache:
        return cache[cache_key]

    if depth >= MAX_TRANSITIVE_DEPTH:
        return set()

    # Mark as in-progress (empty set) to handle circular imports
    cache[cache_key] = set()

    vllm_imports, other_imports = get_all_imports(filepath)

    # Follow imports that point to tests/ helper modules
    for imp in other_imports:
        if not imp.startswith("tests"):
            continue

        # Try to find this module in our index
        helper_path = _find_helper_module(imp, helper_index)
        if helper_path is None:
            continue

        # Recursively get vllm imports from the helper
        transitive = resolve_transitive_vllm_imports(
            helper_path, helper_index, cache, depth + 1
        )
        vllm_imports |= transitive

    cache[cache_key] = vllm_imports
    return vllm_imports


def _find_helper_module(
    module_name: str,
    helper_index: dict[str, Path],
) -> Path | None:
    """Find the file for a tests.* module name.

    Handles both:
      - tests.utils → tests/utils.py
      - tests.utils.SomeClass → tests/utils.py (over-specified import)
    """
    # Direct match
    if module_name in helper_index:
        return helper_index[module_name]

    # Try progressively shorter prefixes
    # e.g., tests.models.utils.SomeClass → tests.models.utils
    parts = module_name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in helper_index:
            return helper_index[prefix]

    return None


# ---------------------------------------------------------------------------
# Conftest / fixture resolution
# ---------------------------------------------------------------------------


def get_conftest_chain(test_file: Path) -> list[Path]:
    """Find all conftest.py files in scope for a test file.

    pytest resolution: conftest.py in the test's directory,
    then each parent directory up to the repo root.
    """
    conftests = []
    directory = test_file.parent

    while True:
        candidate = directory / "conftest.py"
        if candidate.exists():
            conftests.append(candidate)

        if directory == Path(".") or directory == directory.parent:
            break
        directory = directory.parent

    return conftests


def get_fixture_names(conftest_path: Path) -> set[str]:
    """Extract fixture names defined in a conftest.py."""
    try:
        tree = ast.parse(conftest_path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return set()

    fixtures = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if _is_pytest_fixture(decorator):
                    fixtures.add(node.name)
    return fixtures


def _is_pytest_fixture(decorator) -> bool:
    """Check if a decorator is @pytest.fixture or @pytest.fixture(...)."""
    if isinstance(decorator, ast.Attribute):
        return (
            isinstance(decorator.value, ast.Name)
            and decorator.value.id == "pytest"
            and decorator.attr == "fixture"
        )
    elif isinstance(decorator, ast.Call):
        return _is_pytest_fixture(decorator.func)
    return False


def get_used_fixtures(test_file: Path, available_fixtures: set[str]) -> set[str]:
    """Find which fixtures a test file actually uses (via function params)."""
    try:
        tree = ast.parse(test_file.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return set()

    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            for arg in node.args.args:
                if arg.arg in available_fixtures:
                    used.add(arg.arg)
    return used


# ---------------------------------------------------------------------------
# Main dependency builder
# ---------------------------------------------------------------------------


def build_test_dependencies(
    detail: bool = False,
) -> dict[str, set[str]]:
    """Build mapping: test_file → set of vllm modules it depends on.

    Resolves three layers:
      1. Direct vllm imports in the test file
      2. vllm imports from conftest fixtures the test uses
      3. Transitive vllm imports through tests/ helper modules
    """
    dependencies: dict[str, set[str]] = {}

    # Pre-compute conftest fixtures
    conftest_fixtures: dict[Path, set[str]] = {}
    for conftest in TESTS_ROOT.rglob("conftest.py"):
        conftest_fixtures[conftest] = get_fixture_names(conftest)
    root_conftest = Path("conftest.py")
    if root_conftest.exists():
        conftest_fixtures[root_conftest] = get_fixture_names(root_conftest)

    # Build helper module index for transitive resolution
    helper_index = build_helper_module_index()

    # Cache for transitive resolution (shared across all test files)
    transitive_cache: dict[str, set[str]] = {}

    test_files = sorted(TESTS_ROOT.rglob("test_*.py"))
    print(f"Found {len(test_files)} test files", file=sys.stderr)
    print(f"Found {len(conftest_fixtures)} conftest.py files", file=sys.stderr)
    print(f"Found {len(helper_index)} helper modules\n", file=sys.stderr)

    for test_file in test_files:
        # Layer 1: Direct vllm imports
        direct_imports = get_vllm_imports(test_file)

        # Layer 2: Conftest fixture inheritance
        conftest_chain = get_conftest_chain(test_file)

        all_available_fixtures: set[str] = set()
        for conftest in conftest_chain:
            all_available_fixtures |= conftest_fixtures.get(conftest, set())

        used_fixtures = get_used_fixtures(test_file, all_available_fixtures)

        conftest_imports: set[str] = set()
        inherited_from_conftest: list[str] = []
        for conftest in conftest_chain:
            defined = conftest_fixtures.get(conftest, set())
            overlap = used_fixtures & defined
            if overlap:
                # Direct vllm imports from conftest
                c_imports = get_vllm_imports(conftest)
                conftest_imports |= c_imports

                # Also resolve transitive imports from conftest
                c_transitive = resolve_transitive_vllm_imports(
                    conftest, helper_index, transitive_cache
                )
                conftest_imports |= c_transitive

                if detail and (c_imports or c_transitive):
                    inherited_from_conftest.append(
                        f"conftest {conftest} (fixtures: {', '.join(sorted(overlap))})"
                    )

        # Layer 3: Transitive imports through tests/ helpers
        transitive_imports = resolve_transitive_vllm_imports(
            test_file, helper_index, transitive_cache
        )

        all_imports = direct_imports | conftest_imports | transitive_imports

        if detail and all_imports:
            print(f"\n{test_file}:", file=sys.stderr)
            if direct_imports:
                print(f"  direct: {', '.join(sorted(direct_imports))}", file=sys.stderr)
            for line in inherited_from_conftest:
                print(f"  via {line}", file=sys.stderr)
            transitive_only = transitive_imports - direct_imports - conftest_imports
            if transitive_only:
                print(
                    f"  transitive: {', '.join(sorted(transitive_only))}",
                    file=sys.stderr,
                )

        if all_imports:
            dependencies[str(test_file)] = all_imports

    return dependencies


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def invert_mapping(
    deps: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Invert test→source into source_dir→test_dirs."""
    source_to_tests: dict[str, set[str]] = defaultdict(set)

    for test_file, modules in deps.items():
        test_dir = str(Path(test_file).parent) + "/"

        for module in modules:
            source_parts = module.replace(".", "/").split("/")
            if len(source_parts) >= 2:
                source_dir = "/".join(source_parts[:2]) + "/"
            else:
                source_dir = source_parts[0] + "/"

            source_to_tests[source_dir].add(test_dir)

    return source_to_tests


def invert_mapping_file_level(
    deps: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Invert test→source into source_module→test_files (file-level).

    Unlike invert_mapping(), this preserves full granularity on both sides:
      - Source key is the dotted module path (e.g., "vllm.config.model_config")
      - Test value is the individual test file path (e.g., "tests/test_config.py")
    """
    source_to_tests: dict[str, set[str]] = defaultdict(set)

    for test_file, modules in deps.items():
        for module in modules:
            source_to_tests[module].add(test_file)

    return source_to_tests


def lookup_changed_files(
    file_level_mapping: dict[str, set[str]],
    changed_files: list[str],
) -> dict[str, set[str]]:
    """Look up candidate tests for a set of changed files.

    For each changed Python file under vllm/, converts the file path to a
    module path and finds all test files that depend on it (at any level
    of the module hierarchy).

    Returns: changed_file → set of test files
    """
    results: dict[str, set[str]] = {}

    for filepath in changed_files:
        if not filepath.endswith(".py"):
            continue
        if not filepath.startswith("vllm/"):
            continue

        # Convert file path to module name:
        #   vllm/config/model_config.py → vllm.config.model_config
        #   vllm/utils/__init__.py      → vllm.utils
        module = filepath.replace("/", ".").removesuffix(".py")
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")

        # Collect tests that import this exact module OR any parent package.
        # e.g., changing vllm/config/model_config.py should match tests that
        # import "vllm.config.model_config" or "vllm.config" (package-level).
        candidate_tests: set[str] = set()

        for source_module, tests in file_level_mapping.items():
            # Match if one module is equal to or a sub-package of the
            # other.  The "." check avoids false positives where one name
            # is a prefix of an unrelated module (e.g. "vllm.config" must
            # not match "vllm.config_helper").
            if (
                source_module == module
                or source_module.startswith(module + ".")
                or module.startswith(source_module + ".")
            ):
                candidate_tests |= tests

        if candidate_tests:
            results[filepath] = candidate_tests

    return results


def format_candidate_table(candidates: dict[str, set[str]]) -> str:
    """Format pre-filtered candidates as a markdown table."""
    lines = []
    lines.append("| Changed source file | Candidate test files |")
    lines.append("|---|---|")
    for source_file in sorted(candidates):
        tests = ", ".join(f"`{t}`" for t in sorted(candidates[source_file]))
        lines.append(f"| `{source_file}` | {tests} |")
    return "\n".join(lines)


def format_markdown_table(mapping: dict[str, set[str]]) -> str:
    """Format the mapping as a markdown table."""
    lines = []
    lines.append("| Source path | Test directories |")
    lines.append("|---|---|")
    for source_dir in sorted(mapping):
        tests = ", ".join(f"`{t}`" for t in sorted(mapping[source_dir]))
        lines.append(f"| `{source_dir}` | {tests} |")
    return "\n".join(lines)


def print_stats(
    deps: dict[str, set[str]],
    mapping: dict[str, set[str]],
) -> None:
    """Print summary statistics."""
    total_test_files = len(list(TESTS_ROOT.rglob("test_*.py")))
    mapped_test_files = len(deps)

    print("\n--- Statistics ---", file=sys.stderr)
    print(f"Total test files:  {total_test_files}", file=sys.stderr)
    print(
        f"Mapped test files: {mapped_test_files} "
        f"({mapped_test_files / total_test_files:.0%})",
        file=sys.stderr,
    )
    print(
        f"Unmapped (no vllm imports): {total_test_files - mapped_test_files}",
        file=sys.stderr,
    )
    print(f"Source directories: {len(mapping)}", file=sys.stderr)

    # Find unmapped test files
    mapped_set = set(deps.keys())
    unmapped = []
    for test_file in sorted(TESTS_ROOT.rglob("test_*.py")):
        if str(test_file) not in mapped_set:
            unmapped.append(str(test_file))

    if unmapped:
        print(f"\nUnmapped test files ({len(unmapped)} total):", file=sys.stderr)
        for f in unmapped[:20]:
            print(f"  {f}", file=sys.stderr)
        if len(unmapped) > 20:
            print(f"  ... and {len(unmapped) - 20} more", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Build source→test mapping from static import analysis"
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show per-file breakdown of imports and conftest inheritance",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write markdown table to a file instead of stdout",
    )
    parser.add_argument(
        "--files",
        type=str,
        default=None,
        help="Comma-separated list of changed files. Outputs a pre-filtered "
        "file-level mapping containing only candidate tests for these files.",
    )
    args = parser.parse_args()

    print("Analyzing test dependencies...\n", file=sys.stderr)

    deps = build_test_dependencies(detail=args.detail)

    if args.files:
        # File-level pre-filtered mode: look up only the changed files
        changed = [f.strip() for f in args.files.split(",") if f.strip()]
        file_mapping = invert_mapping_file_level(deps)
        candidates = lookup_changed_files(file_mapping, changed)

        total_tests = set()
        for tests in candidates.values():
            total_tests |= tests
        print(
            f"Changed vllm files: {len(candidates)}, "
            f"candidate tests: {len(total_tests)}",
            file=sys.stderr,
        )

        table = format_candidate_table(candidates)

        if args.output:
            Path(args.output).write_text(table + "\n")
            print(f"\nMapping written to {args.output}", file=sys.stderr)
        else:
            print(table)
    else:
        # Full directory-level mapping mode (legacy)
        mapping = invert_mapping(deps)

        table = format_markdown_table(mapping)

        if args.output:
            Path(args.output).write_text(table + "\n")
            print(f"\nMapping written to {args.output}", file=sys.stderr)
        else:
            print(table)

        print_stats(deps, mapping)


if __name__ == "__main__":
    main()
