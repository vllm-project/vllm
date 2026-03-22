#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Automatic test dependency analysis for vLLM CI.

Analyzes Python imports in test files to determine which source modules
each test depends on. Used by the CI pipeline to selectively run only
tests affected by a code change.

Usage:
    # Generate the full dependency graph
    python tools/ci/generate_test_deps.py --generate

    # Compare auto-generated deps with current YAML declarations
    python tools/ci/generate_test_deps.py --compare

    # Given changed files, show which test steps should run
    python tools/ci/generate_test_deps.py --diff vllm/v1/attention/backends/flash_attn.py

    # Output the graph as JSON
    python tools/ci/generate_test_deps.py --generate --output deps.json
"""

import ast
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLLM_SRC = REPO_ROOT / "vllm"
TESTS_DIR = REPO_ROOT / "tests"
TEST_AREAS_DIR = REPO_ROOT / ".buildkite" / "test_areas"

# ---------------------------------------------------------------------------
# Stop modules: too ubiquitous to be useful as specific dependencies.
# Changes to these trigger broad testing via run_all_patterns instead.
# ---------------------------------------------------------------------------
# These are imported by nearly every test — too ubiquitous for selective deps.
# Changes to them should trigger broad testing via run_all_patterns.
STOP_MODULES = frozenset({
    "vllm.logger",
    "vllm.envs",
    "vllm.env_override",
    "vllm.version",
    "vllm.platforms",  # imported by 50%+ of tests via conftest chains
})

# Shared infrastructure files whose imports should NOT be attributed to
# individual tests. They pull in broad swaths of vllm (for fixtures/utilities
# used by many different test types). Changes to these files or their imports
# should trigger broader testing (run_all or dedicated CI steps).
SHARED_INFRA_FILES = frozenset({
    TESTS_DIR / "conftest.py",        # Main conftest — fixtures for all tests
    TESTS_DIR / "utils.py",           # Shared test utilities
    TESTS_DIR / "models" / "utils.py",  # Model test utilities
})

# ---------------------------------------------------------------------------
# Structural heuristics: fallback dependencies for tests where static import
# analysis finds nothing (e.g., subprocess-based tests, C extension tests).
# Maps test directory prefixes to source directory prefixes.
# ---------------------------------------------------------------------------
DIRECTORY_HEURISTICS: dict[str, list[str]] = {
    "tests/entrypoints/openai/": ["vllm/entrypoints/openai/", "vllm/v1/engine/"],
    "tests/entrypoints/": ["vllm/entrypoints/"],
    "tests/kernels/attention/": ["csrc/attention/", "vllm/v1/attention/",
                                  "vllm/_custom_ops.py"],
    "tests/kernels/quantization/": ["csrc/quantization/",
                                     "vllm/model_executor/layers/quantization/",
                                     "vllm/_custom_ops.py"],
    "tests/kernels/moe/": ["csrc/moe/", "vllm/model_executor/layers/fused_moe/",
                            "vllm/_custom_ops.py"],
    "tests/kernels/mamba/": ["csrc/mamba/", "vllm/_custom_ops.py"],
    "tests/kernels/core/": ["csrc/", "vllm/_custom_ops.py"],
    "tests/kernels/": ["csrc/", "vllm/_custom_ops.py"],
    "tests/cuda/": ["vllm/platforms/"],
    "tests/engine/": ["vllm/engine/", "vllm/v1/engine/"],
    "tests/benchmarks/": ["vllm/benchmarks/", "vllm/entrypoints/"],
    "tests/evals/": ["vllm/model_executor/layers/quantization/"],
    "tests/basic_correctness/": ["vllm/entrypoints/", "vllm/v1/engine/"],
    "tests/models/language/": ["vllm/model_executor/models/", "vllm/entrypoints/"],
    "tests/models/multimodal/": ["vllm/model_executor/models/", "vllm/multimodal/",
                                  "vllm/entrypoints/"],
    "tests/models/quantization/": ["vllm/model_executor/layers/quantization/",
                                    "vllm/entrypoints/"],
    "tests/weight_loading/": ["vllm/model_executor/model_loader/"],
    "tests/compile/": ["vllm/compilation/"],
}

# ---------------------------------------------------------------------------
# Parse vllm/__init__.py MODULE_ATTRS to resolve lazy re-exports.
# e.g. "LLM" -> ".entrypoints.llm:LLM" means `from vllm import LLM`
# actually depends on vllm.entrypoints.llm
# ---------------------------------------------------------------------------
def _parse_module_attrs() -> dict[str, str]:
    """Parse MODULE_ATTRS from vllm/__init__.py to resolve re-exports."""
    init_path = VLLM_SRC / "__init__.py"
    if not init_path.exists():
        return {}

    source = init_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "MODULE_ATTRS"
                and isinstance(node.value, ast.Dict)):
            attrs = {}
            for key, val in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and isinstance(val, ast.Constant):
                    # ".entrypoints.llm:LLM" -> "vllm.entrypoints.llm"
                    module_path = str(val.value).split(":")[0]
                    if module_path.startswith("."):
                        module_path = "vllm" + module_path
                    attrs[str(key.value)] = module_path
            return attrs
    return {}


MODULE_ATTRS = _parse_module_attrs()

# Reverse map: module paths that are re-exported from vllm.__init__
_INIT_REEXPORT_MODULES = set(MODULE_ATTRS.values())


# ---------------------------------------------------------------------------
# AST import extraction
# ---------------------------------------------------------------------------
def _resolve_relative_import(
    filepath: Path, level: int, module: Optional[str]
) -> Optional[str]:
    """Resolve a relative import to an absolute module name."""
    try:
        parts = list(filepath.resolve().relative_to(REPO_ROOT).parts)
    except ValueError:
        return None

    # Remove filename to get package parts
    package_parts = parts[:-1]

    # Go up `level - 1` directories (level=1 means current package)
    up = level - 1
    if up >= len(package_parts):
        return None
    if up > 0:
        package_parts = package_parts[:-up]

    base = ".".join(package_parts)
    if module:
        return f"{base}.{module}" if base else module
    return base or None


def extract_imports(filepath: Path) -> set[str]:
    """Extract vllm.* and tests.* module names imported by a Python file."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return set()

    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith(("vllm", "tests")):
                    imports.add(name)

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            level = node.level or 0

            if level > 0:
                # Relative import — resolve to absolute
                resolved = _resolve_relative_import(filepath, level, mod)
                if resolved and resolved.startswith(("vllm", "tests")):
                    imports.add(resolved)
            elif mod.startswith(("vllm", "tests")):
                imports.add(mod)

            # For `from vllm import LLM`, resolve LLM via MODULE_ATTRS
            if (mod == "vllm" or (level > 0 and not mod)) and node.names:
                for alias in node.names:
                    real_module = MODULE_ATTRS.get(alias.name)
                    if real_module:
                        imports.add(real_module)

    return imports


# ---------------------------------------------------------------------------
# Module name → source file/directory path resolution
# ---------------------------------------------------------------------------
def module_to_source_prefix(module_name: str) -> Optional[str]:
    """
    Map a Python module name to a source path prefix suitable for
    source_file_dependencies matching.

    Returns the parent directory of the resolved file (as a prefix string),
    or the file itself for top-level modules.

    Examples:
        vllm.v1.attention.backends.flash_attn -> vllm/v1/attention/backends/
        vllm.lora.model_manager              -> vllm/lora/
        vllm.sampling_params                  -> vllm/sampling_params.py
        vllm.v1.engine                        -> vllm/v1/engine/
    """
    parts = module_name.split(".")
    rel = os.path.join(*parts)

    # Check if it's a package (directory with __init__.py)
    pkg_path = REPO_ROOT / rel
    if pkg_path.is_dir() and (pkg_path / "__init__.py").exists():
        return rel + "/"

    # Check if it's a module file
    file_path = REPO_ROOT / (rel + ".py")
    if file_path.exists():
        # For files directly under vllm/, return the file itself
        # (e.g., vllm/sampling_params.py)
        # For deeper files, return the parent directory
        # (e.g., vllm/lora/ for vllm/lora/model_manager.py)
        parent = str(Path(rel).parent)
        if parent == parts[0]:
            # Top-level module (e.g., vllm/sampling_params.py)
            return rel + ".py"
        else:
            return parent + "/"

    # Check if it's a directory without __init__.py (namespace package)
    if pkg_path.is_dir():
        return rel + "/"

    # Can't resolve — may be a submodule reference
    # Try parent module
    if len(parts) > 2:
        parent_module = ".".join(parts[:-1])
        return module_to_source_prefix(parent_module)

    return None


# ---------------------------------------------------------------------------
# Conftest chain resolution
# ---------------------------------------------------------------------------
def get_conftest_chain(test_file: Path) -> list[Path]:
    """Get all conftest.py files from tests/ root down to the test file's dir."""
    chain = []
    current = test_file.parent
    while current >= TESTS_DIR and current >= REPO_ROOT:
        conftest = current / "conftest.py"
        if conftest.exists():
            chain.append(conftest)
        if current == TESTS_DIR:
            break
        current = current.parent
    return chain


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def analyze_test_file(test_file: Path) -> set[str]:
    """
    Analyze a test file and return its vllm source dependency prefixes.

    Traces:
      1. Direct imports in the test file
      2. Imports from subdirectory conftest.py files (NOT root conftest.py)
      3. One level of test utility imports (excluding shared infra files)

    Shared infrastructure (root conftest.py, tests/utils.py) is excluded
    because it imports broadly from vllm for fixtures/utilities used by all
    test types. Including those would make every test depend on everything.
    """
    all_imports: set[str] = set()

    # 1. Direct imports from the test file itself
    all_imports.update(extract_imports(test_file))

    # 2. Subdirectory conftest chain (skip shared infrastructure)
    for conftest in get_conftest_chain(test_file):
        if conftest.resolve() not in {f.resolve() for f in SHARED_INFRA_FILES}:
            all_imports.update(extract_imports(conftest))

    # 3. Test utility imports (one level deep, skip shared infra)
    test_util_modules = {m for m in all_imports if m.startswith("tests.")}
    for util_mod in test_util_modules:
        parts = util_mod.split(".")
        rel = os.path.join(*parts)
        # Try as file
        util_path = REPO_ROOT / (rel + ".py")
        if util_path.exists():
            if util_path.resolve() not in {f.resolve()
                                            for f in SHARED_INFRA_FILES}:
                all_imports.update(extract_imports(util_path))
        # Try as package __init__
        init_path = REPO_ROOT / rel / "__init__.py"
        if init_path.exists():
            if init_path.resolve() not in {f.resolve()
                                            for f in SHARED_INFRA_FILES}:
                all_imports.update(extract_imports(init_path))

    # Resolve vllm imports to source prefixes
    vllm_imports = {m for m in all_imports if m.startswith("vllm")}
    source_deps: set[str] = set()

    for mod in vllm_imports:
        # Skip bare "vllm" package (re-exports handled via MODULE_ATTRS)
        if mod == "vllm":
            continue
        # Skip stop modules (too ubiquitous to be useful)
        if any(mod == s or mod.startswith(s + ".") for s in STOP_MODULES):
            continue

        prefix = module_to_source_prefix(mod)
        if prefix:
            source_deps.add(prefix)

    # Fallback: if no vllm deps found, apply directory heuristics.
    # This catches subprocess-based tests, C extension tests, etc.
    if not source_deps:
        rel_test = str(test_file.relative_to(REPO_ROOT))
        for dir_prefix, fallback_deps in DIRECTORY_HEURISTICS.items():
            if rel_test.startswith(dir_prefix):
                source_deps.update(fallback_deps)
                break

    return source_deps


def build_graph() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Build bidirectional dependency graph.

    Returns:
        source_to_tests: {source_prefix: [test_file_paths]}
        test_to_sources: {test_file_path: [source_prefixes]}
    """
    source_to_tests: dict[str, set[str]] = defaultdict(set)
    test_to_sources: dict[str, list[str]] = {}

    test_files = sorted(TESTS_DIR.rglob("test_*.py"))
    total = len(test_files)

    for i, test_file in enumerate(test_files):
        rel_test = str(test_file.relative_to(REPO_ROOT))
        deps = analyze_test_file(test_file)

        test_to_sources[rel_test] = sorted(deps)
        for dep in deps:
            source_to_tests[dep].add(rel_test)

        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"  Analyzed {i + 1}/{total} test files", file=sys.stderr)

    # Add csrc/ → test mappings. csrc/ changes are mostly handled by
    # run_all_patterns, but excluded paths (csrc/cpu/, csrc/rocm/) are not.
    # These static mappings ensure proper coverage for those.
    CSRC_TEST_MAPPINGS = {
        "csrc/attention/": [
            "tests/kernels/attention/", "tests/v1/attention/"],
        "csrc/quantization/": [
            "tests/kernels/quantization/", "tests/quantization/"],
        "csrc/moe/": ["tests/kernels/moe/"],
        "csrc/mamba/": ["tests/kernels/mamba/"],
        "csrc/quantization/cutlass_w8a8/moe/": ["tests/kernels/moe/"],
    }
    for csrc_prefix, test_dirs in CSRC_TEST_MAPPINGS.items():
        for test_dir in test_dirs:
            matching = {t for t in test_to_sources if t.startswith(test_dir)}
            if matching:
                source_to_tests[csrc_prefix].update(matching)

    source_to_tests_sorted = {
        k: sorted(v) for k, v in sorted(source_to_tests.items())
    }
    return source_to_tests_sorted, test_to_sources


# ---------------------------------------------------------------------------
# Test area YAML comparison
# ---------------------------------------------------------------------------
def _extract_test_paths_from_commands(commands: list[str]) -> set[str]:
    """
    Extract test file/directory paths from pytest commands.
    e.g. "pytest -v -s v1/attention" -> {"tests/v1/attention"}
         "pytest -v -s lora/test_foo.py" -> {"tests/lora/test_foo.py"}
    """
    paths = set()
    if not commands:
        return paths

    for cmd in commands:
        # Find pytest invocations
        # Match paths after pytest and its flags
        tokens = cmd.split()
        capture = False
        for token in tokens:
            if token in ("pytest", "python3", "bash", "torchrun",
                         "find", "cd", "export", "ENFORCE_EAGER=1"):
                capture = token == "pytest"
                continue
            if capture:
                # Skip flags
                if token.startswith("-") or token.startswith("$"):
                    if token in ("-k", "-m", "--ignore", "--shard-id",
                                 "--num-shards", "--config-list-file",
                                 "--tp-size"):
                        capture = False  # next token is flag value
                        # Re-enable after consuming value
                    continue
                # This looks like a path
                if ("/" in token or token.endswith(".py")
                        or token.startswith("test_")):
                    # Normalize: commands often run from tests/ working dir
                    path = token.rstrip(";")
                    if not path.startswith("tests/"):
                        path = "tests/" + path
                    paths.add(path)
    return paths


def _find_matching_test_files(
    test_paths: set[str], all_test_files: set[str]
) -> set[str]:
    """Find test files that match the given paths (prefix match)."""
    matches = set()
    for test_path in test_paths:
        for test_file in all_test_files:
            if test_file.startswith(test_path) or test_file == test_path:
                matches.add(test_file)
    return matches


def _reduce_prefixes(deps: list[str], threshold: int = 3) -> list[str]:
    """
    Reduce a list of source dependency paths to common prefixes.
    If `threshold` or more files share a directory, collapse to that directory.
    """
    dir_counts: dict[str, list[str]] = defaultdict(list)
    dirs_already: list[str] = []

    for dep in deps:
        if dep.endswith("/"):
            dirs_already.append(dep)
        else:
            parent = str(Path(dep).parent) + "/"
            dir_counts[parent].append(dep)

    result = set(dirs_already)
    for parent, files in dir_counts.items():
        if len(files) >= threshold:
            result.add(parent)
        else:
            result.update(files)

    # Remove redundant entries: if vllm/lora/ is present, remove vllm/lora/foo.py
    final = set()
    sorted_result = sorted(result)
    for entry in sorted_result:
        if not any(entry != other and entry.startswith(other)
                   for other in sorted_result if other.endswith("/")):
            final.add(entry)

    return sorted(final)


def compare_with_yaml(test_to_sources: dict[str, list[str]]) -> list[dict]:
    """
    Compare auto-generated dependencies with current YAML declarations.
    Returns a list of comparison records per step.
    """
    try:
        import yaml
    except ImportError:
        print("PyYAML required for --compare. Install with: pip install pyyaml",
              file=sys.stderr)
        sys.exit(1)

    if not TEST_AREAS_DIR.exists():
        print(f"Test areas directory not found: {TEST_AREAS_DIR}", file=sys.stderr)
        return []

    all_test_files = set(test_to_sources.keys())
    comparisons = []

    for yaml_file in sorted(TEST_AREAS_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        steps = data.get("steps", [])
        for step in steps:
            label = step.get("label", "unknown")
            commands = step.get("commands", [])
            current_deps = step.get("source_file_dependencies") or []

            # Extract test paths from commands
            test_paths = _extract_test_paths_from_commands(commands)

            # Find matching test files
            matching_tests = _find_matching_test_files(test_paths, all_test_files)

            # Union their source dependencies
            all_deps: set[str] = set()
            for test in matching_tests:
                all_deps.update(test_to_sources.get(test, []))

            recommended = _reduce_prefixes(sorted(all_deps))

            has_catch_all = any(
                d in ("vllm/", "vllm") for d in current_deps
            )

            comparisons.append({
                "file": yaml_file.name,
                "label": label,
                "current_deps": current_deps,
                "recommended_deps": recommended,
                "has_catch_all": has_catch_all,
                "test_files_found": len(matching_tests),
                "test_paths_parsed": sorted(test_paths),
            })

    return comparisons


# ---------------------------------------------------------------------------
# Diff-based step selection
# ---------------------------------------------------------------------------
def find_triggered_steps(
    changed_files: list[str],
    source_to_tests: dict[str, list[str]],
    test_to_sources: dict[str, list[str]],
) -> set[str]:
    """
    Given a list of changed files, find which test files should run.
    Uses prefix matching (same as source_file_dependencies).
    """
    triggered_tests: set[str] = set()

    for changed_file in changed_files:
        # Direct match: the changed file is itself a test
        if changed_file.startswith("tests/") and changed_file in test_to_sources:
            triggered_tests.add(changed_file)

        # Check source_to_tests: does any source prefix match?
        for source_prefix, tests in source_to_tests.items():
            if (source_prefix in changed_file
                    or changed_file.startswith(source_prefix)):
                triggered_tests.update(tests)

    return triggered_tests


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_comparison_report(comparisons: list[dict]):
    """Print a human-readable comparison report."""
    catch_all_count = sum(1 for c in comparisons if c["has_catch_all"])
    total = len(comparisons)

    print(f"\n{'=' * 78}")
    print(f"TEST DEPENDENCY COMPARISON REPORT")
    print(f"{'=' * 78}")
    print(f"Total steps analyzed: {total}")
    print(f"Steps using vllm/ catch-all: {catch_all_count} "
          f"({catch_all_count * 100 // total}%)")
    print()

    # Group by file
    by_file: dict[str, list[dict]] = defaultdict(list)
    for c in comparisons:
        by_file[c["file"]].append(c)

    for filename, steps in sorted(by_file.items()):
        catch_all_steps = [s for s in steps if s["has_catch_all"]]
        if not catch_all_steps:
            continue

        print(f"\n--- {filename} ({len(catch_all_steps)}/{len(steps)} "
              f"steps use catch-all) ---")
        for step in catch_all_steps:
            print(f"\n  Step: {step['label']}")
            print(f"  Test files found: {step['test_files_found']}")

            if step["current_deps"]:
                print(f"  Current deps ({len(step['current_deps'])}):")
                for d in step["current_deps"][:5]:
                    marker = " <<<" if d in ("vllm/", "vllm") else ""
                    print(f"    - {d}{marker}")
                if len(step["current_deps"]) > 5:
                    print(f"    ... and {len(step['current_deps']) - 5} more")

            if step["recommended_deps"]:
                print(f"  Recommended deps ({len(step['recommended_deps'])}):")
                for d in step["recommended_deps"][:10]:
                    print(f"    - {d}")
                if len(step["recommended_deps"]) > 10:
                    print(f"    ... and {len(step['recommended_deps']) - 10} more")
            else:
                print("  Recommended deps: (could not determine - "
                      "no test files matched)")


def print_diff_report(
    changed_files: list[str],
    triggered_tests: set[str],
    test_to_sources: dict[str, list[str]],
):
    """Print which tests would be triggered by the given file changes."""
    print(f"\nChanged files ({len(changed_files)}):")
    for f in changed_files:
        print(f"  {f}")

    print(f"\nTriggered tests ({len(triggered_tests)}):")
    for t in sorted(triggered_tests):
        print(f"  {t}")

    # Map to test area directories
    area_counts: dict[str, int] = defaultdict(int)
    for t in triggered_tests:
        parts = t.split("/")
        if len(parts) >= 3:
            area_counts["/".join(parts[:3])] += 1
        elif len(parts) >= 2:
            area_counts["/".join(parts[:2])] += 1

    print(f"\nAffected test areas:")
    for area, count in sorted(area_counts.items(), key=lambda x: -x[1]):
        print(f"  {area}: {count} tests")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Automatic test dependency analysis for vLLM CI")
    parser.add_argument("--generate", action="store_true",
                        help="Generate the dependency graph")
    parser.add_argument("--compare", action="store_true",
                        help="Compare auto-generated deps with current YAML")
    parser.add_argument("--diff", nargs="+", metavar="FILE",
                        help="Show which tests to run for given changed files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary statistics")

    args = parser.parse_args()

    if not any([args.generate, args.compare, args.diff, args.summary]):
        args.generate = True
        args.compare = True
        args.summary = True

    print("Building dependency graph...", file=sys.stderr)
    source_to_tests, test_to_sources = build_graph()
    print(f"Done. {len(test_to_sources)} test files, "
          f"{len(source_to_tests)} source prefixes.", file=sys.stderr)

    if args.output:
        graph = {
            "source_to_tests": source_to_tests,
            "test_to_sources": test_to_sources,
        }
        with open(args.output, "w") as f:
            json.dump(graph, f, indent=2, sort_keys=True)
        print(f"Graph written to {args.output}", file=sys.stderr)

    if args.summary:
        # Count test files per source prefix
        prefix_counts = sorted(
            ((k, len(v)) for k, v in source_to_tests.items()),
            key=lambda x: -x[1]
        )
        print(f"\n{'=' * 78}")
        print("DEPENDENCY GRAPH SUMMARY")
        print(f"{'=' * 78}")
        print(f"Total test files: {len(test_to_sources)}")
        print(f"Total source prefixes: {len(source_to_tests)}")
        print(f"\nTop 20 most-depended-on source prefixes:")
        for prefix, count in prefix_counts[:20]:
            print(f"  {count:4d} tests depend on  {prefix}")

        # Distribution of dependency counts per test
        dep_counts = [len(v) for v in test_to_sources.values()]
        if dep_counts:
            print(f"\nDeps per test file: "
                  f"min={min(dep_counts)}, "
                  f"max={max(dep_counts)}, "
                  f"avg={sum(dep_counts)/len(dep_counts):.1f}, "
                  f"median={sorted(dep_counts)[len(dep_counts)//2]}")

    if args.compare:
        comparisons = compare_with_yaml(test_to_sources)
        if comparisons:
            print_comparison_report(comparisons)

            if args.output:
                comp_path = args.output.replace(".json", "_comparison.json")
                with open(comp_path, "w") as f:
                    json.dump(comparisons, f, indent=2)
                print(f"\nComparison written to {comp_path}", file=sys.stderr)

    if args.diff:
        triggered = find_triggered_steps(
            args.diff, source_to_tests, test_to_sources)
        print_diff_report(args.diff, triggered, test_to_sources)


if __name__ == "__main__":
    main()
