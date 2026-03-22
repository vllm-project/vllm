# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pre-build script for Zensical documentation.

Performs these steps in order:
1. Run content generation hooks (examples, argparse, metrics) — these are
   existing MkDocs hooks that have on_startup() functions.
2. Handle announcement banner removal for tagged releases.
3. Generate API documentation pages with mkdocstrings ::: directives —
   replaces mkdocs-api-autonav.
4. Read docs/.nav.yml, expand glob patterns, and write the TOML nav array
   into zensical.toml — replaces mkdocs-awesome-nav.

Usage: python docs/scripts/zensical_gen_files.py
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# API docs configuration
# ---------------------------------------------------------------------------
TOP_MODULE = "vllm"
API_DOCS_DIR = Path("docs/api")
API_EXCLUDE_PATTERNS = [
    re.compile(r"^vllm\._"),        # Internal modules
    re.compile(r"^vllm\.third_party"),
    re.compile(r"^vllm\.vllm_flash_attn"),
    re.compile(r"^vllm\.grpc\..*_pb2"),  # Auto-generated protobuf files
]

# Files to exclude from nav generation (matches mkdocs exclude_docs)
NAV_EXCLUDE_SUFFIXES = (".inc.md", ".template.md")
NAV_EXCLUDE_DIRS = {"argparse", "generated"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def find_project_root() -> Path:
    """Find the project root by looking for zensical.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "zensical.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find zensical.toml")


# ---------------------------------------------------------------------------
# Step 1: Run existing MkDocs content generation hooks
# ---------------------------------------------------------------------------


def run_content_hooks(project_root: Path):
    """Run existing MkDocs hooks that generate content."""
    hooks_dir = project_root / "docs" / "mkdocs" / "hooks"

    # Run hooks that have on_startup() functions via subprocess to avoid
    # import side-effects (especially generate_argparse which mocks torch)
    hooks = [
        "generate_examples",
        "generate_argparse",
        "generate_metrics",
    ]

    for hook_name in hooks:
        hook_path = hooks_dir / f"{hook_name}.py"
        if not hook_path.exists():
            print(f"  Skipping {hook_name} (not found)")
            continue

        print(f"  Running {hook_name}...")
        result = subprocess.run(
            [
                sys.executable, "-c",
                f"import sys; sys.path.insert(0, {str(hooks_dir)!r}); "
                f"from {hook_name} import on_startup; "
                f"on_startup('build', False)"
            ],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: {hook_name} failed:")
            print(f"    {result.stderr}")
        else:
            print(f"  {hook_name} completed")


# ---------------------------------------------------------------------------
# Step 2: Handle announcement banner
# ---------------------------------------------------------------------------


def handle_announcement(project_root: Path):
    """Remove announcement banner for tagged releases (replicates
    remove_announcement.py hook)."""
    if os.getenv("READTHEDOCS_VERSION_TYPE") == "tag":
        announcement_path = (
            project_root / "docs" / "mkdocs" / "overrides" / "main.html"
        )
        if announcement_path.exists():
            os.remove(announcement_path)
            print("  Removed announcement banner for tagged release")
        else:
            print("  Announcement banner already removed")
    else:
        print("  Keeping announcement banner (not a tagged release)")


# ---------------------------------------------------------------------------
# Step 3: Generate API pages
# ---------------------------------------------------------------------------


def _get_module_path(pkg_dir: Path, module_root: Path) -> str:
    return ".".join(pkg_dir.relative_to(module_root.parent).parts)


def _should_exclude(module_path: str) -> bool:
    for pattern in API_EXCLUDE_PATTERNS:
        if pattern.search(module_path):
            return True
    return False


def _write_api_page(doc_file: Path, module_path: str, title: str):
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_text(
        f"---\ntitle: {title}\n---\n\n::: {module_path}\n",
        encoding="utf-8",
    )


def generate_api_pages(project_root: Path) -> int:
    """Generate API markdown files. Returns number of modules processed."""
    module_root = project_root / TOP_MODULE
    api_dir = project_root / API_DOCS_DIR / TOP_MODULE

    # Clean previously generated API subdirectories (preserve index.md
    # and README.md at the api/ level)
    if api_dir.exists():
        shutil.rmtree(api_dir)

    count = 0
    for init_file in sorted(module_root.rglob("__init__.py")):
        pkg_dir = init_file.parent
        pkg_module_path = _get_module_path(pkg_dir, module_root)

        if _should_exclude(pkg_module_path):
            continue

        rel_parts = pkg_dir.relative_to(module_root.parent).parts
        doc_dir = project_root / API_DOCS_DIR / Path(*rel_parts)
        _write_api_page(doc_dir / "index.md", pkg_module_path, rel_parts[-1])
        count += 1

        for py_file in sorted(pkg_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue

            module_name = py_file.stem
            module_path = f"{pkg_module_path}.{module_name}"

            if _should_exclude(module_path):
                continue

            _write_api_page(
                doc_dir / module_name / "index.md", module_path, module_name
            )
            count += 1

    return count


# ---------------------------------------------------------------------------
# Step 4: Generate TOML nav from .nav.yml
# ---------------------------------------------------------------------------


def _is_excluded_file(path: Path) -> bool:
    """Check if a file should be excluded from nav (snippets, templates)."""
    name = path.name
    return any(name.endswith(suffix) for suffix in NAV_EXCLUDE_SUFFIXES)


def _get_title_from_file(file_path: Path) -> str:
    """Extract title from markdown frontmatter or first heading."""
    try:
        content = file_path.read_text(encoding="utf-8")
        # Check YAML frontmatter
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                frontmatter = content[3:end]
                for line in frontmatter.strip().split("\n"):
                    if line.startswith("title:"):
                        return line[6:].strip().strip('"').strip("'")
        # Check first H1 heading
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip().rstrip("#").strip()
    except Exception:
        pass

    # Derive from filename
    name = file_path.stem
    if name in ("index", "README"):
        name = file_path.parent.name
    return name.replace("-", " ").replace("_", " ").title()


def _expand_glob_dir(
    base_dir: Path,
    docs_dir: Path,
    exclude_paths: set[str] | None = None,
) -> list:
    """Expand a directory into a nested nav structure.

    Subdirectories become sections, markdown files become leaf pages.
    index.md/README.md files are added as bare paths (section indexes).
    Files in exclude_paths are skipped to avoid duplicates.
    """
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    if exclude_paths is None:
        exclude_paths = set()

    entries = []
    index_files = []
    regular_files = []
    subdirs = []

    for item in sorted(base_dir.iterdir()):
        if item.is_file() and item.suffix == ".md":
            if _is_excluded_file(item):
                continue
            rel = str(item.relative_to(docs_dir))
            if rel in exclude_paths:
                continue
            if item.name in ("index.md", "README.md"):
                index_files.append(item)
            else:
                regular_files.append(item)
        elif (item.is_dir()
              and not item.name.startswith(".")
              and item.name not in NAV_EXCLUDE_DIRS):
            subdirs.append(item)

    for idx_file in index_files:
        entries.append(str(idx_file.relative_to(docs_dir)))

    for subdir in subdirs:
        children = _expand_glob_dir(subdir, docs_dir, exclude_paths)
        if children:
            if (subdir / "index.md").exists():
                title = _get_title_from_file(subdir / "index.md")
            elif (subdir / "README.md").exists():
                title = _get_title_from_file(subdir / "README.md")
            else:
                title = (
                    subdir.name.replace("-", " ").replace("_", " ").title()
                )
            entries.append({title: children})

    for f in regular_files:
        title = _get_title_from_file(f)
        entries.append({title: str(f.relative_to(docs_dir))})

    return entries


def _expand_glob_pattern(
    pattern: str,
    docs_dir: Path,
    exclude_paths: set[str] | None = None,
) -> list:
    """Expand a glob pattern (e.g., 'usage/*') into nav entries."""
    if exclude_paths is None:
        exclude_paths = set()

    # Strip trailing /* or * to get the directory
    dir_path = pattern.rstrip("/*").rstrip("*")
    base_dir = docs_dir / dir_path

    if not base_dir.exists() or not base_dir.is_dir():
        return []

    entries = []
    for item in sorted(base_dir.iterdir()):
        if item.is_file() and item.suffix == ".md":
            if _is_excluded_file(item):
                continue
            rel = str(item.relative_to(docs_dir))
            if rel in exclude_paths:
                continue
            if item.name in ("index.md", "README.md"):
                continue  # Skip index files in glob expansion
            title = _get_title_from_file(item)
            entries.append({title: rel})
        elif (item.is_dir()
              and not item.name.startswith(".")
              and item.name not in NAV_EXCLUDE_DIRS):
            children = _expand_glob_dir(item, docs_dir, exclude_paths)
            if children:
                if (item / "README.md").exists():
                    title = _get_title_from_file(item / "README.md")
                elif (item / "index.md").exists():
                    title = _get_title_from_file(item / "index.md")
                else:
                    title = (
                        item.name.replace("-", " ").replace("_", " ").title()
                    )
                entries.append({title: children})

    return entries


def _expand_glob_with_file_pattern(
    pattern: str,
    docs_dir: Path,
    exclude_paths: set[str] | None = None,
) -> list:
    """Expand a glob pattern with file matching (e.g., 'design/*plugin*.md')."""
    if exclude_paths is None:
        exclude_paths = set()

    entries = []
    for match in sorted(docs_dir.glob(pattern)):
        if match.is_file() and match.suffix == ".md":
            if _is_excluded_file(match):
                continue
            rel = str(match.relative_to(docs_dir))
            if rel in exclude_paths:
                continue
            title = _get_title_from_file(match)
            entries.append({title: rel})

    return entries


def _collect_explicit_paths(items: list) -> set[str]:
    """Collect all explicit (non-glob) file paths from a nav item list."""
    paths = set()
    for item in items:
        if isinstance(item, str) and "*" not in item and "://" not in item:
            paths.add(item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str) and "*" not in value and "://" not in value:
                    paths.add(value)
                elif isinstance(value, list):
                    paths.update(_collect_explicit_paths(value))
    return paths


def _is_external_link(value: str) -> bool:
    """Check if a value is an external URL."""
    return isinstance(value, str) and (
        value.startswith("http://") or value.startswith("https://")
    )


def _process_nav_item(
    item,
    docs_dir: Path,
    sibling_paths: set[str] | None = None,
) -> list:
    """Process a single YAML nav item. Returns a list of nav entries."""
    if sibling_paths is None:
        sibling_paths = set()

    if isinstance(item, str):
        if _is_external_link(item):
            return [item]
        if "*" in item:
            # Check if it's a file glob pattern (contains . before *)
            if "." in item.split("*")[0].split("/")[-1] or item.endswith(".md"):
                return _expand_glob_with_file_pattern(
                    item, docs_dir, sibling_paths
                )
            return _expand_glob_pattern(item, docs_dir, sibling_paths)
        # Check if it's a directory reference (no extension)
        path = docs_dir / item
        if path.is_dir():
            return _expand_glob_dir(path, docs_dir, sibling_paths)
        return [item]

    if isinstance(item, dict):
        results = []
        for title, value in item.items():
            # Handle awesome-nav glob: syntax
            if title == "glob" and isinstance(value, str):
                if "*" in value:
                    return _expand_glob_pattern(
                        value, docs_dir, sibling_paths
                    )
                path = docs_dir / value
                if path.is_dir():
                    return _expand_glob_dir(path, docs_dir, sibling_paths)
                return [value]

            if isinstance(value, str):
                if _is_external_link(value):
                    results.append({title: value})
                elif "*" in value:
                    if "." in value.split("*")[0].split("/")[-1] or value.endswith(".md"):
                        expanded = _expand_glob_with_file_pattern(
                            value, docs_dir, sibling_paths
                        )
                    else:
                        expanded = _expand_glob_pattern(
                            value, docs_dir, sibling_paths
                        )
                    if expanded:
                        results.append({title: expanded})
                else:
                    # Check if it's a directory reference
                    path = docs_dir / value
                    if path.is_dir():
                        children = _expand_glob_dir(
                            path, docs_dir, sibling_paths
                        )
                        if children:
                            results.append({title: children})
                    else:
                        results.append({title: value})
            elif isinstance(value, list):
                child_explicit = _collect_explicit_paths(value)
                children = []
                for child in value:
                    # Handle awesome-nav dict items with special keys
                    if isinstance(child, dict) and "glob" in child:
                        glob_val = child["glob"]
                        children.extend(
                            _process_nav_item(
                                {"glob": glob_val}, docs_dir, child_explicit
                            )
                        )
                    else:
                        children.extend(
                            _process_nav_item(
                                child, docs_dir, child_explicit
                            )
                        )
                results.append({title: children})
            elif value is None:
                results.append({title: []})
        return results

    return []


def _escape_toml(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _nav_to_toml(nav_items: list, indent: int = 4) -> str:
    prefix = " " * indent
    lines = []

    for item in nav_items:
        if isinstance(item, str):
            lines.append(f'{prefix}"{_escape_toml(item)}",')
        elif isinstance(item, dict):
            for title, value in item.items():
                t = _escape_toml(title)
                if isinstance(value, str):
                    v = _escape_toml(value)
                    lines.append(f'{prefix}{{ "{t}" = "{v}" }},')
                elif isinstance(value, list):
                    if not value:
                        lines.append(f'{prefix}{{ "{t}" = [] }},')
                    else:
                        lines.append(f'{prefix}{{ "{t}" = [')
                        lines.append(_nav_to_toml(value, indent + 4))
                        lines.append(f"{prefix}]}},")

    return "\n".join(lines)


def _replace_nav_block(config: str, new_nav_block: str) -> str:
    """Replace the nav = [...] block in TOML, handling nested brackets."""
    match = re.search(r"^nav\s*=\s*\[", config, re.MULTILINE)
    if not match:
        raise ValueError("Could not find 'nav = [' in zensical.toml")

    start = match.start()
    bracket_start = match.end() - 1

    depth = 0
    pos = bracket_start
    while pos < len(config):
        ch = config[pos]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = pos + 1
                break
        elif ch == "#":
            while pos < len(config) and config[pos] != "\n":
                pos += 1
        elif ch == '"':
            pos += 1
            while pos < len(config) and config[pos] != '"':
                if config[pos] == "\\":
                    pos += 1
                pos += 1
        pos += 1
    else:
        raise ValueError("Could not find closing ] for nav array")

    return config[:start] + new_nav_block + config[end:]


def generate_nav(project_root: Path) -> int:
    """Read .nav.yml and write the TOML nav into zensical.toml.

    Returns the number of top-level nav entries.
    """
    docs_dir = project_root / "docs"
    nav_file = docs_dir / ".nav.yml"
    config_file = project_root / "zensical.toml"

    with open(nav_file, encoding="utf-8") as f:
        nav_yaml = yaml.safe_load(f)

    raw_nav = nav_yaml.get("nav", [])

    top_level_paths = _collect_explicit_paths(raw_nav)
    nav_items = []
    for item in raw_nav:
        nav_items.extend(_process_nav_item(item, docs_dir, top_level_paths))

    toml_nav = _nav_to_toml(nav_items)
    nav_block = f"nav = [\n{toml_nav}\n]"

    config = config_file.read_text(encoding="utf-8")
    config = _replace_nav_block(config, nav_block)
    config_file.write_text(config, encoding="utf-8")

    return len(nav_items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def clean_nav(project_root: Path):
    """Reset nav in zensical.toml to an empty array."""
    config_file = project_root / "zensical.toml"
    config = config_file.read_text(encoding="utf-8")
    config = _replace_nav_block(config, "nav = []")
    config_file.write_text(config, encoding="utf-8")


def main():
    project_root = find_project_root()

    if "--clean" in sys.argv:
        print("Resetting nav in zensical.toml ...")
        clean_nav(project_root)
        print("Done.")
        return

    print("Running content generation hooks ...")
    run_content_hooks(project_root)

    print("Handling announcement banner ...")
    handle_announcement(project_root)

    print("Generating API pages ...")
    api_count = generate_api_pages(project_root)
    print(f"  {api_count} API modules")

    print("Generating nav in zensical.toml ...")
    nav_count = generate_nav(project_root)
    print(f"  {nav_count} top-level entries")

    print("Done.")


if __name__ == "__main__":
    main()
