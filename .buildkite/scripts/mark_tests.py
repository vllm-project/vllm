#!/usr/bin/env python3
"""
Script to help mark tests with appropriate pytest marks based on directory and content.

Usage:
    python mark_tests.py <test_file_or_directory>

Examples:
    python mark_tests.py tests/v1/attention/
    python mark_tests.py tests/kernels/attention/test_flash_attn.py
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Set


# Mapping of directory patterns to suggested marks
DIRECTORY_MARK_MAP = {
    "tests/v1/attention": ["v1", "attention"],
    "tests/v1/distributed": ["v1", "v1_distributed", "distributed_comm"],
    "tests/v1/core": ["v1", "v1_core"],
    "tests/v1/engine": ["v1", "engine"],
    "tests/v1/spec_decode": ["v1", "spec_decode"],
    "tests/v1/kv_connector": ["v1", "kv_cache"],
    "tests/v1/kv_offload": ["v1", "kv_cache"],
    "tests/v1/e2e": ["v1", "e2e"],
    "tests/v1": ["v1"],

    "tests/kernels/attention": ["kernels", "attention"],
    "tests/kernels/moe": ["kernels", "models_moe", "expert_parallel"],
    "tests/kernels/quantization": ["kernels", "quantization"],
    "tests/kernels": ["kernels"],

    "tests/distributed": ["distributed_comm"],

    "tests/models/language": ["models_language", "correctness"],
    "tests/models/multimodal": ["models_multimodal", "multimodal", "correctness"],
    "tests/models/quantization": ["models_language", "quantization", "correctness"],

    "tests/quantization": ["quantization"],

    "tests/lora": ["lora"],

    "tests/compile": ["compilation"],

    "tests/entrypoints/openai": ["entrypoints", "openai_api"],
    "tests/entrypoints/pooling": ["entrypoints", "pooling"],
    "tests/entrypoints/llm": ["entrypoints", "offline_inference"],
    "tests/entrypoints": ["entrypoints"],

    "tests/basic_correctness": ["correctness", "e2e"],

    "tests/samplers": ["sampling"],

    "tests/multimodal": ["multimodal"],

    "tests/reasoning": ["reasoning"],

    "tests/tool_parsers": ["tool_calling"],
    "tests/tool_use": ["tool_calling"],

    "tests/benchmarks": ["benchmark"],

    "tests/tpu": ["tpu"],
    "tests/rocm": ["rocm"],
    "tests/cuda": ["cuda"],

    "tests/weight_loading": ["model_loading"],

    "tests/engine": ["engine"],
}

# Content patterns to detect additional marks
CONTENT_PATTERNS = {
    "tensor.?parallel|TP|tp_size": "tensor_parallel",
    "pipeline.?parallel|PP|pp_size": "pipeline_parallel",
    "expert.?parallel|EP|ep_size": "expert_parallel",
    "data.?parallel|DP|dp_size": "data_parallel",
    "nccl|all.?reduce|all.?gather": "distributed_comm",
    "fp8|int8|gptq|awq|quant": "quantization",
    "lora|peft": "lora",
    "spec.?decode|speculative|draft": "spec_decode",
    "kv.?cache|cache.?offload": "kv_cache",
    "tool.?call|function.?call": "tool_calling",
    "flash.?attention|flashinfer|flash_attn": "attention",
    "moe|mixtral|deepseek": "models_moe",
    "multimodal|vision|audio|image": "multimodal",
}


def get_directory_marks(file_path: Path) -> Set[str]:
    """Get suggested marks based on directory location."""
    marks = set()

    # Convert to string and normalize path separators
    path_str = str(file_path).replace('\\', '/')

    # Try to extract the tests/... portion
    if 'tests/' in path_str:
        # Get everything from 'tests/' onwards
        path_str = 'tests/' + path_str.split('tests/', 1)[1]

    # Find the longest matching directory pattern
    for pattern, pattern_marks in sorted(DIRECTORY_MARK_MAP.items(), key=lambda x: -len(x[0])):
        if path_str.startswith(pattern):
            marks.update(pattern_marks)
            break

    return marks


def get_content_marks(content: str) -> Set[str]:
    """Get suggested marks based on file content."""
    marks = set()
    content_lower = content.lower()

    for pattern, mark in CONTENT_PATTERNS.items():
        if re.search(pattern, content_lower, re.IGNORECASE):
            marks.add(mark)

    return marks


def has_marks(content: str) -> bool:
    """Check if file already has pytest marks."""
    return bool(re.search(r'@pytest\.mark\.\w+|pytestmark\s*=', content))


def generate_marks_string(marks: Set[str]) -> str:
    """Generate the pytestmark string for a file."""
    if not marks:
        return ""

    sorted_marks = sorted(marks)
    if len(sorted_marks) == 1:
        return f"pytestmark = pytest.mark.{sorted_marks[0]}"
    else:
        marks_list = ", ".join(f"pytest.mark.{m}" for m in sorted_marks)
        return f"pytestmark = [{marks_list}]"


def get_display_path(file_path: Path) -> Path:
    """Get a path suitable for display (relative if possible, absolute otherwise)."""
    try:
        return file_path.relative_to(Path.cwd())
    except ValueError:
        return file_path


def add_marks_to_file(file_path: Path, dry_run: bool = True) -> None:
    """Add appropriate pytest marks to a test file."""
    content = file_path.read_text()
    display_path = get_display_path(file_path)

    # Skip if already has marks
    if has_marks(content):
        print(f"[SKIP] {display_path}: Already marked")
        return

    # Get suggested marks
    dir_marks = get_directory_marks(file_path)
    content_marks = get_content_marks(content)
    all_marks = dir_marks | content_marks

    if not all_marks:
        print(f"[WARN] {display_path}: No marks suggested")
        return

    # Find where to insert marks (after imports, before first test/class)
    lines = content.split('\n')
    insert_line = 0

    # Find last import
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_line = i + 1

    # Skip blank lines after imports
    while insert_line < len(lines) and not lines[insert_line].strip():
        insert_line += 1

    # Generate marks
    marks_str = generate_marks_string(all_marks)

    # Insert marks
    new_lines = lines[:insert_line] + ['', marks_str, ''] + lines[insert_line:]
    new_content = '\n'.join(new_lines)

    if dry_run:
        print(f"[DRY RUN] {display_path}: Would add {marks_str}")
    else:
        file_path.write_text(new_content)
        print(f"[DONE] {display_path}: Added {marks_str}")


def process_path(path: Path, dry_run: bool = True) -> None:
    """Process a file or directory."""
    if path.is_file():
        if path.name.startswith('test_') and path.suffix == '.py':
            add_marks_to_file(path, dry_run)
    elif path.is_dir():
        for test_file in sorted(path.rglob('test_*.py')):
            add_marks_to_file(test_file, dry_run)
    else:
        print(f"Error: {path} is not a file or directory", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Add pytest marks to test files")
    parser.add_argument('path', type=Path, help="Test file or directory to process")
    parser.add_argument('--apply', action='store_true', help="Actually modify files (default is dry-run)")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: {args.path} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"{'DRY RUN - ' if not args.apply else ''}Processing: {args.path}")
    print("-" * 60)

    process_path(args.path, dry_run=not args.apply)

    if not args.apply:
        print("\nNOTE: This was a dry run. Use --apply to actually modify files.")


if __name__ == '__main__':
    main()
