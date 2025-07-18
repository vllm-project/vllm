# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Script to compare intermediate logging outputs from two different runs.

This script compares the tensor outputs from two different intermediate logging
directories and generates a report of the differences.

Usage:
    python compare_intermediate.py --dir1 /path/to/first/log/dir --dir2 /path/to/second/log/dir [options]

Options:
    --dir1 DIR           First intermediate logging directory
    --dir2 DIR           Second intermediate logging directory
    --output FILE        Output file for the report (default: stdout)
    --format {md,json}   Output format (default: md)
    --rtol FLOAT         Relative tolerance for tensor comparison (default: 1e-5)
    --atol FLOAT         Absolute tolerance for tensor comparison (default: 1e-8)
    --steps STEPS        Comma-separated list of steps to compare (default: all)
    --modules MODULES    Comma-separated list of module name patterns to compare (default: all)
    --verbose            Include detailed information about each tensor
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def load_tensor(path: Path) -> torch.Tensor:
    """Load a tensor from a .pt file."""
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Error loading tensor from {path}: {e}")
        return None


def load_json(path: Path) -> Dict:
    """Load a JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {path}: {e}")
        return {}


def extract_diff_metatada(exception_str: str) -> Dict:
    try:
        num_diff_elements = int(
            re.search(r"Mismatched elements: (\d+) /", exception_str).group(1)
        )
        total_elements = int(
            re.search(r"Mismatched elements: \d+ / (\d+)", exception_str).group(1)
        )
        max_abs_diff = float(
            re.search(
                r"Greatest absolute difference: ([\d\.e-]+)", exception_str
            ).group(1)
        )
        max_rel_diff = float(
            re.search(
                r"Greatest relative difference: ([\d\.e-]+)", exception_str
            ).group(1)
        )
        return {
            "num_diff_elements": num_diff_elements,
            "total_elements": total_elements,
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
        }
    except Exception:
        return {"error": exception_str}


def compare_tensors(
    tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: float, atol: float
) -> Dict:
    """Compare two tensors and return a dictionary with comparison results."""
    if tensor1 is None or tensor2 is None:
        return {"match": False, "error": "One or both tensors are None"}

    if tensor1.shape != tensor2.shape:
        return {
            "match": False,
            "error": f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}",
        }

    if tensor1.dtype != tensor2.dtype:
        return {
            "match": False,
            "error": f"Dtype mismatch: {tensor1.dtype} vs {tensor2.dtype}",
        }

    # Check if tensors are close using PyTorch's assert_close
    try:
        torch.testing.assert_close(tensor1, tensor2, rtol=rtol, atol=atol)
    except Exception as e:
        return {"match": False, **extract_diff_metatada(str(e))}
    return {"match": True}


def compare_json_values(value1: Any, value2: Any) -> Dict:
    """Compare two JSON values and return a dictionary with comparison results."""
    if type(value1) is not type(value2):
        return {
            "match": False,
            "error": f"Type mismatch: {type(value1).__name__} vs {type(value2).__name__}",
        }

    if isinstance(value1, dict):
        # Compare dictionaries
        all_keys = set(value1.keys()) | set(value2.keys())
        mismatches = {}

        for key in all_keys:
            if key not in value1:
                mismatches[key] = {"error": "Missing in first dict"}
            elif key not in value2:
                mismatches[key] = {"error": "Missing in second dict"}
            else:
                comparison = compare_json_values(value1[key], value2[key])
                if not comparison["match"]:
                    mismatches[key] = comparison

        if mismatches:
            return {"match": False, "mismatches": mismatches}
        return {"match": True}

    elif isinstance(value1, list):
        # Compare lists
        if len(value1) != len(value2):
            return {
                "match": False,
                "error": f"Length mismatch: {len(value1)} vs {len(value2)}",
            }

        mismatches = {}
        for i, (item1, item2) in enumerate(zip(value1, value2)):
            comparison = compare_json_values(item1, item2)
            if not comparison["match"]:
                mismatches[i] = comparison

        if mismatches:
            return {"match": False, "mismatches": mismatches}
        return {"match": True}

    else:
        # Compare primitive values
        if value1 == value2:
            return {"match": True}
        else:
            return {"match": False, "value1": value1, "value2": value2}


def find_tensor_files(directory: Path) -> Dict[str, Dict[str, Dict[str, List[Path]]]]:
    """
    Find all tensor files in the given directory.

    Returns a dictionary with the structure:
    {
        "step_0": {
            "module_name_123456": {
                "inputs": [Path("inputs_0_cuda_0.pt"), ...],
                "outputs": [Path("output_cuda_0.pt"), ...]
            },
            ...
        },
        ...
    }
    """
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Find all step directories
    step_dirs = [d for d in directory.glob("step_*") if d.is_dir()]

    for step_dir in step_dirs:
        step_name = step_dir.name

        # Find all module directories
        module_dirs = [d for d in step_dir.glob("*") if d.is_dir()]

        for module_dir in module_dirs:
            module_name = module_dir.name

            # Find input tensor files
            input_tensors = list(module_dir.glob("inputs_*.pt"))
            if input_tensors:
                result[step_name][module_name]["inputs"] = input_tensors

            # Find output tensor files
            output_tensors = list(module_dir.glob("output*.pt"))
            if output_tensors:
                result[step_name][module_name]["outputs"] = output_tensors

            # Find JSON metadata files
            inputs_json = module_dir / "inputs.json"
            if inputs_json.exists():
                result[step_name][module_name]["inputs_json"] = [inputs_json]

            outputs_json = module_dir / "outputs.json"
            if outputs_json.exists():
                result[step_name][module_name]["outputs_json"] = [outputs_json]

    return result


def filter_steps_and_modules(
    tensor_files: Dict[str, Dict[str, Dict[str, List[Path]]]],
    steps: Optional[List[str]] = None,
    module_patterns: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, List[Path]]]]:
    """Filter tensor files by steps and module patterns."""
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Filter steps
    if steps:
        step_names = [f"step_{step}" for step in steps]
        steps_to_include = {step: True for step in step_names}
    else:
        steps_to_include = {step: True for step in tensor_files.keys()}

    # Compile module patterns
    if module_patterns:
        compiled_patterns = [re.compile(pattern) for pattern in module_patterns]
    else:
        compiled_patterns = None

    for step_name, modules in tensor_files.items():
        if step_name not in steps_to_include:
            continue

        for module_name, file_types in modules.items():
            # Check if module matches any pattern
            if compiled_patterns:
                if not any(
                    pattern.search(module_name) for pattern in compiled_patterns
                ):
                    continue

            result[step_name][module_name] = file_types

    return result


def compare_directories(
    dir1: Path,
    dir2: Path,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    steps: Optional[List[str]] = None,
    module_patterns: Optional[List[str]] = None,
) -> Dict:
    """Compare two intermediate logging directories and return a report."""
    # Find tensor files in both directories
    tensor_files1 = find_tensor_files(dir1)
    tensor_files2 = find_tensor_files(dir2)

    # Filter by steps and modules
    if steps or module_patterns:
        tensor_files1 = filter_steps_and_modules(tensor_files1, steps, module_patterns)
        tensor_files2 = filter_steps_and_modules(tensor_files2, steps, module_patterns)

    # Get all steps and modules from both directories
    all_steps = set(tensor_files1.keys()) | set(tensor_files2.keys())

    report = {
        "dir1": str(dir1),
        "dir2": str(dir2),
        "rtol": rtol,
        "atol": atol,
        "steps": {},
    }

    # Compare each step
    for step in sorted(all_steps):
        step_report = {
            "modules": {},
            "summary": {
                "total_modules": 0,
                "matching_modules": 0,
                "mismatched_modules": 0,
                "missing_modules": 0,
            },
        }

        # Get all modules from both directories for this step
        modules1 = tensor_files1.get(step, {})
        modules2 = tensor_files2.get(step, {})
        # TODO: read from module calls.txt to get the full module list
        # TODO: check if module calls txt exsits
        dir1_module_call_file = dir1 / step / "module_calls.txt"
        if dir1_module_call_file.exists():
            with open(dir1 / step / "module_calls.txt", "r") as f:
                all_modules = f.read().splitlines()
        else:
            print(
                "Warnings: the module call orders are missed, ordering using module alphbetics"
            )
            all_modules = sorted(set(modules1.keys()) | set(modules2.keys()))
        step_report["module_call_list"] = []
        for module in all_modules:
            module_report = {
                "inputs": {},
                "outputs": {},
                "summary": {
                    "total_tensors": 0,
                    "matching_tensors": 0,
                    "mismatched_tensors": 0,
                    "missing_tensors": 0,
                },
            }

            # Check if module exists in both directories
            if module not in modules1:
                module_report["error"] = f"Module missing in {dir1}"
                step_report["summary"]["missing_modules"] += 1
                step_report["modules"][module] = module_report
                continue

            if module not in modules2:
                module_report["error"] = f"Module missing in {dir2}"
                step_report["summary"]["missing_modules"] += 1
                step_report["modules"][module] = module_report
                continue

            # Compare JSON metadata
            for json_type in ["inputs_json", "outputs_json"]:
                json_files1 = modules1[module].get(json_type, [])
                json_files2 = modules2[module].get(json_type, [])

                if json_files1 and json_files2:
                    json1 = load_json(json_files1[0])
                    json2 = load_json(json_files2[0])

                    json_comparison = compare_json_values(json1, json2)
                    json_name = json_type.replace("_json", "")
                    module_report[f"{json_name}_metadata"] = json_comparison

                    # Add file paths for manual checking when there's a mismatch
                    if not json_comparison.get("match", True):
                        module_report[f"{json_name}_metadata"]["file1"] = str(
                            json_files1[0]
                        )
                        module_report[f"{json_name}_metadata"]["file2"] = str(
                            json_files2[0]
                        )

            # Compare input tensors
            input_tensors1 = {p.name: p for p in modules1[module].get("inputs", [])}
            input_tensors2 = {p.name: p for p in modules2[module].get("inputs", [])}
            all_input_names = set(input_tensors1.keys()) | set(input_tensors2.keys())

            for tensor_name in sorted(all_input_names):
                if tensor_name not in input_tensors1:
                    module_report["inputs"][tensor_name] = {
                        "match": False,
                        "error": f"Tensor missing in {dir1}",
                    }
                    module_report["summary"]["missing_tensors"] += 1
                elif tensor_name not in input_tensors2:
                    module_report["inputs"][tensor_name] = {
                        "match": False,
                        "error": f"Tensor missing in {dir2}",
                    }
                    module_report["summary"]["missing_tensors"] += 1
                else:
                    tensor1 = load_tensor(input_tensors1[tensor_name])
                    tensor2 = load_tensor(input_tensors2[tensor_name])

                    comparison = compare_tensors(tensor1, tensor2, rtol, atol)
                    # Add file paths for manual checking when there's a mismatch
                    if not comparison.get("match", False):
                        comparison["file1"] = str(input_tensors1[tensor_name])
                        comparison["file2"] = str(input_tensors2[tensor_name])

                    module_report["inputs"][tensor_name] = comparison

                    if comparison.get("match", False):
                        module_report["summary"]["matching_tensors"] += 1
                    else:
                        module_report["summary"]["mismatched_tensors"] += 1

                module_report["summary"]["total_tensors"] += 1

            # Compare output tensors
            output_tensors1 = {p.name: p for p in modules1[module].get("outputs", [])}
            output_tensors2 = {p.name: p for p in modules2[module].get("outputs", [])}
            all_output_names = set(output_tensors1.keys()) | set(output_tensors2.keys())

            for tensor_name in sorted(all_output_names):
                if tensor_name not in output_tensors1:
                    module_report["outputs"][tensor_name] = {
                        "match": False,
                        "error": f"Tensor missing in {dir1}",
                    }
                    module_report["summary"]["missing_tensors"] += 1
                elif tensor_name not in output_tensors2:
                    module_report["outputs"][tensor_name] = {
                        "match": False,
                        "error": f"Tensor missing in {dir2}",
                    }
                    module_report["summary"]["missing_tensors"] += 1
                else:
                    tensor1 = load_tensor(output_tensors1[tensor_name])
                    tensor2 = load_tensor(output_tensors2[tensor_name])

                    comparison = compare_tensors(tensor1, tensor2, rtol, atol)
                    # Add file paths for manual checking when there's a mismatch
                    if not comparison.get("match", False):
                        comparison["file1"] = str(output_tensors1[tensor_name])
                        comparison["file2"] = str(output_tensors2[tensor_name])

                    module_report["outputs"][tensor_name] = comparison

                    if comparison.get("match", False):
                        module_report["summary"]["matching_tensors"] += 1
                    else:
                        module_report["summary"]["mismatched_tensors"] += 1

                module_report["summary"]["total_tensors"] += 1

            # Update module status
            if module_report["summary"]["mismatched_tensors"] > 0:
                step_report["summary"]["mismatched_modules"] += 1
            else:
                step_report["summary"]["matching_modules"] += 1

            step_report["summary"]["total_modules"] += 1
            step_report["modules"][module] = module_report
            step_report["module_call_list"].append(module)

        report["steps"][step] = step_report

    # Add overall summary
    report["summary"] = {
        "total_steps": len(all_steps),
        "total_modules": sum(
            step_report["summary"]["total_modules"]
            for step_report in report["steps"].values()
        ),
        "matching_modules": sum(
            step_report["summary"]["matching_modules"]
            for step_report in report["steps"].values()
        ),
        "mismatched_modules": sum(
            step_report["summary"]["mismatched_modules"]
            for step_report in report["steps"].values()
        ),
        "missing_modules": sum(
            step_report["summary"]["missing_modules"]
            for step_report in report["steps"].values()
        ),
        "total_tensors": sum(
            module_report["summary"]["total_tensors"]
            for step_report in report["steps"].values()
            for module_name, module_report in step_report["modules"].items()
            if "summary" in module_report
        ),
        "matching_tensors": sum(
            module_report["summary"]["matching_tensors"]
            for step_report in report["steps"].values()
            for module_name, module_report in step_report["modules"].items()
            if "summary" in module_report
        ),
        "mismatched_tensors": sum(
            module_report["summary"]["mismatched_tensors"]
            for step_report in report["steps"].values()
            for module_name, module_report in step_report["modules"].items()
            if "summary" in module_report
        ),
        "missing_tensors": sum(
            module_report["summary"]["missing_tensors"]
            for step_report in report["steps"].values()
            for module_name, module_report in step_report["modules"].items()
            if "summary" in module_report
        ),
    }

    return report


def generate_markdown_report(report: Dict, verbose: bool = False) -> str:
    """Generate a markdown report from the comparison results."""
    lines = []

    # Add header
    lines.append("# Intermediate Logging Comparison Report")
    lines.append("")
    lines.append("Comparing intermediate logging outputs between:")
    lines.append(f"- **Directory 1**: `{report['dir1']}`")
    lines.append(f"- **Directory 2**: `{report['dir2']}`")
    lines.append("")
    lines.append(f"Comparison parameters:")
    lines.append(f"- Relative tolerance (rtol): {report['rtol']}")
    lines.append(f"- Absolute tolerance (atol): {report['atol']}")
    lines.append("")

    # Add overall summary
    summary = report["summary"]
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Category | Total | Matching | Mismatched | Missing |")
    lines.append("|----------|-------|----------|------------|---------|")
    lines.append(f"| Steps | {summary['total_steps']} | - | - | - |")
    lines.append(
        f"| Modules | {summary['total_modules']} | {summary['matching_modules']} | {summary['mismatched_modules']} | {summary['missing_modules']} |"
    )
    lines.append(
        f"| Tensors | {summary['total_tensors']} | {summary['matching_tensors']} | {summary['mismatched_tensors']} | {summary['missing_tensors']} |"
    )
    lines.append("")

    # Add step details
    for step_name, step_report in sorted(report["steps"].items()):
        step_summary = step_report["summary"]

        lines.append(f"## {step_name}")
        lines.append("")
        lines.append(
            f"**Summary**: {step_summary['matching_modules']} matching modules, {step_summary['mismatched_modules']} mismatched modules, {step_summary['missing_modules']} missing modules"
        )
        lines.append("")

        # Add module details
        for module_name in step_report["module_call_list"]:
            module_report = step_report["modules"][module_name]
            if "error" in module_report:
                lines.append(f"### ❌ {module_name}")
                lines.append("")
                lines.append(f"**Error**: {module_report['error']}")
                lines.append("")
                continue

            module_summary = module_report["summary"]

            # Determine module status
            if module_summary["mismatched_tensors"] > 0:
                status = "❌"
            else:
                status = "✅"

            lines.append(f"### {status} {module_name}")
            lines.append("")
            lines.append(
                f"**Summary**: {module_summary['matching_tensors']} matching tensors, {module_summary['mismatched_tensors']} mismatched tensors, {module_summary['missing_tensors']} missing tensors"
            )
            lines.append("")

            # Add metadata comparison results if available
            for metadata_type in ["inputs_metadata", "outputs_metadata"]:
                if metadata_type in module_report:
                    metadata_comparison = module_report[metadata_type]
                    if not metadata_comparison.get("match", True):
                        file_paths = ""
                        if (
                            "file1" in metadata_comparison
                            and "file2" in metadata_comparison
                        ):
                            file_paths = f" - Files: `{metadata_comparison['file1']}` vs `{metadata_comparison['file2']}`"

                        lines.append(
                            f"**{metadata_type.capitalize()}**: Mismatch detected{file_paths}"
                        )
                        if verbose and "mismatches" in metadata_comparison:
                            lines.append("```json")
                            lines.append(
                                json.dumps(metadata_comparison["mismatches"], indent=2)
                            )
                            lines.append("```")
                        lines.append("")

            # Add tensor comparison details
            if module_summary["mismatched_tensors"] > 0 or verbose:
                # Add input tensor details
                if module_report["inputs"]:
                    lines.append("#### Input Tensors")
                    lines.append("")
                    lines.append("| Tensor | Status | Details |")
                    lines.append("|--------|--------|---------|")

                    for tensor_name, comparison in sorted(
                        module_report["inputs"].items()
                    ):
                        if comparison.get("match", False):
                            status = "✅"
                            details = "Tensors match"
                        elif "error" in comparison:
                            status = "❌"
                            details = comparison["error"]
                        else:
                            status = "❌"
                            details = f"Max abs diff: {comparison.get('max_abs_diff', 'N/A'):.2e}, "
                            details = f"Max relative diff: {comparison.get('max_rel_diff', 'N/A'):.2e}, "
                            details += f"Diff elements: {comparison.get('num_diff_elements', 'N/A')}/{comparison.get('total_elements', 'N/A')}"
                            if "file1" in comparison and "file2" in comparison:
                                details += f"<br>Files: `{comparison['file1']}` vs `{comparison['file2']}`"

                        lines.append(f"| {tensor_name} | {status} | {details} |")

                    lines.append("")

                # Add output tensor details
                if module_report["outputs"]:
                    lines.append("#### Output Tensors")
                    lines.append("")
                    lines.append("| Tensor | Status | Details |")
                    lines.append("|--------|--------|---------|")

                    for tensor_name, comparison in sorted(
                        module_report["outputs"].items()
                    ):
                        if comparison.get("match", False):
                            status = "✅"
                            details = "Tensors match"
                        elif "error" in comparison:
                            status = "❌"
                            details = comparison["error"]
                        else:
                            status = "❌"
                            details = f"Max abs diff: {comparison.get('max_abs_diff', 'N/A')}, "
                            details = f"Max relative diff: {comparison.get('max_rel_diff', 'N/A')}, "
                            details += f"Diff elements: {comparison.get('num_diff_elements', 'N/A')}/{comparison.get('total_elements', 'N/A')}"

                        lines.append(f"| {tensor_name} | {status} | {details} |")

                    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare intermediate logging outputs from two different runs."
    )
    parser.add_argument(
        "--dir1", required=True, help="First intermediate logging directory"
    )
    parser.add_argument(
        "--dir2", required=True, help="Second intermediate logging directory"
    )
    parser.add_argument("--output", help="Output file for the report (default: stdout)")
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance for tensor comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance for tensor comparison (default: 1e-8)",
    )
    parser.add_argument(
        "--steps", help="Comma-separated list of steps to compare (default: all)"
    )
    parser.add_argument(
        "--modules",
        help="Comma-separated list of module name patterns to compare (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include detailed information about each tensor",
    )

    args = parser.parse_args()

    # Parse steps and modules
    steps = args.steps.split(",") if args.steps else None
    module_patterns = args.modules.split(",") if args.modules else None

    # Compare directories
    report = compare_directories(
        Path(args.dir1),
        Path(args.dir2),
        rtol=args.rtol,
        atol=args.atol,
        steps=steps,
        module_patterns=module_patterns,
    )

    # Generate report
    output = generate_markdown_report(report, verbose=args.verbose)

    # Write report
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
            print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()


def invoke_main() -> None:
    main()
