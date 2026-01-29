#!/usr/bin/env python3
"""
Parse JUnit XML test results and generate SIG-grouped failure reports.

Usage:
    python annotate-test-results.py test-results-*.xml

Outputs markdown to stdout for piping to buildkite-agent annotate.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, NamedTuple


class TestFailure(NamedTuple):
    """Represents a single test failure with SIG metadata."""
    test_name: str
    classname: str
    file_path: str
    sig: str  # SIG ownership tag (e.g., "sig-ci")
    failure_type: str
    failure_message: str


def parse_junit_xml(xml_path: Path) -> List[TestFailure]:
    """Extract failures with SIG metadata from JUnit XML."""
    failures = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find all testcases with failures or errors
        for testcase in root.findall('.//testcase'):
            failure_elem = testcase.find('failure')
            error_elem = testcase.find('error')

            if failure_elem is None and error_elem is None:
                continue  # Test passed, skip it

            # Extract test identification
            test_name = testcase.get('name', 'unknown')
            classname = testcase.get('classname', '')
            file_path = testcase.get('file', '')

            # Extract SIG from properties
            sig = None
            properties = testcase.find('properties')
            if properties is not None:
                for prop in properties.findall('property'):
                    if prop.get('name') == 'sig':
                        sig = prop.get('value')
                        break

            # Extract failure details
            failure_or_error = failure_elem if failure_elem is not None else error_elem
            failure_type = failure_or_error.get('type', 'Error')
            failure_message = failure_or_error.get('message', '')

            # Truncate long messages
            if len(failure_message) > 200:
                failure_message = failure_message[:200] + '...'

            failures.append(TestFailure(
                test_name=test_name,
                classname=classname,
                file_path=file_path,
                sig=sig or "unassigned",
                failure_type=failure_type,
                failure_message=failure_message
            ))

    except ET.ParseError as e:
        print(f"Warning: Failed to parse {xml_path}: {e}", file=sys.stderr)

    return failures


def group_by_sig(failures: List[TestFailure]) -> Dict[str, List[TestFailure]]:
    """Group failures by SIG ownership."""
    grouped = defaultdict(list)
    for failure in failures:
        grouped[failure.sig].append(failure)
    return dict(grouped)


def generate_markdown(grouped_failures: Dict[str, List[TestFailure]]) -> str:
    """Generate triage-friendly markdown report."""
    if not grouped_failures:
        return "✅ All tests passed!"

    total_failures = sum(len(failures) for failures in grouped_failures.values())
    num_sigs = len(grouped_failures)

    lines = [
        "# 🔍 Test Failures - SIG Ownership Report",
        "",
        f"**Summary:** {total_failures} failure(s) across {num_sigs} SIG(s)",
        "",
    ]

    # Sort SIGs: unassigned last, rest alphabetically
    sig_order = sorted(grouped_failures.keys(), key=lambda s: (s == "unassigned", s))

    for sig in sig_order:
        failures = grouped_failures[sig]
        emoji = "⚪" if sig == "unassigned" else "🔴"

        lines.append(f"## {emoji} {sig} ({len(failures)} failure(s))")
        lines.append("")
        lines.append("| Test | Type | Message |")
        lines.append("|------|------|---------|")

        for f in failures:
            # Clean test name for display
            display_name = f.test_name
            if len(display_name) > 50:
                display_name = "..." + display_name[-47:]

            # Escape pipe characters in message
            message = f.failure_message.replace('|', '\\|').replace('\n', ' ')

            lines.append(f"| `{display_name}` | {f.failure_type} | {message} |")

        lines.append("")

    if "unassigned" in grouped_failures:
        lines.append("---")
        lines.append("**Note:** Tests in `unassigned` need SIG ownership tags added.")

    return "\n".join(lines)


def main():
    xml_files = sys.argv[1:]

    if not xml_files:
        print("No JUnit XML files specified", file=sys.stderr)
        sys.exit(1)

    all_failures = []
    for xml_file in xml_files:
        xml_path = Path(xml_file)
        if not xml_path.exists():
            print(f"Warning: {xml_file} not found", file=sys.stderr)
            continue
        all_failures.extend(parse_junit_xml(xml_path))

    grouped = group_by_sig(all_failures)
    markdown = generate_markdown(grouped)
    print(markdown)


if __name__ == "__main__":
    main()
