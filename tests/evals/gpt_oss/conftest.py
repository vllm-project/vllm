# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pytest configuration for GPT-OSS evaluation tests.
"""


def pytest_addoption(parser):
    """Add command line options for pytest."""
    parser.addoption("--model", action="store", help="Model name to evaluate")
    parser.addoption(
        "--metric", action="store", type=float, help="Expected metric threshold"
    )
    parser.addoption(
        "--server-args", action="store", default="", help="Additional server arguments"
    )
