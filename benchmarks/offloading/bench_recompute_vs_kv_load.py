#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility entry point for recompute-only benchmark runs."""

try:
    from .bench_offloading_e2e import main
except ImportError:
    from bench_offloading_e2e import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            default_mode="recompute",
            default_sizes=[256, 512, 1024, 2048, 4096, 8192],
        )
    )
