# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-batch policy tables."""


def batch_table(spec):
    """Per-batch table of integers ("1-2:96,3-8:120"), tuples ("16:64.16.4.8") or names
    ("16-64:trtllm"); later entries override earlier ones."""
    table = {}
    for item in filter(None, spec.split(",")):
        span, value = item.split(":")
        lo, _, hi = span.partition("-")
        if "." in value:
            value = tuple(map(int, value.split(".")))
        elif value.lstrip("-").isdigit():
            value = int(value)
        table.update({b: value for b in range(int(lo), int(hi or lo) + 1)})
    return table
