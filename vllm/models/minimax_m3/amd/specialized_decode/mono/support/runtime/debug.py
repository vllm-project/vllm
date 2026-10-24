# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/mono/runtime/debug.py
# Original code is MIT-licensed; see mono/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stable mailbox region identifiers for device-side timeout diagnostics."""

import zlib


def region_id(name: str) -> int:
    """A mailbox region's id in a wait record: its name's CRC-32 (31 bits), the
    same in every kernel and table that names it."""
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def region_ids(names) -> dict[str, int]:
    """Name -> ``region_id``: a debug build's ``Mailbox`` table."""
    return {n: region_id(n) for n in names}
