# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mistral Large 4 specialized decode on MI350X / MI355X (gfx950, TP4) and
MI300X / MI325X (gfx942, TP8), enabled by
`--additional-config '{"decode_backend": "latency_specialized"}'`."""

from .spec import DECODE_CONFIG, DecodeConfig, specialized_decode_config

__all__ = ["DECODE_CONFIG", "DecodeConfig", "specialized_decode_config"]
