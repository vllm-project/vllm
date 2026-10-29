# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Side-effect-free MiniMax-M3 specialized-decode configuration."""

from .spec import DECODE_CONFIG, DecodeConfig, specialized_decode_config

__all__ = ["DECODE_CONFIG", "DecodeConfig", "specialized_decode_config"]
