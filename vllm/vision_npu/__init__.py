# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Vision NPU backend infrastructure for vLLM.

Provides pluggable NPU backends for vision processing in multimodal models.
"""

from .backend import NPUVisionBackend
from .flexmlrt_backend import FlexMLRTVisionBackend

__all__ = ["NPUVisionBackend", "FlexMLRTVisionBackend"]
