# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Distillation capture for speculative decoding diagnostics."""

from vllm.v1.spec_decode.capture.config import (
    CaptureConfig,
    ConfigurationManager,
    LogitsLoggingConfig,
)
from vllm.v1.spec_decode.capture.spec_decode_capture import SpecDecodeCapture
from vllm.v1.spec_decode.capture.safetensors_writer import (
    AsyncSafetensorsWriter,
)
from vllm.v1.spec_decode.capture.percentile_tracker import (
    PercentileTracker,
)
from vllm.v1.spec_decode.capture.rate_limiter import RateLimiter
from vllm.v1.spec_decode.capture.transfer_handler import (
    AsyncTransferHandler,
)

# Backward compatibility alias
LogitsLogger = SpecDecodeCapture

__all__ = [
    "AsyncSafetensorsWriter",
    "AsyncTransferHandler",
    "CaptureConfig",
    "ConfigurationManager",
    "LogitsLogger",  # Backward compatibility alias
    "LogitsLoggingConfig",  # Backward compatibility alias
    "PercentileTracker",
    "RateLimiter",
    "SpecDecodeCapture",
]
