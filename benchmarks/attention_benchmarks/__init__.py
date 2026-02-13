# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM Attention Benchmarking Suite."""

from .batch_spec import (
    BatchRequest,
    format_batch_spec,
    get_batch_stats,
    parse_batch_spec,
    reorder_for_flashinfer,
    split_by_type,
)
from .common import (
    BenchmarkConfig,
    BenchmarkResult,
    MockLayer,
    MockModelConfig,
    ResultsFormatter,
    get_attention_scale,
    is_mla_backend,
    setup_mla_dims,
)

__all__ = [
    # Batch specification
    "BatchRequest",
    "parse_batch_spec",
    "format_batch_spec",
    "reorder_for_flashinfer",
    "split_by_type",
    "get_batch_stats",
    # Benchmarking infrastructure
    "BenchmarkConfig",
    "BenchmarkResult",
    "ResultsFormatter",
    # Mock objects
    "MockLayer",
    "MockModelConfig",
    # Utilities
    "setup_mla_dims",
    "get_attention_scale",
    "is_mla_backend",
]
