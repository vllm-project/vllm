# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-export of attention-backend abstract types from the V1 framework.


Re-exporting -- not subclassing or wrapping -- is required because the V1
framework keys KV-cache group identity on the backend class object
itself.
"""

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)

__all__ = [
    "AttentionBackend",
    "AttentionCGSupport",
    "AttentionMetadata",
    "AttentionMetadataBuilder",
    "CommonAttentionMetadata",
    "MultipleOf",
]
