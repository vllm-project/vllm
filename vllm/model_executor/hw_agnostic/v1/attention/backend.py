# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-export of attention-backend abstract types from the V1 framework.

The hw-agnostic attention tree depends on ``AttentionBackend``,
``AttentionMetadataBuilder``, ``AttentionCGSupport``,
``CommonAttentionMetadata`` and ``MultipleOf`` as the contract for
KV-cache group discovery and metadata building. The V1 framework keys
group identity on the backend class object itself, so the re-exports
here resolve to the same classes — importing from this shim preserves
``isinstance`` / ``full_cls_name()`` identity.

The hw-agnostic isolation lint forbids reaching into
``vllm.v1.attention.backend`` directly; this shim is the single
sanctioned entry point.
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
