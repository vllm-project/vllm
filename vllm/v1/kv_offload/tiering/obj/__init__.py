# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OBJ (S3) secondary tier for multi-tier KV cache offloading."""

from vllm.v1.kv_offload.tiering.obj.tier import ObjSecondaryTier

__all__ = ["ObjSecondaryTier"]
