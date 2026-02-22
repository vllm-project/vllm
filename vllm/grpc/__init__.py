# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM gRPC protocol definitions.

Proto definitions are owned by SMG and published as the smg-grpc-proto package.
This module re-exports the vLLM engine protobuf modules for backwards
compatibility.
"""

from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc

__all__ = [
    "vllm_engine_pb2",
    "vllm_engine_pb2_grpc",
]
