# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM gRPC protocol definitions.

This module contains the protocol buffer definitions for vLLM's gRPC API.
The protobuf files are compiled into Python code using grpcio-tools.
"""

# These imports will be available after protobuf compilation
# from vllm.grpc import vllm_engine_pb2
# from vllm.grpc import vllm_engine_pb2_grpc

__all__ = [
    "vllm_engine_pb2",
    "vllm_engine_pb2_grpc",
]
