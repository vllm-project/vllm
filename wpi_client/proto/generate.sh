#!/usr/bin/env bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Regenerate Python gRPC stubs from wpi.proto.
#
# Prerequisites:
#   pip install grpcio-tools
#
# Usage:
#   cd proto && ./generate.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m grpc_tools.protoc \
    -I"${SCRIPT_DIR}" \
    --python_out="${SCRIPT_DIR}" \
    --grpc_python_out="${SCRIPT_DIR}" \
    "${SCRIPT_DIR}/wpi.proto"

# Fix import path: the generated code uses `import wpi_pb2` but we need
# the fully-qualified `from wpi_client.proto import wpi_pb2`.
sed -i.bak 's/^import wpi_pb2 as wpi__pb2$/from wpi_client.proto import wpi_pb2 as wpi__pb2/' \
    "${SCRIPT_DIR}/wpi_pb2_grpc.py"
rm -f "${SCRIPT_DIR}/wpi_pb2_grpc.py.bak"

echo "Generated wpi_pb2.py and wpi_pb2_grpc.py from wpi.proto"
