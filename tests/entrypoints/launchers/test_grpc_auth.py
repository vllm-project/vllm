# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytest.importorskip("smg_grpc_servicer")

from vllm.entrypoints.launchers import grpc_server  # noqa: E402


@pytest.mark.asyncio
async def test_standalone_grpc_rejects_api_key_environment(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    args = grpc_server._build_arg_parser().parse_args(["--model", "dummy"])

    with pytest.raises(ValueError, match="does not support authentication"):
        await grpc_server.serve_grpc(args)
