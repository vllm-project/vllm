# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

import pytest

pytest.importorskip("smg_grpc_servicer")

from vllm.entrypoints.launchers import grpc_server  # noqa: E402


@pytest.mark.asyncio
async def test_standalone_grpc_rejects_api_key_environment(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    args = grpc_server._build_arg_parser().parse_args(["--model", "dummy"])

    with pytest.raises(ValueError, match="does not support API keys"):
        await grpc_server.serve_grpc(args)


def test_standalone_grpc_main_rejects_api_key_with_rust_servicer(monkeypatch):
    """The Rust servicer switch replaces serve_grpc's body, so main() checks too."""
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    monkeypatch.setenv("SMG_VLLM_SERVICER_IMPL", "rust")
    monkeypatch.setattr(sys, "argv", ["grpc_server", "--model", "dummy"])

    with pytest.raises(ValueError, match="does not support API keys"):
        grpc_server.main()
