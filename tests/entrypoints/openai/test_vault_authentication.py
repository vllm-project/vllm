# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import patch

import pytest
from starlette.datastructures import Headers

from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.server_utils import VaultAuthenticationMiddleware
from vllm.utils.argparse_utils import FlexibleArgumentParser


### arg passer
def _build_vllm_parsers():
    # quick override to be CPU for the purpose of testing just python
    with (
        patch("vllm.platforms.current_platform.device_type", "cpu"),
        patch.dict(os.environ, {"VLLM_TARGET_DEVICE": "cpu"}),
    ):
        vllm_parser = FlexibleArgumentParser()
        subparsers = vllm_parser.add_subparsers(dest="subcommand")
        serve_parser = subparsers.add_parser("serve")
        make_arg_parser(serve_parser)
        return {"vllm": vllm_parser, "vllm serve": serve_parser}


@pytest.fixture
def serve_parser():
    return _build_vllm_parsers()["vllm serve"]


"""Test that Vault-specific CLI arguments are parsed correctly."""


def test_vault_arg_parsing(serve_parser):
    args = serve_parser.parse_args(
        [
            "--vault-url",
            "http://someserver:8200",
            "--vault-token",
            "hvs.test_token",
            "--vault-secret-path",
            "secret/data/vllm",
            "--vault-key",
            "api-token",
        ]
    )
    assert args.vault_url == "http://someserver:8200"
    assert args.vault_token == "hvs.test_token"
    assert args.vault_secret_path == "secret/data/vllm"
    assert args.vault_key == "api-token"


### middleware stub
class MockApp:
    async def __call__(self, scope, receive, send):
        return "called"


@pytest.fixture
def mock_vault_middleware():
    app = MockApp()
    return VaultAuthenticationMiddleware(
        app=app,
        vault_url="http://localhost:8200",
        vault_token="fake-root",
        secret_path="vllm/auth",
        vault_key="api-token",
        mount_point="secret",
    )


"""Test verification of token"""


@pytest.mark.asyncio  # Add this
async def test_verify_token_valid(mock_vault_middleware):
    headers = Headers({"authorization": "Bearer correct-token"})
    mock_response = {"data": {"data": {"api-token": "correct-token"}}}

    with patch.object(
        mock_vault_middleware.client.secrets.kv.v2,
        "read_secret_version",
        return_value=mock_response,
    ):
        # Add await here
        assert await mock_vault_middleware.verify_token(headers) is True


"""Test invalid token"""


@pytest.mark.asyncio
async def test_verify_token_invalid(mock_vault_middleware):
    headers = Headers({"authorization": "Bearer wrong-token"})
    mock_response = {"data": {"data": {"api-token": "correct-token"}}}

    with patch.object(
        mock_vault_middleware.client.secrets.kv.v2,
        "read_secret_version",
        return_value=mock_response,
    ):
        # Add await here
        assert await mock_vault_middleware.verify_token(headers) is False


"""Test failure when the expected key is missing from Vault JSON."""


@pytest.mark.asyncio
async def test_verify_token_missing_key(mock_vault_middleware, caplog):
    headers = Headers({"authorization": "Bearer some-token"})
    mock_response = {"data": {"data": {"wrong_key": "some-token"}}}

    with (
        caplog.at_level("ERROR"),
        patch.object(
            mock_vault_middleware.client.secrets.kv.v2,
            "read_secret_version",
            return_value=mock_response,
        ),
    ):
        # Add await here
        assert await mock_vault_middleware.verify_token(headers) is False


"""Test failure when Vault returns a 403 Forbidden."""


@pytest.mark.asyncio
async def test_verify_token_vault_forbidden(mock_vault_middleware):
    import hvac

    headers = Headers({"authorization": "Bearer some-token"})

    with patch.object(
        mock_vault_middleware.client.secrets.kv.v2,
        "read_secret_version",
        side_effect=hvac.exceptions.Forbidden("Access Denied"),
    ):
        # Add await here
        assert await mock_vault_middleware.verify_token(headers) is False


"""Test that non-API paths (like health) bypass authentication."""


@pytest.mark.asyncio
async def test_middleware_call_bypass(mock_vault_middleware):
    scope = {"type": "http", "method": "GET", "path": "/health", "headers": []}

    with patch.object(
        mock_vault_middleware, "app", side_effect=mock_vault_middleware.app
    ) as mock_app:
        await mock_vault_middleware(scope, None, None)
        mock_app.assert_called_once()
