# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Anthropic-compatible x-api-key authentication.

See https://github.com/vllm-project/vllm/issues/51572.
"""

from starlette.datastructures import Headers

from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware


class TestAuthenticationMiddlewareXApiKey:
    """Anthropic-compatible clients/proxies send x-api-key instead of
    Authorization: Bearer. AuthenticationMiddleware must accept both.
    """

    def setup_method(self):
        self.middleware = AuthenticationMiddleware(app=None, tokens=["secret-token"])

    def test_accepts_authorization_bearer(self):
        headers = Headers({"Authorization": "Bearer secret-token"})
        assert self.middleware.verify_token(headers)

    def test_accepts_x_api_key(self):
        headers = Headers({"x-api-key": "secret-token"})
        assert self.middleware.verify_token(headers)

    def test_rejects_wrong_x_api_key(self):
        headers = Headers({"x-api-key": "wrong-token"})
        assert not self.middleware.verify_token(headers)

    def test_rejects_missing_credentials(self):
        assert not self.middleware.verify_token(Headers({}))

    def test_rejects_non_bearer_authorization_without_x_api_key(self):
        headers = Headers({"Authorization": "Basic secret-token"})
        assert not self.middleware.verify_token(headers)

    def test_x_api_key_works_when_bearer_is_wrong(self):
        headers = Headers(
            {
                "Authorization": "Bearer wrong-token",
                "x-api-key": "secret-token",
            }
        )
        assert self.middleware.verify_token(headers)
