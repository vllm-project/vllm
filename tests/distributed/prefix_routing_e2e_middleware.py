# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Response markers used only by the prefix-routing hardware E2E runner."""

import os

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class PrefixRoutingE2EIdentityMiddleware:
    """Expose the serving node and received DP rank to the E2E runner."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.node_id = os.environ["PREFIX_ROUTING_E2E_NODE_ID"]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rank = next(
            (
                value.decode("latin-1")
                for key, value in scope.get("headers", [])
                if key.lower() == b"x-data-parallel-rank"
            ),
            "",
        )

        async def send_with_identity(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-prefix-routing-e2e-node", self.node_id.encode("utf-8")),
                    (b"x-prefix-routing-e2e-rank", rank.encode("ascii")),
                ]
            await send(message)

        await self.app(scope, receive, send_with_identity)
