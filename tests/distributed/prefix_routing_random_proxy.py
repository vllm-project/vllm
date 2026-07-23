# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic random proxy used by the prefix-routing benchmark."""

import argparse
import asyncio
import json
import random
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class RandomProxy:
    def __init__(self, upstreams: list[str], seed: int, stats_file: Path) -> None:
        self.upstreams = [url.rstrip("/") for url in upstreams]
        self.random = random.Random(seed)
        self.stats_file = stats_file
        self.counts = [0 for _ in upstreams]
        self.session: ClientSession | None = None

    async def start(self, _: web.Application) -> None:
        self.session = ClientSession(timeout=ClientTimeout(total=None))

    async def stop(self, _: web.Application) -> None:
        if self.session is not None:
            await self.session.close()
        self.write_stats()

    def write_stats(self) -> None:
        payload = {
            "policy": "seeded-random",
            "upstreams": self.upstreams,
            "request_counts": {
                upstream: count for upstream, count in zip(self.upstreams, self.counts)
            },
            "total_requests": sum(self.counts),
        }
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        self.stats_file.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    async def health(self, _: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def stats(self, _: web.Request) -> web.Response:
        self.write_stats()
        return web.json_response(
            {
                "upstreams": self.upstreams,
                "request_counts": self.counts,
                "total_requests": sum(self.counts),
            }
        )

    async def forward(self, request: web.Request) -> web.StreamResponse:
        assert self.session is not None
        index = self.random.randrange(len(self.upstreams))
        self.counts[index] += 1
        upstream = self.upstreams[index]
        target = f"{upstream}{request.rel_url}"
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _HOP_BY_HOP_HEADERS and name.lower() != "host"
        }
        body = await request.read()
        try:
            async with self.session.request(
                request.method,
                target,
                headers=headers,
                data=body,
                allow_redirects=False,
            ) as upstream_response:
                response_headers = {
                    name: value
                    for name, value in upstream_response.headers.items()
                    if name.lower() not in _HOP_BY_HOP_HEADERS
                }
                response_headers["x-prefix-routing-benchmark-upstream"] = str(index)
                response = web.StreamResponse(
                    status=upstream_response.status,
                    reason=upstream_response.reason,
                    headers=response_headers,
                )
                await response.prepare(request)
                async for chunk in upstream_response.content.iter_chunked(64 * 1024):
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (OSError, asyncio.TimeoutError) as exc:
            return web.json_response(
                {"error": f"upstream request failed: {type(exc).__name__}"},
                status=502,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", action="append", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if len(args.upstream) < 2:
        raise SystemExit("at least two --upstream values are required")
    proxy = RandomProxy(args.upstream, args.seed, args.stats_file)
    app = web.Application(client_max_size=16 * 1024**2)
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.stop)
    app.router.add_get("/health", proxy.health)
    app.router.add_get("/_prefix_routing_benchmark/stats", proxy.stats)
    app.router.add_route("*", "/{path:.*}", proxy.forward)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
