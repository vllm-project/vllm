# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lab-only single-API-process middleware; never expose management via proxy.

Keep the engine loopback-only. This is not a general autoscaling controller.
The experimental 32 MiB request limit applies equally to both A/B images.
"""

import asyncio
import hmac
import json
import os

from starlette.responses import JSONResponse

LIMIT = 32 * 1024 * 1024
INFERENCE = {"/v1/completions", "/v1/chat/completions"}
READ_ONLY = {"/health", "/metrics", "/v1/models"}


class AdmissionFence:
    def __init__(self, app):
        self.app = app
        self.active = 0
        self.uncertain_disconnect = False
        self.fences = {}
        self.retirement = None
        self.lock = asyncio.Lock()
        self.admin_key = os.environ.get("PEER_LAB_ADMIN_KEY", "")
        if len(self.admin_key) < 32:
            raise RuntimeError("Set a private PEER_LAB_ADMIN_KEY (at least 32 chars)")
        self.expected_ranks = int(os.environ.get("PEER_LAB_TP_SIZE", "4"))

    async def _respond(self, scope, receive, send, data, status=200):
        await JSONResponse(data, status_code=status)(scope, receive, send)

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if path in READ_ONLY and scope["method"] == "GET":
            return await self.app(scope, receive, send)
        if path not in INFERENCE and path != "/peer_lab/control":
            return await self._respond(
                scope, receive, send, {"error": "Endpoint disabled in peer lab"}, 404
            )
        if scope["method"] != "POST":
            return await self._respond(
                scope, receive, send, {"error": "POST required"}, 405
            )
        management = path == "/peer_lab/control"
        if management:
            headers = dict(scope["headers"])
            supplied = headers.get(b"authorization", b"")
            expected = ("Bearer " + self.admin_key).encode()
            if not hmac.compare_digest(supplied, expected):
                return await self._respond(
                    scope, receive, send, {"error": "Unauthorized"}, 401
                )
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            part = message.get("body", b"")
            chunks.append(part)
            size += len(part)
            if size > LIMIT:
                return await self._respond(
                    scope, receive, send, {"error": "Lab body limit"}, 413
                )
            if not message.get("more_body", False):
                break
        raw = b"".join(chunks)
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("Object required")
        except (ValueError, UnicodeDecodeError):
            return await self._respond(
                scope, receive, send, {"error": "Invalid JSON"}, 400
            )
        if management:
            async with self.lock:
                try:
                    result, code = await self._control(scope, body)
                except Exception as exc:
                    result, code = (
                        {"state": "failed", "reason": type(exc).__name__},
                        503,
                    )
            return await self._respond(scope, receive, send, result, code)
        kv = body.get("kv_transfer_params") or {}
        target = kv.get("remote_engine_id") if isinstance(kv, dict) else None
        del body, chunks
        if target is not None and not isinstance(target, str):
            return await self._respond(
                scope, receive, send, {"error": "Invalid engine ID"}, 400
            )
        # No await between checking the fence and accounting the request.
        if target in self.fences:
            return await self._respond(
                scope, receive, send, {"error": "Retired P generation"}, 409
            )
        if self.retirement is not None:
            return await self._respond(
                scope, receive, send, {"error": "Lab retirement in progress"}, 503
            )
        self.active += 1
        delivered = False
        response_complete = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": raw, "more_body": False}
            return await receive()

        async def tracked_send(message):
            nonlocal response_complete
            await send(message)
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                response_complete = True

        try:
            await self.app(scope, replay, tracked_send)
        finally:
            self.active -= 1
            if not response_complete:
                self.uncertain_disconnect = True

    def _valid_ranks(self, results, expected_decode=None):
        return (
            isinstance(results, list)
            and len(results) == self.expected_ranks
            and all(isinstance(r, dict) for r in results)
            and {r.get("tp_rank") for r in results} == set(range(self.expected_ranks))
            and len({r.get("decode_engine_id") for r in results}) == 1
            and results[0].get("decode_engine_id") is not None
            and (
                expected_decode is None
                or results[0]["decode_engine_id"] == expected_decode
            )
        )

    async def _control(self, scope, body):
        engine = scope["app"].state.engine_client
        phase = body.get("phase", "snapshot")
        if phase == "snapshot":
            results = await engine.collective_rpc("peer_lab_snapshot", timeout=30)
            return {
                "results": results,
                "active_http": self.active,
                "fences": self.fences,
                "uncertain_disconnect": self.uncertain_disconnect,
                "retirement": self.retirement,
            }, 200
        if phase == "libraries":
            results = await engine.collective_rpc("peer_lab_libraries", timeout=30)
            return {"results": results}, 200
        if os.environ.get("PEER_LAB_RETIRE_ENABLED") != "1":
            return {"state": "disabled"}, 403
        target, operation = body.get("target_engine_id"), body.get("operation_id")
        expected_decode = body.get("decode_engine_id")
        import re

        if phase not in {"prepare", "commit"} or any(
            not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", v)
            for v in (target, operation, expected_decode)
        ):
            return {"state": "invalid_arguments"}, 400
        if self.active:
            return {"state": "busy", "active_http": self.active}, 409
        if self.uncertain_disconnect:
            return {
                "state": "refused",
                "reason": "aborted_request_cleanup_not_in_scope",
            }, 409
        previous = self.fences.get(target)
        if self.retirement is not None and self.retirement != (target, operation):
            return {"state": "conflict", "reason": "other_retirement_in_progress"}, 409
        if previous is not None and previous["operation_id"] != operation:
            return {"state": "conflict"}, 409
        if phase == "commit" and (previous is None or not previous.get("prepared")):
            return {"state": "prepare_required"}, 409
        if previous is None:
            if len(self.fences) >= 1024:
                return {"state": "tombstone_limit"}, 409
            snapshots = await engine.collective_rpc("peer_lab_snapshot", timeout=30)
            if not self._valid_ranks(snapshots, expected_decode):
                return {"state": "rank_or_generation_mismatch"}, 409
            if not any(target in s["peers"] for s in snapshots):
                return {"state": "unknown_peer"}, 409
            # Recheck after await; a request may have entered while RPC ran.
            if self.active:
                return {"state": "busy", "active_http": self.active}, 409
            if self.uncertain_disconnect:
                return {
                    "state": "refused",
                    "reason": "aborted_request_cleanup_not_in_scope",
                }, 409
            self.fences[target] = {"operation_id": operation, "prepared": False}
        # Freeze all D admissions across every rank's prepare/commit. A failed
        # or partial RPC leaves this freeze in place for explicit lab diagnosis.
        self.retirement = (target, operation)
        results = await engine.collective_rpc(
            "peer_lab_retire",
            timeout=30,
            args=(target, operation, expected_decode, phase),
        )
        wanted = "prepared" if phase == "prepare" else "cleanup_returned"
        # A retried prepare may arrive after some or all ranks already committed.
        # Do not convert an idempotent retry into a permanent admission freeze.
        allowed = {"prepared", "cleanup_returned"} if phase == "prepare" else {wanted}
        complete = self._valid_ranks(results, expected_decode) and all(
            r.get("target_engine_id") == target
            and r.get("operation_id") == operation
            and r.get("state") in allowed
            for r in results
        )
        if phase == "prepare" and complete:
            self.fences[target]["prepared"] = True
        if complete and all(r.get("state") == "cleanup_returned" for r in results):
            wanted = "cleanup_returned"
            self.retirement = None
        return {
            "state": wanted if complete else "pending_or_failed",
            "results": results,
            "native_release_verified": False,
            "gpu_memory_measured": False,
        }, 200 if complete else 409
