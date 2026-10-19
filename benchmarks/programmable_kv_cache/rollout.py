# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real HTTP generation and bounded application control for rollout workloads."""

import asyncio
import hashlib
import json
import math
import time
import uuid
from contextlib import suppress
from typing import Any

import httpx

from benchmarks.programmable_kv_cache.client import ACTIVE, envelope


class RolloutClient:
    def __init__(self, base_url: str, mode: str, *, seed: int = 0):
        self.http = httpx.AsyncClient(
            base_url=base_url, timeout=600, limits=httpx.Limits(max_connections=256)
        )
        self.mode = mode
        self.seed = seed
        self.run_id = uuid.uuid4().hex
        self.rows: list[dict[str, Any]] = []
        self.controls: list[dict[str, Any]] = []
        self.turns: dict[str, int] = {}
        self.targets: dict[str, dict[str, Any]] = {}
        self.background: dict[str, asyncio.Task] = {}
        self.resume_events: dict[str, asyncio.Event] = {}
        self.pending_moves: dict[str, str] = {}
        self.invocation: dict[str, str] = {}
        self.boundary = 1

    async def initialize(self) -> None:
        response = await self.http.get("/v1/models")
        response.raise_for_status()
        self.model = response.json()["data"][0]["id"]
        response = await self.http.get("/v1/kv/capabilities")
        response.raise_for_status()
        engines = response.json()["engines"]
        if len(engines) > 1:
            raise ValueError("Use an HTTP endpoint pinned to one explicit engine")
        self.capabilities = engines[0] if engines else None
        if self.capabilities is not None:
            self.invocation = {
                key: self.capabilities[key]
                for key in ("engine_id", "engine_epoch", "namespace")
            }
            self.boundary = math.lcm(
                *(
                    group["block_size"]
                    for group in self.capabilities["cache_profile"]["groups"]
                    if group["prefix_cacheable"]
                )
            )
        elif self.mode not in ("native", "noop"):
            raise ValueError("The selected arm requires programmable KV control")

    async def call(self, method: str, payload: dict) -> Any:
        response = await self.http.post(
            "/v1/kv/control",
            json={
                "method": method,
                "invocation": self.invocation,
                "payload": payload,
            },
        )
        response.raise_for_status()
        return response.json()

    async def wait(
        self,
        message_id: str,
        action_id: str = "action",
        *,
        timeout: float = 30,
        acknowledge: bool = True,
    ) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = await self.call(
                "operation",
                {
                    "message_id": message_id,
                    "action_id": action_id,
                },
            )
            if result["status"] not in ACTIVE:
                children = [
                    await self.wait(
                        child["message_id"], child["action_id"], acknowledge=False
                    )
                    for child in result.get("derived_operations", [])
                ]
                result["children"] = children
                self.controls.append(result)
                messages = {child["message_id"] for child in children}
                if acknowledge:
                    messages.add(message_id)
                for message in messages:
                    try:
                        await self.call("acknowledge", {"message_id": message})
                    except httpx.HTTPStatusError as error:
                        if error.response.json().get("error", {}).get("code") != "busy":
                            raise
                return result
            await asyncio.sleep(0.01)
        raise TimeoutError(f"Control operation timed out: {message_id}")

    async def action(
        self,
        kind: str,
        payload: dict,
        *,
        version="1.0",
        attempts: int = 2,
        submitted: asyncio.Event | None = None,
    ) -> dict:
        for attempt in range(attempts):
            message = f"{self.run_id}-{uuid.uuid4().hex}"
            receipt = None
            started = time.perf_counter()
            try:
                receipt = await self.call(
                    "submit", envelope(message, kind, payload, version)
                )
                if submitted is not None:
                    submitted.set()
                result = await self.wait(message)
            except (httpx.HTTPError, TimeoutError) as error:
                self.controls.append(
                    {"status": "client_error", "error": str(error), "action_type": kind}
                )
                if receipt is not None:
                    for parent in receipt["results"]:
                        for ref in [parent, *parent.get("derived_operations", [])]:
                            with suppress(httpx.HTTPError):
                                cancelled = await self.call(
                                    "cancel",
                                    {
                                        key: ref[key]
                                        for key in (
                                            "message_id",
                                            "action_id",
                                            "operation_id",
                                        )
                                    },
                                )
                                self.controls.append(cancelled)
                if attempt + 1 == attempts:
                    return {"status": "failed", "reason": str(error)}
            else:
                result["driver_seconds"] = time.perf_counter() - started
                if kind == "signal":
                    result["driver_signal"] = payload["signal"]
                    result["driver_resume_after_seconds"] = payload.get(
                        "resume_after_seconds"
                    )
                if result["status"] != "deferred" or attempt + 1 == attempts:
                    return result
            finally:
                if submitted is not None:
                    submitted.set()
            await asyncio.sleep(0.02 * (2**attempt))
        raise AssertionError("Unreachable")

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict,
        *,
        fixed_tokens: int | None = None,
        **kwargs,
    ) -> dict:
        started = time.perf_counter()
        turn = self.turns.get(request_id, 0)
        self.turns[request_id] = turn + 1
        pending = self.background.pop(request_id, None)
        if pending is not None:
            self.resume_events.pop(request_id).set()
            await pending
        continuation = f"turn-{turn}"
        session = f"{self.run_id}-{request_id}"
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt_ids,
            "temperature": sampling_params.get("temperature", 0),
            "top_p": sampling_params.get("top_p", 1),
            "max_tokens": fixed_tokens or sampling_params.get("max_tokens", 256),
            "seed": self.seed
            + turn
            + int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:4], "big")
            % 1000000,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
            "logprobs": 0,
            "cache_salt": self.run_id,
            "session_id": session,
        }
        for key in ("stop", "stop_token_ids", "repetition_penalty"):
            if key in sampling_params:
                body[key] = sampling_params[key]
        if fixed_tokens is not None:
            body["ignore_eos"] = True
            body.pop("stop", None)
            body.pop("stop_token_ids", None)
        association = None
        if self.mode not in ("native", "noop"):
            end = len(prompt_ids) // self.boundary * self.boundary
            if end:
                association = f"{session}-{continuation}"
                body["kv_hints"] = envelope(
                    association,
                    "associate",
                    {
                        "session_id": session,
                        "continuation_id": continuation,
                        "expected_revision": 0,
                        "observe_growth": True,
                        "task_parent_id": kwargs.get("task_parent_id"),
                        "parent_prefix": kwargs.get("parent_prefix"),
                        "target": {
                            "kind": "request",
                            "request_id": "$request",
                            "end_token": end,
                        },
                    },
                    "1",
                )
                self.targets[request_id] = {
                    "kind": "session",
                    "session_id": session,
                    "continuation_id": continuation,
                }
                if self.mode in ("movement", "composed"):
                    body["kv_hints"]["actions"].append(
                        {
                            "action_id": "move",
                            "action_type": "vllm.move_to",
                            "action_version": "1.0",
                            "payload": {
                                "target": {
                                    "kind": "request",
                                    "request_id": "$request",
                                    "end_token": end,
                                },
                                "src": {"tier": "gpu"},
                                "dst": {"tier": "mooncake"},
                            },
                        }
                    )
                    self.pending_moves[request_id] = association
        sent = time.perf_counter()
        first_token = None
        ids, logprobs = [], []
        text = ""
        usage = {}
        finish = None
        async with self.http.stream("POST", "/v1/completions", json=body) as response:
            if response.is_error:
                await response.aread()
                response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                item = json.loads(line[6:])
                if "error" in item:
                    raise RuntimeError(item["error"])
                if item.get("usage"):
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    tokens = choice.get("token_ids") or []
                    if tokens and first_token is None:
                        first_token = time.perf_counter()
                    ids.extend(tokens)
                    text += choice.get("text", "")
                    logprobs.extend(
                        (choice.get("logprobs") or {}).get("token_logprobs", [])
                    )
                    finish = choice.get("finish_reason") or finish
        finished = time.perf_counter()
        if not ids:
            raise RuntimeError("Generation returned no token IDs")
        row = {
            "request_id": request_id,
            "turn": turn,
            "prompt_ids": prompt_ids,
            "token_ids": ids,
            "log_probs": logprobs,
            "text": text,
            "started": started,
            "finished": finished,
            "seconds": finished - started,
            "prepare_wait_seconds": sent - started,
            "ttft": (first_token or finished) - started,
            "usage": usage,
            "finish_reason": finish,
        }
        self.rows.append(row)
        if association is not None:
            await self.wait(
                association, acknowledge=request_id not in self.pending_moves
            )
        return row

    async def pause(self, request_id: str, predicted_seconds: float = 0.25) -> None:
        target = self.targets.get(request_id)
        if target is None:
            return
        resumed = asyncio.Event()
        self.resume_events[request_id] = resumed
        predicted_at = time.monotonic() + predicted_seconds

        async def prepare():
            if self.mode == "priority":
                await self.action(
                    "retain",
                    {
                        "target": target,
                        "owner_id": target["session_id"],
                        "declaration_id": "rollout",
                        "priority": 50,
                        "ttl_seconds": 30,
                    },
                    version="1",
                )
                return
            movement_target = target
            if self.mode != "policy":
                message = self.pending_moves.pop(request_id, None)
                if message is None:
                    return
                saved = await self.wait(message, "move")
                blocks = [
                    {
                        key: row[key]
                        for key in (
                            "block_hash",
                            "group_id",
                            "start_token",
                            "end_token",
                        )
                    }
                    for row in saved.get("target_outcomes", [])
                    if "cold_copy_stored" in row["completed_steps"]
                ]
                if not blocks:
                    return
                movement_target = {
                    "kind": "hashes",
                    "namespace": self.invocation["namespace"],
                    "blocks": blocks,
                }
            if self.mode == "policy":
                result = await self.action(
                    "signal", {"target": target, "signal": "tool_wait"}, version="1"
                )
                if not result.get("derived_operations"):
                    return
                if resumed.is_set():
                    await self.action(
                        "signal", {"target": target, "signal": "resume"}, version="1"
                    )
                    return
                submitted = asyncio.Event()
                prediction = asyncio.create_task(
                    self.action(
                        "signal",
                        {
                            "target": target,
                            "signal": "resume_prediction",
                            "resume_after_seconds": max(
                                0.001, predicted_at - time.monotonic()
                            ),
                        },
                        version="1",
                        attempts=1,
                        submitted=submitted,
                    )
                )
                try:
                    await submitted.wait()
                    await resumed.wait()
                    await self.action(
                        "signal", {"target": target, "signal": "resume"}, version="1"
                    )
                finally:
                    await prediction
                return
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    resumed.wait(), max(0, predicted_at - time.monotonic() - 0.1)
                )
            await self.action(
                "vllm.copy_to",
                {
                    "target": movement_target,
                    "src": {"tier": "mooncake"},
                    "dst": {"tier": "gpu"},
                },
            )
            if self.mode == "composed":
                await self.action(
                    "retain",
                    {
                        "target": target,
                        "owner_id": target["session_id"],
                        "declaration_id": "rollout",
                        "priority": 50,
                        "ttl_seconds": 30,
                    },
                    version="1",
                )

        self.background[request_id] = asyncio.create_task(prepare())

    async def finish(self, request_id: str) -> None:
        pending = self.background.pop(request_id, None)
        if pending is not None:
            self.resume_events.pop(request_id).set()
            await pending
        move = self.pending_moves.pop(request_id, None)
        if move is not None:
            await self.wait(move, "move")
        target = self.targets.pop(request_id, None)
        if target is None:
            return
        try:
            snapshot = await self.call(
                "associations", {"session_id": target["session_id"]}
            )
        except httpx.HTTPStatusError as error:
            if (
                error.response.json().get("error", {}).get("code")
                == "missing_association"
            ):
                return
            raise
        if snapshot["items"]:
            await self.action(
                "finish_task",
                {
                    "session_id": target["session_id"],
                    "expected_revision": dict(snapshot["session_revisions"])[
                        target["session_id"]
                    ],
                },
                version="1",
            )

    async def close(self) -> None:
        try:
            for event in self.resume_events.values():
                event.set()
            if self.background:
                await asyncio.gather(*self.background.values(), return_exceptions=True)
            for identifier in tuple(self.targets):
                await self.finish(identifier)
        finally:
            await self.http.aclose()
