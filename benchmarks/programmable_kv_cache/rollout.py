# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real HTTP generation and bounded application control for rollout workloads."""

import asyncio
import hashlib
import json
import math
import time
import uuid
from collections import OrderedDict
from contextlib import suppress
from typing import Any

import httpx

from benchmarks.programmable_kv_cache.client import ACTIVE, envelope


class ToolDurationEstimator:
    """Estimate a tool batch using only previously completed batches."""

    def __init__(self, initial_seconds: float, max_batches: int = 64):
        if not math.isfinite(initial_seconds) or initial_seconds <= 0:
            raise ValueError("Initial tool duration must be positive and finite")
        if max_batches < 1:
            raise ValueError("Tool batch history must be positive")
        self.initial_seconds = initial_seconds
        self.max_batches = max_batches
        self.history: OrderedDict[tuple[str, ...], float] = OrderedDict()

    def predict(self, tools: tuple[str, ...]) -> float:
        return self.history.get(tuple(sorted(tools)), self.initial_seconds)

    def observe(self, tools: tuple[str, ...], seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Observed tool duration must be nonnegative and finite")
        key = tuple(sorted(tools))
        previous = self.history.get(key, seconds)
        self.history[key] = max(0.001, 0.75 * previous + 0.25 * seconds)
        self.history.move_to_end(key)
        while len(self.history) > self.max_batches:
            self.history.popitem(last=False)


class RolloutClient:
    def __init__(
        self,
        base_url: str,
        mode: str,
        *,
        seed: int = 0,
        trace_rpc: bool = False,
        control_timeout: float = 30,
    ):
        self.http = httpx.AsyncClient(
            base_url=base_url, timeout=600, limits=httpx.Limits(max_connections=256)
        )
        self.mode = mode
        if not math.isfinite(control_timeout) or control_timeout <= 0:
            raise ValueError("Control timeout must be positive and finite")
        self.control_timeout = control_timeout
        self.seed = seed
        self.run_id = uuid.uuid4().hex
        self.rows: list[dict[str, Any]] = []
        self.controls: list[dict[str, Any]] = []
        self.turns: dict[str, int] = {}
        self.targets: dict[str, dict[str, Any]] = {}
        self.background: dict[str, list[asyncio.Task]] = {}
        self.preparations: dict[str, asyncio.Task] = {}
        self.resume_events: dict[str, asyncio.Event] = {}
        self.pending_moves: dict[str, str] = {}
        self.invocation: dict[str, str] = {}
        self.boundary = 1
        self.rpc_stats: dict[str, dict[str, float]] = {}
        self.trace_rpc = trace_rpc
        self.rpc_rows: list[dict[str, Any]] = []

    def _track(self, request_id: str, task: asyncio.Task) -> None:
        self.background.setdefault(request_id, []).append(task)

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
        elif self.mode not in ("native", "idle", "noop"):
            raise ValueError("The selected arm requires programmable KV control")

    async def call(self, method: str, payload: dict) -> Any:
        started = time.perf_counter()
        response = None
        try:
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
        finally:
            elapsed = time.perf_counter() - started
            stats = self.rpc_stats.setdefault(
                method, {"calls": 0, "seconds": 0, "max_seconds": 0}
            )
            stats["calls"] += 1
            stats["seconds"] += elapsed
            stats["max_seconds"] = max(stats["max_seconds"], elapsed)
            if self.trace_rpc:
                self.rpc_rows.append(
                    {
                        "method": method,
                        "started": started,
                        "seconds": elapsed,
                        "message_id": payload.get("message_id"),
                        "action_id": payload.get("action_id"),
                        "session_id": payload.get("session_id"),
                        "status_code": response.status_code
                        if response is not None
                        else None,
                    }
                )

    async def wait(
        self,
        message_id: str,
        action_id: str = "action",
        *,
        timeout: float | None = None,
        acknowledge: bool = True,
        initial: dict | None = None,
        await_registration: bool = False,
    ) -> dict:
        deadline = time.monotonic() + (
            self.control_timeout if timeout is None else timeout
        )
        registration_delay = 0.01
        while time.monotonic() < deadline:
            try:
                result = (
                    initial
                    if initial is not None
                    else await self.call(
                        "operation",
                        {"message_id": message_id, "action_id": action_id},
                    )
                )
            except httpx.HTTPStatusError as error:
                if not (
                    await_registration
                    and error.response.status_code == 404
                    and error.response.json().get("error", {}).get("code") == "missing"
                ):
                    raise
                await asyncio.sleep(registration_delay)
                registration_delay = min(0.25, registration_delay * 2)
                continue
            await_registration = False
            initial = None
            if result["status"] not in ACTIVE:
                children = [
                    await self.wait(
                        child["message_id"],
                        child["action_id"],
                        acknowledge=False,
                        timeout=max(0, deadline - time.monotonic()),
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
                result = await self.wait(message, initial=receipt["results"][0])
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
                result["driver_started"] = started
                result["driver_action_type"] = kind
                result["driver_target"] = payload.get("target")
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
        resumed = self.resume_events.pop(request_id, None)
        if resumed is not None:
            resumed.set()
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
        if self.mode not in ("native", "idle", "noop", "quota"):
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
            if association is not None:
                self._track(
                    request_id,
                    asyncio.create_task(
                        self.wait(
                            association,
                            acknowledge=request_id not in self.pending_moves,
                            await_registration=True,
                        )
                    ),
                )
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
            "association_message_id": association,
        }
        self.rows.append(row)
        return row

    async def pause(self, request_id: str, predicted_seconds: float = 0.25) -> None:
        target = self.targets.get(request_id)
        if target is None:
            return
        resumed = self.resume_events.setdefault(request_id, asyncio.Event())
        predicted_at = time.monotonic() + predicted_seconds
        message = self.pending_moves.pop(request_id, None)
        previous = self.preparations.get(request_id)

        async def prepare():
            if previous is not None:
                await previous
            if self.mode == "retention":
                await self.action(
                    "signal", {"target": target, "signal": "importance"}, version="1"
                )
                return
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
                    "signal",
                    {
                        "target": target,
                        "signal": "tool_wait",
                        "resume_after_seconds": max(
                            0.001, predicted_at - time.monotonic()
                        ),
                    },
                    version="1",
                )
                if not result.get("derived_operations"):
                    await resumed.wait()
                    await self.action(
                        "signal", {"target": target, "signal": "resume"}, version="1"
                    )
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

        task = asyncio.create_task(prepare())
        self.preparations[request_id] = task
        self._track(request_id, task)

    async def drain(self, request_id: str) -> None:
        resumed = self.resume_events.pop(request_id, None)
        if resumed is not None:
            resumed.set()
        pending = self.background.pop(request_id, [])
        self.preparations.pop(request_id, None)
        if pending:
            outcomes = await asyncio.gather(*pending, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome

    async def finish(self, request_id: str) -> None:
        await self.drain(request_id)
        move = self.pending_moves.pop(request_id, None)
        if move is not None:
            await self.wait(move, "move")
        target = self.targets.get(request_id)
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
                self.targets.pop(request_id, None)
                return
            raise
        if snapshot["items"]:
            result = await self.action(
                "finish_task",
                {
                    "session_id": target["session_id"],
                    "expected_revision": dict(snapshot["session_revisions"])[
                        target["session_id"]
                    ],
                },
                version="1",
            )
            if "associations_removed" not in result.get("completed_steps", []):
                raise RuntimeError(f"Task cleanup did not complete: {result}")
        self.targets.pop(request_id, None)

    async def close(self) -> None:
        try:
            for event in self.resume_events.values():
                event.set()
            outcomes = await asyncio.gather(
                *(task for tasks in self.background.values() for task in tasks),
                return_exceptions=True,
            )
            self.background.clear()
            self.preparations.clear()
            failure = next((r for r in outcomes if isinstance(r, BaseException)), None)
            for identifier in tuple(self.targets):
                try:
                    await self.finish(identifier)
                except Exception as error:
                    failure = failure or error
            if failure is not None:
                raise failure
        finally:
            await self.http.aclose()
