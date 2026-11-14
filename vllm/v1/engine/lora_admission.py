# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Capacity-bounded LoRA epochs shared by internal-DP engine processes."""

import heapq
import weakref
from concurrent.futures import Future
from typing import Any

import msgspec
import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


def gather_metadata(group: ProcessGroup, value: Any) -> list[Any]:
    """Gather CPU metadata without requiring a registered default process group."""
    encoded = msgspec.msgpack.encode(value)
    size = torch.tensor([len(encoded)], dtype=torch.int64, device="cpu")
    sizes = [torch.empty_like(size) for _ in range(group.size())]
    group.allgather([sizes], [size]).wait()
    buffer = torch.zeros(
        max(int(item.item()) for item in sizes), dtype=torch.uint8, device="cpu"
    )
    buffer[: len(encoded)] = torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
    buffers = [torch.empty_like(buffer) for _ in sizes]
    group.allgather([buffers], [buffer]).wait()
    return [
        msgspec.msgpack.decode(item[: int(length.item())].numpy().tobytes())
        for item, length in zip(buffers, sizes)
    ]


class GroupLoRAAdmission:
    def __init__(self, capacity: int, priority: bool = False):
        self.capacity = capacity
        self.priority = priority
        self.ready: frozenset[int] = frozenset()
        self.pinned: set[int] = set()
        self.identities: dict[int, bytes] = {}
        self.fingerprints: dict[int, str] = {}
        self.failed: dict[int, str] = {}
        self.registrations: dict[tuple, int] = {}
        self.next_id = 1
        self.requests: dict[
            int, list[tuple[tuple, weakref.ReferenceType[Request]]]
        ] = {}
        self.live: dict[int, weakref.WeakSet[Request]] = {}
        self.blocked: list[weakref.ReferenceType[Request]] = []
        self.heads: list[tuple[tuple, int]] = []
        self.head_keys: dict[int, tuple] = {}
        self.pending: dict[str, tuple[str, Any, Future]] = {}
        self.target: str | None = None
        self.admitted: frozenset[int] = frozenset()
        self.epoch = 0
        self.max_observed_demand = 0

    def track(self, request: Request) -> None:
        lora = request.lora_request
        if lora is None or request.is_finished():
            return
        key = (
            request.priority if self.priority else 0,
            request.arrival_time,
            request.request_id,
        )
        queue = self.requests.setdefault(lora.lora_int_id, [])
        self.live.setdefault(lora.lora_int_id, weakref.WeakSet()).add(request)
        heapq.heappush(queue, (key, weakref.ref(request)))
        if queue[0][0] == key and self.head_keys.get(lora.lora_int_id) != key:
            self.head_keys[lora.lora_int_id] = key
            heapq.heappush(self.heads, (key, lora.lora_int_id))

    def _candidates(self, exclude: frozenset[int]) -> list[tuple[tuple, bytes]]:
        saved = []
        candidates: list[tuple[tuple, bytes]] = []
        while self.heads and len(candidates) < self.capacity:
            key, adapter_id = heapq.heappop(self.heads)
            if self.head_keys.get(adapter_id) != key:
                continue
            queue = self.requests[adapter_id]
            if len(queue) > max(64, 2 * len(self.live[adapter_id])):
                queue[:] = [
                    entry
                    for entry in queue
                    if (req := entry[1]()) is not None and not req.is_finished()
                ]
                heapq.heapify(queue)
            while queue:
                req = queue[0][1]()
                if req is not None and not req.is_finished():
                    if req.status in {
                        RequestStatus.WAITING,
                        RequestStatus.RUNNING,
                        RequestStatus.PREEMPTED,
                    }:
                        break
                    self.blocked.append(queue[0][1])
                heapq.heappop(queue)
            if not queue:
                del self.requests[adapter_id]
                del self.head_keys[adapter_id]
                continue
            if queue[0][0] != key:
                self.head_keys[adapter_id] = queue[0][0]
                heapq.heappush(self.heads, (queue[0][0], adapter_id))
                continue
            saved.append((key, adapter_id))
            if adapter_id not in exclude and adapter_id not in self.failed:
                req = queue[0][1]()
                assert req is not None
                candidates.append((key, msgspec.msgpack.encode(req.lora_request)))
        for item in saved:
            heapq.heappush(self.heads, item)
        return candidates

    def snapshot(self, running: list[Request]) -> dict:
        blocked, self.blocked = self.blocked, []
        for reference in blocked:
            req = reference()
            if req is not None and not req.is_finished():
                if req.status in {
                    RequestStatus.WAITING,
                    RequestStatus.RUNNING,
                    RequestStatus.PREEMPTED,
                }:
                    self.track(req)
                else:
                    self.blocked.append(reference)
        command = self.pending.get(self.target) if self.target is not None else None
        return {
            "running": sorted(
                {req.lora_request.lora_int_id for req in running if req.lora_request}
            ),
            "candidates": self._candidates(frozenset()),
            "outside": bool(self._candidates(self.ready)),
            "head": min(self.pending) if self.pending else None,
            "target": self.target,
            "command": command[:2] if command else None,
        }

    def enqueue(self, operation: str, method: str, argument: Any) -> Future:
        if method not in {"add_lora", "remove_lora", "pin_lora"}:
            raise ValueError(f"Unsupported group adapter operation: {method}")
        if method == "add_lora":
            request = msgspec.convert(argument, type=LoRARequest)
            if request.load_inplace:
                raise ValueError(
                    "Group admission requires immutable adapter IDs/weights"
                )
            argument = msgspec.msgpack.encode(request)
        future: Future = Future()
        self.pending[operation] = method, argument, future
        return future

    def plan(self, frames: list[dict]) -> tuple[str, Any] | None:
        running = {adapter for frame in frames for adapter in frame["running"]}
        demand = running | {
            msgspec.msgpack.decode(encoded, type=LoRARequest).lora_int_id
            for frame in frames
            for _, encoded in frame["candidates"]
        }
        if len(demand) > self.max_observed_demand:
            self.max_observed_demand = len(demand)
            logger.info(
                "Group LoRA admission: demand=%d budget=%d", len(demand), self.capacity
            )
        if not running <= self.ready:
            raise RuntimeError("A running LoRA request escaped group admission")
        if self.target is None:
            operations = [frame["head"] for frame in frames if frame["head"]]
            if operations:
                self.target = min(operations)
        if self.target is not None:
            self.admitted = frozenset()
            commands = [
                frame["command"]
                for frame in frames
                if frame["target"] == self.target and frame["command"] is not None
            ]
            if running or len(commands) != len(frames):
                return None
            if any(command != commands[0] for command in commands):
                raise RuntimeError("Group adapter operation identities disagree")
            method, argument = commands[0]
            if method == "add_lora":
                request = msgspec.msgpack.decode(argument, type=LoRARequest)
                request.lora_int_id = self.registrations.get(
                    self._registration_key(request), self.next_id
                )
                argument = msgspec.msgpack.encode(request)
            if (
                method == "pin_lora"
                and argument not in self.pinned
                and len(self.pinned) == self.capacity
            ):
                self.reject(
                    (method, argument), "All group LoRA slots are already pinned"
                )
                return None
            return method, argument
        selected = dict.fromkeys(sorted(self.pinned | running))
        for _, encoded in sorted(
            candidate for frame in frames for candidate in frame["candidates"]
        ):
            lora = msgspec.msgpack.decode(encoded, type=LoRARequest)
            previous = self.identities.setdefault(lora.lora_int_id, encoded)
            self.next_id = max(self.next_id, lora.lora_int_id + 1)
            if previous != encoded:
                raise RuntimeError(
                    f"Conflicting identity for adapter {lora.lora_int_id}"
                )
            if len(selected) < self.capacity:
                selected[lora.lora_int_id] = None
        desired = frozenset(selected)
        if desired == self.ready or (running and desired <= self.ready):
            self.admitted = (
                self.ready - running
                if running and any(frame["outside"] for frame in frames)
                else self.ready
            )
            return None
        self.admitted = frozenset()
        return "prepare", [self.identities[adapter] for adapter in sorted(desired)]

    @staticmethod
    def _registration_key(request: LoRARequest) -> tuple:
        return (
            request.lora_name,
            request.lora_path,
            request.base_model_name,
            request.is_3d_lora_weight,
        )

    def validate(self, action: tuple[str, Any], reports: list[dict]) -> dict[int, str]:
        if not reports or any(report["error"] for report in reports):
            errors = [report["error"] for report in reports if report["error"]]
            raise RuntimeError(
                f"Group LoRA preparation failed; admission blocked: {errors}"
            )
        identities: dict[int, str] = {}
        for report in reports:
            for adapter, fingerprint in report["fingerprints"].items():
                old = identities.setdefault(adapter, fingerprint)
                if old != fingerprint:
                    raise RuntimeError(
                        f"Adapter {adapter} has different weights across workers"
                    )
                if self.fingerprints.get(adapter, fingerprint) != fingerprint:
                    raise RuntimeError(
                        f"Adapter {adapter} changed its immutable weights"
                    )
        method, argument = action
        if method in {"add_lora", "pin_lora"}:
            expected = {
                msgspec.msgpack.decode(argument, type=LoRARequest).lora_int_id
                if method == "add_lora"
                else argument
            }
            if any(set(report["fingerprints"]) != expected for report in reports):
                raise RuntimeError(
                    "Not every worker acknowledged the registered adapter"
                )
        if method == "prepare":
            desired = frozenset(
                msgspec.msgpack.decode(item, type=LoRARequest).lora_int_id
                for item in argument
            )
            if any(set(report["fingerprints"]) != desired for report in reports):
                raise RuntimeError(
                    "Not every worker acknowledged the complete LoRA epoch"
                )
        return identities

    def reject(self, action: tuple[str, Any], error: str) -> set[int]:
        self.admitted = self.ready
        message = f"Group LoRA update failed; previous epoch restored: {error}"
        if action[0] == "prepare":
            failed = {
                msgspec.msgpack.decode(item, type=LoRARequest).lora_int_id
                for item in action[1]
            } - self.ready
            if not failed:
                raise RuntimeError(message)
            self.failed.update(dict.fromkeys(failed, message))
            return failed
        assert self.target is not None
        _, _, future = self.pending.pop(self.target)
        self.target = None
        future.set_exception(ValueError(message))
        return set()

    def publish(self, action: tuple[str, Any], reports: list[dict]) -> None:
        identities = self.validate(action, reports)
        self.fingerprints.update(identities)
        method, argument = action
        if method == "prepare":
            desired = frozenset(identities)
            self.ready = self.admitted = desired
            self.epoch += 1
        else:
            if method == "add_lora":
                lora = msgspec.msgpack.decode(argument, type=LoRARequest)
                self.identities[lora.lora_int_id] = argument
                self.registrations[self._registration_key(lora)] = lora.lora_int_id
                self.next_id = max(self.next_id, lora.lora_int_id + 1)
                self.failed.pop(lora.lora_int_id, None)
            elif method == "pin_lora":
                self.pinned.add(argument)
            elif method == "remove_lora":
                self.pinned.discard(argument)
            self.ready = frozenset()
            assert self.target is not None
            _, _, future = self.pending.pop(self.target)
            self.target = None
            future.set_result(
                lora.lora_int_id
                if method == "add_lora"
                else any(report["result"] for report in reports)
            )
