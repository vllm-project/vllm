# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coordinate LoRA updates at the synchronous DP execution boundary."""

from concurrent.futures import Future
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import msgspec
import torch.distributed as dist

from vllm.lora.dp_manager import DPLoRASlots, LoRAUsage, adapter_identity
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestError
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.engine.core import DPEngineCoreProc


class DPLoRACoordinator:
    def __init__(self, engine: "DPEngineCoreProc"):
        self.engine = engine
        config = engine.vllm_config.lora_config
        assert config is not None
        self.state = DPLoRASlots(config.max_loras)
        self.pending: dict[str, tuple[str, bytes, Future]] = {}
        self.results: dict[str, tuple[bool, bool | str]] = {}
        self.seen: dict[str, set[int]] = {}

    def update(self, operation_id: str, kind: str, payload: bytes) -> Future:
        future: Future = Future()
        self.pending[operation_id] = kind, payload, future
        if operation_id in self.results:
            # Advertise this rank's receipt on the next normal synchronization.
            success, operation_result = self.results[operation_id]
            if success:
                future.set_result(operation_result)
            else:
                future.set_exception(ValueError(operation_result))
            return future
        self.engine.engines_running = True
        self.engine.output_queue.put_nowait(
            (-1, EngineCoreOutputs(start_wave=self.engine.current_wave))
        )
        return future

    def _gather(self, value: Any) -> list[Any]:
        values: list[Any] = [None] * self.engine.dp_size
        dist.all_gather_object(values, value, group=self.engine.dp_group)
        return values

    def _worker_errors(self, method: str, *args: Any) -> list[str]:
        results = self.engine.model_executor.collective_rpc(method, args=args)
        return [
            error
            for rank_errors in self._gather(results)
            for error in rank_errors
            if error is not None
        ]

    def _commit(self, candidate: DPLoRASlots) -> None:
        payload = candidate.payload()
        extra_requests = [
            slot.request
            for adapter_id, slot in candidate.adapters.items()
            if slot not in candidate.slots
            and slot != self.state.adapters.get(adapter_id)
        ]
        if payload == self.state.payload() and not extra_requests:
            self.state = candidate
            return
        errors = self._worker_errors("prepare_dp_loras", payload, extra_requests)
        if errors:
            self.engine.model_executor.collective_rpc("abort_dp_loras")
            raise ValueError(f"DP LoRA preparation failed: {errors}")
        errors = self._worker_errors("commit_dp_loras")
        if errors:
            # Every worker retains the previous CPU models until commit succeeds.
            # Roll back successful peers as well as any partially written worker.
            self.engine.model_executor.collective_rpc("rollback_dp_loras")
            self._gather(None)
            raise ValueError(f"DP LoRA commit failed: {errors}")
        self.engine.model_executor.collective_rpc("finalize_dp_loras")
        self.state = candidate

    def _reject(self, usage: list[LoRAUsage], message: str) -> None:
        for item in usage:
            self.engine.scheduler.finish_requests(
                [item.request_id], RequestStatus.FINISHED_ERROR
            )
            self.engine._send_finish_outputs_to_client(
                [item.request_id],
                item.client_index,
                FinishReason.ERROR,
                RequestError(code="invalid_lora", message=message, retryable=False),
            )

    def synchronize(self) -> None:
        local = self.engine.scheduler.get_lora_usage()
        commands = [
            (key, kind, payload) for key, (kind, payload, _) in self.pending.items()
        ]
        gathered = self._gather((msgspec.msgpack.encode(local), commands))
        usage = [
            item
            for packed, _ in gathered
            for item in msgspec.msgpack.decode(packed, type=list[LoRAUsage])
        ]
        referenced = {item.request.lora_int_id for item in usage}
        running = {item.request.lora_int_id for item in usage if item.running}
        operations = {}
        for rank, (_, updates) in enumerate(gathered):
            for key, kind, payload in updates:
                operations[key] = kind, payload
                self.seen.setdefault(key, set()).add(rank)
        for key, (kind, payload) in sorted(operations.items()):
            if key in self.results:
                continue
            candidate = deepcopy(self.state)
            try:
                result = candidate.update(kind, payload, referenced, running)
                self._commit(candidate)
                self.results[key] = True, result
            except ValueError as exc:
                self.results[key] = False, str(exc)

        # Keep running adapters resident; admit waiting adapters in global FCFS order.
        allowed = running | self.state.pinned
        requests: dict[int, LoRARequest] = {}
        invalid = set()
        for item in sorted(
            usage, key=lambda u: (u.priority, u.arrival_time, u.request_id)
        ):
            request = item.request
            adapter_id = request.lora_int_id
            try:
                self.state.validate(request)
                if request.load_inplace:
                    raise ValueError(
                        "Use add_lora to replace adapters between requests"
                    )
                previous = requests.get(adapter_id)
                if previous and adapter_identity(previous) != adapter_identity(request):
                    raise ValueError("Conflicting adapter identity")
            except ValueError:
                invalid.add(item.request_id)
                continue
            requests.setdefault(adapter_id, request)
            if item.ready and (
                adapter_id in allowed or len(allowed) < len(self.state.slots)
            ):
                allowed.add(adapter_id)
        if invalid:
            self._reject(
                [item for item in local if item.request_id in invalid],
                "Conflicting or unsupported LoRA request",
            )
        selected = [requests[i] for i in sorted(allowed) if i in requests]
        resident = {
            slot.request.lora_int_id for slot in self.state.slots if slot is not None
        }
        for request in selected:
            adapter_id = request.lora_int_id
            if adapter_id in resident:
                continue
            candidate = deepcopy(self.state)
            try:
                candidate.acquire([request], allowed)
                self._commit(candidate)
            except ValueError as exc:
                self._reject(
                    [item for item in local if item.request.lora_int_id == adapter_id],
                    str(exc),
                )
                allowed.discard(adapter_id)
        self.state.acquire([r for r in selected if r.lora_int_id in allowed], allowed)
        self.engine.scheduler.set_allowed_loras(allowed)

        for key in list(self.pending):
            success, operation_result = self.results[key]
            _, _, future = self.pending.pop(key)
            if future.done():
                continue
            if success:
                future.set_result(operation_result)
            else:
                future.set_exception(ValueError(operation_result))
        for key in list(self.results):
            if len(self.seen[key]) == self.engine.dp_size:
                del self.results[key]
                del self.seen[key]
