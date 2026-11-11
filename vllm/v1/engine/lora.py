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
        self.engine.engines_running = True
        self.engine.output_queue.put_nowait(
            (-1, EngineCoreOutputs(start_wave=self.engine.current_wave))
        )
        return future

    def _gather(self, value: Any) -> list[Any]:
        values: list[Any] = [None] * self.engine.dp_size
        dist.all_gather_object(values, value, group=self.engine.dp_group)
        return values

    def _commit(self, candidate: DPLoRASlots) -> None:
        payload = candidate.payload()
        if payload == self.state.payload():
            self.state = candidate
            return
        error = None
        try:
            self.engine.model_executor.collective_rpc(
                "prepare_dp_loras", args=(payload,)
            )
        except Exception as exc:
            error = str(exc)
        errors = self._gather(error)
        if any(errors):
            self.engine.model_executor.collective_rpc("abort_dp_loras")
            raise ValueError(f"DP LoRA preparation failed: {errors}")
        error = None
        try:
            self.engine.model_executor.collective_rpc("commit_dp_loras")
        except Exception as exc:
            error = str(exc)
        errors = self._gather(error)
        if any(errors):
            # Every worker retains the previous CPU models until commit succeeds.
            # Roll back successful peers as well as any partially written worker.
            self.engine.model_executor.collective_rpc("rollback_dp_loras")
            self._gather(None)
            raise ValueError(f"DP LoRA commit failed: {errors}")
        self.engine.model_executor.collective_rpc("finalize_dp_loras")
        self.state = candidate

    def _reject(self, usage: list[LoRAUsage], ids: set[int], message: str) -> None:
        for item in usage:
            if item.request.lora_int_id not in ids:
                continue
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
                invalid.add(adapter_id)
            requests.setdefault(adapter_id, request)
            if item.ready and (
                adapter_id in allowed or len(allowed) < len(self.state.slots)
            ):
                allowed.add(adapter_id)
        if invalid:
            self._reject(local, invalid, "Conflicting or unsupported LoRA request")
            allowed -= invalid
        selected = [
            requests[i] for i in sorted(allowed) if i in requests and i not in invalid
        ]
        candidate = deepcopy(self.state)
        try:
            candidate.acquire(selected, allowed)
            self._commit(candidate)
        except ValueError as exc:
            failed = {r.lora_int_id for r in selected}
            self._reject(local, failed, str(exc))
            allowed -= failed
        self.engine.scheduler.set_allowed_loras(allowed)

        for key in list(self.pending):
            success, operation_result = self.results[key]
            _, _, future = self.pending.pop(key)
            if success:
                future.set_result(operation_result)
            else:
                future.set_exception(ValueError(operation_result))
        for key in list(self.results):
            if len(self.seen[key]) == self.engine.dp_size:
                del self.results[key]
                del self.seen[key]
