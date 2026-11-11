# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The native input path (`VLLM_RUST_CORE_NATIVE_IO`): `EngineInputReader`
admits the add requests that meet no Python collaborator as
`NativeAddRequest`s, without a Python `Request`, and the Rust scheduler runs
them as it runs the Python ones; every other message goes to the engine's
`_process_input_message` with its original frames."""

import dataclasses
import itertools
import queue
import threading
from typing import Any

import numpy as np
import pytest
import zmq

from tests.v1.core.rust_core_diff import FuzzConfig, build_scheduler, run_one
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType, PrefixHashConfig
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytest.importorskip("vllm._rust_core")

from vllm._rust_core import EngineInputReader, NativeAddRequest  # noqa: E402

from vllm.v1.core.sched.rust_scheduler import (  # noqa: E402
    RustAsyncScheduler,
    RustScheduler,
)

pytestmark = pytest.mark.cpu_test

_IDENTITY = b"\x00\x00"
_ipc_ids = itertools.count()


class _FakeEngine:
    """What `EngineInputReader` uses of `EngineCoreProc`."""

    def __init__(self):
        self.input_queue: queue.Queue = queue.Queue()
        self.slow_path: list[tuple[bytes, list[bytes]]] = []

    def _process_input_message(self, type_frame: bytes, data_frames: list) -> None:
        self.slow_path.append((type_frame, data_frames))
        self.input_queue.put_nowait(("slow", type_frame))


class _Frontend:
    """The frontend's end of an engine's input socket, and the engine's native
    input thread connected to it."""

    def __init__(self, ipc_dir: str, prefix_hash_config: PrefixHashConfig | None):
        self.socket = zmq.Context.instance().socket(zmq.ROUTER)
        address = f"ipc://{ipc_dir}/input-{next(_ipc_ids)}"
        self.socket.bind(address)
        self.engine = _FakeEngine()
        ready = threading.Event()
        self.reader = EngineInputReader(
            self.engine,
            [address],
            None,
            _IDENTITY,
            b"ready-response",
            ready,
            MsgpackDecoder(SamplingParams).decode,
            MsgpackDecoder(LoRARequest).decode,
            prefix_hash_config,
        )
        assert self.socket.recv_multipart() == [_IDENTITY, b"ready-response"]
        assert ready.wait(10)
        self.encoder = MsgpackEncoder()

    def send(self, request_type: EngineCoreRequestType, payload: Any) -> Any:
        """Send a message; returns what the engine queued for it."""
        frames = self.encoder.encode(payload)
        self.socket.send_multipart([_IDENTITY, request_type.value, *frames])
        return self.engine.input_queue.get(timeout=10)


def _engine_core_request(request: Request) -> EngineCoreRequest:
    """The add request a frontend sends for `request`."""
    return EngineCoreRequest(
        request_id=request.request_id,
        prompt_token_ids=np.asarray(request.prompt_token_ids, dtype=np.uint32),
        mm_features=None,
        sampling_params=request.sampling_params,
        pooling_params=None,
        arrival_time=request.arrival_time,
        lora_request=request.lora_request,
        cache_salt=None,
        data_parallel_rank=None,
        priority=request.priority,
    )


def _native(scheduler_cls: type[RustScheduler], ipc_dir: str, block_size: int):
    """`scheduler_cls` whose requests arrive through the native input path."""

    class Native(scheduler_cls):  # type: ignore[valid-type, misc]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.frontend = _Frontend(
                ipc_dir,
                PrefixHashConfig("sha256_cbor", block_size, kv_cache_utils.NONE_HASH),
            )
            self.num_native = 0
            self.num_native_lora = 0

        def add_request(self, request):
            if not isinstance(request, Request) or request.resumable:
                return super().add_request(request)
            request_type, item = self.frontend.send(
                EngineCoreRequestType.ADD, _engine_core_request(request)
            )
            if request_type == "slow":
                # Requests that need Python preprocessing: structured outputs.
                assert request.use_structured_output
                return super().add_request(request)
            native, wave = item
            assert isinstance(native, NativeAddRequest) and wave == 0
            assert native.lora_request == request.lora_request
            self.num_native += 1
            self.num_native_lora += request.lora_request is not None
            super().add_request(native)

        def finish_requests(self, request_ids, finished_status):
            finished = super().finish_requests(request_ids, finished_status)
            # The engine reads only the id, client index and status of a
            # finished fast-path request; compare the rest through the
            # reference's request, which is not finished yet.
            out = []
            for request in finished:
                if isinstance(request, NativeAddRequest):
                    assert request.status == finished_status
                    request = self.requests.reference[request.request_id]
                out.append(request)
            return out

    return Native


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("sample", ["base", "extended", "structured", "lora"])
@pytest.mark.parametrize("seed", range(3))
def test_matches_python_scheduler(seed: int, sample: str, use_async: bool, tmp_path):
    """Fast-path requests run as the Python scheduler runs their Python
    requests, through preemption, aborts and the end of every request."""
    cfg = getattr(FuzzConfig, f"sample_{sample}" if sample != "base" else "sample")(
        seed
    )
    # The deployments of the native input path.
    cfg = dataclasses.replace(cfg, hash_fn="sha256_cbor", connector=None)
    ref_cls, test_cls = (
        (AsyncScheduler, RustAsyncScheduler)
        if use_async
        else (Scheduler, RustScheduler)
    )
    native_cls = _native(test_cls, str(tmp_path), cfg.block_size)
    schedulers: list[Any] = []

    def make_ref(cfg):
        schedulers.append(build_scheduler(cfg, ref_cls))
        return schedulers[-1]

    def make_test(cfg):
        test = build_scheduler(cfg, native_cls)
        # The fake model reads request state through `requests`, which holds
        # no Python request for fast-path requests: let it read the
        # reference's, while every step's outputs are compared.
        test.requests = _RequestsWithReference(test.requests, schedulers[0].requests)
        schedulers.append(test)
        return test

    ok, msg = run_one(cfg, make_ref, make_test, use_async=use_async)
    assert ok, msg
    assert schedulers[1].num_native > 0
    if cfg.lora:
        assert schedulers[1].num_native_lora > 0


class _RequestsWithReference(dict):
    """`requests` of a scheduler with fast-path requests: the reference
    scheduler's request where the scheduler has a `NativeAddRequest`."""

    def __init__(self, requests, reference):
        super().__init__()
        self._requests = requests
        self.reference = reference

    def __getitem__(self, request_id):
        request = self._requests[request_id]
        if isinstance(request, NativeAddRequest):
            return self.reference[request_id]
        return request

    def __iter__(self):
        return iter(self._requests)

    def __len__(self):
        return len(self._requests)

    def items(self):
        return ((rid, self[rid]) for rid in self._requests)


def test_other_messages_take_the_slow_path(tmp_path):
    """Aborts, utility calls and add requests that need Python preprocessing
    reach `_process_input_message` with their original frames."""
    frontend = _Frontend(str(tmp_path), None)
    encoder = MsgpackEncoder()
    structured = EngineCoreRequest(
        request_id="structured",
        prompt_token_ids=np.arange(4, dtype=np.uint32),
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=4, extra_args={"kv_transfer_params": {"do_remote_prefill": True}}
        ),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    messages = [
        (EngineCoreRequestType.ADD, structured),
        (EngineCoreRequestType.ABORT, ["a", "b"]),
        (EngineCoreRequestType.UTILITY, (0, 1, "is_sleeping", ())),
    ]
    for request_type, payload in messages:
        assert frontend.send(request_type, payload) == ("slow", request_type.value)
    for (request_type, payload), (type_frame, frames) in zip(
        messages, frontend.engine.slow_path
    ):
        assert type_frame == request_type.value
        assert [bytes(f) for f in frames] == [bytes(f) for f in encoder.encode(payload)]
