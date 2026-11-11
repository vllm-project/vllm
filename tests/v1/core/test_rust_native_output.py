# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The native output path (`VLLM_RUST_CORE_NATIVE_IO`): the Rust scheduler
sends the outputs of each step through `EngineOutputWriter`, encoded in the
Rust frontend's wire types, with the content of the Python scheduler's
`EngineCoreOutputs`."""

import itertools
from typing import Any

import msgspec
import numpy as np
import pytest
import torch
import zmq

from tests.v1.core.rust_core_diff import FuzzConfig, build_scheduler, run_one
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, ModelRunnerOutput
from vllm.v1.serial_utils import MsgpackDecoder

pytest.importorskip("vllm._rust_core")

from vllm._rust_core import EngineOutputWriter  # noqa: E402

from vllm.v1.core.sched.rust_scheduler import (  # noqa: E402
    RustAsyncScheduler,
    RustScheduler,
)

pytestmark = pytest.mark.cpu_test

_MARK = b"step-done"
_ipc_ids = itertools.count()


class _Frontend:
    """The frontend's end of an engine's output socket, and the engine's
    native output thread connected to it."""

    def __init__(self, ipc_dir: str):
        self.socket = zmq.Context.instance().socket(zmq.PULL)
        address = f"ipc://{ipc_dir}/output-{next(_ipc_ids)}"
        self.socket.bind(address)
        # A low threshold, so that arrays also go in auxiliary frames.
        self.writer = EngineOutputWriter([address], None, 0, 16)

    def recv_step(self) -> list[list[bytes]]:
        """The messages sent before a marker sent after them."""
        self.writer.send_encoded(0, [_MARK])
        messages: list[list[bytes]] = []
        while True:
            assert self.socket.poll(10_000), "no output within 10 s"
            frames = self.socket.recv_multipart()
            if frames == [_MARK]:
                return messages
            messages.append(frames)


def _native(scheduler_cls: type[RustScheduler], ipc_dir: str):
    """`scheduler_cls` whose `update_from_output` goes through the native
    output path, returning what the Python frontend decodes."""

    class Native(scheduler_cls):  # type: ignore[valid-type, misc]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.frontend = _Frontend(ipc_dir)
            self.set_output_writer(self.frontend.writer)
            self.decoder = MsgpackDecoder(EngineCoreOutputs)

        def update_from_output(self, scheduler_output, model_runner_output):
            self.update_from_output_native(scheduler_output, model_runner_output, None)
            outputs = {}
            for frames in self.frontend.recv_step():
                assert 0 not in outputs, "one batch per client and step"
                outputs[0] = self.decoder.decode(frames)
            return outputs

    return Native


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("sample", ["base", "extended", "nixl", "structured"])
@pytest.mark.parametrize("seed", range(2))
def test_matches_python_scheduler(seed: int, sample: str, use_async: bool, tmp_path):
    """Every step's outputs, decoded by the Python frontend's decoder, match
    the Python scheduler's (`run_one` compares them as the scheduler tests
    do: outputs, finished requests, cache stats)."""
    cfg = getattr(FuzzConfig, f"sample_{sample}" if sample != "base" else "sample")(
        seed
    )
    ref_cls, test_cls = (
        (AsyncScheduler, RustAsyncScheduler)
        if use_async
        else (Scheduler, RustScheduler)
    )
    native_cls = _native(test_cls, str(tmp_path))
    ok, msg = run_one(
        cfg,
        lambda cfg: build_scheduler(cfg, ref_cls),
        lambda cfg: build_scheduler(cfg, native_cls),
        use_async=use_async,
    )
    assert ok, msg


def _wire_array(value: list[Any], frames: list[bytes]) -> np.ndarray:
    """An `(dtype, shape, data)` array of the wire, decoded without the Python
    decoder's types: the Rust frontend reads numpy dtypes and Torch names
    alike."""
    dtype, shape, data = value
    if isinstance(data, int):
        data = frames[data]
    else:
        assert isinstance(data, msgspec.msgpack.Ext) and data.code == 3
        data = data.data
    dtype = {"int64": "<i8", "float32": "<f4", "int32": "<i4"}.get(dtype, dtype)
    return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)


def _wire_logprobs(value: list[Any], frames: list[bytes]) -> list[np.ndarray]:
    """`(logprob_token_ids, logprobs, ranks)` of a logprobs tuple."""
    assert value[3:] in ([], [None], [None, None])
    return [_wire_array(v, frames) for v in value[:3]]


# Positions of `EngineCoreOutput` fields on the wire.
_REQUEST_ID, _NEW_LOGPROBS, _NEW_PROMPT_LOGPROBS = 0, 2, 3


def test_logprobs_payloads(tmp_path):
    """The logprobs and prompt logprobs Python holds as numpy arrays and
    tensors are sliced per request and sent as the Python scheduler's are."""
    num_logprobs = 2
    schedulers = {}
    for name, cls in (("ref", Scheduler), ("test", RustScheduler)):
        scheduler = create_scheduler(scheduler_cls=cls)
        for request in create_requests(
            num_requests=3, num_tokens=8, max_tokens=4, prompt_logprobs=1
        ):
            request.sampling_params.logprobs = num_logprobs
            scheduler.add_request(request)
        schedulers[name] = scheduler
    frontend = _Frontend(str(tmp_path))
    schedulers["test"].set_output_writer(frontend.writer)

    so_ref = schedulers["ref"].schedule()
    so_test = schedulers["test"].schedule()
    req_ids = list(so_ref.num_scheduled_tokens)
    assert req_ids == list(so_test.num_scheduled_tokens)
    num_reqs = len(req_ids)
    rng = np.random.default_rng(0)
    # One generated position per request; rows are requests.
    logprobs = LogprobsLists(
        logprob_token_ids=rng.integers(0, 100, (num_reqs, num_logprobs + 1)),
        logprobs=rng.standard_normal((num_reqs, num_logprobs + 1)).astype(np.float32),
        sampled_token_ranks=rng.integers(1, 10, num_reqs),
    )
    prompt_logprobs = {
        rid: LogprobsTensors(
            logprob_token_ids=torch.arange(14, dtype=torch.int64).reshape(7, 2) + i,
            logprobs=torch.full((7, 2), -float(i), dtype=torch.float32),
            selected_token_ranks=torch.arange(7, dtype=torch.int64),
        )
        for i, rid in enumerate(req_ids)
    }
    output = ModelRunnerOutput(
        req_ids=req_ids,
        sampled_token_ids=[[int(t)] for t in logprobs.logprob_token_ids[:, 0]],
        logprobs=logprobs,
        prompt_logprobs_dict=prompt_logprobs,
    )

    expected = schedulers["ref"].update_from_output(so_ref, output)[0].outputs
    schedulers["test"].update_from_output_native(so_test, output, "v1")
    [frames] = frontend.recv_step()
    assert len(frames) > 1, "large arrays go in auxiliary frames"
    batch = msgspec.msgpack.decode(frames[0])
    assert batch[-1] == "v1"  # weight_version
    actual = batch[1]
    assert [o[_REQUEST_ID] for o in actual] == [o.request_id for o in expected]
    for got, want in zip(actual, expected):
        got_logprobs = _wire_logprobs(got[_NEW_LOGPROBS], frames)
        want_logprobs = want.new_logprobs[:3]
        for g, w in zip(got_logprobs, want_logprobs):
            np.testing.assert_array_equal(g, w)
        got_prompt = _wire_logprobs(got[_NEW_PROMPT_LOGPROBS], frames)
        want_prompt = want.new_prompt_logprobs_tensors[:3]
        for g, w in zip(got_prompt, want_prompt):
            np.testing.assert_array_equal(g, w.numpy())
