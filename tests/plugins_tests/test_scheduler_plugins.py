# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest

from vllm.core.scheduler import Scheduler
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.v1.core.scheduler import Scheduler as V1Scheduler
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine


class DummyScheduler(Scheduler):

    def schedule(self):
        raise Exception("Exception raised by DummyScheduler")


class DummyV1Scheduler(V1Scheduler):

    def schedule(self):
        raise Exception("Exception raised by DummyScheduler")


def test_scheduler_plugins(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V1", "0")
    with pytest.raises(Exception) as exception_info:

        engine_args = EngineArgs(
            model="facebook/opt-125m",
            enforce_eager=True,  # reduce test time
            scheduler_cls=DummyScheduler,
        )

        engine = LLMEngine.from_engine_args(engine_args=engine_args)

        sampling_params = SamplingParams(max_tokens=1)
        engine.add_request("0", "foo", sampling_params)
        engine.step()

    assert str(exception_info.value) == "Exception raised by DummyScheduler"


def test_scheduler_plugins_v1(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V1", "1")

    # V1 engine has more redirection- the worker process will raise with our
    # dummy scheduler error but then client in this process will try to kill
    # the process tree when the worker fails.
    with mock.patch(
            "vllm.v1.engine.core_client.kill_process_tree") as mock_kill:
        mock_kill.side_effect = Exception("kill_process_tree was called")

        with pytest.raises(Exception) as exception_info:

            engine_args = EngineArgs(
                model="facebook/opt-125m",
                enforce_eager=True,  # reduce test time
                scheduler_cls=DummyV1Scheduler,
            )

            engine = V1LLMEngine.from_engine_args(engine_args=engine_args)

            sampling_params = SamplingParams(max_tokens=1)
            engine.add_request("0", "foo", sampling_params)
            engine.step()

    assert str(exception_info.value) == "kill_process_tree was called"
