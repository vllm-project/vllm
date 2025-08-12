# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.core.scheduler import Scheduler
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.v1.core.scheduler import Scheduler as V1Scheduler
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine


class DummyV0Scheduler(Scheduler):

    def schedule(self):
        raise Exception("Exception raised by DummyV0Scheduler")


class DummyV1Scheduler(V1Scheduler):

    def schedule(self):
        raise Exception("Exception raised by DummyV1Scheduler")


def test_scheduler_plugins_v0(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "0")
        with pytest.raises(Exception) as exception_info:

            engine_args = EngineArgs(
                model="facebook/opt-125m",
                enforce_eager=True,  # reduce test time
                scheduler_cls=DummyV0Scheduler,
            )

            engine = LLMEngine.from_engine_args(engine_args=engine_args)

            sampling_params = SamplingParams(max_tokens=1)
            engine.add_request("0", "foo", sampling_params)
            engine.step()

        assert str(
            exception_info.value) == "Exception raised by DummyV0Scheduler"


def test_scheduler_plugins_v1(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        # Explicitly turn off engine multiprocessing so
        # that the scheduler runs in this process
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

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

        assert str(
            exception_info.value) == "Exception raised by DummyV1Scheduler"
