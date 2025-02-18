# SPDX-License-Identifier: Apache-2.0

from vllm.core.scheduler import Scheduler


class DummyScheduler(Scheduler):

    def schedule(self):
        raise Exception("Exception raised by DummyScheduler")


def test_scheduler_plugins():
    import pytest

    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.llm_engine import LLMEngine
    from vllm.sampling_params import SamplingParams

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
