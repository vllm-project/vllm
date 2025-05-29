# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm import LLM

from ...utils import create_new_process_for_each_test


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("backend", ["mp", "ray"])
@create_new_process_for_each_test()
def test_collective_rpc(tp_size, backend, monkeypatch):
    if tp_size == 1 and backend == "ray":
        pytest.skip("Skip duplicate test case")
    if tp_size == 1:
        backend = None

    # intentionally define the method and class in the test function,
    # to test if they can be serialized and sent to the workers
    def echo_rank(self):
        return self.rank

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct",
              enforce_eager=True,
              load_format="dummy",
              tensor_parallel_size=tp_size,
              distributed_executor_backend=backend)
    assert llm.collective_rpc(echo_rank) == list(range(tp_size))
