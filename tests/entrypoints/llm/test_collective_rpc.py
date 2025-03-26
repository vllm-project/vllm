# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm import LLM

from ...utils import create_new_process_for_each_test

# intentionally define the method and class in the test function,
# to test if they can be serialized and sent to the workers


# UPDATE: move outside because in V1 multiprocessing mode,
# the function has to be global to be serialized by pickle
# in the other process
# TODO: find a secure way to serialize the function
def echo_rank(self):
    return self.rank


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("backend", ["mp", "ray"])
@create_new_process_for_each_test()
def test_collective_rpc(tp_size, backend):
    if tp_size == 1 and backend == "ray":
        pytest.skip("Skip duplicate test case")
    if tp_size == 1:
        backend = None

    llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct",
              enforce_eager=True,
              load_format="dummy",
              tensor_parallel_size=tp_size,
              distributed_executor_backend=backend)
    assert llm.collective_rpc(echo_rank) == list(range(tp_size))
