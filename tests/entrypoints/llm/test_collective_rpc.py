import pytest

from vllm import LLM
from vllm.worker.worker import Worker

from ...utils import fork_new_process_for_each_test


def echo_rank(self):
    return self.rank


class MyWorker(Worker):

    def echo_rank(self):
        return self.rank


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("backend", ["mp", "ray"])
@fork_new_process_for_each_test
def test_collective_rpc(tp_size, backend):
    if tp_size == 1 and backend == "ray":
        pytest.skip("Skip duplicate test case")
    if tp_size == 1:
        backend = None
    llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct",
              enforce_eager=True,
              load_format="dummy",
              tensor_parallel_size=tp_size,
              distributed_executor_backend=backend,
              worker_cls=MyWorker)
    for method in ["echo_rank", echo_rank]:
        assert llm.collective_rpc(method) == list(range(tp_size))
