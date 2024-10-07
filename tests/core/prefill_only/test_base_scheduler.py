import pytest

from vllm.core.prefill_only_scheduler import Scheduler
from vllm.inputs.prefill_only.data import Request
from vllm.model_executor.prefill_only.engine_io import RequestOutput


class Scheduler4Test(Scheduler):

    def schedule(self):
        pass


@pytest.mark.parametrize("n_request", [9, 99, 199])
def test_add_request_and_abort_request(n_request: int):
    scheduler = Scheduler4Test(None, None)

    # add requests
    for i in range(1, n_request + 1):
        scheduler.add_request(Request(request_id=str(i), arrival_time=0.))
        assert len(scheduler.waiting) == i
        assert len(scheduler.requests) == i

    # abort irrelevant requests
    for i in range(1, n_request + 1):
        scheduler.abort_request(request_id=str(100000 + i))
        assert len(scheduler.waiting) == n_request
        assert len(scheduler.requests) == n_request

    # abort requests
    for i in range(1, n_request + 1):
        scheduler.abort_request(request_id=str(i))
        assert len(scheduler.waiting) == n_request
        assert len(scheduler.requests) == n_request - i

    # Lazy abort_request, only test whether to abort during scheduling
    assert len(scheduler.waiting) == n_request
    assert len(scheduler.requests) == 0


@pytest.mark.parametrize("n_request", [9, 99, 199])
def test_remove_abort_request(n_request: int):
    scheduler = Scheduler4Test(None, None)

    request_outputs = []
    for i in range(1, n_request + 1):
        scheduler.add_request(Request(request_id=str(i), arrival_time=0.))
        request_outputs.append(
            RequestOutput(request_id=str(i), arrival_time=0., finished=True))
        assert len(scheduler.waiting) == i
        assert len(scheduler.requests) == i
        assert len(request_outputs) == i

    # abort half of requests
    for i in range(1, n_request // 2):
        scheduler.abort_request(request_id=str(i))
        assert len(scheduler.waiting) == n_request
        assert len(scheduler.requests) == n_request - i

    finished_requests = scheduler.remove_abort_request(request_outputs)
    assert len(finished_requests) == n_request - n_request // 2 + 1
    assert len(scheduler.requests) == n_request - n_request // 2 + 1
    assert len(scheduler.aborted_requests) == 0

    finished_request_ids = set(request.request_id
                               for request in finished_requests
                               if request.finished)

    assert len(finished_request_ids - scheduler.requests) == 0
    assert len(scheduler.requests - finished_request_ids) == 0

    scheduler.free_finished_request(finished_requests)
    assert len(scheduler.requests) == 0
