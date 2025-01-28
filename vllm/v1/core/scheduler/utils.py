from typing import Dict, List, Set

from vllm.v1.core.scheduler.interface import (NewRequestData,
                                              ResumedRequestData,
                                              RunningRequestData)
from vllm.v1.request import Request, RequestStatus


class CommonSchedulerState:

    def __init__(self):
        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: Set[str] = set()

        # OPTIMIZATION: Cache the RunningRequestData objects to avoid creating
        # them at each scheduling step.
        # Request id -> RunningRequestData
        self.running_reqs_data: Dict[str, RunningRequestData] = {}

    def make_running_req_data(
        self,
        requests: List[Request],
        req_to_new_block_ids: Dict[str, List[int]],
    ) -> List[RunningRequestData]:
        # OPTIMIZATION: Cache the RunningRequestData objects to avoid creating
        # them at each scheduling step.
        req_data_list: List[RunningRequestData] = []
        for request in requests:
            req_id = request.request_id
            if req_id in self.running_reqs_data:
                req_data = self.running_reqs_data[req_id]
                req_data.new_block_ids = req_to_new_block_ids[req_id]
                req_data.num_computed_tokens = request.num_computed_tokens
            else:
                req_data = RunningRequestData.from_request(
                    request,
                    req_to_new_block_ids[req_id],
                    request.num_computed_tokens,
                )
                self.running_reqs_data[req_id] = req_data
            req_data_list.append(req_data)
        return req_data_list

    def make_new_req_data(
        self,
        requests: List[Request],
        req_to_new_block_ids: Dict[str, List[int]],
    ) -> List[NewRequestData]:
        return [
            NewRequestData.from_request(
                req,
                req_to_new_block_ids[req.request_id],
                req.num_computed_tokens,
            ) for req in requests
        ]

    def make_resumed_req_data(
        self,
        requests: List[Request],
        req_to_new_block_ids: Dict[str, List[int]],
    ) -> List[ResumedRequestData]:
        return [
            ResumedRequestData.from_request(
                req,
                req_to_new_block_ids[req.request_id],
                req.num_computed_tokens,
            ) for req in requests
        ]

    def free(self, request_id: str) -> None:
        self.finished_req_ids.add(request_id)
        self.running_reqs_data.pop(request_id, None)


def check_stop(request: Request, max_model_len: int) -> bool:
    if (request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    sampling_params = request.sampling_params
    last_token_id = request.output_token_ids[-1]
    if (not sampling_params.ignore_eos
            and last_token_id == request.eos_token_id):
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    return False
