# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


class ASDRiskState:
    """GPU regret ledger whose request lifetime includes preemption."""

    def __init__(self, max_num_reqs: int, device: torch.device):
        self.spent = torch.zeros(max_num_reqs, dtype=torch.float64, device=device)
        self.request_slots: dict[str, int] = {}
        self.suspended: dict[str, torch.Tensor] = {}

    def add_request(self, req_id: str, req_idx: int) -> None:
        previous_idx = self.request_slots.get(req_id)
        if previous_idx is not None:
            # Streaming input can re-add an existing request.
            self.spent[req_idx].copy_(self.spent[previous_idx])
        elif req_id in self.suspended:
            self.spent[req_idx].copy_(self.suspended.pop(req_id))
        else:
            self.spent[req_idx].zero_()
        self.request_slots[req_id] = req_idx

    def finish_requests(
        self, finished_req_ids: set[str], preempted_req_ids: set[str]
    ) -> None:
        for req_id in finished_req_ids:
            self.request_slots.pop(req_id, None)
            self.suspended.pop(req_id, None)
        for req_id in preempted_req_ids - finished_req_ids:
            req_idx = self.request_slots.pop(req_id, None)
            if req_idx is not None:
                self.suspended[req_id] = self.spent[req_idx].clone()
