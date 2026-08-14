# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState


class TraceReplayState:
    """Per-request state for inference trace-replay.

    When a request carries ``SamplingParams.trace_decode_token_ids``, the
    sampler overwrites the sampled token at each decode step with the
    predetermined trace token, while real logprobs and ranks are still computed
    from the unmodified logit distribution. The replay step for a request is
    derived entirely from GPU state (``total_len - prompt_len``), so no CPU
    synchronization or async placeholder handling is needed.
    """

    def __init__(self, req_states: RequestState):
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device
        self.trace_token_ids = StagedWriteTensor(
            (self.max_num_reqs, req_states.max_model_len),
            dtype=torch.int32,
            device=self.device,
            uva_instead_of_gpu=True,
        )
        self.trace_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        # Sticky: set once any request replays, so the per-step paths can use an
        # O(1) check. Per-request trace_len values protect non-replay requests.
        self.any_trace = False

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        trace = sampling_params.trace_decode_token_ids
        if trace is not None:
            self.trace_len.np[req_idx] = len(trace)
            self.trace_token_ids.stage_write(req_idx, 0, trace)
            self.any_trace = True
        else:
            self.trace_len.np[req_idx] = 0

    def apply_staged_writes(self) -> None:
        if self.any_trace:
            self.trace_len.copy_to_uva()
            self.trace_token_ids.apply_write()

    def apply_trace(
        self,
        sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        total_len: torch.Tensor,
        prompt_len: torch.Tensor,
    ) -> None:
        if not self.any_trace:
            return
        apply_trace_tokens(
            sampled,
            idx_mapping,
            self.trace_token_ids.gpu,
            self.trace_len.gpu,
            total_len,
            prompt_len,
        )


@triton.jit
def _trace_replay_kernel(
    sampled_ptr,  # [num_reqs], int64, mutated in place
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx
    trace_token_ids_ptr,  # [max_num_reqs, max_model_len], int32
    trace_token_ids_stride,
    trace_len_ptr,  # [max_num_reqs], int32
    total_len_ptr,  # [max_num_reqs], int32
    prompt_len_ptr,  # [max_num_reqs], int32
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_state_idx < 0:
        return

    trace_len = tl.load(trace_len_ptr + req_state_idx)
    if trace_len <= 0:
        return

    # The token being sampled now is output token number
    # (total_len - prompt_len): total_len reflects tokens committed through the
    # previous step (post_update runs after sampling).
    step = tl.load(total_len_ptr + req_state_idx) - tl.load(
        prompt_len_ptr + req_state_idx
    )
    if step < 0 or step >= trace_len:
        return

    token_id = tl.load(
        trace_token_ids_ptr + req_state_idx * trace_token_ids_stride + step
    )
    tl.store(sampled_ptr + batch_idx, token_id.to(tl.int64))


def apply_trace_tokens(
    sampled: torch.Tensor,
    idx_mapping: torch.Tensor,
    trace_token_ids: torch.Tensor,
    trace_len: torch.Tensor,
    total_len: torch.Tensor,
    prompt_len: torch.Tensor,
) -> None:
    """Overwrite ``sampled`` in place with trace tokens for the current step."""
    num_reqs = idx_mapping.shape[0]
    _trace_replay_kernel[(num_reqs,)](
        sampled,
        idx_mapping,
        trace_token_ids,
        trace_token_ids.stride(0),
        trace_len,
        total_len,
        prompt_len,
    )
