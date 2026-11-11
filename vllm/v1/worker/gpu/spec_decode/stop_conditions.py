# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request stop conditions on the device, for truncating speculative acceptance.

With `speculative_config.truncate_acceptance_at_stop` the rejection sampler
cuts each request's accepted tokens at its stop, so the last accepted token of
a step is the request's last token and the device-side state that follows the
accepted tokens (KV cache, recurrent state, computed-token count) ends there.
The conditions mirror the scheduler's `check_stop`: the EOS token (unless
`ignore_eos`), the stop token ids, and the length limit
`min(prompt_len + max_tokens, max_model_len)`. Stop strings and repetition
detection are not mirrored.
"""

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.sample.logit_bias import MAX_NUM_STOP_TOKEN_IDS
from vllm.v1.worker.gpu.states import RequestState


class StopConditions:
    def __init__(self, req_states: RequestState) -> None:
        self.prompt_len_np = req_states.prompt_len.np
        self.max_model_len = req_states.max_model_len
        max_num_reqs = req_states.max_num_reqs
        # -1 when EOS does not stop the request (ignore_eos, or no EOS token).
        self.eos_token_id = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        # The request's total length (prompt and output) at its length limit.
        self.max_len = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.num_stop_token_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.stop_token_ids = StagedWriteTensor(
            (max_num_reqs, MAX_NUM_STOP_TOKEN_IDS),
            dtype=torch.int32,
            device=req_states.device,
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        prompt_len = int(self.prompt_len_np[req_idx])
        # None under ignore_eos.
        eos_token_id = sampling_params.eos_token_id
        self.eos_token_id.np[req_idx] = -1 if eos_token_id is None else eos_token_id
        max_len = self.max_model_len
        if sampling_params.max_tokens is not None:
            max_len = min(max_len, prompt_len + sampling_params.max_tokens)
        self.max_len.np[req_idx] = max_len
        stop_token_ids = sampling_params.stop_token_ids or []
        if len(stop_token_ids) > MAX_NUM_STOP_TOKEN_IDS:
            # Not mirrored: a stop token id inside the accepted drafts is
            # then trimmed by the scheduler as without truncation.
            stop_token_ids = []
        self.num_stop_token_ids.np[req_idx] = len(stop_token_ids)
        self.stop_token_ids.stage_write(req_idx, 0, stop_token_ids)

    def apply_staged_writes(self) -> None:
        self.eos_token_id.copy_to_uva()
        self.max_len.copy_to_uva()
        self.num_stop_token_ids.copy_to_uva()
        self.stop_token_ids.apply_write()
