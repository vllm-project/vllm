# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


@triton.jit
def _rewind_sampled_state_kernel(
    req_indices_ptr,
    num_output_tokens_ptr,
    prompt_len_ptr,
    total_len_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
):
    batch_idx = tl.program_id(0)
    req_idx = tl.load(req_indices_ptr + batch_idx)
    num_output_tokens = tl.load(num_output_tokens_ptr + batch_idx)
    target_total_len = tl.load(prompt_len_ptr + req_idx) + num_output_tokens
    old_total_len = tl.load(total_len_ptr + req_idx)

    # The scheduler only marks requests whose accepted output prefix moved
    # backward. Tokens in this suffix were sampled by in-flight frames whose
    # outputs the scheduler discarded.
    for pos in tl.range(target_total_len, old_total_len):
        token_id = tl.load(all_token_ids_ptr + req_idx * all_token_ids_stride + pos)
        if output_bin_counts_ptr is not None:
            tl.atomic_add(
                output_bin_counts_ptr + req_idx * output_bin_counts_stride + token_id,
                -1,
            )
        tl.store(all_token_ids_ptr + req_idx * all_token_ids_stride + pos, 0)

    last_sampled_token = 0
    if num_output_tokens > 0:
        last_sampled_token = tl.load(
            all_token_ids_ptr + req_idx * all_token_ids_stride + target_total_len - 1
        )
    tl.store(last_sampled_tokens_ptr + req_idx, last_sampled_token)
    tl.store(total_len_ptr + req_idx, target_total_len)


class RequestState:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
        num_prefill_lookahead: int = 1,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size
        self.device = device

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_reqs))

        # NOTE(woosuk): This tensor can be extremely large (e.g., several GBs)
        # depending on the configured max_num_reqs and max_model_len.
        # To save GPU memory, we use UVA instead of GPU for this tensor.
        self.all_token_ids = StagedWriteTensor(
            (self.max_num_reqs, self.max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )
        # NOTE(woosuk): Distinguish clearly between prompt_len and prefill_len:
        # - prompt_len: Number of tokens in the user-provided prompt.
        # - prefill_len: Number of tokens passed into the model runner.
        #   This can include the prompt and additional partial output tokens,
        #   so prefill_len >= prompt_len.
        # Usually, prefill_len equals prompt_len, but in cases such as resumption after
        # preemption, prefill_len may be greater. Differentiating between these values
        # is crucial, as certain features such as prompt logprobs or frequency penalties
        # must treat prompt and output tokens separately.
        self.prompt_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.prefill_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        # total_len = prompt_len + output_len. It grows as the request progresses.
        self.total_len = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

        # Number of computed tokens.
        self.num_computed_prefill_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        # Optimistic CPU mirror of num_computed_tokens (upper bound on GPU value).
        self.num_computed_tokens_np = np.zeros(self.max_num_reqs, dtype=np.int32)

        # Last sampled tokens.
        self.last_sampled_tokens = torch.zeros(
            self.max_num_reqs, 1, dtype=torch.int64, device=device
        )

        # Max total seq length (prompt_len + max_tokens).
        self.max_seq_len = np.zeros(self.max_num_reqs, dtype=np.int32)

        # Draft tokens.
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        self.next_prefill_tokens = torch.zeros(
            num_prefill_lookahead,
            self.max_num_reqs,
            dtype=torch.int32,
            device=device,
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
        max_tokens: int,
    ) -> None:
        assert len(self.free_indices) > 0, "No free indices"
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        self.max_seq_len[req_idx] = prompt_len + max_tokens
        self.prompt_len.np[req_idx] = prompt_len
        prefill_len = len(all_token_ids)
        assert prefill_len >= prompt_len, (
            f"prefill_len {prefill_len} < prompt_len {prompt_len}"
        )
        self.prefill_len.np[req_idx] = prefill_len
        self.total_len.stage_write_elem(req_idx, prefill_len)
        self.all_token_ids.stage_write(req_idx, 0, all_token_ids)
        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        self.num_computed_tokens_np[req_idx] = num_computed_tokens
        self.num_computed_tokens.stage_write_elem(req_idx, num_computed_tokens)

        self.draft_tokens[req_idx].zero_()

    def apply_staged_writes(self) -> None:
        self.prompt_len.copy_to_uva()
        self.prefill_len.copy_to_uva()
        self.total_len.apply_write()
        self.all_token_ids.apply_write()
        self.num_computed_tokens.apply_write()

    def rewind_sampled_state(
        self,
        req_indices: list[int],
        num_output_tokens: list[int],
        output_bin_counts: torch.Tensor | None,
    ) -> None:
        """Restore each request to the scheduler's accepted output prefix."""
        assert req_indices
        assert len(req_indices) == len(num_output_tokens)
        assert all(num_tokens >= 0 for num_tokens in num_output_tokens)

        req_indices_gpu = async_tensor_h2d(
            req_indices, dtype=torch.int32, device=self.device
        )
        num_output_tokens_gpu = async_tensor_h2d(
            num_output_tokens, dtype=torch.int32, device=self.device
        )
        _rewind_sampled_state_kernel[(len(req_indices),)](
            req_indices_gpu,
            num_output_tokens_gpu,
            self.prompt_len.gpu,
            self.total_len.gpu,
            self.all_token_ids.gpu,
            self.all_token_ids.gpu.stride(0),
            self.last_sampled_tokens,
            output_bin_counts,
            output_bin_counts.stride(0) if output_bin_counts is not None else 0,
            num_warps=1,
        )

        req_indices_long = req_indices_gpu.long()
        self.draft_tokens.index_fill_(0, req_indices_long, 0)
        self.next_prefill_tokens.index_fill_(1, req_indices_long, 0)

    def remove_request(self, req_id: str) -> int | None:
        """Return the freed slot index, or None if the request was not found."""
        req_idx = self.req_id_to_index.pop(req_id, None)
        if req_idx is None:
            return None
        self.index_to_req_id.pop(req_idx, None)
        self.free_indices.append(req_idx)
        return req_idx
