# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sampler import gumbel_sample
from vllm.v1.worker.gpu.states import SamplingMetadata


class EagleSpeculator:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.method = self.speculative_config.method
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.draft_model_config = self.speculative_config.draft_model_config

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens

        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

    def load_model(self, target_model: nn.Module) -> None:
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("eagle_head"):
            self.model = get_model(
                vllm_config=self.vllm_config, model_config=self.draft_model_config
            )

        share_lm_head = True
        if share_lm_head and hasattr(target_model, "lm_head"):
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_model.lm_head

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        sampling_metadata: SamplingMetadata,
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [max_num_reqs, 1]
        last_sampled: torch.Tensor,
        # [num_reqs]
        next_prefill_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if aux_hidden_states:
            assert self.method == "eagle3"
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states

        # Get the input ids and last token indices for the speculator.
        last_token_indices = prepare_eagle_inputs(
            self.input_ids,
            input_batch,
            num_sampled,
            last_sampled,
            next_prefill_tokens,
        )
        input_ids = self.input_ids[: input_batch.num_tokens_after_padding]

        # Prefill: Run the eagle speculator with eager mode.
        with set_forward_context(
            input_batch.attn_metadata,
            self.vllm_config,
            num_tokens=input_batch.num_tokens_after_padding,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ):
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=input_batch.positions,
                hidden_states=hidden_states,
            )
        if self.method == "mtp":
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        num_reqs = input_batch.num_reqs
        cu_num_logits = input_batch.cu_num_logits[:num_reqs]
        temperature = sampling_metadata.temperature[cu_num_logits]
        seed = sampling_metadata.seeds[cu_num_logits]
        # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise
        # used for draft and target sampling.
        pos = input_batch.positions[last_token_indices] + 1
        draft_tokens = gumbel_sample(
            logits, temperature, seed, pos, apply_temperature=True
        )
        if self.num_speculative_steps == 1:
            # Early exit.
            return draft_tokens.view(-1, 1)
        raise NotImplementedError("num_speculative_steps > 1 is not supported yet.")


@triton.jit
def _prepare_eagle_inputs_kernel(
    last_token_indices_ptr,
    eagle_input_ids_ptr,
    target_input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    query_start_loc_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    # Get the true query length and next token after accounting for rejected tokens.
    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    if num_sampled > 0:
        req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)

        logits_start = tl.load(cu_num_logits_ptr + batch_idx)
        logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
        num_logits = logits_end - logits_start

        num_rejected = num_logits - num_sampled
        query_len -= num_rejected
    else:
        # Chunked prefilling.
        # Get the next prefill token.
        next_token = tl.load(next_prefill_tokens_ptr + batch_idx)

    # Shift target_input_ids by one.
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(eagle_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)

    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + batch_idx, last_token_index)
    tl.store(eagle_input_ids_ptr + last_token_index, next_token)


def prepare_eagle_inputs(
    eagle_input_ids: torch.Tensor,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [max_num_reqs, 1]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
) -> torch.Tensor:
    num_reqs = input_batch.num_reqs
    last_token_indices = torch.empty(
        num_reqs,
        dtype=torch.int64,
        device=eagle_input_ids.device,
    )
    _prepare_eagle_inputs_kernel[(num_reqs,)](
        last_token_indices,
        eagle_input_ids,
        input_batch.input_ids,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        input_batch.query_start_loc,
        input_batch.cu_num_logits,
        BLOCK_SIZE=1024,
    )
    return last_token_indices
