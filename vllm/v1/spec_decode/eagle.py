# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata


class EagleProposer:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.model = None
        self.vllm_config = vllm_config
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens)
        self.block_size = vllm_config.cache_config.block_size
        self.arange = torch.arange(vllm_config.scheduler_config.max_num_seqs,
                                   device=device)

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [num_tokens]
        target_slot_mapping: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        # [batch_size + 1] starting with 0
        cu_num_tokens: torch.Tensor,
        # [batch_size, max_num_blocks_per_req]
        block_table: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:  # [batch_size, num_speculative_tokens]
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        last_token_indices = cu_num_tokens[1:] - 1

        input_ids = torch.empty_like(target_token_ids)
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        input_ids[:-1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        input_ids[last_token_indices] = next_token_ids

        seq_lens = target_positions[last_token_indices] + 1
        # FIXME(woosuk): The below two ops cause synchronization. Optimize.
        max_seq_len = seq_lens.max().item()
        max_num_tokens = (cu_num_tokens[1:] - cu_num_tokens[:-1]).max().item()
        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=max_num_tokens,
            query_start_loc=cu_num_tokens,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=target_slot_mapping,
            # TODO(woosuk): Support cascade attention.
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )

        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=input_ids,
                hidden_states=target_hidden_states,
                positions=target_positions,
            )
        sample_hidden_states = hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)

        # Sample the next token.
        draft_token_ids = sample_token_ids(logits, sampling_metadata)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            return draft_token_ids.view(-1, 1)

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]
        positions = target_positions[last_token_indices]
        hidden_states = sample_hidden_states
        attn_metadata.num_actual_tokens = batch_size
        attn_metadata.max_query_len = 1
        attn_metadata.query_start_loc = self.arange[:batch_size]
        for _ in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            input_ids = draft_token_ids_list[-1]
            positions += 1
            attn_metadata.max_seq_len += 1
            attn_metadata.seq_lens += 1
            # Compute the slot mapping.
            block_numbers = positions // self.block_size
            block_ids = block_table.gather(dim=1,
                                           index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            attn_metadata.slot_mapping = (block_ids * self.block_size +
                                          positions % self.block_size)

            # Run the model.
            with set_forward_context(attn_metadata, self.vllm_config):
                hidden_states = self.model(
                    input_ids=input_ids,
                    hidden_states=hidden_states,
                    positions=positions,
                )
            logits = self.model.compute_logits(hidden_states, None)
            draft_token_ids = sample_token_ids(logits, sampling_metadata)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        return torch.stack(draft_token_ids_list, dim=1)

    @staticmethod
    def prepare_inputs(
        # [batch_size + 1]
        cu_target_query_lens: torch.Tensor,
        # [batch_size]
        num_rejected_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cu_target_query_lens: [0, a, a + b, a + b + c]
        # num_rejected_tokens: [n1, n2, n3]
        # num_tokens_per_req: [a - n1, b - n2, c - n3]
        # cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
        # token_indices: [0, 1, ..., a - n1 - 1,
        #                 a, a + 1, ..., a + b - n2 - 1,
        #                 a + b, a + b + 1, ..., a + b + c - n3 - 1]

        # [0, a, a + b, a + b + c] -> [a, b, c]
        query_len_per_req = (cu_target_query_lens[1:] -
                             cu_target_query_lens[:-1])
        # [a, b, c] -> [a - n1, b - n2, c - n3]
        num_tokens_per_req = query_len_per_req - num_rejected_tokens

        cu_num_tokens = torch.empty_like(cu_target_query_lens)
        torch.cumsum(num_tokens_per_req, dim=0, out=cu_num_tokens[1:])
        cu_num_tokens[0] = 0

        # FIXME(woosuk): Avoid synchronization.
        num_tokens = cu_num_tokens[-1].item()
        token_indices = torch.empty(
            num_tokens,
            dtype=torch.int32,
            device=cu_num_tokens.device,
        )

        batch_size = num_rejected_tokens.shape[0]
        BLOCK_SIZE = 1024
        prepare_input_kernel[(batch_size, )](
            token_indices,
            cu_target_query_lens,
            cu_num_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return cu_num_tokens, token_indices


# TODO(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def sample_token_ids(
    logits: torch.Tensor,  # [batch_size, vocab_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:  # [batch_size]
    # NOTE(woosuk): We don't need to apply all the sampling parameters
    # for generating the draft tokens.
    if sampling_metadata.all_greedy:
        # All greedy.
        next_token_ids = logits.argmax(dim=-1)
    else:
        logits.div_(sampling_metadata.temperature)
        probs = logits.softmax(dim=-1, dtype=torch.float32)

        # TODO(woosuk): Consider seeds?
        q = torch.empty_like(logits)
        q.exponential_()
        next_token_ids = probs.div_(q).argmax(dim=-1).view(-1)

        if not sampling_metadata.all_random:
            greedy_token_ids = logits.argmax(dim=-1)
            next_token_ids = torch.where(
                sampling_metadata.temperature == -1,
                greedy_token_ids,
                next_token_ids,
            )
    return next_token_ids


@triton.jit
def prepare_input_kernel(
    out_ptr,
    cu_query_lens_ptr,
    cu_num_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # [start_pos, end_pos)
    start_pos = tl.load(cu_num_tokens_ptr + pid)
    end_pos = tl.load(cu_num_tokens_ptr + pid + 1)
    num_tokens = end_pos - start_pos

    index_start = tl.load(cu_query_lens_ptr + pid)
    indices = index_start + tl.arange(0, BLOCK_SIZE)

    num_blocks = tl.cdiv(num_tokens, BLOCK_SIZE)
    for i in tl.range(num_blocks):
        offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(
            out_ptr + start_pos + offset,
            indices,
            mask=offset < num_tokens,
        )
