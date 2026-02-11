# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.eagle_cudagraph import EagleCudaGraphManager

logger = init_logger(__name__)


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
        self.max_model_len = vllm_config.model_config.max_model_len
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.idx_mapping = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.temperature = torch.zeros(
            self.max_num_reqs, dtype=torch.float32, device=device
        )
        self.seeds = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        self.cudagraph_manager = EagleCudaGraphManager(vllm_config, device)

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

    def set_attn(
        self,
        kv_cache_config: KVCacheConfig,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        block_tables: BlockTables,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.attn_metadata_builders = attn_metadata_builders
        self.block_tables = block_tables

    @torch.inference_mode()
    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
        ):
            ret_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
            )
        if self.method == "mtp":
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

    def generate_draft(
        self,
        num_reqs: int,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        pos = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]
        for step in range(1, self.num_speculative_steps):
            # Run the eagle model.
            last_hidden_states, hidden_states = self.run_model(
                num_reqs, attn_metadata, slot_mappings, num_tokens_across_dp
            )
            logits = self.model.compute_logits(last_hidden_states)

            # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise
            # used for draft and target sampling.
            draft_tokens = gumbel_sample(
                logits,
                idx_mapping,
                self.temperature,
                self.seeds,
                pos + 1,
                apply_temperature=True,
            )
            self.draft_tokens[:num_reqs, step] = draft_tokens

            if step < self.num_speculative_steps - 1:
                # Update the inputs for the next step.
                update_eagle_inputs(
                    draft_tokens,
                    hidden_states,
                    self.input_buffers,
                    self.hidden_states,
                    self.max_model_len,
                )
                self.block_tables.compute_slot_mappings(
                    idx_mapping, query_start_loc, pos
                )

    def capture_model(self) -> None:
        if self.num_speculative_steps == 1:
            return
        logger.info("Capturing model for Eagle speculator...")
        self.cudagraph_manager.capture(
            self.generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_metadata_builders,
            self.kv_cache_config,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        # P/D injection data (optional).
        injection_data: dict[str, dict] | None = None,
    ) -> torch.Tensor:
        # NOTE(woosuk): To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of eagle's input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions. By doing so,
        # we can also reuse the attention metadata (e.g., query_start_loc,
        # seq_lens) of the target model.
        if aux_hidden_states:
            assert self.method == "eagle3"
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states

        num_reqs = input_batch.num_reqs
        idx_mapping = self.idx_mapping[:num_reqs]
        idx_mapping.copy_(input_batch.idx_mapping)
        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)

        # ── P/D injection: expanded EAGLE "prefill" for warm-up ──
        if injection_data:
            return self._propose_with_injection(
                input_batch,
                hidden_states,
                injection_data,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
            )

        # ── Normal path (no injection) ──
        num_tokens = input_batch.num_tokens_after_padding
        self.hidden_states[:num_tokens] = hidden_states

        # Get the input ids and last token indices for the speculator.
        last_token_indices = prepare_eagle_inputs(
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
        )

        # Prefill: Run the eagle speculator with eager mode.
        # TODO(woosuk): Support CUDA graph for prefill.
        last_hidden_states, hidden_states = self.run_model(
            num_tokens,
            input_batch.attn_metadata,
            input_batch.slot_mappings,
            num_tokens_across_dp=None,  # FIXME
        )
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        # NOTE(woosuk): For draft sampling, we only consider the temperature
        # and ignore the other sampling parameters such as top_k and top_p,
        # for simplicity and performance.
        # While this may slightly degrade the acceptance rate, it does not
        # affect the output distribution after rejection sampling.
        # Gather the values and copy them to the pre-allocated buffers.
        pos = self.input_buffers.positions[:num_reqs]
        torch.gather(input_batch.positions, 0, last_token_indices, out=pos)
        # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise
        # used for draft and target sampling.
        draft_tokens = gumbel_sample(
            logits,
            idx_mapping,
            self.temperature,
            self.seeds,
            pos + 1,
            apply_temperature=True,
        )
        if self.num_speculative_steps == 1:
            # Early exit.
            return draft_tokens.view(-1, 1)

        # Save the draft tokens for the first step.
        self.draft_tokens[:num_reqs, 0] = draft_tokens
        # Prepare the inputs for the decode steps.
        prepare_eagle_decode(
            draft_tokens,
            hidden_states,
            last_token_indices,
            input_batch.seq_lens,
            num_rejected,
            self.input_buffers,
            self.hidden_states,
            self.max_model_len,
            self.max_num_reqs,
        )
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        slot_mappings = self.block_tables.compute_slot_mappings(
            idx_mapping, query_start_loc, pos
        )

        cudagraph_size = self.cudagraph_manager.get_cudagraph_size(num_reqs)
        if cudagraph_size is not None:
            # Run CUDA graph.
            self.cudagraph_manager.run(cudagraph_size)
            return self.draft_tokens[:num_reqs]

        # Run eager mode.
        query_start_loc_cpu = torch.arange(
            num_reqs + 1, dtype=torch.int32, device="cpu"
        )
        block_tables = [x[:num_reqs] for x in self.block_tables.input_block_tables]

        # FIXME(woosuk): This is UNSAFE!!
        attn_metadata = build_attn_metadata(
            attn_metadata_builders=self.attn_metadata_builders,
            num_reqs=num_reqs,
            num_tokens=num_reqs,
            query_start_loc_gpu=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=1,
            seq_lens=self.input_buffers.seq_lens[:num_reqs],
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )
        self.generate_draft(
            num_reqs, attn_metadata, slot_mappings_by_layer, num_tokens_across_dp=None
        )  # FIXME
        return self.draft_tokens[:num_reqs]

    @torch.inference_mode()
    def _propose_with_injection(
        self,
        input_batch: InputBatch,
        target_hidden_states: torch.Tensor,
        injection_data: dict[str, dict],
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Run EAGLE's first forward pass with expanded full-prefix inputs
        for requests that have transferred hidden states from P/D.

        For injected requests, EAGLE receives N tokens (the full prompt)
        instead of 1 token, so its KV cache is warmed up properly.
        Non-injected requests in the same batch keep their original 1-token
        input (with zero hidden states as fallback).
        """
        num_reqs = input_batch.num_reqs
        idx_mapping = self.idx_mapping[:num_reqs]

        # ── Build expanded tensors per request ──
        all_input_ids: list[torch.Tensor] = []
        all_positions: list[torch.Tensor] = []
        all_hidden_states: list[torch.Tensor] = []
        query_lens: list[int] = []

        for batch_idx in range(num_reqs):
            req_id = input_batch.req_ids[batch_idx]
            req_state_idx = int(input_batch.idx_mapping[batch_idx].item())
            inj = injection_data.get(req_id)

            if inj is not None:
                N = inj["prompt_len"]
                hs = inj["hs"]  # [N, H]
                tok_ids = inj["token_ids"]  # [N] int32

                # EAGLE input_ids are shifted by 1: [t1, t2, ..., t_{N-1}, last_sampled]
                eagle_ids = torch.empty(N, dtype=torch.int32, device=self.device)
                eagle_ids[: N - 1] = tok_ids[1:]
                if int(num_sampled[batch_idx].item()) > 0:
                    eagle_ids[N - 1] = last_sampled[req_state_idx].to(torch.int32)
                else:
                    eagle_ids[N - 1] = tok_ids[N - 1]

                positions = torch.arange(N, device=self.device, dtype=torch.long)

                all_input_ids.append(eagle_ids)
                all_positions.append(positions)
                all_hidden_states.append(hs)
                query_lens.append(N)
            else:
                # Fallback: 1-token decode (same as normal path).
                num_sampled_i = int(num_sampled[batch_idx].item())
                if num_sampled_i > 0:
                    tok = last_sampled[req_state_idx].view(1).to(torch.int32)
                else:
                    tok = input_batch.input_ids[
                        int(input_batch.query_start_loc[batch_idx].item())
                    ].view(1)
                # Position = seq_len - 1  (last computed position).
                pos = (input_batch.seq_lens[batch_idx] - 1).view(1).long()
                # Use target hidden state at this request's position.
                start = int(input_batch.query_start_loc[batch_idx].item())
                hs_1 = target_hidden_states[start : start + 1]

                all_input_ids.append(tok)
                all_positions.append(pos)
                all_hidden_states.append(hs_1)
                query_lens.append(1)

        # ── Concatenate into batch tensors ──
        cat_input_ids = torch.cat(all_input_ids)
        cat_positions = torch.cat(all_positions)
        cat_hidden_states = torch.cat(all_hidden_states, dim=0)
        total_tokens = cat_input_ids.shape[0]

        # Copy into speculator buffers.
        self.input_buffers.input_ids[:total_tokens] = cat_input_ids
        self.input_buffers.positions[:total_tokens] = cat_positions
        self.hidden_states[:total_tokens] = cat_hidden_states

        # ── Build query_start_loc and seq_lens ──
        query_starts = [0]
        for ql in query_lens:
            query_starts.append(query_starts[-1] + ql)
        query_start_loc_gpu = torch.tensor(
            query_starts, device=self.device, dtype=torch.int32
        )
        query_start_loc_cpu = torch.tensor(query_starts, dtype=torch.int32)
        seq_lens_list = []
        for batch_idx in range(num_reqs):
            inj = injection_data.get(input_batch.req_ids[batch_idx])
            if inj is not None:
                seq_lens_list.append(inj["prompt_len"])
            else:
                seq_lens_list.append(int(input_batch.seq_lens[batch_idx].item()))
        seq_lens_gpu = torch.tensor(
            seq_lens_list, device=self.device, dtype=torch.int32
        )
        max_query_len = max(query_lens)
        max_seq_len = max(seq_lens_list)

        # ── Compute slot_mappings via speculator's block tables ──
        # First, gather the block tables for the current batch.
        self.block_tables.gather_block_tables(idx_mapping)
        slot_mappings = self.block_tables.compute_slot_mappings(
            idx_mapping,
            query_start_loc_gpu,
            cat_positions,
        )

        # ── Build attention metadata ──
        block_tables = [x[:num_reqs] for x in self.block_tables.input_block_tables]
        attn_metadata = build_attn_metadata(
            attn_metadata_builders=self.attn_metadata_builders,
            num_reqs=num_reqs,
            num_tokens=total_tokens,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=seq_lens_gpu,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )

        # ── Run EAGLE model (first pass — expanded prefill) ──
        last_hidden_states_out, hidden_states_out = self.run_model(
            total_tokens,
            attn_metadata,
            slot_mappings_by_layer,
            num_tokens_across_dp=None,
        )

        # ── Sample first draft token from last position per request ──
        last_token_indices = torch.tensor(
            [query_starts[i + 1] - 1 for i in range(num_reqs)],
            dtype=torch.int64,
            device=self.device,
        )
        sample_hidden_states = last_hidden_states_out[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        pos = self.input_buffers.positions[:num_reqs]
        torch.gather(cat_positions, 0, last_token_indices, out=pos)
        draft_tokens = gumbel_sample(
            logits,
            idx_mapping,
            self.temperature,
            self.seeds,
            pos + 1,
            apply_temperature=True,
        )
        if self.num_speculative_steps == 1:
            return draft_tokens.view(-1, 1)

        # ── Prepare decode steps (same as normal path) ──
        self.draft_tokens[:num_reqs, 0] = draft_tokens
        prepare_eagle_decode(
            draft_tokens,
            hidden_states_out,
            last_token_indices,
            seq_lens_gpu,
            torch.zeros(num_reqs, dtype=torch.int32, device=self.device),
            self.input_buffers,
            self.hidden_states,
            self.max_model_len,
            self.max_num_reqs,
        )
        query_start_loc_decode = self.input_buffers.query_start_loc[: num_reqs + 1]
        slot_mappings_decode = self.block_tables.compute_slot_mappings(
            idx_mapping, query_start_loc_decode, pos
        )

        cudagraph_size = self.cudagraph_manager.get_cudagraph_size(num_reqs)
        if cudagraph_size is not None:
            self.cudagraph_manager.run(cudagraph_size)
            return self.draft_tokens[:num_reqs]

        # Eager decode steps.
        decode_qsl_cpu = torch.arange(num_reqs + 1, dtype=torch.int32, device="cpu")
        decode_block_tables = [
            x[:num_reqs] for x in self.block_tables.input_block_tables
        ]
        decode_attn_metadata = build_attn_metadata(
            attn_metadata_builders=self.attn_metadata_builders,
            num_reqs=num_reqs,
            num_tokens=num_reqs,
            query_start_loc_gpu=query_start_loc_decode,
            query_start_loc_cpu=decode_qsl_cpu,
            max_query_len=1,
            seq_lens=self.input_buffers.seq_lens[:num_reqs],
            max_seq_len=self.max_model_len,
            block_tables=decode_block_tables,
            slot_mappings=slot_mappings_decode,
            kv_cache_config=self.kv_cache_config,
        )
        decode_slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings_decode, self.kv_cache_config
        )
        self.generate_draft(
            num_reqs,
            decode_attn_metadata,
            decode_slot_mappings_by_layer,
            num_tokens_across_dp=None,
        )
        return self.draft_tokens[:num_reqs]


@triton.jit
def _prepare_eagle_inputs_kernel(
    last_token_indices_ptr,
    eagle_input_ids_ptr,
    eagle_positions_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    # Get the true query length and next token after accounting for rejected tokens.
    num_rejected = tl.load(num_rejected_ptr + batch_idx)
    query_len -= num_rejected

    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling.
        # Get the next prefill token.
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    # Shift target_input_ids by one.
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(eagle_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)

    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + batch_idx, last_token_index)
    tl.store(eagle_input_ids_ptr + last_token_index, next_token)

    # Copy positions.
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(eagle_positions_ptr + query_start + block, target_pos, mask=mask)


def prepare_eagle_inputs(
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
) -> torch.Tensor:
    num_reqs = input_batch.num_reqs
    last_token_indices = torch.empty(
        num_reqs,
        dtype=torch.int64,
        device=num_sampled.device,
    )
    _prepare_eagle_inputs_kernel[(num_reqs,)](
        last_token_indices,
        input_buffers.input_ids,
        input_buffers.positions,
        input_batch.input_ids,
        input_batch.positions,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_batch.query_start_loc,
        BLOCK_SIZE=1024,
    )
    return last_token_indices


@triton.jit
def _prepare_eagle_docode_kernel(
    draft_tokens_ptr,
    output_hidden_states_ptr,
    output_hidden_states_stride,
    last_token_indices_ptr,
    target_seq_lens_ptr,
    num_rejected_ptr,
    input_ids_ptr,
    positions_ptr,
    input_hidden_states_ptr,
    input_hidden_states_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    hidden_size,
    max_model_len,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        # Compute query_start_loc. Pad it with the last query_start_loc
        # for CUDA graphs.
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            q = tl.where(block < num_reqs, block, num_reqs)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, q, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    # draft token -> input id.
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    tl.store(input_ids_ptr + req_idx, draft_token)

    # output hidden states -> input hidden states.
    src_idx = tl.load(last_token_indices_ptr + req_idx)
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        output_hidden_states = tl.load(
            output_hidden_states_ptr + src_idx * output_hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            input_hidden_states_ptr + req_idx * input_hidden_states_stride + block,
            output_hidden_states,
            mask=mask,
        )

    # Compute position and seq_lens.
    # NOTE(woosuk): To prevent out-of-range access, we clamp these values
    # if they reach the max model length.
    position = tl.load(positions_ptr + req_idx)
    position = tl.minimum(position + 1, max_model_len - 1)
    tl.store(positions_ptr + req_idx, position)

    target_seq_len = tl.load(target_seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    seq_len = target_seq_len - num_rejected
    seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def prepare_eagle_decode(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    last_token_indices: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_buffers: InputBuffers,
    input_hidden_states: torch.Tensor,
    max_model_len: int,
    max_num_reqs: int,
):
    num_reqs = draft_tokens.shape[0]
    hidden_size = output_hidden_states.shape[-1]
    _prepare_eagle_docode_kernel[(num_reqs + 1,)](
        draft_tokens,
        output_hidden_states,
        output_hidden_states.stride(0),
        last_token_indices,
        target_seq_lens,
        num_rejected,
        input_buffers.input_ids,
        input_buffers.positions,
        input_hidden_states,
        input_hidden_states.stride(0),
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        hidden_size,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=1024,
    )


@triton.jit
def _update_eagle_inputs_kernel(
    input_ids_ptr,
    positions_ptr,
    input_hidden_states_ptr,
    input_hidden_states_stride,
    seq_lens_ptr,
    max_model_len,
    draft_tokens_ptr,
    output_hidden_states_ptr,
    output_hidden_states_stride,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    # Draft token -> Input ID.
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Output hidden states -> Input hidden states.
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        output_hidden_states = tl.load(
            output_hidden_states_ptr + req_idx * output_hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            input_hidden_states_ptr + req_idx * input_hidden_states_stride + block,
            output_hidden_states,
            mask=mask,
        )

    # Increment position and seq_lens.
    # NOTE(woosuk): To prevent out-of-range access, we clamp these values
    # if they reach the max model length.
    position = tl.load(positions_ptr + req_idx)
    position = tl.minimum(position + 1, max_model_len - 1)
    tl.store(positions_ptr + req_idx, position)

    seq_len = tl.load(seq_lens_ptr + req_idx)
    seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def update_eagle_inputs(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    hidden_states: torch.Tensor,
    max_model_len: int,
):
    num_reqs, hidden_size = output_hidden_states.shape
    _update_eagle_inputs_kernel[(num_reqs,)](
        input_buffers.input_ids,
        input_buffers.positions,
        hidden_states,
        hidden_states.stride(0),
        input_buffers.seq_lens,
        max_model_len,
        draft_tokens,
        output_hidden_states,
        output_hidden_states.stride(0),
        hidden_size,
        BLOCK_SIZE=1024,
    )
