# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables, _compute_slot_mappings_kernel
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionStatePair,
    BatchExecutionDescriptor,
    get_uniform_token_count,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    DecodeSpeculatorCudaGraphManager,
    PrefillSpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)


class AutoRegressiveSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )

        self.supports_mm_inputs = MULTIMODAL_REGISTRY.supports_multimodal_inputs(
            self.draft_model_config
        )
        if self.supports_mm_inputs:
            self.inputs_embeds = torch.zeros(
                self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
            )

        self.prefill_cudagraph_manager: PrefillSpeculatorCudaGraphManager | None = None
        self.decode_cudagraph_manager: DecodeSpeculatorCudaGraphManager | None = None

        # PEagle (parallel drafting) support
        self.parallel_drafting = self.speculative_config.parallel_drafting
        self.parallel_drafting_token_id: int = 0
        self.parallel_drafting_hidden_state_tensor: torch.Tensor | None = None
        if self.parallel_drafting:
            self._init_parallel_drafting_params()
            N = self.num_speculative_steps
            max_extended = self.max_num_tokens + self.max_num_reqs * (N - 1)
            self.peagle_max_extended = max_extended
            self.peagle_input_ids = torch.zeros(
                max_extended, dtype=torch.int32, device=device
            )
            self.peagle_positions = torch.zeros(
                max_extended, dtype=torch.int64, device=device
            )
            self.peagle_hidden_states = torch.zeros(
                (max_extended, self.hidden_size), dtype=self.dtype, device=device
            )
            self.peagle_query_start_loc = torch.zeros(
                self.max_num_reqs + 1, dtype=torch.int32, device=device
            )
            self.peagle_seq_lens = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=device
            )
            self.peagle_is_pard_mask = torch.zeros(
                max_extended, dtype=torch.bool, device=device
            )
            self.peagle_eagle_idx = torch.zeros(
                max_extended, dtype=torch.int32, device=device
            )
            self.peagle_slot_mappings: torch.Tensor | None = None
            self.peagle_is_stale_mask = torch.zeros(
                max_extended, dtype=torch.bool, device=device
            )

    def _init_parallel_drafting_params(self) -> None:
        model_hf_config = self.draft_model_config.hf_config
        if hasattr(model_hf_config, "pard_token"):
            self.parallel_drafting_token_id = model_hf_config.pard_token
        elif hasattr(model_hf_config, "ptd_token_id"):
            self.parallel_drafting_token_id = model_hf_config.ptd_token_id
        else:
            raise ValueError(
                "For PEagle (parallel drafting), the draft model config must "
                "have `pard_token` or `ptd_token_id` in its config.json."
            )

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if self.parallel_drafting:
            self._init_parallel_drafting_hidden_state()

    def _init_parallel_drafting_hidden_state(self) -> None:
        # Extract the mask hidden state from the model for use at pard positions.
        # For Eagle3 with aux hidden states, project through combine_hidden_states.
        # If the model has no mask_hidden (e.g., plain Eagle), fall back to zeros.
        if not hasattr(self.model, "mask_hidden"):
            self.parallel_drafting_hidden_state_tensor = torch.zeros(
                self.hidden_size, dtype=self.dtype, device=self.device
            )
            return
        flat_mask = self.model.mask_hidden.view(-1)
        if hasattr(self.model, "combine_hidden_states"):
            projected = self.model.combine_hidden_states(flat_mask)
        else:
            projected = flat_mask
        self.parallel_drafting_hidden_state_tensor = projected.detach().clone()

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)
        if self.parallel_drafting:
            num_kv_groups = len(kv_cache_config.kv_cache_groups)
            self.peagle_slot_mappings = torch.zeros(
                (num_kv_groups, self.peagle_max_extended),
                dtype=torch.int64,
                device=self.device,
            )

    @property
    def advance_draft_positions(self) -> bool:
        """
        Whether to increment positions and seq_lens between draft steps.

        True for Eagle/standard MTP (each step produces new KV).
        False for Gemma4 MTP (Q-only, shares target KV, constant positions).
        """
        return True

    @property
    def model_returns_tuple(self) -> bool:
        """
        Whether the draft model's forward() returns a tuple.

        True: returns (last_hidden_states, hidden_states) — Eagle, Gemma4 MTP.
        False: returns a single tensor used for both — standard MTP (DeepSeek).
        """
        return True

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Initialize cudagraph manager for draft prefill (draft position 0).
        self.prefill_cudagraph_manager = PrefillSpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_speculative_steps + 1,
        )

        # PIECEWISE cudagraphs are not supported for draft decodes.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        # Initialize cudagraph manager for draft decodes (draft positions > 0).
        self.decode_cudagraph_manager = DecodeSpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=1,
        )

    def capture(
        self,
        attn_states: dict[BatchExecutionDescriptor, AttentionStatePair],
    ) -> None:
        logger.info("Capturing model for speculator...")
        # Reset indices to zeros to prevent stale values from prior
        # dummy runs to cause out-of-bounds indexing during capture.
        self.last_token_indices.zero_()

        # Capture the prefill routine (model forward + compute_logits +
        # sample).
        # For FULL graphs, the entire routine is recorded as one graph.
        # For PIECEWISE, only the model's compiled regions are captured
        # and the rest (compute_logits, gumbel_sample) runs eagerly.
        assert self.prefill_cudagraph_manager is not None
        if self.prefill_cudagraph_manager.use_breakable_cg:
            self.prefill_cudagraph_manager.init_breakable_cg_runner(self.model)
        self.prefill_cudagraph_manager.capture(
            self._prefill,
            attn_states,
            progress_bar_desc="Capturing prefill CUDA graphs",
        )

        if self.num_speculative_steps == 1:
            return

        # Capture the decode draft generation routine (model forward +
        # sample + update_draft_inputs) for a single
        # step.
        assert self.decode_cudagraph_manager is not None
        self.decode_cudagraph_manager.capture(
            self._generate_draft,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing decode CUDA graphs",
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
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
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_tokens = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        max_query_len = input_batch.num_scheduled_tokens.max()
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_speculative_steps, self.max_model_len
        )

        # NOTE(woosuk): To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of input_ids and
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
        self.hidden_states[:num_tokens].copy_(hidden_states)

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        # Get the input ids and last token indices for the speculator.
        prepare_prefill_inputs(
            self.last_token_indices,
            self.current_draft_step,
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_num_reqs,
        )

        # PEagle (parallel drafting): generate all N draft tokens in one pass.
        if self.parallel_drafting:
            self.parallel_eagle(
                num_reqs,
                input_batch.num_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                eagle_query_start_loc_np=input_batch.query_start_loc_np,
                num_rejected=num_rejected,
                skip_attn=dummy_run and skip_attn_for_dummy_run,
                is_profile=is_profile,
                mm_inputs=mm_inputs,
            )
            return self.draft_tokens[:num_reqs]

        # When all requests are decoding (no true prefills), each has
        # num_speculative_steps + 1 tokens, enabling FULL graph replay.
        uniform_token_count = get_uniform_token_count(
            num_reqs,
            # Use the actual number of tokens without padding added by
            # the target model during FULL cudagraph.
            input_batch.num_tokens,
            max_query_len,
        )
        prefill_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.prefill_cudagraph_manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Replay the full graph for draft prefill.
            assert self.prefill_cudagraph_manager is not None
            self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)
        else:
            # The target model's attention metadata and slot mappings
            # can directly be used for draft prefill, because of the
            # identical batch shape and KV cache layout.
            self._prefill(
                num_reqs,
                prefill_batch_desc.num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=prefill_batch_desc.cg_mode,
                mm_inputs=mm_inputs,
            )

        if self.num_speculative_steps == 1:
            # Early exit.
            return self.draft_tokens[:num_reqs, :1]

        # Prepare the inputs for the decode steps.
        prepare_decode_inputs(
            self.draft_tokens[:num_reqs, 0],
            input_batch.seq_lens,
            num_rejected,
            self.input_buffers,
            self.max_model_len,
            self.max_num_reqs,
            advance_draft_positions=self.advance_draft_positions,
        )

        # Each request produces exactly 1 token per draft generation step,
        # enabling FULL graph replay.
        decode_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager,
            num_reqs,
            num_reqs,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        # Generate the remaining num_speculative_steps - 1 draft tokens.
        self._multi_step_decode(
            num_reqs,
            dummy_run and skip_attn_for_dummy_run,
            decode_batch_desc,
            num_tokens_across_dp,
        )

        return self.draft_tokens[:num_reqs]

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            inputs_embeds = None
            if self.supports_mm_inputs:
                # Merge multimodal embeddings with input ids.
                mm_embeds, is_mm_embed = mm_inputs or (None, None)
                num_input_tokens = (
                    is_mm_embed.shape[0] if is_mm_embed is not None else num_tokens
                )
                self.inputs_embeds[:num_input_tokens] = self.model.embed_input_ids(
                    self.input_buffers.input_ids[:num_input_tokens],
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
                inputs_embeds = self.inputs_embeds[:num_tokens]

            model_inputs = dict(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                # Draft prefill with PIECEWISE cudagraph (compiled PW or breakable),
                # chosen inside run_pw_graph.
                assert self.prefill_cudagraph_manager is not None
                ret_hidden_states = self.prefill_cudagraph_manager.run_pw_graph(
                    self.model, model_inputs
                )
            else:
                # Eager (NONE): call the raw model directly.
                ret_hidden_states = self.model(**model_inputs)
        if self.model_returns_tuple:
            last_hidden_states, hidden_states = ret_hidden_states
        else:
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        last_token_indices = self.last_token_indices[:num_reqs]
        positions = self.input_buffers.positions[last_token_indices]
        idx_mapping = self.idx_mapping[:num_reqs]

        last_hidden_states, hidden_states = self._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            mm_inputs=mm_inputs,
        )
        sample_hidden_states = last_hidden_states[last_token_indices]

        self.draft_tokens[:num_reqs, 0] = self.sample_draft(
            sample_hidden_states,
            positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )
        self.hidden_states[:num_reqs] = hidden_states[last_token_indices]
        self.input_buffers.positions[:num_reqs] = positions

    def parallel_eagle(
        self,
        num_reqs: int,
        num_tokens: int,
        num_tokens_across_dp: torch.Tensor | None,
        eagle_query_start_loc_np: np.ndarray,
        num_rejected: torch.Tensor,
        skip_attn: bool = False,
        is_profile: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """PEagle single-pass forward: generates all N draft tokens at once.
        Called instead of _prefill() + _multi_step_decode() when
        parallel_drafting is enabled. Extends each request's eagle input by
        N-1 pard tokens, runs one forward pass, and samples all N draft tokens.
        """
        N = self.num_speculative_steps

        # During profile_run, skip the N-1 pard extension to keep num_tokens
        # within the model's compile range (1, max_num_batched_tokens).
        if is_profile:
            self._run_model(
                num_tokens,
                None,
                None,
                num_tokens_across_dp,
                CUDAGraphMode.NONE,
                mm_inputs,
            )
            return

        # 1. Build extended PEagle input from the existing eagle inputs
        #    (populated by prepare_prefill_inputs).
        prepare_peagle_prefill_inputs(
            last_token_indices=self.last_token_indices,
            peagle_input_ids=self.peagle_input_ids,
            peagle_positions=self.peagle_positions,
            peagle_query_start_loc=self.peagle_query_start_loc,
            peagle_seq_lens=self.peagle_seq_lens,
            peagle_is_pard_mask=self.peagle_is_pard_mask,
            peagle_is_stale_mask=self.peagle_is_stale_mask,
            peagle_eagle_idx=self.peagle_eagle_idx,
            eagle_input_ids=self.input_buffers.input_ids,
            eagle_positions=self.input_buffers.positions,
            eagle_query_start_loc=self.input_buffers.query_start_loc,
            eagle_seq_lens=self.input_buffers.seq_lens,
            num_rejected=num_rejected,
            pard_token_id=self.parallel_drafting_token_id,
            num_speculative_steps=N,
            max_num_reqs=self.max_num_reqs,
            num_reqs=num_reqs,
        )
        # 2. Extended token count: each request gains N-1 pard slots on top of
        #    its eagle query length; eagle total == num_tokens.
        num_extended = num_tokens + num_reqs * (N - 1)

        # 3. Fill hidden states: eagle positions get target hidden states;
        #    pard and stale positions get mask_hidden.
        assert self.parallel_drafting_hidden_state_tensor is not None
        mask_h = self.parallel_drafting_hidden_state_tensor
        eagle_idx = self.peagle_eagle_idx[:num_extended].long()
        is_pard = self.peagle_is_pard_mask[:num_extended]
        is_stale = self.peagle_is_stale_mask[:num_extended]

        self.peagle_hidden_states[:num_extended] = self.hidden_states[eagle_idx]
        needs_mask_h = is_pard | is_stale
        torch.where(
            needs_mask_h.unsqueeze(1),
            mask_h,
            self.peagle_hidden_states[:num_extended],
            out=self.peagle_hidden_states[:num_extended],
        )

        # 4. Compute slot mappings for the extended batch.
        if skip_attn:
            extended_slot_mappings_by_layer = None
            attn_metadata = None
        else:
            assert self.peagle_slot_mappings is not None
            num_kv_groups = self.peagle_slot_mappings.shape[0]
            _compute_slot_mappings_kernel[(num_kv_groups, num_reqs + 1)](
                num_extended,
                self.idx_mapping[:num_reqs],
                self.peagle_query_start_loc[: num_reqs + 1],
                self.peagle_positions,
                self.block_tables.block_table_ptrs,
                self.block_tables.block_table_strides,
                self.block_tables.block_sizes_tensor,
                self.peagle_slot_mappings,
                self.peagle_slot_mappings.stride(0),
                self.block_tables.cp_rank,
                CP_SIZE=self.block_tables.cp_size,
                CP_INTERLEAVE=self.block_tables.cp_interleave,
                PAD_ID=PAD_SLOT_ID,
                TRITON_BLOCK_SIZE=1024,
            )
            # Suppress KV writes for stale positions so they don't corrupt
            # the KV cache.
            stale_mask = self.peagle_is_stale_mask[:num_extended]
            self.peagle_slot_mappings[:, :num_extended].masked_fill_(
                stale_mask.unsqueeze(0), PAD_SLOT_ID
            )
            extended_slot_mappings = self.peagle_slot_mappings[:, :num_extended]
            extended_slot_mappings_by_layer = build_slot_mappings_by_layer(
                extended_slot_mappings, self.kv_cache_config
            )
            # 5. Build attention metadata: compute query_start_loc on CPU to
            #    avoid GPU sync: peagle_qsl[i] = eagle_qsl[i] + i*(N-1).
            eagle_qsl_np = eagle_query_start_loc_np[: num_reqs + 1]
            offsets = np.arange(num_reqs + 1, dtype=np.int32) * (N - 1)
            peagle_qsl_np = eagle_qsl_np + offsets
            peagle_qsl_cpu = torch.from_numpy(peagle_qsl_np)
            max_query_len_extended = (
                int((peagle_qsl_np[1:] - peagle_qsl_np[:-1]).max())
                if num_reqs > 0
                else 1
            )
            block_tables = [x[:num_reqs] for x in self.block_tables.input_block_tables]
            attn_metadata = self._build_peagle_attn_metadata(
                num_reqs=num_reqs,
                num_extended=num_extended,
                max_query_len_extended=max_query_len_extended,
                block_tables=block_tables,
                extended_slot_mappings=extended_slot_mappings,
                peagle_qsl_cpu=peagle_qsl_cpu,
            )

        # 6. Run the model forward pass with the extended batch.
        batch_descriptor = BatchDescriptor(num_tokens=num_extended)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_extended,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=extended_slot_mappings_by_layer,
            batch_descriptor=batch_descriptor,
        ):
            ret_hidden_states = self.model(
                input_ids=self.peagle_input_ids[:num_extended],
                positions=self.peagle_positions[:num_extended],
                hidden_states=self.peagle_hidden_states[:num_extended],
            )
        if self.model_returns_tuple:
            last_hidden_states, _ = ret_hidden_states
        else:
            last_hidden_states = ret_hidden_states

        # 7. Sample all N draft tokens with a single compute_logits call.
        #    all_sample_indices = [last_eagle_tok, last+1, ..., last+N-1] per req
        last_token_indices = self.last_token_indices[:num_reqs]
        first_sample_positions = self.peagle_positions[last_token_indices]
        idx_mapping = self.idx_mapping[:num_reqs]

        steps = torch.arange(N, device=self.device)
        all_sample_indices = (
            last_token_indices.unsqueeze(1) + steps.unsqueeze(0)
        ).reshape(-1)  # [B*N]
        all_logits = self.model.compute_logits(
            last_hidden_states[all_sample_indices]
        )  # [B*N, vocab]

        if self.draft_logits is None:
            self.draft_tokens[:num_reqs] = all_logits.argmax(dim=-1).view(num_reqs, N)
        else:
            all_logits_view = all_logits.view(num_reqs, N, -1)
            for step in range(N):
                self.current_draft_step.fill_(step)
                self.draft_tokens[:num_reqs, step] = gumbel_sample(
                    all_logits_view[:, step, :].contiguous(),
                    idx_mapping,
                    self.temperature,
                    self.seeds,
                    first_sample_positions + step + 1,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=self.current_draft_step,
                    use_fp64=self.use_fp64_gumbel,
                )

    def _build_peagle_attn_metadata(
        self,
        num_reqs: int,
        num_extended: int,
        max_query_len_extended: int,
        block_tables: list[torch.Tensor],
        extended_slot_mappings: torch.Tensor,
        peagle_qsl_cpu: torch.Tensor,
    ) -> dict[str, Any] | None:
        if not self.draft_attn_layer_names:
            return None
        return build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_extended,
            query_start_loc_gpu=self.peagle_query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=peagle_qsl_cpu,
            max_query_len=max_query_len_extended,
            seq_lens=self.peagle_seq_lens[:num_reqs],
            max_seq_len=self.draft_max_seq_len,
            block_tables=block_tables,
            slot_mappings=extended_slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )

    def _multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, self.num_speculative_steps):
            # Rebuild every step when positions advance, or just once
            # on the first step when positions are constant (Gemma4 MTP).
            if not skip_attn and (self.advance_draft_positions or step == 1):
                slot_mappings = self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    batch_desc.num_tokens,
                )
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config
                )
                attn_metadata = self._build_draft_attn_metadata(
                    num_reqs=num_reqs,
                    num_reqs_padded=batch_desc.num_reqs or num_reqs,
                    num_tokens_padded=batch_desc.num_tokens,
                )

            # Update the current draft step.
            self.current_draft_step.fill_(step)

            # Generate draft tokens for the current step.
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode,
                )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        # Run the draft model forward pass.
        last_hidden_states, hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        last_hidden_states = last_hidden_states[:num_reqs]

        # Sample the draft tokens.
        draft_tokens = self.sample_draft(
            last_hidden_states,
            positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )

        # Update the inputs for the next step.
        update_draft_inputs(
            draft_tokens,
            self.current_draft_step,
            hidden_states,
            self.draft_tokens,
            self.hidden_states,
            self.input_buffers,
            num_reqs,
            self.max_model_len,
            self.num_speculative_steps,
            advance_draft_positions=self.advance_draft_positions,
        )


@triton.jit
def _prepare_prefill_inputs_kernel(
    last_token_indices_ptr,
    draft_current_step_ptr,
    draft_input_ids_ptr,
    draft_positions_ptr,
    draft_query_start_loc_ptr,
    draft_seq_lens_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)

    # Get the true query length and next token after accounting for rejected tokens.
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    query_len -= num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
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
        tl.store(draft_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)

    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    tl.store(draft_input_ids_ptr + last_token_index, next_token)

    # Copy positions.
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(draft_positions_ptr + query_start + block, target_pos, mask=mask)

    # Copy query start locations.
    tl.store(draft_query_start_loc_ptr + req_idx, query_start)
    # Copy sequence lengths.
    tl.store(draft_seq_lens_ptr + req_idx, seq_len)
    if req_idx == (num_reqs - 1):
        # Reset the current draft step to 0.
        tl.store(draft_current_step_ptr, 0)
        # Pad query_start_loc for CUDA graphs.
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(draft_query_start_loc_ptr + block, query_end, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(draft_seq_lens_ptr + block, 0, mask=mask)
        # Pad last_token_indices for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


def prepare_prefill_inputs(
    # [num_reqs]
    last_token_indices: torch.Tensor,
    current_draft_step: torch.Tensor,
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
    max_num_reqs,
) -> torch.Tensor:
    num_reqs = input_batch.num_reqs
    _prepare_prefill_inputs_kernel[(num_reqs,)](
        last_token_indices,
        current_draft_step,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        input_batch.input_ids,
        input_batch.positions,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        max_num_reqs,
        BLOCK_SIZE=1024,
    )
    return last_token_indices


@triton.jit
def _prepare_decode_inputs_kernel(
    draft_tokens_ptr,
    draft_tokens_stride,
    target_seq_lens_ptr,
    num_rejected_ptr,
    input_ids_ptr,
    positions_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_model_len,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
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
    draft_token = tl.load(draft_tokens_ptr + req_idx * draft_tokens_stride)
    tl.store(input_ids_ptr + req_idx, draft_token)

    if ADVANCE_DRAFT_POSITIONS:
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


def prepare_decode_inputs(
    draft_tokens: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_buffers: InputBuffers,
    max_model_len: int,
    max_num_reqs: int,
    advance_draft_positions: bool = True,
):
    num_reqs = draft_tokens.shape[0]
    _prepare_decode_inputs_kernel[(num_reqs + 1,)](
        draft_tokens,
        draft_tokens.stride(0),
        target_seq_lens,
        num_rejected,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
    )


@triton.jit
def _update_draft_inputs_kernel(
    output_draft_tokens_ptr,
    output_draft_tokens_stride,
    next_input_hidden_states_ptr,
    next_input_hidden_states_stride,
    input_ids_ptr,
    positions_ptr,
    seq_lens_ptr,
    draft_tokens_ptr,
    current_draft_step_ptr,
    hidden_states_ptr,
    hidden_states_stride,
    hidden_size,
    max_model_len,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
):
    req_idx = tl.program_id(0)

    # Write the sampled draft token into self.draft_tokens[req_idx, step].
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    step = tl.load(current_draft_step_ptr)
    tl.store(
        output_draft_tokens_ptr + req_idx * output_draft_tokens_stride + step,
        draft_token,
    )

    if step >= num_speculative_steps - 1:
        # This is the final step. Skip updating draft forward inputs.
        return

    # Write the sampled draft token into the input ids tensor for the next
    # forward pass.
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Copy hidden states into the input hidden states tensor for the next
    # forward pass.
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        hidden_states = tl.load(
            hidden_states_ptr + req_idx * hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            next_input_hidden_states_ptr
            + req_idx * next_input_hidden_states_stride
            + block,
            hidden_states,
            mask=mask,
        )

    if ADVANCE_DRAFT_POSITIONS:
        # Increment position and seq_lens.
        # NOTE(woosuk): To prevent out-of-range access, we clamp these values
        # if they reach the max model length.
        position = tl.load(positions_ptr + req_idx)
        position = tl.minimum(position + 1, max_model_len - 1)
        tl.store(positions_ptr + req_idx, position)

        seq_len = tl.load(seq_lens_ptr + req_idx)
        seq_len = tl.minimum(seq_len + 1, max_model_len)
        tl.store(seq_lens_ptr + req_idx, seq_len)


def update_draft_inputs(
    draft_tokens: torch.Tensor,
    current_draft_step: torch.Tensor,
    hidden_states: torch.Tensor,
    output_draft_tokens: torch.Tensor,
    next_input_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    num_reqs: int,
    max_model_len: int,
    num_speculative_steps: int,
    advance_draft_positions: bool = True,
):
    _, hidden_size = hidden_states.shape
    _update_draft_inputs_kernel[(num_reqs,)](
        output_draft_tokens,
        output_draft_tokens.stride(0),
        next_input_hidden_states,
        next_input_hidden_states.stride(0),
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.seq_lens,
        draft_tokens,
        current_draft_step,
        hidden_states,
        hidden_states.stride(0),
        hidden_size,
        max_model_len,
        num_speculative_steps,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
    )


# ---------------------------------------------------------------------------
# PEagle (Parallel Eagle) helpers
# ---------------------------------------------------------------------------


@triton.jit
def _prepare_peagle_prefill_inputs_kernel(
    # Outputs: extended PEagle buffer
    out_input_ids_ptr,  # [max_extended]
    out_positions_ptr,  # [max_extended]
    out_query_start_loc_ptr,  # [max_num_reqs + 1]
    out_seq_lens_ptr,  # [max_num_reqs]
    out_is_pard_mask_ptr,  # [max_extended] bool
    out_is_stale_mask_ptr,  # [max_extended] bool
    out_eagle_idx_ptr,  # [max_extended] int32: extended pos → eagle pos
    last_token_indices_ptr,  # [max_num_reqs] (mutated: set to last valid eagle pos)
    # Inputs: from prepare_prefill_inputs
    eagle_input_ids_ptr,  # [max_num_tokens]
    eagle_positions_ptr,  # [max_num_tokens]
    eagle_query_start_loc_ptr,  # [max_num_reqs + 1]
    eagle_seq_lens_ptr,  # [max_num_reqs]
    num_rejected_ptr,  # [max_num_reqs]
    pard_token_id,
    num_speculative_steps,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    N = num_speculative_steps

    eagle_q_start = tl.load(eagle_query_start_loc_ptr + req_idx)
    eagle_q_end = tl.load(eagle_query_start_loc_ptr + req_idx + 1)
    eagle_q_len = eagle_q_end - eagle_q_start

    # Use adjusted query length to exclude stale (rejected) tokens from the
    # previous speculative round.
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    adjusted_q_len = eagle_q_len - num_rejected

    # Extended buffer layout per request:
    #   [stale(num_rejected) | eagle(adjusted_q_len) | pard(N-1)]
    #
    # Placing stale tokens FIRST ensures valid eagle tokens have higher
    # q_idx values under FlashAttention's relative causal mask, so they
    # can attend to all required KV positions.
    out_start = eagle_q_start + req_idx * (N - 1)
    valid_eagle_start = out_start + num_rejected

    # Copy valid eagle input IDs.
    for i in range(0, adjusted_q_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < adjusted_q_len
        ids = tl.load(eagle_input_ids_ptr + eagle_q_start + block, mask=mask, other=0)
        tl.store(out_input_ids_ptr + valid_eagle_start + block, ids, mask=mask)

    # Fill pard token IDs after valid eagle tokens.
    for i in range(0, N - 1, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < (N - 1)
        tl.store(
            out_input_ids_ptr + valid_eagle_start + adjusted_q_len + block,
            pard_token_id,
            mask=mask,
        )

    # Copy valid eagle positions.
    for i in range(0, adjusted_q_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < adjusted_q_len
        pos = tl.load(eagle_positions_ptr + eagle_q_start + block, mask=mask, other=0)
        tl.store(out_positions_ptr + valid_eagle_start + block, pos, mask=mask)

    # Fill pard positions: last valid eagle pos + 1, +2, ...
    last_pos = tl.load(eagle_positions_ptr + eagle_q_start + adjusted_q_len - 1)
    for i in range(0, N - 1, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < (N - 1)
        tl.store(
            out_positions_ptr + valid_eagle_start + adjusted_q_len + block,
            last_pos + 1 + block,
            mask=mask,
        )

    # is_stale_mask: True for stale slots [out_start, valid_eagle_start).
    # is_pard_mask:  False for eagle tokens, True for pard tokens.
    for i in range(0, num_rejected, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < num_rejected
        tl.store(out_is_stale_mask_ptr + out_start + block, 1, mask=mask)
    for i in range(0, adjusted_q_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < adjusted_q_len
        tl.store(out_is_pard_mask_ptr + valid_eagle_start + block, 0, mask=mask)
        tl.store(out_is_stale_mask_ptr + valid_eagle_start + block, 0, mask=mask)
    for i in range(0, N - 1, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < (N - 1)
        tl.store(
            out_is_pard_mask_ptr + valid_eagle_start + adjusted_q_len + block,
            1,
            mask=mask,
        )
        tl.store(
            out_is_stale_mask_ptr + valid_eagle_start + adjusted_q_len + block,
            0,
            mask=mask,
        )

    # Eagle index mapping: valid eagle positions → original eagle buffer index.
    # Pard and stale positions map to 0 (placeholder; hidden state overridden).
    for i in range(0, adjusted_q_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < adjusted_q_len
        tl.store(
            out_eagle_idx_ptr + valid_eagle_start + block,
            eagle_q_start + block,
            mask=mask,
        )
    for i in range(0, N - 1, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < (N - 1)
        tl.store(
            out_eagle_idx_ptr + valid_eagle_start + adjusted_q_len + block, 0, mask=mask
        )

    # Update last_token_indices to the last valid eagle token (sampling start).
    tl.store(last_token_indices_ptr + req_idx, valid_eagle_start + adjusted_q_len - 1)

    # Extended query metadata.
    tl.store(out_query_start_loc_ptr + req_idx, out_start)
    eagle_seq_len = tl.load(eagle_seq_lens_ptr + req_idx)
    tl.store(out_seq_lens_ptr + req_idx, eagle_seq_len - num_rejected + N - 1)

    if req_idx == num_reqs - 1:
        out_end = out_start + eagle_q_len + N - 1
        tl.store(out_query_start_loc_ptr + num_reqs, out_end)
        # Pad remaining entries for CUDA graphs.
        for i in range(num_reqs + 1, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(out_query_start_loc_ptr + block, out_end, mask=mask)
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(out_seq_lens_ptr + block, 0, mask=mask)


def prepare_peagle_prefill_inputs(
    last_token_indices: torch.Tensor,
    peagle_input_ids: torch.Tensor,
    peagle_positions: torch.Tensor,
    peagle_query_start_loc: torch.Tensor,
    peagle_seq_lens: torch.Tensor,
    peagle_is_pard_mask: torch.Tensor,
    peagle_is_stale_mask: torch.Tensor,
    peagle_eagle_idx: torch.Tensor,
    eagle_input_ids: torch.Tensor,
    eagle_positions: torch.Tensor,
    eagle_query_start_loc: torch.Tensor,
    eagle_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    pard_token_id: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    num_reqs: int,
) -> None:
    _prepare_peagle_prefill_inputs_kernel[(num_reqs,)](
        peagle_input_ids,
        peagle_positions,
        peagle_query_start_loc,
        peagle_seq_lens,
        peagle_is_pard_mask,
        peagle_is_stale_mask,
        peagle_eagle_idx,
        last_token_indices,
        eagle_input_ids,
        eagle_positions,
        eagle_query_start_loc,
        eagle_seq_lens,
        num_rejected,
        pard_token_id,
        num_speculative_steps,
        max_num_reqs,
        BLOCK_SIZE=32,
    )
