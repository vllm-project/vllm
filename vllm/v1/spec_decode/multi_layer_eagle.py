# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer

logger = init_logger(__name__)

PADDING_SLOT_ID = -1
BLOCK_HIDDEN = 128
BLOCK_TOKENS = 128


class DraftInputStates:
    def __init__(
        self,
        len: torch.Tensor,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        # Keep this as a tensor to avoid device syncs from `.item()`.
        self.len = len
        self.token_ids = token_ids
        self.hidden_states = hidden_states
        self.positions = positions
        self.slot_mapping = slot_mapping


class MultiLayerEagleProposer(EagleProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner)

        self.layer_num: int = getattr(
            self.speculative_config.draft_model_config.hf_text_config, "n_predict", 0
        )
        self.num_speculative_tokens: int = (
            self.speculative_config.num_speculative_tokens
        )
        if self.num_speculative_tokens != self.layer_num:
            logger.warning_once(
                "For multi_layer_eagle, num_speculative_tokens "
                "does not match layer_num, adjusting to layer_num"
            )
            self.num_speculative_tokens = self.layer_num
        self.running_req_ids: list[str] | None = None
        self.draft_input_states_pool: dict[str, DraftInputStates] = {}

    def set_running_req_ids(self, req_ids: list[str]):
        self.running_req_ids = req_ids

    def _get_draft_input_states(self, req_id: str, len: int) -> DraftInputStates:
        draft_input_states = self.draft_input_states_pool.get(req_id, None)
        assert draft_input_states is not None
        assert draft_input_states.len >= len
        return draft_input_states

    def clean_req_cache(self, req_id: str):
        self.draft_input_states_pool.pop(req_id, None)

    def adjust_input(
        self,
        batch_size: int,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        MAX_SHIFT = self.layer_num
        assert MAX_SHIFT > 0

        device = target_token_ids.device
        hidden_size = int(target_hidden_states.shape[1])

        prev_token_ids = target_token_ids.clone()
        prev_positions = target_positions.clone()
        prev_hidden_states = target_hidden_states.clone()
        slot_mapping = common_attn_metadata.slot_mapping

        start_token_indices = common_attn_metadata.query_start_loc[:-1]
        end_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        pos_for_shift = (
            target_positions[0] if target_positions.dim() == 2 else target_positions
        )
        start_token_pos = pos_for_shift[start_token_indices]

        shift = torch.minimum(
            end_token_indices - last_token_indices,
            start_token_pos.to(end_token_indices.dtype),
        )
        shift = torch.clamp(shift, min=0).to(torch.int32)

        # Metadata updates (matches the original reference implementation).
        last_token_indices.add_(shift.to(last_token_indices.dtype))
        common_attn_metadata.seq_lens.sub_(
            shift.to(common_attn_metadata.seq_lens.dtype)
        )

        # NOTE: ignore cpu data to avoid device sync
        # common_attn_metadata.seq_lens_cpu.copy_(common_attn_metadata.seq_lens,
        #                                         non_blocking=True)
        # query_lens = common_attn_metadata.query_start_loc[
        #     1:] - common_attn_metadata.query_start_loc[:-1]
        # num_computed_tokens = common_attn_metadata.seq_lens - query_lens.to(
        #     common_attn_metadata.seq_lens.dtype)
        # common_attn_metadata.num_computed_tokens_cpu.copy_(
        #     num_computed_tokens.to(
        #         common_attn_metadata.num_computed_tokens_cpu.dtype),
        #     non_blocking=True,
        # )
        # common_attn_metadata.max_seq_len =
        #       int(common_attn_metadata.seq_lens_cpu.max().item())

        cached_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        cached_prev_token_ids = torch.zeros(
            batch_size, MAX_SHIFT, dtype=prev_token_ids.dtype, device=device
        )
        if prev_positions.dim() == 1:
            cached_prev_positions: Any = torch.zeros(
                batch_size, MAX_SHIFT, dtype=prev_positions.dtype, device=device
            )
        else:
            cached_prev_positions = [
                torch.zeros(
                    batch_size, MAX_SHIFT, dtype=prev_positions.dtype, device=device
                )
                for _ in range(3)
            ]
        cached_prev_hidden_states = torch.zeros(
            (batch_size, MAX_SHIFT, hidden_size),
            dtype=prev_hidden_states.dtype,
            device=device,
        )
        cached_slot_mappings = torch.zeros(
            batch_size, MAX_SHIFT, dtype=slot_mapping.dtype, device=device
        )

        assert self.running_req_ids is not None
        for i in range(batch_size):
            req_id = self.running_req_ids[i]
            draft_input_states = self.draft_input_states_pool.get(req_id, None)
            if draft_input_states is None:
                continue

            cached_lens[i] = draft_input_states.len
            cached_prev_token_ids[i].copy_(draft_input_states.token_ids)

            if prev_positions.dim() == 1:
                assert (
                    cached_prev_positions[i].shape == draft_input_states.positions.shape
                )
                cached_prev_positions[i].copy_(draft_input_states.positions)
            else:
                assert prev_positions.dim() == 2
                assert (
                    cached_prev_positions[:, i].shape
                    == draft_input_states.positions.shape
                )
                cached_prev_positions[:, i].copy_(draft_input_states.positions)

            cached_prev_hidden_states[i].copy_(draft_input_states.hidden_states)
            cached_slot_mappings[i].copy_(draft_input_states.slot_mapping)

        shift = torch.minimum(shift, cached_lens)

        # [batch_size]
        len_buffer = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # [batch_size, MAX_SHIFT]
        token_ids_buffer = torch.zeros(
            (batch_size, MAX_SHIFT), dtype=prev_token_ids.dtype, device=device
        )
        if prev_positions.dim() == 1:
            positions_buffer: Any = torch.zeros(
                (batch_size, MAX_SHIFT), dtype=prev_positions.dtype, device=device
            )
        else:
            positions_buffer = [
                torch.zeros(
                    (batch_size, MAX_SHIFT), dtype=prev_positions.dtype, device=device
                )
                for _ in range(3)
            ]
        # [batch_size, MAX_SHIFT, hidden_size]
        hidden_states_buffer = torch.zeros(
            (batch_size, MAX_SHIFT, hidden_size),
            dtype=prev_hidden_states.dtype,
            device=device,
        )
        # [batch_size, MAX_SHIFT]
        slot_mapping_buffer = torch.zeros(
            (batch_size, MAX_SHIFT), dtype=slot_mapping.dtype, device=device
        )

        _multi_layer_eagle_shift_and_cache(
            batch_size=batch_size,
            max_shift=MAX_SHIFT,
            src_token_ids=target_token_ids,
            dst_token_ids=prev_token_ids,
            src_positions=target_positions,
            dst_positions=prev_positions,
            src_hidden_states=target_hidden_states,
            dst_hidden_states=prev_hidden_states,
            src_slot_mapping=slot_mapping,
            dst_slot_mapping=slot_mapping,
            start_token_indices=start_token_indices,
            end_token_indices=end_token_indices,
            last_token_indices=last_token_indices,
            shift=shift,
            cached_lens=cached_lens,
            cached_prev_token_ids=cached_prev_token_ids,
            cached_prev_positions=cached_prev_positions,
            cached_prev_hidden_states=cached_prev_hidden_states,
            cached_slot_mappings=cached_slot_mappings,
            out_cached_lens=len_buffer,
            out_cached_token_ids=token_ids_buffer,
            out_cached_positions=positions_buffer,
            out_cached_hidden_states=hidden_states_buffer,
            out_cached_slot_mappings=slot_mapping_buffer,
            common_attn_metadata=common_attn_metadata,
        )
        for i in range(batch_size):
            req_id = self.running_req_ids[i]
            if prev_positions.dim() == 1:
                cached_positions = positions_buffer[i].clone()
            else:
                cached_positions = torch.stack(
                    [buf[i] for buf in positions_buffer], dim=0
                ).clone()
            self.draft_input_states_pool[req_id] = DraftInputStates(
                len=len_buffer[i].clone(),
                token_ids=token_ids_buffer[i].clone(),
                hidden_states=hidden_states_buffer[i].clone(),
                positions=cached_positions,
                slot_mapping=slot_mapping_buffer[i].clone(),
            )

        return prev_token_ids, prev_positions, prev_hidden_states, common_attn_metadata

    def initial_inputs_for_forward(
        self,
        num_tokens: int,
        prev_token_ids: torch.Tensor,
        prev_positions: torch.Tensor,
        prev_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
        spec_step_idx: int = 0,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ):
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[: num_tokens - 1] = prev_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids
        self._set_positions(num_tokens, prev_positions)
        self.hidden_states[:num_tokens] = prev_hidden_states[:num_tokens]
        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

    def draft_model_forward(
        self,
        num_tokens: int,
        per_layer_attn_metadata: dict[str, Any],
        last_token_indices: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        common_attn_metadata: CommonAttentionMetadata,
        spec_step_idx: int = 0,
    ):
        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=num_tokens, num_tokens_padded=num_tokens
        )

        cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens_dp_padded
        )
        num_input_tokens = batch_desc.num_tokens

        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        if self.supports_mm_inputs:
            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_input_tokens),
            "hidden_states": self.hidden_states[:num_input_tokens],
            "inputs_embeds": inputs_embeds,
            "spec_step_idx": spec_step_idx,
        }

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                num_input_tokens, common_attn_metadata.slot_mapping
            ),
        ):
            last_hidden_states = self.model(**model_kwargs)

        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(
            sample_hidden_states, spec_step_idx=spec_step_idx
        )

        draft_token_ids = logits.argmax(dim=-1)

        return draft_token_ids, last_hidden_states

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        assert self.method == "mtp"
        assert self.runner is not None

        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        prev_token_ids, prev_positions, prev_hidden_states, common_attn_metadata = (
            self.adjust_input(
                batch_size=batch_size,
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                last_token_indices=last_token_indices,
                common_attn_metadata=common_attn_metadata,
            )
        )

        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder

        assert isinstance(attn_metadata_builder, AttentionMetadataBuilder)

        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )

        # FIXME: support hybrid kv for draft model (remove separate indexer)
        if self.draft_indexer_metadata_builder:
            draft_indexer_metadata = (
                self.draft_indexer_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=0,
                )
            )
        else:
            draft_indexer_metadata = None

        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        for layer_name in self.indexer_layer_names:
            assert draft_indexer_metadata is not None
            per_layer_attn_metadata[layer_name] = draft_indexer_metadata

        if isinstance(attn_metadata, TreeAttentionMetadata):
            raise NotImplementedError(
                "Tree attention is not supported for multi layer eagle."
            )

        if self.allowed_attn_types is not None and not isinstance(
            attn_metadata, self.allowed_attn_types
        ):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding for multi layer eagle: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}"
            )

        # Generate the remaining draft tokens.
        draft_token_ids_list: list[torch.Tensor] = []

        for token_index in range(self.num_speculative_tokens):
            if token_index != 0:
                prev_token_ids = self.input_ids[:num_tokens].clone()
                next_token_ids = draft_token_ids_list[-1].int()

            self.initial_inputs_for_forward(
                num_tokens=num_tokens,
                prev_token_ids=prev_token_ids,
                prev_positions=prev_positions,
                prev_hidden_states=prev_hidden_states,
                next_token_ids=next_token_ids,
                last_token_indices=last_token_indices,
                spec_step_idx=token_index,
                mm_embed_inputs=mm_embed_inputs,
            )

            draft_token_ids, prev_hidden_states = self.draft_model_forward(
                num_tokens=num_tokens,
                per_layer_attn_metadata=per_layer_attn_metadata,
                last_token_indices=last_token_indices,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                spec_step_idx=token_index,
            )

            # Early exit if there is only one draft token to be generated.
            if self.num_speculative_tokens == 1:
                return draft_token_ids.view(-1, 1)

            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)

        return draft_token_ids

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        raise Exception(
            "speculative_config.disable_padded_drafter_batch"
            " is not supported now for MultiLayerEagleProposer."
        )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=num_tokens, num_tokens_padded=num_tokens
        )
        if use_cudagraphs:
            cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                num_tokens_dp_padded
            )
            num_input_tokens = batch_desc.num_tokens
        else:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
            num_input_tokens = num_tokens_dp_padded
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        # Make sure to use EAGLE's own buffer during cudagraph capture.
        if (
            self.attn_layer_names
            and slot_mappings is not None
            and self.attn_layer_names[0] in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        self.set_running_req_ids([f"dummy_req_{i}" for i in range(1)])

        adjust_input_kwargs = {
            "batch_size": 1,
            "target_token_ids": self.input_ids[:num_input_tokens],
            "target_positions": self._get_positions(num_input_tokens),
            "target_hidden_states": self.hidden_states[:num_input_tokens],
            "last_token_indices": torch.tensor(
                [num_input_tokens - 1], dtype=torch.int32, device=self.device
            ),
            "common_attn_metadata": CommonAttentionMetadata(
                query_start_loc=torch.tensor(
                    [0, num_input_tokens], dtype=torch.int32, device=self.device
                ),
                query_start_loc_cpu=torch.tensor(
                    [0, num_input_tokens], dtype=torch.int32, device="cpu"
                ),
                seq_lens=torch.tensor(
                    [num_input_tokens], dtype=torch.int32, device=self.device
                ),
                num_reqs=1,
                num_actual_tokens=num_input_tokens,
                max_query_len=num_input_tokens,
                max_seq_len=self.max_model_len,
                block_table_tensor=torch.tensor(
                    [], dtype=torch.int32, device=self.device
                ),
                slot_mapping=self.arange[:num_input_tokens],
                logits_indices_padded=None,
                num_logits_indices=None,
                causal=True,
                encoder_seq_lens=None,
            ),
        }
        # NOTE ensure the jit kernel in _adjust_input can be compiled
        self.adjust_input(**adjust_input_kwargs)
        self.clean_req_cache("dummy_req_0")

        for fwd_idx in range(self.layer_num):
            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None

                model_kwargs = {
                    "input_ids": input_ids,
                    "positions": self._get_positions(num_input_tokens),
                    "hidden_states": self.hidden_states[:num_input_tokens],
                    "inputs_embeds": inputs_embeds,
                    "spec_step_idx": fwd_idx,
                }

                self.model(**model_kwargs)


def _multi_layer_eagle_shift_and_cache(
    *,
    batch_size: int,
    max_shift: int,
    src_token_ids: torch.Tensor,
    dst_token_ids: torch.Tensor,
    src_positions: torch.Tensor,
    dst_positions: torch.Tensor,
    src_hidden_states: torch.Tensor,
    dst_hidden_states: torch.Tensor,
    src_slot_mapping: torch.Tensor,
    dst_slot_mapping: torch.Tensor,
    start_token_indices: torch.Tensor,
    end_token_indices: torch.Tensor,
    last_token_indices: torch.Tensor,
    shift: torch.Tensor,
    cached_lens: torch.Tensor,
    cached_prev_token_ids: torch.Tensor,
    cached_prev_positions: Any,
    cached_prev_hidden_states: torch.Tensor,
    cached_slot_mappings: torch.Tensor,
    out_cached_lens: torch.Tensor,
    out_cached_token_ids: torch.Tensor,
    out_cached_positions: Any,
    out_cached_hidden_states: torch.Tensor,
    out_cached_slot_mappings: torch.Tensor,
    common_attn_metadata: CommonAttentionMetadata,
):
    if batch_size == 0:
        return

    assert max_shift > 0

    start_token_indices_i32 = start_token_indices.to(torch.int32)
    end_token_indices_i32 = end_token_indices.to(torch.int32)

    # If src/dst are the same tensor, shifting is unsafe without a separate src.
    if src_slot_mapping.data_ptr() == dst_slot_mapping.data_ptr():
        src_slot_mapping = src_slot_mapping.clone()

    # Cache extraction for the next call.
    cache_start = torch.maximum(
        start_token_indices_i32,
        (last_token_indices.to(torch.int32) + 1 - max_shift),
    )
    cache_len = torch.clamp(
        last_token_indices.to(torch.int32) - cache_start + 1,
        min=0,
        max=max_shift,
    ).to(torch.int32)
    out_cached_lens.copy_(cache_len)

    padded_shift = triton.next_power_of_2(max_shift)

    hidden_size = int(dst_hidden_states.shape[1])
    # Hidden blocking avoids extremely large Triton tiles (and huge cubins)
    # when hidden_size is large.
    num_hidden_blocks = max(1, (hidden_size + BLOCK_HIDDEN - 1) // BLOCK_HIDDEN)
    # Avoid device sync: query length == (end - start + 1) == diff of
    # query_start_loc (CPU copy).
    max_window_len = int(
        (
            common_attn_metadata.query_start_loc_cpu[1:]
            - common_attn_metadata.query_start_loc_cpu[:-1]
        )
        .max()
        .item()
    )
    num_blocks = max(1, (max_window_len + BLOCK_TOKENS - 1) // BLOCK_TOKENS)

    def _shift_and_gather_cache_1d_kernel(
        src: torch.Tensor,
        dst: torch.Tensor,
        cached_prev: torch.Tensor,
        out_cached: torch.Tensor,
    ):
        _shift_1d_kernel[(batch_size, num_blocks)](
            src,
            dst,
            cached_prev,
            start_token_indices_i32,
            end_token_indices_i32,
            shift,
            cached_lens,
            MAX_SHIFT=max_shift,
            BLOCK_TOKENS=BLOCK_TOKENS,
        )

        _gather_cache_1d_kernel[(batch_size,)](
            dst,
            out_cached,
            cache_start,
            cache_len,
            MAX_SHIFT=max_shift,
            PADDED_SHIFT=padded_shift,
        )

    _shift_and_gather_cache_1d_kernel(
        src_token_ids,
        dst_token_ids,
        cached_prev_token_ids,
        out_cached_token_ids,
    )

    _shift_and_gather_cache_1d_kernel(
        src_slot_mapping,
        dst_slot_mapping,
        cached_slot_mappings,
        out_cached_slot_mappings,
    )

    if dst_positions.dim() == 1:
        assert isinstance(cached_prev_positions, torch.Tensor)
        assert isinstance(out_cached_positions, torch.Tensor)
        _shift_and_gather_cache_1d_kernel(
            src_positions,
            dst_positions,
            cached_prev_positions,
            out_cached_positions,
        )
    else:
        assert isinstance(cached_prev_positions, list)
        assert isinstance(out_cached_positions, list)
        for row in range(3):
            _shift_and_gather_cache_1d_kernel(
                src_positions[row],
                dst_positions[row],
                cached_prev_positions[row],
                out_cached_positions[row],
            )

    _shift_hidden_kernel[(batch_size, num_blocks, num_hidden_blocks)](
        src_hidden_states,
        dst_hidden_states,
        cached_prev_hidden_states,
        start_token_indices_i32,
        end_token_indices_i32,
        shift,
        cached_lens,
        src_hidden_states.stride(0),
        src_hidden_states.stride(1),
        dst_hidden_states.stride(0),
        dst_hidden_states.stride(1),
        cached_prev_hidden_states.stride(0),
        cached_prev_hidden_states.stride(1),
        cached_prev_hidden_states.stride(2),
        MAX_SHIFT=max_shift,
        HIDDEN_SIZE=hidden_size,
        BLOCK_TOKENS=BLOCK_TOKENS,
        BLOCK_HIDDEN=BLOCK_HIDDEN,
        num_warps=4,
    )

    _gather_cache_hidden_kernel[(batch_size, num_hidden_blocks)](
        dst_hidden_states,
        out_cached_hidden_states,
        cache_start,
        cache_len,
        dst_hidden_states.stride(0),
        dst_hidden_states.stride(1),
        out_cached_hidden_states.stride(0),
        out_cached_hidden_states.stride(1),
        out_cached_hidden_states.stride(2),
        MAX_SHIFT=max_shift,
        HIDDEN_SIZE=hidden_size,
        PADDED_SHIFT=padded_shift,
        BLOCK_HIDDEN=BLOCK_HIDDEN,
        num_warps=4,
    )
    return


@triton.jit
def _shift_1d_kernel(
    src_ptr,
    dst_ptr,
    cached_ptr,
    start_ptr,
    end_ptr,
    shift_ptr,
    cached_len_ptr,
    MAX_SHIFT: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_blk = tl.program_id(1)

    start = tl.load(start_ptr + pid_seq).to(tl.int32)
    end = tl.load(end_ptr + pid_seq).to(tl.int32)
    shift = tl.load(shift_ptr + pid_seq).to(tl.int32)
    cached_len = tl.load(cached_len_ptr + pid_seq).to(tl.int32)

    window_len = end - start + 1
    base_cached = cached_ptr + pid_seq * MAX_SHIFT
    k = tl.arange(0, BLOCK_TOKENS)

    base = pid_blk * BLOCK_TOKENS
    offs = base + k
    mask = offs < window_len

    dst_idx = start + offs
    head = offs < shift

    cached_idx = cached_len - shift + offs
    cached_in_range = (
        head & (cached_idx >= 0) & (cached_idx < cached_len) & (cached_idx < MAX_SHIFT)
    )
    cached_idx_safe = tl.where(cached_in_range, cached_idx, 0)
    val_head = tl.load(
        base_cached + cached_idx_safe, mask=mask & cached_in_range, other=0
    )

    src_idx = start + offs - shift
    src_idx = tl.where(head, 0, src_idx)
    val_body = tl.load(src_ptr + src_idx, mask=mask & ~head, other=0)

    val = tl.where(head, val_head, val_body)
    tl.store(dst_ptr + dst_idx, val, mask=mask)


@triton.jit
def _shift_hidden_kernel(
    src_ptr,
    dst_ptr,
    cached_ptr,
    start_ptr,
    end_ptr,
    shift_ptr,
    cached_len_ptr,
    src_s0,
    src_s1,
    dst_s0,
    dst_s1,
    cached_s0,
    cached_s1,
    cached_s2,
    MAX_SHIFT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_blk = tl.program_id(1)
    pid_hid = tl.program_id(2)

    start = tl.load(start_ptr + pid_seq).to(tl.int32)
    end = tl.load(end_ptr + pid_seq).to(tl.int32)
    shift = tl.load(shift_ptr + pid_seq).to(tl.int32)
    cached_len = tl.load(cached_len_ptr + pid_seq).to(tl.int32)

    window_len = end - start + 1
    cached_base = cached_ptr + pid_seq * cached_s0
    k = tl.arange(0, BLOCK_TOKENS)

    base = pid_blk * BLOCK_TOKENS
    m = base + k
    m_mask = m < window_len
    head = m < shift

    dst_tok = start + m
    src_tok = start + m - shift
    cached_tok = cached_len - shift + m

    cached_in_range = (
        head & (cached_tok >= 0) & (cached_tok < cached_len) & (cached_tok < MAX_SHIFT)
    )
    cached_tok_safe = tl.where(cached_in_range, cached_tok, 0)
    src_tok = tl.where(head, 0, src_tok)

    n = pid_hid * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    n_mask = n < HIDDEN_SIZE
    mask = m_mask[:, None] & n_mask[None, :]

    src_ptrs = src_ptr + src_tok[:, None] * src_s0 + n[None, :] * src_s1
    dst_ptrs = dst_ptr + dst_tok[:, None] * dst_s0 + n[None, :] * dst_s1
    cached_ptrs = (
        cached_base + cached_tok_safe[:, None] * cached_s1 + n[None, :] * cached_s2
    )

    val_head = tl.load(cached_ptrs, mask=mask & cached_in_range[:, None], other=0)
    val_body = tl.load(src_ptrs, mask=mask & ~head[:, None], other=0)
    val = tl.where(head[:, None], val_head, val_body)
    tl.store(dst_ptrs, val, mask=mask)


@triton.jit
def _gather_cache_1d_kernel(
    src_ptr,
    out_ptr,
    cache_start_ptr,
    cache_len_ptr,
    MAX_SHIFT: tl.constexpr,
    PADDED_SHIFT: tl.constexpr,
):
    pid_seq = tl.program_id(0)

    cache_start = tl.load(cache_start_ptr + pid_seq).to(tl.int32)
    cache_len = tl.load(cache_len_ptr + pid_seq).to(tl.int32)

    out_base = out_ptr + pid_seq * MAX_SHIFT
    k = tl.arange(0, PADDED_SHIFT)
    k_mask = k < MAX_SHIFT
    src_idx = cache_start + k
    val = tl.load(src_ptr + src_idx, mask=k_mask & (k < cache_len), other=0)
    tl.store(out_base + k, val, mask=k_mask)


@triton.jit
def _gather_cache_hidden_kernel(
    src_ptr,
    out_ptr,
    cache_start_ptr,
    cache_len_ptr,
    src_s0,
    src_s1,
    out_s0,
    out_s1,
    out_s2,
    MAX_SHIFT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    PADDED_SHIFT: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_hid = tl.program_id(1)

    cache_start = tl.load(cache_start_ptr + pid_seq).to(tl.int32)
    cache_len = tl.load(cache_len_ptr + pid_seq).to(tl.int32)
    m = tl.arange(0, PADDED_SHIFT)
    m_mask = (m < MAX_SHIFT) & (m < cache_len)
    tok = cache_start + m

    n = pid_hid * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    n_mask = n < HIDDEN_SIZE
    mask = m_mask[:, None] & n_mask[None, :]

    src_ptrs = src_ptr + tok[:, None] * src_s0 + n[None, :] * src_s1
    out_ptrs = out_ptr + pid_seq * out_s0 + m[:, None] * out_s1 + n[None, :] * out_s2
    val = tl.load(src_ptrs, mask=mask, other=0)
    tl.store(out_ptrs, val, mask=mask)
