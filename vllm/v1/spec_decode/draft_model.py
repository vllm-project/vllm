# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.speculative import SpeculativeConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.model_loader import get_model
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    extend_all_queries_by_1,
)
from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID, SpecDecodeBaseProposer

logger = init_logger(__name__)


class DraftModelProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self._raise_if_multimodal()
        self._raise_if_mrope()
        self._raise_if_padded_drafter_batch_disabled()
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

    def _block_size(self) -> int:
        builder = self._get_attention_metadata_builder()
        return builder.kv_cache_spec.block_size

    def _raise_if_multimodal(self):
        if self.supports_mm_inputs:
            raise NotImplementedError(
                "Speculative Decoding with draft models "
                "does not support multimodal models yet"
            )

    def _raise_if_mrope(self):
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError(
                "Speculative Decoding with draft models does not support M-RoPE yet"
            )

    def _raise_if_padded_drafter_batch_disabled(self):
        if self.vllm_config.speculative_config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "Speculative Decoding with draft models only supports "
                "padded drafter batch. Please don't pass --disable-padded-drafter-batch"
                " in the speculative_config."
            )

    def _raise_if_vocab_size_mismatch(self):
        self.vllm_config.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _raise_if_draft_tp_mismatch(self):
        # Note(Tomas Ruiz) If we run the target model with TP > 1 and
        # the draft model with TP = 1, then the different TP ranks collide.
        # Specifically when all ranks compile the draft model on rank 0
        # (because TP=1), then the torch compile cache is overwritten and corrupted.
        # We need a mechanism like this: https://github.com/vllm-project/vllm/pull/5414
        # To prevent this error, we assert that both TP sizes must be the same.
        spec_cfg: SpeculativeConfig = self.vllm_config.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' "
                f"must be the same. Got {draft_tp} and {tgt_tp}. "
                "Please pass 'draft_tensor_parallel_size' in the speculative_config."
            )

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        batch_size = cad.batch_size()
        grid = (batch_size,)
        start_locs = cad.query_start_loc[:-1]
        end_locs = cad.query_start_loc[1:] - 1
        if num_rejected_tokens_gpu is not None:
            end_locs -= num_rejected_tokens_gpu

        num_tokens = target_token_ids.shape[0] + batch_size
        is_rejected_tok = torch.empty(
            (num_tokens,), device=self.input_ids.device, dtype=torch.bool
        )
        merge_toks_kernel[grid](
            target_toks_ptr=target_token_ids,
            next_toks_ptr=next_token_ids,
            query_start_locs_ptr=start_locs,
            query_end_locs_ptr=end_locs,
            out_ptr_merged_toks=self.input_ids,
            out_ptr_is_rejected_tok=is_rejected_tok,
            target_toks_size=target_token_ids.shape[0],
            # passing a negative rejected_tok_fill value will raise an error
            # when the value is used to index into embeddings.
            # Therefore, we pass a valid integer, e.g. 0.
            rejected_tok_fill=0,
        )
        merge_toks_kernel[grid](
            target_toks_ptr=target_positions,
            next_toks_ptr=target_positions[end_locs] + 1,
            query_start_locs_ptr=start_locs,
            query_end_locs_ptr=end_locs,
            out_ptr_merged_toks=self.positions,
            out_ptr_is_rejected_tok=is_rejected_tok,
            target_toks_size=target_positions.shape[0],
            rejected_tok_fill=0,
        )

        # recompute slot mapping
        new_slot_mapping = compute_new_slot_mapping(
            cad=cad,
            new_positions=self.positions[:num_tokens],
            is_rejected_token_mask=is_rejected_tok,
            block_size=self._block_size(),
            max_model_len=self.max_model_len,
        )
        # update common_attn_metadata
        new_cad: CommonAttentionMetadata = extend_all_queries_by_1(
            cad,
            arange=self.arange,
            new_slot_mapping=new_slot_mapping,
        )

        new_last_token_indices = new_cad.query_start_loc[1:] - 1
        if num_rejected_tokens_gpu is not None:
            new_last_token_indices -= num_rejected_tokens_gpu

        return num_tokens, new_last_token_indices, new_cad

    def load_model(self, target_model: Any) -> None:
        """Takes target_model to satisfy the type checker."""

        # This must be computed before loading the draft model
        # because that mutates the forward_context of the vllm_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys()
        )

        from vllm.compilation.backends import set_model_tag

        draft_vllm_config: VllmConfig = create_vllm_config_for_draft_model(
            target_model_vllm_config=self.vllm_config
        )
        logger.info(
            "Starting to load draft model %s. TP=%d, rank=%d",
            draft_vllm_config.model_config.model,
            draft_vllm_config.parallel_config.tensor_parallel_size,
            draft_vllm_config.parallel_config.rank,
        )
        with set_model_tag("draft_model"):
            self.model = get_model(vllm_config=draft_vllm_config, prefix="draft_model")

        # This must be computed after loading the draft model
        # because that mutates the forward_context of the vllm_config
        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys()
            - target_attn_layer_names
        )
        self.attn_layer_names = list(draft_attn_layer_names)


def create_vllm_config_for_draft_model(
    target_model_vllm_config: VllmConfig,
) -> VllmConfig:
    """The vllm_config is configured for the target model, e.g.
    its quant_config and parallel_config. But the draft model is potentially
    quantized differently, and has potentially different tensor_parallel_size.
    This function creates a new vllm_config configured for the draft model.
    The vllm_config is useful when loading the draft model with get_model().
    """
    old = target_model_vllm_config
    new_parallel_config = old.speculative_config.draft_parallel_config.replace(
        rank=old.parallel_config.rank
    )
    new: VllmConfig = old.replace(
        quant_config=None,  # quant_config is recomputed in __init__()
        model_config=old.speculative_config.draft_model_config,
        parallel_config=new_parallel_config,
    )
    return new


def compute_new_slot_mapping(
    cad: CommonAttentionMetadata,
    new_positions: torch.Tensor,
    is_rejected_token_mask: torch.Tensor,
    block_size: int,
    max_model_len: int,
):
    batch_size, n_blocks_per_req = cad.block_table_tensor.shape
    req_indices = torch.arange(batch_size, device=cad.query_start_loc.device)
    req_indices = torch.repeat_interleave(
        req_indices, cad.naive_query_lens() + 1, output_size=len(new_positions)
    )
    # Clamp the positions to prevent an out-of-bounds error when indexing
    # into block_table_tensor.
    clamped_positions = torch.clamp(new_positions, max=max_model_len - 1)
    block_table_indices = (
        req_indices * n_blocks_per_req + clamped_positions // block_size
    )
    block_nums = cad.block_table_tensor.view(-1)[block_table_indices]
    block_offsets = clamped_positions % block_size
    new_slot_mapping = block_nums * block_size + block_offsets
    # Mask out the position ids that exceed the max model length.
    exceeds_max_model_len = new_positions >= max_model_len
    new_slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)
    # Mask out rejected tokens to prevent saves to the KV cache.
    new_slot_mapping.masked_fill_(is_rejected_token_mask, PADDING_SLOT_ID)
    return new_slot_mapping


@triton.jit
def merge_toks_kernel(
    target_toks_ptr,
    next_toks_ptr,
    query_start_locs_ptr,
    query_end_locs_ptr,
    out_ptr_merged_toks,
    out_ptr_is_rejected_tok,
    target_toks_size,
    rejected_tok_fill,
):
    """
    Merges the `target_toks_ptr` and the `next_toks_ptr` into a new tensor
    called `out_ptr_merged_toks`. Rejected tokens are those after the
    `query_end_locs_ptr` and before the next `query_start_locs_ptr`. Fills the
    rejected tokens positions with the value `rejected_tok_fill`. Also fills a mask
    of the rejected tokens in `out_ptr_is_rejected_tok`.
    """
    pid = tl.program_id(0)
    start_loc = tl.load(query_start_locs_ptr + pid)
    is_last_program = pid == tl.num_programs(0) - 1
    if is_last_program:
        next_start_loc = target_toks_size.to(tl.int32)
    else:
        next_start_loc = tl.load(query_start_locs_ptr + pid + 1).to(tl.int32)

    end_loc = tl.load(query_end_locs_ptr + pid)
    new_val = tl.load(next_toks_ptr + pid)
    for i in range(start_loc, next_start_loc + 1):
        if i <= end_loc:  # copy existing tokens
            old_val = tl.load(target_toks_ptr + i)
            tl.store(out_ptr_merged_toks + pid + i, old_val)
            tl.store(out_ptr_is_rejected_tok + pid + i, False)
        elif i == end_loc + 1:  # copy bonus token
            tl.store(out_ptr_merged_toks + pid + i, new_val)
            tl.store(out_ptr_is_rejected_tok + pid + i, False)
        else:  # fill rejected tokens
            tl.store(out_ptr_merged_toks + pid + i, rejected_tok_fill)
            tl.store(out_ptr_is_rejected_tok + pid + i, True)
