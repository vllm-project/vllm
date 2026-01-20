# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer

logger = init_logger(__name__)

PTD_METHODS = ("eagle-ptd", "eagle3-ptd")


@triton.jit
def ptd_prepare_inputs_kernel(
    target_token_ids_ptr,
    target_positions_ptr,
    target_hidden_ptr,
    mask_hidden_ptr,
    next_token_ids_ptr,
    last_token_indices_ptr,
    original_slot_mapping_ptr,
    block_table_ptr,
    in_query_start_loc_ptr,
    out_query_start_loc_ptr,
    out_input_ids_ptr,
    out_positions_ptr,
    out_hidden_ptr,
    out_slot_mapping_ptr,
    batch_size: tl.constexpr,
    K: tl.constexpr,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks: tl.constexpr,
    mask_token_id: tl.constexpr,
    max_model_len: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Prepares inputs for PTD (Parallel Token Decoding) by constructing:
    - Input token IDs: [shifted verified tokens, next_token, mask, mask, ...]
    - Positions: [verified positions, incremented positions for draft tokens]
    - Hidden states: [verified hidden states, mask_hidden for draft positions]
    - Slot mapping: [verified slots, computed slots for draft positions]
    """
    out_idx = tl.program_id(0)
    h_block = tl.program_id(1)

    req_idx = 0
    for r in range(batch_size):
        out_start = tl.load(out_query_start_loc_ptr + r)
        out_end = tl.load(out_query_start_loc_ptr + r + 1)
        req_idx = tl.where((out_idx >= out_start) & (out_idx < out_end), r, req_idx)

    in_start = tl.load(in_query_start_loc_ptr + req_idx)
    out_start = tl.load(out_query_start_loc_ptr + req_idx)
    global_last_idx = tl.load(last_token_indices_ptr + req_idx)
    last_idx = global_last_idx - in_start

    local_idx = out_idx - out_start
    is_verified = local_idx <= last_idx

    if h_block == 0:
        if is_verified:
            if local_idx < last_idx:
                out_tok = tl.load(target_token_ids_ptr + in_start + local_idx + 1)
            else:
                out_tok = tl.load(next_token_ids_ptr + req_idx)
        else:
            out_tok = mask_token_id
        tl.store(out_input_ids_ptr + out_idx, out_tok)

        if is_verified:
            out_pos = tl.load(target_positions_ptr + in_start + local_idx)
        else:
            last_pos = tl.load(target_positions_ptr + in_start + last_idx)
            out_pos = last_pos + (local_idx - last_idx)
            out_pos = tl.where(out_pos >= max_model_len, 0, out_pos)
        tl.store(out_positions_ptr + out_idx, out_pos)

        if is_verified:
            slot = tl.load(original_slot_mapping_ptr + in_start + local_idx)
        else:
            last_pos = tl.load(target_positions_ptr + in_start + last_idx)
            draft_pos = last_pos + (local_idx - last_idx)
            draft_pos = tl.where(draft_pos >= max_model_len, 0, draft_pos)
            block_num = draft_pos // block_size
            block_offset = draft_pos % block_size
            block_id = tl.load(block_table_ptr + req_idx * max_blocks + block_num)
            slot = block_id * block_size + block_offset
        tl.store(out_slot_mapping_ptr + out_idx, slot)

    h_start = h_block * BLOCK_H
    h_offs = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offs < hidden_size

    if is_verified:
        out_vals = tl.load(
            target_hidden_ptr + (in_start + local_idx) * hidden_size + h_offs,
            mask=h_mask, other=0.0
        )
    else:
        out_vals = tl.load(mask_hidden_ptr + h_offs, mask=h_mask, other=0.0)

    tl.store(out_hidden_ptr + out_idx * hidden_size + h_offs, out_vals, mask=h_mask)


class PtdEagleProposer(EagleProposer):
    """Generates K draft tokens in a single forward pass using mask tokens."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner)
        self._raise_if_unsupported_method()
        self._raise_if_multimodal()

        self.K = self.num_speculative_tokens
        self.slot_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self.arange_K = torch.arange(self.K, device=device, dtype=torch.int64)

        self.mask_hidden: torch.Tensor | None = None
        self.mask_token_id: int | None = None
        self.block_size = vllm_config.cache_config.block_size

    def _raise_if_unsupported_method(self):
        if self.method not in PTD_METHODS:
            raise ValueError(
                f"PtdEagleProposer only supports methods {PTD_METHODS}, "
                f"got {self.method}"
            )

    def _raise_if_multimodal(self):
        if self.supports_mm_inputs:
            raise NotImplementedError(
                "PTD speculative decoding does not support multimodal models"
            )

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        if self.method != "eagle3-ptd":
            return False
        use_aux = True
        eagle_config = getattr(
            self.draft_model_config.hf_config, "eagle_config", None
        )
        if eagle_config is not None:
            use_aux = eagle_config.get("use_aux_hidden_state", True)
        return use_aux

    def load_model(self, target_model: nn.Module) -> None:
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        )

        from vllm.compilation.backends import set_model_tag
        with set_model_tag("ptd_eagle_head"):
            self.model = get_model(
                vllm_config=self.vllm_config, model_config=draft_model_config
            )

        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
            - target_attn_layer_names
        )
        self.attn_layer_names = list(draft_attn_layer_names)

        config = self.draft_model_config.hf_config
        self.mask_token_id = getattr(config, 'ptd_token_id', None)
        if self.mask_token_id is None:
            raise ValueError(
                "PTD requires 'ptd_token_id' in draft model config.json"
            )
        self.mask_token_id = int(self.mask_token_id)

        self.mask_hidden = self._load_mask_hidden()

        if self.method == "eagle3-ptd" and self.eagle3_use_aux_hidden_state:
            expected_aux_size = self.hidden_size * 3
            if self.mask_hidden.shape[-1] == expected_aux_size:
                self.mask_hidden = self.model.combine_hidden_states(self.mask_hidden)
                logger.info(
                    "Transformed mask_hidden from aux format to hidden_size"
                )

    def _load_mask_hidden(self) -> torch.Tensor:
        checkpoint_path = Path(self.draft_model_config.model)

        def normalize_shape(t: torch.Tensor) -> torch.Tensor:
            t = t.to(device=self.device, dtype=self.dtype)
            while t.dim() > 2 and t.size(0) == 1:
                t = t.squeeze(0)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return t

        from safetensors.torch import load_file

        # Look for mask_hidden in main model safetensors files
        safetensor_files = list(checkpoint_path.glob("*.safetensors"))
        for path in sorted(safetensor_files):
            try:
                weights = load_file(str(path))
                if "mask_hidden" in weights:
                    logger.info(f"Loaded mask_hidden from {path}")
                    return normalize_shape(weights["mask_hidden"])
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        # Fallback: use embedding of mask token
        logger.warning(
            "mask_hidden not found in checkpoint, "
            "using embedding of ptd_token_id as fallback"
        )
        embed = self.model.model.embed_tokens.weight[self.mask_token_id]
        return normalize_shape(embed)

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mm_embed_inputs is not None:
            raise NotImplementedError(
                "PTD speculative decoding does not support multimodal inputs"
            )

        batch_size = next_token_ids.shape[0]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        if self.method == "eagle3-ptd":
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )

        if self.attn_metadata_builder is None:
            self.attn_metadata_builder = self._get_attention_metadata_builder()

        K = self.K
        draft_len = K - 1
        in_qsl = common_attn_metadata.query_start_loc

        accepted_lengths = last_token_indices - in_qsl[:batch_size] + 1
        out_lens = accepted_lengths + draft_len

        out_qsl = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=self.device
        )
        out_qsl[1:] = torch.cumsum(out_lens, dim=0)

        total_out = (
            common_attn_metadata.num_actual_tokens + batch_size * draft_len
        )

        in_qsl_cpu = common_attn_metadata.query_start_loc_cpu
        accepted_lengths_cpu = (
            in_qsl_cpu[1:batch_size+1] - in_qsl_cpu[:batch_size]
        )
        out_qsl_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
        out_qsl_cpu[1:] = torch.cumsum(accepted_lengths_cpu + draft_len, dim=0)

        slot_mapping = self._prepare_ptd_inputs(
            target_token_ids, target_positions, target_hidden_states,
            next_token_ids, last_token_indices,
            common_attn_metadata.slot_mapping,
            common_attn_metadata.block_table_tensor,
            in_qsl, out_qsl, total_out, batch_size
        )

        seq_lens = common_attn_metadata.seq_lens.clone()
        if num_rejected_tokens_gpu is not None:
            seq_lens = seq_lens - num_rejected_tokens_gpu
        seq_lens = (seq_lens + K).to(common_attn_metadata.seq_lens.dtype)

        common_attn_metadata.query_start_loc = out_qsl
        common_attn_metadata.query_start_loc_cpu = out_qsl_cpu
        common_attn_metadata.seq_lens = seq_lens
        common_attn_metadata.num_actual_tokens = total_out
        common_attn_metadata.max_query_len = (
            common_attn_metadata.max_query_len + draft_len
        )
        common_attn_metadata.max_seq_len = (
            common_attn_metadata.max_seq_len + draft_len
        )
        common_attn_metadata.slot_mapping = slot_mapping
        common_attn_metadata._seq_lens_cpu = None
        common_attn_metadata._num_computed_tokens_cpu = None

        attn_metadata = self.attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )
        per_layer_metadata = {
            name: attn_metadata for name in self.attn_layer_names
        }

        num_input, cudagraph_mode = self._get_ptd_cudagraph_config(total_out)

        hidden = self._run_ptd_forward(
            num_input, total_out, per_layer_metadata, cudagraph_mode
        )

        ends = out_qsl[1:batch_size+1]
        starts = ends - K
        indices = starts.unsqueeze(1) + self.arange_K
        hidden_selected = hidden[indices.flatten()]

        logits = self.model.compute_logits(hidden_selected)
        return logits.argmax(dim=-1).view(batch_size, K)

    def _prepare_ptd_inputs(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        in_qsl: torch.Tensor,
        out_qsl: torch.Tensor,
        total_out: int,
        batch_size: int,
    ) -> torch.Tensor:
        BLOCK_H = 256
        num_h_blocks = (self.hidden_size + BLOCK_H - 1) // BLOCK_H

        ptd_prepare_inputs_kernel[(total_out, num_h_blocks)](
            target_token_ids, target_positions, target_hidden_states,
            self.mask_hidden, next_token_ids, last_token_indices,
            slot_mapping, block_table, in_qsl, out_qsl,
            self.input_ids, self.positions, self.hidden_states, self.slot_buffer,
            batch_size=batch_size, K=self.K, hidden_size=self.hidden_size,
            block_size=self.block_size, max_blocks=block_table.shape[1],
            mask_token_id=self.mask_token_id, max_model_len=self.max_model_len,
            BLOCK_H=BLOCK_H
        )
        return self.slot_buffer[:total_out]

    def _get_ptd_cudagraph_config(
        self, num_tokens: int
    ) -> tuple[int, CUDAGraphMode]:
        num_padded, _ = self._pad_batch_across_dp(num_tokens, num_tokens)

        if (
            self.use_cuda_graph
            and num_padded <= self.compilation_config.max_cudagraph_capture_size
        ):
            num_input = self.vllm_config.pad_for_cudagraph(num_padded)
            return num_input, CUDAGraphMode.PIECEWISE

        return num_padded, CUDAGraphMode.NONE

    def _run_ptd_forward(
        self,
        num_input: int,
        num_out: int,
        per_layer_metadata: dict,
        cudagraph_mode: CUDAGraphMode,
    ) -> torch.Tensor:
        with set_forward_context(
            per_layer_metadata, self.vllm_config,
            num_tokens=num_input, cudagraph_runtime_mode=cudagraph_mode
        ):
            result = self.model(
                input_ids=self.input_ids[:num_input],
                positions=self._get_positions(num_input),
                hidden_states=self.hidden_states[:num_input],
                inputs_embeds=None,
            )
            hidden = result[0] if isinstance(result, tuple) else result
        return hidden[:num_out]
