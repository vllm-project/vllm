# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id

logger = init_logger(__name__)


class OrthrusSpeculator(DraftModelSpeculator):
    """Orthrus diffusion drafter for ModelRunnerV2.

    Orthrus reuses the target model as its drafter and runs the model's
    diffusion attention path over a separate query block:
    [bonus_token, mask, ..., mask]. Unlike DFlash, the draft tokens are sampled
    from diffusion logits[:, :-1], so the bonus position is sampled and the
    final mask position is skipped.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "orthrus"
        super().__init__(vllm_config, device)

        self.supports_mm_inputs = False
        self.num_query_per_req = 1 + self.num_speculative_steps
        self.parallel_drafting_token_id = get_parallel_drafting_token_id(
            self.draft_model_config.hf_config
        )

        self.context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self.context_slot_mapping = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        max_num_sampled_tokens = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_pos = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_idx_mapping = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int32, device=device
        )
        self.sample_col = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)

        self.query_cudagraph_manager: DFlashCudaGraphManager | None = None
        self.draft_kv_cache_group_id: int = -1

    def load_model(self, target_model: nn.Module) -> None:
        self.model = target_model
        if not hasattr(self.model, "get_top_tokens"):
            raise ValueError(
                "Orthrus enables local argmax reduction for draft token "
                f"generation, but {self.model.__class__.__name__} does not "
                "implement get_top_tokens()."
            )

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self.draft_attn_layer_names = {
            name for name in all_attn_layers if name.endswith(".attn_diff")
        }
        if not self.draft_attn_layer_names:
            raise ValueError(
                "Orthrus requires diffusion attention layers ending in "
                "'.attn_diff', but none were found in the target model."
            )
        logger.info(
            "Using local argmax reduction for Orthrus draft token generation "
            "(communication: O(2*tp_size) vs O(vocab_size))."
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        raise NotImplementedError(
            "Orthrus reuses the target model and initializes it in load_model()."
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.num_query_per_req,
            causal=False,
        )

    def capture(self, attn_states: dict | None = None) -> None:
        logger.info("Capturing model for Orthrus speculator...")
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.zero_()
        assert self.query_cudagraph_manager is not None
        self.query_cudagraph_manager.capture(
            self._generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            progress_bar_desc="Capturing orthrus CUDA graphs",
        )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)
        draft_groups = [gid for gid, group in enumerate(self.attn_groups) if group]
        assert len(draft_groups) == 1, (
            "Orthrus currently requires all diffusion attention layers to share "
            "a single kv-cache group."
        )
        self.draft_kv_cache_group_id = draft_groups[0]
        self.draft_block_size = self.block_tables.block_sizes[
            self.draft_kv_cache_group_id
        ]

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
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
            ret = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
                is_diffusion_pass=True,
            )
        if isinstance(ret, tuple):
            return ret[0]
        return ret

    def _sample_draft(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        idx_mapping: torch.Tensor,
    ) -> torch.Tensor:
        if self.draft_logits is None:
            return self.model.get_top_tokens(hidden_states)
        return self.sample_draft(
            hidden_states,
            positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.sample_col[: hidden_states.shape[0]],
            self.draft_logits,
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
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        draft_tokens = self._sample_draft(
            sample_hidden_states,
            self.sample_pos[:num_sample],
            self.sample_idx_mapping[:num_sample],
        )
        self.draft_tokens[:num_reqs] = draft_tokens.view(
            num_reqs, self.num_speculative_steps
        )

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        num_query_per_req: int | None = None,
        causal: bool = False,
    ) -> dict[str, Any] | None:
        if not self.draft_attn_layer_names:
            return None
        assert num_query_per_req is None
        return super()._build_draft_attn_metadata(
            num_reqs,
            num_reqs_padded,
            num_tokens_padded,
            num_query_per_req=self.num_query_per_req,
            causal=causal,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_query_tokens = num_reqs * self.num_query_per_req
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_query_per_req, self.max_model_len
        )

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        if dummy_run and skip_attn_for_dummy_run:
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            return self.draft_tokens[:num_reqs]

        assert self.draft_kv_cache_group_id >= 0
        query_slot_mapping = self.block_tables.slot_mappings[
            self.draft_kv_cache_group_id
        ]
        prepare_dflash_inputs(
            self.input_buffers,
            query_slot_mapping,
            self.context_positions,
            self.context_slot_mapping,
            self.sample_indices,
            self.sample_pos,
            self.sample_idx_mapping,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.block_tables.input_block_tables[self.draft_kv_cache_group_id],
            self.draft_block_size,
            self.parallel_drafting_token_id,
            self.num_query_per_req,
            self.num_speculative_steps,
            self.max_num_reqs,
            self.max_num_tokens,
            sample_bonus_token=True,
        )

        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_query_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens_padded = batch_desc.num_tokens
        draft_attn_metadata = self._build_draft_attn_metadata(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
        )
        draft_slot_mappings_by_layer = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_tokens_padded],
            self.kv_cache_config,
        )

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_draft(
                num_reqs_padded,
                num_tokens_padded,
                draft_attn_metadata,
                draft_slot_mappings_by_layer,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )

        return self.draft_tokens[:num_reqs]
