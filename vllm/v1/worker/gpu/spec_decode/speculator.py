# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    init_attn_backend,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import maybe_prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

logger = init_logger(__name__)


def _target_feeds_hc_residual(vllm_config: VllmConfig) -> bool:
    """Whether the target replaces the drafter's input with its HC residual.

    Keyed on the same hook the model runner calls to perform that swap. It is
    resolved from the target model class because speculators are built before
    the target model is instantiated.
    """
    from vllm.model_executor.model_loader.utils import get_model_cls

    target_cls = get_model_cls(vllm_config.model_config)
    return hasattr(target_cls, "get_mtp_target_hidden_states")


class BaseSpeculator(ABC):
    @abstractmethod
    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    @abstractmethod
    def capture(self) -> None:
        pass

    @abstractmethod
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
        # [num_prefill_lookahead, max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        pass


class DraftModelSpeculator(BaseSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.method = self.speculative_config.method
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.draft_model_config = self.speculative_config.draft_model_config

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.draft_max_seq_len = self.max_model_len
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        # Widen for HC-multiplexed residuals: a target that implements
        # get_mtp_target_hidden_states() (e.g. DeepSeek V4) hands the drafter
        # its pre-hc_head (T, hc_mult * hidden_size) residual in place of the
        # collapsed hidden states, so the drafter's buffers must match. Key off
        # that hook rather than hc_mult alone -- HY V4 runs iHC in its backbone
        # (hc_mult=4) but its MTP head consumes the collapsed states, so
        # widening it feeds propose() a 4x-too-wide buffer.
        if _target_feeds_hc_residual(vllm_config):
            hc_mult = getattr(self.draft_model_config.hf_config, "hc_mult", 1)
            self.hidden_size = self.hidden_size * hc_mult
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype
        self.use_fp64_gumbel = vllm_config.model_config.use_fp64_gumbel
        self.use_local_argmax_reduction = (
            self.speculative_config.use_local_argmax_reduction
        )

        # DP configuration
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        self.eplb_state: EplbState | None = None

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
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
        self.arange_np = np.arange(self.max_num_reqs + 1, dtype=np.int32)
        self.draft_is_prefilling = torch.zeros(self.max_num_reqs, dtype=torch.bool)

        self.draft_logits: torch.Tensor | None = None
        if self.speculative_config.draft_sample_method == "probabilistic":
            # Pre-temperature logits, cached from the previous decode step.
            dtype, fill = self.draft_logits_spec(vllm_config)
            self.draft_logits = torch.full(
                (
                    self.max_num_reqs,
                    self.num_speculative_steps,
                    self.vocab_size,
                ),
                fill,
                dtype=dtype,
                device=device,
            )

        self.supports_mm_inputs = False
        self.pcp_manager: PCPManager | None = None

    @abstractmethod
    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        pass

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        self.model = self.load_draft_model(target_model, target_attn_layer_names)
        self._validate_local_argmax_reduction()

        all_attn_layers = set[str](
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )
        self.draft_attn_layer_names = all_attn_layers - target_attn_layer_names

        target_supports_mm = MULTIMODAL_REGISTRY.supports_multimodal_inputs(
            self.vllm_config.model_config
        )
        draft_supports_mm = supports_multimodal_embeddings(self.model)
        self.supports_mm_inputs = target_supports_mm and draft_supports_mm
        if target_supports_mm and not draft_supports_mm:
            logger.warning_once(
                "Draft model %s does not support external multimodal embeddings. "
                "Embeddings from the target model will not be passed to the "
                "drafter; using text-only draft inputs instead.",
                type(self.model).__name__,
            )

    def set_eplb_state(self, eplb_state: EplbState) -> None:
        """Inject EPLB state after construction."""
        self.eplb_state = eplb_state

    def _prepare_eplb_forward(self, num_unpadded_tokens: int) -> None:
        """Call EPLB prepare_forward if EPLB is active for the draft model."""
        if self.eplb_state is not None:
            self.eplb_state.prepare_forward(
                self.speculative_config.draft_model_config,
                num_unpadded_tokens,
            )

    @property
    def attn_vllm_config(self) -> VllmConfig:
        """Config for the draft's attention metadata builders. Overridden by
        speculators whose attention mode differs from the target's."""
        return self.vllm_config

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.attn_groups, self.attn_cg_support, _ = init_attn_backend(
            kv_cache_config,
            self.attn_vllm_config,
            self.device,
            active_layer_names=self.draft_attn_layer_names,
        )
        self.block_tables = block_tables
        # The target model runner's buffers and attention groups. Draft
        # prefill reuses the target model's attention metadata, so its
        # cudagraph capture must build dummy metadata through the same
        # builders and buffers.
        self.target_input_buffers = target_input_buffers
        self.target_attn_groups = target_attn_groups

    def _build_attn_metadata(
        self,
        num_reqs: int,
        batch_desc: BatchExecutionDescriptor,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        # A FULL graph replays a captured shape whose padded requests each hold
        # a full query width, so attention must see the padded token count.
        # PIECEWISE/eager needs the actual token count, because batch_desc may
        # carry graph or DP padding that no request owns, which would desync it
        # from query_start_loc.
        num_tokens = (
            batch_desc.num_tokens
            if batch_desc.cg_mode == CUDAGraphMode.FULL
            else int(query_start_loc_np[-1])
        )
        query_start_loc_cpu = torch.empty(num_reqs_padded + 1, dtype=torch.int32)
        query_start_loc_cpu[: num_reqs + 1] = torch.from_numpy(
            query_start_loc_np[: num_reqs + 1]
        )
        query_start_loc_cpu[num_reqs:] = query_start_loc_cpu[num_reqs]
        max_query_len = int((query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).max())
        block_tables = [
            x[:num_reqs_padded] for x in self.block_tables.input_block_tables
        ]
        slot_mappings = self.block_tables.slot_mappings[:, :num_tokens]
        draft_seq_lens_cpu_upper_bound = torch.zeros(
            num_reqs_padded, dtype=torch.int32, device="cpu"
        )
        torch.add(
            seq_lens_cpu_upper_bound[:num_reqs],
            step,
            out=draft_seq_lens_cpu_upper_bound[:num_reqs],
        )
        draft_seq_lens_cpu_upper_bound[:num_reqs].clamp_(max=self.max_model_len)
        if dcp_local_seq_lens is None and self.block_tables.cp_size > 1:
            # Draft steps advance and rewind their own global sequence lengths,
            # so the target model's DCP-local lengths may already be stale.
            dcp_local_seq_lens = maybe_prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.block_tables.cp_size,
                self.block_tables.cp_rank,
                self.block_tables.cp_interleave,
            )
        attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens,
            query_start_loc_gpu=self.input_buffers.query_start_loc[
                : num_reqs_padded + 1
            ],
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            dcp_local_seq_lens=(
                None
                if dcp_local_seq_lens is None
                else dcp_local_seq_lens[:num_reqs_padded]
            ),
            max_seq_len=self.draft_max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
            causal=causal,
            seq_lens_cpu_upper_bound=draft_seq_lens_cpu_upper_bound,
            is_prefilling=self.draft_is_prefilling[:num_reqs],
        )
        return attn_metadata

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        """Dtype and fill for the cached proposal distribution.

        Speculators that write only a subset of columns each step override this.
        """
        return vllm_config.model_config.head_dtype, 0.0

    def _validate_local_argmax_reduction(self) -> None:
        if not self.use_local_argmax_reduction:
            return
        if self.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "use_local_argmax_reduction is not compatible with "
                "draft_sample_method='probabilistic'."
            )
        if not hasattr(self.model, "get_top_tokens"):
            raise ValueError(
                "use_local_argmax_reduction is enabled but draft model "
                f"{self.model.__class__.__name__} does not implement "
                "get_top_tokens()."
            )
        logger.info(
            "Using local argmax reduction for draft token generation "
            "(communication: O(2*tp_size) vs O(vocab_size))."
        )

    def _greedy_sample_draft(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)
        logits = self.model.compute_logits(hidden_states)
        return logits.argmax(dim=-1)

    def sample_draft(
        self,
        hidden_states: torch.Tensor,
        sample_src_positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if draft_logits is not None:
            logits = self.model.compute_logits(hidden_states)
            return gumbel_sample(
                logits,
                idx_mapping,
                temperature,
                seeds,
                sample_src_positions,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=draft_logits,
                logits_cache_col=draft_step,
                use_fp64=self.use_fp64_gumbel,
            )
        return self._greedy_sample_draft(hidden_states)

    def _copy_request_inputs(
        self,
        num_reqs: int,
        # [num_reqs]
        idx_mapping: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
    ) -> None:
        # Copy temperature, seeds, and idx mapping to the pre-allocated buffers.
        # NOTE(woosuk): For draft sampling, we only consider the temperature
        # and ignore the other sampling parameters such as top_k and top_p,
        # for simplicity and performance.
        # While this may slightly degrade the acceptance rate, it does not
        # affect the output distribution after rejection sampling.
        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        self.idx_mapping[:num_reqs].copy_(idx_mapping)
        # idx_mapping for CG padded requests points to -1, which is ignored
        # during sampling to prevent writing stale values to draft logits.
        self.idx_mapping[num_reqs:].fill_(-1)

    def _build_uniform_batch_dp_sync(
        self,
        target_dp_sync: DPSyncState,
        num_reqs: int,
        num_query_per_req: int = 1,
    ) -> tuple[DPSyncState, int]:
        num_batch_tokens = target_dp_sync.num_reqs * num_query_per_req
        assert num_reqs * num_query_per_req <= num_batch_tokens, (
            "reusing a DP sync that does not cover this batch's requests"
        )
        return replace(
            target_dp_sync,
            num_tokens_across_dp=torch.full_like(
                target_dp_sync.num_tokens_across_dp, num_batch_tokens
            ),
            uniform_token_count=num_query_per_req,
            eager=False,
        ), num_batch_tokens

    def _build_uniform_attn_metadata(
        self,
        batch_desc: BatchExecutionDescriptor,
        num_reqs: int,
        num_query_per_req: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        query_start_loc_np = self.arange_np[: num_reqs + 1] * num_query_per_req
        return self._build_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=batch_desc,
            query_start_loc_np=query_start_loc_np,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=step,
            causal=causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
        )
