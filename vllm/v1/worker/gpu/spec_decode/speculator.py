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
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.watermarking import create_watermarker
from vllm.v1.watermarking.spec_decode import (
    DraftWatermarker,
    create_speculative_draft_watermarker,
)
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
    init_attn_backend,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.batch_shard import batch_sharded_sampling_enabled
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    OnlineAcceptanceEstimator,
)
from vllm.v1.worker.gpu.spec_decode.cudagraph_utils import SpeculatorCudaGraphManager
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
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
    num_query_per_req: int = 1
    # Extra query slots reserved per request outside the regular queries.
    num_extra_query_per_req: int = 0

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
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        pass


class DraftModelSpeculator(BaseSpeculator):
    # Draft prefill runs on the target's batch, so it can reuse the target
    # forward's attention metadata. Otherwise the drafter rebuilds it.
    can_reuse_target_attn_metadata: bool = True

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
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        if vllm_config.parallel_config.use_ubatching:
            # A microbatched target forward builds only per-microbatch metadata.
            self.can_reuse_target_attn_metadata = False

        self.eplb_state: EplbState | None = None

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.idx_mapping = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
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

        self.draft_token_confidence_probs = torch.empty_like(
            self.draft_tokens, dtype=torch.float32
        )
        self.enable_adaptive_verification = (
            self.speculative_config.enable_adaptive_verification
        )
        self.use_acceptance_estimator = self.enable_adaptive_verification
        self.acceptance_estimator: OnlineAcceptanceEstimator | None = None

        # Pre-temperature logits, cached from the previous decode step.
        # Allocated in load_model. See _init_draft_logits.
        self.draft_logits: torch.Tensor | None = None
        self.use_draft_logits = (
            self.speculative_config.draft_sample_method == "probabilistic"
        )
        self.draft_logits_num_shards = 1
        self.draft_logits_shard_idx = 0

        self.draft_watermarker: DraftWatermarker | None = None
        watermark_config = getattr(vllm_config, "watermark_config", None)
        if watermark_config is not None:
            watermarker = create_watermarker(watermark_config)
            self.draft_watermarker = create_speculative_draft_watermarker(
                watermarker,
                self.max_num_reqs,
                device,
                watermark_config.allow_target_only_watermarking,
                self.num_speculative_steps,
                watermark_config.deduplicate_contexts,
                watermark_config.deduplicate_contexts_max_history,
            )

        self.supports_mm_inputs = False
        self.pcp_manager: PCPManager | None = None
        # Whether every draft attention builder can refresh its decode metadata
        # in place, so the refresh can be recorded in the draft CUDA graph.
        self.supports_decode_metadata_update = False
        self.prefill_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None

    @property
    def max_model_len(self) -> int:
        # Auto-fit can reduce the limit after the speculator is built.
        return self.vllm_config.model_config.max_model_len

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_speculative_steps + 1,
        )

        # PIECEWISE cudagraphs are not supported for draft decodes.
        decode_mode = (
            CUDAGraphMode.FULL_DECODE_ONLY
            if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            else CUDAGraphMode.NONE
        )
        self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config, self.device, decode_mode, decode_query_len=1
        )

    @abstractmethod
    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
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
        if self.use_draft_logits:
            self._init_draft_logits(target_model)

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self.draft_attn_layer_names = set(all_attn_layers) - target_attn_layer_names
        for layer_name in self.draft_attn_layer_names:
            all_attn_layers[layer_name].is_draft_layer = True

        target_supports_mm = self.vllm_config.model_config.supports_multimodal_inputs
        draft_supports_mm = supports_multimodal_embeddings(self.model)
        self.supports_mm_inputs = target_supports_mm and draft_supports_mm
        if target_supports_mm and not draft_supports_mm:
            logger.warning_once(
                "Draft model %s does not support external multimodal embeddings. "
                "Embeddings from the target model will not be passed to the "
                "drafter; using text-only draft inputs instead.",
                type(self.model).__name__,
            )

        if self.use_acceptance_estimator:
            if self.use_local_argmax_reduction:
                raise ValueError(
                    "Adaptive verification without a confidence head estimates "
                    "per-position acceptance from the draft logits, which "
                    "use_local_argmax_reduction never materializes. Disable one "
                    "of them."
                )
            self.acceptance_estimator = OnlineAcceptanceEstimator(
                self.max_num_reqs,
                self.num_speculative_steps,
                self.device,
            )

    def supports_batch_sharded_draft_logits(self) -> bool:
        """Whether every draft logits write on this speculator's path honors
        draft_logits_num_shards and draft_logits_shard_idx."""
        return False

    def _init_draft_logits(self, target_model: nn.Module) -> None:
        """Allocate the draft logits cache.

        Under batch-sharded sampling, the rank that verifies a request is the
        only one that reads its draft logits, so each TP rank keeps just the
        rows of the requests it owns, at row
        req_state_idx // draft_logits_num_shards.
        """
        parallel_config = self.vllm_config.parallel_config
        if (
            batch_sharded_sampling_enabled(self.vllm_config, target_model)
            # PCPManager.sync_drafts indexes the draft logits by request state,
            # so it must see every request's logits.
            and parallel_config.prefill_context_parallel_size == 1
            and self.draft_watermarker is None
            and self.supports_batch_sharded_draft_logits()
        ):
            tp_group = get_tp_group()
            self.draft_logits_num_shards = tp_group.world_size
            self.draft_logits_shard_idx = tp_group.rank_in_group
        dtype, fill = self.draft_logits_spec(self.vllm_config)
        self.draft_logits = torch.full(
            (
                cdiv(self.max_num_reqs, self.draft_logits_num_shards),
                self.num_speculative_steps,
                self.vocab_size,
            ),
            fill,
            dtype=dtype,
            device=self.device,
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
        self._configure_decode_metadata_update()

    def _build_drafter_attn_metadata(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        for_capture: bool = False,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Attention metadata and slot mappings for draft prefill on the target
        batch: the target forward's when reusable, else rebuilt on the draft
        builders from the block tables and slot mappings the target prepared.
        Must run before draft decode steps overwrite the slot mappings.
        """
        if self.can_reuse_target_attn_metadata:
            return attn_metadata, slot_mappings
        block_tables = tuple(
            block_table[: input_batch.num_reqs_after_padding]
            for block_table in self.block_tables.input_block_tables
        )
        target_slot_mappings = self.block_tables.slot_mappings[
            :, : input_batch.num_tokens_after_padding
        ]
        attn_metadata = self.model_state.prepare_attn(
            input_batch,
            cg_mode,
            block_tables,
            target_slot_mappings,
            self.attn_groups,
            self.kv_cache_config,
            for_capture=for_capture,
        )
        return attn_metadata, build_slot_mappings_by_layer(
            target_slot_mappings, self.kv_cache_config
        )

    def _configure_decode_metadata_update(self) -> None:
        unsupported_backends = sorted(
            {
                attn_group.backend.get_name()
                for attn_groups in self.attn_groups
                for attn_group in attn_groups
                if not attn_group.supports_draft_decode_metadata_update
            }
        )
        self.supports_decode_metadata_update = not unsupported_backends
        if unsupported_backends:
            logger.info_once(
                "Attention backend(s) %s cannot update draft decode metadata in "
                "place; falling back to rebuilding attention metadata eagerly.",
                ", ".join(unsupported_backends),
            )

    def _build_attn_metadata(
        self,
        num_reqs: int,
        batch_desc: BatchExecutionDescriptor,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
        slot_mappings: torch.Tensor | None = None,
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
        if slot_mappings is None:
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
        if dcp_local_seq_lens is None and self.dcp_size > 1:
            # Draft steps advance and rewind their own global sequence lengths,
            # so the target model's DCP-local lengths may already be stale.
            dcp_local_seq_lens = prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.dcp_size,
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
            max_seq_len=min(self.draft_max_seq_len, self.max_model_len),
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
            causal=causal,
            seq_lens_cpu_upper_bound=draft_seq_lens_cpu_upper_bound,
            positions=self.input_buffers.positions[:num_tokens],
            is_prefilling=self.draft_is_prefilling[:num_reqs_padded],
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

    def compute_draft_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int
    ) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    def get_draft_top_tokens(
        self, hidden_states: torch.Tensor, spec_step_idx: int
    ) -> torch.Tensor:
        return self.model.get_top_tokens(hidden_states)

    def sample_draft(
        self,
        hidden_states: torch.Tensor,
        sample_src_positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
        spec_step_idx: int = 0,
        num_steps: int = 1,
    ) -> torch.Tensor:
        if draft_logits is None and self.use_local_argmax_reduction:
            return self.get_draft_top_tokens(hidden_states, spec_step_idx)

        logits = self.compute_draft_logits(hidden_states, spec_step_idx)
        if draft_logits is not None and self.draft_watermarker is not None:
            sampled = self.draft_watermarker.sample(
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
                num_steps=num_steps,
            )
        elif draft_logits is not None:
            sampled = gumbel_sample(
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
                logits_cache_num_shards=self.draft_logits_num_shards,
                logits_cache_shard_idx=self.draft_logits_shard_idx,
            )
        else:
            sampled = logits.argmax(dim=-1)
        self._maybe_predict_acceptance(logits, idx_mapping, draft_step)
        return sampled

    def _maybe_predict_acceptance(
        self, logits: torch.Tensor, idx_mapping: torch.Tensor, draft_step: torch.Tensor
    ) -> None:
        if self.acceptance_estimator is not None:
            self.acceptance_estimator.predict(
                logits,
                idx_mapping,
                draft_step,
                self.draft_token_confidence_probs,
                self.temperature,
            )

    def observe_verification(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        """Fold the target's verdict on the last drafts into the estimator.

        Must run before the next `propose`, which overwrites the per-slot
        features the verdict grades.
        """
        if self.acceptance_estimator is not None:
            self.acceptance_estimator.step(idx_mapping, num_sampled, num_rejected)

    def prepare_watermarking(
        self,
        sampler: "GPUWatermarkSampler",
        idx_mapping: torch.Tensor,
    ) -> None:
        if self.draft_watermarker is None:
            return
        self.draft_watermarker.prepare(
            sampler._get_contexts(idx_mapping),
            sampler.watermarking.gpu[idx_mapping],
            sampler.req_states.all_token_ids.gpu,
            sampler.req_states.prompt_len.gpu,
            sampler.req_states.total_len.gpu,
        )

    def _copy_request_inputs(
        self,
        num_reqs: int,
        # [num_reqs]
        idx_mapping: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        dummy_run: bool = False,
    ) -> None:
        # Copy temperature, seeds, and idx mapping to the pre-allocated buffers.
        # NOTE(woosuk): For draft sampling, we only consider the temperature
        # and ignore the other sampling parameters such as top_k and top_p,
        # for simplicity and performance.
        # While this may slightly degrade the acceptance rate, it does not
        # affect the output distribution after rejection sampling.
        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        # idx_mapping == -1 marks a row the drafter must not act on: sampling
        # skips it and compute_slot_mappings emits PAD. CUDA-graph padded rows
        # always get it; a dummy batch gets it for every row, since its arange
        # idx_mapping names request-state slots with stale block tables.
        if dummy_run:
            self.idx_mapping.fill_(-1)
        else:
            self.idx_mapping[:num_reqs].copy_(idx_mapping)
            self.idx_mapping[num_reqs:].fill_(-1)

    def _build_uniform_batch_dp_sync(
        self,
        target_dp_sync_state: DPSyncState,
        num_reqs: int,
        num_query_per_req: int = 1,
    ) -> tuple[DPSyncState, int]:
        num_batch_tokens = target_dp_sync_state.num_reqs * num_query_per_req
        assert num_reqs * num_query_per_req <= num_batch_tokens, (
            "reusing a DP sync that does not cover this batch's requests"
        )
        return replace(
            target_dp_sync_state,
            num_tokens_across_dp=torch.full_like(
                target_dp_sync_state.num_tokens_across_dp, num_batch_tokens
            ),
            uniform_token_count=num_query_per_req,
            max_query_len=num_query_per_req,
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

    def _update_draft_decode_metadata(
        self, attn_metadata: dict[str, Any], num_reqs: int
    ) -> None:
        if self.block_tables.cp_size > 1:
            prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.block_tables.cp_size,
                self.block_tables.cp_rank,
                self.block_tables.cp_interleave,
            )
        for groups in self.attn_groups:
            for group in groups:
                group.update_draft_decode_metadata(attn_metadata)
