# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class AutoRegressiveSpeculator(DraftModelSpeculator):
    """Shared CUDA graph and decode loop for autoregressive speculators."""

    draft_seq_len_offset = 0

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.prefill_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None

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
    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Run the draft prefill and sample draft token zero."""

    @abstractmethod
    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
        num_speculative_steps: int,
    ) -> None:
        """Generate the current draft token and prepare the next decode step."""

    @abstractmethod
    def _prepare_decode_inputs(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
    ) -> None:
        """Prepare initial decode inputs for this speculator."""

    def _prefill_capture_inputs(
        self,
    ) -> tuple[InputBuffers, list[list[AttentionGroup]]]:
        return self.input_buffers, self.attn_groups

    def _configure_decode_metadata_update(self) -> None:
        if self.num_speculative_steps == 1:
            self.supports_decode_metadata_update = False
            return
        super()._configure_decode_metadata_update()

    def on_prefill_begin(self, num_reqs: int) -> None: ...

    def on_prefill_end(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_begin(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_end(self, num_reqs: int) -> None: ...

    @property
    def advance_draft_positions(self) -> bool:
        """Whether each draft step writes a new KV position."""
        return True

    def capture(self) -> None:
        logger.info("Capturing autoregressive speculator...")
        self.last_token_indices.zero_()
        self.idx_mapping.zero_()
        self.current_draft_step.zero_()

        assert self.prefill_cudagraph_manager is not None
        if self.prefill_cudagraph_manager.use_breakable_cg:
            self.prefill_cudagraph_manager.init_breakable_cg_runner(self.model)

        input_buffers, attn_groups = self._prefill_capture_inputs()
        self.on_prefill_begin(self.max_num_reqs)
        self.prefill_cudagraph_manager.capture(
            self._prefill,
            self.model_state,
            input_buffers,
            self.block_tables,
            attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing prefill CUDA graphs",
        )
        self.on_prefill_end(self.max_num_reqs)

        if self.num_speculative_steps == 1:
            return

        assert self.decode_cudagraph_manager is not None
        self.on_multi_step_decode_begin(self.max_num_reqs)
        decode_fn = (
            self._generate_fused_drafts
            if self.supports_decode_metadata_update
            else self._generate_draft
        )
        self.decode_cudagraph_manager.capture(
            decode_fn,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing decode CUDA graphs",
            # 看下这个地方
            specialize_spec_tokens=self.supports_decode_metadata_update
            and self.speculative_config.uses_dynamic_speculative_decoding(),
        )
        self.on_multi_step_decode_end(self.max_num_reqs)

    def _decode(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        num_reqs: int,
        skip_attn: bool,
        dp_sync_state: DPSyncState | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
        is_profile: bool,
        num_speculative_steps: int,
    ) -> None:
        decode_batch_sync, num_batch_tokens = (
            self._build_uniform_batch_dp_sync(dp_sync_state, num_reqs)
            if dp_sync_state is not None
            else (None, num_reqs)
        )
        decode_batch_desc, decode_batch_sync = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager,
            num_reqs,
            num_batch_tokens,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
            dp_sync_state=decode_batch_sync,
        )

        self._prepare_decode_inputs(
            input_batch, num_rejected, num_reqs, skip_attn, decode_batch_desc
        )

        if self.decode_cudagraph_manager is not None:
            decode_batch_desc = self.decode_cudagraph_manager.specialize_spec_tokens(
                decode_batch_desc, num_speculative_steps
            )
        num_tokens_across_dp = (
            decode_batch_sync.num_tokens_across_dp
            if decode_batch_sync is not None
            else None
        )

        self.on_multi_step_decode_begin(num_reqs)
        decode_fn = (
            self._fused_multi_step_decode
            if self.supports_decode_metadata_update
            else self._multi_step_decode
        )
        decode_fn(
            num_reqs,
            skip_attn,
            decode_batch_desc,
            num_tokens_across_dp,
            seq_lens_cpu_upper_bound,
            num_speculative_steps,
        )
        self.on_multi_step_decode_end(num_reqs)

    def _multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
        num_speculative_steps: int,
    ) -> None:
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, num_speculative_steps):
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
                attn_metadata = self._build_uniform_attn_metadata(
                    num_reqs=num_reqs,
                    batch_desc=batch_desc,
                    num_query_per_req=1,
                    seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                    step=step + self.draft_seq_len_offset,
                )

            self.current_draft_step.fill_(step)
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp,
                    batch_desc.cg_mode,
                    num_speculative_steps,
                )

    def _fused_multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
        num_speculative_steps: int,
    ) -> None:
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            for groups in self.attn_groups:
                for group in groups:
                    group.map_to_kernel_block_table(
                        self.block_tables.input_block_tables[group.kv_cache_group_id]
                    )
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            return

        attn_metadata = None
        slot_mappings_by_layer = None
        if not skip_attn:
            slot_mappings = self.block_tables.compute_slot_mappings(
                self.idx_mapping[:num_reqs],
                self.input_buffers.query_start_loc[: num_reqs + 1],
                self.input_buffers.positions[:num_reqs],
                batch_desc.num_tokens,
            )
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            attn_metadata = self._build_uniform_attn_metadata(
                batch_desc=batch_desc,
                num_reqs=num_reqs,
                num_query_per_req=1,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=1 + self.draft_seq_len_offset,
            )

        self._generate_fused_drafts(
            num_reqs,
            batch_desc.num_tokens,
            attn_metadata,
            slot_mappings_by_layer,
            num_tokens_across_dp,
            batch_desc.cg_mode,
            num_speculative_steps,
        )

    def _generate_fused_drafts(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        num_speculative_steps: int | None = None,
    ) -> None:
        if num_speculative_steps is None:
            num_speculative_steps = self.num_speculative_steps

        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]

        for step in range(1, num_speculative_steps):
            if attn_metadata is not None and (
                self.advance_draft_positions or step == 1
            ):
                self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    num_tokens_padded,
                )
                self._update_draft_decode_metadata(attn_metadata, num_reqs)

            self.current_draft_step.fill_(step)
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                num_speculative_steps,
            )
