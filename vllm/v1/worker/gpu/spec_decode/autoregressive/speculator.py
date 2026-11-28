# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from abc import abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.sample.sampler import Sampler

logger = init_logger(__name__)


@contextlib.contextmanager
def capture_while_body(pred: torch.Tensor) -> Iterator[None]:
    """Captures the enclosed work as the body of a CUDA graph WHILE node.

    The body runs while `pred` is true. `pred` is read before the first iteration
    and after every iteration, so the body must leave it updated.
    """
    graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    graph.begin_capture_to_while_node(pred)
    # PyTorch captures the body on a stream it makes current in C++, which the
    # current_stream() that vLLM caches does not follow.
    torch.cuda.set_stream(torch.cuda.current_stream())
    yield
    graph.set_conditional_handle_for_current_node(pred)
    graph.end_capture_to_conditional_node()
    torch.cuda.set_stream(torch.cuda.current_stream())


class AutoRegressiveSpeculator(DraftModelSpeculator):
    """Shared CUDA graph and decode loop for autoregressive speculators."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.prefill_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None

        # Draft early exit: the captured decode graph runs draft steps 1..k-1 in a
        # CUDA graph WHILE node, which stops once one more draft position is
        # expected to add fewer accepted tokens than it costs. Adaptive
        # verification sets the stopping threshold and its draft budget here every
        # step (see _draft_early_exit_kernel); until it first does, a zero
        # threshold drafts every position.
        self.draft_early_exit = bool(self.speculative_config.draft_early_exit)
        self.draft_stop_params = torch.tensor(
            [0.0, self.max_num_reqs * self.num_speculative_steps],
            dtype=torch.float32,
            device=device,
        )
        # Draft positions the last propose produced, and per request, those of the
        # propose that made its current drafts (see store_num_drafted).
        self.num_drafted = torch.zeros((), dtype=torch.int32, device=device)
        self.num_drafted_per_req = torch.full(
            (self.max_num_reqs,),
            self.num_speculative_steps,
            dtype=torch.int32,
            device=device,
        )
        self._draft_continue = torch.ones((), dtype=torch.bool, device=device)
        self._draft_positions = torch.arange(self.num_speculative_steps, device=device)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Subclasses create prefill_cudagraph_manager before calling this.
        # PIECEWISE cudagraphs are not supported for draft decodes.
        decode_mode = (
            CUDAGraphMode.FULL_DECODE_ONLY
            if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            else CUDAGraphMode.NONE
        )
        self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            decode_mode,
            decode_query_lens={1: self.max_num_reqs},
            decode_request_sizes=(
                self.vllm_config.get_draft_decode_cudagraph_request_sizes()
            ),
        )
        if self.draft_early_exit:
            self._check_draft_early_exit()

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
    def _decode(
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

    def _prefill_capture_slot_mappings(self) -> torch.Tensor | None:
        return None

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
        # As in dummy runs, -1 keeps draft steps from writing KV through a
        # request-state row's stale block ids.
        self.idx_mapping.fill_(-1)
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
            slot_mappings_buffer=self._prefill_capture_slot_mappings(),
        )
        self.on_prefill_end(self.max_num_reqs)

        if self.num_speculative_steps == 1:
            return

        assert self.decode_cudagraph_manager is not None
        self.on_multi_step_decode_begin(self.max_num_reqs)
        decode_fn = (
            self._fused_multi_step_decode
            if self.supports_decode_metadata_update
            else self._decode
        )
        with self._decode_capture_collectives():
            self.decode_cudagraph_manager.capture(
                decode_fn,
                self.model_state,
                self.input_buffers,
                self.block_tables,
                self.attn_groups,
                self.kv_cache_config,
                progress_bar_desc="Capturing decode CUDA graphs",
                specialize_spec_tokens=self.supports_decode_metadata_update
                and self.speculative_config.uses_dynamic_speculative_decoding(),
            )
        self.on_multi_step_decode_end(self.max_num_reqs)

    def _check_draft_early_exit(self) -> None:
        """The loop runs only in FULL CUDA graphs of the fused draft decode."""
        assert self.decode_cudagraph_manager is not None
        if not (
            self.supports_decode_metadata_update
            and self.decode_cudagraph_manager.needs_capture()
        ):
            raise ValueError(
                "draft_early_exit needs FULL CUDA graphs for draft decodes, with "
                "attention backends that update draft decode metadata in place."
            )
        if self.acceptance_estimator is None:
            raise ValueError(
                "draft_early_exit needs the acceptance estimator of adaptive "
                "verification to score each draft."
            )
        if not self.advance_draft_positions:
            raise ValueError(
                "draft_early_exit needs draft steps that advance their positions."
            )

    def _decode_capture_collectives(self) -> contextlib.AbstractContextManager:
        """With draft early exit, the decode graphs gather the draft logits through
        symmetric memory: a conditional node's body may hold no NCCL launch, since
        NCCL's default graph mixing adds event nodes."""
        tp_group = get_tp_group()
        if not self.draft_early_exit or tp_group.world_size == 1:
            return contextlib.nullcontext()
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )

        # The largest decode batch's logits span the LM head's padded vocabulary,
        # in the head_dtype of the config the drafter is built under.
        vocab_size = max(
            (
                module.num_embeddings_padded
                for module in self.model.modules()
                if isinstance(module, ParallelLMHead)
            ),
            default=self.vocab_size,
        )
        logits_bytes = self.vllm_config.model_config.head_dtype.itemsize
        communicator = tp_group.device_communicator
        assert isinstance(communicator, CudaCommunicator)
        return communicator.symm_mem_all_gather(
            self.max_num_reqs * vocab_size * logits_bytes
        )

    def _run_multi_step_decode(
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
        decode_batch_sync, num_batch_tokens = None, num_reqs
        if dp_sync_state is not None:
            # Padded dummy batches can exceed the requests the DP sync covers.
            num_reqs = min(num_reqs, dp_sync_state.num_reqs)
            decode_batch_sync, num_batch_tokens = self._build_uniform_batch_dp_sync(
                dp_sync_state, num_reqs
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
            parallel_config=self.vllm_config.parallel_config,
        )
        if self.decode_cudagraph_manager is not None:
            decode_batch_desc = self.decode_cudagraph_manager.specialize_spec_tokens(
                decode_batch_desc, num_speculative_steps
            )

        self._prepare_decode_inputs(
            input_batch, num_rejected, num_reqs, skip_attn, decode_batch_desc
        )

        num_tokens_across_dp = (
            decode_batch_sync.num_tokens_across_dp
            if decode_batch_sync is not None
            else None
        )

        if self.draft_early_exit:
            # A captured decode graph overwrites it when it stops early.
            self.num_drafted.fill_(num_speculative_steps)
        self.on_multi_step_decode_begin(num_reqs)
        multi_step_decode_fn = (
            self._run_fused_multi_step_decode
            if self.supports_decode_metadata_update
            else self._multi_step_decode
        )
        multi_step_decode_fn(
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
                    step=step,
                )

            self.current_draft_step.fill_(step)
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._decode(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp,
                    batch_desc.cg_mode,
                    num_speculative_steps,
                )

    def _run_fused_multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
        num_speculative_steps: int,
    ) -> None:
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            self._map_kernel_block_tables()
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
                step=1,
            )

        self._fused_multi_step_decode(
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
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
        num_speculative_steps: int,
    ) -> None:
        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]

        if self.draft_early_exit and torch.cuda.is_current_stream_capturing():
            self._capture_early_exit_decode(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                num_speculative_steps,
            )
            return

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
            self._decode(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                num_speculative_steps,
            )

    def _capture_early_exit_decode(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
        num_speculative_steps: int,
    ) -> None:
        """Captures draft steps 1..k-1 as one CUDA graph WHILE node.

        The body is one draft step for the whole batch. It reads the step index,
        positions and attention metadata from device buffers, so a single capture
        serves every iteration. On exit, num_drafted holds the number of draft
        positions this propose produced.
        """
        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]

        self.current_draft_step.fill_(1)
        self._decide_draft_continue(num_reqs, num_speculative_steps, advance=False)
        with capture_while_body(self._draft_continue):
            if attn_metadata is not None:
                self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    num_tokens_padded,
                )
                self._update_draft_decode_metadata(attn_metadata, num_reqs)
            self._decode(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                num_speculative_steps,
            )
            self._decide_draft_continue(num_reqs, num_speculative_steps, advance=True)

        self.num_drafted.copy_(self.current_draft_step)
        # Skipped positions still hold an earlier propose's confidences, and draft
        # tokens that may be -1 (a propose that drafted nothing, as with
        # skip_drafts). Verification treats their slots as placeholders, but the
        # target still reads their tokens, so they must be valid token ids.
        skipped = self._draft_positions >= self.current_draft_step
        self.draft_token_confidence_probs[:num_reqs].masked_fill_(skipped, 0.0)
        self.draft_tokens[:num_reqs].masked_fill_(skipped, 0)

    def _decide_draft_continue(
        self, num_reqs: int, num_speculative_steps: int, advance: bool
    ) -> None:
        """Sets _draft_continue on the GPU (see _draft_early_exit_kernel); with
        `advance`, first counts the draft step that just ran in current_draft_step.
        """
        block_r = triton.next_power_of_2(num_reqs)
        block_k = triton.next_power_of_2(num_speculative_steps)
        _draft_early_exit_kernel[(1,)](
            self._draft_continue,
            self.current_draft_step,
            self.draft_stop_params,
            self.draft_token_confidence_probs,
            self.draft_token_confidence_probs.stride(0),
            self.idx_mapping,
            num_reqs,
            num_speculative_steps,
            ADVANCE=advance,
            BLOCK_R=block_r,
            BLOCK_K=block_k,
            num_warps=8 if block_r * block_k >= 4096 else 4,
        )

    def store_num_drafted(
        self, idx_mapping: torch.Tensor, num_speculative_tokens: int
    ) -> None:
        """Stores the draft count of the propose that just ran with each request's
        drafts. Verification treats the draft slots past it as placeholders."""
        if num_speculative_tokens > 1:
            self.num_drafted_per_req[idx_mapping] = self.num_drafted
        else:
            # No multi-step decode ran.
            self.num_drafted_per_req[idx_mapping] = num_speculative_tokens

    def prepare_propose(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        sampler: "Sampler",
    ) -> None:
        if self.draft_early_exit and input_batch.num_draft_tokens > 0:
            # The acceptance estimator grades only the drafts this drafter made.
            num_rejected = rejections_of_drafted(
                num_rejected,
                input_batch.cu_num_logits,
                self.model_state.num_new_sampled_tokens_per_step,
                self.num_drafted_per_req[input_batch.idx_mapping],
            )
        super().prepare_propose(input_batch, num_sampled, num_rejected, sampler)


def rejections_of_drafted(
    num_rejected: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_bonus_tokens: int,
    req_num_drafted: torch.Tensor,
) -> torch.Tensor:
    """`num_rejected` without the placeholder slots verified past the drafts the
    drafter made. Verification stops at the first placeholder, so all of them
    count as rejected."""
    num_verified = cu_num_logits[1:] - cu_num_logits[:-1] - num_bonus_tokens
    num_skipped = (num_verified - req_num_drafted).clamp_min(0)
    return (num_rejected - num_skipped).clamp_min(0)


@triton.jit
def _draft_early_exit_kernel(
    continue_ptr,
    current_draft_step_ptr,
    # [2]: threshold, budget
    params_ptr,
    confidence_ptr,
    confidence_stride,
    idx_mapping_ptr,
    num_reqs,
    num_speculative_steps,
    ADVANCE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Whether to draft position j, with positions 0..j-1 drafted (j is the
    current draft step).

    A request's survival probability at j-1, the product of its confidences at
    0..j-1, times its confidence at j-1 predicts its survival at j. Adaptive
    verification admits the `budget` drafts with the highest survivals, so a
    draft at j adds only what its survival exceeds the edge by: the budget-th
    highest survival among the positions drafted so far, or 0 while there are
    fewer. Drafting j pays for itself if the batch's sum of these gains is at
    least `threshold`. A position must also be left, and a budget above 0.

    CUDA graph padding rows (idx_mapping < 0) have unwritten, possibly NaN,
    confidences and are selected away; a NaN from a real row keeps drafting.
    """
    step = tl.load(current_draft_step_ptr)
    if ADVANCE:
        step += 1
        tl.store(current_draft_step_ptr, step)
    threshold = tl.load(params_ptr)
    budget = tl.load(params_ptr + 1).to(tl.int32)

    rows = tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_K)
    is_real = tl.load(idx_mapping_ptr + rows, mask=rows < num_reqs, other=-1) >= 0
    drafted = is_real[:, None] & (cols[None, :] < step)
    confidences = tl.load(
        confidence_ptr + rows[:, None] * confidence_stride + cols[None, :],
        mask=drafted,
        other=1.0,
    )
    survival = tl.cumprod(confidences, axis=1)
    newest = cols[None, :] == step - 1
    predicted = tl.sum(tl.where(newest, survival * confidences, 0.0), axis=1)

    # Survival probabilities are non-negative, so their float bits order like
    # the values, as in adaptive verification's ranking; undrafted slots rank
    # last.
    bits = tl.where(drafted, survival.to(tl.int32, bitcast=True), -1)
    num_drafted = tl.sum(drafted.to(tl.int32))
    edge = 0.0
    if (budget > 0) & (budget <= num_drafted):
        # Binary search the largest survival that budget drafts reach.
        lo = 0
        hi = 0x3F800000  # 1.0
        while lo < hi:
            mid = lo + (hi - lo + 1) // 2
            if tl.sum((bits >= mid).to(tl.int32)) >= budget:
                lo = mid
            else:
                hi = mid - 1
        edge = tl.cast(lo, tl.float32, bitcast=True)
    over_edge = tl.maximum(predicted - edge, 0.0, propagate_nan=tl.PropagateNan.ALL)
    gain = tl.sum(tl.where(is_real, over_edge, 0.0))
    keep = (step < num_speculative_steps) & (budget > 0) & ~(gain < threshold)
    tl.store(continue_ptr, keep)
