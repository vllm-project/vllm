# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from tqdm import tqdm

from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture, is_global_first_rank
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


@triton.jit(
    do_not_specialize=[
        "num_reqs",
        "num_tokens",
        "num_reqs_padded",
        "num_tokens_after_padding",
    ]
)
def _copy_inputs_with_padding_kernel(
    # Input buffers.
    src_idx_mapping_ptr,
    src_cu_num_logits_ptr,
    src_logits_indices_ptr,
    src_expanded_idx_mapping_ptr,
    src_expanded_local_pos_ptr,
    # Destination buffers.
    dst_idx_mapping_ptr,
    dst_cu_num_logits_ptr,
    dst_logits_indices_ptr,
    dst_expanded_idx_mapping_ptr,
    dst_expanded_local_pos_ptr,
    # Actual sizes.
    num_reqs,
    num_tokens,
    # Padded sizes.
    num_reqs_padded,
    num_tokens_after_padding,
    # Padding value for expanded_local_pos.
    expanded_local_pos_pad_value,
):
    pid = tl.program_id(0)

    # Copy idx_mapping values. Set padded values to 0.
    if pid < num_reqs:
        tl.store(
            dst_idx_mapping_ptr + pid,
            tl.load(src_idx_mapping_ptr + pid),
        )
    elif pid < num_reqs_padded:
        tl.store(dst_idx_mapping_ptr + pid, 0)

    # Copy cu_num_logits values. Set padding values to the last real value.
    if pid < num_reqs + 1:
        tl.store(
            dst_cu_num_logits_ptr + pid,
            tl.load(src_cu_num_logits_ptr + pid),
        )
    elif pid < num_reqs_padded + 1:
        tl.store(
            dst_cu_num_logits_ptr + pid,
            tl.load(src_cu_num_logits_ptr + num_reqs),
        )

    # Copy logits_indices, expanded_idx_mapping, and expanded_local_pos values.
    # Set padded values to 0.
    if pid < num_tokens:
        tl.store(
            dst_logits_indices_ptr + pid,
            tl.load(src_logits_indices_ptr + pid),
        )
        tl.store(
            dst_expanded_idx_mapping_ptr + pid,
            tl.load(src_expanded_idx_mapping_ptr + pid),
        )
        tl.store(
            dst_expanded_local_pos_ptr + pid,
            tl.load(src_expanded_local_pos_ptr + pid),
        )
    elif pid < num_tokens_after_padding:
        tl.store(dst_logits_indices_ptr + pid, 0)
        tl.store(dst_expanded_idx_mapping_ptr + pid, 0)
        tl.store(dst_expanded_local_pos_ptr + pid, expanded_local_pos_pad_value)


def _copy_inputs_with_padding(
    src_idx_mapping: torch.Tensor,
    src_cu_num_logits: torch.Tensor,
    src_logits_indices: torch.Tensor,
    src_expanded_idx_mapping: torch.Tensor,
    src_expanded_local_pos: torch.Tensor,
    dst_idx_mapping: torch.Tensor,
    dst_cu_num_logits: torch.Tensor,
    dst_logits_indices: torch.Tensor,
    dst_expanded_idx_mapping: torch.Tensor,
    dst_expanded_local_pos: torch.Tensor,
    num_reqs: int,
    num_tokens: int,
    num_reqs_padded: int,
    num_tokens_after_padding: int,
    num_speculative_steps: int,
) -> None:
    grid = (max(num_reqs_padded + 1, num_tokens_after_padding),)
    _copy_inputs_with_padding_kernel[grid](
        src_idx_mapping,
        src_cu_num_logits,
        src_logits_indices,
        src_expanded_idx_mapping,
        src_expanded_local_pos,
        dst_idx_mapping,
        dst_cu_num_logits,
        dst_logits_indices,
        dst_expanded_idx_mapping,
        dst_expanded_local_pos,
        num_reqs,
        num_tokens,
        num_reqs_padded,
        num_tokens_after_padding,
        num_speculative_steps,
    )


class RejectionSamplerCudaGraphManager:
    """CUDA graph manager for the rejection-sampler chain.

    Captures `compute_logits` → sampling-params kernels (unconditional) →
    `rejection_sample` → `get_num_sampled_and_rejected` as a single FULL
    graph, per `BatchExecutionDescriptor`.

    Only captures for **uniform spec-decode** batch descriptors
    (`uniform_token_count == num_speculative_tokens + 1`), since the
    rejection-sample kernel grids depend on `num_logits` and the uniform
    case has a fixed, predictable shape. Non-uniform batches fall back to
    the eager path in `GPUModelRunner.sample`.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        max_num_reqs: int,
        num_speculative_steps: int,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_reqs = max_num_reqs
        self.num_speculative_steps = num_speculative_steps
        self.decode_query_len = num_speculative_steps + 1
        self.pool = current_platform.get_global_graph_pool()

        # Static input buffers, sized to the maxima. Host code copies into
        # these before each replay so the captured graph reads from stable
        # GPU addresses.
        max_num_logits = max_num_reqs * self.decode_query_len
        self.idx_mapping = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.cu_num_logits = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.logits_indices = torch.zeros(
            max_num_logits, dtype=torch.int64, device=device
        )
        self.expanded_idx_mapping = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )
        self.expanded_local_pos = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )

        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[
            BatchExecutionDescriptor,
            tuple[SamplerOutput, torch.Tensor, torch.Tensor],
        ] = {}

    def has_graph(self, desc: BatchExecutionDescriptor) -> bool:
        return desc in self.graphs

    def _is_capturable_desc(self, desc: BatchExecutionDescriptor) -> bool:
        return (
            desc.cg_mode == CUDAGraphMode.FULL
            and desc.uniform_token_count == self.decode_query_len
            and desc.num_reqs is not None
        )

    @torch.inference_mode()
    def capture(
        self,
        descs: list[BatchExecutionDescriptor],
        compute_logits_fn: Callable[[torch.Tensor], tuple[torch.Tensor, int | None]],
        rejection_sampler: RejectionSampler,
        hidden_states: torch.Tensor,
        draft_logits: torch.Tensor | None,
        input_buffers,
        req_states: RequestState,
        progress_bar_desc: str = "Capturing rejection sampler CUDA graphs",
    ) -> None:
        """Capture rejection-sampler cudagraphs for each capturable descriptor."""
        capturable = [d for d in descs if self._is_capturable_desc(d)]
        if not capturable:
            return

        # Capture in decreasing num_tokens order so the largest activations
        # are allocated first (same convention used by ModelCudaGraphManager).
        capturable.sort(key=lambda d: d.num_tokens, reverse=True)

        iterator = capturable
        if is_global_first_rank():
            iterator = tqdm(capturable, desc=progress_bar_desc)

        with graph_capture(device=self.device):
            for desc in iterator:
                num_reqs = desc.num_reqs
                assert num_reqs is not None
                num_tokens = desc.num_tokens

                self._init_persistent_buffers_for_capture(num_reqs, num_tokens)

                # Warmup the _copy_inputs_with_padding kernel.
                _copy_inputs_with_padding(
                    self.idx_mapping,
                    self.cu_num_logits,
                    self.logits_indices,
                    self.expanded_idx_mapping,
                    self.expanded_local_pos,
                    self.idx_mapping,
                    self.cu_num_logits,
                    self.logits_indices,
                    self.expanded_idx_mapping,
                    self.expanded_local_pos,
                    num_reqs,
                    num_tokens,
                    num_reqs,
                    num_tokens,
                    self.num_speculative_steps,
                )

                # Warmup the _sample operation.
                self._sample(
                    num_reqs=num_reqs,
                    num_tokens=num_tokens,
                    hidden_states=hidden_states,
                    compute_logits_fn=compute_logits_fn,
                    rejection_sampler=rejection_sampler,
                    draft_logits=draft_logits,
                    input_buffers=input_buffers,
                    req_states=req_states,
                )

                # Capture.
                graph = torch.cuda.CUDAGraph()
                get_offloader().sync_prev_onload()
                with torch.cuda.graph(graph, self.pool):
                    sampler_output, num_sampled, num_rejected = self._sample(
                        num_reqs=num_reqs,
                        num_tokens=num_tokens,
                        hidden_states=hidden_states,
                        compute_logits_fn=compute_logits_fn,
                        rejection_sampler=rejection_sampler,
                        draft_logits=draft_logits,
                        input_buffers=input_buffers,
                        req_states=req_states,
                    )
                self.graphs[desc] = graph
                self.outputs[desc] = (sampler_output, num_sampled, num_rejected)
                compilation_counter.num_cudagraph_captured += 1

        logger.info("Captured %d rejection-sampler cudagraphs.", len(self.graphs))

    def _init_persistent_buffers_for_capture(
        self, num_reqs: int, num_tokens: int
    ) -> None:
        """Initialize persistent buffers with a valid uniform-spec-decode pattern.

        Values must not cause the kernels to crash during capture/warmup.
        For uniform spec decode: each request has exactly `decode_query_len`
        logits. cu_num_logits = [0, K, 2K, ...]. idx_mapping is the identity.
        """
        K = self.decode_query_len
        assert num_tokens == num_reqs * K

        device = self.device
        self.idx_mapping[:num_reqs].copy_(
            torch.arange(num_reqs, dtype=torch.int32, device=device)
        )
        self.cu_num_logits[: num_reqs + 1].copy_(
            torch.arange(num_reqs + 1, dtype=torch.int32, device=device) * K
        )
        arange_tokens = torch.arange(num_tokens, dtype=torch.int32, device=device)
        self.logits_indices[:num_tokens].copy_(
            torch.arange(num_tokens, dtype=torch.int64, device=device)
        )
        self.expanded_idx_mapping[:num_tokens].copy_(arange_tokens // K)
        self.expanded_local_pos[:num_tokens].copy_(arange_tokens % K)

    def _sample(
        self,
        num_reqs: int,
        num_tokens: int,
        rejection_sampler: RejectionSampler,
        compute_logits_fn: Callable[[torch.Tensor], tuple[torch.Tensor, int | None]],
        hidden_states: torch.Tensor,
        draft_logits: torch.Tensor,
        input_buffers,
        req_states: RequestState,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        idx_mapping = self.idx_mapping[:num_reqs]
        cu_num_logits = self.cu_num_logits[: num_reqs + 1]
        logits_indices = self.logits_indices[:num_tokens]
        expanded_idx_mapping = self.expanded_idx_mapping[:num_tokens]
        expanded_local_pos = self.expanded_local_pos[:num_tokens]

        sample_hidden_states = hidden_states[logits_indices]
        logits, shard_vocab_start = compute_logits_fn(sample_hidden_states)

        sampler_output = rejection_sampler.forward_fast(
            logits,
            input_buffers.input_ids,
            input_buffers.positions,
            cu_num_logits,
            idx_mapping,
            logits_indices,
            expanded_idx_mapping,
            expanded_local_pos,
            draft_logits,
            shard_vocab_start,
        )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            sampler_output.num_sampled,
            input_buffers.seq_lens[:num_reqs],
            cu_num_logits,
            idx_mapping,
            req_states.prefill_len.gpu,
        )
        return (
            sampler_output,
            num_sampled,
            num_rejected,
        )

    @torch.inference_mode()
    def run(
        self,
        desc: BatchExecutionDescriptor,
        input_batch: InputBatch,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        assert desc in self.graphs, (
            f"No rejection sampler graph exists for descriptor: {desc}"
        )

        num_reqs = input_batch.num_reqs
        num_reqs_padded = input_batch.num_reqs_after_padding
        num_tokens = input_batch.num_tokens
        num_tokens_after_padding = input_batch.num_tokens_after_padding

        # Copy this step's inputs into the persistent buffers, and fill the
        # padded slots with kernel-friendly sentinel values.
        _copy_inputs_with_padding(
            input_batch.idx_mapping,
            input_batch.cu_num_logits,
            input_batch.logits_indices,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.idx_mapping,
            self.cu_num_logits,
            self.logits_indices,
            self.expanded_idx_mapping,
            self.expanded_local_pos,
            num_reqs,
            num_tokens,
            num_reqs_padded,
            num_tokens_after_padding,
            self.num_speculative_steps,
        )

        get_offloader().sync_prev_onload()
        self.graphs[desc].replay()

        # Remove padding from outputs.
        padded_sampler_output, num_sampled, num_rejected = self.outputs[desc]
        sampler_output = SamplerOutput(
            sampled_token_ids=padded_sampler_output.sampled_token_ids[:num_reqs],
            # Output logprobs are not supported during cudagraph capture.
            logprobs_tensors=None,
            num_nans=padded_sampler_output.num_nans[:num_reqs]
            if padded_sampler_output.num_nans is not None
            else None,
            num_sampled=padded_sampler_output.num_sampled[:num_reqs]
            if padded_sampler_output.num_sampled is not None
            else None,
        )
        return sampler_output, num_sampled[:num_reqs], num_rejected[:num_reqs]
