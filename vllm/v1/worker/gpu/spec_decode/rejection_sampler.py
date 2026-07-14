# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.config import SpeculativeConfig
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

# Cap on the fp32 scratch that apply_sampling_params materializes.
# Spec decode verifies num_reqs * (num_speculative_steps + 1) logits, so chunking
# lets us cap the peak memory usage.
# TODO(mgoin): Chunking is a workaround. The rejection kernels already upcast
# per vocab block on load and apply ops like temperature and gumbel, so folding
# sampling-param application into those kernels would remove this buffer and
# its traffic entirely.
MAX_CHUNK_BYTES = 512 * 1024 * 1024


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ):
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True

    def _get_logprobs_tensors(
        self,
        input_batch: InputBatch,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
    ) -> LogprobsTensors | None:
        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = input_batch.cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            input_batch.cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != input_batch.idx_mapping.shape[0]
        return compute_topk_logprobs(
            logits,
            max_num_logprobs,
            flat_sampled,
            input_batch.cu_num_logits_np.tolist() if expanded_logits else None,
        )

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
        )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
        )
        return processed_logits, sampled, num_sampled

    def _verify_chunked(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        max_chunk_logits: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Requests are independent, so verify them in request-aligned chunks. Each
        # chunk's fp32 scratch is freed before the next one is allocated, capping
        # the peak at max_chunk_logits rows instead of the whole batch.
        cu_num_logits_np = input_batch.cu_num_logits_np
        reqs_per_chunk = max(1, max_chunk_logits // (self.num_speculative_steps + 1))

        sampled_chunks = []
        num_sampled_chunks = []
        for start in range(0, input_batch.num_reqs, reqs_per_chunk):
            end = min(start + reqs_per_chunk, input_batch.num_reqs)
            lo = int(cu_num_logits_np[start])
            hi = int(cu_num_logits_np[end])
            _, sampled, num_sampled = self._verify(
                logits[lo:hi],
                draft_logits,
                draft_sampled[lo:hi],
                pos[lo:hi],
                input_batch.cu_num_logits[start : end + 1] - lo,
                input_batch.idx_mapping[start:end],
                input_batch.idx_mapping_np[start:end],
                input_batch.expanded_idx_mapping[lo:hi],
                input_batch.expanded_local_pos[lo:hi],
            )
            sampled_chunks.append(sampled)
            num_sampled_chunks.append(num_sampled)
        return torch.cat(sampled_chunks), torch.cat(num_sampled_chunks)

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]

        # Only processed_logprobs needs the processed logits for the whole batch;
        # raw_logprobs reads the (untouched) input logits, which chunking preserves.
        need_processed_logits = (
            self.sampler.logprobs_mode == "processed_logprobs"
            and self.sampler.sampling_states.max_num_logprobs(
                input_batch.idx_mapping_np
            )
            != NO_LOGPROBS
        )
        max_chunk_logits = max(1, MAX_CHUNK_BYTES // (logits.shape[1] * 4))
        if need_processed_logits or logits.shape[0] <= max_chunk_logits:
            processed_logits, sampled, num_sampled = self._verify(
                logits,
                draft_logits,
                draft_sampled,
                pos,
                input_batch.cu_num_logits,
                input_batch.idx_mapping,
                input_batch.idx_mapping_np,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
            )
        else:
            processed_logits = None
            sampled, num_sampled = self._verify_chunked(
                logits, input_batch, draft_logits, draft_sampled, pos, max_chunk_logits
            )

        logprobs_tensors = self._get_logprobs_tensors(
            input_batch,
            sampled,
            num_sampled,
            processed_logits
            if self.sampler.logprobs_mode == "processed_logprobs"
            else logits,
        )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
