# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbatching (DBO) helpers for the V2 GPU model runner."""

import threading
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    DPMetadata,
    ForwardContext,
    create_forward_context,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import current_stream
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    UBatchSlices,
    check_ubatch_thresholds,
    create_sm_control_context,
)
from vllm.v1.worker.ubatching import make_ubatch_contexts

logger = init_logger(__name__)


def slice_input_batch(
    input_batch: InputBatch,
    ubatch_slice: UBatchSlice,
    ubatch_idx: int,
    input_buffers: InputBuffers,
) -> InputBatch:
    """Build the sub-`InputBatch` a single microbatch runs on.

    The sub-batch covers the requests in `ubatch_slice.request_slice` and the
    tokens in `ubatch_slice.token_slice`. When a request straddles the boundary
    it appears in both microbatches, with its query truncated to the tokens each
    one owns.

    Everything except `query_start_loc` and `seq_lens` is a view of the buffers
    the full batch already uses, so slicing costs no allocation. Those two need
    to be rebased onto the microbatch's token range, so they are written into
    the per-microbatch buffers of `input_buffers`; all writes are plain tensor
    ops on persistent memory, which keeps the result usable under CUDA graph
    capture and replay.

    The sub-batch describes the forward pass only (attention metadata and model
    inputs). Sampling runs once over the merged batch, so `logits_indices`,
    `cu_num_logits` and the draft-token fields are carried over unchanged and
    must not be read off a sub-batch.
    """
    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"
    assert ubatch_idx < len(input_buffers.ubatch_query_start_loc), (
        f"No microbatch buffers for ubatch {ubatch_idx}; InputBuffers was "
        f"created with num_ubatches={input_buffers.num_ubatches}"
    )

    req_start = ubatch_slice.request_slice.start
    req_stop = ubatch_slice.request_slice.stop
    tok_start = ubatch_slice.token_slice.start
    tok_stop = ubatch_slice.token_slice.stop

    num_reqs_padded = req_stop - req_start
    num_tokens_padded = tok_stop - tok_start
    # The trailing microbatch may extend past the real batch into CUDA graph
    # padding; the unpadded counts stop at the last real request/token.
    num_reqs = max(0, min(req_stop, input_batch.num_reqs) - req_start)
    num_tokens = max(0, min(tok_stop, input_batch.num_tokens) - tok_start)

    query_start_loc = _slice_query_start_loc(
        input_batch.query_start_loc,
        input_buffers.ubatch_query_start_loc[ubatch_idx][: num_reqs_padded + 1],
        req_start,
        req_stop,
        tok_start,
        num_tokens_padded,
    )
    query_start_loc_np = np.clip(
        input_batch.query_start_loc_np[req_start : req_stop + 1] - tok_start,
        0,
        num_tokens_padded,
    ).astype(np.int32)

    seq_lens = _slice_seq_lens(
        input_batch.seq_lens,
        input_buffers.ubatch_seq_lens[ubatch_idx][:num_reqs_padded],
        input_batch.query_start_loc,
        query_start_loc,
        req_start,
        req_stop,
    )
    # Same truncation as `_slice_seq_lens`, on the host-side upper bound.
    seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound[
        req_start:req_stop
    ].clone()
    tokens_truncated = max(0, int(input_batch.query_start_loc_np[req_stop]) - tok_stop)
    if tokens_truncated:
        seq_lens_cpu_upper_bound[-1] -= tokens_truncated

    # Query lengths of the truncated requests, so consumers that derive
    # max_query_len from this array see the microbatch's own lengths.
    num_scheduled_tokens = np.diff(query_start_loc_np)[:num_reqs]

    dcp_local_seq_lens = input_batch.dcp_local_seq_lens
    if dcp_local_seq_lens is not None:
        # NOTE: a request split across microbatches keeps its full local
        # seq_len here. DCP is not yet validated with DBO.
        dcp_local_seq_lens = dcp_local_seq_lens[req_start:req_stop]

    return replace(
        input_batch,
        req_ids=input_batch.req_ids[req_start : min(req_stop, input_batch.num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs_padded,
        idx_mapping=input_batch.idx_mapping[req_start:req_stop],
        idx_mapping_np=input_batch.idx_mapping_np[req_start:req_stop],
        num_scheduled_tokens=num_scheduled_tokens,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens_padded,
        query_start_loc=query_start_loc,
        query_start_loc_np=query_start_loc_np,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        dcp_local_seq_lens=dcp_local_seq_lens,
        num_computed_tokens_np=input_batch.num_computed_tokens_np[req_start:req_stop],
        prefill_len_np=input_batch.prefill_len_np[req_start:req_stop],
        num_computed_prefill_tokens_np=input_batch.num_computed_prefill_tokens_np[
            req_start:req_stop
        ],
        is_prefilling_np=input_batch.is_prefilling_np[req_start:req_stop],
        max_seq_len_np=(
            None
            if input_batch.max_seq_len_np is None
            else input_batch.max_seq_len_np[req_start:req_stop]
        ),
        input_ids=input_batch.input_ids[tok_start:tok_stop],
        positions=input_batch.positions[tok_start:tok_stop],
        is_padding=input_batch.is_padding[tok_start:tok_stop],
        prompt_lens=(
            None
            if input_batch.prompt_lens is None
            else input_batch.prompt_lens[req_start:req_stop]
        ),
    )


def _slice_query_start_loc(
    query_start_loc: torch.Tensor,
    out: torch.Tensor,
    req_start: int,
    req_stop: int,
    tok_start: int,
    num_tokens_padded: int,
) -> torch.Tensor:
    """Rebase query_start_loc onto the microbatch's token range.

    Clamping to [0, num_tokens_padded] truncates the two requests that can
    straddle the microbatch boundary: the leading one loses the tokens the
    previous microbatch owns, the trailing one the tokens the next one owns.
    """
    torch.sub(query_start_loc[req_start : req_stop + 1], tok_start, out=out)
    return out.clamp_(0, num_tokens_padded)


def _slice_seq_lens(
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    query_start_loc: torch.Tensor,
    ubatch_query_start_loc: torch.Tensor,
    req_start: int,
    req_stop: int,
) -> torch.Tensor:
    """Slice seq_lens, shortening the request truncated at the boundary.

    A request continuing into the next microbatch has fewer of its tokens in
    this one, so its sequence ends earlier. The truncation is computed from the
    already-rebased query_start_loc rather than from Python ints, so the whole
    thing stays valid inside a CUDA graph.
    """
    out.copy_(seq_lens[req_start:req_stop])
    last = req_stop - req_start - 1
    query_len = query_start_loc[req_stop] - query_start_loc[req_stop - 1]
    ubatch_query_len = ubatch_query_start_loc[last + 1] - ubatch_query_start_loc[last]
    out[last] -= query_len - ubatch_query_len
    return out


def slice_model_inputs(
    model_inputs: dict[str, Any], token_slice: slice
) -> dict[str, Any]:
    """Narrow the model's per-token inputs to one microbatch."""
    sliced = dict(model_inputs)
    for key in ("input_ids", "inputs_embeds"):
        value = model_inputs.get(key)
        if value is not None:
            sliced[key] = value[token_slice]

    positions = model_inputs["positions"]
    # M-RoPE carries a leading section dim.
    sliced["positions"] = (
        positions[:, token_slice] if positions.ndim == 2 else positions[token_slice]
    )

    intermediate_tensors = model_inputs.get("intermediate_tensors")
    if intermediate_tensors is not None:
        sliced["intermediate_tensors"] = intermediate_tensors[token_slice]
    return sliced


def merge_ubatch_outputs(outputs: list[Any]) -> Any:
    """Reassemble the full-batch model output from the per-microbatch ones.

    Models return hidden states, a tuple of those plus auxiliary hidden states
    (EAGLE3), or IntermediateTensors on non-final pipeline ranks.
    """
    first = outputs[0]
    if isinstance(first, IntermediateTensors):
        return IntermediateTensors(
            {
                key: torch.cat([out.tensors[key] for out in outputs], dim=0)
                for key in first.tensors
            }
        )
    if isinstance(first, tuple):
        return tuple(torch.cat(parts, dim=0) for parts in zip(*outputs))
    return torch.cat(outputs, dim=0)


class UBatchRunner:
    """Runs the model on two or more microbatches that overlap each other.

    Each microbatch gets a thread and its own forward context. Only one thread
    holds the GPU at a time; they hand off to each other at the communication
    points inside the model (see `vllm.v1.worker.ubatching`), so one
    microbatch's expert all-to-all overlaps the other's compute.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config
        self.num_ubatches = self.parallel_config.num_ubatches
        self.device = device
        self.comm_stream = torch.cuda.Stream(device=device)
        # The microbatch threads plus the thread that starts them.
        self.ready_barrier = threading.Barrier(self.num_ubatches + 1)
        self.sm_control = create_sm_control_context(vllm_config)

    def wants_ubatch(self, num_tokens: int, uniform_decode: bool) -> bool:
        """Whether this rank would like to microbatch this step.

        The answer still has to be agreed with the other DP ranks before it can
        be acted on: microbatching is all-or-nothing across the group.
        """
        return check_ubatch_thresholds(
            self.parallel_config, num_tokens, uniform_decode=uniform_decode
        )

    def make_forward_contexts(
        self,
        attn_metadata: list[dict[str, Any]],
        slot_mappings_by_layer: list[dict[str, torch.Tensor]],
        ubatch_slices: UBatchSlices,
        is_padding: list[torch.Tensor | None],
    ) -> list[ForwardContext]:
        """Build one forward context per microbatch.

        Each context carries its own DP metadata: the expert all-to-all runs
        once per microbatch, so it has to see that microbatch's token count
        rather than the whole batch's. All DP ranks are padded to the same size
        when microbatching, which is what lets every rank assume the same token
        count for the other ranks' microbatches.
        """
        dp_size = self.parallel_config.data_parallel_size
        forward_contexts = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            num_tokens = ubatch_slice.num_tokens
            num_tokens_across_dp = torch.full(
                (dp_size,), num_tokens, dtype=torch.int32, device="cpu"
            )
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i],
                    self.vllm_config,
                    dp_metadata=DPMetadata.make(
                        self.parallel_config, num_tokens, num_tokens_across_dp
                    ),
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    slot_mapping=slot_mappings_by_layer[i],
                    is_padding=is_padding[i],
                )
            )
        return forward_contexts

    def run(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        forward_contexts: list[ForwardContext],
        ubatch_slices: UBatchSlices,
    ) -> Any:
        assert len(forward_contexts) == len(ubatch_slices) == self.num_ubatches

        ubatch_contexts = make_ubatch_contexts(
            num_micro_batches=self.num_ubatches,
            comm_stream=self.comm_stream,
            compute_stream=current_stream(),
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        outputs: dict[int, Any] = {}
        errors: dict[int, BaseException] = {}

        @torch.inference_mode()
        def run_ubatch(ubatch_context, inputs: dict[str, Any]) -> None:
            try:
                with ubatch_context:
                    outputs[ubatch_context.id] = model(**inputs)
            except BaseException as e:  # noqa: BLE001
                # Recorded so the join below can report which microbatch broke.
                # Note the siblings are not told to unwind: microbatches hand
                # control to each other, so one that dies mid-forward leaves
                # the others parked on a handoff that never comes and the step
                # hangs. The V1 runner behaves the same way; fixing it means
                # changing the shared handoff protocol in ubatching.py, which
                # is deliberately out of scope here.
                errors[ubatch_context.id] = e

        # The threads manage the forward context themselves; clear it here so
        # it is restored correctly once they are done.
        with override_forward_context(None), self.sm_control:
            threads = []
            for ubatch_context, ubatch_slice in zip(ubatch_contexts, ubatch_slices):
                thread = threading.Thread(
                    target=run_ubatch,
                    args=(
                        ubatch_context,
                        slice_model_inputs(model_inputs, ubatch_slice.token_slice),
                    ),
                )
                threads.append(thread)
                thread.start()

            # Wait for every thread to reach its context, then start the first.
            self.ready_barrier.wait()
            ubatch_contexts[0].cpu_wait_event.set()
            for thread in threads:
                thread.join()

        if errors:
            failed = min(errors)
            raise RuntimeError(
                f"Microbatch {failed} of {self.num_ubatches} failed"
            ) from errors[failed]
        return merge_ubatch_outputs([outputs[i] for i in range(self.num_ubatches)])
