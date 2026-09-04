# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbatching (DBO) helpers for the V2 GPU model runner."""

import threading
from dataclasses import replace
from typing import Any, NamedTuple

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
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    UBatchSlices,
    create_sm_control_context,
    maybe_create_ubatch_slices,
)
from vllm.v1.worker.ubatching import make_ubatch_contexts
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class UBatchState(NamedTuple):
    """Per-microbatch inputs for one dual-batch-overlap step."""

    slices: UBatchSlices
    forward_contexts: list[ForwardContext]


def create_ubatch_slices(input_batch: InputBatch, num_ubatches: int) -> UBatchSlices:
    """Split a DP-padded batch into the slices its microbatches run on.

    Splitting at the midpoint of the DP-padded token count leaves the trailing
    microbatch anywhere from full to entirely inside the padding. Clamping its
    request slice onto the last request leaves it holding that request with
    zero query tokens -- the shape a straddling request already takes -- so it
    stays well-formed with no work to do, like a dummy run.
    """
    _, ubatch_slices = maybe_create_ubatch_slices(
        True,
        input_batch.num_scheduled_tokens,
        input_batch.num_tokens_after_padding,
        input_batch.num_reqs_after_padding,
        num_ubatches,
    )
    assert ubatch_slices is not None
    return [
        UBatchSlice(
            slice(
                min(s.request_slice.start, input_batch.num_reqs - 1),
                min(s.request_slice.stop, input_batch.num_reqs_after_padding),
            ),
            s.token_slice,
        )
        for s in ubatch_slices
    ]


def _slice_input_batch(
    input_batch: InputBatch,
    ubatch_slice: UBatchSlice,
    query_start_loc_buf: torch.Tensor,
    seq_lens_buf: torch.Tensor,
) -> InputBatch:
    """Build the sub-`InputBatch` a single microbatch runs on.

    A request straddling the boundary appears in both microbatches, its query
    truncated to the tokens each one owns.

    Everything except `query_start_loc` and `seq_lens` is a view of the full
    batch's buffers, so slicing costs no allocation. Those two are rebased onto
    the microbatch's token range and written to the caller's buffers, which
    keeps them valid under CUDA graph capture and replay.

    The sub-batch describes the forward pass only. Sampling runs once over the
    merged batch, so `logits_indices`, `cu_num_logits` and the draft-token
    fields are carried over unchanged and must not be read off a sub-batch.
    """
    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

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
        query_start_loc_buf[: num_reqs_padded + 1],
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
        seq_lens_buf[:num_reqs_padded],
        input_batch.query_start_loc,
        req_start,
        req_stop,
        tok_stop,
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

    The clamp truncates the requests straddling either boundary to the tokens
    this microbatch owns.
    """
    torch.sub(query_start_loc[req_start : req_stop + 1], tok_start, out=out)
    return out.clamp_(0, num_tokens_padded)


def _slice_seq_lens(
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    query_start_loc: torch.Tensor,
    req_start: int,
    req_stop: int,
    tok_stop: int,
) -> torch.Tensor:
    """Slice seq_lens, shortening the request truncated at the boundary.

    Only tokens past the end of this microbatch are missing; tokens it owns in
    an *earlier* microbatch are computed before this one reads them, so they
    still count. Deriving the truncation from this microbatch's own query
    lengths would subtract them twice. It is computed from tensors, not Python
    ints, so this stays valid inside a CUDA graph.
    """
    out.copy_(seq_lens[req_start:req_stop])
    last = req_stop - req_start - 1
    out[last] -= (query_start_loc[req_stop] - tok_stop).clamp_(min=0)
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

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        model_state: ModelState,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_num_reqs: int,
    ):
        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config
        self.num_ubatches = self.parallel_config.num_ubatches
        self.device = device
        self.model_state = model_state
        self.attn_groups = attn_groups
        self.kv_cache_config = kv_cache_config
        # `query_start_loc` and `seq_lens` are rebased onto each microbatch's
        # own token range, so they cannot be views of the full batch's buffers.
        # Allocating up front keeps their addresses stable across replays.
        self.ubatch_query_start_loc = [
            torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
            for _ in range(self.num_ubatches)
        ]
        self.ubatch_seq_lens = [
            torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
            for _ in range(self.num_ubatches)
        ]
        self.comm_stream = torch.cuda.Stream(device=device)
        # The microbatch threads plus the thread that starts them.
        self.ready_barrier = threading.Barrier(self.num_ubatches + 1)
        self.sm_control = create_sm_control_context(self.parallel_config)

    def prepare(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
    ) -> UBatchState:
        """Split the batch into the microbatches the step will run on.

        Attention metadata is built per microbatch and carried in the forward
        contexts the threads install, not in the caller's.
        """
        ubatch_slices = create_ubatch_slices(input_batch, self.num_ubatches)

        attn_metadata = []
        slot_mappings_by_layer = []
        is_padding = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            ubatch = _slice_input_batch(
                input_batch,
                ubatch_slice,
                self.ubatch_query_start_loc[i],
                self.ubatch_seq_lens[i],
            )
            ubatch_slot_mappings = slot_mappings[:, ubatch_slice.token_slice]
            ubatch_block_tables = tuple(
                block_table[ubatch_slice.request_slice] for block_table in block_tables
            )
            attn_metadata.append(
                self.model_state.prepare_attn(
                    ubatch,
                    CUDAGraphMode.NONE,
                    ubatch_block_tables,
                    ubatch_slot_mappings,
                    self.attn_groups,
                    self.kv_cache_config,
                    ubatch_idx=i,
                )
            )
            slot_mappings_by_layer.append(
                build_slot_mappings_by_layer(ubatch_slot_mappings, self.kv_cache_config)
            )
            is_padding.append(ubatch.is_padding)

        return UBatchState(
            slices=ubatch_slices,
            forward_contexts=self._make_forward_contexts(
                ubatch_slices, attn_metadata, slot_mappings_by_layer, is_padding
            ),
        )

    def _make_forward_contexts(
        self,
        ubatch_slices: UBatchSlices,
        attn_metadata: list[dict[str, Any]],
        slot_mappings_by_layer: list[dict[str, torch.Tensor]],
        is_padding: list[torch.Tensor | None],
    ) -> list[ForwardContext]:
        """Build one forward context per microbatch.

        The expert all-to-all runs once per microbatch, so each context's DP
        metadata carries that microbatch's token count. Microbatching pads every
        DP rank alike, so each rank can assume that count for the others too.
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
        ubatch_state: UBatchState,
    ) -> Any:
        ubatch_slices = ubatch_state.slices
        assert len(ubatch_slices) == len(ubatch_state.forward_contexts)
        assert len(ubatch_slices) == self.num_ubatches

        ubatch_contexts = make_ubatch_contexts(
            num_micro_batches=self.num_ubatches,
            comm_stream=self.comm_stream,
            compute_stream=current_stream(),
            forward_contexts=ubatch_state.forward_contexts,
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


def maybe_build_ubatch_runner(
    vllm_config: VllmConfig,
    device: torch.device,
    model_state: ModelState,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_num_reqs: int,
) -> UBatchRunner | None:
    """Build the microbatch runner, or None when DBO is not in use.

    Microbatching needs the DP handshake to agree on it, so it is only
    available with more than one DP rank (as in the V1 runner).
    """
    parallel_config = vllm_config.parallel_config
    if not parallel_config.use_ubatching or parallel_config.data_parallel_size <= 1:
        return None

    logger.info_once(
        "Dual batch overlap is enabled. Microbatched steps run without "
        "CUDA graphs on the V2 model runner."
    )
    return UBatchRunner(
        vllm_config, device, model_state, attn_groups, kv_cache_config, max_num_reqs
    )
