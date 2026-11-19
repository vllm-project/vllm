# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph capture for the plain-path rejection sampler.

Replays the four rejection-sampling Triton kernels (local logits stats,
rejection, resample, insert-resampled) as a single CUDA graph per decode
step, replacing their per-kernel host dispatches with one replay.

The graph captures the kernels at the maximum verify-batch geometry
(``max_num_reqs x (num_speculative_steps + 1)`` logits rows). Per step, the
live tensors are copied into static input buffers whose padded tail is
filled so that padded requests become no-ops inside the graph:

- ``cu_num_logits`` repeats the total, so padded requests have empty row
  ranges;
- ``expanded_idx_mapping``/``idx_mapping`` point padded rows at state slot
  0 (valid memory, results discarded);
- ``expanded_local_pos`` is zeroed past the live width. The stats kernel
  skips a row when the next row starts a new request, so every padded row
  and the live row that closes the range take that early return, just as
  the masked lookahead does on the eager path.

Output buffers are graph-static: consumers must complete their reads
before the next replay overwrites them. Upstream consumes them through
``AsyncModelRunnerOutput``, which enqueues the D2H on a dedicated copy
stream ordered after production via ``wait_stream`` and blocks in
``get_output()`` on the copy event before the scheduler releases the next
step — the same contract already applied to the graph-static
``sampled_token_ids`` buffer.
"""

from types import SimpleNamespace

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

logger = init_logger(__name__)


# n_reqs/n_logits vary per step; their ==1/%16 specialization keys would
# recompile this graph-external launch on shape changes.
@triton.jit(do_not_specialize=["n_reqs", "n_logits"])
def _pack_rejection_sample_metadata_kernel(
    # [n_reqs + 1] -> [MAX_REQS + 1]
    cu_src_ptr,
    cu_dst_ptr,
    # [n_reqs] -> [MAX_REQS]
    idx_src_ptr,
    idx_dst_ptr,
    # [n_logits] -> [MAX_LOGITS]
    exp_idx_src_ptr,
    exp_idx_dst_ptr,
    # [n_logits] -> [MAX_LOGITS]
    elp_src_ptr,
    elp_dst_ptr,
    # [n_logits] -> [MAX_LOGITS]
    pos_src_ptr,
    pos_dst_ptr,
    # [n_logits] -> [MAX_LOGITS + 1]
    draft_sampled_src_ptr,
    draft_sampled_dst_ptr,
    # [MAX_REQS] -> [MAX_REQS]
    temp_src_ptr,
    temp_dst_ptr,
    # [MAX_REQS] -> [MAX_REQS]
    seed_src_ptr,
    seed_dst_ptr,
    n_reqs,
    n_logits,
    total_logits,
    MAX_REQS: tl.constexpr,
    MAX_LOGITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy the small rejection-sampling inputs into the static buffers.

    One launch replaces eight ``copy_``/``fill_`` dispatches. Each program
    handles one segment; lanes past the live region receive the padding
    value that turns padded requests into no-ops for the captured kernels.
    """
    seg = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    # Each segment keeps a dedicated result variable: variables assigned in
    # every branch of an if/elif chain must share one type, and the segments
    # mix int32/int64/fp32.
    if seg == 0:
        # Padded requests repeat the total, so cu_num_logits[r] ==
        # cu_num_logits[r + 1] and their token loops never execute.
        cu_vals = tl.load(cu_src_ptr + lanes, mask=lanes <= n_reqs, other=total_logits)
        tl.store(cu_dst_ptr + lanes, cu_vals, mask=lanes < MAX_REQS + 1)
    elif seg == 1:
        idx_vals = tl.load(idx_src_ptr + lanes, mask=lanes < n_reqs, other=0)
        tl.store(idx_dst_ptr + lanes, idx_vals, mask=lanes < MAX_REQS)
    elif seg == 2:
        exp_idx_vals = tl.load(exp_idx_src_ptr + lanes, mask=lanes < n_logits, other=0)
        tl.store(exp_idx_dst_ptr + lanes, exp_idx_vals, mask=lanes < MAX_LOGITS)
    elif seg == 3:
        # A row is skipped by the stats kernel when the next row starts a new
        # request. Zeroing the padded tail makes every padded row, and the
        # live row that closes the range, take that early return.
        elp_vals = tl.load(elp_src_ptr + lanes, mask=lanes < n_logits, other=0)
        tl.store(elp_dst_ptr + lanes, elp_vals, mask=lanes < MAX_LOGITS)
    elif seg == 4:
        pos_vals = tl.load(pos_src_ptr + lanes, mask=lanes < n_logits, other=0)
        tl.store(pos_dst_ptr + lanes, pos_vals, mask=lanes < MAX_LOGITS)
    elif seg == 5:
        # One extra slot: the finalize path reads draft_sampled[row + 1] for
        # the last live row, which falls outside the live width.
        ds_vals = tl.load(draft_sampled_src_ptr + lanes, mask=lanes < n_logits, other=0)
        tl.store(draft_sampled_dst_ptr + lanes, ds_vals, mask=lanes < MAX_LOGITS + 1)
    elif seg == 6:
        temp_vals = tl.load(temp_src_ptr + lanes, mask=lanes < MAX_REQS, other=0.0)
        tl.store(temp_dst_ptr + lanes, temp_vals, mask=lanes < MAX_REQS)
    else:
        seed_vals = tl.load(seed_src_ptr + lanes, mask=lanes < MAX_REQS, other=0)
        tl.store(seed_dst_ptr + lanes, seed_vals, mask=lanes < MAX_REQS)


class RejectionSamplerGraph:
    """Manages capture/replay of the plain-path rejection sampler kernels.

    Only the plain probabilistic path is captured; block verification,
    synthetic rates, FP64, watermarking, FLy, sharded draft caches and
    chunked verification keep the eager implementation. Capture failures
    permanently disable the graph, release the static buffers, and fall
    back to eager sampling.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._graph = None
        self._disabled = False
        self._max_reqs = 0
        self._max_logits = 0
        self._num_speculative_steps = 0
        self._has_draft = False
        self._vocab = 0
        self._target_dtype = None
        self._draft_shape = None
        self._pack_block = 0
        self._cu_buf = None
        self._idx_buf = None
        self._exp_idx_buf = None
        self._elp_buf = None
        self._pos_buf = None
        self._ds_buf = None
        self._temp_buf = None
        self._seed_buf = None
        self._target_buf = None
        self._draft_buf = None
        self._sampled_buf = None
        self._num_sampled_buf = None

    @staticmethod
    def _is_stream_capturing() -> bool:
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except AttributeError:
            return False
        except Exception:
            return False

    def can_use(
        self,
        target_logits: torch.Tensor,
        draft_logits_cache,
        cu_num_logits: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        num_speculative_steps: int,
    ) -> bool:
        if self._disabled:
            return False
        if self._is_stream_capturing():
            return False
        num_reqs = cu_num_logits.shape[0] - 1
        if num_reqs < 1:
            return False
        # A draft cache carrying a per-row mapping (sharded draft) is not
        # capturable; keep it on the eager path.
        if (
            draft_logits_cache is not None
            and draft_logits_cache.row_mapping is not None
        ):
            return False
        has_draft = draft_logits_cache is not None
        if self._graph is not None:
            draft_values = draft_logits_cache.values if has_draft else None
            return (
                num_speculative_steps == self._num_speculative_steps
                and has_draft == self._has_draft
                and target_logits.device == self._device
                and target_logits.dtype == self._target_dtype
                and target_logits.shape[-1] == self._vocab
                and num_reqs <= self._max_reqs
                and target_logits.shape[0] <= self._max_logits
                and temperature.shape[0] == self._max_reqs
                and seed.shape[0] == self._max_reqs
                and (not has_draft or tuple(draft_values.shape) == self._draft_shape)
            )
        return (
            target_logits.device.type == "cuda"
            and temperature.shape[0] == seed.shape[0]
        )

    def _fill_static_inputs(
        self,
        target_logits: torch.Tensor,
        draft_values: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
        pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
    ) -> None:
        n_reqs = cu_num_logits.shape[0] - 1
        n_logits = target_logits.shape[0]
        self._target_buf[:n_logits].copy_(target_logits, non_blocking=True)
        if self._has_draft:
            self._draft_buf.copy_(draft_values, non_blocking=True)
        _pack_rejection_sample_metadata_kernel[(8,)](
            cu_num_logits,
            self._cu_buf,
            idx_mapping,
            self._idx_buf,
            expanded_idx_mapping,
            self._exp_idx_buf,
            expanded_local_pos,
            self._elp_buf,
            pos,
            self._pos_buf,
            draft_sampled,
            self._ds_buf,
            temperature,
            self._temp_buf,
            seed,
            self._seed_buf,
            n_reqs,
            n_logits,
            n_logits,
            MAX_REQS=self._max_reqs,
            MAX_LOGITS=self._max_logits,
            BLOCK=self._pack_block,
        )

    def _capture(
        self,
        target_logits: torch.Tensor,
        draft_logits_cache,
        draft_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
        pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        num_speculative_steps: int,
    ) -> bool:
        try:
            device = target_logits.device
            max_reqs = temperature.shape[0]
            max_logits = max_reqs * (num_speculative_steps + 1)
            vocab = target_logits.shape[-1]
            has_draft = draft_logits_cache is not None
            self._max_reqs = max_reqs
            self._max_logits = max_logits
            self._num_speculative_steps = num_speculative_steps
            self._has_draft = has_draft
            self._vocab = vocab
            self._target_dtype = target_logits.dtype
            self._draft_shape = (
                tuple(draft_logits_cache.values.shape) if has_draft else None
            )
            self._pack_block = triton.next_power_of_2(max(max_logits + 1, max_reqs + 1))

            def zeros(shape, dtype):
                return torch.zeros(shape, dtype=dtype, device=device)

            self._cu_buf = zeros((max_reqs + 1,), cu_num_logits.dtype)
            self._idx_buf = zeros((max_reqs,), idx_mapping.dtype)
            self._exp_idx_buf = zeros((max_logits,), expanded_idx_mapping.dtype)
            self._elp_buf = zeros((max_logits,), expanded_local_pos.dtype)
            self._pos_buf = zeros((max_logits,), pos.dtype)
            self._ds_buf = zeros((max_logits + 1,), draft_sampled.dtype)
            self._temp_buf = zeros((max_reqs,), temperature.dtype)
            self._seed_buf = zeros((max_reqs,), seed.dtype)
            self._target_buf = zeros((max_logits, vocab), target_logits.dtype)
            if has_draft:
                self._draft_buf = zeros(
                    draft_logits_cache.values.shape, draft_logits_cache.values.dtype
                )
            self._sampled_buf = zeros(
                (max_reqs, num_speculative_steps + 1), torch.int64
            )
            self._num_sampled_buf = zeros((max_reqs,), torch.int32)
            static_draft_cache = (
                SimpleNamespace(values=self._draft_buf, row_mapping=None)
                if has_draft
                else None
            )
            static_args = (
                self._target_buf,
                static_draft_cache,
                self._ds_buf,
                self._cu_buf,
                self._pos_buf,
                self._idx_buf,
                self._exp_idx_buf,
                self._elp_buf,
                self._temp_buf,
                self._seed_buf,
                num_speculative_steps,
                None,
                False,
                False,
            )
            self._fill_static_inputs(
                target_logits,
                draft_logits_cache.values if has_draft else None,
                draft_sampled,
                cu_num_logits,
                pos,
                idx_mapping,
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                seed,
            )
            rejection_sample(
                *static_args,
                skip_masked_vocab_blocks=False,
                sampled=self._sampled_buf,
                num_sampled=self._num_sampled_buf,
            )
            torch.cuda.synchronize()
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                rejection_sample(
                    *static_args,
                    skip_masked_vocab_blocks=False,
                    sampled=self._sampled_buf,
                    num_sampled=self._num_sampled_buf,
                )
            logger.info(
                "Captured rejection sampler graph (num_reqs=%d, num_logits=%d, "
                "vocab=%d, draft_logits=%s)",
                max_reqs,
                max_logits,
                vocab,
                has_draft,
            )
            return True
        except Exception as e:
            logger.warning(
                "Rejection sampler graph capture failed; using eager "
                "rejection sampling: %s",
                e,
            )
            self._disabled = True
            self._graph = None
            self._cu_buf = None
            self._idx_buf = None
            self._exp_idx_buf = None
            self._elp_buf = None
            self._pos_buf = None
            self._ds_buf = None
            self._temp_buf = None
            self._seed_buf = None
            self._target_buf = None
            self._draft_buf = None
            self._sampled_buf = None
            self._num_sampled_buf = None
            return False

    def run(
        self,
        target_logits: torch.Tensor,
        draft_logits_cache,
        draft_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
        pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        num_speculative_steps: int,
    ):
        if self._graph is None:
            if not self._capture(
                target_logits,
                draft_logits_cache,
                draft_sampled,
                cu_num_logits,
                pos,
                idx_mapping,
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                seed,
                num_speculative_steps,
            ):
                # Capture failed: serve this step eagerly with the live
                # tensors.
                return rejection_sample(
                    target_logits,
                    draft_logits_cache,
                    draft_sampled,
                    cu_num_logits,
                    pos,
                    idx_mapping,
                    expanded_idx_mapping,
                    expanded_local_pos,
                    temperature,
                    seed,
                    num_speculative_steps,
                )
            # Capture succeeded: the warm-up run inside _capture already
            # produced this step's outputs into the static buffers.
            num_reqs = cu_num_logits.shape[0] - 1
            return self._sampled_buf[:num_reqs], self._num_sampled_buf[:num_reqs]
        # Existing graph: unconditionally refresh the static inputs and
        # replay.
        self._fill_static_inputs(
            target_logits,
            draft_logits_cache.values if draft_logits_cache is not None else None,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seed,
        )
        self._graph.replay()
        num_reqs = cu_num_logits.shape[0] - 1
        return self._sampled_buf[:num_reqs], self._num_sampled_buf[:num_reqs]
