# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The levers on vLLM's GPU model runner.

``VLLM_K31_BEAT_EAGERPREP``: when the step's logits_indices is the identity
(every scheduled token is a logits row, e.g. a pure verify batch), the sampler
takes a view of the leading hidden-state rows instead of a gather launch.
``prepare_inputs`` marks the batch, ``sample`` reads the mark.

``VLLM_K31_BEAT_DRAFTMETA``: ``execute_model`` marks this thread as in the
target step, the one ``prepare_attn`` call that skips the draft-only attention
groups (``KimiK3ModelState.target_attn_groups``); CUDA-graph capture and the
drafter's own calls run outside it."""

import contextlib
import functools
import threading

import numpy as np
import torch

from vllm.models.kimi_k3.amd.specialized_decode.levers import (
    DRAFTMETA,
    EAGERPREP,
    lever_on,
    wrap,
)

# The InputBatch attribute ``prepare_inputs`` sets.
IDENTITY = "sd_logits_indices_is_identity"

# Whether this thread runs ``GPUModelRunner.execute_model``.
_local = threading.local()


def install() -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    wrap(GPUModelRunner, "execute_model", _wrap_execute_model)
    wrap(GPUModelRunner, "prepare_inputs", _wrap_prepare_inputs)
    wrap(GPUModelRunner, "sample", _wrap_sample)


def in_target_step() -> bool:
    return getattr(_local, "target_step", False)


@contextlib.contextmanager
def target_step():
    """Marks this thread as in ``execute_model``."""
    prev = in_target_step()
    _local.target_step = True
    try:
        yield
    finally:
        _local.target_step = prev


def _wrap_execute_model(stock):
    @functools.wraps(stock)
    def execute_model(self, *args, **kwargs):
        if not lever_on(DRAFTMETA):
            return stock(self, *args, **kwargs)
        with target_step():
            return stock(self, *args, **kwargs)

    return execute_model


def logits_indices_is_identity(
    cu_num_logits_np: np.ndarray, query_start_loc_np: np.ndarray, num_reqs: int
) -> bool:
    """Whether ``combine_sampled_and_draft_tokens``'s logits_indices is the
    identity. It puts request i's num_logits rows at [query_end - num_logits,
    query_end) from cu_num_logits[i]; when both cumsums coincide, every request's
    num_logits equals its scheduled tokens and row j is token j."""
    n = num_reqs + 1
    return bool(np.array_equal(cu_num_logits_np[:n], query_start_loc_np[:n]))


def select_logits_rows(
    hidden_states: torch.Tensor, logits_indices: torch.Tensor, is_identity: bool
) -> torch.Tensor:
    """``hidden_states[logits_indices]``; a view of the leading rows (no
    gather launch) when logits_indices is the identity."""
    if is_identity:
        return hidden_states[: logits_indices.shape[0]]
    return hidden_states[logits_indices]


def _wrap_prepare_inputs(stock):
    @functools.wraps(stock)
    def prepare_inputs(
        self, scheduler_output, batch_req_state, batch_desc, num_active_loras
    ):
        input_batch = stock(
            self, scheduler_output, batch_req_state, batch_desc, num_active_loras
        )
        # Adaptive verification reallocates the drafts on the GPU only (the host
        # cumsums no longer describe logits_indices); PCP repartitions.
        setattr(
            input_batch,
            IDENTITY,
            lever_on(EAGERPREP)
            and not (
                self.adaptive_verification is not None
                and scheduler_output.scheduled_spec_decode_tokens
            )
            and input_batch.num_draft_tokens_per_req is not None
            and self.pcp_manager is None
            and logits_indices_is_identity(
                input_batch.cu_num_logits_np,
                input_batch.query_start_loc_np,
                input_batch.num_reqs,
            ),
        )
        return input_batch

    return prepare_inputs


def _wrap_sample(stock):
    @functools.wraps(stock)
    def sample(self, hidden_states, input_batch, grammar_output):
        if (
            self.batch_sharder is not None
            or not lever_on(EAGERPREP)
            or not getattr(input_batch, IDENTITY, False)
        ):
            return stock(self, hidden_states, input_batch, grammar_output)
        return _sample_identity(self, hidden_states, input_batch, grammar_output)

    return sample


def _sample_identity(self, hidden_states, input_batch, grammar_output):
    """``GPUModelRunner.sample`` without the batch sharder, the logits rows a
    view (``select_logits_rows``)."""
    from vllm.v1.worker.gpu.structured_outputs import grammar_invalid_drafts

    sample_hidden_states = select_logits_rows(
        hidden_states, input_batch.logits_indices, True
    )
    logits = self.model.compute_logits(sample_hidden_states)

    invalid_drafts = None
    # A diffusion prefill has no logit rows even when a bitmask row arrived for
    # it.
    if grammar_output is not None and logits.shape[0] > 0:
        # Apply grammar bitmask to the logits in-place.
        assert self.structured_outputs_worker is not None
        self.structured_outputs_worker.apply_grammar_bitmask(
            logits,
            input_batch,
            grammar_output.structured_output_request_ids,
            grammar_output.grammar_bitmask,
        )
        invalid_drafts = grammar_invalid_drafts(
            input_batch,
            grammar_output.structured_output_request_ids,
            grammar_output.num_acceptable_drafts,
        )

    if input_batch.num_reqs == 0:
        sampler_output = None
    elif input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
        assert self.sampler is not None
        sampler_output = self.sampler(logits, input_batch)
    else:
        # Rejection sampling for spec decoding.
        assert self.rejection_sampler is not None
        assert self.speculator is not None
        sampler_output = self.rejection_sampler(
            logits,
            input_batch,
            # Draft logits are needed for probabilistic rejection sampling.
            self.speculator.draft_logits,
            invalid_drafts,
        )

    assert sampler_output is not None
    return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected
