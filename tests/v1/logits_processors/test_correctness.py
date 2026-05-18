# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from collections.abc import Callable
from typing import NamedTuple, TypeAlias

import numpy as np
import pytest
import torch

from tests.utils import create_new_process_for_each_test
from tests.v1.sample.utils import (
    LogitsprocsTestFakes,
    create_fake_logits,
    create_penalty_tensor,
    create_prompt_tokens_tensor,
    fake_apply_logitsprocs,
    fake_update_logitsprocs_state,
)
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    BatchUpdateBuilder,
    LogitBiasLogitsProcessor,
    LogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
    MoveDirectionality,
    build_logitsprocs,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.thinking_budget_state import (
    ThinkingBudgetStateHolder,
    maybe_create_thinking_budget_state_holder,
)

PIN_MEMORY_AVAILABLE = is_pin_memory_available()
MAX_NUM_REQS = 256
VOCAB_SIZE = 1024
NUM_OUTPUT_TOKENS = 20
DEVICE_TYPE = current_platform.device_type
DEVICES = [
    f"{DEVICE_TYPE}:{i}"
    for i in range(1 if current_platform.device_count() == 1 else 2)
]
MAX_NUM_PROMPT_TOKENS = 64
MIN_TOKENS_LEN_THRESHOLD = 5
REQS_PER_LOGITPROC = 50
STR_NO_LOGITPROC = "none"
# Thinking budget uses ``ThinkingBudgetStateHolder`` (not a logits processor).
STR_THINKING_BUDGET = "thinking_budget"

# Thinking token budget testing constants
THINKING_TOKEN_BUDGET = 5
THINK_START_TOKEN_ID = 999
THINK_END_TOKEN_ID = 998

# LogitsProcessor subclass or "none"
LogitprocType: TypeAlias = type[LogitsProcessor] | str


class LogitsProcsRequestParams:
    """Encapsulates key params for a single request in a batch.

    Params can be customized based on the enabled logitproc
    """

    workload_index: int
    logitproc_type: LogitprocType  # Logitproc enabled, specified by str id
    out_tokens: list[int]  # Output tokens required for min tokens test
    prompt_tokens: list[int]  # Dummy prompt tokens placeholder
    params: SamplingParams  # Settings customized for logitproc

    def __init__(self, workload_index: int, logitproc_type: LogitprocType):
        self.workload_index = workload_index
        self.logitproc_type = logitproc_type
        # Number of output tokens is randomly 0 or twice the min-tokens
        # threshold which will be used in testing.
        # Generate diverse random tokens for all processors (more realistic)
        num_tokens = MIN_TOKENS_LEN_THRESHOLD * random.randint(0, 2)
        if num_tokens > 0:
            # Use diverse random tokens
            self.out_tokens = [random.randint(1, 950) for _ in range(num_tokens)]
            # Think-start seed for ``STR_THINKING_BUDGET`` rows.
            if logitproc_type == STR_THINKING_BUDGET:
                self.out_tokens[0] = THINK_START_TOKEN_ID
        else:
            self.out_tokens = []
        self.prompt_tokens = []
        self.params = _sampling_params_from_logitproc(logitproc_type)

    def __str__(self):
        """For debugging"""
        summ = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"MyClass({summ})"


class MockReasoningConfig:
    """Minimal reasoning config for ``ThinkingBudgetStateHolder`` tests."""

    reasoning_start_token_ids = [THINK_START_TOKEN_ID]
    reasoning_end_token_ids = [THINK_END_TOKEN_ID]
    enabled = True


def _generate_fake_sampling_metadata(
    num_output_tokens: int,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> SamplingMetadata:
    """Generate fake sampling metadata with fake logitsprocs"""
    output_token_ids: list[list[int]] = []
    prompt_token_ids: list[list[int]] = []
    for _ in range(batch_size):
        output_token_ids.append(
            np.random.randint(0, vocab_size, size=num_output_tokens).tolist()
        )
        prompt_token_ids.append(
            np.random.randint(
                0, vocab_size, size=np.random.randint(1, MAX_NUM_PROMPT_TOKENS)
            ).tolist()
        )

    vllm_config = VllmConfig()
    vllm_config.reasoning_config = MockReasoningConfig()

    logitsprocs = build_logitsprocs(
        vllm_config=vllm_config,
        device=device,
        is_pin_memory=PIN_MEMORY_AVAILABLE,
        is_pooling_model=False,
    )
    num_spec = (
        vllm_config.speculative_config.num_speculative_tokens
        if vllm_config.speculative_config
        else 0
    )
    thinking_holder = maybe_create_thinking_budget_state_holder(
        vllm_config.reasoning_config,
        vllm_config.scheduler_config.max_num_seqs,
        num_spec,
        device,
        PIN_MEMORY_AVAILABLE,
    )
    fake_sampling_metadata = SamplingMetadata(
        temperature=torch.full((batch_size,), 0.0),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=0,
        prompt_token_ids=create_prompt_tokens_tensor(
            prompt_token_ids, vocab_size, device
        ),
        output_token_ids=output_token_ids,
        frequency_penalties=create_penalty_tensor(batch_size, 0.0, device),
        presence_penalties=create_penalty_tensor(batch_size, 0.0, device),
        repetition_penalties=create_penalty_tensor(batch_size, 1.0, device),
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=logitsprocs,
        thinking_budget_state_holder=thinking_holder,
    )
    return fake_sampling_metadata


def _generate_test_fakes(batch_size: int, device: str) -> LogitsprocsTestFakes:
    """Generate fake logits and sampling metadata"""
    fake_logits = create_fake_logits(batch_size, VOCAB_SIZE)
    # Create one dominant token per batch, to support min-p test
    for i in range(batch_size):
        fake_logits[i, 0] = 10.0  # High logit for first token
        fake_logits[i, 1:] = 1e-2  # Others remain low
    sampling_metadata = _generate_fake_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device)
    )
    return LogitsprocsTestFakes(
        logits=fake_logits,
        sampling_metadata=sampling_metadata,
    )


def _sampling_params_from_logitproc(logitproc_type: LogitprocType) -> SamplingParams:
    """Customize request SamplingParams for a specified logitproc"""
    # SamplingParams for req with no logitproc
    kwargs = {"min_p": 0.0, "logit_bias": None, "min_tokens": 0}
    if fxn := logitsprocs_test_mapping[logitproc_type].gen_request_fxn:
        fxn(kwargs)
    return SamplingParams(**kwargs)


def _generate_mixed_logitsprocs_batch_params(
    reqs_per_logitproc: int,
    logitsprocs_types: list[LogitprocType],
) -> list[LogitsProcsRequestParams]:
    """Define key params for a batch of requests with a different
    logitproc enabled per request.

    The batch will have `reqs_per_logitproc` repeats for all
    `logitsprocs_types` under test, including the case where
    no logitsproc is enabled. The batch is randomly shuffled. The
    size of the batch is `reqs_per_logitproc` times
    `n = len(logitsprocs_types)`

    Args:
      reqs_per_logitproc: number of requests using each logitproc
      logitsprocs_types: logitsprocs under test

    Returns:
      List of per-request params which configure the engine for that request's
      enabled logitproc
    """
    batch_size = len(logitsprocs_types) * reqs_per_logitproc
    # Generate multiple repeats of key params for each logitproc;
    # apply random inverse permutation to the iteration
    # over logitsprocs, such that logitsprocs are shuffled.
    batch_perm = random.sample(range(batch_size), k=batch_size)
    return [
        LogitsProcsRequestParams(
            workload_index=idx,
            logitproc_type=logitsprocs_types[pdx // reqs_per_logitproc],
        )
        for idx, pdx in enumerate(batch_perm)
    ]


def _raise_error_invalid(
    msg_suffix: str,
    batch_index: int,
    request_params: LogitsProcsRequestParams,
    step_idx: int,
    err_cls: type[Exception] = ValueError,
) -> None:
    raise err_cls(
        f"Validation failed for step={step_idx}, "
        f"batch_index={batch_index}, "
        f"workload_index={request_params.workload_index}, "
        f"req_params={request_params}. Reason: {msg_suffix}"
    )


def _logit_bias_params(kwargs: dict) -> None:
    """Logit bias config"""
    kwargs["logit_bias"] = {
        random.randint(0, VOCAB_SIZE - 1): random.choice([-0.1, 0.2])
    }


def _logit_bias_validate(
    test_fakes: LogitsprocsTestFakes,
    persistent_batch: list[LogitsProcsRequestParams],
    logits_new: torch.Tensor,
    batch_index: int,
    request_params: LogitsProcsRequestParams,
    step_idx: int,
) -> None:
    """Validate logit bias logitproc applied correctly"""
    logit_bias = request_params.params.logit_bias
    logits_old = test_fakes.logits[persistent_batch[batch_index].workload_index].cpu()
    logits_new = logits_new[batch_index].cpu()
    for token_id in range(VOCAB_SIZE):
        logit_old_value = logits_old[token_id]
        logit_new_value = logits_new[token_id]
        if token_id in logit_bias:
            bias_value = logit_bias[token_id]
            exp_value = bias_value + logit_old_value
            if logit_new_value != pytest.approx(exp_value):
                _raise_error_invalid(
                    msg_suffix=(
                        f"Biased token {token_id} logit value {logit_new_value} "
                        f"does not match expected value {exp_value} "
                        f"given bias {bias_value}"
                    ),
                    batch_index=batch_index,
                    request_params=request_params,
                    step_idx=step_idx,
                )

        else:
            if logit_new_value != pytest.approx(logit_old_value):
                _raise_error_invalid(
                    msg_suffix=(
                        f"Unbiased token {token_id} logit value {logit_new_value} "
                        f"does not match expected value {logit_old_value}"
                    ),
                    batch_index=batch_index,
                    request_params=request_params,
                    step_idx=step_idx,
                )


def _min_p_params(kwargs: dict) -> None:
    """Min-p logitproc config"""
    kwargs["min_p"] = 0.1


def _min_p_validate(
    test_fakes: LogitsprocsTestFakes,
    persistent_batch: list[LogitsProcsRequestParams],
    logits_new: torch.Tensor,
    batch_index: int,
    request_params: LogitsProcsRequestParams,
    step_idx: int,
) -> None:
    """Validate min-p logitproc applied correctly"""
    for token_id in range(VOCAB_SIZE):
        logits_for_token = logits_new[batch_index][token_id]
        if token_id == 0:
            # Dominant token should always be unmasked
            if logits_for_token == -float("inf"):
                _raise_error_invalid(
                    msg_suffix="Invalid: dominant token 0 masked (-inf)",
                    batch_index=batch_index,
                    request_params=request_params,
                    step_idx=step_idx,
                )
        else:
            if request_params.params.min_p > 0.0:
                # Non-dominant tokens should be masked when min_p > 0
                if logits_for_token != -float("inf"):
                    _raise_error_invalid(
                        msg_suffix=f"Invalid: non-dominant token {token_id} not masked",
                        batch_index=batch_index,
                        request_params=request_params,
                        step_idx=step_idx,
                    )
            else:
                # No masking when min_p is 0
                if logits_for_token == -float("inf"):
                    _raise_error_invalid(
                        msg_suffix=f"Invalid: token {token_id} masked when min_p=0.0",
                        batch_index=batch_index,
                        request_params=request_params,
                        step_idx=step_idx,
                    )


def _min_tokens_params(kwargs: dict) -> None:
    """Min-tokens logitproc config"""
    kwargs["min_tokens"] = MIN_TOKENS_LEN_THRESHOLD
    kwargs["stop_token_ids"] = [
        np.random.randint(0, VOCAB_SIZE - 1)
        for _ in range(np.random.randint(0, VOCAB_SIZE))
    ]


def _min_tokens_validate(
    test_fakes: LogitsprocsTestFakes,
    persistent_batch: list[LogitsProcsRequestParams],
    logits_new: torch.Tensor,
    batch_index: int,
    request_params: LogitsProcsRequestParams,
    step_idx: int,
) -> None:
    """Validate min-tokens logitsproc applied correctly"""
    ref_num_out_tokens = len(request_params.out_tokens)
    min_reached = ref_num_out_tokens >= MIN_TOKENS_LEN_THRESHOLD
    ref_all_stop_token_ids = request_params.params.all_stop_token_ids
    mt_lp: MinTokensLogitsProcessor = next(
        test_fakes.get_logitsprocs_by_cls(MinTokensLogitsProcessor)
    )
    assert isinstance(mt_lp, MinTokensLogitsProcessor)
    min_tok = mt_lp.min_toks.get(batch_index, None)

    # Validate min-token logits processor state
    if min_tok:
        (_, out_tok, all_stop_token_ids) = min_tok
        num_out_tokens = len(out_tok)
        if num_out_tokens != ref_num_out_tokens:
            _raise_error_invalid(
                msg_suffix=(
                    "Number of output tokens in min-token logit processor "
                    f"request metadata ({num_out_tokens}) does not match "
                    f"reference ({ref_num_out_tokens})."
                ),
                batch_index=batch_index,
                request_params=request_params,
                step_idx=step_idx,
            )
        if ref_all_stop_token_ids != all_stop_token_ids:
            _raise_error_invalid(
                msg_suffix=(
                    "Stop token ids do not match reference; all_stop_token_ids: "
                    f"{sorted(all_stop_token_ids)}, ref_all_stop_token_ids: "
                    f"{sorted(ref_all_stop_token_ids)}"
                ),
                batch_index=batch_index,
                request_params=request_params,
                step_idx=step_idx,
            )
        if min_reached:
            _raise_error_invalid(
                msg_suffix=(
                    "Expected min-tokens request with min reached, but batch "
                    "index is recognized by min-tokens logits processor."
                ),
                batch_index=batch_index,
                request_params=request_params,
                step_idx=step_idx,
                err_cls=RuntimeError,
            )

    elif not min_reached:
        _raise_error_invalid(
            msg_suffix=(
                "Expected min-tokens request with min not reached, but batch "
                "index is not recognized by min-tokens logits processor."
            ),
            batch_index=batch_index,
            request_params=request_params,
            step_idx=step_idx,
            err_cls=RuntimeError,
        )

    # Validate min-token logits
    for token_id in range(VOCAB_SIZE):
        logits_for_token = logits_new[batch_index][token_id]
        if token_id in ref_all_stop_token_ids and not min_reached:
            if logits_for_token != -float("inf"):
                _raise_error_invalid(
                    msg_suffix=(
                        f"Token {token_id} is a stop token and "
                        "the sequence has not reached min length, "
                        "but the token is not masked "
                        f"(logit={logits_for_token})"
                    ),
                    batch_index=batch_index,
                    request_params=request_params,
                    step_idx=step_idx,
                )
        else:
            if logits_for_token == -float("inf"):
                _raise_error_invalid(
                    msg_suffix=(
                        f"Token {token_id} should not be masked but "
                        f"is (output len={ref_num_out_tokens})"
                    ),
                    batch_index=batch_index,
                    request_params=request_params,
                    step_idx=step_idx,
                )


def _thinking_budget_params(kwargs: dict) -> None:
    """Set SamplingParams kwargs for thinking token budget tests"""
    kwargs["thinking_token_budget"] = THINKING_TOKEN_BUDGET


def _thinking_budget_validate(
    test_fakes: LogitsprocsTestFakes,
    persistent_batch: list[LogitsProcsRequestParams],
    logits_new: torch.Tensor,
    batch_index: int,
    request_params: LogitsProcsRequestParams,
    step_idx: int,
) -> None:
    """Validate ``ThinkingBudgetStateHolder`` thinking-budget behavior.

    State is keyed by **batch slot** (same index space as logits rows), matching
    ``sync_batch`` / sampler integration (see PR #34668 discussion).
    """
    holder = test_fakes.sampling_metadata.thinking_budget_state_holder
    assert holder is not None
    state = holder._state.get(batch_index)
    params = request_params.params

    if hasattr(params, "thinking_token_budget") and params.thinking_token_budget:
        if state is None:
            _raise_error_invalid(
                msg_suffix=(
                    f"Expected holder state for batch slot {batch_index} "
                    f"with thinking_token_budget={params.thinking_token_budget}"
                ),
                batch_index=batch_index,
                request_params=request_params,
                step_idx=step_idx,
            )

        expected_budget = params.thinking_token_budget
        actual_budget = state["thinking_token_budget"]
        if actual_budget != expected_budget:
            _raise_error_invalid(
                msg_suffix=(
                    f"Budget mismatch: expected {expected_budget}, got {actual_budget}"
                ),
                batch_index=batch_index,
                request_params=request_params,
                step_idx=step_idx,
            )

        output_tokens = request_params.out_tokens
        start_tokens = holder.think_start_token_ids
        thinking_started = False
        if len(start_tokens) > 0:
            for i in range(len(output_tokens) - len(start_tokens) + 1):
                if output_tokens[i : i + len(start_tokens)] == start_tokens:
                    thinking_started = True
                    break

        if thinking_started:
            think_count = state["think_count"]
            budget = state["thinking_token_budget"]
            if think_count >= budget and not state["in_end"]:
                _raise_error_invalid(
                    msg_suffix=(
                        f"Budget exceeded ({think_count} >= {budget}) but "
                        "in_end is false"
                    ),
                    batch_index=batch_index,
                    request_params=request_params,
                    step_idx=step_idx,
                )

            end_tokens = holder.think_end_token_ids
            if (
                think_count >= budget
                and state["in_end"]
                and len(end_tokens) > 0
                and holder.has_tracked_requests()
            ):
                expected_end_token_id = end_tokens[
                    min(state["end_count"], len(end_tokens) - 1)
                ]
                # Holder bumps forced vocab positions to 1e9 (does not -inf others).
                forced_logit = float(logits_new[batch_index, expected_end_token_id])
                if forced_logit < 1.0e8:
                    _raise_error_invalid(
                        msg_suffix=(
                            f"Expected forced end token {expected_end_token_id} "
                            f"with large logit, got {forced_logit}"
                        ),
                        batch_index=batch_index,
                        request_params=request_params,
                        step_idx=step_idx,
                    )


def _none_validate(
    test_fakes: LogitsprocsTestFakes,
    persistent_batch: list[LogitsProcsRequestParams],
    logits_new: torch.Tensor,
    batch_index: int,
    request_params: LogitsProcsRequestParams,
    step_idx: int,
) -> None:
    """Validate that no logits processors are applied"""
    logits = test_fakes.logits[persistent_batch[batch_index].workload_index].cpu()
    ref_logits = logits_new[batch_index]
    if not torch.all(ref_logits == logits):
        mismatch_toks = (ref_logits != logits).nonzero(as_tuple=True)[0].tolist()
        mismatch_strs = []
        for token in mismatch_toks:
            val = float(logits[token])
            ref_val = float(ref_logits[token])
            mismatch_strs.append(f"({token=},{val=},{ref_val=})")
        _raise_error_invalid(
            msg_suffix=(
                f"Unexpected modification of logits: {','.join(mismatch_strs)}"
            ),
            batch_index=batch_index,
            request_params=request_params,
            step_idx=step_idx,
        )


class LogitsprocTestHelpers(NamedTuple):
    """Supports setting up and validating logitsprocs unit tests."""

    eval_fxn: Callable
    gen_request_fxn: Callable | None = None


logitsprocs_test_mapping = {
    STR_NO_LOGITPROC: LogitsprocTestHelpers(eval_fxn=_none_validate),
    LogitBiasLogitsProcessor: LogitsprocTestHelpers(
        gen_request_fxn=_logit_bias_params, eval_fxn=_logit_bias_validate
    ),
    MinPLogitsProcessor: LogitsprocTestHelpers(
        gen_request_fxn=_min_p_params, eval_fxn=_min_p_validate
    ),
    MinTokensLogitsProcessor: LogitsprocTestHelpers(
        gen_request_fxn=_min_tokens_params, eval_fxn=_min_tokens_validate
    ),
    STR_THINKING_BUDGET: LogitsprocTestHelpers(
        gen_request_fxn=_thinking_budget_params, eval_fxn=_thinking_budget_validate
    ),
}


def _get_test_cases() -> list[list[str]]:
    """Each test case is a set of logitsprocs"""
    logitsprocs_types = list(logitsprocs_test_mapping.keys())

    # Isolate thinking-budget handling from other processors to avoid cross-talk.
    thinking_id: LogitprocType = STR_THINKING_BUDGET
    other_processors = [
        p for p in logitsprocs_types if p != STR_NO_LOGITPROC and p != thinking_id
    ]

    return (
        [[STR_NO_LOGITPROC]]
        + [[logitproc_type, STR_NO_LOGITPROC] for logitproc_type in other_processors]
        + [other_processors]
        + [[thinking_id]]
    )


def _generate_fake_step_update(
    persistent_batch: list[LogitsProcsRequestParams],
    workload_params: list[LogitsProcsRequestParams],
    wdx: int,
    batch_update_builder: BatchUpdateBuilder,
) -> tuple[BatchUpdate | None, int, int]:
    batch_size = len(persistent_batch)
    workload_size = len(workload_params)
    workload_reqs_remaining = workload_size - wdx
    max_add_remove_per_step = max(1, int(0.2 * workload_size))

    # 50% of steps: add no reqs
    # Other 50%: add a limited number of reqs (less than the number
    # of workload reqs remaining, less than an arbitrary max)
    # If no workload reqs remain: 100% of steps have 0 adds
    num_step_add = (
        random.choice(
            [
                0,
                random.randint(
                    1, min(max_add_remove_per_step, workload_reqs_remaining)
                ),
            ]
        )
        if workload_reqs_remaining
        else 0
    )

    # 50% of steps: remove no requests
    # Other 50%: remove a limited number of reqs (less than the number
    # persistent batch reqs remaining, less than an arbitrary max)
    # If persistent batch is empty: 100% of steps have 0 removals until
    # more requests are added. Assume that removed requests are always
    # drawn from the current batch, before new adds
    num_step_remove = (
        random.choice([0, random.randint(1, min(max_add_remove_per_step, batch_size))])
        if batch_size
        else 0
    )

    num_step_add_replace = min(num_step_add, num_step_remove)

    # Generate fake removed request indices drawn from persistent batch indices
    for removal in random.sample(range(batch_size), num_step_remove):
        batch_update_builder.removed_append(removal)

    # Get added requests from workload
    for add_req_params in workload_params[wdx : (wdx + num_step_add_replace)]:
        # Replace as many removed requests as possible with added requests
        add_remove_idx = batch_update_builder.pop_removed()
        batch_update_builder.added.append(
            (
                add_remove_idx,
                add_req_params.params,
                add_req_params.prompt_tokens,
                add_req_params.out_tokens,
            )
        )
        persistent_batch[add_remove_idx] = add_req_params

    # Append remaining added requests to end of batch
    add_reqs_append = workload_params[
        (wdx + num_step_add_replace) : (wdx + num_step_add)
    ]
    batch_update_builder.added.extend(
        [
            (
                adx + batch_size,
                add_req_params.params,
                add_req_params.prompt_tokens,
                add_req_params.out_tokens,
            )
            for adx, add_req_params in enumerate(add_reqs_append)
        ]
    )
    persistent_batch.extend(add_reqs_append)
    pre_condense_batch_size = len(persistent_batch)
    wdx += num_step_add  # Update workload offset

    # Simulate condensing persistent batch
    last_nonempty_index = pre_condense_batch_size - 1
    condensed_to_idxs = set()
    while batch_update_builder.removed:
        if (
            last_nonempty_index in batch_update_builder.removed
            or last_nonempty_index in condensed_to_idxs
        ):
            last_nonempty_index -= 1
            continue
        # last_nonempty_index is the highest persistent batch index that was
        # not removed
        first_empty_index = batch_update_builder.peek_removed()
        assert first_empty_index is not None
        if first_empty_index > last_nonempty_index:
            break
        # first_empty_index is the lowest removed persistent batch index
        # that is less than last_nonempty_index
        #
        # move last_nonempty_index -> first_empty_index
        batch_update_builder.pop_removed()
        condensed_to_idxs.add(first_empty_index)
        persistent_batch[first_empty_index] = persistent_batch[last_nonempty_index]
        batch_update_builder.moved.append(
            (last_nonempty_index, first_empty_index, MoveDirectionality.UNIDIRECTIONAL)
        )

        last_nonempty_index -= 1

    # Now removed requests & gaps left by non-removed requests that got
    # moved downward are grouped consecutively in the upper indices of
    # the persistent batch. Truncate them to get condensed persistent batch
    condensed_batch_size = batch_size + num_step_add - num_step_remove
    persistent_batch[:] = persistent_batch[0:condensed_batch_size]

    if condensed_batch_size > 1:
        # Simulate arbitrary batch ordering in the kernel backend
        # Generate a random number k of non-overlapping swap tuples
        k = random.randint(0, condensed_batch_size // 2)
        idxs = list(range(condensed_batch_size))
        random.shuffle(idxs)
        swaps = [tuple(sorted([idxs[2 * i], idxs[2 * i + 1]])) for i in range(k)]
        batch_update_builder.moved.extend(
            [(sw[0], sw[1], MoveDirectionality.SWAP) for sw in swaps]
        )
        for adx, bdx in swaps:
            persistent_batch[adx], persistent_batch[bdx] = (
                persistent_batch[bdx],
                persistent_batch[adx],
            )

    return (
        batch_update_builder.get_and_reset(condensed_batch_size),
        wdx,
        workload_size - wdx,
    )


def _assert_valid(
    batch_size: int,
    persistent_batch: list[LogitsProcsRequestParams],
    test_fakes: LogitsprocsTestFakes,
    slice_idxs: list[int],
    logits_w_lp: torch.Tensor,
    step_idx: int,
) -> None:
    if not slice_idxs:
        # Trivial case of empty persistent batch
        assert len(persistent_batch) == 0
        if logits_w_lp.shape[0] != 0:
            raise ValueError(
                "Fake persistent batch is empty but logitsprocs "
                f"output batch has shape {logits_w_lp.shape}"
            )
        return

    # Validate logits for each fake request
    for batch_index in range(batch_size):
        request_params = persistent_batch[batch_index]
        # Invoke the appropriate validation function for
        # the logitproc employed by this request
        fxn = logitsprocs_test_mapping[request_params.logitproc_type].eval_fxn
        fxn(
            test_fakes=test_fakes,
            persistent_batch=persistent_batch,
            logits_new=logits_w_lp,
            batch_index=batch_index,
            request_params=request_params,
            step_idx=step_idx,
        )


def _slot_outputs_for_metadata(
    persistent_batch: list[LogitsProcsRequestParams], pad_len: int
) -> list[list[int]]:
    """Per-batch-slot output token ids aligned with ``SamplingMetadata`` rows."""
    rows: list[list[int]] = [[] for _ in range(pad_len)]
    for i, req in enumerate(persistent_batch):
        if i < pad_len:
            rows[i] = list(req.out_tokens)
    return rows


@create_new_process_for_each_test()
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reqs_per_logitproc", [REQS_PER_LOGITPROC])
@pytest.mark.parametrize("logitsprocs_under_test", _get_test_cases())
def test_logitsprocs(
    device: str, reqs_per_logitproc: int, logitsprocs_under_test: list[LogitprocType]
):
    random.seed(40)
    torch.set_default_device(device)

    # Define a shuffled batch of requests which individually use a different
    # logitproc, or no logitproc at all
    workload_params = _generate_mixed_logitsprocs_batch_params(
        reqs_per_logitproc=reqs_per_logitproc, logitsprocs_types=logitsprocs_under_test
    )
    workload_size = len(workload_params)

    # Create fake test data structures for testing.
    test_fakes = _generate_test_fakes(workload_size, device)

    wdx = 0  # Next request index in workload to add
    persistent_batch: list[
        LogitsProcsRequestParams
    ] = []  # Persistent batch state, as list of workload indices

    # Generate fake removed request indices from current persistent
    # batch before adds
    batch_update_builder = BatchUpdateBuilder()

    # Break when entire workload has been added previously and persistent
    # batch is empty
    workload_reqs_remaining = workload_size
    batch_size = 0
    step_idx = 0
    while True:
        if not (workload_reqs_remaining or batch_size):
            break

        (
            batch_update,
            wdx,
            workload_reqs_remaining,
        ) = _generate_fake_step_update(
            persistent_batch=persistent_batch,
            workload_params=workload_params,
            wdx=wdx,
            batch_update_builder=batch_update_builder,
        )
        batch_size = len(persistent_batch)

        # Apply fake batch update to logitsprocs
        fake_update_logitsprocs_state(test_fakes, batch_update)

        # Emulate application of logits processors + thinking holder (sampler order).
        slice_idxs = [req.workload_index for req in persistent_batch]
        slot_rows = _slot_outputs_for_metadata(persistent_batch, workload_size)
        logits_w_lp = fake_apply_logitsprocs(test_fakes, slice_idxs, slot_rows).cpu()

        _assert_valid(
            batch_size=batch_size,
            persistent_batch=persistent_batch,
            test_fakes=test_fakes,
            slice_idxs=slice_idxs,
            logits_w_lp=logits_w_lp,
            step_idx=step_idx,
        )

        step_idx += 1


class MockReasoningNoEndTokens:
    """Reasoning config with no end token ids (disables enforcement in holder)."""

    reasoning_start_token_ids = [THINK_START_TOKEN_ID]
    reasoning_end_token_ids: list[int] = []


def test_maybe_create_thinking_budget_holder_without_reasoning():
    cfg = VllmConfig()
    assert cfg.reasoning_config is None
    assert (
        maybe_create_thinking_budget_state_holder(
            None,
            cfg.scheduler_config.max_num_seqs,
            0,
            torch.device("cpu"),
            False,
        )
        is None
    )


def test_thinking_budget_holder_has_tracked_after_sync_add():
    vc = VllmConfig()
    vc.reasoning_config = MockReasoningConfig()
    h = ThinkingBudgetStateHolder(
        vc.reasoning_config,
        vc.scheduler_config.max_num_seqs,
        0,
        torch.device("cpu"),
        False,
    )
    assert not h.has_tracked_requests()
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[
                (
                    0,
                    SamplingParams(thinking_token_budget=3),
                    None,
                    [THINK_START_TOKEN_ID],
                )
            ],
            moved=(),
        )
    )
    assert h.has_tracked_requests()
    assert h._state[0]["thinking_token_budget"] == 3


def test_thinking_budget_holder_sync_remove_clears_state():
    vc = VllmConfig()
    vc.reasoning_config = MockReasoningConfig()
    h = ThinkingBudgetStateHolder(
        vc.reasoning_config,
        vc.scheduler_config.max_num_seqs,
        0,
        torch.device("cpu"),
        False,
    )
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[
                (
                    0,
                    SamplingParams(thinking_token_budget=3),
                    None,
                    [],
                )
            ],
            moved=(),
        )
    )
    assert h.has_tracked_requests()
    h.sync_batch(BatchUpdate(batch_size=0, removed=(0,), added=(), moved=()))
    assert not h.has_tracked_requests()


def test_thinking_budget_holder_sync_add_without_budget_drops_row():
    vc = VllmConfig()
    vc.reasoning_config = MockReasoningConfig()
    h = ThinkingBudgetStateHolder(
        vc.reasoning_config,
        vc.scheduler_config.max_num_seqs,
        0,
        torch.device("cpu"),
        False,
    )
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[(0, SamplingParams(), None, [])],
            moved=(),
        )
    )
    assert not h.has_tracked_requests()


def test_thinking_budget_holder_swap_exchanges_state():
    vc = VllmConfig()
    vc.reasoning_config = MockReasoningConfig()
    h = ThinkingBudgetStateHolder(
        vc.reasoning_config,
        vc.scheduler_config.max_num_seqs,
        0,
        torch.device("cpu"),
        False,
    )
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (
                    0,
                    SamplingParams(thinking_token_budget=3),
                    None,
                    [],
                ),
                (
                    1,
                    SamplingParams(thinking_token_budget=7),
                    None,
                    [],
                ),
            ],
            moved=(),
        )
    )
    b0, b1 = h._state[0]["thinking_token_budget"], h._state[1]["thinking_token_budget"]
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert h._state[0]["thinking_token_budget"] == b1
    assert h._state[1]["thinking_token_budget"] == b0


def test_thinking_budget_holder_unidirectional_move():
    vc = VllmConfig()
    vc.reasoning_config = MockReasoningConfig()
    h = ThinkingBudgetStateHolder(
        vc.reasoning_config,
        vc.scheduler_config.max_num_seqs,
        0,
        torch.device("cpu"),
        False,
    )
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (
                    1,
                    SamplingParams(thinking_token_budget=4),
                    None,
                    [],
                ),
            ],
            moved=(),
        )
    )
    assert 1 in h._state and 0 not in h._state
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(1, 0, MoveDirectionality.UNIDIRECTIONAL)],
        )
    )
    assert 0 in h._state and 1 not in h._state
    assert h._state[0]["thinking_token_budget"] == 4


def test_thinking_budget_holder_update_state_repeat_indices_last_row_wins():
    vc = VllmConfig()
    vc.reasoning_config = MockReasoningConfig()
    h = ThinkingBudgetStateHolder(
        vc.reasoning_config,
        vc.scheduler_config.max_num_seqs,
        0,
        torch.device("cpu"),
        False,
    )
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[
                (
                    0,
                    SamplingParams(thinking_token_budget=5),
                    None,
                    [THINK_START_TOKEN_ID],
                )
            ],
            moved=(),
        )
    )
    out_lists = [[THINK_START_TOKEN_ID], [THINK_START_TOKEN_ID, 10, 11, 12, 13, 14]]
    h.update_state(
        out_lists,
        None,
        torch.tensor([0, 0], dtype=torch.long),
    )
    assert h._state[0]["output_tok_ids"] == out_lists[1]


def test_thinking_budget_holder_spec_mode_tensor_layout():
    h = ThinkingBudgetStateHolder(
        MockReasoningConfig(),
        8,
        2,
        torch.device("cpu"),
        False,
    )
    assert h.in_spec_mode
    assert h.mask.shape[0] == 8 * (2 + 1)


def test_thinking_budget_holder_empty_end_tokens_disables_row():
    vc = VllmConfig()
    vc.reasoning_config = MockReasoningNoEndTokens()
    h = ThinkingBudgetStateHolder(
        vc.reasoning_config,
        vc.scheduler_config.max_num_seqs,
        0,
        torch.device("cpu"),
        False,
    )
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[
                (
                    0,
                    SamplingParams(thinking_token_budget=5),
                    None,
                    [THINK_START_TOKEN_ID],
                )
            ],
            moved=(),
        )
    )
    h.update_state([[THINK_START_TOKEN_ID, 1]], None, None)
    assert h._state[0]["thinking_token_budget"] == -1


def test_thinking_budget_enforced_without_penalties():
    """Regression test for gpu_input_batch.py bug.

    When thinking_budget_tracks_reqs=True and no penalties/bad_words are set,
    the old code computed needs_output_token_ids=False (inverted condition:
    ``or not thinking_budget_tracks_reqs``), causing update_state to receive
    an empty list and skip _update_think_state for every request.

    Fix: changed ``or not thinking_budget_tracks_reqs`` to
    ``or thinking_budget_tracks_reqs`` so that output_token_ids is populated
    whenever the thinking budget state holder has tracked requests.

    This test verifies that update_state correctly calls _update_think_state
    (setting in_end=True) when given the real output_token_ids, and that
    passing an empty list (the pre-fix behavior) prevents budget enforcement.
    """
    vc = VllmConfig()
    vc.reasoning_config = MockReasoningConfig()
    budget = 3  # allow 3 thinking tokens

    h = ThinkingBudgetStateHolder(
        vc.reasoning_config,
        vc.scheduler_config.max_num_seqs,
        0,
        torch.device("cpu"),
        False,
    )
    output_token_ids: list[int] = []
    h.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[
                (
                    0,
                    SamplingParams(thinking_token_budget=budget),
                    None,
                    output_token_ids,
                )
            ],
            moved=(),
        )
    )
    assert h.has_tracked_requests()

    # Simulate the buggy behavior: update_state receives empty list.
    # _update_think_state is skipped → in_end stays False → no budget enforcement.
    h.update_state([], None, None)
    assert not h._state[0].get("in_end", False), (
        "With empty output_token_ids, in_end should stay False (budget not yet tracked)"
    )

    # Simulate the correct behavior: output_token_ids is the live list.
    # Step 1: think-start token appears.
    output_token_ids.append(THINK_START_TOKEN_ID)
    h.update_state([output_token_ids], None, None)
    assert not h._state[0].get("in_end", False), (
        "Still within budget after 0 think tokens"
    )

    # Steps 2–4: 3 thinking tokens (hits the budget exactly).
    for tok in [1, 2, 3]:
        output_token_ids.append(tok)
        h.update_state([output_token_ids], None, None)

    # After exactly `budget` thinking tokens the holder should force end token.
    assert h._state[0].get("in_end", False), (
        "Budget exceeded: in_end should be True so that apply_to_logits "
        "forces the end token"
    )
