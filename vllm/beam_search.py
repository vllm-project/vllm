# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.inputs import TokensPrompt
from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict


@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """

    # The tokens include the prompt.
    tokens: list[int]
    logprobs: list[dict[int, Logprob]]
    lora_request: LoRARequest | None = None
    cum_logprob: float = 0.0
    text: str | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    multi_modal_data: "MultiModalDataDict | None" = None
    mm_processor_kwargs: dict[str, Any] | None = None


@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """

    sequences: list[BeamSearchSequence]


class BeamSearchInstance:
    def __init__(
        self,
        prompt_tokens: list[int],
        lora_request: LoRARequest | None = None,
        logprobs: list[dict[int, Logprob]] | None = None,
        **kwargs,
    ):
        self.beams: list[BeamSearchSequence] = [
            BeamSearchSequence(
                tokens=prompt_tokens,
                logprobs=[] if logprobs is None else list(logprobs),
                lora_request=lora_request,
                **kwargs,
            )
        ]
        self.completed: list[BeamSearchSequence] = []


def get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score with length penalty.

    Adapted from

    https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    """
    seq_len = len(tokens)
    if tokens[-1] == eos_token_id:
        seq_len -= 1

    return cumulative_logprob / (seq_len**length_penalty)


def create_sort_beams_key_function(eos_token_id: int, length_penalty: float):
    def sort_beams_key(x: BeamSearchSequence) -> float:
        return get_beam_search_score(
            x.tokens, x.cum_logprob, eos_token_id, length_penalty
        )

    return sort_beams_key


def _get_token_prompts(tensor: torch.Tensor, pad_token_id: int) -> list[TokensPrompt]:
    """
    Vectorized removal of leading padding and returns TokensPrompt list.
    tensor: [batch*num_beams, seq_len] on the model device (torch.Tensor)
    """
    mask = tensor.ne(pad_token_id)
    any_nonpad = mask.any(dim=1)
    first_nonpad = torch.where(
        any_nonpad,
        mask.float().argmax(dim=1),
        torch.full(
            (tensor.size(0),), tensor.size(1), device=tensor.device, dtype=torch.long
        ),
    )

    result_prompts = []
    for i in range(tensor.size(0)):
        start = int(first_nonpad[i].item())
        # all-pads -> empty sequence
        processed = [] if start >= tensor.size(1) else tensor[i, start:].tolist()
        result_prompts.append(TokensPrompt(prompt_token_ids=processed))
    return result_prompts


def _flatten_beam_dim(tensor: torch.Tensor) -> torch.Tensor:
    """[batch_size, num_beams, ...] -> [batch_size * num_beams, ...]"""
    shape = list(tensor.shape)
    return torch.reshape(tensor, [shape[0] * shape[1]] + shape[2:])


def _unflatten_beam_dim(
    tensor: torch.Tensor, batch_size: int, num_beams: int
) -> torch.Tensor:
    """[batch_size * num_beams, ...] -> [batch_size, num_beams, ...]"""
    shape = list(tensor.shape)
    return torch.reshape(tensor, [batch_size, num_beams] + shape[1:])


def _gather_beams(tensor: torch.Tensor, beam_indices: torch.Tensor) -> torch.Tensor:
    """Gathers the beam slices indexed by beam_indices into new beam array."""
    while len(beam_indices.shape) < len(tensor.shape):
        beam_indices = beam_indices.unsqueeze(-1)
    gathered_tensor = torch.take_along_dim(input=tensor, indices=beam_indices, dim=1)
    return gathered_tensor


def _check_early_stop_heuristic(
    is_early_stop_heuristic_unsatisfied: torch.Tensor,
    running_beam_scores: torch.Tensor,
    beam_scores: torch.Tensor,
    is_sent_finished: torch.Tensor,
    cur_len: int,
    max_length: int,
    decoder_prompt_len: int,
    early_stopping: bool | str,
    length_penalty: float,
):
    """
    Determine whether early stopping is possible by checking if the best possible
    score of running beams could still improve upon the finished ones.

    Mechanism:
    - Without a length penalty, beam scores typically decrease as more tokens are
    generated. So, if the *best possible* score from any running beam is already
    worse than the *worst* finished beam, we can safely stop early.
    - With a length penalty, scores may increase with longer sequences. In this
    case, we use heuristics to estimate the best possible score — though this estimate
    may not always be correct — and stop if no further improvement seems likely.

    We apply different heuristics depending on the value of `early_stopping`:
    1. `early_stopping == False`:
    -> Use a heuristic that assumes the best score comes from the current length
    minus the decoder prompt length.
    -> See detailed discussion:
    https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565

    2. `early_stopping == "never"`:
    -> Estimate the best score using either `max_length` or `cur_len`, depending
    on the sign of `length_penalty`.
    -> A positive length penalty favors longer sequences, so we use `max_length`
    in that case.

    NOTE: the canonical beam search implementation can be replicated with
    `early_stopping="never"` and `length_penalty=0.0`, which are NOT the default
    flags. The default behavior was empirically found to produce better sequences
    (prior to 2022), and changing it is BC breaking.
    """
    if early_stopping == "never" and length_penalty > 0.0:
        best_hypothetical_length = max_length - decoder_prompt_len
    else:
        best_hypothetical_length = cur_len - decoder_prompt_len
    best_possible_running_score = running_beam_scores[:, :1] / (
        best_hypothetical_length**length_penalty
    )
    worst_finished_score = torch.where(
        is_sent_finished, torch.min(beam_scores, dim=1, keepdim=True)[0], -1.0e9
    )
    return is_early_stop_heuristic_unsatisfied & torch.any(
        best_possible_running_score > worst_finished_score, dim=-1, keepdim=True
    )


def _beam_search_has_unfinished_sequences(
    is_early_stop_heuristic_unsatisfied: torch.Tensor,
    is_sent_finished: torch.Tensor,
    next_token_hits_stopping_criteria: torch.Tensor,
    early_stopping: bool | str,
):
    """
    Beam Search stopping condition -- halts the generation loop if any of these
    conditions becomes False
    """
    # a. Can the open beams improve the top completed scores?
    improvement_possible = torch.any(is_early_stop_heuristic_unsatisfied)

    # b. Is there still a beam without fully completed sequences? This is only
    # relevant if early_stopping is enabled, where we want to finish as soon as
    # all beams have a completed sequence.
    exists_open_beam = ~(torch.all(is_sent_finished) & (early_stopping is True))

    # c. Have we hit a stopping criteria with all running sequences and have no
    # way to continue? e.g. we have reached `max_length``
    valid_continuations = ~torch.all(next_token_hits_stopping_criteria)

    return improvement_possible & exists_open_beam & valid_continuations


def _get_top_k_continuations(
    model_outputs,
    running_sequences: torch.Tensor,
    running_beam_indices: torch.Tensor,
    running_beam_scores: torch.Tensor,
    cur_len: int,
    decoder_prompt_len: int,
    beams_to_keep: int,
    num_beams: int,
    batch_size: int,
    logprobs_size: int,
    is_first_step: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_log_probs = torch.full(
        (batch_size * num_beams, logprobs_size),
        -1.0e9,
        dtype=torch.float32,
        device=running_sequences.device,
    )
    output_token_ids = torch.zeros(
        (batch_size * num_beams, logprobs_size),
        dtype=torch.long,
        device=running_sequences.device,
    )

    if is_first_step:
        output_log_probs = _unflatten_beam_dim(output_log_probs, batch_size, num_beams)
        output_token_ids = _unflatten_beam_dim(output_token_ids, batch_size, num_beams)
        for idx in range(batch_size):
            out = model_outputs[idx]
            lp_map = out.outputs[0].logprobs[0]
            for item_idx, (token_id, logprob_obj) in enumerate(lp_map.items()):
                # token_id might be string or int (be defensive)
                output_token_ids[
                    idx, item_idx // logprobs_size, item_idx % logprobs_size
                ] = int(token_id)
                output_log_probs[
                    idx, item_idx // logprobs_size, item_idx % logprobs_size
                ] = float(logprob_obj.logprob)
    else:
        for idx in range(batch_size * num_beams):
            out = model_outputs[idx]
            lp_map = out.outputs[0].logprobs[0]
            for item_idx, (token_id, logprob_obj) in enumerate(lp_map.items()):
                # token_id might be string or int (be defensive)
                output_token_ids[idx, item_idx] = int(token_id)
                output_log_probs[idx, item_idx] = float(logprob_obj.logprob)
        output_log_probs = _unflatten_beam_dim(output_log_probs, batch_size, num_beams)
        output_token_ids = _unflatten_beam_dim(output_token_ids, batch_size, num_beams)

    # accumulate
    output_log_probs = output_log_probs + running_beam_scores[:, :, None]
    # flatten to [batch, num_beams * logprobs_size]
    output_log_probs = torch.reshape(
        output_log_probs, (batch_size, num_beams * logprobs_size)
    )
    output_token_ids = torch.reshape(
        output_token_ids, (batch_size, num_beams * logprobs_size)
    )

    topk_log_probs, topk_indices = torch.topk(output_log_probs, k=beams_to_keep, dim=-1)

    # Gather K top beams, recover the beam index by floor division and token id
    # by modulo division
    topk_ids = torch.gather(output_token_ids, dim=-1, index=topk_indices)
    topk_current_beam_indices = topk_indices // logprobs_size
    topk_running_beam_indices = _gather_beams(
        running_beam_indices, topk_current_beam_indices
    )
    topk_running_sequences = _gather_beams(running_sequences, topk_current_beam_indices)

    # Update sequences for the K top-k new sequences.
    topk_running_sequences[:, :, cur_len] = topk_ids

    # we want to store the beam indices with batch information ->
    # real beam index = beam index % num beams
    batch_offset = (
        torch.arange(batch_size, device=topk_ids.device).view(-1, 1) * num_beams
    )
    batch_modified_indices = topk_current_beam_indices + batch_offset
    topk_running_beam_indices[:, :, cur_len - decoder_prompt_len] = (
        batch_modified_indices
    )

    return topk_log_probs, topk_running_sequences, topk_running_beam_indices


def _get_running_beams_for_next_iteration(
    topk_log_probs: torch.Tensor,
    topk_running_sequences: torch.Tensor,
    topk_running_beam_indices: torch.Tensor,
    next_token_hits_stopping_criteria: torch.Tensor,
    num_beams: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given the top-K continuations, their scores, and whether they hit a stopping
    criteria, select the best non-finished beams to continue beam search in the
    next iteration.
    """
    # To prevent these just finished sequences from being used in subsequent
    # iterations, set their log probs to a very large negative value
    topk_running_log_probs = (
        topk_log_probs + next_token_hits_stopping_criteria.to(torch.float32) * -1.0e9
    )

    next_topk_indices = torch.topk(topk_running_log_probs, k=num_beams)[1]
    running_sequences = _gather_beams(topk_running_sequences, next_topk_indices)
    running_beam_scores = _gather_beams(topk_running_log_probs, next_topk_indices)
    running_beam_indices = _gather_beams(topk_running_beam_indices, next_topk_indices)
    return running_sequences, running_beam_scores, running_beam_indices


def _update_finished_beams(
    sequences: torch.Tensor,
    topk_running_sequences: torch.Tensor,
    beam_scores: torch.Tensor,
    topk_log_probs: torch.Tensor,
    beam_indices: torch.Tensor,
    topk_running_beam_indices: torch.Tensor,
    is_early_stop_heuristic_unsatisfied: torch.Tensor,
    is_sent_finished: torch.Tensor,
    next_token_hits_stopping_criteria: torch.Tensor,
    top_num_beam_mask: torch.Tensor,
    num_beams: int,
    cur_len: int,
    decoder_prompt_len: int,
    length_penalty: float,
    early_stopping: bool | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Updates the finished beams if (and only if) there are new completed sequences
    that have a higher score than the current finished sequences.
    """
    # Only the top `num_beam` sequences can be considered for the final returned
    # sequences. Remember: the remaining sequences only exist as a backup to ensure
    # that we have at least `num_beams` sequences to continue.
    did_top_num_beams_just_finished = (
        next_token_hits_stopping_criteria & top_num_beam_mask[None, :]
    )

    # Further process topk logits for the finished beams
    # - add length penalty
    topk_log_probs = topk_log_probs / (
        (cur_len + 1 - decoder_prompt_len) ** length_penalty
    )
    # - make sure no scores can be added anymore if beam is full and early stopping
    # is on
    beams_in_batch_are_full = torch.all(is_sent_finished, axis=-1, keepdims=True) & (
        early_stopping is True
    )
    topk_log_probs += beams_in_batch_are_full.to(torch.float32) * -1.0e9
    # - make sure no scores can be added anymore if improvement is not possible
    topk_log_probs += (~is_early_stop_heuristic_unsatisfied).to(torch.float32) * -1.0e9

    # - make sure still running sequences cannot be chosen as finalized beam
    topk_log_probs += (~did_top_num_beams_just_finished) * -1.0e9

    # Get finalized  `num_beam` sequences for the next generation step -- combine the
    # previous finalized data with the new finalized sequences (if any, non-finalized
    # sequences have a very large negative score in this step), and keep the best
    # `num_beams` sequences.
    merged_sequences = torch.cat((sequences, topk_running_sequences), dim=1)
    merged_scores = torch.cat((beam_scores, topk_log_probs), dim=1)
    merged_beam_indices = torch.cat((beam_indices, topk_running_beam_indices), dim=1)
    merged_is_sent_finished = torch.cat(
        (is_sent_finished, did_top_num_beams_just_finished), dim=1
    )
    topk_merged_indices = torch.topk(merged_scores, k=num_beams)[1]
    sequences = _gather_beams(merged_sequences, topk_merged_indices)
    beam_scores = _gather_beams(merged_scores, topk_merged_indices)
    beam_indices = _gather_beams(merged_beam_indices, topk_merged_indices)
    is_sent_finished = _gather_beams(merged_is_sent_finished, topk_merged_indices)
    return sequences, beam_scores, beam_indices, is_sent_finished
