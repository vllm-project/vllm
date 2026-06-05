# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Speculative Speculative Decoding (SSD) for vLLM.

Eliminates the sequential gap between target verification and draft
pre-speculation by overlapping them on two separate CUDA streams.

Problem (standard spec decode):
  |--- draft K tokens ---|--- target verify ---|--- draft K tokens ---|
                                               ^
                                     GPU idle here (draft waits for verify)

SSD solution:
  Stream A (verify): |--- verify batch_i ---|--- verify batch_{i+1} ---|
  Stream B (draft):       |-- predict outcome_i --|-- pre-draft for predicted --|

If prediction correct (~70-80%): next batch's draft is READY immediately.
If prediction wrong: re-draft from correct continuation (same cost as standard).

Net speedup: accuracy x (draft_fraction_of_step_time)
At 75% accuracy, draft = 30% of step: 0.75 x 0.30 = 22.5% throughput improvement.

This is the first implementation of the SSD algorithm (vLLM issue #36037).

Reference: AAAI 2026 SSD paper, vLLM issue #36037
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.outcome_predictor import OutcomePredictor

logger = init_logger(__name__)


class SSDAsyncOverlap:
    """
    Manages the two-stream async overlap for SSD.

    Handles:
      - Two CUDA streams (verify_stream, draft_stream)
      - CUDA event synchronization between streams
      - Pre-speculation cache (stores pre-drafted tokens + correctness flag)
      - Metrics tracking (prediction accuracy)

    Usage::

        ssd = SSDAsyncOverlap(outcome_predictor_path="/path/to/predictor.pt")

        # Each decode step:
        target_out, accept_mask, pred_correct = ssd.run_verify_phase(
            verify_fn, draft_tokens, draft_scores
        )
        ssd.run_predraft_phase(
            draft_fn, draft_logits, draft_hidden, draft_tokens
        )
        accepted_tokens = ssd.sync_and_get_result(
            accept_mask, target_out, draft_tokens, redraft_fn
        )
    """

    def __init__(
        self,
        outcome_predictor_path: Optional[str] = None,
        hidden_size: int = 2048,
        num_speculative_tokens: int = 4,
        device: Optional[torch.device] = None,
    ) -> None:
        self._device = device or torch.device("cuda")

        # Two CUDA streams
        self.verify_stream = torch.cuda.Stream(device=self._device)
        self.draft_stream = torch.cuda.Stream(device=self._device)

        # Cross-stream synchronization events
        # kv_ready:      fires when KV cache is updated (draft stream needs this)
        # verify_done:   fires when full verification is complete
        # predraft_done: fires when pre-speculation is complete
        self.kv_ready_event = torch.cuda.Event()
        self.verify_done_event = torch.cuda.Event()
        self.predraft_done_event = torch.cuda.Event()

        # Outcome predictor (tiny MLP -- loads lazily)
        self._predictor: Optional[OutcomePredictor] = None
        if outcome_predictor_path:
            self._load_predictor(
                outcome_predictor_path,
                hidden_size=hidden_size,
                K=num_speculative_tokens,
            )

        # Pre-speculation cache
        self._pre_spec_tokens: Optional[torch.Tensor] = None  # [batch, K] int
        self._pre_spec_scores: Optional[torch.Tensor] = None  # [batch, K] float
        self._pre_spec_mask: Optional[torch.Tensor] = None  # [batch, K] bool
        self._pre_spec_valid: bool = False

        # Accumulated state from last draft step (needed for outcome predictor)
        self._last_draft_logits: Optional[torch.Tensor] = None  # [batch, K, vocab]
        self._last_draft_hidden: Optional[torch.Tensor] = None  # [batch, hidden]
        self._last_draft_tokens: Optional[torch.Tensor] = None  # [batch, K] int

        # Metrics
        self._correct_predictions: int = 0
        self._total_predictions: int = 0

    def _load_predictor(
        self,
        path: str,
        hidden_size: int,
        K: int,
    ) -> None:
        self._predictor = OutcomePredictor.from_pretrained(
            path=path,
            hidden_size=hidden_size,
            K=K,
            device=self._device,
        )

    @property
    def prediction_accuracy(self) -> float:
        return self._correct_predictions / max(1, self._total_predictions)

    def run_verify_phase(
        self,
        verify_fn: Callable[
            [torch.Tensor, torch.Tensor],
            Tuple[Any, torch.Tensor],
        ],
        draft_tokens: torch.Tensor,  # [batch, K]
        draft_scores: torch.Tensor,  # [batch, K]
    ) -> Tuple[Any, torch.Tensor, Optional[bool]]:
        """
        Run target verification on verify_stream.

        Args:
            verify_fn: callable that verifies draft tokens; returns
                       (target_output, accept_mask [batch, K] bool).
            draft_tokens: [batch, K] integer draft token ids.
            draft_scores: [batch, K] draft log-probabilities.

        Returns:
            (target_output, actual_accept_mask, was_prediction_correct)
            was_prediction_correct is None when no pre-speculation was cached.
        """
        prediction_correct: Optional[bool] = None

        with torch.cuda.stream(self.verify_stream):
            target_output, actual_accept_mask = verify_fn(draft_tokens,
                                                          draft_scores)

            # Signal that KV cache is updated (draft stream can now pre-speculate)
            self.kv_ready_event.record(self.verify_stream)

            # Check if our pre-speculation prediction was correct
            if self._pre_spec_valid and self._pre_spec_mask is not None:
                prediction_correct = bool(
                    torch.equal(actual_accept_mask, self._pre_spec_mask))
                self._total_predictions += 1
                if prediction_correct:
                    self._correct_predictions += 1

            self.verify_done_event.record(self.verify_stream)

        return target_output, actual_accept_mask, prediction_correct

    def run_predraft_phase(
        self,
        draft_fn: Callable[
            [torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        draft_logits: torch.Tensor,  # [batch, K, vocab]
        draft_hidden: torch.Tensor,  # [batch, hidden]
        draft_tokens: torch.Tensor,  # [batch, K] int
    ) -> None:
        """
        Run outcome predictor + pre-speculation on draft_stream.

        Runs concurrently with verify_stream, overlapping target verification.

        Args:
            draft_fn: callable that given a continuation token [batch] produces
                      (tokens [batch,K], scores [batch,K], hidden [batch,H],
                       logits [batch,K,vocab]).
            draft_logits: [batch, K, vocab] logits from the last draft step.
            draft_hidden: [batch, hidden] hidden state from last draft model.
            draft_tokens: [batch, K] last set of draft token ids.
        """
        if self._predictor is None:
            return

        with torch.cuda.stream(self.draft_stream):
            # Wait for KV to be ready (not for full verify -- only need KV state)
            self.draft_stream.wait_event(self.kv_ready_event)

            # Predict acceptance mask using tiny MLP -- no .item() calls
            with torch.no_grad():
                predicted_mask = self._predictor.predict_acceptance_mask(
                    draft_logits, draft_hidden)  # [batch, K] bool

            # Determine predicted continuation token
            predicted_continuation = self._get_predicted_continuation(
                predicted_mask, draft_tokens)

            # Pre-draft tokens for the predicted continuation
            pre_tokens, pre_scores, new_hidden, new_logits = draft_fn(
                predicted_continuation)

            # Cache pre-speculated tokens for next verify phase
            self._pre_spec_tokens = pre_tokens
            self._pre_spec_scores = pre_scores
            self._pre_spec_mask = predicted_mask
            self._pre_spec_valid = True

            # Update accumulated state
            self._last_draft_hidden = new_hidden
            self._last_draft_logits = new_logits
            self._last_draft_tokens = pre_tokens

            self.predraft_done_event.record(self.draft_stream)

    def _get_predicted_continuation(
        self,
        predicted_mask: torch.Tensor,  # [batch, K] bool
        draft_tokens: torch.Tensor,  # [batch, K] int
    ) -> torch.Tensor:
        """
        Determine the continuation token given the predicted acceptance mask.

        For each sequence, finds the first non-accepted position and returns
        the draft token there as the continuation start token.

        Returns [batch] int tensor -- stays on GPU, no .item() calls.
        """
        # predicted_mask: True = accepted, False = rejected
        # First False position is where speculation continues from
        # If all accepted: use the last draft token
        batch_size = predicted_mask.shape[0]
        K = predicted_mask.shape[1]

        # Inverted mask: True where rejected
        rejected = ~predicted_mask  # [batch, K]

        # For each sequence: index of first rejection, or K-1 if all accepted
        # Use argmax on rejected (first True in each row)
        has_rejection = rejected.any(dim=-1)  # [batch] bool
        first_rejection = rejected.int().argmax(dim=-1)  # [batch] int64

        # If no rejection: fall back to last position (K-1)
        cont_idx = torch.where(has_rejection, first_rejection,
                               torch.full_like(first_rejection,
                                               K - 1))  # [batch]

        # Gather the continuation token for each batch element
        cont_idx_expanded = cont_idx.unsqueeze(-1)  # [batch, 1]
        continuation = draft_tokens.gather(1,
                                           cont_idx_expanded).squeeze(-1)  # [batch]
        return continuation

    def sync_and_get_result(
        self,
        actual_accept_mask: torch.Tensor,  # [batch, K] bool
        target_output: Any,
        draft_tokens: torch.Tensor,  # [batch, K]
        redraft_fn: Callable[
            [torch.Tensor, torch.Tensor],
            List[torch.Tensor],
        ],
    ) -> List[torch.Tensor]:
        """
        Finalize the result for this step.

        If pre-speculation prediction was correct -> use pre-drafted tokens.
        If wrong -> re-draft from correct continuation (standard spec decode cost).

        Output is bit-for-bit identical to standard spec decode; the only
        change is whether the next draft was pre-computed.

        Args:
            actual_accept_mask: [batch, K] bool acceptance mask from verify.
            target_output: raw target model output (for bonus token lookup).
            draft_tokens: [batch, K] tokens that were verified.
            redraft_fn: callable(continuation [batch], accept_mask [batch, K])
                        -> list of accepted token tensors.

        Returns:
            List of accepted token tensors, one per batch element.
        """
        # Wait for verify stream before finalizing
        torch.cuda.current_stream().wait_stream(self.verify_stream)

        if self._pre_spec_valid and self._pre_spec_mask is not None:
            if torch.equal(actual_accept_mask, self._pre_spec_mask):
                # Pre-speculation correct -- use cached result
                return self._apply_acceptance(actual_accept_mask, draft_tokens)
            else:
                # Prediction wrong -- re-draft from correct continuation
                correct_cont = self._get_correct_continuation(
                    actual_accept_mask, draft_tokens, target_output)
                return redraft_fn(correct_cont, actual_accept_mask)

        return self._apply_acceptance(actual_accept_mask, draft_tokens)

    def _apply_acceptance(
        self,
        accept_mask: torch.Tensor,  # [batch, K] bool
        draft_tokens: torch.Tensor,  # [batch, K] int
    ) -> List[torch.Tensor]:
        """Extract accepted tokens using acceptance mask (no .item() calls)."""
        batch_size = accept_mask.shape[0]
        results: List[torch.Tensor] = []
        for b in range(batch_size):
            mask = accept_mask[b]
            tokens = draft_tokens[b]
            results.append(tokens[mask])
        return results

    def _get_correct_continuation(
        self,
        actual_mask: torch.Tensor,  # [batch, K] bool
        draft_tokens: torch.Tensor,  # [batch, K] int
        target_output: Any,
    ) -> torch.Tensor:
        """
        Get the continuation token given actual acceptance mask.

        Prefers bonus tokens from the target model when available.
        Falls back to the first rejected draft token.

        Returns [batch] int tensor on the same device as draft_tokens.
        """
        batch_size = actual_mask.shape[0]
        K = actual_mask.shape[1]

        rejected = ~actual_mask  # [batch, K]
        has_rejection = rejected.any(dim=-1)  # [batch] bool
        first_rejection = rejected.int().argmax(dim=-1)  # [batch]

        # Default: use draft token at first rejected position
        cont_idx = torch.where(has_rejection, first_rejection,
                               torch.full_like(first_rejection, K - 1))
        result = draft_tokens.gather(1, cont_idx.unsqueeze(-1)).squeeze(-1)

        # Override with target model bonus tokens if available
        if (target_output is not None
                and hasattr(target_output, "bonus_tokens")
                and target_output.bonus_tokens is not None):
            bonus = target_output.bonus_tokens  # [batch]
            result = torch.where(has_rejection, bonus, result)

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Return SSD prediction accuracy metrics."""
        return {
            "ssd_prediction_accuracy": self.prediction_accuracy,
            "ssd_correct_predictions": self._correct_predictions,
            "ssd_total_predictions": self._total_predictions,
        }
