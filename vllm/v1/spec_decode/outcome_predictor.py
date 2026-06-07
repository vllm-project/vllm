# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Outcome Predictor for Speculative Speculative Decoding (SSD).

Predicts the acceptance mask for a set of K draft tokens before the
target model verifies them. Enables pre-speculation to overlap verify and draft.

Architecture: tiny 2-layer MLP (runs concurrently with target verification).
  - Input: draft_logits [batch, K, vocab] -> compressed to [batch, K, 34 features]
  - Hidden: 2 layers, 256 dim each
  - Output: acceptance probability [batch, K]

Target: <5ms inference on RTX 3070Ti for batch=16, K=4.

Reference: AAAI 2026 SSD paper, vLLM issue #36037
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


class OutcomePredictor(nn.Module):
    """
    Tiny MLP that predicts per-draft-token acceptance probability.

    Runs on draft_stream concurrently with target model verification on
    verify_stream. Must be small enough that its latency is less than
    target verification latency.

    Input features (compressed from full logit distribution):
      - top_k=32 logit values: highest-signal features for distribution shape
      - entropy (1): measures distribution spread/uncertainty
      - max_logit (1): draft confidence signal
      Total: 34 features per draft position

    Hidden state from draft model (context for this sequence's continuation)
    is concatenated with per-position features.
    """

    NUM_TOP_K_FEATURES: int = 32
    NUM_EXTRA_FEATURES: int = 2  # entropy + max_logit
    FEATURE_SIZE: int = NUM_TOP_K_FEATURES + NUM_EXTRA_FEATURES  # = 34

    def __init__(
        self,
        hidden_size: int = 2048,  # TinyLlama hidden size; override for larger models
        K: int = 4,  # number of draft positions
        mlp_hidden: int = 256,  # MLP hidden dimension
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.K = K

        mlp_input_size = hidden_size + self.FEATURE_SIZE

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # one logit per draft position
        )

    def extract_features(self, draft_logits: torch.Tensor) -> torch.Tensor:
        """
        Compress full logit distribution to compact feature vector.

        [batch, K, vocab] -> [batch, K, FEATURE_SIZE]

        Uses: top-32 logit values + entropy + max_logit
        """
        # Top-32 logit values (distribution shape signal)
        top_logits = draft_logits.topk(
            self.NUM_TOP_K_FEATURES, dim=-1
        ).values  # [batch, K, 32]

        # Entropy: H(p) = -sum(p * log(p)) -- uncertainty measure
        probs = draft_logits.softmax(dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(
            dim=-1, keepdim=True
        )  # [batch, K, 1]

        # Max logit: draft confidence
        max_logit = draft_logits.max(dim=-1, keepdim=True).values  # [batch, K, 1]

        return torch.cat([top_logits, entropy, max_logit], dim=-1)  # [batch, K, 34]

    def forward(
        self,
        draft_logits: torch.Tensor,  # [batch, K, vocab]
        hidden_state: torch.Tensor,  # [batch, hidden_size] -- from draft model
    ) -> torch.Tensor:
        """
        Returns acceptance probability [batch, K] in [0, 1].
        """
        features = self.extract_features(draft_logits)  # [batch, K, 34]

        # Expand hidden state to per-position: [batch, 1, H] -> [batch, K, H]
        hidden_exp = hidden_state.unsqueeze(1).expand(-1, features.shape[1], -1)

        # MLP input: [batch, K, hidden+34]
        mlp_input = torch.cat([hidden_exp, features], dim=-1)

        return self.mlp(mlp_input).squeeze(-1).sigmoid()  # [batch, K]

    def predict_acceptance_mask(
        self,
        draft_logits: torch.Tensor,  # [batch, K, vocab]
        hidden_state: torch.Tensor,  # [batch, hidden_size]
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Returns binary acceptance mask [batch, K] (bool tensor).
        No .item() calls -- stays on GPU.
        """
        probs = self.forward(draft_logits, hidden_state)
        return probs > threshold

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        hidden_size: int = 2048,
        K: int = 4,
        mlp_hidden: int = 256,
        device: torch.device | None = None,
    ) -> OutcomePredictor:
        """Load a trained OutcomePredictor from disk."""
        predictor = cls(hidden_size=hidden_size, K=K, mlp_hidden=mlp_hidden)
        state_dict = torch.load(path, map_location=device or torch.device("cpu"))
        predictor.load_state_dict(state_dict)
        predictor.eval()
        if device is not None:
            predictor = predictor.to(device)
        logger.info("Loaded SSD OutcomePredictor from %s", path)
        return predictor
