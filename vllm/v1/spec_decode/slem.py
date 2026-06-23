# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SLEM (String-Level Exact Match) speculative decoding for heterogeneous vocabs.

Implements Algorithm 2 from:
  "Accelerating LLM Inference with Lossless Speculative Decoding Algorithms
   for Heterogeneous Vocabularies" — Timor et al., ICML 2025.
  https://arxiv.org/abs/2502.05202

The key insight: draft tokens are decoded to text and re-tokenized with the
target tokenizer at the STRING level (not per-token). This ensures lossless
mapping even when token boundaries differ between vocabularies.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class SlemMapper:
    """String-Level Exact Match mapper for heterogeneous-vocabulary spec decode.

    The draft model generates tokens using its full vocabulary (no intersection
    constraint). After generating K draft tokens, the entire sequence is decoded
    to text and re-tokenized with the target tokenizer for verification.

    This guarantees lossless mapping: any text the draft model produces will be
    faithfully represented in the target vocabulary for scoring.
    """

    def __init__(
        self,
        target_tokenizer,
        draft_tokenizer,
        device: torch.device,
    ):
        self.target_tokenizer = target_tokenizer
        self.draft_tokenizer = draft_tokenizer
        self.device = device

        self.target_eos_id: int = getattr(target_tokenizer, "eos_token_id", 0) or 0
        self.draft_eos_id: int = getattr(draft_tokenizer, "eos_token_id", 0) or 0

        draft_vocab_size = len(draft_tokenizer.get_vocab())
        target_vocab_size = len(target_tokenizer.get_vocab())

        # Precompute 1:1 lookup tables for the common case where a single
        # token maps cleanly between vocabularies. Used ONLY for
        # set_inputs_first_pass (target→draft for feeding context to draft
        # model), where approximate mapping for KV context is acceptable.
        # The forward path (draft→target proposals) always uses string-level
        # mapping for losslessness.
        d2t = torch.full((draft_vocab_size,), self.target_eos_id, dtype=torch.int64)
        mapped_d2t = 0
        for did in range(draft_vocab_size):
            text = draft_tokenizer.decode([did], skip_special_tokens=False)
            target_ids = target_tokenizer.encode(text, add_special_tokens=False)
            if len(target_ids) == 1:
                d2t[did] = target_ids[0]
                mapped_d2t += 1

        t2d = torch.full((target_vocab_size,), self.draft_eos_id, dtype=torch.int64)
        mapped_t2d = 0
        for tid in range(target_vocab_size):
            text = target_tokenizer.decode([tid], skip_special_tokens=False)
            draft_ids = draft_tokenizer.encode(text, add_special_tokens=False)
            if len(draft_ids) == 1:
                t2d[tid] = draft_ids[0]
                mapped_t2d += 1

        self._draft_to_target_table = d2t.to(device)
        self._target_to_draft_table = t2d.to(device)

        logger.info(
            "SlemMapper initialized: target_vocab=%d, draft_vocab=%d, "
            "draft→target 1:1=%d (%.1f%%), target→draft 1:1=%d (%.1f%%)",
            target_vocab_size,
            draft_vocab_size,
            mapped_d2t,
            100.0 * mapped_d2t / draft_vocab_size,
            mapped_t2d,
            100.0 * mapped_t2d / target_vocab_size,
        )

    def map_draft_to_target_ids(
        self,
        draft_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Map draft token IDs to target vocab via STRING-LEVEL re-encoding.

        This is the core SLEM operation: decode the draft token sequence to
        text, then re-encode with the target tokenizer. The result is padded
        or truncated to match the input shape.

        Args:
            draft_token_ids: [batch_size, num_draft_tokens] in draft vocab

        Returns:
            target_token_ids: [batch_size, num_draft_tokens] in target vocab
        """
        batch_size, num_draft_tokens = draft_token_ids.shape
        device = draft_token_ids.device

        result = torch.full(
            (batch_size, num_draft_tokens),
            self.target_eos_id,
            dtype=torch.int64,
            device=device,
        )

        draft_ids_cpu = draft_token_ids.cpu().tolist()

        for i in range(batch_size):
            seq = draft_ids_cpu[i]

            # Find effective length (strip trailing EOS padding)
            eff_len = num_draft_tokens
            for j in range(num_draft_tokens - 1, -1, -1):
                if seq[j] != self.draft_eos_id:
                    eff_len = j + 1
                    break
            else:
                eff_len = 0

            if eff_len == 0:
                continue

            # Decode draft tokens to text (string-level)
            text = self.draft_tokenizer.decode(seq[:eff_len], skip_special_tokens=False)

            if not text:
                continue

            # Re-encode with target tokenizer
            target_ids = self.target_tokenizer.encode(text, add_special_tokens=False)

            if not target_ids:
                continue

            # Fill result, truncating if needed
            fill_len = min(len(target_ids), num_draft_tokens)
            result[i, :fill_len] = torch.tensor(
                target_ids[:fill_len], dtype=torch.int64, device=device
            )

        return result

    def map_target_to_draft_ids(
        self,
        target_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Map target token IDs to draft vocab for feeding context to draft.

        Uses the 1:1 lookup table for efficiency. This is used in
        set_inputs_first_pass where target token IDs (from accepted tokens)
        need to be converted to draft vocab to feed the draft model's next
        forward pass. The draft model's KV cache already contains the correct
        context from its own generation, so this mapping only affects the
        input_ids for the first token of the next proposal round (the last
        accepted token). A 1:1 lookup is acceptable here because:
        - For tokens with clean 1:1 mapping (majority), it's exact
        - For tokens without 1:1 mapping, the draft model will still generate
          reasonable continuations (it just sees a slightly different token
          for context), and any errors are caught by the target model's
          verification step
        """
        return self._target_to_draft_table[target_token_ids.long()]
