# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Token-Level Intersection (TLI) vocabulary mapping for heterogeneous-vocab
speculative decoding.

Based on Algorithm 4 from:
  "Accelerating LLM Inference with Lossless Speculative Decoding Algorithms
   for Heterogeneous Vocabularies" — Timor et al., ICML 2025.
  https://arxiv.org/abs/2502.05202
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _detect_space_prefix(tokenizer) -> tuple[str, ...]:
    """Detect the space-prefix character(s) by tokenizing a literal space.

    Different tokenizer families mark word-initial spaces differently:
    BPE uses 'G' (U+0120), SentencePiece uses '_' (U+2581). Probing at
    runtime avoids hardcoding assumptions.
    """
    try:
        space_ids = tokenizer.encode(" a", add_special_tokens=False)
        if space_ids:
            tok_str = tokenizer.convert_ids_to_tokens(space_ids[0])
            if (
                isinstance(tok_str, str)
                and len(tok_str) > 1
                and tok_str.endswith("a")
                and tok_str[0] not in (" ", "\t")
            ):
                return (tok_str[:-1],)
    except Exception:
        pass
    return ("Ġ", "▁")


def _normalize_token(token: str, space_prefixes: tuple[str, ...]) -> str:
    for prefix in space_prefixes:
        if token.startswith(prefix):
            return " " + token[len(prefix) :]
    return token


def _get_unk_token_id(tokenizer, role: str) -> int:
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None:
        return unk
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        logger.warning(
            "VocabMapping: %s has no unk_token_id; "
            "falling back to eos_token_id=%d for out-of-intersection tokens",
            role,
            eos,
        )
        return eos
    raise ValueError(
        f"VocabMapping: {role} has neither unk_token_id nor eos_token_id; "
        "cannot safely map out-of-intersection tokens"
    )


class VocabMapping:
    def __init__(
        self,
        target_tokenizer,
        draft_tokenizer,
        target_vocab_size: int,
        draft_vocab_size: int,
        device: torch.device,
    ):
        self.target_vocab_size = target_vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.device = device
        self.target_unk_token_id = _get_unk_token_id(
            target_tokenizer, "target tokenizer"
        )
        self.draft_unk_token_id = _get_unk_token_id(draft_tokenizer, "draft tokenizer")

        target_prefixes = _detect_space_prefix(target_tokenizer)
        draft_prefixes = _detect_space_prefix(draft_tokenizer)

        target_vocab = target_tokenizer.get_vocab()
        draft_vocab = draft_tokenizer.get_vocab()

        target_normalized: dict[str, int] = {}
        for token, tid in target_vocab.items():
            norm = _normalize_token(token, target_prefixes)
            if norm not in target_normalized:
                target_normalized[norm] = tid

        draft_normalized: dict[str, int] = {}
        for token, tid in draft_vocab.items():
            norm = _normalize_token(token, draft_prefixes)
            if norm not in draft_normalized:
                draft_normalized[norm] = tid

        common_tokens = set(target_normalized.keys()) & set(draft_normalized.keys())

        draft_to_target = torch.full((draft_vocab_size,), -1, dtype=torch.long)
        target_to_draft = torch.full((target_vocab_size,), -1, dtype=torch.long)
        intersection_mask_draft = torch.zeros(draft_vocab_size, dtype=torch.bool)

        for norm_token in common_tokens:
            t_id = target_normalized[norm_token]
            d_id = draft_normalized[norm_token]
            if t_id < target_vocab_size and d_id < draft_vocab_size:
                draft_to_target[d_id] = t_id
                target_to_draft[t_id] = d_id
                intersection_mask_draft[d_id] = True

        self.draft_to_target_ids = draft_to_target.to(device)
        self.target_to_draft_ids = target_to_draft.to(device)
        self.intersection_mask_draft = intersection_mask_draft.to(device)
        self.intersection_size = int(intersection_mask_draft.sum().item())

        logger.info(
            "VocabMapping initialized: target_vocab=%d, draft_vocab=%d, "
            "intersection=%d (%.1f%% of draft, %.1f%% of target)",
            target_vocab_size,
            draft_vocab_size,
            self.intersection_size,
            100.0 * self.intersection_size / max(draft_vocab_size, 1),
            100.0 * self.intersection_size / max(target_vocab_size, 1),
        )

        if self.intersection_size < 100:
            logger.warning(
                "Very small vocabulary intersection (%d tokens).",
                self.intersection_size,
            )

    def map_target_to_draft_ids(self, target_ids: torch.Tensor) -> torch.Tensor:
        draft_ids = self.target_to_draft_ids[target_ids]
        missing = draft_ids == -1
        if missing.any():
            draft_ids = draft_ids.clone()
            draft_ids[missing] = self.draft_unk_token_id
        return draft_ids.to(target_ids.dtype)

    def map_draft_to_target_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        target_ids = self.draft_to_target_ids[draft_ids]
        missing = target_ids == -1
        if missing.any():
            target_ids = target_ids.clone()
            target_ids[missing] = self.target_unk_token_id
        return target_ids.to(draft_ids.dtype)

    def constrain_draft_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~self.intersection_mask_draft, float("-inf"))
