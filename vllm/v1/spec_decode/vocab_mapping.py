# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _detect_space_prefix(tokenizer) -> tuple[str, ...]:
    """Detect the space-prefix character(s) by tokenizing a literal space.

    Different tokenizer families mark word-initial spaces differently:
    BPE uses 'Ġ' (U+0120), SentencePiece uses '▁' (U+2581). Probing at
    runtime avoids hardcoding assumptions and correctly handles mixed-family
    pairs (e.g. BPE draft + SentencePiece target).
    """
    try:
        space_ids = tokenizer.encode(" ", add_special_tokens=False)
        if space_ids:
            tok_str = tokenizer.convert_ids_to_tokens(space_ids[0])
            if isinstance(tok_str, str) and len(tok_str) > 1 \
                    and tok_str[0] not in (" ", "\t"):
                return (tok_str[0],)
    except Exception:
        pass
    # Fallback: cover both BPE (Ġ U+0120) and SentencePiece (▁ U+2581)
    return ("\u0120", "\u2581")


def _normalize_token(token: str, space_prefixes: tuple[str, ...]) -> str:
    for prefix in space_prefixes:
        if token.startswith(prefix):
            return " " + token[len(prefix):]
    return token


def _get_unk_token_id(tokenizer, role: str) -> int:
    """Return a safe fallback token ID for out-of-intersection tokens.

    Preferred: unk_token_id → eos_token_id → ValueError.
    Checking with ``is not None`` is required because token ID 0 is a valid
    (and common) unk ID on many tokenizers; using ``or 0`` would silently
    mis-handle those cases.
    """
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None:
        return unk
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        logger.warning(
            "VocabMapping: %s has no unk_token_id; "
            "falling back to eos_token_id=%d for out-of-intersection tokens",
            role, eos,
        )
        return eos
    raise ValueError(
        f"VocabMapping: {role} has neither unk_token_id nor eos_token_id; "
        "cannot safely map out-of-intersection tokens"
    )


class VocabMapping:
    def __init__(self, target_tokenizer, draft_tokenizer,
                 target_vocab_size, draft_vocab_size, device):
        self.target_vocab_size = target_vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.device = device
        self.target_unk_token_id = _get_unk_token_id(target_tokenizer,
                                                      "target tokenizer")
        self.draft_unk_token_id  = _get_unk_token_id(draft_tokenizer,
                                                      "draft tokenizer")

        target_prefixes = _detect_space_prefix(target_tokenizer)
        draft_prefixes  = _detect_space_prefix(draft_tokenizer)

        target_vocab = target_tokenizer.get_vocab()
        draft_vocab  = draft_tokenizer.get_vocab()

        target_normalized = {}
        for token, tid in target_vocab.items():
            norm = _normalize_token(token, target_prefixes)
            if norm not in target_normalized:
                target_normalized[norm] = tid

        draft_normalized = {}
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
            target_vocab_size, draft_vocab_size, self.intersection_size,
            100.0 * self.intersection_size / max(draft_vocab_size, 1),
            100.0 * self.intersection_size / max(target_vocab_size, 1),
        )

        if self.intersection_size < 100:
            logger.warning(
                "Very small vocabulary intersection (%d tokens).",
                self.intersection_size,
            )

    def map_target_to_draft_ids(self, target_ids):
        draft_ids = self.target_to_draft_ids[target_ids]  # new tensor; no clone needed
        missing = draft_ids == -1
        if missing.any():
            draft_ids[missing] = self.draft_unk_token_id
        return draft_ids.to(target_ids.dtype)

    def map_draft_to_target_ids(self, draft_ids):
        target_ids = self.draft_to_target_ids[draft_ids]  # new tensor; no clone needed
        missing = target_ids == -1
        if missing.any():
            target_ids[missing] = self.target_unk_token_id
        return target_ids.to(draft_ids.dtype)

    def constrain_draft_logits(self, logits):
        # masked_fill returns a new tensor; no clone needed
        return logits.masked_fill(~self.intersection_mask_draft, float("-inf"))
