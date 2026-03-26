# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_SPACE_PREFIXES = ("\u0120", "\u2581")

def _normalize_token(token: str) -> str:
    for prefix in _SPACE_PREFIXES:
        if token.startswith(prefix):
            return " " + token[len(prefix):]
    return token

class VocabMapping:
    def __init__(self, target_tokenizer, draft_tokenizer, target_vocab_size, draft_vocab_size, device):
        self.target_vocab_size = target_vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.device = device
        self.target_unk_token_id = getattr(target_tokenizer, "unk_token_id", None) or 0
        self.draft_unk_token_id = getattr(draft_tokenizer, "unk_token_id", None) or 0

        target_vocab = target_tokenizer.get_vocab()
        draft_vocab = draft_tokenizer.get_vocab()

        target_normalized = {}
        for token, tid in target_vocab.items():
            norm = _normalize_token(token)
            if norm not in target_normalized:
                target_normalized[norm] = tid

        draft_normalized = {}
        for token, tid in draft_vocab.items():
            norm = _normalize_token(token)
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

        logger.info("VocabMapping initialized: target_vocab=%d, draft_vocab=%d, intersection=%d (%.1f%% of draft, %.1f%% of target)",
            target_vocab_size, draft_vocab_size, self.intersection_size,
            100.0 * self.intersection_size / max(draft_vocab_size, 1),
            100.0 * self.intersection_size / max(target_vocab_size, 1))

        if self.intersection_size < 100:
            logger.warning("Very small vocabulary intersection (%d tokens).", self.intersection_size)

    def map_target_to_draft_ids(self, target_ids):
        draft_ids = self.target_to_draft_ids[target_ids]
        not_in_intersection = draft_ids == -1
        draft_ids = draft_ids.clone()
        draft_ids[not_in_intersection] = self.draft_unk_token_id
        return draft_ids.to(target_ids.dtype)

    def map_draft_to_target_ids(self, draft_ids):
        target_ids = self.draft_to_target_ids[draft_ids]
        not_in_intersection = target_ids == -1
        target_ids = target_ids.clone()
        target_ids[not_in_intersection] = self.target_unk_token_id
        return target_ids.to(draft_ids.dtype)

    def constrain_draft_logits(self, logits):
        logits = logits.clone()
        logits[:, ~self.intersection_mask_draft] = float("-inf")
        return logits
