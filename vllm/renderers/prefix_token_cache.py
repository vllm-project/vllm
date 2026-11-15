# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token ids of recent long prompts, reused by the next turn of a conversation.

A multi-turn (agentic) chat request carries the whole conversation, and the next
turn's rendered prompt starts with the previous turn's text. An HF fast tokenizer
splits its added tokens (the chat template's special tokens) out of the text
before it normalizes and pre-tokenizes the pieces between them, so the ids of a
text equal the ids of its pieces cut right after any added token, concatenated.
This cache keeps, for each recent long prompt, its ids and a rolling digest of the
text at every such cut; a new prompt reuses the ids up to its deepest cut that a
cached prompt shares and encodes only the rest, piece by piece.

Measured on 600 InferenceX AgentX requests (MiniMax-M3.1, prompts p50 479K
characters / 157K tokens): encode p50 152 -> 9 ms, p90 525 -> 103 ms, ids equal on
every request; 98% of the characters are reused at the median.

Used only when every added token matches the raw text (no ``lstrip`` / ``rstrip``
/ ``single_word`` / ``normalized``), for prompts of at least ``MIN_CHARS`` encoded
without special tokens and without offsets. The first ``VERIFY_FIRST`` prompts it
serves are also encoded whole and compared; a mismatch turns it off for the
process.
"""

import array
import hashlib
import re
import threading
from collections import OrderedDict

from vllm.logger import init_logger

logger = init_logger(__name__)

MIN_CHARS = 16384  # shorter prompts encode in a few ms
MAX_TOKENS = 32 << 20  # cached ids, 4 bytes each (128 MiB)
VERIFY_FIRST = 8


class PrefixTokenCache:
    def __init__(self, backend, pattern: re.Pattern) -> None:
        self._backend = backend  # a private tokenizers.Tokenizer, no truncation
        self._pattern = pattern
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, tuple[array.array, list[bytes]]] = OrderedDict()
        self._by_digest: dict[bytes, tuple[int, int]] = {}
        self._tokens = 0
        self._next_id = 0
        self._verify_left = VERIFY_FIRST
        self.enabled = True

    @classmethod
    def maybe_create(cls, tokenizer) -> "PrefixTokenCache | None":
        """A cache for an HF fast tokenizer whose added tokens match the raw
        text, else None."""
        try:
            from tokenizers import Tokenizer
        except ImportError:
            return None
        backend = getattr(tokenizer, "_tokenizer", None)
        if not isinstance(backend, Tokenizer) or getattr(
            tokenizer, "split_special_tokens", False
        ):
            return None
        added = list(backend.get_added_tokens_decoder().values())
        if not added or any(
            a.lstrip or a.rstrip or a.single_word or a.normalized for a in added
        ):
            return None
        own = Tokenizer.from_str(backend.to_str())
        own.no_truncation()
        own.no_padding()
        contents = sorted({a.content for a in added}, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(c) for c in contents))
        logger.info(
            "Prefix token cache on for prompts of %d+ characters (%d added tokens)",
            MIN_CHARS,
            len(contents),
        )
        return cls(own, pattern)

    def applies(self, text: str) -> bool:
        return self.enabled and len(text) >= MIN_CHARS

    def encode(self, text: str) -> list[int]:
        """The ids of ``text`` encoded without special tokens."""
        cuts = [m.end() for m in self._pattern.finditer(text)]
        h = hashlib.blake2b(digest_size=16)
        digests, prev = [], 0
        for c in cuts:
            h.update(text[prev:c].encode())
            digests.append(h.copy().digest())
            prev = c
        best, head = -1, None
        with self._lock:
            for i in range(len(digests) - 1, -1, -1):
                hit = self._by_digest.get(digests[i])
                if hit is not None:
                    best = i
                    self._entries.move_to_end(hit[0])
                    head = self._entries[hit[0]][0][: hit[1]]
                    break
        ids = head if head is not None else array.array("i")
        start = cuts[best] if best >= 0 else 0
        pieces, ends, p = [], [], start
        for c in cuts[best + 1 :] + [len(text)]:
            if c > p:
                pieces.append(text[p:c])
                ends.append(c)
            p = c
        counts: dict[int, int] = {}
        k = best + 1
        if pieces:
            for enc, end in zip(
                self._backend.encode_batch(pieces, add_special_tokens=False), ends
            ):
                ids.extend(enc.ids)
                while k < len(cuts) and cuts[k] <= end:
                    counts[k] = len(ids)
                    k += 1
        if self._verify_left > 0:
            whole = self._backend.encode(text, add_special_tokens=False).ids
            with self._lock:
                self._verify_left -= 1
            if whole != ids.tolist():
                self.enabled = False
                logger.warning(
                    "Prefix token cache off: ids of a %d-character prompt differ "
                    "from a whole encode",
                    len(text),
                )
                return whole
        with self._lock:
            for i in range(best + 1):  # the reused cuts now point at this prompt
                old = self._by_digest.get(digests[i])
                if old is not None:
                    counts[i] = old[1]
            eid = self._next_id
            self._next_id += 1
            self._entries[eid] = (ids, [digests[i] for i in counts])
            self._tokens += len(ids)
            for i, n in counts.items():
                self._by_digest[digests[i]] = (eid, n)
            while self._tokens > MAX_TOKENS and len(self._entries) > 1:
                old_id, (old_ids, old_digests) = self._entries.popitem(last=False)
                self._tokens -= len(old_ids)
                for d in old_digests:
                    if self._by_digest.get(d, (None,))[0] == old_id:
                        del self._by_digest[d]
        return ids.tolist()
