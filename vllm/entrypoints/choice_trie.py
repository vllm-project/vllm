# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-level prefix trie for constrained beam search over a fixed choice set.

Built once per request from the root ChoiceTrie.  At each beam step, each beam
re-walks the trie from the root over its generated tokens to find its current
node.  This is O(generation_length) per beam per step, but generation is short
(tool names are at most ~5 tokens) and trie lookup is pure dict access, so it
is fast in practice and eliminates all state-sync bugs that come from
maintaining a forward pointer across beam forks.

Each choice has an EOS token appended in the trie, mirroring the xgrammar
backend behaviour: after all choice tokens are emitted, the only allowed next
token is EOS.  The beam then terminates via the normal EOS-handling path with
the correct cumulative log probability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike


class _TrieNode:
    __slots__ = ("children", "is_terminal")

    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.is_terminal: bool = False


class ChoiceTrie:
    """Token-level prefix trie over a fixed list of string choices.

    Build once per request with :meth:`build`.  At each beam step call
    :meth:`allowed_tokens_for` with the beam's generated token IDs to get the
    set of valid next tokens — no per-beam state, no pointer to maintain.

    The EOS token is appended to each choice sequence in the trie so that
    completed choices produce EOS as their final token, consistent with the
    xgrammar backend.
    """

    def __init__(self, root: _TrieNode) -> None:
        self._root = root

    @staticmethod
    def build(
        choices: list[str],
        tokenizer: TokenizerLike,
        eos_token_id: int | None = None,
    ) -> ChoiceTrie:
        """Build a trie from *choices* using *tokenizer* to tokenize each one.

        Uses a newline prefix so SentencePiece produces continuation-context
        token IDs (no spurious leading space that would appear when tokenizing
        a string in isolation).

        If *eos_token_id* is provided, it is appended to each choice sequence
        in the trie so that beam search terminates choices via the normal EOS
        path — identical to how xgrammar and constrained.py handle terminals.
        """
        _PREFIX = "\n"
        prefix_ids: list[int] = tokenizer.encode(_PREFIX, add_special_tokens=False)
        prefix_len = len(prefix_ids)

        root = _TrieNode()

        for choice in choices:
            combined_ids: list[int] = tokenizer.encode(
                _PREFIX + choice, add_special_tokens=False
            )
            token_ids = combined_ids[prefix_len:]
            if not token_ids:
                continue
            if eos_token_id is not None:
                token_ids = token_ids + [eos_token_id]
            node = root
            for tid in token_ids:
                if tid not in node.children:
                    node.children[tid] = _TrieNode()
                node = node.children[tid]
            node.is_terminal = True

        return ChoiceTrie(root)

    def allowed_tokens_for(self, generated_ids: list[int]) -> list[int] | None:
        """Return allowed next token IDs given the beam's generated tokens.

        Re-walks the trie from the root over *generated_ids* each call.
        Returns ``None`` when the sequence has reached a terminal (all choice
        tokens plus EOS have been emitted).
        Returns ``[]`` if the generated prefix is not in the trie at all
        (should not happen during constrained beam search).
        """
        node = self._root
        for tid in generated_ids:
            child = node.children.get(tid)
            if child is None:
                return []
            node = child

        if node.is_terminal:
            return None

        return list(node.children.keys())
