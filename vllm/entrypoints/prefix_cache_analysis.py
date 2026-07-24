# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline, no-GPU analysis of prefix-cache friendliness for a request
dataset.

This mirrors the full-block hash-chain construction vLLM's V1 scheduler
uses for automatic prefix caching (see ``vllm/v1/core/kv_cache_utils.py``),
so the reported cacheability estimates match production block-hash
behavior rather than an approximation of it.

v1 scope (see https://github.com/vllm-project/vllm/issues/47993):
  - Plain-prompt JSONL input only. OpenAI chat/batch JSONL is a
    follow-up once the reporting shape here is agreed on.
  - No LoRA / multimodal / cache-salt extra-key support yet. Requests
    are hashed as plain token sequences.
  - This computes an offline upper-bound on cacheability. It does not
    model eviction, memory pressure, request arrival order, or
    disaggregated/KV-offloaded serving.

Open design question carried over from the RFC: ``hash_block_tokens``
and friends live under ``vllm/v1/core`` today, which is engine-internal.
This module imports them directly for a first version rather than
waiting on a new public helper -- see the RFC thread before extending
this beyond the CLI use case.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vllm.tokenizers import get_tokenizer
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import BlockHash, hash_block_tokens, init_none_hash


@dataclass
class PromptRecord:
    """One parsed line from the input JSONL."""

    request_id: str
    text: str


@dataclass
class PrefixGroup:
    """A set of requests that share a common full-block hash-chain prefix."""

    shared_block_hashes: tuple[BlockHash, ...]
    request_ids: list[str] = field(default_factory=list)

    @property
    def shared_prefix_tokens(self) -> int:
        return len(self.shared_block_hashes)  # multiplied by block_size by caller


@dataclass
class AnalysisReport:
    block_size: int
    num_requests: int
    total_prompt_tokens: int
    total_full_block_tokens: int
    estimated_reusable_full_block_tokens: int
    top_prefix_groups: list[PrefixGroup]

    @property
    def cacheability_ratio(self) -> float:
        if self.total_full_block_tokens == 0:
            return 0.0
        return self.estimated_reusable_full_block_tokens / self.total_full_block_tokens

    def to_dict(self) -> dict:
        return {
            "block_size": self.block_size,
            "num_requests": self.num_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_full_block_tokens": self.total_full_block_tokens,
            "estimated_reusable_full_block_tokens": (
                self.estimated_reusable_full_block_tokens
            ),
            "cacheability_ratio": round(self.cacheability_ratio, 4),
            "top_prefix_groups": [
                {
                    "shared_full_blocks": len(g.shared_block_hashes),
                    "shared_tokens": len(g.shared_block_hashes) * self.block_size,
                    "num_requests": len(g.request_ids),
                    "request_ids": g.request_ids,
                }
                for g in self.top_prefix_groups
            ],
        }

    def render_text(self) -> str:
        lines = [
            f"requests analyzed        : {self.num_requests}",
            f"block size                : {self.block_size}",
            f"total prompt tokens       : {self.total_prompt_tokens}",
            f"total full-block tokens   : {self.total_full_block_tokens}",
            f"est. reusable full blocks : {self.estimated_reusable_full_block_tokens}",
            f"est. cacheability ratio   : {self.cacheability_ratio:.2%}",
            "",
            "top shared-prefix groups (offline upper bound, not a hit-rate guarantee):",
        ]
        if not self.top_prefix_groups:
            lines.append("  (no requests shared a full block-aligned prefix)")
        for i, group in enumerate(self.top_prefix_groups, start=1):
            shared_tokens = len(group.shared_block_hashes) * self.block_size
            lines.append(
                f"  {i}. {len(group.request_ids)} requests share "
                f"{shared_tokens} full-block tokens "
                f"({len(group.shared_block_hashes)} blocks) "
                f"-> {', '.join(group.request_ids[:5])}"
                + (" ..." if len(group.request_ids) > 5 else "")
            )
        return "\n".join(lines)


def load_plain_prompt_jsonl(path: str | Path) -> list[PromptRecord]:
    """Parse a JSONL file of plain-prompt requests.

    Each line must be a JSON object with a ``prompt`` string field and an
    optional ``id`` field (defaults to the 0-indexed line number).
    """
    records: list[PromptRecord] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                raise ValueError(
                    f"{path}:{line_no + 1}: missing required 'prompt' field"
                )
            records.append(
                PromptRecord(
                    request_id=str(obj.get("id", line_no)),
                    text=obj["prompt"],
                )
            )
    return records


def _full_block_hash_chain(
    token_ids: Sequence[int],
    block_size: int,
    hash_fn,
) -> list[BlockHash]:
    """Compute the full-block hash chain for a token sequence, matching
    ``hash_block_tokens`` block-by-block. Partial trailing blocks are
    dropped, mirroring the engine's own full-block-only hashing.
    """
    chain: list[BlockHash] = []
    parent: BlockHash | None = None
    num_full_blocks = len(token_ids) // block_size
    for i in range(num_full_blocks):
        start = i * block_size
        end = start + block_size
        block_hash = hash_block_tokens(hash_fn, parent, token_ids[start:end])
        chain.append(block_hash)
        parent = block_hash
    return chain


class _TrieNode:
    """One node in the shared-prefix trie built by
    ``_group_requests_by_shared_prefix``. ``block_hash``/``parent`` identify
    the edge this node was reached by, so a winning node's full prefix can be
    reconstructed by walking parents back to the root -- without needing
    every node to carry its own path.
    """

    __slots__ = ("children", "request_ids", "parent", "block_hash")

    def __init__(
        self,
        parent: _TrieNode | None = None,
        block_hash: BlockHash | None = None,
    ) -> None:
        self.children: dict[BlockHash, _TrieNode] = {}
        self.request_ids: list[str] = []
        self.parent = parent
        self.block_hash = block_hash


def _path_to_node(node: _TrieNode) -> tuple[BlockHash, ...]:
    path: list[BlockHash] = []
    while node.parent is not None:
        assert node.block_hash is not None
        path.append(node.block_hash)
        node = node.parent
    path.reverse()
    return tuple(path)


def _group_requests_by_shared_prefix(
    chains: dict[str, list[BlockHash]], top_k_groups: int
) -> list[PrefixGroup]:
    """Find the top-K largest shared hash-chain prefixes across requests.

    Builds a trie over the chains in a single O(total blocks) pass -- each
    chain step is one dict lookup/insert -- instead of materializing a
    ``tuple(chain[:depth])`` key for every depth of every chain, which is
    O(chain_length ** 2) per request. That quadratic cost lands hardest on
    exactly the workload this tool exists to highlight: a long shared system
    prompt or tool schema reused across many requests (e.g. an 8k-token
    shared prefix at block_size=16 is ~512 blocks; at 10k requests the old
    approach did on the order of 10_000 * 512**2 ~= 2.6B tuple-element
    operations for grouping alone).

    A winning group's actual ``shared_block_hashes`` tuple is only
    reconstructed (via ``_path_to_node``) for the at-most-``top_k_groups``
    nodes that survive selection, not for every trie node.
    """
    root = _TrieNode()
    for request_id, chain in chains.items():
        node = root
        for block_hash in chain:
            child = node.children.get(block_hash)
            if child is None:
                child = _TrieNode(parent=node, block_hash=block_hash)
                node.children[block_hash] = child
            node = child
            node.request_ids.append(request_id)

    # Depth is tracked as a plain counter alongside each stack entry, not by
    # materializing a path, so this collection pass is O(total trie nodes)
    # <= O(total blocks).
    candidates: list[tuple[int, _TrieNode]] = []
    stack: list[tuple[_TrieNode, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        # Each request_id is appended at most once per node (chains cannot
        # revisit a node -- block hashes are chained forward-only), so
        # request_ids is already duplicate-free here.
        if depth > 0 and len(node.request_ids) > 1:
            candidates.append((depth, node))
        for child in node.children.values():
            stack.append((child, depth + 1))

    # Prefer longer shared prefixes first, then more requests sharing them.
    # Ties (equal depth and count) are broken by the lexicographically
    # smallest request_id in the group, so selection is a deterministic
    # function of the input and doesn't depend on dict/stack iteration
    # order -- two runs on the same input always report the same winners.
    candidates.sort(
        key=lambda c: (-c[0], -len(c[1].request_ids), min(c[1].request_ids))
    )

    # Deduplicate so a longer prefix's requests aren't also reported at
    # every shorter prefix depth -- keep only maximal groups. This is a
    # display-only reduction: a request set can legitimately appear in more
    # than one *distinct* group (e.g. {a,b} at depth 5 and {a,b,c} at depth
    # 2 are both real, different-membership sharing clusters), so this only
    # collapses redundant same-membership entries at shorter depths.
    top_groups: list[PrefixGroup] = []
    covered: set[tuple[str, ...]] = set()
    for depth, node in candidates:
        request_ids = sorted(node.request_ids)
        key = tuple(request_ids)
        if key in covered:
            continue
        covered.add(key)
        top_groups.append(
            PrefixGroup(
                shared_block_hashes=_path_to_node(node),
                request_ids=request_ids,
            )
        )
        if len(top_groups) >= top_k_groups:
            break

    return top_groups


def _reusable_tokens_from_chains(
    chains: dict[str, list[BlockHash]], block_size: int
) -> int:
    """Estimate reusable full-block tokens across a set of hash chains.

    Because each block's hash is chained on its parent's hash
    (``hash_block_tokens(hash_fn, parent, tokens)``), a given ``BlockHash``
    value already uniquely identifies one node in the shared-prefix trie
    across every chain that reaches it. Counting reuse once per distinct
    ``BlockHash`` -- regardless of its depth -- gives the correct,
    non-overlapping total: a block shared by ``count`` requests is computed
    once and reused ``count - 1`` times.

    This must be computed independently of any maximal-prefix-group
    selection (e.g. for display purposes): two reported groups can share
    block-hash nodes (``{a,b}`` at depth 5 and ``{a,b,c}`` at depth 2 both
    include the same first two blocks for ``a`` and ``b``), so summing
    per-group full-prefix lengths would double-count that overlap.
    """
    block_hash_counts: dict[BlockHash, int] = defaultdict(int)
    for chain in chains.values():
        for block_hash in chain:
            block_hash_counts[block_hash] += 1
    return sum(
        block_size * (count - 1) for count in block_hash_counts.values() if count > 1
    )


def analyze(
    records: Iterable[PromptRecord],
    *,
    model: str,
    block_size: int,
    hash_algo: str = "sha256",
    trust_remote_code: bool = False,
    top_k_groups: int = 10,
) -> AnalysisReport:
    """Tokenize each record, compute its full-block hash chain, and group
    requests by shared chain prefixes.
    """
    tokenizer = get_tokenizer(model, trust_remote_code=trust_remote_code)
    hash_fn = get_hash_fn_by_name(hash_algo)
    init_none_hash(hash_fn)

    chains: dict[str, list[BlockHash]] = {}
    total_prompt_tokens = 0
    total_full_block_tokens = 0

    for record in records:
        if record.request_id in chains:
            raise ValueError(
                f"duplicate request_id {record.request_id!r} in input records -- "
                "each record must have a unique id, or reuse would be silently "
                "undercounted (the later record would overwrite the earlier "
                "one's chain after both had already been tokenized)"
            )
        token_ids = tokenizer.encode(record.text)
        total_prompt_tokens += len(token_ids)
        chain = _full_block_hash_chain(token_ids, block_size, hash_fn)
        total_full_block_tokens += len(chain) * block_size
        chains[record.request_id] = chain

    top_groups = _group_requests_by_shared_prefix(chains, top_k_groups)
    reusable_tokens = _reusable_tokens_from_chains(chains, block_size)

    return AnalysisReport(
        block_size=block_size,
        num_requests=len(chains),
        total_prompt_tokens=total_prompt_tokens,
        total_full_block_tokens=total_full_block_tokens,
        estimated_reusable_full_block_tokens=reusable_tokens,
        top_prefix_groups=top_groups,
    )
