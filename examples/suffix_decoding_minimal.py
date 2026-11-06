# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Minimal, self-contained demonstration of suffix decoding ideas without external
dependencies. This is a toy version that:
- Uses the last up-to max_tree_depth tokens as the pattern.
- Finds all occurrences of that pattern's suffixes in the existing context.
- Chooses the next token greedily by frequency among matches.
- Extends the draft sequence while limits are respected:
  * max number of speculative tokens
  * max_spec_factor * matched_prefix_len
  * min_token_prob threshold for the chosen next token

Run:
  python examples/suffix_decoding_minimal.py
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence


class ToySuffixDecoding:
    def __init__(
        self,
        num_speculative_tokens: int = 8,
        max_tree_depth: int = 24,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
    ):
        self.num_speculative_tokens = num_speculative_tokens
        self.max_tree_depth = max_tree_depth
        self.max_spec_factor = max_spec_factor
        self.min_token_prob = min_token_prob

    def _find_matches_with_next_tokens(
        self, context: Sequence[str], pattern: Sequence[str]
    ) -> list[str]:
        """
        Find all positions where 'pattern' appears in 'context', and collect
        the token that follows that occurrence (if any).
        Returns the list of "next tokens" that followed each match.
        """
        next_tokens: list[str] = []
        n = len(context)
        m = len(pattern)
        if m == 0 or n == 0 or m > n:
            return next_tokens
        for i in range(n - m + 1):
            if context[i : i + m] == list(pattern) and i + m < n:
                next_tokens.append(context[i + m])
        return next_tokens

    def _best_next_token(
        self, candidates: Iterable[str]
    ) -> tuple[str | None, float, int]:
        """
        Pick the most frequent token among candidates and return:
        (token, probability, total_count)
        """
        counts = Counter(candidates)
        if not counts:
            return None, 0.0, 0
        token, count = counts.most_common(1)[0]
        total = sum(counts.values())
        prob = count / total if total > 0 else 0.0
        return token, prob, total

    def find_next_token_stats(
        self, context: Sequence[str], pattern: Sequence[str]
    ) -> tuple[list[str], str | None, float, int]:
        """
        Public helper to expose (candidates, best_token, prob, total_samples)
        for demos without accessing protected members.
        """
        candidates = self._find_matches_with_next_tokens(context, pattern)
        token, prob, total = self._best_next_token(candidates)
        return candidates, token, prob, total

    def propose(self, context: list[str]) -> list[str]:
        """
        Produce a dynamic-length draft continuation from the given context.
        Greedy, frequency-based extension as a minimal stand-in for the full
        suffix decoding algorithm.
        """
        if not context:
            return []

        # Use the last up-to max_tree_depth tokens as the base pattern window.
        window_start = max(0, len(context) - self.max_tree_depth)
        window = context[window_start:]

        # We try suffixes of the window from longest to shortest
        # and grow a continuation greedily.
        for suffix_len in range(len(window), 0, -1):
            pattern = window[-suffix_len:]

            # Compute the max number of tokens allowed by spec factor for this match.
            spec_limit_by_factor = int(self.max_spec_factor * suffix_len)
            max_allowed = min(self.num_speculative_tokens, max(spec_limit_by_factor, 0))
            if max_allowed <= 0:
                continue

            draft: list[str] = []
            working_context = list(context)
            working_pattern = list(pattern)
            steps = 0

            while steps < max_allowed:
                # Find all next-token candidates following the current pattern.
                next_tokens = self._find_matches_with_next_tokens(
                    working_context, working_pattern
                )
                token, prob, _ = self._best_next_token(next_tokens)
                if token is None or prob < self.min_token_prob:
                    break

                # Append the chosen token and update state.
                draft.append(token)
                working_context.append(token)
                working_pattern = (working_pattern + [token])[-len(working_pattern) :]
                steps += 1

            if draft:
                return draft

        return []


def _step_by_step_demo():
    """
    Walk through a tiny example with clear prints.
    """
    toy = ToySuffixDecoding(
        num_speculative_tokens=5,
        max_tree_depth=6,
        max_spec_factor=1.5,
        min_token_prob=0.34,  # require the most frequent next-token to have >= 34%
    )

    # Example "tokenization": simple whitespace split
    # Prompt + some generated tokens so far:
    context = ["the", "cat", "sat", "on", "the", "mat", "the", "cat"]
    print("Context:", context)
    print("Parameters:")
    print("  num_speculative_tokens =", toy.num_speculative_tokens)
    print("  max_tree_depth         =", toy.max_tree_depth)
    print("  max_spec_factor        =", toy.max_spec_factor)
    print("  min_token_prob         =", toy.min_token_prob)
    print()

    # Compute the pattern window
    window_start = max(0, len(context) - toy.max_tree_depth)
    window = context[window_start:]
    print("Pattern search window (last max_tree_depth tokens):", window)
    print()

    # Try suffixes from longest to shortest, and show what happens.
    for suffix_len in range(len(window), 0, -1):
        pattern = window[-suffix_len:]
        spec_limit_by_factor = int(toy.max_spec_factor * suffix_len)
        max_allowed = min(toy.num_speculative_tokens, max(spec_limit_by_factor, 0))
        print(f"Trying suffix length = {suffix_len}")
        print("  Pattern:", pattern)
        print("  Max speculative tokens allowed by limits:", max_allowed)
        if max_allowed <= 0:
            print("  -> Skipped (no allowance)\n")
            continue

        # Gather candidates for the next token
        next_tokens, token, prob, total = toy.find_next_token_stats(context, pattern)
        print("  Next-token candidates after matches:", next_tokens)
        print(f"  Best next token: {token!r} with prob={prob:.2f} over {total} samples")
        if token is None or prob < toy.min_token_prob:
            print("  -> No viable token at this suffix length (below threshold)\n")
            continue

        # Greedy grow the draft
        draft: list[str] = []
        working_context = list(context)
        working_pattern = list(pattern)
        steps = 0
        while steps < max_allowed:
            candidates, t, p, _ = toy.find_next_token_stats(
                working_context, working_pattern
            )
            print(
                f"    Step {steps}: candidates={candidates}, "
                f"chosen={t!r}, p={p:.2f}"
            )
            if t is None or p < toy.min_token_prob:
                print("    Stop: no candidate or below prob threshold.")
                break
            draft.append(t)
            working_context.append(t)
            working_pattern = (working_pattern + [t])[-len(working_pattern) :]
            steps += 1

        if draft:
            print("\nDraft proposed tokens:", draft)
            return
        else:
            print("  -> Could not extend draft at this suffix length.\n")

    print("No speculative tokens proposed.")


if __name__ == "__main__":
    _step_by_step_demo()
    print("\n---- Probability variation demo ----")
    # Minimal demonstration where probability < 1.0:
    # Pattern ["a","b"] is followed by both "x" and "y" across the context,
    # so the most frequent next token will have prob 2/3 ~= 0.67.
    toy_demo = ToySuffixDecoding(
        num_speculative_tokens=3,
        max_tree_depth=6,
        max_spec_factor=2.0,
        min_token_prob=0.0,
    )
    ambiguous_context = ["a", "b", "x", "a", "b", "y", "a", "b", "x"]
    demo_pattern = ["a", "b"]
    print("Context:", ambiguous_context)
    print("Pattern:", demo_pattern)
    demo_candidates, demo_token, demo_prob, demo_total = toy_demo.find_next_token_stats(
        ambiguous_context, demo_pattern
    )
    print("Next-token candidates after matches:", demo_candidates)
    print(
        f"Chosen next token: {demo_token!r} with prob={demo_prob:.2f} "
        f"over {demo_total} samples"
    )


