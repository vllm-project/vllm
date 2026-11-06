# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the shadow grammar manager."""

import json

import pytest

xgr = pytest.importorskip("xgrammar")


class TestShadowGrammarState:
    """Test the shadow grammar state machine logic using xgrammar directly,
    without requiring full vllm initialization."""

    @pytest.fixture
    def simple_json_schema(self):
        return json.dumps(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            }
        )

    @pytest.fixture
    def compiler(self):
        """Create a grammar compiler with a simple tokenizer."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True,
        )
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=tokenizer.vocab_size
        )
        return xgr.GrammarCompiler(
            tokenizer_info, max_threads=1, cache_enabled=True
        ), tokenizer

    def test_matcher_basic_flow(self, compiler, simple_json_schema):
        """Test that a GrammarMatcher can accept, rollback, and fill bitmask."""
        grammar_compiler, tokenizer = compiler
        ctx = grammar_compiler.compile_json_schema(simple_json_schema)
        matcher = xgr.GrammarMatcher(ctx, max_rollback_tokens=7)

        # Encode a valid JSON prefix
        tokens = tokenizer.encode('{"name":', add_special_tokens=False)

        # Accept tokens one by one
        for t in tokens:
            assert matcher.accept_token(t), f"Token {t} rejected"

        # Rollback all
        matcher.rollback(len(tokens))

        # Re-accept — should work identically
        for t in tokens:
            assert matcher.accept_token(t), f"Token {t} rejected after rollback"

        # Rollback again
        matcher.rollback(len(tokens))

    def test_bitmask_constrains_tokens(self, compiler, simple_json_schema):
        """Test that fill_next_token_bitmask produces a valid constraint."""
        import torch

        grammar_compiler, tokenizer = compiler
        ctx = grammar_compiler.compile_json_schema(simple_json_schema)
        matcher = xgr.GrammarMatcher(ctx, max_rollback_tokens=7)

        # Allocate bitmask
        bitmask = xgr.allocate_token_bitmask(1, tokenizer.vocab_size)
        matcher.fill_next_token_bitmask(bitmask, 0)

        # Apply bitmask to uniform logits
        logits = torch.zeros(1, tokenizer.vocab_size)
        xgr.apply_token_bitmask_inplace(logits, bitmask)

        # The first valid token for a JSON object should be '{'
        valid_token = logits[0].argmax().item()
        decoded = tokenizer.decode([valid_token])
        assert "{" in decoded, f"Expected '{{' but got '{decoded}'"

    def test_draft_with_grammar_mask(self, compiler, simple_json_schema):
        """Simulate the full draft-with-grammar flow:
        fill bitmask → sample → accept → repeat → rollback all."""
        import torch

        grammar_compiler, tokenizer = compiler
        ctx = grammar_compiler.compile_json_schema(simple_json_schema)
        matcher = xgr.GrammarMatcher(ctx, max_rollback_tokens=7)

        bitmask = xgr.allocate_token_bitmask(1, tokenizer.vocab_size)
        num_draft_tokens = 0
        draft_tokens = []

        # Simulate k=5 draft steps with grammar constraint
        for _ in range(5):
            if matcher.is_terminated():
                break
            # Fill bitmask
            matcher.fill_next_token_bitmask(bitmask, 0)
            # Create logits and apply mask
            logits = torch.zeros(1, tokenizer.vocab_size)
            xgr.apply_token_bitmask_inplace(logits, bitmask)
            # "Sample" (greedy from masked logits)
            token_id = logits[0].argmax().item()
            # Accept in matcher
            assert matcher.accept_token(token_id), (
                f"Grammar rejected token {token_id} that passed its own mask"
            )
            draft_tokens.append(token_id)
            num_draft_tokens += 1

        # Rollback all draft tokens
        if num_draft_tokens > 0 and not matcher.is_terminated():
            matcher.rollback(num_draft_tokens)

        # Verify: all draft tokens should be re-acceptable
        # (validates that rollback worked correctly)
        for t in draft_tokens:
            assert matcher.accept_token(t), (
                f"Token {t} rejected after rollback — state inconsistency"
            )

        print(f"Draft tokens: {draft_tokens}")
        print(f"Decoded: {tokenizer.decode(draft_tokens)}")

    def test_validate_tokens_matches_masked_sampling(
        self, compiler, simple_json_schema
    ):
        """Key correctness test: tokens sampled under grammar mask
        must ALL pass validate_tokens (no truncation)."""
        import torch

        grammar_compiler, tokenizer = compiler
        ctx = grammar_compiler.compile_json_schema(simple_json_schema)

        # Create two matchers: one for drafting, one simulating scheduler
        draft_matcher = xgr.GrammarMatcher(ctx, max_rollback_tokens=7)
        scheduler_matcher = xgr.GrammarMatcher(ctx, max_rollback_tokens=7)

        bitmask = xgr.allocate_token_bitmask(1, tokenizer.vocab_size)
        draft_tokens = []

        # Generate k=7 draft tokens with grammar mask
        for _ in range(7):
            if draft_matcher.is_terminated():
                break
            draft_matcher.fill_next_token_bitmask(bitmask, 0)
            logits = torch.zeros(1, tokenizer.vocab_size)
            xgr.apply_token_bitmask_inplace(logits, bitmask)
            token_id = logits[0].argmax().item()
            assert draft_matcher.accept_token(token_id)
            draft_tokens.append(token_id)

        # Now simulate scheduler's validate_tokens
        # (accept then rollback — same as XgrammarGrammar.validate_tokens)
        accepted = []
        for token in draft_tokens:
            if scheduler_matcher.accept_token(token):
                accepted.append(token)
            else:
                break
        if accepted:
            scheduler_matcher.rollback(len(accepted))

        # ALL draft tokens should be accepted (no truncation!)
        assert len(accepted) == len(draft_tokens), (
            f"Grammar-masked draft produced {len(draft_tokens)} tokens "
            f"but scheduler only accepted {len(accepted)}. "
            f"This means the shadow grammar optimization is broken."
        )
        print(
            f"SUCCESS: All {len(draft_tokens)} grammar-masked draft tokens "
            f"pass validate_tokens without truncation."
        )
        print(f"Decoded: {tokenizer.decode(draft_tokens)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
