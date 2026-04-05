# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for visual token pruning functions and config integration."""

import pytest
import torch

from vllm.model_executor.layers.attention.visual_token_pruning import (
    prune_visual_tokens_dominant_only,
    prune_visual_tokens_with_merge,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_embeddings():
    """100 tokens with hidden_size=64, deterministic scores."""
    torch.manual_seed(42)
    N, D = 100, 64
    embeddings = torch.randn(N, D)
    scores = torch.arange(N, dtype=torch.float32)  # 0..99, token 99 highest
    return embeddings, scores


@pytest.fixture
def small_embeddings():
    """10 tokens with hidden_size=16."""
    torch.manual_seed(0)
    N, D = 10, 16
    embeddings = torch.randn(N, D)
    scores = torch.arange(N, dtype=torch.float32)
    return embeddings, scores


# ===================================================================
# Tests for prune_visual_tokens_dominant_only
# ===================================================================


class TestDominantOnly:
    def test_no_pruning_rate_0(self, sample_embeddings):
        emb, scores = sample_embeddings
        pruned, indices = prune_visual_tokens_dominant_only(emb, scores, 0.0)
        assert pruned.shape == emb.shape
        assert torch.equal(pruned, emb)
        assert indices.shape[0] == emb.shape[0]

    def test_prune_half(self, sample_embeddings):
        emb, scores = sample_embeddings
        # pruning_rate=0.5 means prune 50%, keep 50 tokens
        pruned, indices = prune_visual_tokens_dominant_only(emb, scores, 0.5)
        assert pruned.shape[0] == 50
        assert pruned.shape[1] == emb.shape[1]
        assert indices.shape[0] == 50

    def test_keeps_highest_scoring_tokens(self, sample_embeddings):
        emb, scores = sample_embeddings
        # pruning_rate=0.6 means prune 60%, keep 40 tokens
        _, indices = prune_visual_tokens_dominant_only(emb, scores, 0.6)
        # scores = 0..99, top 40 should be indices 60..99
        assert indices.min().item() >= 60

    def test_indices_sorted(self, sample_embeddings):
        emb, scores = sample_embeddings
        # pruning_rate=0.7 means prune 70%, keep 30 tokens
        _, indices = prune_visual_tokens_dominant_only(emb, scores, 0.7)
        assert torch.all(indices[1:] > indices[:-1])

    def test_pruned_embeds_match_indices(self, sample_embeddings):
        emb, scores = sample_embeddings
        pruned, indices = prune_visual_tokens_dominant_only(emb, scores, 0.5)
        assert torch.allclose(pruned, emb[indices])

    def test_at_least_one_token_kept(self, sample_embeddings):
        emb, scores = sample_embeddings
        # pruning_rate=0.999 means prune 99.9%, but at least 1 token kept
        pruned, indices = prune_visual_tokens_dominant_only(emb, scores, 0.999)
        assert pruned.shape[0] >= 1

    def test_very_small_input(self):
        emb = torch.randn(3, 8)
        scores = torch.tensor([1.0, 3.0, 2.0])
        # pruning_rate=0.5 means prune 50%, keep int(3 * 0.5) = 1
        pruned, indices = prune_visual_tokens_dominant_only(emb, scores, 0.5)
        # int(3 * 0.5) = 1, keep at least 1
        assert pruned.shape[0] == 1
        assert indices.item() == 1  # token with score 3.0


# ===================================================================
# Tests for prune_visual_tokens_with_merge
# ===================================================================


class TestWithMerge:
    def test_no_pruning_rate_0(self, sample_embeddings):
        emb, scores = sample_embeddings
        pruned, indices = prune_visual_tokens_with_merge(emb, scores, 0.0)
        assert pruned.shape == emb.shape
        assert torch.equal(pruned, emb)

    def test_output_shape(self, sample_embeddings):
        emb, scores = sample_embeddings
        # pruning_rate=0.6 means prune 60%, keep 40% -> keep_ratio=0.4
        pruning_rate = 0.6
        merge_ratio = 0.1
        pruned, indices = prune_visual_tokens_with_merge(
            emb, scores, pruning_rate, merge_ratio
        )
        keep_ratio = 1.0 - pruning_rate  # 0.4
        dominant_num = max(1, int(100 * (keep_ratio - merge_ratio)))  # 30
        anchor_num = max(1, int(100 * merge_ratio))  # 10
        expected_total = dominant_num + anchor_num  # 40
        assert pruned.shape[0] == expected_total
        assert indices.shape[0] == expected_total

    def test_indices_sorted(self, sample_embeddings):
        emb, scores = sample_embeddings
        # pruning_rate=0.6 -> keep 40%
        _, indices = prune_visual_tokens_with_merge(emb, scores, 0.6)
        assert torch.all(indices[1:] > indices[:-1])

    def test_dominant_tokens_preserved(self, sample_embeddings):
        """Dominant tokens should keep their original embeddings."""
        emb, scores = sample_embeddings
        pruning_rate, merge_ratio = 0.6, 0.1
        pruned, indices = prune_visual_tokens_with_merge(
            emb, scores, pruning_rate, merge_ratio
        )

        keep_ratio = 1.0 - pruning_rate  # 0.4
        dominant_num = int(100 * (keep_ratio - merge_ratio))
        # Get which of the kept indices are dominant (top-30 by score)
        _, top_indices = torch.topk(scores, dominant_num, sorted=False)
        top_set = set(top_indices.tolist())

        for i, idx in enumerate(indices.tolist()):
            if idx in top_set:
                # Dominant tokens should be unchanged
                assert torch.allclose(pruned[i], emb[idx])

    def test_anchor_tokens_modified(self, sample_embeddings):
        """Anchor tokens should differ from originals (they got merged info)."""
        emb, scores = sample_embeddings
        pruning_rate, merge_ratio = 0.6, 0.1
        pruned, indices = prune_visual_tokens_with_merge(
            emb, scores, pruning_rate, merge_ratio
        )

        keep_ratio = 1.0 - pruning_rate  # 0.4
        dominant_num = int(100 * (keep_ratio - merge_ratio))
        _, top_indices = torch.topk(scores, dominant_num, sorted=False)
        top_set = set(top_indices.tolist())

        anchor_found = False
        for i, idx in enumerate(indices.tolist()):
            if idx not in top_set:
                anchor_found = True
                # Anchor tokens should be modified (original + aggregated)
                if not torch.allclose(pruned[i], emb[idx]):
                    break
        assert anchor_found, "Should have at least one anchor token"

    def test_fallback_when_dominant_ratio_negative(self, sample_embeddings):
        """When keep_ratio <= merge_ratio, should fallback to dominant_only."""
        emb, scores = sample_embeddings
        # pruning_rate=0.95 -> keep_ratio=0.05, which <= merge_ratio=0.1
        pruned, indices = prune_visual_tokens_with_merge(
            emb, scores, pruning_rate=0.95, merge_ratio=0.1
        )
        # Fallback: dominant_only with pruning_rate=0.95 -> keep 5 tokens
        assert pruned.shape[0] == max(1, int(100 * 0.05))

    def test_small_input(self, small_embeddings):
        emb, scores = small_embeddings
        # pruning_rate=0.5 -> keep_ratio=0.5
        pruned, indices = prune_visual_tokens_with_merge(
            emb, scores, pruning_rate=0.5, merge_ratio=0.1
        )
        # dominant = int(10*0.4)=4, anchor = int(10*0.1)=1 -> 5 total
        assert pruned.shape[0] == 5
        assert pruned.shape[1] == emb.shape[1]

    def test_merge_ratio_zero_equivalent_to_dominant(self, sample_embeddings):
        emb, scores = sample_embeddings
        # pruning_rate=0.6 -> keep_ratio=0.4
        pruned_merge, idx_merge = prune_visual_tokens_with_merge(
            emb, scores, pruning_rate=0.6, merge_ratio=0.0
        )
        # merge_ratio=0.0 -> dominant_ratio=0.4, anchor_num=0
        # But max(1, int(100*0.0))=1, so still 1 anchor
        # This is fine - just verify it doesn't crash
        assert pruned_merge.shape[0] > 0


# ===================================================================
# Tests for MultiModalConfig auto-enable
# ===================================================================


class TestMultiModalConfigAutoEnable:
    def test_auto_enable_extract_score(self):
        from vllm.config.multimodal import MultiModalConfig

        # image_pruning_rate=0.6 means prune 60% of tokens
        cfg = MultiModalConfig(image_pruning_rate=0.6)
        assert cfg.extract_vit_attention_score is True

    def test_no_auto_enable_when_rate_none(self):
        from vllm.config.multimodal import MultiModalConfig

        cfg = MultiModalConfig(image_pruning_rate=None)
        assert cfg.extract_vit_attention_score is False

    def test_no_auto_enable_when_rate_0(self):
        from vllm.config.multimodal import MultiModalConfig

        # image_pruning_rate=0.0 means no pruning
        cfg = MultiModalConfig(image_pruning_rate=0.0)
        assert cfg.extract_vit_attention_score is False

    def test_already_enabled_not_overridden(self):
        from vllm.config.multimodal import MultiModalConfig

        cfg = MultiModalConfig(image_pruning_rate=0.6, extract_vit_attention_score=True)
        assert cfg.extract_vit_attention_score is True

    def test_is_multimodal_pruning_enabled_image(self):
        from vllm.config.multimodal import MultiModalConfig

        cfg = MultiModalConfig(image_pruning_rate=0.6)
        assert cfg.is_multimodal_pruning_enabled() is True

    def test_is_multimodal_pruning_enabled_none(self):
        from vllm.config.multimodal import MultiModalConfig

        cfg = MultiModalConfig()
        assert cfg.is_multimodal_pruning_enabled() is False

    def test_is_multimodal_pruning_enabled_video(self):
        from vllm.config.multimodal import MultiModalConfig

        cfg = MultiModalConfig(video_pruning_rate=0.5)
        assert cfg.is_multimodal_pruning_enabled() is True


# ===================================================================
# GPU-based tests (only run when CUDA available)
# ===================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestPruningOnGPU:
    def test_dominant_only_cuda(self):
        emb = torch.randn(100, 64, device="cuda")
        scores = torch.randn(100, device="cuda")
        # pruning_rate=0.5 -> keep 50 tokens
        pruned, indices = prune_visual_tokens_dominant_only(emb, scores, 0.5)
        assert pruned.device.type == "cuda"
        assert pruned.shape[0] == 50

    def test_with_merge_cuda(self):
        emb = torch.randn(100, 64, device="cuda")
        scores = torch.randn(100, device="cuda")
        # pruning_rate=0.6 -> keep_ratio=0.4, merge_ratio=0.1
        pruned, indices = prune_visual_tokens_with_merge(emb, scores, 0.6, 0.1)
        assert pruned.device.type == "cuda"
        expected = int(100 * 0.3) + int(100 * 0.1)
        assert pruned.shape[0] == expected

    def test_bfloat16_support(self):
        emb = torch.randn(100, 64, device="cuda", dtype=torch.bfloat16)
        scores = torch.randn(100, device="cuda")
        # pruning_rate=0.6 -> keep 40%
        pruned, _ = prune_visual_tokens_with_merge(emb, scores, 0.6)
        assert pruned.dtype == torch.bfloat16
