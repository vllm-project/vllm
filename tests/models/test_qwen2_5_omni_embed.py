# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Qwen2.5-Omni embed_input_ids to verify embeddings are
correctly assigned to audio/image/video token positions.

Regression test for: https://github.com/vllm-project/vllm/issues/34506
  - Non-interleaved mixed modalities (audio + image + video) should correctly
    assign audio embeddings to audio positions, image to image, video to video.
  - Interleaved (use_audio_in_video) should also work correctly.
"""

from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.models.qwen2_5_omni_thinker import (
    check_interleaved_audio_video,
    merge_interleaved_embeddings,
)

# Fake token IDs
AUDIO_TOKEN_ID = 1001
IMAGE_TOKEN_ID = 1002
VIDEO_TOKEN_ID = 1003
TEXT_TOKEN_ID = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_token_seq(
    audio_n: int, image_n: int, video_n: int, text_prefix: int = 3, text_sep: int = 2
):
    """
    Build a flat token sequence:
      [text_prefix] [AUDIO * audio_n] [text_sep] [IMAGE * image_n]
      [text_sep] [VIDEO * video_n] [text_sep]
    Returns (input_ids tensor, is_multimodal mask, positions dict).
    """
    tokens = (
        [TEXT_TOKEN_ID] * text_prefix
        + [AUDIO_TOKEN_ID] * audio_n
        + [TEXT_TOKEN_ID] * text_sep
        + [IMAGE_TOKEN_ID] * image_n
        + [TEXT_TOKEN_ID] * text_sep
        + [VIDEO_TOKEN_ID] * video_n
        + [TEXT_TOKEN_ID] * text_sep
    )
    input_ids = torch.tensor(tokens)
    is_multimodal = (
        (input_ids == AUDIO_TOKEN_ID)
        | (input_ids == IMAGE_TOKEN_ID)
        | (input_ids == VIDEO_TOKEN_ID)
    )
    return input_ids, is_multimodal


def make_interleaved_seq(
    video_chunks: list[int], audio_chunks: list[int], text_prefix: int = 2
):
    """
    Build an interleaved sequence like use_audio_in_video:
      [text] [V*v0] [A*a0] [V*v1] [A*a1] ...
    """
    tokens = [TEXT_TOKEN_ID] * text_prefix
    for v, a in zip(video_chunks, audio_chunks):
        tokens += [VIDEO_TOKEN_ID] * v + [AUDIO_TOKEN_ID] * a
    input_ids = torch.tensor(tokens)
    is_multimodal = (input_ids == VIDEO_TOKEN_ID) | (input_ids == AUDIO_TOKEN_ID)
    return input_ids, is_multimodal


# ---------------------------------------------------------------------------
# Tests for check_interleaved_audio_video
# ---------------------------------------------------------------------------


class TestCheckInterleavedAudioVideo:
    def test_non_interleaved_audio_then_video(self):
        """Audio entirely before video → not interleaved."""
        input_ids, is_multimodal = make_token_seq(5, 0, 4)
        is_video = is_multimodal & (input_ids == VIDEO_TOKEN_ID)
        is_audio = is_multimodal & (input_ids == AUDIO_TOKEN_ID)
        assert not check_interleaved_audio_video(
            is_video, is_audio, is_video.sum().item(), is_audio.sum().item()
        )

    def test_non_interleaved_with_image(self):
        """Audio + image + video (the mixed_modalities case) → not interleaved."""
        input_ids, is_multimodal = make_token_seq(5, 4, 6)
        is_video = is_multimodal & (input_ids == VIDEO_TOKEN_ID)
        is_audio = is_multimodal & (input_ids == AUDIO_TOKEN_ID)
        assert not check_interleaved_audio_video(
            is_video, is_audio, is_video.sum().item(), is_audio.sum().item()
        )

    def test_no_audio(self):
        """Video only → not interleaved."""
        input_ids, is_multimodal = make_token_seq(0, 0, 6)
        is_video = is_multimodal & (input_ids == VIDEO_TOKEN_ID)
        is_audio = is_multimodal & (input_ids == AUDIO_TOKEN_ID)
        assert not check_interleaved_audio_video(
            is_video, is_audio, is_video.sum().item(), is_audio.sum().item()
        )

    def test_interleaved(self):
        """V A V A interleaved → True."""
        input_ids, is_multimodal = make_interleaved_seq([4, 4], [3, 3])
        is_video = is_multimodal & (input_ids == VIDEO_TOKEN_ID)
        is_audio = is_multimodal & (input_ids == AUDIO_TOKEN_ID)
        assert check_interleaved_audio_video(
            is_video, is_audio, is_video.sum().item(), is_audio.sum().item()
        )


# ---------------------------------------------------------------------------
# Tests for embed_input_ids via a minimal mock
# ---------------------------------------------------------------------------


def make_mock_model(hidden: int = 8):
    """
    Return a minimal mock of Qwen2_5OmniThinkerForConditionalGeneration
    that has enough structure to run embed_input_ids.
    """
    from vllm.model_executor.models.qwen2_5_omni_thinker import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    model = Mock(spec=Qwen2_5OmniThinkerForConditionalGeneration)

    # Config with token IDs
    cfg = Mock()
    cfg.video_token_index = VIDEO_TOKEN_ID
    cfg.audio_token_index = AUDIO_TOKEN_ID
    model.config = cfg

    # embed_input_ids: simply embed each token as a one-hot-like vector
    # token_id * ones so we can verify which embedding ends up where.
    def fake_lm_embed(ids: torch.Tensor) -> torch.Tensor:
        # Use .clone() so the tensor is contiguous (expand() creates a strided
        # view with shared memory, which masked_scatter_ cannot handle).
        return ids.float().unsqueeze(-1).expand(-1, hidden).clone()

    lang_model = Mock()
    lang_model.embed_input_ids = fake_lm_embed
    model.get_language_model = Mock(return_value=lang_model)

    # _embed_text_input_ids: delegate to SupportsMultiModal's implementation
    from vllm.model_executor.models.interfaces import SupportsMultiModal

    model._embed_text_input_ids = (
        lambda *a, **kw: SupportsMultiModal._embed_text_input_ids(model, *a, **kw)
    )

    # super().embed_input_ids → use SupportsMultiModal.embed_input_ids
    def fake_super_embed(
        ids, mm_embs=None, *, is_multimodal=None, handle_oov_mm_token=False
    ):
        return SupportsMultiModal.embed_input_ids(
            model,
            ids,
            mm_embs,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    # Bind embed_input_ids as the real method
    model.embed_input_ids = (
        lambda *a, **kw: Qwen2_5OmniThinkerForConditionalGeneration.embed_input_ids(
            model, *a, **kw
        )
    )

    # Store super-embed for use inside the method
    model._super_embed_input_ids = fake_super_embed

    return model, hidden


def build_mm_embeds(
    audio_n, image_n, video_n, hidden, audio_val=10.0, image_val=20.0, video_val=30.0
):
    """
    Build multimodal_embeddings list in position order (audio, image, video).
    Each embedding is filled with a distinct constant so we can verify placement.
    """
    embs = []
    if audio_n:
        embs.append(torch.full((audio_n, hidden), audio_val))
    if image_n:
        embs.append(torch.full((image_n, hidden), image_val))
    if video_n:
        embs.append(torch.full((video_n, hidden), video_val))
    return embs


class TestEmbedInputIds:
    def _run(self, audio_n, image_n, video_n, hidden=8):
        """
        Run embed_input_ids for a non-interleaved mixed-modality sequence.
        Returns (result_embeds, input_ids, is_multimodal).
        """
        input_ids, is_multimodal = make_token_seq(audio_n, image_n, video_n)
        mm_embeds = build_mm_embeds(audio_n, image_n, video_n, hidden)

        model, _ = make_mock_model(hidden)
        result = model.embed_input_ids(
            input_ids, mm_embeds, is_multimodal=is_multimodal
        )
        return result, input_ids, is_multimodal

    def test_audio_only(self):
        """Audio-only: audio positions get audio embeddings."""
        audio_n, hidden = 5, 8
        audio_val = 10.0
        result, input_ids, is_multimodal = self._run(audio_n, 0, 0, hidden)

        audio_pos = (input_ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
        assert result[audio_pos].allclose(torch.full((audio_n, hidden), audio_val)), (
            "Audio positions should get audio embeddings"
        )

    def test_video_only(self):
        """Video-only: video positions get video embeddings."""
        video_n, hidden = 6, 8
        video_val = 30.0
        result, input_ids, is_multimodal = self._run(0, 0, video_n, hidden)

        video_pos = (input_ids == VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]
        assert result[video_pos].allclose(torch.full((video_n, hidden), video_val)), (
            "Video positions should get video embeddings"
        )

    def test_mixed_modalities_audio_goes_to_audio_pos(self):
        """
        Regression test for GitHub issue #34506:
        With audio + image + video (non-interleaved), audio positions must
        receive audio embeddings (not image or video embeddings).
        """
        audio_n, image_n, video_n, hidden = 5, 4, 6, 8
        audio_val, image_val, video_val = 10.0, 20.0, 30.0

        result, input_ids, is_multimodal = self._run(audio_n, image_n, video_n, hidden)

        audio_pos = (input_ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
        image_pos = (input_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        video_pos = (input_ids == VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]

        mean_a = result[audio_pos].mean().item()
        assert result[audio_pos].allclose(torch.full((audio_n, hidden), audio_val)), (
            f"Audio emb wrong: expected {audio_val}, got mean={mean_a:.1f}"
        )

        mean_i = result[image_pos].mean().item()
        assert result[image_pos].allclose(torch.full((image_n, hidden), image_val)), (
            f"Image emb wrong: expected {image_val}, got mean={mean_i:.1f}"
        )

        mean_v = result[video_pos].mean().item()
        assert result[video_pos].allclose(torch.full((video_n, hidden), video_val)), (
            f"Video emb wrong: expected {video_val}, got mean={mean_v:.1f}"
        )

    def test_text_positions_unchanged(self):
        """Text positions should keep their text embeddings."""
        audio_n, image_n, video_n, hidden = 3, 2, 4, 8
        result, input_ids, is_multimodal = self._run(audio_n, image_n, video_n, hidden)

        text_pos = (~is_multimodal).nonzero(as_tuple=True)[0]
        # Text tokens have value TEXT_TOKEN_ID=0, so embed → 0.0
        assert result[text_pos].allclose(torch.zeros(len(text_pos), hidden)), (
            "Text positions should keep text embeddings"
        )

    def test_interleaved_use_audio_in_video(self):
        """
        Interleaved (use_audio_in_video): video chunks interleaved with audio.
        Video embeddings must go to video positions, audio to audio positions.
        """
        hidden = 8
        audio_val, video_val = 10.0, 30.0
        # Two video chunks of 4, two audio chunks of 3
        video_chunks = [4, 4]
        audio_chunks = [3, 3]
        input_ids, is_multimodal = make_interleaved_seq(video_chunks, audio_chunks)

        video_n = sum(video_chunks)  # 8
        audio_n = sum(audio_chunks)  # 6

        # mm_embeds come in [video, audio] order (video feature first in
        # mm_features when positions are the same for use_audio_in_video)
        mm_embeds = [
            torch.full((video_n, hidden), video_val),
            torch.full((audio_n, hidden), audio_val),
        ]

        model, _ = make_mock_model(hidden)
        result = model.embed_input_ids(
            input_ids, mm_embeds, is_multimodal=is_multimodal
        )

        video_pos = (input_ids == VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]
        audio_pos = (input_ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]

        assert result[video_pos].allclose(torch.full((video_n, hidden), video_val)), (
            "Interleaved: video positions should get video embeddings"
        )

        assert result[audio_pos].allclose(torch.full((audio_n, hidden), audio_val)), (
            "Interleaved: audio positions should get audio embeddings"
        )


# ---------------------------------------------------------------------------
# Tests for merge_interleaved_embeddings helper
# ---------------------------------------------------------------------------


class TestMergeInterleavedEmbeddings:
    def test_basic_interleaved(self):
        """Video chunks + audio chunks scattered to correct positions."""
        hidden = 4
        input_ids, is_multimodal = make_interleaved_seq([3, 3], [2, 2])

        is_video = is_multimodal & (input_ids == VIDEO_TOKEN_ID)
        is_audio = is_multimodal & (input_ids == AUDIO_TOKEN_ID)
        num_video = is_video.sum().item()  # 6
        num_audio = is_audio.sum().item()  # 4

        inputs_embeds = torch.zeros(len(input_ids), hidden)
        mm_embeds = [
            torch.full((num_video, hidden), 30.0),
            torch.full((num_audio, hidden), 10.0),
        ]

        result = merge_interleaved_embeddings(
            inputs_embeds,
            mm_embeds,
            is_video,
            is_audio,
            is_multimodal,
            num_video,
            num_audio,
        )

        video_pos = is_video.nonzero(as_tuple=True)[0]
        audio_pos = is_audio.nonzero(as_tuple=True)[0]
        assert result[video_pos].allclose(torch.full((num_video, hidden), 30.0))
        assert result[audio_pos].allclose(torch.full((num_audio, hidden), 10.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
