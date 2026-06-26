# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for VoxtralRealtimeGeneration.embed_input_ids fix.

Verifies that embed_input_ids always returns a tensor of shape
[num_input_tokens, embed_dim], regardless of whether the batch contains
a mix of multimodal (prefill) and non-multimodal (decode) tokens.

This is the fix for issue #39202: the engine crashed with a tensor size
mismatch when a batch contained both prefill and decode requests.
"""
import contextlib
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm import EngineArgs, LLM, SamplingParams
from vllm.assets.audio import AudioAsset

from ....utils import ROCM_ENGINE_KWARGS

MODEL_NAME = "mistralai/Voxtral-Mini-4B-Realtime-2602"

# Typical values — actual values come from the model config but these are
# representative for constructing test tensors.
EMBED_DIM = 1024
BLOCK_POOL_SIZE = 2
D_MODEL = EMBED_DIM // BLOCK_POOL_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(embed_dim=EMBED_DIM, d_model=D_MODEL,
                     block_pool_size=BLOCK_POOL_SIZE, dtype=torch.float32):
    """Create a minimal mock that has the attributes embed_input_ids needs."""
    from types import SimpleNamespace
    mock = MagicMock()
    audio_config = SimpleNamespace(
        d_model=d_model,
        block_pool_size=block_pool_size,
    )
    mock.config = SimpleNamespace(audio_config=audio_config)
    # whisper_encoder.dtype controls the output dtype
    mock.whisper_encoder = SimpleNamespace(dtype=dtype)
    return mock


def _make_embeddings(num_tokens, embed_dim=EMBED_DIM, dtype=torch.float32,
                     device="cpu"):
    """Create a list-of-tensors (NestedTensors) simulating multimodal embeds.

    Returns a list with a single tensor of shape [num_tokens, embed_dim],
    filled with non-zero values so we can distinguish them from zero-padding.
    """
    if num_tokens == 0:
        return []
    t = torch.randn(num_tokens, embed_dim, dtype=dtype, device=device)
    return [t]


# ---------------------------------------------------------------------------
# Unit tests for embed_input_ids output shape contract
# ---------------------------------------------------------------------------

@pytest.mark.skip_global_cleanup
class TestEmbedInputIdsOutputShape:
    """The primary contract: output shape must be [input_ids.shape[0], embed_dim]."""

    @pytest.fixture(autouse=True)
    def import_model(self):
        """Lazily import to avoid CUDA issues in environments without GPU."""
        try:
            from vllm.model_executor.models.voxtral_realtime import (
                VoxtralRealtimeGeneration,
            )
            self.model_cls = VoxtralRealtimeGeneration
        except ImportError:
            pytest.skip("voxtral_realtime model not available")

    def _call_embed(self, mock_model, input_ids, multimodal_embeddings=None,
                    is_multimodal=None):
        """Call embed_input_ids as an unbound method on our mock."""
        return self.model_cls.embed_input_ids(
            mock_model, input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def test_all_multimodal_tokens(self):
        """All tokens are multimodal — output should exactly match input count."""
        num_tokens = 5
        mock = _make_mock_model()
        input_ids = torch.zeros(num_tokens, dtype=torch.long)
        is_mm = torch.ones(num_tokens, dtype=torch.bool)
        mm_embeds = _make_embeddings(num_tokens)

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=mm_embeds,
                                  is_multimodal=is_mm)

        assert result.shape == (num_tokens, EMBED_DIM)

    def test_mixed_batch_multimodal_and_decode(self):
        """The crash scenario: some tokens are multimodal, some are not.

        This is the core regression test. Previously, the method returned
        only mm_embeds_flat (3 rows) instead of output for all 4 tokens.
        """
        num_mm_tokens = 3
        num_total_tokens = 4  # 3 multimodal + 1 decode
        mock = _make_mock_model()
        input_ids = torch.zeros(num_total_tokens, dtype=torch.long)
        is_mm = torch.tensor([True, True, True, False])
        mm_embeds = _make_embeddings(num_mm_tokens)

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=mm_embeds,
                                  is_multimodal=is_mm)

        assert result.shape == (num_total_tokens, EMBED_DIM)

    def test_no_multimodal_tokens_empty_list(self):
        """All decode tokens — multimodal_embeddings is an empty list."""
        num_tokens = 3
        mock = _make_mock_model()
        input_ids = torch.zeros(num_tokens, dtype=torch.long)

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=[])

        assert result.shape == (num_tokens, EMBED_DIM)

    def test_no_multimodal_tokens_none(self):
        """All decode tokens — multimodal_embeddings is None."""
        num_tokens = 3
        mock = _make_mock_model()
        input_ids = torch.zeros(num_tokens, dtype=torch.long)

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=None)

        assert result.shape == (num_tokens, EMBED_DIM)

    def test_single_token_multimodal(self):
        """Single multimodal token — boundary case."""
        mock = _make_mock_model()
        input_ids = torch.zeros(1, dtype=torch.long)
        is_mm = torch.tensor([True])
        mm_embeds = _make_embeddings(1)

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=mm_embeds,
                                  is_multimodal=is_mm)

        assert result.shape == (1, EMBED_DIM)

    def test_single_token_non_multimodal(self):
        """Single non-multimodal (decode) token — boundary case."""
        mock = _make_mock_model()
        input_ids = torch.zeros(1, dtype=torch.long)

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=None)

        assert result.shape == (1, EMBED_DIM)

    def test_large_mixed_batch(self):
        """Larger batch with many decode tokens and few multimodal tokens."""
        num_total = 128
        num_mm = 10
        mock = _make_mock_model()
        input_ids = torch.zeros(num_total, dtype=torch.long)
        is_mm = torch.zeros(num_total, dtype=torch.bool)
        is_mm[:num_mm] = True
        mm_embeds = _make_embeddings(num_mm)

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=mm_embeds,
                                  is_multimodal=is_mm)

        assert result.shape == (num_total, EMBED_DIM)


# ---------------------------------------------------------------------------
# Tests for correct embedding placement
# ---------------------------------------------------------------------------

@pytest.mark.skip_global_cleanup
class TestEmbedInputIdsPlacement:
    """Multimodal embeddings must land at is_multimodal positions; zeros elsewhere."""

    @pytest.fixture(autouse=True)
    def import_model(self):
        try:
            from vllm.model_executor.models.voxtral_realtime import (
                VoxtralRealtimeGeneration,
            )
            self.model_cls = VoxtralRealtimeGeneration
        except ImportError:
            pytest.skip("voxtral_realtime model not available")

    def _call_embed(self, mock_model, input_ids, multimodal_embeddings=None,
                    is_multimodal=None):
        return self.model_cls.embed_input_ids(
            mock_model, input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def test_non_multimodal_positions_are_zeros(self):
        """Decode token positions should be all zeros."""
        num_total = 5
        num_mm = 3
        mock = _make_mock_model()
        input_ids = torch.zeros(num_total, dtype=torch.long)
        is_mm = torch.tensor([True, True, True, False, False])
        mm_embeds = _make_embeddings(num_mm)

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=mm_embeds,
                                  is_multimodal=is_mm)

        # Non-multimodal positions (indices 3, 4) must be all zeros
        assert torch.all(result[3] == 0), "Decode token at index 3 should be zeros"
        assert torch.all(result[4] == 0), "Decode token at index 4 should be zeros"

    def test_multimodal_positions_are_nonzero(self):
        """Multimodal positions should contain the actual embeddings, not zeros."""
        num_total = 4
        num_mm = 3
        mock = _make_mock_model()
        input_ids = torch.zeros(num_total, dtype=torch.long)
        is_mm = torch.tensor([True, True, True, False])
        # Use known non-zero embeddings
        mm_data = torch.ones(num_mm, EMBED_DIM)
        mm_embeds = [mm_data]

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=mm_embeds,
                                  is_multimodal=is_mm)

        for i in range(num_mm):
            assert torch.all(result[i] == 1.0), (
                f"Multimodal position {i} should have the embedding values"
            )

    def test_no_multimodal_all_zeros(self):
        """When no multimodal data, entire output should be zeros."""
        num_tokens = 4
        mock = _make_mock_model()
        input_ids = torch.zeros(num_tokens, dtype=torch.long)

        result = self._call_embed(mock, input_ids, multimodal_embeddings=[])

        assert torch.all(result == 0), "All-decode batch should return all zeros"

    def test_scattered_multimodal_positions(self):
        """Multimodal tokens at non-contiguous positions."""
        num_total = 6
        mock = _make_mock_model()
        input_ids = torch.zeros(num_total, dtype=torch.long)
        # Multimodal at positions 0, 2, 4 (non-contiguous)
        is_mm = torch.tensor([True, False, True, False, True, False])
        num_mm = int(is_mm.sum())
        mm_data = torch.ones(num_mm, EMBED_DIM) * 42.0
        mm_embeds = [mm_data]

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=mm_embeds,
                                  is_multimodal=is_mm)

        assert result.shape == (num_total, EMBED_DIM)
        # Multimodal positions should have value 42
        for idx in [0, 2, 4]:
            assert torch.allclose(result[idx], torch.tensor(42.0)), (
                f"Position {idx} should be 42.0"
            )
        # Non-multimodal positions should be zero
        for idx in [1, 3, 5]:
            assert torch.all(result[idx] == 0), (
                f"Position {idx} should be zeros"
            )


# ---------------------------------------------------------------------------
# Tests for dtype handling
# ---------------------------------------------------------------------------

@pytest.mark.skip_global_cleanup
class TestEmbedInputIdsDtype:
    """Output dtype should match whisper_encoder.dtype."""

    @pytest.fixture(autouse=True)
    def import_model(self):
        try:
            from vllm.model_executor.models.voxtral_realtime import (
                VoxtralRealtimeGeneration,
            )
            self.model_cls = VoxtralRealtimeGeneration
        except ImportError:
            pytest.skip("voxtral_realtime model not available")

    def _call_embed(self, mock_model, input_ids, multimodal_embeddings=None,
                    is_multimodal=None):
        return self.model_cls.embed_input_ids(
            mock_model, input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def test_output_dtype_float32(self):
        """Output should be float32 when whisper_encoder.dtype is float32."""
        mock = _make_mock_model(dtype=torch.float32)
        input_ids = torch.zeros(3, dtype=torch.long)
        is_mm = torch.ones(3, dtype=torch.bool)
        mm_embeds = _make_embeddings(3, dtype=torch.float32)

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=mm_embeds,
                                  is_multimodal=is_mm)

        assert result.dtype == torch.float32

    def test_output_dtype_float16(self):
        """Output should be float16 when whisper_encoder.dtype is float16."""
        mock = _make_mock_model(dtype=torch.float16)
        input_ids = torch.zeros(3, dtype=torch.long)
        is_mm = torch.ones(3, dtype=torch.bool)
        mm_embeds = _make_embeddings(3, dtype=torch.float16)

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=mm_embeds,
                                  is_multimodal=is_mm)

        assert result.dtype == torch.float16

    def test_dtype_cast_when_embeddings_differ(self):
        """Embeddings in float32 should be cast to match float16 encoder dtype."""
        mock = _make_mock_model(dtype=torch.float16)
        input_ids = torch.zeros(3, dtype=torch.long)
        is_mm = torch.ones(3, dtype=torch.bool)
        # Embeddings are float32 but encoder is float16
        mm_embeds = _make_embeddings(3, dtype=torch.float32)

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=mm_embeds,
                                  is_multimodal=is_mm)

        assert result.dtype == torch.float16

    def test_zero_output_dtype_no_embeddings(self):
        """Even with no embeddings, output dtype should match encoder dtype."""
        mock = _make_mock_model(dtype=torch.bfloat16)
        input_ids = torch.zeros(3, dtype=torch.long)

        result = self._call_embed(mock, input_ids, multimodal_embeddings=None)

        assert result.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Tests for is_multimodal=None fallback
# ---------------------------------------------------------------------------

@pytest.mark.skip_global_cleanup
class TestEmbedInputIdsNoMask:
    """When is_multimodal is None, fallback behavior should still work."""

    @pytest.fixture(autouse=True)
    def import_model(self):
        try:
            from vllm.model_executor.models.voxtral_realtime import (
                VoxtralRealtimeGeneration,
            )
            self.model_cls = VoxtralRealtimeGeneration
        except ImportError:
            pytest.skip("voxtral_realtime model not available")

    def _call_embed(self, mock_model, input_ids, multimodal_embeddings=None,
                    is_multimodal=None):
        return self.model_cls.embed_input_ids(
            mock_model, input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def test_no_mask_exact_match(self):
        """No mask, embeddings count == input count — direct use."""
        num_tokens = 5
        mock = _make_mock_model()
        input_ids = torch.zeros(num_tokens, dtype=torch.long)
        mm_embeds = _make_embeddings(num_tokens)

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=mm_embeds,
                                  is_multimodal=None)

        assert result.shape == (num_tokens, EMBED_DIM)

    def test_no_mask_fewer_embeddings(self):
        """No mask, fewer embeddings than input tokens — should not crash."""
        num_tokens = 5
        num_mm = 3
        mock = _make_mock_model()
        input_ids = torch.zeros(num_tokens, dtype=torch.long)
        mm_embeds = _make_embeddings(num_mm)

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=mm_embeds,
                                  is_multimodal=None)

        # Must return full size regardless
        assert result.shape == (num_tokens, EMBED_DIM)

    def test_is_multimodal_all_false(self):
        """is_multimodal provided but all False — should return all zeros."""
        num_tokens = 4
        mock = _make_mock_model()
        input_ids = torch.zeros(num_tokens, dtype=torch.long)
        is_mm = torch.zeros(num_tokens, dtype=torch.bool)
        mm_embeds = _make_embeddings(0)  # empty

        result = self._call_embed(mock, input_ids,
                                  multimodal_embeddings=mm_embeds,
                                  is_multimodal=is_mm)

        assert result.shape == (num_tokens, EMBED_DIM)
        assert torch.all(result == 0)


# ---------------------------------------------------------------------------
# Tests for error conditions
# ---------------------------------------------------------------------------

@pytest.mark.skip_global_cleanup
class TestEmbedInputIdsErrors:
    """Edge cases that should either handle gracefully or raise clear errors."""

    @pytest.fixture(autouse=True)
    def import_model(self):
        try:
            from vllm.model_executor.models.voxtral_realtime import (
                VoxtralRealtimeGeneration,
            )
            self.model_cls = VoxtralRealtimeGeneration
        except ImportError:
            pytest.skip("voxtral_realtime model not available")

    def _call_embed(self, mock_model, input_ids, multimodal_embeddings=None,
                    is_multimodal=None):
        return self.model_cls.embed_input_ids(
            mock_model, input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def test_mask_count_mismatch_raises(self):
        """is_multimodal says 3 tokens but embeddings has 2 — should error.

        This indicates a bug in _gather_mm_embeddings and should propagate
        a clear error rather than silently producing wrong results.
        """
        mock = _make_mock_model()
        input_ids = torch.zeros(5, dtype=torch.long)
        is_mm = torch.tensor([True, True, True, False, False])  # 3 True
        mm_embeds = _make_embeddings(2)  # only 2 embeddings — mismatch!

        with pytest.raises((RuntimeError, IndexError)):
            self._call_embed(mock, input_ids,
                             multimodal_embeddings=mm_embeds,
                             is_multimodal=is_mm)


# ---------------------------------------------------------------------------
# Integration-style test: mixed batch with real engine
# ---------------------------------------------------------------------------

ENGINE_CONFIG = {
    "model": MODEL_NAME,
    "max_model_len": 8192,
    "max_num_seqs": 4,
    "limit_mm_per_prompt": {"audio": 1},
    "config_format": "mistral",
    "load_format": "mistral",
    "tokenizer_mode": "mistral",
    "enforce_eager": True,
    "gpu_memory_utilization": 0.9,
    **ROCM_ENGINE_KWARGS,
}


@pytest.fixture
def engine():
    engine_args = EngineArgs(**ENGINE_CONFIG)
    llm = LLM.from_engine_args(engine_args)
    try:
        yield llm
    finally:
        with contextlib.suppress(Exception):
            llm.llm_engine.engine_core.shutdown()
        torch.accelerator.empty_cache()


@pytest.fixture
def audio_assets():
    return [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]


@pytest.fixture
def tokenizer():
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    return MistralTokenizer.from_hf_hub(MODEL_NAME)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Requires GPU")
def test_concurrent_requests_no_crash(audio_assets, tokenizer, engine):
    """Regression test: concurrent requests should not crash with tensor
    size mismatch when the batch contains both prefill and decode tokens.

    This exercises the scenario from issue #39202 where a mixed batch
    (new prefill + cached decode) caused a RuntimeError.
    """
    from mistral_common.audio import Audio
    from mistral_common.protocol.instruct.chunk import RawAudio
    from mistral_common.protocol.transcription.request import (
        StreamingMode,
        TranscriptionRequest,
    )

    audio_config = tokenizer.instruct_tokenizer.tokenizer.audio

    def from_file(file_path):
        audio = Audio.from_file(file_path, strict=False)
        req = TranscriptionRequest(
            audio=RawAudio.from_audio(audio),
            streaming=StreamingMode.OFFLINE,
            language=None,
        )
        tokenized = tokenizer.instruct_tokenizer.encode_transcription(req)
        return tokenized.tokens, tokenized.audios[0].audio_array

    tokenized_list = [
        from_file(asset.get_local_path()) for asset in audio_assets
    ]

    inputs = []
    sampling_params = []
    for tokens, audio_array in tokenized_list:
        num_samples = audio_array.shape[0]
        max_tokens = audio_config.num_audio_tokens(num_samples) - len(tokens) - 1
        sampling_params.append(
            SamplingParams(temperature=0.0, max_tokens=max_tokens)
        )
        inputs.append({
            "multi_modal_data": {"audio": [(audio_array, None)]},
            "prompt_token_ids": tokens,
        })

    # Submit all at once to maximize chance of mixed batches.
    # The old code would crash here with RuntimeError about tensor size mismatch.
    outputs = engine.generate(inputs, sampling_params=sampling_params)

    # Basic sanity: we got output for each input, no crash
    assert len(outputs) == len(inputs)
    for out in outputs:
        assert len(out.outputs) > 0
        assert len(out.outputs[0].text) > 0, "Output should not be empty"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Requires GPU")
def test_repeated_sequential_requests_no_crash(audio_assets, tokenizer, engine):
    """Multiple sequential requests should not crash.

    After the first request completes, subsequent requests exercise the
    path where decode tokens from prior requests may linger in the batch.
    """
    from mistral_common.audio import Audio
    from mistral_common.protocol.instruct.chunk import RawAudio
    from mistral_common.protocol.transcription.request import (
        StreamingMode,
        TranscriptionRequest,
    )

    audio_config = tokenizer.instruct_tokenizer.tokenizer.audio

    for _round in range(3):
        asset = audio_assets[0]
        audio = Audio.from_file(asset.get_local_path(), strict=False)
        req = TranscriptionRequest(
            audio=RawAudio.from_audio(audio),
            streaming=StreamingMode.OFFLINE,
            language=None,
        )
        tokenized = tokenizer.instruct_tokenizer.encode_transcription(req)
        tokens = tokenized.tokens
        audio_array = tokenized.audios[0].audio_array

        num_samples = audio_array.shape[0]
        max_tokens = audio_config.num_audio_tokens(num_samples) - len(tokens) - 1
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        outputs = engine.generate(
            [{"multi_modal_data": {"audio": [(audio_array, None)]},
              "prompt_token_ids": tokens}],
            sampling_params=[sp],
        )

        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].text) > 0
