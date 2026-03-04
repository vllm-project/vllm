# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-backend CUDA graph padding values.

These tests verify the mechanism that allows different attention backends
to declare their own safe padding values for unused (padded) entries in
seq_lens and block_table during CUDA graph capture/replay.
"""

from unittest import mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionMetadataBuilder


# --------------------------------------------------------------------------- #
# 1. Base class default padding values
# --------------------------------------------------------------------------- #
class TestAttentionMetadataBuilderDefaults:
    """Verify the default CG padding values on the base class."""

    def test_default_cg_pad_seq_lens(self):
        assert AttentionMetadataBuilder.cg_pad_seq_lens == 0

    def test_default_cg_pad_block_table(self):
        assert AttentionMetadataBuilder.cg_pad_block_table == -1


# --------------------------------------------------------------------------- #
# 2. FlashMLAMetadataBuilder FP8 overrides
# --------------------------------------------------------------------------- #
def _flashmla_importable() -> bool:
    """Check if FlashMLAMetadataBuilder can be imported."""
    try:
        from vllm.v1.attention.backends.mla.flashmla import (  # noqa: F401
            FlashMLAMetadataBuilder,
        )

        return True
    except (ImportError, Exception):
        return False


@pytest.mark.skipif(
    not _flashmla_importable(),
    reason="FlashMLAMetadataBuilder not importable on this platform",
)
class TestFlashMLAPaddingOverrides:
    """Verify FlashMLAMetadataBuilder overrides padding for FP8."""

    @staticmethod
    def _create_builder(cache_dtype: str = "fp8_e4m3"):
        """Create a FlashMLAMetadataBuilder with mocked config.

        Uses mock objects to avoid downloading the real model weights.
        Only the attributes that FlashMLAMetadataBuilder.__init__ actually
        reads are wired up.
        """
        from vllm.model_executor.layers.attention.mla_attention import (
            MLACommonMetadataBuilder,
        )
        from vllm.v1.attention.backends.mla.flashmla import (
            FlashMLAMetadataBuilder,
        )

        # Lightweight mock vllm_config — no real model needed.
        vllm_config = mock.MagicMock()
        vllm_config.cache_config.cache_dtype = cache_dtype
        vllm_config.model_config.get_num_attention_heads.return_value = 128
        # Disable full cudagraphs so CG buffer allocation (which needs a
        # real CUDA device) is skipped.
        vllm_config.compilation_config.cudagraph_mode \
            .has_full_cudagraphs.return_value = False
        vllm_config.scheduler_config.max_num_seqs = 256

        device = torch.device("cuda:0")
        kv_cache_spec = mock.MagicMock()

        # Patch the heavy parent __init__ (MLACommonMetadataBuilder) so it
        # only sets the few attributes that FlashMLAMetadataBuilder reads.
        def _lightweight_init(self, kv_cache_spec, layer_names,
                              vllm_config, device, metadata_cls=None,
                              **kwargs):
            self.compilation_config = vllm_config.compilation_config
            self.device = device

        with mock.patch.object(
            MLACommonMetadataBuilder, "__init__", _lightweight_init
        ):
            builder = FlashMLAMetadataBuilder(
                kv_cache_spec=kv_cache_spec,
                layer_names=["model.layers.0.self_attn"],
                vllm_config=vllm_config,
                device=device,
            )
        return builder

    def test_fp8_overrides_padding(self):
        """FP8 cache should override padding to safe values."""
        builder = self._create_builder(cache_dtype="fp8_e4m3")
        assert builder.cg_pad_seq_lens == 1, (
            "FP8 FlashMLA must pad seq_lens with 1, not 0"
        )
        assert builder.cg_pad_block_table == 0, (
            "FP8 FlashMLA must pad block_table with 0, not -1"
        )

    def test_non_fp8_keeps_defaults(self):
        """Non-FP8 cache should keep default padding values."""
        builder = self._create_builder(cache_dtype="auto")
        assert builder.cg_pad_seq_lens == 0
        assert builder.cg_pad_block_table == -1

    def test_fp8_allocates_cg_buffers_when_cudagraph_enabled(self):
        """FP8 + CG mode should allocate persistent buffers."""
        builder = self._create_builder(cache_dtype="fp8_e4m3")
        if builder._use_cg_buf:
            assert builder.cg_buf_tile_scheduler_metadata is not None
            assert builder.cg_buf_num_splits is not None
        else:
            # CG not enabled in the test config → buffers should be None
            assert builder.cg_buf_tile_scheduler_metadata is None
            assert builder.cg_buf_num_splits is None


# --------------------------------------------------------------------------- #
# 3. Consolidated padding value computation logic
# --------------------------------------------------------------------------- #
class TestConsolidatedPaddingComputation:
    """Test the max-across-builders padding consolidation logic."""

    @staticmethod
    def _make_mock_builder(pad_seq_lens: int = 0, pad_block_table: int = -1):
        """Create a mock AttentionMetadataBuilder with given padding values."""
        builder = mock.MagicMock()
        builder.cg_pad_seq_lens = pad_seq_lens
        builder.cg_pad_block_table = pad_block_table
        return builder

    @staticmethod
    def _compute_consolidated_padding(builders):
        """Replicate the consolidation logic from model runners."""
        cg_pad_seq_lens = 0
        cg_pad_block_table = -1
        for builder in builders:
            cg_pad_seq_lens = max(cg_pad_seq_lens, builder.cg_pad_seq_lens)
            cg_pad_block_table = max(
                cg_pad_block_table, builder.cg_pad_block_table
            )
        return cg_pad_seq_lens, cg_pad_block_table

    def test_all_defaults(self):
        """All default builders → default padding values."""
        builders = [self._make_mock_builder() for _ in range(3)]
        pad_sl, pad_bt = self._compute_consolidated_padding(builders)
        assert pad_sl == 0
        assert pad_bt == -1

    def test_one_fp8_builder(self):
        """One FP8 builder among defaults → FP8 values win (max)."""
        builders = [
            self._make_mock_builder(),
            self._make_mock_builder(pad_seq_lens=1, pad_block_table=0),
            self._make_mock_builder(),
        ]
        pad_sl, pad_bt = self._compute_consolidated_padding(builders)
        assert pad_sl == 1
        assert pad_bt == 0

    def test_multiple_fp8_builders(self):
        """Multiple FP8 builders → max values are taken."""
        builders = [
            self._make_mock_builder(pad_seq_lens=1, pad_block_table=0),
            self._make_mock_builder(pad_seq_lens=2, pad_block_table=1),
        ]
        pad_sl, pad_bt = self._compute_consolidated_padding(builders)
        assert pad_sl == 2
        assert pad_bt == 1

    def test_empty_builders(self):
        """No builders → default values."""
        pad_sl, pad_bt = self._compute_consolidated_padding([])
        assert pad_sl == 0
        assert pad_bt == -1


# --------------------------------------------------------------------------- #
# 4. prepare_pos_seq_lens Triton kernel with custom pad_value
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA required for Triton kernels",
)
class TestPreparePosSqlLensPadding:
    """Test that prepare_pos_seq_lens respects the pad_value parameter."""

    @staticmethod
    def _run_prepare(
        num_reqs: int,
        max_num_reqs: int,
        pad_value: int = 0,
    ):
        """Run prepare_pos_seq_lens and return the seq_lens tensor."""
        from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens

        device = torch.device("cuda:0")

        # idx_mapping: identity mapping for simplicity
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)

        # query_start_loc: each request has query_len=1 (decode)
        query_start_loc = torch.arange(
            num_reqs + 1, dtype=torch.int32, device=device
        )

        # num_computed_tokens: each request already has some context
        num_computed_tokens = torch.full(
            (max_num_reqs,), 100, dtype=torch.int32, device=device
        )

        # Output tensors
        pos = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        seq_lens = torch.full(
            (max_num_reqs,), -999, dtype=torch.int32, device=device
        )

        prepare_pos_seq_lens(
            idx_mapping=idx_mapping,
            query_start_loc=query_start_loc,
            num_computed_tokens=num_computed_tokens,
            pos=pos,
            seq_lens=seq_lens,
            pad_value=pad_value,
        )
        return seq_lens

    def test_default_pad_value_zero(self):
        """Unused slots should be filled with 0 by default."""
        num_reqs, max_num_reqs = 3, 8
        seq_lens = self._run_prepare(num_reqs, max_num_reqs, pad_value=0)

        # Active entries should have seq_len = num_computed_tokens + query_len
        # = 100 + 1 = 101
        for i in range(num_reqs):
            assert seq_lens[i].item() == 101, (
                f"Active seq_lens[{i}] should be 101, got {seq_lens[i].item()}"
            )

        # Padded entries should be 0
        for i in range(num_reqs, max_num_reqs):
            assert seq_lens[i].item() == 0, (
                f"Padded seq_lens[{i}] should be 0, got {seq_lens[i].item()}"
            )

    def test_custom_pad_value_one(self):
        """When pad_value=1, unused slots should be filled with 1."""
        num_reqs, max_num_reqs = 3, 8
        seq_lens = self._run_prepare(num_reqs, max_num_reqs, pad_value=1)

        # Active entries unchanged
        for i in range(num_reqs):
            assert seq_lens[i].item() == 101

        # Padded entries should be 1
        for i in range(num_reqs, max_num_reqs):
            assert seq_lens[i].item() == 1, (
                f"Padded seq_lens[{i}] should be 1, got {seq_lens[i].item()}"
            )

    def test_all_slots_active(self):
        """When all slots are active, no padding should occur."""
        num_reqs, max_num_reqs = 5, 5
        seq_lens = self._run_prepare(num_reqs, max_num_reqs, pad_value=42)

        # All entries should be 101 (active), none should be 42
        for i in range(max_num_reqs):
            assert seq_lens[i].item() == 101

    def test_large_pad_value(self):
        """Verify arbitrary large pad values work correctly."""
        num_reqs, max_num_reqs = 2, 6
        pad_value = 9999
        seq_lens = self._run_prepare(num_reqs, max_num_reqs, pad_value=pad_value)

        for i in range(num_reqs):
            assert seq_lens[i].item() == 101

        for i in range(num_reqs, max_num_reqs):
            assert seq_lens[i].item() == pad_value

