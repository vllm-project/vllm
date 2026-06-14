# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NIXL PD transfer with equal-size decode context parallelism.

With identical TP and DCP sizes on both sides, each decode rank reads its
same-index prefill rank's shard through the existing rank-to-rank path; the
only connector change is the DCP-scaled scheduler block accounting, which
these tests cover.

No GPU or NIXL required.
"""

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
    NixlConnectorScheduler,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus

from .utils import create_request, create_vllm_config, make_kv_cache_config


class TestDcpSchedulerAccounting:
    @pytest.mark.parametrize("dcp_size", [1, 8])
    def test_block_size_is_resolved(self, dcp_size, default_vllm_config):
        physical = 16
        vllm_config = create_vllm_config(block_size=physical)
        vllm_config.parallel_config.decode_context_parallel_size = dcp_size
        scheduler = NixlConnectorScheduler(
            vllm_config, "engine", make_kv_cache_config(block_size=physical)
        )
        assert scheduler.block_size == physical * dcp_size

    def test_pcp_scales_block_size(self, default_vllm_config):
        """PCP scales the resolved block size, same as the engine scheduler."""
        physical = 16
        vllm_config = create_vllm_config(block_size=physical)
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 2
        scheduler = NixlConnectorScheduler(
            vllm_config, "engine", make_kv_cache_config(block_size=physical)
        )
        assert scheduler.block_size == physical * 2

    def test_boundary_lengths_dcp8(self, default_vllm_config):
        """Block counts at the resolved-block-size boundaries.

        Raw per-rank accounting diverges from the resolved accounting at
        every length that is not a multiple of the resolved block size, so
        an implementation using the raw cache block size fails these.
        """
        physical, dcp_size = 16, 8
        vllm_config = create_vllm_config(block_size=physical)
        vllm_config.parallel_config.decode_context_parallel_size = dcp_size
        scheduler = NixlConnectorScheduler(
            vllm_config, "engine", make_kv_cache_config(block_size=physical)
        )
        resolved = scheduler.block_size
        assert resolved == physical * dcp_size

        boundary_expectations = [
            (physical - 1, 1),
            (physical, 1),
            (resolved - 1, 1),
            (resolved, 1),
            (resolved + 1, 2),
        ]
        for num_tokens, expected_blocks in boundary_expectations:
            assert cdiv(num_tokens, scheduler.block_size) == expected_blocks

        # The raw per-rank block size over-counts inside one resolved block.
        assert cdiv(resolved - 1, physical) == 8
        assert cdiv(resolved - 1, resolved) == 1

    def test_sliding_window_blocks_scale_with_cp(self, default_vllm_config):
        """blocks_per_sw uses the CP-scaled per-group block capacity.

        Each block id covers block_size * dcp * pcp tokens of the sequence,
        so the window clip count shrinks accordingly; the raw per-group size
        would over-count and let get_sw_clipped_blocks keep HMA's leading
        null-marked blocks.
        """
        physical, dcp_size, pcp_size, sw_size = 16, 4, 2, 512
        vllm_config = create_vllm_config(block_size=physical)
        vllm_config.parallel_config.decode_context_parallel_size = dcp_size
        vllm_config.parallel_config.prefill_context_parallel_size = pcp_size
        # Single SWA group: hybrid (multi-group) layouts are still rejected
        # by resolve_kv_cache_block_sizes under CP.
        kv_cache_config = KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer0"],
                    SlidingWindowSpec(
                        block_size=physical,
                        num_kv_heads=4,
                        head_size=16,
                        dtype=torch.float16,
                        sliding_window=sw_size,
                    ),
                )
            ],
        )
        scheduler = NixlConnectorScheduler(vllm_config, "engine", kv_cache_config)
        assert scheduler.blocks_per_sw == [
            cdiv(sw_size, physical * dcp_size * pcp_size) + 1
        ]

    def test_request_finished_emits_exact_token_count(self, default_vllm_config):
        """remote_num_tokens stays the exact computed count, unrounded."""
        physical, dcp_size = 16, 8
        vllm_config = create_vllm_config(block_size=physical)
        vllm_config.parallel_config.decode_context_parallel_size = dcp_size
        scheduler = NixlConnectorScheduler(
            vllm_config, "engine", make_kv_cache_config(block_size=physical)
        )
        num_tokens = physical * dcp_size + 3  # not block-aligned
        request = create_request(
            request_id=1,
            block_size=physical,
            num_tokens=num_tokens,
            do_remote_decode=True,
        )
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        request.num_computed_tokens = num_tokens

        delay_free, params = scheduler.request_finished(request, ([1, 2],))
        assert delay_free
        assert params is not None
        assert params["remote_num_tokens"] == num_tokens


class TestDcpCompatibilityHash:
    """CP sizes are part of the handshake compatibility hash, so P/D pairs
    with mismatched DCP/PCP sizes are rejected at handshake time."""

    @staticmethod
    def _hash(dcp_size: int = 1, pcp_size: int = 1) -> str:
        vllm_config = create_vllm_config()
        vllm_config.parallel_config.decode_context_parallel_size = dcp_size
        vllm_config.parallel_config.prefill_context_parallel_size = pcp_size
        return compute_nixl_compatibility_hash(
            vllm_config, attn_backend_name="FLASH_ATTN", cross_layers_blocks=False
        )

    def test_identical_cp_sizes_match(self, default_vllm_config):
        assert self._hash(dcp_size=8) == self._hash(dcp_size=8)

    def test_dcp_mismatch_changes_hash(self, default_vllm_config):
        assert self._hash(dcp_size=2) != self._hash(dcp_size=4)

    def test_pcp_mismatch_changes_hash(self, default_vllm_config):
        assert self._hash(pcp_size=1) != self._hash(pcp_size=2)

    def test_equal_cp_product_different_split_changes_hash(self, default_vllm_config):
        """dcp*pcp products match but rank interleavings differ; reject."""
        assert self._hash(dcp_size=4, pcp_size=1) != self._hash(dcp_size=1, pcp_size=4)
