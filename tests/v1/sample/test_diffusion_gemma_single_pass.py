# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaModelState,
    DiffusionSampler,
    _CanvasSplit,
    _prefill_canvas_rows,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.cpu_attn import (
    CPUAttentionBackend,
    CPUAttentionBackendImpl,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.worker.utils import AttentionGroup

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 128
BLOCK_SIZE = 128


def _canvas_batch(rows: list[tuple[int, int, int]]) -> SimpleNamespace:
    computed, prefill_len, num_logits = (np.array(col) for col in zip(*rows))
    return SimpleNamespace(
        num_reqs=len(rows),
        num_computed_prefill_tokens_np=computed,
        prefill_len_np=prefill_len,
        cu_num_logits_np=np.concatenate(([0], np.cumsum(num_logits))),
    )


def _int32(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


@pytest.fixture
def cpu_attention(dtype, sliding_window):
    if torch.cpu._is_amx_tile_supported():
        torch.cpu._init_amx()
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        dtype=dtype, is_mm_prefix_lm=False, max_model_len=256
    )
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=dtype,
    )
    with set_current_vllm_config(config):
        layer = Attention(
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            scale=HEAD_SIZE**-0.5,
            num_kv_heads=NUM_KV_HEADS,
            per_layer_sliding_window=sliding_window,
            prefix="attn",
            attn_backend=CPUAttentionBackend,
        )
        impl = cast(CPUAttentionBackendImpl, layer.impl)
        group = AttentionGroup(CPUAttentionBackend, ["attn"], spec, 0)
        group.create_metadata_builders(config, torch.device("cpu"))
    cache_config = KVCacheConfig(8, [], [KVCacheGroupSpec(["attn"], spec)])
    state = DiffusionGemmaModelState.__new__(DiffusionGemmaModelState)
    state.model_config = config.model_config
    state.single_pass_reads = True
    state._causal_buf = torch.zeros(4, dtype=torch.int32)
    state.diffusion_states = SimpleNamespace(
        is_encoder_phase=torch.zeros(4, dtype=torch.bool)
    )

    def attend(query, key, value, rows, block_table, cache):
        computed, prompt_lens, query_lens, widths = np.array(rows).T
        starts = np.concatenate(([0], np.cumsum(query_lens))).astype(np.int32)
        slots = torch.arange(len(rows) - 1, -1, -1)
        state.diffusion_states.is_encoder_phase[slots] = torch.from_numpy(
            computed < prompt_lens
        )
        positions = torch.cat(
            [torch.arange(c, c + n) for c, n in zip(computed, query_lens)]
        )
        blocks = block_table.repeat_interleave(torch.from_numpy(query_lens), dim=0)
        slot_mapping = blocks.gather(1, (positions // BLOCK_SIZE)[:, None]).flatten()
        slot_mapping = slot_mapping * BLOCK_SIZE + positions % BLOCK_SIZE
        batch = SimpleNamespace(
            num_reqs=len(rows),
            num_tokens=len(query),
            num_computed_prefill_tokens_np=computed,
            prefill_len_np=prompt_lens,
            num_scheduled_tokens=query_lens,
            cu_num_logits_np=np.concatenate(([0], np.cumsum(widths))),
            query_start_loc_np=starts,
            query_start_loc=torch.from_numpy(starts),
            seq_lens=torch.from_numpy((computed + query_lens).astype(np.int32)),
            idx_mapping=slots,
        )
        metadata = state.prepare_attn(
            batch,
            CUDAGraphMode.NONE,
            (block_table,),
            slot_mapping[None],
            [[group]],
            cache_config,
        )["attn"]
        impl.do_kv_cache_update(layer, key, value, cache, metadata.slot_mapping)
        return impl.forward(
            layer, query, key, value, cache, metadata, torch.empty_like(query)
        )

    return attend


@pytest.fixture
def split() -> _CanvasSplit:
    return _CanvasSplit.from_batch(
        np.array([9, 16, 12, 5], dtype=np.int32),
        np.array([8, 8, 8, 0], dtype=np.int32),
        np.array([0, 2]),
    )


class TestCanvasRows:
    def test_prefill_canvas_rows_pick_requests_that_finish_their_prompt(self):
        rows = [(0, 10, 0), (0, 12, 8), (12, 12, 8), (3, 9, 8), (4, 10, 0)]
        assert _prefill_canvas_rows(_canvas_batch(rows)).tolist() == [1, 3]

    def test_fused_requests_split_into_prompt_and_canvas_rows(self, split):
        assert split.query_lens.tolist() == [1, 8, 16, 4, 8, 5]
        assert split.row_map.tolist() == [0, 0, 1, 2, 2, 3]
        assert split.query_start_loc.tolist() == [0, 1, 9, 25, 29, 37, 42]

    def test_prompt_rows_end_where_their_canvas_starts(self, split):
        seq_lens = split.seq_lens(_int32(20, 30, 12, 7))
        assert seq_lens.tolist() == [12, 20, 30, 4, 12, 7]

    def test_prompt_rows_are_causal_and_canvas_rows_bidirectional(self, split):
        causal = split.causal(torch.tensor([True, False, True, True]))
        assert causal.tolist() == [1, 0, 0, 1, 0, 1]


class TestFusedCanvasStart:
    def test_only_fused_rows_leave_the_encoder_phase(self):
        states = SimpleNamespace(is_encoder_phase=torch.ones(4, dtype=torch.bool))
        batch = _canvas_batch([(0, 10, 0), (0, 12, 8), (12, 12, 8), (3, 9, 8)])
        DiffusionSampler._start_fused_canvases(
            SimpleNamespace(diffusion_states=states), batch, np.array([3, 2, 1, 0])
        )
        assert states.is_encoder_phase.tolist() == [False, True, False, True]


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU attention kernels")
class TestFusedAttention:
    @pytest.mark.parametrize(
        "dtype, tol", [(torch.float32, 1e-4), (torch.bfloat16, 2e-2)]
    )
    @pytest.mark.parametrize("sliding_window", [None, 64])
    @pytest.mark.parametrize(
        "computed", [0, 128, 149], ids=["uncached", "cached", "last-token"]
    )
    def test_fused_attention_matches_two_passes(
        self, cpu_attention, dtype, tol, computed
    ):
        rows = [
            (computed, 150, 166 - computed, 16),
            (0, 137, 145, 8),
            (96, 96, 8, 8),
            (20, 50, 11, 0),
        ]
        generator = torch.Generator().manual_seed(0)
        query = torch.randn(
            4, 256, NUM_HEADS, HEAD_SIZE, dtype=dtype, generator=generator
        )
        key = torch.randn(
            4, 256, NUM_KV_HEADS, HEAD_SIZE, dtype=dtype, generator=generator
        )
        value = torch.randn(key.shape, dtype=dtype, generator=generator)
        block_table = torch.arange(8, dtype=torch.int32).flip(0).view(4, 2)
        cache = torch.zeros(8, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE, dtype=dtype)
        for i, (start, prompt, _, _) in enumerate(rows):
            if start:
                cpu_attention(
                    query[i, :start],
                    key[i, :start],
                    value[i, :start],
                    [(0, prompt, start, 0)],
                    block_table[i : i + 1],
                    cache,
                )
        reference_cache = cache.clone()
        scheduled = [
            torch.cat([t[i, c : c + n] for i, (c, _, n, _) in enumerate(rows)])
            for t in (query, key, value)
        ]
        fused = cpu_attention(*scheduled, rows, block_table, cache)
        expected = []
        for i, (start, prompt, length, width) in enumerate(rows):
            spans = [(start, length, width)]
            if start < prompt and width:
                spans = [(start, length - width, 0), (prompt, width, width)]
            for offset, count, logits in spans:
                expected.append(
                    cpu_attention(
                        query[i, offset : offset + count],
                        key[i, offset : offset + count],
                        value[i, offset : offset + count],
                        [(offset, prompt, count, logits)],
                        block_table[i : i + 1],
                        reference_cache,
                    )
                )
        torch.testing.assert_close(fused, torch.cat(expected), atol=tol, rtol=tol)
        torch.testing.assert_close(cache, reference_cache, atol=0, rtol=0)
