# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-graph capture and replay of ROCM_AITER_MLA_SPARSE uniform decodes.

The builder declares ``UNIFORM_BATCH``: uniform decode batches (q_len 1, or
1 + spec tokens) run under FULL graphs. A graph is captured on one batch and
replayed on batches with different sequence lengths (on both sides of topk),
block tables and top-k indices, as the runner does, and compared with the
float32 reference. Runner-level modes (NONE / PIECEWISE / FULL_DECODE_ONLY /
FULL_AND_PIECEWISE) are covered end to end in
tests/v1/e2e/attention/test_rocm_aiter_mla_sparse_runtime.py.
"""

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_mla_sparse.sparse_harness import (
    AITER_FP8_LSE_STALE_ROWS,
    DEVICE,
    SparseCase,
    make_case,
)
from tests.v1.attention.model_shapes import DEEPSEEK_V3_2, GLM_5_2, GLM_5_3_FLASH
from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import set_forward_context
from vllm.platforms.rocm import on_gfx950
from vllm.v1.attention.backend import CommonAttentionMetadata

NUM_REQS = 8
MAX_BLOCKS = 64


class StaticInputs:
    """Runner-style persistent inputs that a captured graph reads."""

    def __init__(self, case: SparseCase, q_len: int):
        self.case = case
        self.q_len = q_len
        n = NUM_REQS * q_len
        self.n = n
        self.query = case.query(n)
        self.block_table = torch.zeros(
            NUM_REQS, MAX_BLOCKS, dtype=torch.int32, device=DEVICE
        )
        self.seq_lens = torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE)
        self.qsl = torch.arange(0, n + 1, q_len, dtype=torch.int32, device=DEVICE)
        self.slot_mapping = torch.zeros(n, dtype=torch.int64, device=DEVICE)
        # The batch's own latent rows, rewritten to the cache by the forward.
        self.kv = torch.zeros(n, case.shape.head_size, dtype=case.dtype, device=DEVICE)

    def load(self, seq_lens: list[int], seed: int):
        """Copy a new batch into the static tensors; returns (batch, indices)."""
        case = self.case
        case.seed = seed
        batch = case.batch([self.q_len] * NUM_REQS, seq_lens)
        bt = batch.common.block_table_tensor
        self.block_table.zero_()
        self.block_table[:, : bt.shape[1]] = bt
        self.seq_lens.copy_(batch.common.seq_lens)
        self.slot_mapping.copy_(batch.common.slot_mapping)
        self.query.copy_(case.query(self.n))
        self.kv.copy_(case.kv_rows[batch.query_slots])
        common = CommonAttentionMetadata(
            query_start_loc=self.qsl,
            query_start_loc_cpu=self.qsl.cpu(),
            seq_lens=self.seq_lens,
            seq_lens_cpu_upper_bound=batch.common.seq_lens_cpu_upper_bound,
            num_reqs=NUM_REQS,
            num_actual_tokens=self.n,
            max_query_len=self.q_len,
            max_seq_len=max(seq_lens),
            block_table_tensor=self.block_table,
            slot_mapping=self.slot_mapping,
        )
        indices = case.indices(batch)
        case.topk_buffer[: self.n] = indices
        return batch, indices, case.builder.build(0, common)


def _forward(case: SparseCase, inputs: StaticInputs, metadata):
    lora = case.shape.kv_lora_rank
    rows = inputs.kv
    names = case.names
    out_shape = torch.Size((inputs.n, case.shape.num_heads * case.shape.v_head_dim))
    with set_forward_context(
        {name: metadata for name in names},
        case.config,
        slot_mapping={name: inputs.slot_mapping for name in names},
    ):
        return [
            case.layers[name].forward(
                inputs.query,
                rows[:, :lora],
                rows[:, lora:].unsqueeze(1),
                output_shape=out_shape,
            )
            for name in names
        ]


def _capture_and_replay(case: SparseCase, q_len: int, capture_lens, replay_lens):
    inputs = StaticInputs(case, q_len)
    _, _, metadata = inputs.load(capture_lens, seed=0)
    _forward(case, inputs, metadata)  # warm up outside the graph
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = _forward(case, inputs, metadata)
    for seed, seq_lens in enumerate(replay_lens, start=1):
        batch, indices, new_md = inputs.load(seq_lens, seed=seed)
        # The captured kernels read the builder's persistent buffers.
        assert new_md.paged_kv_indptr.data_ptr() == metadata.paged_kv_indptr.data_ptr()
        graph.replay()
        for name, out in zip(case.names, outputs):
            case.check(batch, inputs.query, indices, out, layer=case.layers[name])


CAPTURE = [64] * NUM_REQS
REPLAYS = [
    [100, 255, 256, 257, 900, 1500, 2000, 3000],
    [70] * NUM_REQS,
    [3000] * NUM_REQS,
]


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("q_len", [1, 4])
@pytest.mark.parametrize(
    "shape", [DEEPSEEK_V3_2.shard(8), GLM_5_2.shard(4)], ids=lambda s: s.id
)
def test_asm_decode_replays_new_lengths_and_tables(env, shape, q_len, kv_cache_dtype):
    case = make_case(
        shape, [3000] * NUM_REQS, kv_cache_dtype=kv_cache_dtype, spec_tokens=3
    )
    _capture_and_replay(case, q_len, CAPTURE, REPLAYS)


@pytest.mark.parametrize("q_len", [1, 4])
def test_nope_triton_decode_replays_longer_contexts(env, q_len):
    """The Triton decode picks its split count from max_seq_len at capture;
    replays with longer contexts must still be exact."""
    case = make_case(GLM_5_3_FLASH.shard(8), [3000] * NUM_REQS, spec_tokens=3)
    _capture_and_replay(case, q_len, CAPTURE, REPLAYS)


@pytest.mark.parametrize(
    "kv_cache_dtype", ["auto", pytest.param("fp8", marks=AITER_FP8_LSE_STALE_ROWS)]
)
def test_sink_decode_replays(env, kv_cache_dtype):
    case = make_case(
        DEEPSEEK_V3_2.shard(8),
        [3000] * NUM_REQS,
        kv_cache_dtype=kv_cache_dtype,
        use_sinks=True,
    )
    _capture_and_replay(case, 1, CAPTURE, REPLAYS)


@pytest.mark.skipif(not on_gfx950(), reason="AITER Triton sparse MLA is gfx950")
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_aiter_triton_decode_replays(env, monkeypatch, kv_cache_dtype):
    monkeypatch.setattr(rocm_aiter_ops, "_TRITON_SPARSE_MLA", True)
    case = make_case(
        DEEPSEEK_V3_2.shard(8), [3000] * NUM_REQS, kv_cache_dtype=kv_cache_dtype
    )
    _capture_and_replay(case, 1, CAPTURE, REPLAYS)


def test_shared_layers_replay(env):
    case = make_case(GLM_5_2.shard(8), [3000] * NUM_REQS, num_layers=3)
    _capture_and_replay(case, 1, CAPTURE, REPLAYS)
