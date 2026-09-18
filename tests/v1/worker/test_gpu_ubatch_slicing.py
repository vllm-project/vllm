# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dual batch overlap (DBO) in the V2 model runner.

The V2 runner slices the batch *before* building attention metadata, where V1
builds metadata for the whole batch and slices it afterwards. Both must describe
the same microbatches, so the main test here pins V2's slicing against V1's
`split_attn_metadata`. The rest covers the decision to microbatch and the
threaded execution of the microbatches.
"""

import threading
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.forward_context import create_forward_context
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker import gpu_ubatch_wrapper as legacy_gpu_ubatch_wrapper
from vllm.v1.worker.gpu import cp_utils as gpu_cp_utils
from vllm.v1.worker.gpu import cudagraph_utils, dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import (
    UBatchRunner,
    UBatchState,
    _slice_input_batch,
    create_ubatch_slices,
    restore_staged_inputs,
    slice_model_inputs,
    stage_decode_tokens,
)
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    maybe_create_ubatch_slices,
    split_attn_metadata,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id, dbo_yield

MAX_NUM_REQS = 32
MAX_NUM_TOKENS = 128


def _make_buffers() -> InputBuffers:
    return InputBuffers(
        max_num_reqs=MAX_NUM_REQS,
        max_num_tokens=MAX_NUM_TOKENS,
        device=torch.device("cpu"),
    )


def _make_ubatch_buffers(num_ubatches: int = 2) -> list[tuple[torch.Tensor, ...]]:
    """The per-microbatch (query_start_loc, seq_lens) buffers UBatchRunner owns."""
    return [
        (
            torch.zeros(MAX_NUM_REQS + 1, dtype=torch.int32),
            torch.zeros(MAX_NUM_REQS, dtype=torch.int32),
        )
        for _ in range(num_ubatches)
    ]


def _make_input_batch(
    query_lens: list[int],
    seq_lens: list[int],
    buffers: InputBuffers,
    num_reqs_padded: int | None = None,
    num_tokens_padded: int | None = None,
) -> InputBatch:
    """Build an InputBatch shaped like the one `prepare_inputs` produces."""
    num_reqs = len(query_lens)
    num_tokens = int(sum(query_lens))
    num_reqs_padded = num_reqs_padded or num_reqs
    num_tokens_padded = num_tokens_padded or num_tokens

    # make_dummy writes the shared buffers, so fill them afterwards.
    base = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    query_start_loc_np = np.zeros(num_reqs_padded + 1, dtype=np.int32)
    np.cumsum(query_lens, out=query_start_loc_np[1 : num_reqs + 1])
    # Padded entries repeat the token count, as `prepare_inputs` does.
    query_start_loc_np[num_reqs + 1 :] = num_tokens
    buffers.query_start_loc[: num_reqs_padded + 1] = torch.from_numpy(
        query_start_loc_np
    )

    buffers.seq_lens[:num_reqs] = torch.tensor(seq_lens, dtype=torch.int32)
    buffers.seq_lens[num_reqs:num_reqs_padded] = 0

    seq_lens_upper_bound = np.zeros(num_reqs_padded, dtype=np.int32)
    seq_lens_upper_bound[:num_reqs] = seq_lens

    return replace(
        base,
        req_ids=[f"req_{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs_padded,
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
        num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens_padded,
        query_start_loc=buffers.query_start_loc[: num_reqs_padded + 1],
        query_start_loc_np=query_start_loc_np,
        seq_lens=buffers.seq_lens[:num_reqs_padded],
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_upper_bound),
        num_computed_tokens_np=np.array(seq_lens, dtype=np.int32)
        - np.array(query_lens, dtype=np.int32),
        prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
        input_ids=buffers.input_ids[:num_tokens_padded],
        positions=buffers.positions[:num_tokens_padded],
        is_padding=buffers.is_padding[:num_tokens_padded],
    )


# (query_lens, seq_lens): uniform decode, a boundary that lands inside a
# request, and a boundary that lands exactly on a request start.
BATCHES = {
    "uniform_decode": ([1] * 8, [128 + 4 * i for i in range(8)]),
    "split_request": ([1, 1, 10, 1, 1], [64, 96, 512, 32, 48]),
    "aligned_prefills": ([6, 2, 4, 4], [6, 130, 260, 4]),
    "split_first_and_last": ([3, 9, 3], [70, 300, 41]),
    # A long prefill trailing a decode: it straddles the boundary and is also
    # the last request, so the second microbatch holds nothing else.
    "trailing_prefill_spans_boundary": ([2, 20], [64, 300]),
}


@pytest.mark.parametrize("batch_name", list(BATCHES))
def test_slicing_matches_v1_split_attn_metadata(batch_name: str):
    """V2's pre-sliced inputs describe the same microbatches as V1's."""
    query_lens, seq_lens = BATCHES[batch_name]
    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    num_tokens = int(num_scheduled_tokens.sum())

    ubatch_slices, _ = maybe_create_ubatch_slices(
        True, num_scheduled_tokens, num_tokens, len(query_lens), num_ubatches=2
    )
    assert ubatch_slices is not None

    v1_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=16,
        device=torch.device("cpu"),
    )
    v1_ubatches = split_attn_metadata(ubatch_slices, v1_metadata)

    buffers = _make_buffers()
    ubatch_buffers = _make_ubatch_buffers()
    input_batch = _make_input_batch(query_lens, seq_lens, buffers)

    for i, (ubatch_slice, v1_ubatch) in enumerate(zip(ubatch_slices, v1_ubatches)):
        v2_ubatch = _slice_input_batch(input_batch, ubatch_slice, *ubatch_buffers[i])

        assert v2_ubatch.num_reqs == v1_ubatch.num_reqs
        assert v2_ubatch.num_tokens == v1_ubatch.num_actual_tokens
        assert int(v2_ubatch.num_scheduled_tokens.max()) == v1_ubatch.max_query_len
        torch.testing.assert_close(v2_ubatch.query_start_loc, v1_ubatch.query_start_loc)
        torch.testing.assert_close(v2_ubatch.seq_lens, v1_ubatch.seq_lens)
        torch.testing.assert_close(
            torch.from_numpy(v2_ubatch.query_start_loc_np),
            v1_ubatch.query_start_loc_cpu,
        )
        # `prepare_attn` derives max_seq_len from the upper bound. Compare the
        # bound itself rather than V1's max_seq_len: V1 floors it with the full
        # batch's max_seq_len to keep the value CUDA-graph capture installed,
        # which V2 gets from `prepare_attn(for_capture=True)` instead.
        torch.testing.assert_close(
            v2_ubatch.seq_lens_cpu_upper_bound, v1_ubatch.seq_lens_cpu_upper_bound
        )


def test_microbatches_do_not_share_buffers():
    """Both microbatches are live at once, so their buffers must be distinct."""
    query_lens, seq_lens = BATCHES["split_request"]
    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    num_tokens = int(num_scheduled_tokens.sum())

    ubatch_slices, _ = maybe_create_ubatch_slices(
        True, num_scheduled_tokens, num_tokens, len(query_lens), num_ubatches=2
    )
    assert ubatch_slices is not None

    buffers = _make_buffers()
    ubatch_buffers = _make_ubatch_buffers()
    input_batch = _make_input_batch(query_lens, seq_lens, buffers)

    first = _slice_input_batch(input_batch, ubatch_slices[0], *ubatch_buffers[0])
    first_query_start_loc = first.query_start_loc.clone()
    first_seq_lens = first.seq_lens.clone()

    # Slicing the second microbatch must not disturb the first.
    _slice_input_batch(input_batch, ubatch_slices[1], *ubatch_buffers[1])

    torch.testing.assert_close(first.query_start_loc, first_query_start_loc)
    torch.testing.assert_close(first.seq_lens, first_seq_lens)


def test_trailing_microbatch_absorbs_cudagraph_padding():
    """The padded microbatch keeps padded rows empty and query_start_loc flat."""
    query_lens, seq_lens = [1] * 6, [64] * 6
    num_tokens_padded = 8

    buffers = _make_buffers()
    input_batch = _make_input_batch(
        query_lens,
        seq_lens,
        buffers,
        num_reqs_padded=num_tokens_padded,
        num_tokens_padded=num_tokens_padded,
    )
    ubatch_slices_padded = create_ubatch_slices(input_batch, num_ubatches=2)

    last = _slice_input_batch(
        input_batch, ubatch_slices_padded[-1], *_make_ubatch_buffers()[1]
    )

    # Two real decodes plus two padded rows.
    assert last.num_reqs == 2
    assert last.num_tokens == 2
    assert last.num_reqs_after_padding == 4
    assert last.num_tokens_after_padding == 4
    torch.testing.assert_close(
        last.query_start_loc, torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        last.seq_lens, torch.tensor([64, 64, 0, 0], dtype=torch.int32)
    )


DECODE_THRESHOLD = 32
PREFILL_THRESHOLD = 128
DECODE_QUERY_LEN = 1


def test_dummy_batch_can_be_microbatched():
    """A DP rank with no work still has to split, so its dummy batch must slice.

    Microbatching is all-or-nothing across the group, so a rank whose step is a
    dummy batch runs it microbatched like everyone else.
    """
    num_reqs, num_tokens = 1, 32
    buffers = _make_buffers()
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    ubatch_slices_padded = create_ubatch_slices(input_batch, num_ubatches=2)

    ubatch_buffers = _make_ubatch_buffers()
    ubatches = [
        _slice_input_batch(input_batch, ubatch_slice, *ubatch_buffers[i])
        for i, ubatch_slice in enumerate(ubatch_slices_padded)
    ]

    assert [u.num_tokens for u in ubatches] == [16, 16]
    # The one dummy request spans both microbatches, 16 of its tokens in each.
    assert all(u.num_reqs == 1 for u in ubatches)
    for ubatch in ubatches:
        torch.testing.assert_close(
            ubatch.query_start_loc, torch.tensor([0, 16], dtype=torch.int32)
        )


def _sync_dp(
    num_tokens_per_rank: list[int],
    uniform_token_count_per_rank: list[int] | None = None,
    allow_ubatching: bool = True,
    cudagraph_manager: Any = None,
    num_reqs_per_rank: list[int] | None = None,
    num_ubatches: int = 2,
    dp_rank: int = 0,
) -> tuple[BatchExecutionDescriptor, dp_utils.DPSyncState | None]:
    """Run the DP handshake with the all-reduce stubbed out.

    The microbatching decision lives inside `sync_cudagraph_and_dp_padding`, so
    the cases below stand in for the collective rather than for the decision,
    and need no process group. Every rank asks for eager, which keeps the
    non-microbatched path off the cudagraph manager. This rank is rank 0.
    """
    dp_size = len(num_tokens_per_rank)
    uniform_token_counts = uniform_token_count_per_rank or [0] * dp_size
    reduced = torch.zeros(6, dp_size, dtype=torch.int32)
    reduced[0] = torch.tensor(num_tokens_per_rank, dtype=torch.int32)
    reduced[1] = CUDAGraphMode.NONE.value
    reduced[2] = torch.tensor(uniform_token_counts, dtype=torch.int32)
    reduced[3] = -1  # max_query_len, -1 means None
    reduced[4] = int(allow_ubatching)
    reduced[5] = torch.tensor(num_reqs_per_rank) if num_reqs_per_rank is not None else 8

    with (
        patch.object(dp_utils.dist, "all_reduce", lambda t, group: t.copy_(reduced)),
        patch.object(dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=None)),
    ):
        return dp_utils.sync_cudagraph_and_dp_padding(
            cudagraph_manager=cudagraph_manager,
            desired_batch_desc=BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens_per_rank[dp_rank],
                num_reqs=8,
            ),
            num_tokens=num_tokens_per_rank[dp_rank],
            num_reqs=8,
            uniform_token_count=uniform_token_counts[dp_rank] or None,
            dp_size=dp_size,
            dp_rank=dp_rank,
            parallel_config=ParallelConfig(
                enable_dbo=num_ubatches == 2,
                ubatch_size=num_ubatches,
                dbo_decode_token_threshold=DECODE_THRESHOLD,
                dbo_prefill_token_threshold=PREFILL_THRESHOLD,
            ),
            allow_ubatching=allow_ubatching,
            uniform_decode=uniform_token_counts[dp_rank] == DECODE_QUERY_LEN,
        )


def test_every_dp_rank_must_agree_to_microbatch():
    """One rank below the threshold is enough to keep the group on one batch.

    The thresholds are not communicated -- they are checked against the token
    counts the all-reduce already carries -- so this also pins that the whole
    vector is consulted, not just this rank's own entry.
    """
    assert _sync_dp([256, 256])[0].num_ubatches == 2
    assert _sync_dp([256, 100])[0].num_ubatches == 1
    assert _sync_dp([100, 256])[0].num_ubatches == 1


def test_decode_threshold_only_applies_when_every_rank_is_a_uniform_decode():
    """Otherwise the group is held to the prefill threshold."""
    # 64 tokens clears the decode threshold but not the prefill one, so it only
    # microbatches when both ranks are uniform decodes.
    assert _sync_dp([64, 64], uniform_token_count_per_rank=[1, 1])[0].num_ubatches == 2
    assert _sync_dp([64, 64], uniform_token_count_per_rank=[1, 0])[0].num_ubatches == 1


def test_microbatching_pads_all_ranks_to_the_largest():
    """Ranks must run the same token count so microbatch sizes line up."""
    desc, dp_sync = _sync_dp([200, 256])
    assert desc.num_tokens == 256
    assert dp_sync is not None
    assert dp_sync.num_tokens_across_dp.tolist() == [256, 256]


def test_microbatching_survives_a_rank_that_cannot_fill_it():
    """A rank whose real tokens all land in the first microbatch still splits.

    Rank 0 has 128 real tokens but the split point is at 512 // 2 = 256, so its
    second microbatch is pure padding. That microbatch does no work, which is
    fine -- it still has to run so the expert all-to-all stays collective.
    """
    desc, _ = _sync_dp([128, 512])
    assert desc.num_tokens == 512
    assert desc.num_ubatches == 2


def test_all_padding_microbatch_has_no_work_to_do():
    """The microbatch past the last real token is well-formed and empty.

    It carries the last request with none of its query tokens, so attention and
    sampling see nothing to do while the microbatch still runs.
    """
    query_lens, seq_lens = [1] * 6, [64] * 6
    num_tokens_padded = 16  # split at 8, past all 6 real tokens

    buffers = _make_buffers()
    input_batch = _make_input_batch(
        query_lens, seq_lens, buffers, num_tokens_padded=num_tokens_padded
    )
    ubatch_slices_padded = create_ubatch_slices(input_batch, num_ubatches=2)
    ubatch_buffers = _make_ubatch_buffers()

    first = _slice_input_batch(input_batch, ubatch_slices_padded[0], *ubatch_buffers[0])
    last = _slice_input_batch(input_batch, ubatch_slices_padded[1], *ubatch_buffers[1])

    # The first microbatch keeps every real request; the second gets none.
    assert first.num_reqs == 6
    assert first.num_tokens == 6
    assert last.num_tokens == 0
    assert last.num_reqs == 1
    assert last.num_reqs_after_padding == 1
    assert last.num_tokens_after_padding == 8

    # Zero query tokens for the one request it holds, so no attention work.
    torch.testing.assert_close(
        last.query_start_loc, torch.tensor([0, 0], dtype=torch.int32)
    )
    assert last.num_scheduled_tokens.tolist() == [0]
    # ...and its sequence still ends where its tokens do: they all live in the
    # first microbatch, which computes them before this one runs.
    torch.testing.assert_close(last.seq_lens, torch.tensor([64], dtype=torch.int32))
    torch.testing.assert_close(
        last.seq_lens_cpu_upper_bound, torch.tensor([64], dtype=torch.int32)
    )


def test_microbatching_off_when_a_rank_does_not_allow_it():
    assert _sync_dp([256, 256], allow_ubatching=False)[0].num_ubatches == 1


def test_slice_model_inputs_handles_mrope_positions():
    model_inputs = {
        "input_ids": torch.arange(8),
        # M-RoPE positions carry a leading section dim.
        "positions": torch.arange(24).view(3, 8),
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }
    sliced = slice_model_inputs(model_inputs, slice(4, 8))

    torch.testing.assert_close(sliced["input_ids"], torch.arange(4, 8))
    torch.testing.assert_close(sliced["positions"], torch.arange(24).view(3, 8)[:, 4:8])
    assert sliced["inputs_embeds"] is None
    assert sliced["intermediate_tensors"] is None


@pytest.mark.parametrize(
    ("cudagraph_runtime_mode", "execution_method"),
    [
        (CUDAGraphMode.FULL, "_capture_ubatches"),
        (CUDAGraphMode.NONE, "_run_ubatches"),
    ],
)
def test_legacy_ubatch_contexts_slice_padding_mask(
    cudagraph_runtime_mode: CUDAGraphMode, execution_method: str
):
    """Replacement contexts preserve each microbatch's padding rows."""
    ubatch_slices = [
        UBatchSlice(slice(0, 2), slice(0, 3)),
        UBatchSlice(slice(2, 4), slice(3, 6)),
    ]
    is_padding = torch.tensor([False, True, False, True, False, True])
    forward_context = SimpleNamespace(
        batch_descriptor=object(),
        ubatch_slices=ubatch_slices,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        attn_metadata=[{}, {}],
        slot_mapping={},
        dp_metadata=object(),
        is_padding=is_padding,
    )

    wrapper = object.__new__(legacy_gpu_ubatch_wrapper.UBatchWrapper)
    wrapper.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2)
    )
    wrapper.comm_stream = object()
    wrapper.ready_barrier = object()
    wrapper.cudagraphs = {}
    wrapper.sm_control = nullcontext()
    wrapper.runnable = object()

    def make_contexts(*, forward_contexts, **kwargs):
        return [
            SimpleNamespace(id=i, forward_context=context)
            for i, context in enumerate(forward_contexts)
        ]

    def create_context(attn_metadata, vllm_config, **kwargs):
        return SimpleNamespace(attn_metadata=attn_metadata, **kwargs)

    model_inputs = {
        "input_ids": torch.arange(6),
        "positions": torch.arange(6),
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }
    expected = object()
    with (
        patch.object(
            legacy_gpu_ubatch_wrapper,
            "get_forward_context",
            return_value=forward_context,
        ),
        patch.object(
            legacy_gpu_ubatch_wrapper.torch.cuda,
            "current_stream",
            return_value=object(),
        ),
        patch.object(
            legacy_gpu_ubatch_wrapper.DPMetadata, "make", return_value=object()
        ),
        patch.object(
            legacy_gpu_ubatch_wrapper,
            "create_forward_context",
            side_effect=create_context,
        ),
        patch.object(
            legacy_gpu_ubatch_wrapper,
            "make_ubatch_contexts",
            side_effect=make_contexts,
        ),
        patch.object(wrapper, execution_method, return_value=expected) as execute,
    ):
        result = wrapper(**model_inputs)

    assert result is expected
    contexts = [
        metadata.context.forward_context for metadata in execute.call_args.args[0]
    ]
    torch.testing.assert_close(contexts[0].is_padding, is_padding[:3])
    torch.testing.assert_close(contexts[1].is_padding, is_padding[3:])


def _make_dbo_config() -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", dtype="float16", seed=0),
        parallel_config=ParallelConfig(
            enable_dbo=True, all2all_backend="deepep_low_latency"
        ),
    )


@pytest.mark.parametrize("graph_size", [127, 128, 129])
@pytest.mark.parametrize("num_ubatches", [2, 3, 4])
@torch.inference_mode()
def test_flash_attention_staging_multistep_graph(
    graph_size, num_ubatches, tmp_path, monkeypatch
):
    """Real FA3 metadata/replay preserves outputs and every KV slot across layouts.

    This isolates attention from MoE communication. Production capability must
    enable the real FA3 builder; the test does not establish DP support.
    """
    from transformers import OPTConfig

    from tests.kernels.attention.test_flash_attn import ref_paged_attn
    from vllm.config import AttentionConfig, SchedulerConfig, set_current_vllm_config
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionBackend,
        FlashAttentionImpl,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    from vllm.v1.worker.utils import AttentionGroup
    from vllm.vllm_flash_attn import is_fa_version_supported

    if not torch.cuda.is_available() or not is_fa_version_supported(3):
        pytest.skip("This staging integration test requires CUDA and FA3")
    device = torch.device("cuda:0")
    torch.manual_seed(57184)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    OPTConfig(
        hidden_size=256,
        word_embed_proj_dim=256,
        num_attention_heads=4,
        num_hidden_layers=1,
        ffn_dim=512,
        vocab_size=256,
        max_position_embeddings=64,
        architectures=["OPTForCausalLM"],
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(model=str(tmp_path), dtype="bfloat16", seed=0),
        parallel_config=ParallelConfig(
            ubatch_size=num_ubatches, all2all_backend="nixl_ep"
        ),
        attention_config=AttentionConfig(flash_attn_version=3),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=64,
            max_num_seqs=graph_size,
            max_num_batched_tokens=graph_size,
        ),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=[graph_size],
            max_cudagraph_capture_size=graph_size,
        ),
    )
    block_size, blocks_per_req, heads, kv_heads, dim = 16, 4, 4, 2, 64
    num_blocks = graph_size * blocks_per_req + 2
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=kv_heads,
        head_size=dim,
        dtype=torch.bfloat16,
    )
    cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["attn"], spec)],
    )
    with set_current_vllm_config(config):
        impl = FlashAttentionImpl(heads, dim, dim**-0.5, kv_heads, None, None, "auto")
        group = AttentionGroup(FlashAttentionBackend, ["attn"], spec, 0)
        group.create_metadata_builders(
            config, device, num_metadata_builders=num_ubatches
        )
        ref_group = AttentionGroup(FlashAttentionBackend, ["attn"], spec, 0)
        ref_group.create_metadata_builders(config, device)
        model_state = DefaultModelState(config, torch.nn.Identity(), None, device)
        runner = UBatchRunner(
            config, device, model_state, [[group]], cache_config, graph_size
        )
    assert impl.vllm_flash_attn_version == 3, "Test must exercise real FA3 scheduling"
    assert all(
        b.aot_schedule and b.use_full_cuda_graph for b in group.metadata_builders
    )
    assert runner.stage_real_tokens, runner._real_token_staging_unsupported_reason()
    # DPMetadata.make requires a distributed group; this test runs attention only.
    # Keep the real prepare/build path and omit just the distributed wrappers.
    monkeypatch.setattr(
        runner,
        "_make_forward_contexts",
        lambda slices, metadata, mappings, padding: [
            SimpleNamespace(attn_metadata=m) for m in metadata
        ],
    )
    layer = torch.nn.Module()
    layer._q_scale = layer._k_scale = layer._v_scale = torch.ones((), device=device)
    buffers = InputBuffers(graph_size, graph_size, device)
    ref_buffers = InputBuffers(graph_size, graph_size, device)
    block_table = torch.zeros(
        (graph_size, blocks_per_req), dtype=torch.int32, device=device
    )
    slots = torch.full((1, graph_size), -1, dtype=torch.int64, device=device)
    # Disjoint, shuffled blocks plus two guard blocks; no accidental prefix mapping.
    request_blocks = (
        (torch.randperm(graph_size * blocks_per_req, device=device) + 1)
        .reshape(graph_size, blocks_per_req)
        .int()
    )
    lengths = np.arange(graph_size) % 27 + 7
    cache = torch.randn(
        num_blocks,
        kv_heads,
        block_size,
        2 * dim,
        dtype=torch.bfloat16,
        device=device,
    )
    reference_cache = cache.clone()
    expected_cache = cache.clone()
    q_bank = torch.randn(graph_size + 1, heads, dim, dtype=cache.dtype, device=device)
    k_bank = torch.randn(
        graph_size + 1, kv_heads, dim, dtype=cache.dtype, device=device
    )
    v_bank = torch.randn_like(k_bank)
    output = torch.empty_like(q_bank[:graph_size])

    def prepare(n, step, *, capture=False):
        # Flip request order, not just counts, to expose stale per-request metadata.
        reqs = np.arange(graph_size)[:: -1 if step % 2 else 1][:n].copy()
        req_gpu = torch.tensor(reqs, device=device)
        lens = lengths[reqs] + 1
        batch = _make_input_batch(
            [1] * n, lens.tolist(), buffers, graph_size, graph_size
        )
        buffers.input_ids.zero_()
        buffers.input_ids[:n] = req_gpu + 1
        buffers.positions.zero_()
        buffers.positions[:n] = torch.tensor(lens - 1, device=device)
        buffers.is_padding.fill_(True)
        buffers.is_padding[:n] = False
        block_table.zero_()
        block_table[:n] = request_blocks[req_gpu]
        slot_positions = buffers.positions[:n]
        logical_slots = (
            block_table[
                torch.arange(n, device=device), slot_positions // block_size
            ].long()
            * block_size
            + slot_positions % block_size
        )
        slots.fill_(-1)
        slots[0, :n] = logical_slots
        logical_blocks = block_table[:n].clone()
        state = runner.prepare(
            batch, (block_table,), slots, CUDAGraphMode.FULL, for_capture=capture
        )
        return batch, state, reqs, req_gpu, lens, logical_slots, logical_blocks

    batch, captured, *_ = prepare(graph_size, 0, capture=True)

    def run_attention():
        # Model projections would consume staged input rows in this same order.
        q, key, value = (bank[buffers.input_ids] for bank in (q_bank, k_bank, v_bank))
        for region, context in zip(captured.slices, captured.forward_contexts):
            s = region.token_slice
            metadata = context.attn_metadata["attn"]
            impl.do_kv_cache_update(layer, key[s], value[s], cache, slots[0, s])
            impl.forward(layer, q[s], key[s], value[s], cache, metadata, output[s])

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run_attention()
        run_attention()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run_attention()
    torch.cuda.current_stream().wait_stream(stream)
    cache.copy_(reference_cache)  # Discard warmup/capture writes only.
    last_start = captured.slices[-1].token_slice.start

    def metadata_pointers(state):
        return [
            tuple(
                getattr(ctx.attn_metadata["attn"], name).data_ptr()
                for name in (
                    "query_start_loc",
                    "seq_lens",
                    "block_table",
                    "scheduler_metadata",
                )
            )
            for ctx in state.forward_contexts
        ]

    pointers = metadata_pointers(captured)
    input_pointers = [
        x.data_ptr()
        for x in (
            buffers.input_ids,
            buffers.positions,
            buffers.is_padding,
            block_table,
            slots,
        )
    ]
    # Alternating staged/ordinary FULL steps, including equality and odd capacities.
    counts = [
        num_ubatches,
        last_start + 1,
        last_start,
        graph_size,
        last_start - 1,
        num_ubatches + 1,
        graph_size,
        last_start,
    ]
    max_error = 0.0
    max_oracle_error = 0.0
    staged_steps = 0
    for step, n in enumerate(counts):
        batch, state, reqs, req_gpu, lens, logical_slots, logical_blocks = prepare(
            n, step
        )
        staged_rows = state.staged_rows
        assert (staged_rows is not None) == (n <= last_start)
        assert metadata_pointers(state) == pointers, "Captured metadata address changed"
        if staged_rows is not None:
            staged_steps += 1
            expected_rows = [
                row
                for i, region in enumerate(captured.slices)
                for row in range(
                    region.token_slice.start,
                    region.token_slice.start
                    + n // num_ubatches
                    + (i < n % num_ubatches),
                )
            ]
            assert staged_rows.tolist() == expected_rows
            logical_start = 0
            for i, (region, ctx) in enumerate(
                zip(state.slices, state.forward_contexts)
            ):
                count = n // num_ubatches + (i < n % num_ubatches)
                capacity = region.num_tokens
                meta = ctx.attn_metadata["attn"]
                assert meta.query_start_loc.tolist() == [
                    min(j, count) for j in range(capacity + 1)
                ], f"query offsets: step={step}, ubatch={i}"
                assert meta.seq_lens.tolist() == (
                    lens[logical_start : logical_start + count].tolist()
                    + [0] * (capacity - count)
                ), f"sequence lengths: step={step}, ubatch={i}"
                logical_start += count
        q, key, value = (bank[req_gpu + 1] for bank in (q_bank, k_bank, v_bank))
        ref_batch = _make_input_batch([1] * n, lens.tolist(), ref_buffers)
        ref_metadata = model_state.prepare_attn(
            ref_batch,
            CUDAGraphMode.NONE,
            (logical_blocks,),
            logical_slots[None],
            [[ref_group]],
            cache_config,
        )["attn"]
        ref_output = torch.empty_like(q)
        impl.do_kv_cache_update(layer, key, value, reference_cache, logical_slots)
        impl.forward(layer, q, key, value, reference_cache, ref_metadata, ref_output)
        # An independent cache oracle catches shared mistakes in both kernel paths.
        expected_cache[
            logical_slots // block_size, :, logical_slots % block_size, :dim
        ] = key
        expected_cache[
            logical_slots // block_size, :, logical_slots % block_size, dim:
        ] = value
        graph.replay()
        torch.accelerator.synchronize()
        if staged_rows is not None:
            output[:n] = output[staged_rows]
            restore_staged_inputs(batch, (block_table,), slots, staged_rows)
        label = f"N={graph_size}, k={num_ubatches}, step={step}, n={n}"
        torch.testing.assert_close(
            output[:n], ref_output, atol=1.5e-2, rtol=1e-2, msg=label
        )
        oracle_k, oracle_v = expected_cache.transpose(1, 2).split(dim, dim=-1)
        with torch.device(device):
            oracle_output = ref_paged_attn(
                q.clone(),
                oracle_k.contiguous(),
                oracle_v.contiguous(),
                [1] * n,
                lens.tolist(),
                logical_blocks,
                dim**-0.5,
            )
        torch.testing.assert_close(
            output[:n], oracle_output, atol=1.5e-2, rtol=1e-2, msg=label
        )
        torch.testing.assert_close(
            reference_cache, expected_cache, atol=0, rtol=0, msg=label
        )
        torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0, msg=label)
        assert buffers.input_ids[:n].tolist() == (req_gpu + 1).tolist()
        assert buffers.positions[:n].tolist() == (lens - 1).tolist()
        assert not buffers.is_padding[:n].any()
        assert torch.equal(block_table[:n], logical_blocks)
        assert torch.equal(slots[0, :n], logical_slots)
        assert not block_table[n:].count_nonzero()
        assert torch.all(slots[:, n:] == -1)
        assert not buffers.input_ids[n:].count_nonzero()
        assert not buffers.positions[n:].count_nonzero()
        assert buffers.is_padding[n:].all()
        assert [
            x.data_ptr()
            for x in (
                buffers.input_ids,
                buffers.positions,
                buffers.is_padding,
                block_table,
                slots,
            )
        ] == input_pointers
        max_error = max(max_error, (output[:n] - ref_output).abs().max().item())
        max_oracle_error = max(
            max_oracle_error, (output[:n] - oracle_output).abs().max().item()
        )
        lengths[reqs] += 1
    print(
        f"FA3 N={graph_size} k={num_ubatches}: captures=1 replays={len(counts)} "
        f"staged={staged_steps} max_abs_vs_compact={max_error} "
        f"max_abs_vs_torch={max_oracle_error} full_KV_exact=True"
    )


def _make_model_inputs(num_tokens: int, device: torch.device) -> dict[str, Any]:
    return {
        "input_ids": torch.arange(num_tokens, device=device),
        "positions": torch.arange(num_tokens, device=device),
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }


@pytest.fixture(autouse=True)
def _no_comm_sm_reservation(monkeypatch):
    """CI runs on MIG slices with fewer SMs than the default comm reservation."""
    monkeypatch.setenv("VLLM_DBO_COMM_SMS", "0")


def _make_execution_runner(vllm_config: VllmConfig) -> UBatchRunner:
    """A runner for the execution tests below, which never call `prepare()`.

    The attention-side dependencies are only read while building per-microbatch
    metadata, so they are left out here.
    """
    return UBatchRunner(
        vllm_config,
        torch.device("cuda:0"),
        model_state=cast(ModelState, None),
        attn_groups=[],
        kv_cache_config=cast(KVCacheConfig, None),
        max_num_reqs=MAX_NUM_REQS,
    )


def _make_ubatch_state(
    vllm_config: VllmConfig, ubatch_slices: list[UBatchSlice]
) -> UBatchState:
    return UBatchState(
        slices=ubatch_slices,
        forward_contexts=[
            create_forward_context(None, vllm_config) for _ in ubatch_slices
        ],
    )


def test_thresholds_below_the_microbatch_count_are_rejected():
    """A batch with fewer tokens than microbatches cannot be split at all.

    Nothing downstream copes with an unsplittable batch, so the thresholds are
    what keep one out: a one-token decode split two ways leaves an empty
    microbatch.
    """
    with pytest.raises(ValueError, match="number of microbatches"):
        ParallelConfig(enable_dbo=True, dbo_decode_token_threshold=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DBO needs a GPU")
def test_runner_allocates_one_buffer_pair_per_microbatch():
    """All microbatches are live at once, so none may share rebase buffers."""
    vllm_config = _make_dbo_config()
    runner = _make_execution_runner(vllm_config)

    for buffers in (runner.ubatch_query_start_loc, runner.ubatch_seq_lens):
        assert len(buffers) == runner.num_ubatches
        storages = {b.untyped_storage().data_ptr() for b in buffers}
        assert len(storages) == runner.num_ubatches


class _YieldingModel(torch.nn.Module):
    """Toy model that hands off to the other microbatch mid-forward."""

    def __init__(self, trace: list[tuple[str, int]]):
        super().__init__()
        self.trace = trace

    def forward(self, input_ids, positions, intermediate_tensors=None, **kwargs):
        self.trace.append(("enter", dbo_current_ubatch_id()))
        out = input_ids.float().unsqueeze(-1) * 2 + positions.float().unsqueeze(-1)
        # Stands in for the expert all-to-all handoff point.
        dbo_yield()
        self.trace.append(("exit", dbo_current_ubatch_id()))
        return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DBO needs a GPU")
def test_ubatch_runner_overlaps_and_matches_single_batch():
    """Microbatches interleave at the yield point and produce the same output."""
    vllm_config = _make_dbo_config()
    device = torch.device("cuda:0")
    runner = _make_execution_runner(vllm_config)

    model_inputs = _make_model_inputs(16, device)
    ubatch_state = _make_ubatch_state(
        vllm_config,
        [
            UBatchSlice(slice(0, 4), slice(0, 8)),
            UBatchSlice(slice(4, 8), slice(8, 16)),
        ],
    )

    trace: list[tuple[str, int]] = []
    model = _YieldingModel(trace)
    output = runner.run(model, model_inputs, ubatch_state)

    expected = model_inputs["input_ids"].float().unsqueeze(-1) * 2 + model_inputs[
        "positions"
    ].float().unsqueeze(-1)
    torch.testing.assert_close(output, expected)

    # Both microbatches reach the handoff before either finishes.
    assert trace == [("enter", 0), ("enter", 1), ("exit", 0), ("exit", 1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DBO needs a GPU")
def test_ubatch_runner_names_the_microbatch_that_failed():
    """A failing microbatch surfaces as a named error, not a downstream KeyError.

    The model here never yields, so the microbatches run back to back and no
    handoff is left outstanding when one of them dies. A microbatch that fails
    *while its sibling is parked at a yield* hangs the step instead -- the
    shared handoff protocol in `ubatching.py` has no way to unwind a parked
    microbatch, and the V1 runner has the same gap. Fixing that means changing
    `ubatching.py`, which is out of scope for the V2 runner.
    """
    vllm_config = _make_dbo_config()
    device = torch.device("cuda:0")
    runner = _make_execution_runner(vllm_config)

    class _FailingModel(torch.nn.Module):
        def forward(self, input_ids, positions, **kwargs):
            if dbo_current_ubatch_id() == 0:
                raise ValueError("boom")
            return input_ids.float().unsqueeze(-1)

    model_inputs = _make_model_inputs(8, device)
    ubatch_state = _make_ubatch_state(
        vllm_config,
        [
            UBatchSlice(slice(0, 2), slice(0, 4)),
            UBatchSlice(slice(2, 4), slice(4, 8)),
        ],
    )

    result: dict[str, BaseException] = {}

    def _call():
        try:
            runner.run(_FailingModel(), model_inputs, ubatch_state)
        except BaseException as e:  # noqa: BLE001
            result["error"] = e

    # Run behind a watchdog: `UBatchRunner.run` joins without a timeout, so a
    # regression here would wedge the suite rather than fail it.
    caller = threading.Thread(target=_call, daemon=True)
    caller.start()
    caller.join(timeout=60.0)
    assert not caller.is_alive(), "UBatchRunner.run hung on a failing microbatch"

    assert isinstance(result["error"], RuntimeError)
    assert "Microbatch 0" in str(result["error"])
    assert isinstance(result["error"].__cause__, ValueError)


# DCP x DBO: each microbatch recomputes DCP-local seq_lens from its own
# truncated seq_lens into its own persistent buffer (the parent's values for
# a straddling request are wrong for the leading microbatch).

DCP_SIZE = 2
CP_INTERLEAVE = 1


def _make_cuda_input_batch(
    query_lens: list[int], seq_lens: list[int]
) -> tuple[InputBatch, InputBuffers]:
    buffers = InputBuffers(
        max_num_reqs=MAX_NUM_REQS,
        max_num_tokens=MAX_NUM_TOKENS,
        device=torch.device("cuda:0"),
    )
    return _make_input_batch(query_lens, seq_lens, buffers), buffers


def _make_dcp_ubatch_buffers(
    num_ubatches: int = 2,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per-microbatch (query_start_loc, seq_lens, dcp_local_seq_lens)."""
    return [
        (
            torch.zeros(MAX_NUM_REQS + 1, dtype=torch.int32, device="cuda:0"),
            torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device="cuda:0"),
            torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device="cuda:0"),
        )
        for _ in range(num_ubatches)
    ]


def _slice_with_dcp(
    input_batch: InputBatch, dcp_rank: int, num_ubatches: int = 2
) -> tuple[list[InputBatch], list[torch.Tensor]]:
    dcp_buffers = [
        torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device="cuda:0")
        for _ in range(num_ubatches)
    ]
    ubatch_buffers = _make_dcp_ubatch_buffers(num_ubatches)
    ubatches = [
        _slice_input_batch(
            input_batch,
            ubatch_slice,
            ubatch_buffers[i][0],
            ubatch_buffers[i][1],
            dcp_buffers[i],
            dcp_size=DCP_SIZE,
            dcp_rank=dcp_rank,
            cp_interleave=CP_INTERLEAVE,
        )
        for i, ubatch_slice in enumerate(
            create_ubatch_slices(input_batch, num_ubatches)
        )
    ]
    return ubatches, dcp_buffers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel needs CUDA")
@pytest.mark.parametrize("dcp_rank", [0, 1])
def test_microbatches_recompute_dcp_lens_from_truncated_seq_lens(dcp_rank: int):
    """A boundary-split request keeps a truncated seq_len on the leading
    microbatch, whose DCP-local length may differ (in parity) from the full
    batch's. Microbatches must therefore derive DCP lengths from their own
    seq_lens, not reuse the parent's row.
    """
    input_batch, buffers = _make_cuda_input_batch(
        [1, 1, 10, 1, 1], [64, 96, 512, 32, 48]
    )
    # execute_model has populated the merged batch's DCP metadata already.
    input_batch.dcp_local_seq_lens = gpu_cp_utils.prepare_dcp_local_seq_lens(
        buffers.dcp_local_seq_lens,
        input_batch.seq_lens,
        input_batch.num_reqs,
        DCP_SIZE,
        dcp_rank,
        CP_INTERLEAVE,
        num_reqs_padded=input_batch.num_reqs_after_padding,
    )
    parent_lens = input_batch.dcp_local_seq_lens.clone()

    ubatches, _ = _slice_with_dcp(input_batch, dcp_rank)

    for ubatch in ubatches:
        assert ubatch.dcp_local_seq_lens is not None
        expected = get_dcp_local_seq_lens(
            ubatch.seq_lens.cpu(), DCP_SIZE, dcp_rank, CP_INTERLEAVE
        )
        assert torch.equal(
            ubatch.dcp_local_seq_lens[: ubatch.num_reqs].cpu(),
            expected.to(torch.int32),
        )

    # The straddling request (512 tokens, 5 of its 10 query tokens truncated)
    # lands on this rank with a different local length than the full batch
    # reports (507 vs 512 -> 254 vs 256 for dcp_size=2, interleave=1); copying
    # the parent's row would ship the stale value.
    assert ubatches[0].num_reqs == 3
    assert ubatches[0].seq_lens[2].item() == 507
    assert ubatches[0].dcp_local_seq_lens[2].item() != parent_lens[2].item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel needs CUDA")
def test_each_microbatch_owns_its_dcp_buffer():
    """Microbatch DCP lengths live in per-microbatch persistent buffers so
    CPU-visible shapes stay valid under CUDA graph capture and replay."""
    input_batch, _ = _make_cuda_input_batch([4, 4, 4, 4], [8, 9, 10, 11])

    ubatches, dcp_buffers = _slice_with_dcp(input_batch, dcp_rank=0)

    for ubatch, dcp_buffer in zip(ubatches, dcp_buffers):
        assert ubatch.dcp_local_seq_lens is not None
        assert ubatch.dcp_local_seq_lens.data_ptr() == dcp_buffer.data_ptr()
    assert dcp_buffers[0].data_ptr() != dcp_buffers[1].data_ptr()


def test_slicing_drops_stale_dcp_metadata_when_dcp_is_off():
    """dcp_size == 1 yields None even when the parent carries a value, so a
    microbatch can never consume DCP metadata this deployment did not
    enable.
    """
    buffers = _make_buffers()
    input_batch = _make_input_batch([4, 4, 4, 4], [8, 9, 10, 11], buffers)
    input_batch.dcp_local_seq_lens = buffers.dcp_local_seq_lens[:4]

    ubatch_buffers = _make_ubatch_buffers()
    ubatches = [
        _slice_input_batch(input_batch, ubatch_slice, *ubatch_buffers[i])
        for i, ubatch_slice in enumerate(create_ubatch_slices(input_batch, 2))
    ]

    assert all(ubatch.dcp_local_seq_lens is None for ubatch in ubatches)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DBO needs a GPU")
def test_capturable_run_replays_as_a_cudagraph(monkeypatch):
    """Microbatched graph replay reads updated persistent input buffers."""

    def unexpected_staging(*args, **kwargs):
        raise AssertionError("ordinary graph execution must not stage or restore")

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.ubatch_utils.stage_decode_tokens", unexpected_staging
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.ubatch_utils.restore_staged_inputs", unexpected_staging
    )
    vllm_config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", dtype="float16", seed=0),
        parallel_config=ParallelConfig(
            enable_dbo=True, all2all_backend="deepep_low_latency"
        ),
    )
    device = torch.device("cuda:0")
    runner = _make_execution_runner(vllm_config)

    num_tokens = 16
    input_ids = torch.arange(num_tokens, device=device)
    positions = torch.arange(num_tokens, device=device)
    model_inputs = {
        "input_ids": input_ids,
        "positions": positions,
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }
    ubatch_state = _make_ubatch_state(
        vllm_config,
        [
            UBatchSlice(slice(0, 4), slice(0, 8)),
            UBatchSlice(slice(4, 8), slice(8, 16)),
        ],
    )

    model = _YieldingModel([])
    graph = torch.cuda.CUDAGraph()
    finish = runner.begin_capturable_run(
        model, model_inputs, ubatch_state, for_capture=True
    )
    with torch.cuda.graph(graph, stream=runner.capture_stream):
        captured = finish()

    input_ids.fill_(3)
    positions.fill_(5)
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=num_tokens,
        num_reqs=8,
        num_ubatches=2,
    )
    manager = object.__new__(CudaGraphManager)
    manager.graphs = {desc: graph}
    manager.run_fullgraph(desc)
    torch.accelerator.synchronize()

    expected = torch.full(
        (num_tokens, 1), 3 * 2 + 5, dtype=torch.float32, device=device
    )
    torch.testing.assert_close(captured, expected)


def _request_slices(input_batch: InputBatch) -> list[slice]:
    return [s.request_slice for s in create_ubatch_slices(input_batch, 2)]


def test_captured_split_survives_a_replay_with_enough_requests():
    """Replay preserves the captured split only if real requests reach it."""
    buffers = _make_buffers()
    captured = _request_slices(InputBatch.make_dummy(16, 16, buffers))

    def replay(num_reqs: int) -> list[slice]:
        return _request_slices(
            _make_input_batch(
                [1] * num_reqs,
                [128] * num_reqs,
                buffers,
                num_reqs_padded=16,
                num_tokens_padded=16,
            )
        )

    assert replay(16) == captured
    assert replay(9) == captured
    assert replay(4) != captured


def _make_cudagraph_manager(capture_sizes: list[int]) -> CudaGraphManager:
    vllm_config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", dtype="float16", seed=0),
        parallel_config=ParallelConfig(
            enable_dbo=True,
            all2all_backend="deepep_low_latency",
            dbo_decode_token_threshold=DECODE_THRESHOLD,
            dbo_prefill_token_threshold=PREFILL_THRESHOLD,
        ),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=capture_sizes,
        ),
    )
    with patch.object(
        cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    ):
        manager = CudaGraphManager(
            vllm_config,
            torch.device("cuda:0"),
            CUDAGraphMode.FULL,
            decode_query_len=1,
            ubatch_runner=cast(UBatchRunner, SimpleNamespace(stage_real_tokens=False)),
        )
    manager._graphs_captured = True
    return manager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a graph pool")
def test_microbatched_graphs_are_only_offered_to_uniform_batches():
    """Mixed batches cannot reuse graphs captured with uniform query lengths."""
    manager = _make_cudagraph_manager([64, 128])

    desc = manager.dispatch(
        num_reqs=64,
        num_tokens=64,
        uniform_token_count=1,
        num_active_loras=0,
        num_ubatches=2,
    )
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_ubatches == 2

    desc = manager.dispatch(
        num_reqs=8,
        num_tokens=64,
        uniform_token_count=None,
        num_active_loras=0,
        num_ubatches=2,
    )
    assert desc.cg_mode == CUDAGraphMode.NONE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a graph pool")
def test_microbatched_graph_needs_every_rank_to_reach_the_split():
    """Fall back to eager on all ranks if any rank cannot reach the captured split."""
    manager = _make_cudagraph_manager([64, 128])
    uniform = [1, 1]

    desc, _ = _sync_dp([64, 64], uniform, cudagraph_manager=manager)
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_ubatches == 2

    # Rank 1 pads the group up to a 128-token graph, but rank 0 only reaches 33.
    desc, dp_sync = _sync_dp([33, 128], uniform, cudagraph_manager=manager)
    assert desc.cg_mode == CUDAGraphMode.NONE
    assert desc.num_ubatches == 2
    assert dp_sync is not None and dp_sync.eager


@pytest.mark.parametrize(
    "padded,k,n",
    [
        (64, 2, 2),
        (64, 2, 3),
        (64, 2, 31),
        (64, 2, 32),
        (128, 2, 63),
        (128, 2, 64),
        (256, 2, 127),
        (256, 2, 128),
        (127, 2, 63),
        (129, 2, 64),
        (11, 3, 5),
        (11, 3, 6),
        (17, 4, 12),
        (19, 5, 12),
    ],
)
def test_conditional_decode_preserves_token_position_and_kv_mapping(padded, k, n):
    """Scatter preserves source order even when k > 2 makes rows overlap."""
    buffers = InputBuffers(padded, padded, torch.device("cpu"))
    batch = _make_input_batch(
        [1] * n, [100 + i for i in range(n)], buffers, padded, padded
    )
    batch.input_ids.copy_(torch.arange(padded))
    batch.positions.copy_(torch.arange(padded) + 100)
    blocks = torch.arange(padded * 4).reshape(padded, 4)
    slots = torch.arange(padded * 2).reshape(2, padded) + 1000
    expected_blocks = blocks[:n].clone()
    expected_slots = slots[:, :n].clone()
    ptrs = [x.data_ptr() for x in (batch.input_ids, batch.positions, blocks, slots)]
    slices = create_ubatch_slices(batch, k)
    indices = stage_decode_tokens(batch, (blocks,), slots, slices)
    expected_indices = [
        row
        for i in range(k)
        for row in range(i * (padded // k), i * (padded // k) + n // k + (i < n % k))
    ]
    assert indices.tolist() == expected_indices
    torch.testing.assert_close(batch.input_ids[indices], torch.arange(n).int())
    torch.testing.assert_close(batch.positions[indices], torch.arange(n) + 100)
    torch.testing.assert_close(blocks[indices], expected_blocks)
    torch.testing.assert_close(slots[:, indices], expected_slots)
    assert not batch.is_padding[indices].any()
    assert batch.is_padding.sum() == padded - n
    assert (slots[:, batch.is_padding] == -1).all()
    assert (blocks[batch.is_padding] == 0).all()
    assert ptrs == [
        x.data_ptr() for x in (batch.input_ids, batch.positions, blocks, slots)
    ]
    output = batch.input_ids[:, None].clone()
    output[: indices.numel()] = output[indices]
    torch.testing.assert_close(output[:n, 0], torch.arange(n).int())
    restore_staged_inputs(batch, (blocks,), slots, indices)
    # The sampler still indexes these buffers by the original logits indices.
    torch.testing.assert_close(batch.input_ids[:n], torch.arange(n).int())
    torch.testing.assert_close(batch.positions[:n], torch.arange(n) + 100)
    torch.testing.assert_close(blocks[:n], expected_blocks)
    torch.testing.assert_close(slots[:, :n], expected_slots)
    assert not batch.is_padding[:n].any()
    assert batch.is_padding[n:].all()
    assert (slots[:, n:] == -1).all()
    for values in (batch.input_ids, batch.positions, blocks):
        assert (values[n:padded] == 0).all()


@pytest.mark.parametrize("n", [0, 1, 65, 80])
def test_conditional_staging_rejects_ineligible_counts(n):
    batch = SimpleNamespace(num_tokens=n, num_tokens_after_padding=128)
    slices = [
        UBatchSlice(slice(0, 64), slice(0, 64)),
        UBatchSlice(slice(64, 128), slice(64, 128)),
    ]
    with pytest.raises(AssertionError):
        stage_decode_tokens(batch, (), torch.empty(1, 128), slices)


@pytest.mark.parametrize(
    "padded,k,counts", [(11, 3, [5, 11, 6, 3]), (17, 4, [12, 17, 4, 7])]
)
def test_staged_prepare_matches_captured_regions_across_steps(padded, k, counts):
    """Unequal physical regions keep stable buffers and clear stale metadata."""
    runner = object.__new__(UBatchRunner)
    runner.num_ubatches = k
    runner.stage_real_tokens = True
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.ubatch_query_start_loc = [
        torch.zeros(padded + 1, dtype=torch.int32) for _ in range(k)
    ]
    runner.ubatch_seq_lens = [torch.zeros(padded, dtype=torch.int32) for _ in range(k)]
    runner.ubatch_dcp_local_seq_lens = [None] * k
    runner.attn_groups = []
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    seen = []

    def prepare_attn(batch, *args, **kwargs):
        seen.append(batch)
        return {}

    runner.model_state = SimpleNamespace(prepare_attn=prepare_attn)
    runner._make_forward_contexts = lambda *args: []
    buffers = InputBuffers(padded, padded, torch.device("cpu"))
    captured = create_ubatch_slices(InputBatch.make_dummy(padded, padded, buffers), k)
    for n in counts:
        seen.clear()
        batch = _make_input_batch(
            [1] * n, list(range(100, 100 + n)), buffers, padded, padded
        )
        batch.input_ids.copy_(torch.arange(padded))
        # Ordinary FULL preparation must never invoke the staging helper.
        guard = (
            patch(
                "vllm.v1.worker.gpu.ubatch_utils.stage_decode_tokens",
                side_effect=AssertionError("normal FULL must not stage"),
            )
            if n == padded
            else nullcontext()
        )
        with guard:
            state = runner.prepare(
                batch, (), torch.zeros(0, padded), CUDAGraphMode.FULL
            )
        assert state.slices == captured
        assert (state.staged_rows is not None) == (n != padded)
        offset = 0
        for i, ubatch in enumerate(seen):
            capacity = captured[i].num_tokens
            count = capacity if n == padded else n // k + (i < n % k)
            assert ubatch.num_tokens == ubatch.num_reqs == count
            assert (
                ubatch.num_tokens_after_padding
                == ubatch.num_reqs_after_padding
                == capacity
            )
            assert ubatch.req_ids == [f"req_{j}" for j in range(offset, offset + count)]
            expected_seq = torch.zeros(capacity, dtype=torch.int32)
            expected_seq[:count] = torch.arange(100 + offset, 100 + offset + count)
            torch.testing.assert_close(ubatch.seq_lens, expected_seq)
            torch.testing.assert_close(ubatch.seq_lens_cpu_upper_bound, expected_seq)
            np.testing.assert_array_equal(
                ubatch.query_start_loc_np, np.minimum(np.arange(capacity + 1), count)
            )
            torch.testing.assert_close(
                ubatch.query_start_loc, torch.from_numpy(ubatch.query_start_loc_np)
            )
            assert (
                ubatch.query_start_loc.data_ptr()
                == runner.ubatch_query_start_loc[i].data_ptr()
            )
            torch.testing.assert_close(
                ubatch.input_ids[:count], torch.arange(offset, offset + count).int()
            )
            offset += count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a graph pool")
@pytest.mark.parametrize("loads", [[63, 97], [63, 127], [63, 64], [32, 96], [64, 97]])
def test_conditional_graph_guard_requires_supported_single_token_layout(loads):
    manager = _make_cudagraph_manager([128])
    manager.ubatch_runner = SimpleNamespace(stage_real_tokens=True)
    desc, sync = _sync_dp(
        loads, [1, 1], cudagraph_manager=manager, num_reqs_per_rank=loads
    )
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_tokens == 128 and desc.num_ubatches == 2
    assert sync is not None and not sync.eager


@pytest.mark.parametrize("padded,k", [(129, 2), (131, 3), (133, 4)])
@pytest.mark.parametrize("delta,capable", [(0, True), (0, False), (1, False)])
def test_staging_gate_uses_last_region_start(padded, k, delta, capable):
    """An empty last region needs staging; one real row there does not."""
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=padded,
        num_reqs=padded,
        num_ubatches=k,
    )
    manager = SimpleNamespace(
        ubatch_runner=SimpleNamespace(stage_real_tokens=capable),
        dispatch=lambda *args, **kwargs: desc,
    )
    n = padded // k * (k - 1) + delta
    loads = [n, padded, padded - 1, padded - 2]
    result, sync = _sync_dp(
        loads,
        [1] * 4,
        cudagraph_manager=manager,
        num_reqs_per_rank=loads,
        num_ubatches=k,
    )
    expected = CUDAGraphMode.FULL if capable or delta else CUDAGraphMode.NONE
    assert result.cg_mode == expected
    assert result.num_ubatches == k
    assert sync is not None and sync.eager == (expected == CUDAGraphMode.NONE)


@pytest.mark.parametrize("uniform,n", [(1, 2), (2, 32)])
def test_staging_gate_rejects_too_few_rows_or_multitoken_decode(uniform, n):
    k = 3
    manager = SimpleNamespace(
        ubatch_runner=SimpleNamespace(stage_real_tokens=True),
        dispatch=lambda *args, **kwargs: BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=131,
            num_reqs=131,
            num_ubatches=k,
        ),
    )
    loads = [n, 100, 100, 100]
    with patch.object(dp_utils, "check_ubatch_thresholds", return_value=True):
        desc, _ = _sync_dp(
            loads,
            [uniform] * 4,
            cudagraph_manager=manager,
            num_reqs_per_rank=loads,
            num_ubatches=k,
        )
    assert desc.cg_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize("padded", [127, 128, 129])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_decode_staging_boundary_matrix(padded, k):
    """The DP contract requires nonempty regions and only repairs an empty tail."""
    last_start = padded // k * (k - 1)
    for n in sorted({k - 1, k, last_start - 1, last_start, last_start + 1, padded}):
        loads = [n, padded]
        manager = SimpleNamespace(
            ubatch_runner=SimpleNamespace(stage_real_tokens=True),
            dispatch=lambda *args, **kwargs: BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=padded,
                num_reqs=padded,
                num_ubatches=k,
            ),
        )
        with patch.object(dp_utils, "check_ubatch_thresholds", return_value=True):
            desc, _ = _sync_dp(
                loads,
                [1, 1],
                cudagraph_manager=manager,
                num_reqs_per_rank=loads,
                num_ubatches=k,
            )
        assert desc.cg_mode == (CUDAGraphMode.NONE if n < k else CUDAGraphMode.FULL)
        if not k <= n <= last_start:
            continue
        buffers = InputBuffers(padded, padded, torch.device("cpu"))
        batch = _make_input_batch([1] * n, [100] * n, buffers, padded, padded)
        batch.input_ids.copy_(torch.arange(padded))
        slices = create_ubatch_slices(batch, k)
        rows = stage_decode_tokens(batch, (), torch.zeros(1, padded), slices)
        expected = [
            s.token_slice.start + j
            for i, s in enumerate(slices)
            for j in range(n // k + (i < n % k))
        ]
        assert rows.tolist() == expected
        assert len(set(expected)) == n and max(expected) < padded
        torch.testing.assert_close(batch.input_ids[rows], torch.arange(n).int())


@pytest.mark.parametrize(
    "loads,padded",
    [
        ([63, 97], 128),
        ([80] * 4, 128),
        ([63, 97, 80, 80], 128),
        ([32, 32, 32, 224], 256),
    ],
)
def test_decode_staging_dp_distributions(loads, padded):
    manager = SimpleNamespace(
        ubatch_runner=SimpleNamespace(stage_real_tokens=True),
        dispatch=lambda *args, **kwargs: BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=padded,
            num_reqs=padded,
            num_ubatches=2,
        ),
    )
    for rank in range(len(loads)):
        desc, sync = _sync_dp(
            loads,
            [1] * len(loads),
            cudagraph_manager=manager,
            num_reqs_per_rank=loads,
            dp_rank=rank,
        )
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert sync is not None
        assert sync.num_tokens_across_dp.tolist() == [padded] * len(loads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph replay")
@pytest.mark.parametrize("padded,k", [(127, 2), (128, 3), (129, 4)])
def test_staged_cuda_graph_preserves_outputs_and_mla_cache(padded, k):
    """Real replay and the MLA cache kernel preserve all slots over changing loads.

    This tests staging/kernel integration, not an end-to-end MoE model.
    """
    from vllm import _custom_ops as ops

    device = torch.device("cuda:0")
    ids = torch.zeros(padded, dtype=torch.int32, device=device)
    positions = torch.zeros(padded, dtype=torch.int64, device=device)
    padding = torch.ones(padded, dtype=torch.bool, device=device)
    slots = torch.full((1, padded), -1, dtype=torch.int64, device=device)
    blocks = torch.zeros(padded, 4, dtype=torch.int32, device=device)
    cache = torch.zeros(32, 16, 576, dtype=torch.bfloat16, device=device)
    reference_cache = torch.zeros_like(cache)
    scale = torch.tensor(1.0, device=device)
    slices = create_ubatch_slices(
        _make_input_batch(
            [1] * padded,
            [1] * padded,
            InputBuffers(padded, padded, torch.device("cpu")),
        ),
        k,
    )

    def forward():
        outputs = []
        for s in slices:
            region = s.token_slice
            kv = (
                (ids[region, None].float() / 32)
                .to(torch.bfloat16)
                .expand(-1, 512)
                .contiguous()
            )
            pe = positions[region, None].to(torch.bfloat16).expand(-1, 64).contiguous()
            ops.concat_and_cache_mla(kv, pe, cache, slots[0, region], "auto", scale)
            outputs.append(ids[region].float() * 2 + positions[region].float())
        return torch.cat(outputs)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    for step, n in enumerate([k, padded // k * (k - 1), k + 1]):
        ids.copy_(torch.arange(padded, device=device) + step * 10)
        positions.fill_(step + 1)
        original = ids[:n].clone()
        slots.fill_(-1)
        logical_slots = torch.arange(n, device=device) * 2 + step
        slots[0, :n] = logical_slots
        batch = SimpleNamespace(
            num_tokens=n,
            num_reqs=n,
            has_prefill=False,
            num_draft_tokens=0,
            num_scheduled_tokens=np.ones(n, dtype=np.int32),
            num_tokens_after_padding=padded,
            input_ids=ids,
            positions=positions,
            is_padding=padding,
        )
        rows = stage_decode_tokens(batch, (blocks,), slots, slices)
        graph.replay()
        output[:n] = output[rows]
        torch.testing.assert_close(output[:n], original.float() * 2 + step + 1)
        expected = torch.cat(
            (
                (original[:, None].float() / 32).to(torch.bfloat16).expand(-1, 512),
                torch.full((n, 64), step + 1, dtype=torch.bfloat16, device=device),
            ),
            dim=1,
        )
        reference_cache.view(-1, 576)[logical_slots] = expected
        torch.testing.assert_close(cache, reference_cache, rtol=0, atol=0)
        restore_staged_inputs(batch, (blocks,), slots, rows)
        torch.testing.assert_close(ids[:n], original)
        torch.testing.assert_close(
            positions[:n], torch.full_like(positions[:n], step + 1)
        )
        torch.testing.assert_close(slots[0, :n], logical_slots)
        for values in (ids, positions, blocks):
            assert (values[n:padded] == 0).all()
        assert (slots[:, n:padded] == -1).all()
        assert padding[n:padded].all()


def _make_staging_capability_runner(model_config, excluded=None):
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    runner = object.__new__(UBatchRunner)
    state = object.__new__(DefaultModelState)
    state.supports_mm_inputs = excluded == "multimodal"
    state.rope_state = object() if excluded == "rope" else None
    runner.model_state = object() if excluded == "model_state" else state
    runner.parallel_config = SimpleNamespace(
        tensor_parallel_size=2 if excluded == "tp" else 1,
        pipeline_parallel_size=2 if excluded == "pp" else 1,
        prefill_context_parallel_size=2 if excluded == "pcp" else 1,
        enable_eplb=excluded == "eplb",
        all2all_backend="deepep_low_latency"
        if excluded == "communication"
        else "nixl_ep",
    )
    runner.dcp_size = 2 if excluded == "dcp" else 1
    runner.num_ubatches = {"k0": 0, "k3": 3, "k4": 4}.get(excluded, 2)
    runner.attn_groups = [
        [
            SimpleNamespace(
                backend=SimpleNamespace(
                    get_name=lambda: "TRITON_ATTN"
                    if excluded == "backend"
                    else "FLASH_ATTN_MLA"
                )
            )
        ]
    ]
    runner.vllm_config = SimpleNamespace(
        model_config=model_config,
        lora_config=object() if excluded == "lora" else None,
        speculative_config=object() if excluded == "spec" else None,
        kv_transfer_config=object() if excluded == "kv_transfer" else None,
    )
    model_config.enable_prompt_embeds = excluded == "prompt_embeds"
    model_config.enable_return_routed_experts = excluded == "routing"
    runner.stage_real_tokens = (
        runner._supports_real_token_staging()
        and runner._backend_supports_staging_layout()
    )
    return runner


def test_random_staging_layouts_match_independent_reference():
    """A fixed random corpus checks arbitrary remainders and overlapping copies."""
    rng = np.random.default_rng(57184)
    for _ in range(300):
        k = int(rng.integers(2, 13))
        padded = int(rng.integers(2 * k, 514))
        n = int(rng.integers(k, padded // k * (k - 1) + 1))
        buffers = InputBuffers(padded, padded, torch.device("cpu"))
        batch = _make_input_batch([1] * n, [100] * n, buffers, padded, padded)
        batch.input_ids.copy_(torch.arange(padded) + 1)
        regions = create_ubatch_slices(batch, k)
        counts = [0] * k
        for token in range(n):
            counts[token % k] += 1
        expected = []
        for region, count in zip(regions, counts):
            assert 0 < count <= region.num_tokens
            expected.extend(
                list(range(region.token_slice.start, region.token_slice.stop))[:count]
            )
        slots = torch.arange(2 * padded).reshape(2, padded)
        original_slots = slots[:, :n].clone()
        rows = stage_decode_tokens(batch, (), slots, regions)
        assert rows.tolist() == expected
        assert rows.unique().numel() == n
        torch.testing.assert_close(batch.input_ids[rows], torch.arange(n).int() + 1)
        assert (slots[:, batch.is_padding] == -1).all()
        restore_staged_inputs(batch, (), slots, rows)
        torch.testing.assert_close(batch.input_ids[:n], torch.arange(n).int() + 1)
        torch.testing.assert_close(slots[:, :n], original_slots)


@pytest.fixture
def staging_model_config(tmp_path):
    from transformers import DeepseekV2Config

    DeepseekV2Config(architectures=["DeepseekV2ForCausalLM"]).save_pretrained(tmp_path)
    return ModelConfig(model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True)


@pytest.mark.parametrize(
    "excluded",
    [
        None,
        "multimodal",
        "rope",
        "model_state",
        "prompt_embeds",
        "tp",
        "pp",
        "pcp",
        "dcp",
        "lora",
        "spec",
        "kv_transfer",
        "eplb",
        "routing",
        "backend",
        "communication",
        "k0",
        "k3",
        "k4",
    ],
)
def test_staging_capability_checks_data_contract(excluded, staging_model_config):
    runner = _make_staging_capability_runner(staging_model_config, excluded)
    assert runner._supports_real_token_staging() == (
        excluded in (None, "k0", "k3", "k4", "communication")
    )
    assert runner._backend_supports_staging_layout() == (
        excluded not in ("k0", "communication")
    )
    assert runner.stage_real_tokens == (excluded in (None, "k3", "k4"))


@pytest.mark.parametrize("excluded", [None, "backend", "routing", "k3", "k4"])
def test_dp_gate_uses_real_staging_capability(staging_model_config, excluded):
    runner = _make_staging_capability_runner(staging_model_config, excluded)
    k = runner.num_ubatches
    manager = SimpleNamespace(
        ubatch_runner=runner,
        dispatch=lambda *args, **kwargs: BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=128,
            num_reqs=128,
            num_ubatches=k,
        ),
    )
    desc, _ = _sync_dp(
        [63, 97],
        [1, 1],
        cudagraph_manager=manager,
        num_reqs_per_rank=[63, 97],
        num_ubatches=k,
    )
    expected = (
        CUDAGraphMode.FULL if excluded in (None, "k3", "k4") else CUDAGraphMode.NONE
    )
    assert desc.cg_mode == expected


@pytest.mark.parametrize(
    "invalid", ["prefill", "draft", "requests", "queries", "region"]
)
def test_staging_rejects_invalid_local_contract_before_mutation(invalid):
    buffers = InputBuffers(128, 128, torch.device("cpu"))
    batch = _make_input_batch([1] * 63, [100] * 63, buffers, 128, 128)
    slices = create_ubatch_slices(batch, 2)
    if invalid == "prefill":
        batch.has_prefill = True
    elif invalid == "draft":
        batch.num_draft_tokens = 1
    elif invalid == "requests":
        batch.num_reqs = 62
    elif invalid == "queries":
        batch.num_scheduled_tokens[:2] = [0, 2]
    else:
        slices[0] = UBatchSlice(slice(0, 10), slice(0, 10))
        slices[1] = UBatchSlice(slice(10, 128), slice(10, 128))
        # Keep the empty-final-region premise, but overflow an earlier region.
        slices.insert(1, UBatchSlice(slice(10, 64), slice(10, 64)))
        slices[-1] = UBatchSlice(slice(64, 128), slice(64, 128))
    original = batch.input_ids.clone()
    with pytest.raises(AssertionError):
        stage_decode_tokens(batch, (), torch.zeros(1, 128), slices)
    torch.testing.assert_close(batch.input_ids, original)


@pytest.mark.parametrize(
    "excluded, expected",
    [
        ("backend", "TRITON_ATTN"),
        ("routing", "enable_return_routed_experts"),
        ("k0", "at least one microbatch"),
        ("communication", "deepep_low_latency"),
    ],
)
def test_staging_rejection_identifies_incompatible_contract(
    staging_model_config, excluded, expected
):
    runner = _make_staging_capability_runner(staging_model_config, excluded)
    reason = (
        runner._real_token_staging_unsupported_reason()
        or runner._staging_layout_unsupported_reason()
    )
    assert not runner.stage_real_tokens
    assert expected in reason


def test_empty_attention_groups_cannot_enable_staging(staging_model_config):
    runner = _make_staging_capability_runner(staging_model_config)
    runner.attn_groups = [[]]
    assert not runner._supports_real_token_staging()
    assert (
        "attention groups are empty" in runner._real_token_staging_unsupported_reason()
    )


@pytest.mark.parametrize("family", ["qwen3_moe", "mixtral"])
@pytest.mark.parametrize(
    "incompatible",
    [
        None,
        "fa_version",
        "graph",
        "dtype",
        "window",
        "chunk",
        "non_causal",
        "builders",
        "weights",
        "quantization",
        "spec",
        "builder_type",
        "builder_count",
    ],
)
def test_staging_fa3_capability_depends_on_data_contract(
    tmp_path, family, incompatible
):
    """Family names do not decide eligibility; unverified FA layouts stay closed."""
    from transformers import MixtralConfig, Qwen3MoeConfig

    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    cls, architecture = {
        "qwen3_moe": (Qwen3MoeConfig, "Qwen3MoeForCausalLM"),
        "mixtral": (MixtralConfig, "MixtralForCausalLM"),
    }[family]
    cls(architectures=[architecture]).save_pretrained(tmp_path)
    model = ModelConfig(model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True)
    if incompatible == "weights":
        model.dtype = torch.float16
    if incompatible == "quantization":
        model.quantization = "fp8"
    runner = _make_staging_capability_runner(model)
    # This CPU test supplies builder state; the single-layer GPU test constructs
    # the actual builder and exercises its captured scheduler buffers.
    builder = object.__new__(FlashAttentionMetadataBuilder)
    builder.aot_schedule = incompatible != "fa_version"
    builder.use_full_cuda_graph = incompatible != "graph"
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float16 if incompatible == "dtype" else torch.bfloat16,
        sliding_window=32 if incompatible == "window" else None,
        attention_chunk_size=32 if incompatible == "chunk" else None,
        non_causal=incompatible == "non_causal",
    )
    if incompatible == "spec":
        spec = SlidingWindowSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=32,
        )
    builders = [builder, builder]
    if incompatible == "builders":
        builders = []
    elif incompatible == "builder_type":
        builders = [object(), object()]
    elif incompatible == "builder_count":
        builders = [builder]
    runner.attn_groups = [
        [
            SimpleNamespace(
                backend=SimpleNamespace(get_name=lambda: "FLASH_ATTN"),
                kv_cache_spec=spec,
                metadata_builders=builders,
            )
        ]
    ]
    reason = runner._real_token_staging_unsupported_reason()
    assert (reason is None) == (incompatible is None), reason
    if reason is not None:
        assert "FLASH_ATTN" in reason
        assert "model_type" not in reason


def test_fa3_staging_rejects_batch_invariant_scheduler(
    staging_model_config, monkeypatch
):
    """Batch invariance disables AOT even when the builder flag remains true."""
    from vllm import envs
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    runner = _make_staging_capability_runner(staging_model_config)
    builder = object.__new__(FlashAttentionMetadataBuilder)
    builder.aot_schedule = builder.use_full_cuda_graph = True
    runner.attn_groups = [
        [
            SimpleNamespace(
                backend=SimpleNamespace(get_name=lambda: "FLASH_ATTN"),
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=128,
                    dtype=torch.bfloat16,
                ),
                metadata_builders=[builder, builder],
            )
        ]
    ]
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    reason = runner._real_token_staging_unsupported_reason()
    assert reason is not None and "VLLM_BATCH_INVARIANT" in reason
