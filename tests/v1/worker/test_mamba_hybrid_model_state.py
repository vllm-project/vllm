# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.v1.attention.test_gdn_metadata_builder import (
    BLOCK_SIZE,
    DEVICE,
    _create_gdn_builder,
)
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from tests.v1.worker.test_gpu_ubatch_slicing import (
    MAX_NUM_REQS,
    MAX_NUM_TOKENS,
    _make_buffers,
    _make_input_batch,
    _make_model_inputs,
    _make_ubatch_buffers,
)
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import create_forward_context, override_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.gpu.ubatch_utils import (
    UBatchRunner,
    _slice_input_batch,
    create_ubatch_slices,
)
from vllm.v1.worker.ubatching import dbo_yield
from vllm.v1.worker.utils import AttentionGroup


def _recoverssm_attn_groups(layer_name: str) -> list[list[AttentionGroup]]:
    """RecoverSSMState.record_step reads only each group's ``layer_names``."""
    group = SimpleNamespace(layer_names=[layer_name])
    return [[group]]  # type: ignore[list-item]  # layer_names-only stand-in


GPU_DEVICE = current_platform.device_type
HAS_GPU = current_platform.is_cuda_alike() or current_platform.is_xpu()


def test_prepare_attn_forwards_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=0)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state.recoverssm = None

    positions = torch.tensor([1536], dtype=torch.int64)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1,
        num_reqs_after_padding=1,
        num_tokens_after_padding=1,
        query_start_loc_np=torch.tensor([0, 1], dtype=torch.int32).numpy(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        num_scheduled_tokens=torch.tensor([1], dtype=torch.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=torch.tensor([1537], dtype=torch.int32),
        seq_lens=torch.tensor([1537], dtype=torch.int32),
        is_prefilling_np=torch.tensor([False]).numpy(),
        prefill_runs_as_decode_np=None,
        dcp_local_seq_lens=None,
        positions=positions,
        prompt_lens=torch.tensor([1024], dtype=torch.int32),
    )
    expected_metadata = {"layer": object()}
    build_attn_metadata = Mock(return_value=expected_metadata)
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", build_attn_metadata)

    metadata = state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )

    assert metadata is expected_metadata
    assert build_attn_metadata.call_args.kwargs["positions"] is positions


def test_add_request_seeds_state_idx_in_mamba_blocks() -> None:
    """A drafter group can lower cache_config.block_size below the mamba
    block size; the seeded column must still be in mamba blocks."""
    state = object.__new__(MambaHybridModelState)
    state.cache_config = SimpleNamespace(block_size=16, mamba_block_size=880)
    state._align_mode = True
    state.rope_state = None
    state.prompt_embeds_state = None
    state.num_accepted_tokens_gpu = torch.ones(2, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.zeros(2, dtype=torch.int32)

    state.add_request(1, SimpleNamespace(num_computed_tokens=107_360))

    assert state._mamba_state_idx_gpu[1] == 121


def test_padded_prompt_tail_builds_as_spec_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-token prompt tail over prior state, padded with K placeholder
    drafts (e.g. a P/D decode-node arrival), must reach the GDN builder as a
    spec-decode row. Built as a prefill, the placeholder tokens are folded into
    the recurrent state and can't be rolled back.
    """
    k = 3
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=k)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state.recoverssm = None
    state.num_accepted_tokens_gpu = torch.ones(4, dtype=torch.int32)

    # A verify decode, a padded prompt tail (128 of 129 prompt tokens already
    # computed), and a fresh prompt chunk of the same length.
    query_lens = [k + 1] * 3
    seq_lens = [50, 128 + k + 1, k + 1]
    is_prefilling = [False, True, True]
    query_start_loc = np.array([0, 4, 8, 12], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=3,
        num_tokens=12,
        num_reqs_after_padding=3,
        num_tokens_after_padding=12,
        idx_mapping=torch.arange(3),
        query_start_loc_np=query_start_loc,
        query_start_loc=torch.from_numpy(query_start_loc),
        num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
        num_draft_tokens_per_req=np.array([k, k, 0], dtype=np.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        is_prefilling_np=np.array(is_prefilling),
        prefill_runs_as_decode_np=np.array([False, True, False]),
        dcp_local_seq_lens=None,
        positions=torch.zeros(12, dtype=torch.int64),
        prompt_lens=None,
    )
    build_attn_metadata = Mock(return_value={})
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", build_attn_metadata)
    state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )
    mamba_metadata = build_attn_metadata.call_args.kwargs[
        "model_specific_attn_metadata"
    ]
    assert mamba_metadata.is_prefilling.tolist() == [False, False, True]

    builder = _create_gdn_builder(num_speculative_tokens=k)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens), BLOCK_SIZE, DEVICE
    ).replace(**mamba_metadata.get_extra_common_attn_kwargs(0, 3))
    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=mamba_metadata.num_accepted_tokens,
        num_decode_draft_tokens_cpu=mamba_metadata.num_decode_draft_tokens_cpu,
    )

    # Only the fresh prompt chunk needs the prefill kernels.
    assert meta.num_spec_decodes == 2
    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == k + 1


def _mamba_hybrid_state(num_speculative_tokens: int) -> MambaHybridModelState:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state.recoverssm = None
    # A request that accepted 3 drafts (plus the bonus token) on the previous
    # step; the others have never had a draft accepted.
    state.num_accepted_tokens_gpu = torch.tensor([4, 1, 1], dtype=torch.int32)
    return state


def _input_batch(
    num_scheduled_tokens: list[int],
    num_draft_tokens_per_req: list[int] | None,
    is_prefilling: list[bool],
    num_reqs_after_padding: int | None = None,
) -> SimpleNamespace:
    """`num_draft_tokens_per_req=None` is a batch where nobody proposed a
    draft this step, e.g. because ngram found no match for anyone."""
    num_reqs = len(num_scheduled_tokens)
    query_start_loc = np.zeros(num_reqs + 1, dtype=np.int32)
    query_start_loc[1:] = np.cumsum(num_scheduled_tokens)
    num_tokens = int(query_start_loc[-1])
    seq_lens = torch.full((num_reqs,), 10, dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs_after_padding or num_reqs,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        query_start_loc_np=query_start_loc,
        num_scheduled_tokens=np.array(num_scheduled_tokens, dtype=np.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=seq_lens,
        is_prefilling_np=np.array(is_prefilling),
        prefill_runs_as_decode_np=None,
        num_draft_tokens_per_req=None
        if num_draft_tokens_per_req is None
        else np.array(num_draft_tokens_per_req, dtype=np.int32),
        idx_mapping=torch.arange(num_reqs),
        query_start_loc=torch.from_numpy(query_start_loc),
        seq_lens=seq_lens,
        dcp_local_seq_lens=None,
        positions=torch.zeros(num_tokens, dtype=torch.int64),
        prompt_lens=None,
    )


def _prepare_attn_metadata(
    state: MambaHybridModelState,
    monkeypatch,
    input_batch: SimpleNamespace | None = None,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
) -> SimpleNamespace:
    captured = {}

    def fake_build_attn_metadata(**kwargs):
        captured.update(kwargs)
        return Mock()

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_states.mamba_hybrid.build_attn_metadata",
        fake_build_attn_metadata,
    )

    if input_batch is None:
        input_batch = _input_batch([1, 1], None, [False, False])
    state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=cudagraph_mode,
        block_tables=(),
        slot_mappings={},
        attn_groups=[],
        kv_cache_config=Mock(),
    )
    return captured["model_specific_attn_metadata"]


def test_prepare_attn_builds_each_microbatch_on_its_requests_state_slots():
    """DBO decode microbatches use their own builder and their requests' slots."""
    builders = [_create_gdn_builder(), _create_gdn_builder()]
    for builder in builders:
        # Hybrid DBO requires prefix caching off.
        builder.vllm_config.cache_config.mamba_cache_mode = "none"
    group = AttentionGroup(
        backend=GDNAttentionBackend,
        layer_names=["layer.0"],
        kv_cache_spec=builders[0].kv_cache_spec,
        kv_cache_group_id=0,
        metadata_builders=builders,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], group.kv_cache_spec)],
    )
    batch = _make_input_batch([1] * 4, [10, 20, 30, 40], _make_buffers())
    state_slots = torch.tensor([[7], [3], [9], [5]], dtype=torch.int32)
    ubatch_buffers = _make_ubatch_buffers()

    for ubatch_idx, ubatch_slice in enumerate(create_ubatch_slices(batch, 2)):
        ubatch = _slice_input_batch(batch, ubatch_slice, *ubatch_buffers[ubatch_idx])
        metadata = _mamba_hybrid_state(num_speculative_tokens=0).prepare_attn(
            input_batch=ubatch,
            cudagraph_mode=CUDAGraphMode.NONE,
            block_tables=(state_slots[ubatch_slice.request_slice],),
            slot_mappings=torch.zeros(1, ubatch.num_tokens, dtype=torch.int64),
            attn_groups=[[group]],
            kv_cache_config=kv_cache_config,
            ubatch_idx=ubatch_idx,
        )["layer.0"]

        assert metadata.num_decodes == 2 and metadata.num_prefills == 0
        assert metadata.non_spec_state_indices_tensor.tolist() == (
            state_slots[ubatch_slice.request_slice, 0].tolist()
        )


# Small GDN dims, as in tests/kernels/mamba/test_gdn_forward_core_split.py.
GDN_H, GDN_HV, GDN_K, GDN_V, GDN_CONV_KERNEL = 4, 8, 128, 128, 4
GDN_CONV_DIM = 2 * GDN_H * GDN_K + GDN_HV * GDN_V
GDN_PREFIX = "model.layers.0.linear_attn"


def _gdn_decode_layer(conv_state, ssm_state, device):
    """The real GDN ``_forward_core`` bound to random weights and state pools."""
    layer = SimpleNamespace(
        prefix=GDN_PREFIX,
        enable_packed_recurrent_decode=False,
        tp_size=1,
        num_k_heads=GDN_H,
        num_v_heads=GDN_HV,
        head_k_dim=GDN_K,
        head_v_dim=GDN_V,
        key_dim=GDN_H * GDN_K,
        value_dim=GDN_HV * GDN_V,
        activation="silu",
        A_log=torch.randn(GDN_HV, device=device) * 0.1,
        dt_bias=torch.randn(GDN_HV, device=device) * 0.1,
        conv1d=SimpleNamespace(
            weight=torch.randn(
                GDN_CONV_DIM, 1, GDN_CONV_KERNEL, dtype=torch.bfloat16, device=device
            )
            * 0.1,
            bias=torch.randn(GDN_CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1,
        ),
        kv_cache=(conv_state, ssm_state),
    )
    for name in ("rearrange_mixed_qkv", "_forward_core"):
        method = getattr(QwenGatedDeltaNetAttention, name)
        setattr(layer, name, types.MethodType(method, layer))
    return layer


def _gdn_core_attn(layer, mixed_qkv, b, a):
    out = torch.zeros(
        mixed_qkv.shape[0], GDN_HV, GDN_V, dtype=mixed_qkv.dtype, device=b.device
    )
    layer._forward_core(
        mixed_qkv=mixed_qkv.clone(), b=b.clone(), a=a.clone(), core_attn_out=out
    )
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DBO needs a GPU")
def test_dbo_decode_steps_match_the_unsplit_recurrent_update(monkeypatch):
    """Two DBO decode microbatches running the real GDN kernels on interleaved
    threads update the recurrent state pool exactly like one unsplit batch,
    across steps with reordered requests and DP padding."""
    monkeypatch.setenv("VLLM_DBO_COMM_SMS", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    # Only the config of the hybrid MoE is used; no weights are loaded. Expert
    # parallelism stays off so the runner needs no EP process group.
    dbo_config = VllmConfig(
        model_config=ModelConfig(
            model="Qwen/Qwen3-Next-80B-A3B-Instruct",
            max_model_len=512,
            enforce_eager=True,
        ),
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_model_len=512, is_encoder_decoder=False, async_scheduling=False
        ),
        parallel_config=ParallelConfig(
            data_parallel_size=2,
            enable_dbo=True,
            all2all_backend="nixl_ep",
            distributed_executor_backend="mp",
            nnodes=2,
        ),
    )

    gdn_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": GDN_K},
    )
    gdn_config.cache_config.mamba_cache_mode = "none"
    gdn_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    spec = MambaSpec(block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,))
    group = AttentionGroup(
        backend=GDNAttentionBackend,
        layer_names=[GDN_PREFIX],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        metadata_builders=[
            GDNAttentionMetadataBuilder(spec, [GDN_PREFIX], gdn_config, device)
            for _ in range(2)
        ],
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([GDN_PREFIX], spec)],
    )
    model_state = _mamba_hybrid_state(num_speculative_tokens=0)
    runner = UBatchRunner(
        dbo_config,
        device,
        model_state=model_state,
        attn_groups=[[group]],
        kv_cache_config=kv_cache_config,
        max_num_reqs=MAX_NUM_REQS,
    )

    # Slot 0 is the reserved null block; five requests own scattered slots.
    pool_size = 12
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, GDN_H, GDN_HV, GDN_K, GDN_V, GDN_CONV_KERNEL, num_spec=0
    )
    conv_state = torch.randn(
        pool_size, *conv_shape, dtype=torch.bfloat16, device=device
    )
    ssm_state = torch.randn(pool_size, *ssm_shape, device=device) * 0.05
    initial_pools = (conv_state.clone(), ssm_state.clone())
    unsplit = _gdn_decode_layer(conv_state.clone(), ssm_state.clone(), device)
    split = _gdn_decode_layer(conv_state, ssm_state, device)
    for name in ("A_log", "dt_bias", "conv1d"):
        setattr(split, name, getattr(unsplit, name))

    class _GDNModel(torch.nn.Module):
        def forward(self, input_ids, positions, **kwargs):
            out = _gdn_core_attn(split, *(x[input_ids] for x in inputs))
            # Hand off to the other microbatch, as at the expert all-to-all.
            dbo_yield()
            return out

    buffers = InputBuffers(MAX_NUM_REQS, MAX_NUM_TOKENS, device)
    slots = [11, 2, 7, 4, 9]
    seq_lens = [20, 33, 7, 48, 15]
    num_padded_tokens = 8
    for step in range(2):
        order = list(range(5)) if step == 0 else list(reversed(range(5)))
        batch = _make_input_batch(
            [1] * 5,
            [seq_lens[i] + step for i in order],
            buffers,
            num_tokens_padded=num_padded_tokens,
        )
        block_table = torch.zeros(5, 1, dtype=torch.int32, device=device)
        block_table[:, 0] = torch.tensor([slots[i] for i in order])
        slot_mappings = torch.zeros(
            1, num_padded_tokens, dtype=torch.int64, device=device
        )
        inputs = (
            torch.randn(num_padded_tokens, GDN_CONV_DIM, device=device) * 0.1,
            torch.randn(num_padded_tokens, GDN_HV, device=device) * 0.1,
            torch.randn(num_padded_tokens, GDN_HV, device=device) * 0.1,
        )
        inputs = tuple(x.to(torch.bfloat16) for x in inputs)

        desc = BatchExecutionDescriptor(
            CUDAGraphMode.NONE, num_padded_tokens, 5, num_ubatches=2
        )
        ubatch_state = runner.prepare(batch, (block_table,), slot_mappings, desc)
        split_out = runner.run(
            _GDNModel(), _make_model_inputs(num_padded_tokens, device), ubatch_state
        )

        metadata = model_state.prepare_attn(
            batch,
            CUDAGraphMode.NONE,
            (block_table,),
            slot_mappings,
            [[group]],
            kv_cache_config,
        )
        with override_forward_context(create_forward_context(metadata, dbo_config)):
            unsplit_out = _gdn_core_attn(unsplit, *inputs)

        assert len(ubatch_state.slices) == 2
        torch.testing.assert_close(split_out[:5], unsplit_out[:5], rtol=0, atol=0)
        for pool, initial in zip(split.kv_cache, initial_pools):
            assert not torch.equal(pool[slots], initial[slots])
        for split_pool, unsplit_pool in zip(split.kv_cache, unsplit.kv_cache):
            torch.testing.assert_close(split_pool[1:], unsplit_pool[1:], rtol=0, atol=0)


def test_prepare_attn_keeps_the_accepted_offset_when_the_batch_drafted_nothing(
    monkeypatch,
):
    """`num_accepted_tokens` must reach the attention builders whenever the
    speculative config is on, even when *this* batch scheduled no draft
    tokens at all. Before the fix, `use_spec_decode` was gated on this step's
    scheduled drafts (`len(scheduled_spec_decode_tokens) > 0` in the V1
    runner) rather than on the speculative config, so a request that accepted
    drafts on a previous step would silently lose that offset the moment
    ngram found no match for anyone in the batch.
    """
    state = _mamba_hybrid_state(num_speculative_tokens=3)

    metadata = _prepare_attn_metadata(state, monkeypatch)

    assert metadata.num_accepted_tokens is not None
    assert metadata.num_accepted_tokens.tolist() == [4, 1]
    assert metadata.num_decode_draft_tokens_cpu is not None


def test_prepare_attn_omits_the_offset_without_a_speculative_config(monkeypatch):
    """Mirrors the check above: with speculative decoding off entirely, there
    is no accepted-token offset to carry, and the metadata must say so
    explicitly (`None`) rather than a stale or default value."""
    state = _mamba_hybrid_state(num_speculative_tokens=0)

    metadata = _prepare_attn_metadata(state, monkeypatch)

    assert metadata.num_accepted_tokens is None
    assert metadata.num_decode_draft_tokens_cpu is None


def test_zero_draft_decode_row_is_marked_as_a_speculative_row(monkeypatch):
    """A decode row that got no drafts still has to reach the speculative path.

    The recurrent backends only apply a row's accepted-token offset on that
    path, and a row that accepted drafts on the previous step keeps that offset
    even when the current step proposes nothing (the scheduler drops drafts
    whenever the token budget truncates the step). Marking such a row with the
    -1 sentinel sends it to the plain decode path, which reads the state at
    column 0 and ignores the offset.
    """
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([3, 1, 1], [2, 0, 0], [False, False, True])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [
        2,  # decode row with drafts
        0,  # decode row with no drafts: speculative path, offset applied
        -1,  # one-token prefill tail: no offset to apply, stays a plain row
    ]


def test_batch_without_any_drafts_still_marks_its_decode_rows(monkeypatch):
    """`num_draft_tokens_per_req` is None whenever no row in the batch drafted
    anything. Those rows are ordinary decode rows carrying an offset from the
    previous step, so the absence of drafts in this step must not be read as
    speculative decoding being off."""
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([1, 1], None, [False, True])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [0, -1]


def test_padded_rows_keep_the_sentinel(monkeypatch):
    """Full CUDA-graph capture pads the batch; padded rows describe no request
    and must not be picked up as speculative rows."""
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([1, 1], [0, 0], [False, False], num_reqs_after_padding=4)

    metadata = _prepare_attn_metadata(
        state, monkeypatch, input_batch, cudagraph_mode=CUDAGraphMode.FULL
    )

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [0, 0, -1, -1]


def test_rows_wider_than_the_speculative_state_window_keep_the_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _mamba_hybrid_state(num_speculative_tokens=7)
    input_batch = _input_batch([12, 8, 1], None, [False, False, False])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [-1, 0, 0]


def test_chunked_prefill_tail_of_two_or_three_tokens_keeps_the_sentinel(
    monkeypatch,
):
    """A prefill continuation is excluded by `is_prefilling`, not by its token
    count, so tails wider than one token are covered by the same rule."""
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([2, 3], [0, 0], [True, True])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [-1, -1]


@pytest.mark.skipif(not HAS_GPU, reason="Requires CUDA or XPU")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device=GPU_DEVICE
    )
    state._align_mode = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device=GPU_DEVICE)

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device=GPU_DEVICE
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    attn_groups = _recoverssm_attn_groups("layer")

    state.record_step({"layer": metadata}, attn_groups, for_capture=False)
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )

    metadata.commit_recoverssm_state.assert_called_once_with(num_sampled)


@pytest.mark.skipif(not HAS_GPU, reason="Requires CUDA or XPU")
@pytest.mark.parametrize(
    ("computed", "sampled", "expected_column"),
    [(4, 3, 0), (4, 4, 0), (4, 5, 1), (8, 8, 1), (8, 9, 2)],
)
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias(
    computed: int, sampled: int, expected_column: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full(
        (5,), -1, dtype=torch.int32, device=GPU_DEVICE
    )
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device=GPU_DEVICE
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device=GPU_DEVICE),
        num_computed_tokens=torch.tensor(
            [6, computed], dtype=torch.int32, device=GPU_DEVICE
        ),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device=GPU_DEVICE),
    )
    num_sampled = torch.tensor([2, sampled], dtype=torch.int32, device=GPU_DEVICE)
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device=GPU_DEVICE)
    attn_groups = _recoverssm_attn_groups("layer")

    state.recoverssm.record_step({"layer": metadata}, attn_groups, for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, expected_column, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    expected_accepted = [9, 1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted
