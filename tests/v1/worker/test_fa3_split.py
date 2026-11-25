# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

import numpy as np
import pytest
import torch

import vllm.envs as envs
from tests.v1.core.utils import new_kv_cache_group
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CUDAGraphMode,
    KernelConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.attention import AttentionConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends import flash_attn as flash_attn_mod
from vllm.v1.attention.backends.fa3_split import (
    FA3_MAX_DETERMINISTIC_SPLITS,
    FA3_MAX_DYNAMIC_SPLIT_REQUESTS,
    get_split_counts,
    get_split_workspace,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
)
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.utils import AttentionGroup
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager


class _DummyPPGroup:
    is_first_rank = True
    is_last_rank = True


class _RecordingMetadataBuilder:
    supports_update_block_table = False

    def __init__(self) -> None:
        self.recorded: CommonAttentionMetadata | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CommonAttentionMetadata:
        self.recorded = common_attn_metadata
        return common_attn_metadata


class _DummyBackend:
    pass


def _make_split_vllm_config(
    *,
    cudagraph_mode: CUDAGraphMode,
    decode_request_sizes: list[int] | None = None,
    prefill_token_sizes: list[int] | None = None,
    enable_jit_warmup: bool = True,
) -> VllmConfig:
    comp_kwargs: dict[str, object] = {
        "cudagraph_mode": cudagraph_mode,
        "max_cudagraph_capture_size": 8,
    }
    if decode_request_sizes is not None:
        comp_kwargs["max_decode_cudagraph_capture_requests"] = 4
        comp_kwargs["decode_cudagraph_capture_request_sizes"] = decode_request_sizes
    if prefill_token_sizes is not None:
        comp_kwargs["prefill_cudagraph_capture_token_sizes"] = prefill_token_sizes
    return VllmConfig(
        model_config=ModelConfig(max_model_len=64, skip_tokenizer_init=True),
        parallel_config=ParallelConfig(),
        cache_config=CacheConfig(),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=16,
            max_model_len=64,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(**comp_kwargs),
        attention_config=AttentionConfig(deterministic_attention_split_stride=512),
        kernel_config=KernelConfig(enable_jit_warmup=enable_jit_warmup),
    )


def _make_common_attn_metadata(
    seq_lens: list[int],
    deterministic_splits: torch.Tensor | None,
) -> CommonAttentionMetadata:
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.zeros((2, 4), dtype=torch.int32),
        slot_mapping=torch.zeros(2, dtype=torch.int64),
        causal=True,
        deterministic_attention_splits=deterministic_splits,
    )


def test_get_split_counts_decode_prefill_and_padding() -> None:
    # Includes one padded tail slot (len(seq_lens) == 5 > len(query_lens) == 4).
    seq_lens = np.array([256, 513, 8192, 2048, 9999], dtype=np.int32)
    query_lens = np.array([1, 1, 1, 1], dtype=np.int32)
    is_prefilling = np.array([False, False, False, True], dtype=np.bool_)

    splits = get_split_counts(seq_lens, query_lens, is_prefilling, stride=512)
    assert splits.dtype == np.int32
    assert np.array_equal(
        splits,
        np.array([1, 2, FA3_MAX_DETERMINISTIC_SPLITS, 1, 1], dtype=np.int32),
    )

    # Multi-token query decode (speculative verify) stays unsplit.
    multi_q = np.array([1, 2, 1, 1], dtype=np.int32)
    splits_multi_q = get_split_counts(seq_lens, multi_q, is_prefilling, stride=512)
    assert np.array_equal(
        splits_multi_q,
        np.array([1, 1, FA3_MAX_DETERMINISTIC_SPLITS, 1, 1], dtype=np.int32),
    )


def test_get_split_workspace_shapes() -> None:
    init_workspace_manager(torch.device("cpu"))
    try:
        output, lse = get_split_workspace(num_tokens=16, num_heads=4, head_size=128)
        assert output.shape == (FA3_MAX_DETERMINISTIC_SPLITS, 4, 16, 128)
        assert lse.shape == (FA3_MAX_DETERMINISTIC_SPLITS, 4, 16)
        assert output.dtype == torch.float32
        assert lse.dtype == torch.float32
    finally:
        reset_workspace_manager()


@pytest.mark.parametrize(
    ("cg_mode", "varlen_decode", "dispatch_kwargs", "expect_full"),
    [
        (
            CUDAGraphMode.FULL_DECODE_ONLY,
            False,
            {"num_reqs": 2, "num_tokens": 2, "uniform_token_count": 1},
            True,
        ),
        (
            CUDAGraphMode.FULL_AND_PIECEWISE,
            False,
            {"num_reqs": 2, "num_tokens": 2, "uniform_token_count": 1},
            True,
        ),
        (
            CUDAGraphMode.FULL_AND_PIECEWISE,
            False,
            {"num_reqs": 2, "num_tokens": 4, "uniform_token_count": None},
            False,
        ),
        (
            CUDAGraphMode.FULL,
            False,
            {"num_reqs": 2, "num_tokens": 2, "uniform_token_count": 1},
            True,
        ),
        (
            CUDAGraphMode.FULL_DECODE_ONLY,
            True,
            {
                "num_reqs": 2,
                "num_tokens": 2,
                "uniform_token_count": None,
                "max_query_len": 2,
            },
            True,
        ),
    ],
)
def test_cudagraph_manager_dual_capture_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    cg_mode: CUDAGraphMode,
    varlen_decode: bool,
    dispatch_kwargs: dict[str, int | None],
    expect_full: bool,
) -> None:
    monkeypatch.setattr(cudagraph_utils, "get_pp_group", lambda: _DummyPPGroup())
    monkeypatch.setattr(
        cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: None,
    )

    decode_lens = {1: 4, 2: 4} if varlen_decode else {1: 4}
    vllm_config = _make_split_vllm_config(
        cudagraph_mode=cg_mode,
        decode_request_sizes=[1, 2, 4],
        prefill_token_sizes=[1, 2, 4, 8],
    )

    manager = CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=cg_mode,
        decode_query_lens=decode_lens,
        varlen_decode=varlen_decode,
        decode_request_sizes=[1, 2, 4],
    )
    manager._graphs_captured = True

    full_descs = manager._capture_descs[CUDAGraphMode.FULL]
    unsplit_shapes = {
        (d.num_reqs, d.num_tokens, d.uniform_token_count, d.max_query_len)
        for d in full_descs
        if not d.split_attention
    }
    split_shapes = {
        (d.num_reqs, d.num_tokens, d.uniform_token_count, d.max_query_len)
        for d in full_descs
        if d.split_attention
    }
    assert unsplit_shapes and unsplit_shapes == split_shapes

    unsplit = manager.dispatch(
        num_active_loras=0, split_attention=False, **dispatch_kwargs
    )
    assert unsplit.split_attention is False
    expected_mode = CUDAGraphMode.FULL if expect_full else CUDAGraphMode.PIECEWISE
    assert unsplit.cg_mode == expected_mode

    split = manager.dispatch(
        num_active_loras=0, split_attention=True, **dispatch_kwargs
    )
    assert split.cg_mode == expected_mode
    assert split.split_attention is expect_full


def test_input_buffers_preallocates_deterministic_attention_splits() -> None:
    buffers = InputBuffers(
        max_num_reqs=4, max_num_tokens=16, device=torch.device("cpu")
    )
    assert buffers.deterministic_attention_splits.shape == (4,)
    assert buffers.deterministic_attention_splits.dtype == torch.int32
    assert torch.all(buffers.deterministic_attention_splits == 1)


def test_deterministic_split_stride_caps_default_max_num_seqs_and_rejects_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = EngineArgs(deterministic_attention_split_stride=512)
    monkeypatch.setattr(
        args,
        "get_batch_defaults",
        lambda _world_size: (
            {UsageContext.OPENAI_API_SERVER: 8192},
            {UsageContext.OPENAI_API_SERVER: 1024},
        ),
    )
    model_config = ModelConfig(max_model_len=64, skip_tokenizer_init=True)
    args._set_default_max_num_seqs_and_batched_tokens_args(
        UsageContext.OPENAI_API_SERVER,
        model_config,
        ParallelConfig(),
    )
    assert args.max_num_seqs == FA3_MAX_DYNAMIC_SPLIT_REQUESTS

    with pytest.raises(ValueError, match="at most 992 padded requests"):
        VllmConfig(
            model_config=model_config,
            parallel_config=ParallelConfig(),
            cache_config=CacheConfig(),
            scheduler_config=SchedulerConfig(
                max_num_seqs=FA3_MAX_DYNAMIC_SPLIT_REQUESTS + 1,
                max_num_batched_tokens=2048,
                max_model_len=64,
                is_encoder_decoder=False,
            ),
            attention_config=AttentionConfig(deterministic_attention_split_stride=512),
        )


def test_build_attn_metadata_slices_deterministic_splits() -> None:
    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[new_kv_cache_group(["layer.0"], spec)],
    )
    builder = _RecordingMetadataBuilder()
    group = AttentionGroup(
        backend=_DummyBackend,
        layer_names=["layer.0"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        metadata_builders=[cast(Any, builder)],
    )

    # Padded length is 4 (e.g. prepare_inputs), unpadded num_reqs is 2 (PIECEWISE).
    splits_padded = torch.tensor([3, 2, 1, 1], dtype=torch.int32)
    build_attn_metadata(
        attn_groups=[[group]],
        num_reqs=2,
        num_tokens=4,
        query_start_loc_gpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        max_query_len=2,
        seq_lens=torch.tensor([1024, 600, 0, 0], dtype=torch.int32),
        max_seq_len=1024,
        block_tables=[torch.zeros((2, 4), dtype=torch.int32)],
        slot_mappings=torch.zeros((1, 4), dtype=torch.int64),
        kv_cache_config=kv_cache_config,
        deterministic_attention_splits=splits_padded,
    )
    assert builder.recorded is not None
    assert builder.recorded.deterministic_attention_splits is not None
    assert builder.recorded.deterministic_attention_splits.shape == (2,)
    assert torch.equal(
        builder.recorded.deterministic_attention_splits,
        torch.tensor([3, 2], dtype=torch.int32),
    )


def test_flash_attn_metadata_builder_length_keyed_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(flash_attn_mod, "get_flash_attn_version", lambda **kwargs: 3)
    monkeypatch.setattr(
        flash_attn_mod,
        "get_dcp_world_size_and_rank",
        lambda _sharded: (1, 0),
    )
    monkeypatch.setattr(
        flash_attn_mod,
        "_get_sliding_window_configs",
        lambda _cfg: {(-1, -1)},
    )
    monkeypatch.setattr(
        flash_attn_mod,
        "get_num_attention_heads_from_layers",
        lambda _cfg, _layers: 4,
    )

    recorded_sched_calls: list[dict[str, object]] = []

    def fake_sched_meta(**kwargs: object) -> torch.Tensor:
        recorded_sched_calls.append(kwargs)
        return torch.ones(4, dtype=torch.int32)

    monkeypatch.setattr(
        flash_attn_mod,
        "get_scheduler_metadata",
        fake_sched_meta,
        raising=False,
    )
    init_workspace_manager(torch.device("cpu"))
    try:
        vllm_config = _make_split_vllm_config(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            enable_jit_warmup=False,
        )
        spec = AttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
        )
        builder = flash_attn_mod.FlashAttentionMetadataBuilder(
            spec, ["layer.0"], vllm_config, torch.device("cpu")
        )

        common_no_split = _make_common_attn_metadata([256, 256], None)
        meta_no_split = builder.build(0, common_no_split)
        assert meta_no_split.max_num_splits == 1
        assert meta_no_split.split_workspace is None
        assert meta_no_split.use_fixed_mma_warpgroups is True
        assert recorded_sched_calls == []

        split_tensor = torch.tensor([1, 2], dtype=torch.int32)
        common_split = _make_common_attn_metadata([256, 700], split_tensor)
        meta_split = builder.build(0, common_split)
        assert meta_split.max_num_splits == FA3_MAX_DETERMINISTIC_SPLITS
        assert meta_split.split_workspace is not None
        assert meta_split.use_fixed_mma_warpgroups is True
        assert len(recorded_sched_calls) == 1
        assert recorded_sched_calls[0]["num_splits_per_request"] is split_tensor
        assert recorded_sched_calls[0]["use_fixed_mma_warpgroups"] is True
    finally:
        reset_workspace_manager()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Requires Hopper SM90 CUDA GPU",
)
def test_fa3_split_external_workspace_large_capacity() -> None:
    """Verify split accumulators when external workspace capacity exceeds total_q."""
    from vllm.vllm_flash_attn import flash_attn_interface as fa

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads, num_kv_heads, head_size = 32, 8, 128
    block_size = 64
    seqlens = [200, 1400, 1700, 2200]
    splits = [1, 3, 4, 5]
    max_seqlen_k = 4096

    torch.manual_seed(0)
    max_blocks_per_req = (max_seqlen_k + block_size - 1) // block_size
    num_blocks = len(seqlens) * max_blocks_per_req
    k_cache = torch.randn(
        (num_blocks, block_size, num_kv_heads, head_size), dtype=dtype, device=device
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        len(seqlens), max_blocks_per_req
    )
    q = torch.randn((len(seqlens), num_heads, head_size), dtype=dtype, device=device)
    cu_seqlens_q = torch.arange(len(seqlens) + 1, dtype=torch.int32, device=device)
    seqused_k = torch.tensor(seqlens, dtype=torch.int32, device=device)
    splits_t = torch.tensor(splits, dtype=torch.int32, device=device)

    def _run_fa3(
        q_in: torch.Tensor,
        cu_q: torch.Tensor,
        sk: torch.Tensor,
        bt: torch.Tensor,
        *,
        num_splits: int,
        splits_per_req: torch.Tensor | None = None,
        split_workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        sched_meta = fa.get_scheduler_metadata(
            batch_size=int(sk.numel()),
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=num_heads,
            num_heads_kv=num_kv_heads,
            headdim=head_size,
            cache_seqlens=sk,
            qkv_dtype=dtype,
            cu_seqlens_q=cu_q,
            page_size=block_size,
            causal=True,
            num_splits=num_splits,
            num_splits_per_request=splits_per_req,
            use_fixed_mma_warpgroups=True,
        )
        return fa.flash_attn_varlen_func(
            q_in,
            k_cache,
            v_cache,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=max_seqlen_k,
            seqused_k=sk,
            causal=True,
            block_table=bt,
            scheduler_metadata=sched_meta,
            num_splits=num_splits,
            fa_version=3,
            split_workspace=split_workspace,
            use_fixed_mma_warpgroups=True,
        )

    ref_out = _run_fa3(q, cu_seqlens_q, seqused_k, block_table, num_splits=1)

    cap = 256
    out_accum = torch.full(
        (FA3_MAX_DETERMINISTIC_SPLITS, num_heads, cap, head_size),
        1234.5,
        dtype=torch.float32,
        device=device,
    )
    lse_accum = torch.full(
        (FA3_MAX_DETERMINISTIC_SPLITS, num_heads, cap),
        1234.5,
        dtype=torch.float32,
        device=device,
    )
    split_out = _run_fa3(
        q,
        cu_seqlens_q,
        seqused_k,
        block_table,
        num_splits=FA3_MAX_DETERMINISTIC_SPLITS,
        splits_per_req=splits_t,
        split_workspace=(out_accum, lse_accum),
    )
    assert torch.allclose(split_out, ref_out, atol=2e-2, rtol=2e-2)

    # Single-request replay against a dirtied large-capacity workspace matches bitwise.
    out_1 = _run_fa3(
        q[2:3],
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        seqused_k[2:3],
        block_table[2:3],
        num_splits=FA3_MAX_DETERMINISTIC_SPLITS,
        splits_per_req=splits_t[2:3],
        split_workspace=(out_accum, lse_accum),
    )
    assert torch.equal(split_out[2:3], out_1)
