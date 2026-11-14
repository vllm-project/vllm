# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query-gather ordering, using real MLA layers and distributed groups."""

from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    KernelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_dcp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    MLACommonMetadata,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.v1.attention.backends.mla.prefill.cpu_sdpa import CPUSDPAMLAPrefillBackend
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend


def _make_layer(
    config, rank, world_size, rope_dim, qrep, prefix, num_heads=8, nope_dim=128
):
    with set_current_vllm_config(config):
        projection = ColumnParallelLinear(
            512,
            world_size * num_heads * (nope_dim + 128),
            bias=False,
            prefix=prefix + ".kv_b",
        ).cuda()
        # Different weights on each rank expose head-ordering mistakes.
        torch.manual_seed(42 + rank)
        projection.weight.normal_(std=0.02)
        layer = MLAAttention(
            num_heads=num_heads,
            scale=(nope_dim + rope_dim) ** -0.5,
            qk_nope_head_dim=nope_dim,
            qk_rope_head_dim=rope_dim,
            v_head_dim=128,
            q_lora_rank=None,
            kv_lora_rank=512,
            kv_b_proj=projection,
            dcp_q_replicate=qrep,
            cache_config=config.cache_config,
            attn_backend=TritonMLABackend,
            prefill_backend_cls=CPUSDPAMLAPrefillBackend,
            prefix=prefix,
        )
        layer.process_weights_after_loading()
        return layer


def _query_tensor(query):
    return torch.cat(query, dim=-1) if isinstance(query, tuple) else query


def _check_prepared_query(config, rank, world_size):
    """Narrow gathering defers preparation and its KV write until normal forward."""
    torch.set_default_dtype(torch.bfloat16)
    config.attention_config = AttentionConfig()
    baseline = _make_layer(config, rank, world_size, 64, False, "overlap.base")
    config.attention_config = AttentionConfig(mla_dcp_query_gather_min_tokens=1)
    candidate = _make_layer(config, rank, world_size, 64, False, "overlap.new")
    q = torch.randn(4, 8, 192, device="cuda")
    kv = torch.randn(4, 512, device="cuda")
    pe = torch.randn(4, 1, 64, device="cuda")
    cache = torch.zeros(1, 1, 16, 576, device="cuda")
    candidate.bind_kv_cache(cache)
    slots = torch.arange(4, device="cuda", dtype=torch.int64)
    metadata = MLACommonMetadata(
        num_reqs=1,
        max_query_len=4,
        max_seq_len=4,
        num_actual_tokens=4,
        query_start_loc=torch.tensor([0, 4], device="cuda", dtype=torch.int32),
        slot_mapping=slots,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
    )
    expected = _query_tensor(baseline._prepare_mqa_query(q, False))
    with set_forward_context(
        {candidate.layer_name: metadata},
        config,
        num_tokens=4,
        slot_mapping={candidate.layer_name: slots},
    ):
        prepared = candidate.prepare_mqa_query(q, kv, pe)
    if world_size > 1:
        assert prepared is None
        assert torch.count_nonzero(cache) == 0
        prepared = _query_tensor(candidate._prepare_mqa_query(q, False))
    else:
        assert prepared is not None
        torch.testing.assert_close(cache[0, 0, :4, :512], kv, rtol=0, atol=0)
    torch.testing.assert_close(prepared, expected, rtol=0, atol=0)


@torch.inference_mode()
def _worker(rank, world_size, init_method, local_rank=None):
    local_rank = rank if local_rank is None else local_rank
    torch.accelerator.set_device_index(local_rank)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size,
            decode_context_parallel_size=world_size,
            disable_custom_all_reduce=True,
            distributed_executor_backend="external_launcher",
        ),
        cache_config=CacheConfig(cache_dtype="auto"),
        scheduler_config=SchedulerConfig(
            max_model_len=8192, max_num_batched_tokens=8192, is_encoder_decoder=False
        ),
        kernel_config=KernelConfig(enable_jit_warmup=False),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            decode_context_model_parallel_size=world_size,
        )
    try:
        for dtype in (torch.bfloat16, torch.float16):
            torch.set_default_dtype(dtype)
            for nope_dim, rope_dim in ((64, 64), (128, 64), (128, 0)):
                for qrep in (False, True):
                    suffix = f"{dtype}.{nope_dim}.{rope_dim}.{qrep}"
                    config.attention_config = AttentionConfig()
                    baseline = _make_layer(
                        config,
                        rank,
                        world_size,
                        rope_dim,
                        qrep,
                        "base." + suffix,
                        nope_dim=nope_dim,
                    )
                    config.attention_config = AttentionConfig(
                        mla_dcp_query_gather_min_tokens=2048
                    )
                    candidate = _make_layer(
                        config,
                        rank,
                        world_size,
                        rope_dim,
                        qrep,
                        "new." + suffix,
                        nope_dim=nope_dim,
                    )
                    assert candidate.dcp_query_gather_min_tokens == (
                        2048 if world_size > 1 else 0
                    )
                    assert (candidate.W_UK_T_dcp_qrep is not None) == (
                        qrep or world_size > 1
                    )
                    for rows in (1, 2047, 2048, 2049, 8192):
                        q = torch.randn(rows, 8, nope_dim + rope_dim, device="cuda")
                        if qrep:
                            q = get_dcp_group().all_gather(q, dim=1)
                        expected = _query_tensor(baseline._prepare_mqa_query(q, qrep))
                        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU],
                            record_shapes=True,
                        ) as profile:
                            actual = _query_tensor(
                                candidate._prepare_mqa_query(q, qrep)
                            )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        bmm = [e for e in profile.events() if e.name == "aten::bmm"]
                        assert len(bmm) == 1
                        replicated = qrep or (world_size > 1 and rows >= 2048)
                        assert bmm[0].input_shapes[0][0] == (
                            8 * world_size if replicated else 8
                        )

                    # Replay with changed inputs and reloaded weights: neither
                    # the gathered query nor the replicated weights may go stale.
                    with get_dcp_group().graph_capture() as context:
                        for _ in range(3):
                            candidate._prepare_mqa_query(q, qrep)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=context.stream):
                            captured = _query_tensor(
                                candidate._prepare_mqa_query(q, qrep)
                            )
                    q.mul_(0.5)
                    expected = _query_tensor(baseline._prepare_mqa_query(q, qrep))
                    graph.replay()
                    torch.testing.assert_close(captured, expected, rtol=0, atol=0)

                    weight = candidate.W_UK_T_dcp_qrep
                    weight_ptr = None if weight is None else weight.data_ptr()
                    for layer in (baseline, candidate):
                        layer.kv_b_proj.weight.mul_(0.5)
                        layer.process_weights_after_loading()
                    if weight_ptr is not None:
                        assert candidate.W_UK_T_dcp_qrep.data_ptr() == weight_ptr
                    expected = _query_tensor(baseline._prepare_mqa_query(q, qrep))
                    graph.replay()
                    torch.testing.assert_close(captured, expected, rtol=0, atol=0)

        _check_prepared_query(config, rank, world_size)
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize("world_size", [1, 2])
def test_mla_query_gather(world_size: int, tmp_path: Path):
    init_method = f"file://{tmp_path / 'dist_init'}"
    mp.spawn(_worker, args=(world_size, init_method), nprocs=world_size, join=True)
