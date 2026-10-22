# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import faulthandler
from copy import copy

import pytest
import ray
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import (
    AttentionConfig,
    CompilationConfig,
    CUDAGraphMode,
    KVTransferConfig,
    SchedulerConfig,
    SpeculativeConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform

from ..utils import init_test_distributed_environment, multi_process_parallel


@ray.remote(num_gpus=1, max_calls=1)
def _indexer_sharding(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.models.minimax_m3.nvidia import indexer_msa as msa

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        config = create_vllm_config(
            model_name="Qwen/Qwen3-0.6B",
            tensor_parallel_size=tp_size,
            block_size=128,
            max_model_len=4096,
            max_num_seqs=4,
            max_num_batched_tokens=256,
            hf_config_override={
                "sparse_attention_config": {
                    "sparse_num_index_heads": 4,
                    "sparse_index_dim": 128,
                }
            },
        )
        config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=7
        )
        config.attention_config.indexer_cache_sharding = "on"
        with set_current_vllm_config(config):
            init_test_distributed_environment(
                tp_size, pp_size, rank, distributed_init_port
            )
            _check_initialization_fallback(config, device, rank)
            assert msa.resolve_index_k_shard(config, 4, "fp8", 16) == tp_size
            comm = msa.get_comm()
            assert comm is not None
            _check_indexer_forwards(config, device, rank, comm)
            _check_sharding_policy(config, comm)
        get_tp_group().barrier()


def _check_initialization_fallback(config, device, rank):
    import torch.distributed._symmetric_memory as symm_mem

    from vllm.models.minimax_m3.nvidia import indexer_msa as msa

    for name, value in (("enable_dbo", True), ("ubatch_size", 2)):
        with pytest.MonkeyPatch.context() as m:
            m.setattr(config.parallel_config, name, value)
            assert msa.resolve_index_k_shard(config, 4, "fp8", 16) == 1
            assert msa.get_comm() is None
            assert not msa._COMM_FAILED

    original_empty = symm_mem.empty
    for failure in (
        "topology",
        "recv_allocation",
    ):
        with pytest.MonkeyPatch.context() as m:
            m.setattr(msa, "_COMM", None)
            m.setattr(msa, "_COMM_FAILED", False)
            injected = False

            def fail_allocation(*args, failure=failure, **kwargs):
                nonlocal injected
                shape = args[0]
                if failure == "recv_allocation" and len(shape) == 5:
                    injected = True
                    raise RuntimeError("injected indexer allocation failure")
                return original_empty(*args, **kwargs)

            def fail_topology(_):
                nonlocal injected
                injected = True
                return False

            if rank == 1:
                if failure == "topology":
                    m.setattr(current_platform, "is_fully_connected", fail_topology)
                else:
                    m.setattr(symm_mem, "empty", fail_allocation)

            # A rank stranded in rendezvous or GPU polling cannot raise a Python
            # timeout. Exit that worker so Ray reports failure and stops peers.
            faulthandler.dump_traceback_later(90, exit=True)
            try:
                shard = msa.resolve_index_k_shard(config, 4, "fp8", 16)
                assert injected == (rank == 1)
                assert shard == 1
                assert msa.get_comm() is None
                assert msa._COMM_FAILED
            finally:
                faulthandler.cancel_dump_traceback_later()


def _check_indexer_forwards(config, device, rank, comm):
    from vllm.models.minimax_m3.nvidia import indexer_msa as msa

    torch.manual_seed(123)
    world = comm.world
    local_heads = 4 // world
    # Five layers exercise all three receive slot sets, including wrap across
    # steps. Keep the actual production communication objects and dispatch.
    layers = []
    for sharded in (False, True):
        layers.append(
            [
                msa.MiniMaxM3IndexerMSAImpl(
                    num_kv_heads=local_heads,
                    scale=128**-0.5,
                    topk_blocks=16,
                    sparse_block_size=128,
                    num_index_heads=local_heads,
                    index_head_dim=128,
                    init_blocks=1,
                    local_blocks=1,
                    prefix=f"indexer_{sharded}_{layer}",
                    cache_config=config.cache_config,
                    indexer_kv_dtype="fp8",
                    index_k_shard=world if sharded else 1,
                    topk_indices_buffer=torch.empty(
                        (256, local_heads, 16), dtype=torch.int32, device=device
                    ),
                )
                for layer in range(5)
            ]
        )
    builders = [
        msa.MiniMaxM3IndexerMSAMetadataBuilder(
            impls[0].index_cache.get_kv_cache_spec(config),
            [impl.index_cache.prefix for impl in impls],
            config,
            device,
        )
        for impls in layers
    ]

    def run(sharded, metadata, queries):
        impls = layers[sharded]
        md = {impl.index_cache.prefix: metadata[sharded] for impl in impls}
        with set_forward_context(md, config):
            for impl, query in zip(impls, queries):
                impl(
                    query
                    if sharded
                    else query[
                        :, rank * local_heads : (rank + 1) * local_heads
                    ].contiguous()
                )

    def check(metadata, queries, num_tokens):
        run(False, metadata, queries)
        for reference, sharded in zip(*layers):
            torch.testing.assert_close(
                sharded.topk_indices_buffer[:num_tokens].sort(dim=-1).values,
                reference.topk_indices_buffer[:num_tokens].sort(dim=-1).values,
                rtol=0,
                atol=0,
            )
        if metadata[1].prefill_msa is not None:
            nd = metadata[1].num_decode_tokens
            torch.testing.assert_close(
                metadata[1].unified_scores[nd:],
                metadata[0].unified_scores[nd:],
                rtol=0,
                atol=0,
            )
        assert comm.errors.item() == 0
        assert torch.count_nonzero(comm.recv[..., 32]).item() == 0

    batches = [
        BatchSpec(seq_lens=[129, 2305], query_lens=[1, 1]),
        BatchSpec(seq_lens=[128, 2306], query_lens=[4, 4]),
        BatchSpec(seq_lens=[128, 2307], query_lens=[8, 8]),
        BatchSpec(seq_lens=[2305, 2561, 2624, 2720], query_lens=[1, 1, 64, 96]),
        BatchSpec(seq_lens=[129, 2305], query_lens=[1, 1]),
    ]
    for batch in batches:
        common = create_common_attn_metadata(
            batch, 128, device, arange_block_indices=True
        )
        num_pages = common.block_table_tensor.numel()
        common.block_table_tensor = torch.randperm(num_pages, device=device)[
            common.block_table_tensor.long()
        ].int()
        if batch.query_lens[-1] > 8:
            # Shared prefix pages must retain each request's logical order.
            common.block_table_tensor[-1, :8] = common.block_table_tensor[-2, :8]
        common.positions = torch.cat(
            [
                torch.arange(s - q, s, device=device)
                for s, q in zip(batch.seq_lens, batch.query_lens)
            ]
        )
        num_tokens = batch.compute_num_tokens()
        bits = torch.arange(7, device=device)
        page_codes = (
            (torch.arange(num_pages, device=device)[:, None] >> bits) & 1
        ) * 64
        head_codes = torch.tensor([0, 21, 42, 85], device=device)[:, None]
        queries, caches = [], []
        for layer, (reference, sharded) in enumerate(zip(*layers)):
            # Page bits dominate the small random row contribution: each head
            # has distinct block scores, with a different ranking per head.
            cache = torch.randint(-4, 5, (num_pages, 128, 128), device=device)
            cache[:, :, :7] = page_codes[:, None]
            cache = cache.to(torch.float8_e4m3fn)
            reference.index_cache.bind_kv_cache(cache.unsqueeze(1))
            shard = cache[:, rank::world].contiguous()
            sharded.index_cache.bind_kv_cache(
                shard.view(num_pages, 1, 128, 128 // world)
            )
            caches.append(cache)
            query = (
                torch.randint(-2, 3, (num_tokens, 4, 128), device=device).float() / 64
            )
            query[:, :, :7] = (1 << bits) * (
                1 - 2 * (((head_codes ^ (layer * 7)) >> bits) & 1)
            )
            queries.append(query.to(torch.float8_e4m3fn))
        metadata = [builder.build(0, common) for builder in builders]

        # Compile before capture. No host assertion/barrier between layers or
        # repeated steps: those would hide receive-slot reuse races.
        run(False, metadata, queries)
        for _ in range(3):
            run(True, metadata, queries)
        check(metadata, queries, num_tokens)

        if metadata[1].prefill_msa is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(True, metadata, queries)
            for _ in range(3):
                for query, cache, impl in zip(queries, caches, layers[1]):
                    query.copy_((-query.float()).to(query.dtype))
                    cache.view(torch.uint8).copy_(cache.view(torch.uint8).roll(1, 0))
                    impl.index_cache.kv_cache.copy_(cache[:, rank::world])
                graph.replay()
            check(metadata, queries, num_tokens)
        else:
            pages = torch.tensor([num_pages - 1, 0, 7, 0], device=device)
            gathered = [
                msa.gather_prefill_pages(impl.index_cache.kv_cache, pages, comm)
                for impl in layers[1]
            ]
            for actual, cache in zip(gathered, caches):
                expected = cache.view(torch.uint8).index_select(0, pages)
                assert torch.equal(actual.view(torch.uint8), expected)


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="MiniMax M3 sharded MSA indexer requires SM100",
)
@pytest.mark.parametrize("tp_size", [2, 4])
def test_msa_indexer_tp_matches_replicated(monkeypatch, tp_size):
    """Real TP score/TopK, prefill gather, and mailbox reuse under graph replay."""
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} GPUs")
    if not all(
        torch.cuda.can_device_access_peer(src, dst)
        for src in range(tp_size)
        for dst in range(tp_size)
        if src != dst
    ):
        pytest.skip("Requires peer access between all TP GPUs")
    multi_process_parallel(monkeypatch, tp_size, 1, _indexer_sharding)


def _check_sharding_policy(config, original_comm):
    from vllm.models.minimax_m3.nvidia import indexer_msa as msa

    sparse_config = config.model_config.hf_text_config.sparse_attention_config
    policy = create_vllm_config(
        tensor_parallel_size=original_comm.world,
        block_size=128,
        max_model_len=4096,
        max_num_seqs=32,
        hf_config_override={"sparse_attention_config": sparse_config},
    )
    model = copy(policy.model_config)
    for mode, length, cap, active_cap, q, eager, graph_mode, expected in [
        ("on", 4096, 1, None, 4, False, CUDAGraphMode.FULL, original_comm.world),
        ("off", 1024 * 1024, 32, None, 1, False, CUDAGraphMode.FULL, 1),
        ("on", 4096, 1, None, 1, True, CUDAGraphMode.NONE, original_comm.world),
        ("off", 1024 * 1024, 32, None, 1, True, CUDAGraphMode.NONE, 1),
    ]:
        policy.model_config = copy(model)
        policy.model_config.max_model_len = length
        policy.model_config.enforce_eager = eager
        policy.scheduler_config = SchedulerConfig(
            max_model_len=length,
            is_encoder_decoder=False,
            max_num_seqs=cap,
            max_num_active_seqs=active_cap,
            max_num_batched_tokens=8192,
            enable_chunked_prefill=True,
        )
        policy.attention_config = AttentionConfig(indexer_cache_sharding=mode)
        policy.compilation_config = CompilationConfig(cudagraph_mode=graph_mode)
        policy.speculative_config = (
            SpeculativeConfig(method="ngram", num_speculative_tokens=q - 1)
            if q > 1
            else None
        )
        before = msa.get_comm()
        assert msa.resolve_index_k_shard(policy, 4, "fp8", 16) == expected
        current = msa.get_comm()
        if expected == 1:
            assert current is before
        else:
            assert current is not before
            assert current.max_decode_tokens == cap * q
            assert msa.get_comm(policy, 4) is current
    assert original_comm.num_layers == 5
    assert original_comm.max_decode_tokens == 4 * 8

    policy.model_config = copy(model)
    policy.model_config.max_model_len = 1024 * 1024
    policy.compilation_config = CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL)
    policy.scheduler_config = SchedulerConfig(
        max_model_len=1024 * 1024,
        is_encoder_decoder=False,
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
    )
    for mode, dtype, topk, pcp, dcp, q, transfer in [
        ("on", "bf16", 16, 1, 1, 1, False),
        ("on", "fp8", 8, 1, 1, 1, False),
        ("on", "fp8", 16, 2, 1, 1, False),
        ("on", "fp8", 16, 1, 2, 1, False),
        ("on", "fp8", 16, 1, 1, 9, False),
        ("on", "fp8", 16, 1, 1, 1, True),
    ]:
        policy.attention_config = AttentionConfig(indexer_cache_sharding=mode)
        policy.parallel_config = copy(config.parallel_config)
        policy.parallel_config.prefill_context_parallel_size = pcp
        policy.parallel_config.decode_context_parallel_size = dcp
        policy.speculative_config = (
            SpeculativeConfig(method="ngram", num_speculative_tokens=q - 1)
            if q > 1
            else None
        )
        policy.kv_transfer_config = (
            KVTransferConfig(kv_connector="NixlConnector", kv_role="kv_consumer")
            if transfer
            else None
        )
        before = msa.get_comm()
        assert msa.resolve_index_k_shard(policy, 4, dtype, topk) == 1
        assert msa.get_comm() is before
