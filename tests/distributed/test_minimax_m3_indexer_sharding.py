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

PREFILL_CHUNK_SIZE = 1024
PREFILL_SCORE_BYTES = 32 << 20


@ray.remote(num_gpus=1, max_calls=1)
def _indexer_sharding(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.models.minimax_m3.nvidia import indexer_msa as msa

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.setattr(msa, "PREFILL_FMHA_CHUNK_SIZE", PREFILL_CHUNK_SIZE)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        config = create_vllm_config(
            model_name="Qwen/Qwen3-0.6B",
            tensor_parallel_size=tp_size,
            block_size=128,
            max_model_len=4096,
            max_num_seqs=4,
            max_num_batched_tokens=1536,
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
            assert comm.scores.shape == (comm.max_decode_tokens, 4, 32)
            for page_size in (128, 256):
                config.cache_config.block_size = page_size
                comm = msa.get_comm(config, 4)
                assert comm is not None
                _check_indexer_forwards(config, device, rank, comm)
            config.cache_config.block_size = 128
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

    for failure in (
        "topology",
        "recv_allocation",
    ):
        with pytest.MonkeyPatch.context() as m:
            m.setattr(msa, "_COMM", None)
            m.setattr(msa, "_COMM_FAILED", False)
            injected = False

            def fail_allocation(*args, **kwargs):
                nonlocal injected
                injected = True
                raise RuntimeError("injected indexer allocation failure")

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
    page_size = config.cache_config.block_size
    local_heads = 4 // world
    # Five layers exercise all three receive slot sets, including wrap across
    # steps. Keep the actual production communication objects and dispatch.
    comm.num_layers = 0
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
                    prefix=f"indexer_p{page_size}_{sharded}_{layer}",
                    cache_config=config.cache_config,
                    indexer_kv_dtype="fp8",
                    index_k_shard=world if sharded else 1,
                    topk_indices_buffer=torch.empty(
                        (1536, local_heads, 16), dtype=torch.int32, device=device
                    ),
                )
                for layer in range(5)
            ]
        )
    assert [impl.index_k_ordinal for impl in layers[1]] == list(range(5))
    assert all(impl.index_k_comm is comm for impl in layers[1])
    builders = []
    for impls, prefill_mode in (
        (layers[0], "gather"),
        (layers[1], "gather"),
        (layers[1], "local_score"),
        (layers[1], "local_score_unified"),
    ):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(config.attention_config, "indexer_prefill_cp", prefill_mode)
            builders.append(
                msa.MiniMaxM3IndexerMSAMetadataBuilder(
                    impls[0].index_cache.get_kv_cache_spec(config),
                    [impl.index_cache.prefix for impl in impls],
                    config,
                    device,
                )
            )
    assert all(builder.unified_scores_buffer is not None for builder in builders[:2])
    assert all(builder.unified_scores_buffer is None for builder in builders[2:])
    prefill_storage = [
        (
            builder.prefill_scores_buffer.data_ptr(),
            builder.prefill_candidates_buffer.data_ptr(),
        )
        for builder in builders[2:]
    ]

    active_layers = 5

    def local_shard(cache):
        if page_size == 256:
            rows = page_size // world
            return cache[:, rank * rows : (rank + 1) * rows]
        return cache[:, rank::world]

    def run(sharded, metadata, queries):
        impls = layers[sharded][:active_layers]
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

    def check_local(metadata, queries, num_tokens):
        run(False, metadata, queries)
        for reference, sharded in zip(*(impls[:active_layers] for impls in layers)):
            torch.testing.assert_close(
                sharded.topk_indices_buffer[:num_tokens].sort(dim=-1).values,
                reference.topk_indices_buffer[:num_tokens].sort(dim=-1).values,
                rtol=0,
                atol=0,
            )
        if (
            metadata[1].prefill_msa is not None
            and metadata[1].prefill_msa.local_chunks is None
        ):
            nd = metadata[1].num_decode_tokens
            torch.testing.assert_close(
                metadata[1].unified_scores[nd:],
                metadata[0].unified_scores[nd:],
                rtol=0,
                atol=0,
            )
        assert comm.errors.item() == 0
        assert torch.count_nonzero(comm.recv[..., 32]).item() == 0
        if (
            metadata[1].prefill_msa is None
            and active_layers > 1
            and comm.tagged_enabled
        ):
            tags = (comm.tagged_recv[..., :num_tokens, :] >> 32) & 0xFFFFFFFF
            generations = comm.expected_generations[:, None, :, :num_tokens, None]
            torch.testing.assert_close(
                tags, (generations.long() & 0xFFFFFFFF).expand_as(tags)
            )

    def check(metadata, queries, num_tokens):
        error = None
        try:
            check_local(metadata, queries, num_tokens)
        except AssertionError as failure:
            error = failure
        # No rank starts publishing the next phase before every snapshot completes.
        ok = torch.tensor([int(error is None)], dtype=torch.int32, device="cpu")
        torch.distributed.all_reduce(
            ok, op=torch.distributed.ReduceOp.MIN, group=comm.group.cpu_group
        )
        if not ok.item():
            raise AssertionError(
                f"Indexer validation failed on a TP rank: {error}"
            ) from error

    exchanges = []
    all_to_all = torch.distributed.all_to_all_single

    def exchange(received, send, **kwargs):
        exchanges.append(tuple(send.shape))
        return all_to_all(received, send, **kwargs)

    batches = [
        BatchSpec(seq_lens=[129, 2305], query_lens=[1, 1]),
        BatchSpec(seq_lens=[128, 2306], query_lens=[4, 4]),
        BatchSpec(seq_lens=[128, 2307], query_lens=[8, 8]),
        BatchSpec(seq_lens=[2305, 2561, 2691, 2720], query_lens=[1, 1, 1027, 127]),
        BatchSpec(seq_lens=[2308, 2564, 2691, 2720], query_lens=[4, 4, 1027, 127]),
        BatchSpec(seq_lens=[2305, 2562, 2691, 2720], query_lens=[1, 2, 3, 4]),
        BatchSpec(seq_lens=[2691], query_lens=[1027]),
        BatchSpec(seq_lens=[129, 2305], query_lens=[1, 1]),
    ]
    for batch, active_layers in zip(batches, (5, 2, 1, 5, 5, 2, 1, 5), strict=True):
        # Exercise wrap-safe rotation, single-layer ACK fallback and mixed transitions.
        comm.num_layers = active_layers
        common = create_common_attn_metadata(
            batch, page_size, device, arange_block_indices=True
        )
        num_pages = common.block_table_tensor.numel()
        common.block_table_tensor = torch.randperm(num_pages, device=device)[
            common.block_table_tensor.long()
        ].int()
        # Runtime tables keep their configured width across batches.
        table_width = (config.model_config.max_model_len + page_size - 1) // page_size
        common.block_table_tensor = torch.nn.functional.pad(
            common.block_table_tensor,
            (0, table_width - common.block_table_tensor.shape[1]),
        )
        if len(set(batch.query_lens)) > 1:
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
        blocks_per_page = page_size // 128
        num_blocks = num_pages * blocks_per_page
        assert num_blocks <= 1 << bits.numel()
        page_codes = (
            (torch.arange(num_blocks, device=device)[:, None] >> bits) & 1
        ) * 64
        head_codes = torch.tensor([0, 21, 42, 85], device=device)[:, None]
        queries, caches = [], []
        for layer, (reference, sharded) in enumerate(zip(*layers)):
            # Logical B128 codes keep P256 half-pages distinct at the cutoff.
            cache = torch.randint(-4, 5, (num_pages, page_size, 128), device=device)
            cache.view(num_pages, blocks_per_page, 128, 128)[..., :7] = page_codes.view(
                num_pages, blocks_per_page, 1, 7
            )
            cache = cache.to(torch.float8_e4m3fn)
            reference.index_cache.bind_kv_cache(cache.unsqueeze(1))
            shard = local_shard(cache).contiguous()
            sharded.index_cache.bind_kv_cache(
                shard.view(num_pages, 1, page_size, 128 // world)
            )
            caches.append(cache)
            query = (
                torch.randint(-2, 3, (num_tokens, 4, 128), device=device).float() / 64
            )
            query[:, :, :7] = (1 << bits) * (
                1 - 2 * (((head_codes ^ (layer * 7)) >> bits) & 1)
            )
            query[PREFILL_CHUNK_SIZE:] *= -1
            queries.append(query.to(torch.float8_e4m3fn))
        metadata = [builder.build(0, common) for builder in builders]
        for md, storage in zip(metadata[2:], prefill_storage, strict=True):
            assert md.unified_scores is None
            if md.prefill_msa is not None:
                prefill = md.prefill_msa
                assert (
                    prefill.local_scores.untyped_storage().data_ptr(),
                    prefill.candidate_buffers.untyped_storage().data_ptr(),
                ) == storage
                assert all(arm.is_contiguous() for arm in prefill.candidate_buffers)
        expected_decode = {
            (4, 4, 1027, 127): (2, 8),
            (1, 2, 3, 4): (1, 1),
        }.get(tuple(batch.query_lens))
        if expected_decode is not None:
            assert all(
                (md.num_decodes, md.num_decode_tokens) == expected_decode
                for md in metadata
            )

        # Compile before capture. No host assertion/barrier between layers or
        # repeated steps: those would hide receive-slot reuse races.
        run(False, metadata, queries)
        for _ in range(3):
            run(True, metadata, queries)
        check(metadata, queries, num_tokens)

        if metadata[1].prefill_msa is None:
            for decode_metadata in metadata[1:]:
                metadata[1] = decode_metadata
                run(True, metadata, queries)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(True, metadata, queries)
                for _ in range(3):
                    for query, cache, impl in zip(queries, caches, layers[1]):
                        query.copy_((-query.float()).to(query.dtype))
                        cache.view(torch.uint8).copy_(
                            cache.view(torch.uint8).roll(1, 0)
                        )
                        impl.index_cache.kv_cache.copy_(local_shard(cache))
                    graph.replay()
                check(metadata, queries, num_tokens)
        else:
            for local_metadata in metadata[2:]:
                metadata[1] = local_metadata
                prefill = local_metadata.prefill_msa
                scratch = prefill.local_scores
                assert scratch.shape[0] <= PREFILL_CHUNK_SIZE
                assert scratch.numel() * scratch.element_size() <= PREFILL_SCORE_BYTES
                chunks = prefill.local_chunks
                assert chunks[0].start == 0
                assert chunks[-1].end == prefill.candidate_buffers.shape[2]
                if chunks[-1].end > PREFILL_CHUNK_SIZE:
                    assert len(chunks) > 1
                assert all(chunk.plan is not None for chunk in chunks)
                # Both dispatches reuse scratch across layers and steps; unified
                # mixed scoring exchanges decode and prefill candidates together.
                exchanges.clear()
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(torch.distributed, "all_to_all_single", exchange)
                    for _ in range(3):
                        run(True, metadata, queries)
                exchanged_tokens = (
                    num_tokens if prefill.unified else local_metadata.num_prefill_tokens
                )
                assert exchanges == [(world, local_heads, exchanged_tokens, 33)] * (
                    3 * active_layers
                )
                check(metadata, queries, num_tokens)
            pages = torch.tensor([num_pages - 1, 0, 7, 0], device=device)
            gathered = [
                msa.gather_prefill_pages(
                    impl.index_cache.kv_cache, pages, comm, page_size=page_size
                )
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
    """Real TP score/TopK, both prefill paths, and decode graph replay."""
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
            policy.model_config.max_model_len = length * 2
            resized = msa.get_comm(policy, 4)
            assert resized is not current
            assert resized.scores.shape[-1] == 2 * current.scores.shape[-1]
            policy.model_config.max_model_len = length
            assert msa.get_comm(policy, 4) is not resized
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
