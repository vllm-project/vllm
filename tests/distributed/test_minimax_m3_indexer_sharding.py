# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import faulthandler

import pytest
import ray
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config
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
            assert msa._GATHER is not None
            _check_indexer_forwards(config, device, rank, comm)
        get_tp_group().barrier()


def _check_initialization_fallback(config, device, rank):
    import torch.distributed._symmetric_memory as symm_mem

    from vllm.models.minimax_m3.nvidia import indexer_msa as msa
    from vllm.models.minimax_m3.nvidia.ops.index_k_gather import (
        _index_k_pull_pages_kernel,
    )

    original_empty = symm_mem.empty
    for failure in ("recv_allocation", "gather_allocation", "gather_compile"):
        with pytest.MonkeyPatch.context() as m:
            m.setattr(msa, "_COMM", None)
            m.setattr(msa, "_COMM_FAILED", False)
            m.setattr(msa, "_GATHER", None)
            injected = False

            def fail_allocation(*args, failure=failure, **kwargs):
                nonlocal injected
                shape = args[0]
                if (failure == "recv_allocation" and len(shape) == 5) or (
                    failure == "gather_allocation" and len(shape) == 1
                ):
                    injected = True
                    raise RuntimeError("injected indexer allocation failure")
                return original_empty(*args, **kwargs)

            def fail_compile(*args, **kwargs):
                nonlocal injected
                injected = True
                raise RuntimeError("injected indexer compilation failure")

            if rank == 1:
                if failure == "gather_compile":
                    m.setattr(_index_k_pull_pages_kernel, "warmup", fail_compile)
                else:
                    m.setattr(symm_mem, "empty", fail_allocation)

            # A rank stranded in rendezvous or GPU polling cannot raise a Python
            # timeout. Exit that worker so Ray reports failure and stops peers.
            faulthandler.dump_traceback_later(90, exit=True)
            try:
                shard = msa.resolve_index_k_shard(config, 4, "fp8", 16)
                assert injected == (rank == 1)
                assert msa._GATHER is None
                if failure == "recv_allocation":
                    assert shard == 1
                    assert msa.get_comm() is None
                    assert msa._COMM_FAILED
                else:
                    assert shard == 4
                    assert msa.get_comm() is not None
                    local = torch.full(
                        (1, 32, 128), rank, dtype=torch.uint8, device=device
                    )
                    pages = torch.tensor([0], device=device)
                    full = msa.gather_prefill_pages(local, pages)
                    expected = torch.arange(128, device=device) % 4
                    assert torch.equal(full[0, :, 0].long(), expected)
            finally:
                faulthandler.cancel_dump_traceback_later()


def _check_indexer_forwards(config, device, rank, comm):
    from vllm.models.minimax_m3.nvidia import indexer_msa as msa

    torch.manual_seed(123)
    # Five layers exercise all three receive slot sets, including wrap across
    # steps. Keep the actual production communication objects and dispatch.
    layers = []
    for sharded in (False, True):
        layers.append(
            [
                msa.MiniMaxM3IndexerMSAImpl(
                    num_kv_heads=1,
                    scale=128**-0.5,
                    topk_blocks=16,
                    sparse_block_size=128,
                    num_index_heads=1,
                    index_head_dim=128,
                    init_blocks=1,
                    local_blocks=1,
                    prefix=f"indexer_{sharded}_{layer}",
                    cache_config=config.cache_config,
                    indexer_kv_dtype="fp8",
                    index_k_shard=4 if sharded else 1,
                    topk_indices_buffer=torch.empty(
                        (256, 1, 16), dtype=torch.int32, device=device
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
                impl(query if sharded else query[:, rank : rank + 1].contiguous())

    def check(metadata, queries, num_tokens):
        run(False, metadata, queries)
        for reference, sharded in zip(*layers):
            torch.testing.assert_close(
                sharded.topk_indices_buffer[:num_tokens].sort(dim=-1).values,
                reference.topk_indices_buffer[:num_tokens].sort(dim=-1).values,
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            metadata[1].unified_scores,
            metadata[0].unified_scores,
            rtol=0,
            atol=0,
        )
        assert comm.errors.item() == 0

    batches = [
        BatchSpec(seq_lens=[129, 2305], query_lens=[1, 1]),
        BatchSpec(seq_lens=[128, 2306], query_lens=[4, 4]),
        BatchSpec(seq_lens=[128, 2307], query_lens=[8, 8]),
        BatchSpec(seq_lens=[2305, 2561, 2624, 2720], query_lens=[1, 1, 64, 96]),
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
            # The two prefill requests share prefix pages, which the sharded
            # builder must deduplicate and remap before the score-only call.
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
            shard = cache[:, rank::4].contiguous()
            sharded.index_cache.bind_kv_cache(shard.view(num_pages, 1, 128, 32))
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
                for query in queries:
                    query.copy_((-query.float()).to(query.dtype))
                graph.replay()
            check(metadata, queries, num_tokens)
        else:
            pages = torch.tensor([num_pages - 1, 0, 7, 0], device=device)
            gathered = [
                msa.gather_prefill_pages(impl.index_cache.kv_cache, pages)
                for impl in layers[1]
            ]
            with pytest.MonkeyPatch.context() as m:
                m.setattr(msa, "_GATHER", None)
                fallback = [
                    msa.gather_prefill_pages(impl.index_cache.kv_cache, pages)
                    for impl in layers[1]
                ]
                for _ in range(3):
                    run(True, metadata, queries)
                check(metadata, queries, num_tokens)
            for actual, nccl, cache in zip(gathered, fallback, caches):
                expected = cache.view(torch.uint8).index_select(0, pages)
                assert torch.equal(actual.view(torch.uint8), expected)
                assert torch.equal(nccl.view(torch.uint8), expected)


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="MiniMax M3 sharded MSA indexer requires SM100",
)
def test_msa_indexer_tp4_matches_replicated(monkeypatch):
    """Real TP4 score/TopK, prefill gather, and mailbox reuse under graph replay."""
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four GPUs")
    if not all(
        torch.cuda.can_device_access_peer(src, dst)
        for src in range(4)
        for dst in range(4)
        if src != dst
    ):
        pytest.skip("Requires peer access between all four GPUs")
    multi_process_parallel(monkeypatch, 4, 1, _indexer_sharding)
