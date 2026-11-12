# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest
import ray
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import (
    CompilationConfig,
    KVTransferConfig,
    SpeculativeConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform

from ..utils import init_test_distributed_environment, multi_process_parallel

PREFILL_CHUNK_SIZE = 8192


@pytest.fixture
def cake_indexer():
    cake = pytest.importorskip(
        "flashinfer.experimental.cake_msa_icp_indexer.cake_indexer"
    )
    if not hasattr(cake.MSAIcpIndexerPlan, "layer_graphs"):
        pytest.skip("FlashInfer CAKE layer_graphs API required")


def _indexer_sharding(
    monkeypatch, tp_size, pp_size, rank, distributed_init_port, scenario
):
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
            max_model_len=16384,
            max_num_seqs=4,
            max_num_batched_tokens=8400,
            hf_config_override={
                "index_n_heads": 4,
                "index_head_dim": 128,
                "index_local_blocks": 1,
            },
        )
        config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=7
        )
        config.attention_config.indexer_cache_sharding = True
        config.kv_transfer_config = KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both"
        )
        with set_current_vllm_config(config):
            init_test_distributed_environment(
                tp_size, pp_size, rank, distributed_init_port
            )
            assert msa.resolve_index_k_shard(config, 4, "fp8", 16) == tp_size
            for num_layers in (1, 2, 5):
                config.compilation_config = CompilationConfig()
                comm = msa.IndexKShardComm(config, 4)
                _check_indexer_forwards(
                    config, device, rank, comm, num_layers, scenario
                )
        get_tp_group().barrier()


def _roundtrip_index_caches(config, impls):
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
    from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
    from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

    caches = {impl.index_cache.prefix: impl.index_cache.kv_cache for impl in impls}
    num_pages = next(iter(caches.values())).shape[0]
    placements = [
        KVCacheTensor(
            size=cache.nbytes,
            layers=[name],
            layer_stride=cache.nbytes,
            block_stride=impl.index_cache.get_kv_cache_spec(config).page_size_bytes,
        )
        for impl, (name, cache) in zip(impls, caches.items())
    ]
    worker = SimpleCPUOffloadWorker(
        config,
        KVCacheConfig(num_pages, placements, []),
        sum(cache.nbytes for cache in caches.values()),
    )
    gpu_pages = list(range(num_pages))
    cpu_pages = gpu_pages[::-1]
    try:
        worker.register_kv_caches(caches)
        expected = [cache.view(torch.uint8).clone() for cache in caches.values()]
        worker.bind_connector_metadata(
            SimpleCPUOffloadMetadata(
                store_event=0, store_gpu_blocks=gpu_pages, store_cpu_blocks=cpu_pages
            )
        )
        worker.wait_for_save()
        deadline = time.monotonic() + 30
        while True:
            worker.get_finished(set())
            completed = worker.build_connector_worker_meta()
            if completed is not None:
                assert completed.completed_store_events == {0: 1}
                break
            assert time.monotonic() < deadline, "Index-cache CPU store did not finish"
        for cache in caches.values():
            cache.view(torch.uint8).zero_()
        torch.accelerator.current_stream().synchronize()
        worker.bind_connector_metadata(
            SimpleCPUOffloadMetadata(
                load_event=0,
                load_gpu_blocks=gpu_pages,
                load_cpu_blocks=cpu_pages,
                load_event_to_reqs={0: ["restored"]},
            )
        )
        worker.start_load_kv()
        deadline = time.monotonic() + 30
        while worker.get_finished(set())[1] != {"restored"}:
            assert time.monotonic() < deadline, "Index-cache CPU load did not finish"
        for cache, original in zip(caches.values(), expected):
            assert torch.equal(cache.view(torch.uint8), original)
    finally:
        if worker._backend is not None:
            worker._backend.shutdown()
        torch.accelerator.synchronize()
        for cache in (worker.cpu_kv_caches or {}).values():
            if cache.is_pinned():
                torch.cuda.cudart().cudaHostUnregister(cache.data_ptr())


def _check_indexer_forwards(config, device, rank, comm, num_layers, scenario):
    from vllm.models.minimax_m3.nvidia import indexer_msa as msa

    torch.manual_seed(123)
    world = comm.world
    local_heads = 4 // world
    # Each arm shares one output across layers, as the production model does.
    outputs = [
        torch.empty((8400, local_heads, 16), dtype=torch.int32, device=device)
        for _ in range(2)
    ]
    snapshots = [
        [torch.empty_like(output) for _ in range(num_layers)] for output in outputs
    ]
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
                    init_blocks=0,
                    local_blocks=1,
                    prefix=f"indexer_{sharded}_{layer}",
                    cache_config=config.cache_config,
                    indexer_kv_dtype="fp8",
                    index_k_shard=world if sharded else 1,
                    topk_indices_buffer=outputs[sharded],
                )
                for layer in range(num_layers)
            ]
        )
    for impl in layers[1]:
        impl.set_shard_comm(comm)
    builders = [
        msa.MiniMaxM3IndexerMSAMetadataBuilder(
            impls[0].index_cache.get_kv_cache_spec(config),
            [impl.index_cache.prefix for impl in impls],
            config,
            device,
        )
        for impls in layers
    ]
    prefill_storage = [
        (
            builder.prefill_candidate_pool.data_ptr(),
            builder.prefill_candidates_buffer.data_ptr(),
        )
        for builder in builders[1:]
    ]

    def run(sharded, metadata, queries):
        impls = layers[sharded]
        md = {impl.index_cache.prefix: metadata[sharded] for impl in impls}
        with set_forward_context(md, config):
            for impl, query, snapshot in zip(impls, queries, snapshots[sharded]):
                impl(
                    query
                    if sharded
                    else query[
                        :, rank * local_heads : (rank + 1) * local_heads
                    ].contiguous()
                )
                tokens = metadata[sharded].num_actual_tokens
                snapshot[:tokens].copy_(impl.topk_indices_buffer[:tokens])

    def check_local(metadata, queries, num_tokens):
        run(False, metadata, queries)
        for reference, sharded in zip(*snapshots):
            if scenario == "replay":
                reference[num_tokens - 1].fill_(-1)
                if metadata[1].prefill_msa is not None:
                    reference[num_tokens - 1, :, 0] = 0
            torch.testing.assert_close(
                sharded[:num_tokens].sort(dim=-1).values,
                reference[:num_tokens].sort(dim=-1).values,
                rtol=0,
                atol=0,
            )
        assert comm.errors.item() == 0
        assert torch.count_nonzero(comm.recv[..., 32]).item() == 0
        if metadata[1].prefill_msa is None and num_layers > 1 and comm.tagged_enabled:
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
        BatchSpec(seq_lens=[2305, 2561, 10371, 2720], query_lens=[1, 1, 8193, 127]),
        BatchSpec(seq_lens=[2308, 2564, 10371, 2720], query_lens=[4, 4, 8193, 127]),
        BatchSpec(seq_lens=[2305, 2562, 2691, 2720], query_lens=[1, 2, 3, 4]),
        BatchSpec(seq_lens=[2691], query_lens=[1027]),
        BatchSpec(seq_lens=[129, 2305], query_lens=[1, 1]),
    ]
    if scenario == "offload":
        batches = batches[:1]
    for batch in batches:
        # Exercise wrap-safe rotation, single-layer ACK fallback and mixed transitions.
        common = create_common_attn_metadata(
            batch, 128, device, arange_block_indices=True
        )
        num_pages = common.block_table_tensor.numel()
        common.block_table_tensor = torch.randperm(num_pages, device=device)[
            common.block_table_tensor.long()
        ].int()
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
        num_page_bits = max(7, (num_pages - 1).bit_length())
        bits = torch.arange(num_page_bits, device=device)
        page_codes = (
            (torch.arange(num_pages, device=device)[:, None] >> bits) & 1
        ) * 64
        head_codes = torch.tensor([0, 21, 42, 85], device=device)[:, None]
        queries, caches = [], []
        for layer, (reference, sharded) in enumerate(zip(*layers)):
            # Page bits dominate the small random row contribution: each head
            # has distinct block scores, with a different ranking per head.
            cache = torch.randint(-4, 5, (num_pages, 128, 128), device=device)
            cache[:, :, :num_page_bits] = page_codes[:, None]
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
            query[:, :, :num_page_bits] = (1 << bits) * (
                1 - 2 * (((head_codes ^ (layer * 7)) >> bits) & 1)
            )
            query[PREFILL_CHUNK_SIZE:] *= -1
            queries.append(query.to(torch.float8_e4m3fn))
        if scenario == "offload":
            _roundtrip_index_caches(config, layers[1])
        if scenario == "replay":
            # Padding must be excluded even when its position is nonnegative.
            common.slot_mapping[num_tokens - 1] = -1
        metadata = [builders[0].build(0, common)]
        torch.cuda.set_sync_debug_mode("error")
        try:
            metadata.append(builders[1].build(0, common))
        finally:
            torch.cuda.set_sync_debug_mode("default")
        for md, storage in zip(metadata[1:], prefill_storage, strict=True):
            assert isinstance(md, msa.MiniMaxM3IndexerMSAShardedMetadata)
            if md.prefill_msa is not None:
                prefill = md.prefill_msa
                assert (
                    prefill.candidate_pool.untyped_storage().data_ptr(),
                    prefill.candidate_buffers.untyped_storage().data_ptr(),
                ) == storage
                assert all(arm.is_contiguous() for arm in prefill.candidate_buffers)
        expected_decode = {
            (4, 4, 8193, 127): (2, 8),
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

        if scenario != "replay":
            continue
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
                        impl.index_cache.kv_cache.copy_(cache[:, rank::world])
                    graph.replay()
                check(metadata, queries, num_tokens)
        else:
            for local_metadata in metadata[1:]:
                metadata[1] = local_metadata
                prefill = local_metadata.prefill_msa
                assert prefill.candidate_pool.dtype == torch.uint8
                chunks = prefill.local_chunks
                assert chunks[0][0] == 0
                assert chunks[-1][1] == prefill.candidate_buffers.shape[2]
                if chunks[-1][1] > PREFILL_CHUNK_SIZE:
                    assert len(chunks) > 1
                assert all(plan is not None for _, _, _, plan in chunks)
                # Reuse scratch across layers and steps.
                exchanges.clear()
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(torch.distributed, "all_to_all_single", exchange)
                    for _ in range(3):
                        run(True, metadata, queries)
                exchanged_tokens = local_metadata.num_prefill_tokens
                assert exchanges == [(world, local_heads, exchanged_tokens, 33)] * (
                    3 * num_layers
                )
                check(metadata, queries, num_tokens)
                assert prefill.graph_step is not None
                if num_layers == 5:
                    builder = builders[1]
                    graphs = builder.prefill_graphs
                    before = graphs.stats["replayed_layers"]
                    for _ in range(3):
                        for query in queries:
                            query.copy_((-query.float()).to(query.dtype))
                        metadata[1] = builder.build(0, common)
                        run(True, metadata, queries)
                        check(metadata, queries, num_tokens)
                    assert graphs.stats["replayed_layers"] > before


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="MiniMax M3 sharded MSA indexer requires SM100",
)
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("scenario", ["parity", "offload", "replay"])
def test_indexer_sharding(monkeypatch, tp_size, scenario, cake_indexer):
    """Each case exercises one CP contract through real TP indexer objects."""
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} GPUs")
    if not all(
        torch.cuda.can_device_access_peer(src, dst)
        for src in range(tp_size)
        for dst in range(tp_size)
        if src != dst
    ):
        pytest.skip("Requires peer access between all TP GPUs")

    @ray.remote(num_gpus=1, max_calls=1)
    def run(monkeypatch, tp_size, pp_size, rank, port):
        _indexer_sharding(monkeypatch, tp_size, pp_size, rank, port, scenario)

    multi_process_parallel(monkeypatch, tp_size, 1, run)


def _indexer_shards(model):
    return [
        layer.self_attn.indexer.index_cache.tp_shard for layer in model.model.layers
    ]


@pytest.fixture
def tiny_m3_checkpoint(tmp_path):
    from safetensors.torch import save_file
    from transformers import MiniMaxM3VLConfig

    config = MiniMaxM3VLConfig(
        architectures=["MiniMaxM3SparseForCausalLM"],
        text_config={
            "num_hidden_layers": 2,
            "hidden_size": 256,
            "vocab_size": 256,
            "dense_intermediate_size": 512,
            "mlp_layer_types": ["dense", "dense"],
            "layer_types": ["minimax_m3_sparse", "minimax_m3_sparse"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 5000000.0,
                "partial_rotary_factor": 0.5,
            },
        },
    )
    config.save_pretrained(tmp_path)
    cfg = config.text_config
    hidden = cfg.hidden_size
    generator = torch.Generator().manual_seed(123)
    weights = {}

    def add(name, *shape):
        weights[name + ".weight"] = (
            torch.randn(shape, generator=generator) * 0.02
        ).bfloat16()

    add("model.embed_tokens", cfg.vocab_size, hidden)
    add("lm_head", cfg.vocab_size, hidden)
    add("model.norm", hidden)
    for layer in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        for name in ("input_layernorm", "post_attention_layernorm"):
            add(f"{prefix}.{name}", hidden)
        for name, heads in (
            ("q", cfg.num_attention_heads),
            ("k", cfg.num_key_value_heads),
            ("v", cfg.num_key_value_heads),
            ("index_q", 4),
            ("index_k", 1),
        ):
            add(f"{prefix}.self_attn.{name}_proj", heads * 128, hidden)
        for name in ("q", "k", "index_q", "index_k"):
            add(f"{prefix}.self_attn.{name}_norm", 128)
        add(f"{prefix}.self_attn.o_proj", hidden, cfg.num_attention_heads * 128)
        for name in ("gate_proj", "up_proj"):
            add(f"{prefix}.mlp.{name}", cfg.dense_intermediate_size, hidden)
        add(f"{prefix}.mlp.down_proj", hidden, cfg.dense_intermediate_size)
    save_file(weights, tmp_path / "model.safetensors")
    return str(tmp_path)


def _model_options(tp_size, mode):
    return dict(
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        distributed_executor_backend="mp",
        block_size=128,
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        kv_cache_memory_bytes=128 << 20,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        compilation_config={
            "cudagraph_mode": "PIECEWISE",
            "cudagraph_capture_sizes": [1, 2, 4],
        },
        attention_config={"indexer_kv_dtype": "fp8", "indexer_cache_sharding": mode},
    )


@pytest.mark.timeout(900)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="SM100 required"
)
@pytest.mark.parametrize("tp_size", [2, 4])
def test_sharded_model_matches_replicated(
    vllm_runner, tiny_m3_checkpoint, monkeypatch, tp_size, cake_indexer
):
    """The same checkpoint must produce the same greedy tokens with CP on/off."""
    from vllm import SamplingParams

    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} GPUs")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    prompts = [
        {"prompt_token_ids": [i % 251 for i in range(length)]} for length in (129, 2305)
    ]
    results = []
    for mode in (False, True):
        with vllm_runner(tiny_m3_checkpoint, **_model_options(tp_size, mode)) as runner:
            shards = tp_size if mode else 1
            assert runner.llm.apply_model(_indexer_shards) == [[shards] * 2] * tp_size
            cohorts = []
            for _ in range(2):
                outputs = runner.llm.generate(
                    prompts,
                    SamplingParams(
                        temperature=0, max_tokens=8, ignore_eos=True, detokenize=False
                    ),
                    use_tqdm=False,
                )
                cohorts.append([out.outputs[0].token_ids for out in outputs])
            assert all(out.num_cached_tokens > 0 for out in outputs)
            assert cohorts[0] == cohorts[1]
            results.append(cohorts)
    assert results[0] == results[1]


def _dummy_preserves_caches(worker):
    model = worker.model_runner.get_model()
    caches = [
        cache
        for layer in model.model.layers
        for cache in (
            layer.self_attn.kv_cache,
            layer.self_attn.indexer.index_cache.kv_cache,
        )
    ]
    before = [cache.view(torch.uint8).clone() for cache in caches]
    worker.execute_dummy_batch()
    return all(
        torch.equal(cache.view(torch.uint8), saved)
        for cache, saved in zip(caches, before)
    )


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="SM100 required"
)
@pytest.mark.parametrize("eager", [False, True], ids=["piecewise", "eager"])
def test_dummy_batch_preserves_index_cache(
    vllm_runner, tiny_m3_checkpoint, monkeypatch, eager, cake_indexer
):
    """An idle worker's dummy batch must not overwrite cached prefix pages."""
    from vllm import SamplingParams

    if torch.accelerator.device_count() < 2:
        pytest.skip("Requires 2 GPUs")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        tiny_m3_checkpoint, **_model_options(2, True), enforce_eager=eager
    ) as runner:
        assert runner.llm.apply_model(_indexer_shards) == [[2, 2]] * 2
        prompt = [{"prompt_token_ids": [i % 251 for i in range(2305)]}]
        params = SamplingParams(
            temperature=0, max_tokens=8, ignore_eos=True, detokenize=False
        )
        reference = runner.llm.generate(prompt, params, use_tqdm=False)
        for _ in range(3):
            assert runner.llm.collective_rpc(_dummy_preserves_caches) == [True, True]
            replay = runner.llm.generate(prompt, params, use_tqdm=False)
            assert replay[0].num_cached_tokens > 0
            assert replay[0].outputs[0].token_ids == reference[0].outputs[0].token_ids
