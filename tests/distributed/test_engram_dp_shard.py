# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram tables shared by co-located DP replicas.

Four GPUs on one node split the hash heads four ways, whether as DEP
(TP1 x DP4, the single-node MoE layout) or TP2 x DP2. A lookup only returns
the right rows if the ids of every replica are gathered, each rank looks up
its own heads, and each replica trades those tokens back for the heads its
peers own.

With shared host storage, DP replicas instead map the same TP slice and
prefetch only their own tokens without DP gathers.
"""

from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import (
    EngramConfig,
    VllmConfig,
    get_current_vllm_config_or_none,
    set_current_vllm_config,
)
from vllm.config.load import LoadConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.distributed import cleanup_dist_env_and_memory, parallel_state
from vllm.distributed.parallel_state import (
    get_engram_dp_group,
    get_engram_dp_size,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.models.deepseek_v4_1.common.engram import EngramLayout
from vllm.models.deepseek_v4_1.nvidia import engram as engram_ops
from vllm.models.deepseek_v4_1.nvidia.engram import (
    Engram,
    ParallelEngramEmbedding,
    engram_head_shard_rank,
    gather_engram_hashes,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

WORLD_SIZE = 4
DIM, BLOCK = 64, 32
# Prime-sized buckets, as the model's layout builds them.
HEAD_SIZES = (97, 101, 103, 107, 109, 113, 127, 131)


def _token_counts(dp_size: int) -> tuple[tuple[int, ...], ...]:
    """Even loads, then uneven ones with an idle replica to pad around."""
    return ((8,) * dp_size, (7, 0, 5, 1)[:dp_size], (7,) * dp_size)


def _full_table(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The unsharded table, identical on every rank."""
    torch.manual_seed(0)
    weight = (torch.randn(rows, DIM) * 4).to(torch.float8_e4m3fn)
    scale_inv = torch.randint(120, 134, (rows, DIM // BLOCK), dtype=torch.uint8)
    return weight, scale_inv


def _make_ids(head_sizes: tuple[int, ...], num_tokens: int, seed: int) -> torch.Tensor:
    """Ids of each head, drawn from that head's own bucket range."""
    torch.manual_seed(seed)
    offset = 0
    columns = []
    for size in head_sizes:
        columns.append(torch.randint(offset, offset + size, (num_tokens, 1)))
        offset += size
    return torch.cat(columns, dim=1).to(torch.int32).cuda()


def _reference(weight, scale_inv, ids) -> torch.Tensor:
    """Dequantized lookup against the whole table."""
    valid = ids >= 0
    safe_ids = ids.clamp_min(0).long()
    values = torch.nn.functional.embedding(safe_ids, weight).float()
    scales = torch.nn.functional.embedding(safe_ids, scale_inv)
    scales = (scales.to(torch.int32) << 23).view(torch.float32)
    values = values.unflatten(-1, (-1, BLOCK)) * scales.unsqueeze(-1)
    return values.flatten(-2).to(torch.bfloat16).masked_fill(~valid.unsqueeze(-1), 0)


@pytest.mark.parametrize(
    "architecture,options,edp_ranks,etp_ranks",
    [
        ("DeepseekV41ForCausalLM", {}, [0, 2], [0, 1]),
        (
            "DeepseekV41ForCausalLM",
            {"enable_engram_dp_sharding": False},
            None,
            [0, 1],
        ),
        (
            "DeepseekV41ForCausalLM",
            {
                "enable_engram_dp_sharding": False,
                "enable_engram_dp_shared_memory": True,
            },
            [0, 2],
            [0, 1],
        ),
        ("Qwen4ExpForCausalLM", {}, None, [0, 1]),
        (
            "Qwen4ExpForCausalLM",
            {"embedding_across_dp": True},
            None,
            [0, 1, 2, 3],
        ),
        ("Qwen4ExpForConditionalGeneration", {}, None, [0, 1]),
    ],
)
def test_engram_group_creation_honors_model_and_options(
    monkeypatch, architecture, options, edp_ranks, etp_ranks
):
    """Disabling DS sharding removes EDP unless storage sharing needs it;
    Qwen uses ETP alone. Exercise initialization without GPU communicators.
    """
    config = SimpleNamespace(
        model_config=SimpleNamespace(architecture=architecture, is_moe=False),
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            data_parallel_size=2,
            distributed_executor_backend="mp",
        ),
        engram_config=EngramConfig(**options),
    )
    for name in (
        "_TP",
        "_ETP",
        "_ENGRAM_DP",
        "_DCP",
        "_PCP",
        "_PP",
        "_DP",
        "_EP",
        "_EPLB",
    ):
        monkeypatch.setattr(parallel_state, name, None)
    monkeypatch.setattr(parallel_state, "_WORLD", SimpleNamespace(local_rank=0))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_node_count", lambda: 1)

    def make_group(group_ranks, *args, **kwargs):
        ranks = next(ranks for ranks in group_ranks if 0 in ranks)
        return SimpleNamespace(ranks=ranks, rank_in_group=0, world_size=len(ranks))

    monkeypatch.setattr(parallel_state, "init_model_parallel_group", make_group)
    with set_current_vllm_config(config):
        initialize_model_parallel(tensor_model_parallel_size=2, backend="nccl")
    group = get_engram_dp_group()
    assert (group.ranks if group is not None else None) == edp_ranks
    assert parallel_state.get_etp_group().ranks == etp_ranks


def _worker(rank: int, tp_size: int, port: int) -> None:
    torch.accelerator.set_device_index(rank)
    update_environment_variables(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(WORLD_SIZE),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
    )
    dp_size = WORLD_SIZE // tp_size
    dp_rank, tp_rank = divmod(rank, tp_size)
    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=8,
            is_encoder_decoder=False,
            max_num_batched_tokens=8,
            max_num_seqs=8,
        ),
    )
    # Supply the resolved Engram settings without constructing a full model.
    vllm_config.engram_config = EngramConfig()
    vllm_config.model_config = SimpleNamespace(
        architecture="DeepseekV41ForCausalLM", is_moe=True
    )
    try:
        init_distributed_environment()
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(tensor_model_parallel_size=tp_size)
        assert get_engram_dp_size() == dp_size
        assert get_engram_dp_group().ranks == [
            tp_rank + replica * tp_size for replica in range(dp_size)
        ]
        assert engram_head_shard_rank() == tp_rank * dp_size + dp_rank

        for failure in ("create", "open", "mmap", "register", "pinned", None):
            _check_shared_storage_lifetime_and_failures(failure)

        # Reuse the processes and communication groups across storage modes.
        for (cpu_offload, dp_shared_memory), n_heads, sequence_parallel in product(
            [(False, False), (True, False), (True, True)],
            [8, 7],
            [False, True] if tp_size > 1 else [False],
        ):
            _check_table(
                vllm_config,
                cpu_offload,
                dp_shared_memory,
                n_heads,
                sequence_parallel,
            )
    finally:
        cleanup_dist_env_and_memory()


def _check_shared_storage_lifetime_and_failures(failure):
    """One peer's failure must roll back every mapping/registration, without hangs."""
    import gc
    import weakref

    group = get_engram_dp_group()
    runtime = torch.cuda.cudart()
    make_file = engram_ops.tempfile.NamedTemporaryFile
    map_file = engram_ops.mmap.mmap
    paths, mappings, registrations, unregistrations = [], [], [], []
    failed_rank = 0 if failure == "create" else 1

    def fail():
        raise OSError(f"injected {failure} failure")

    def temporary_file(*args, **kwargs):
        if failure == "create":
            fail()
        file = make_file(*args, **kwargs)
        paths.append(Path(file.name))
        return file

    def open_file(*args, **kwargs):
        if failure == "open" and group.rank_in_group == failed_rank:
            fail()
        return open(*args, **kwargs)

    def mapping(*args, **kwargs):
        if failure == "mmap" and group.rank_in_group == failed_rank:
            fail()
        result = map_file(*args, **kwargs)
        mappings.append(weakref.ref(result))
        return result

    def register(pointer, size, flags):
        if failure == "register" and group.rank_in_group == failed_rank:
            return SimpleNamespace(value=1)
        result = runtime.cudaHostRegister(pointer, size, flags)
        assert result.value == 0
        registrations.append(pointer)
        return result

    def unregister(pointer):
        unregistrations.append(pointer)
        result = runtime.cudaHostUnregister(pointer)
        assert result.value == 0
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            engram_ops,
            "tempfile",
            SimpleNamespace(NamedTemporaryFile=temporary_file),
        )
        patch.setattr(engram_ops, "open", open_file, raising=False)
        patch.setattr(
            engram_ops,
            "mmap",
            SimpleNamespace(mmap=mapping, MAP_SHARED=engram_ops.mmap.MAP_SHARED),
        )
        patch.setattr(
            torch.cuda,
            "cudart",
            lambda: SimpleNamespace(
                cudaHostRegister=register,
                cudaHostUnregister=unregister,
            ),
        )
        if failure == "pinned" and group.rank_in_group == failed_rank:
            patch.setattr(torch.Tensor, "is_pinned", lambda self: False)
        if failure is not None:
            stage = "cudaHostRegister" if failure in ("register", "pinned") else failure
            with pytest.raises(RuntimeError, match=f"EDP rank {failed_rank}.*{stage}"):
                engram_ops.DPSharedEngramStorage(128, DIM, BLOCK, group)
            assert unregistrations == registrations
        else:
            storage = engram_ops.DPSharedEngramStorage(128, DIM, BLOCK, group)
            assert all(not path.exists() for path in paths)
            storage_ref = weakref.ref(storage)
            parameter = torch.nn.Parameter(storage.weight, requires_grad=False)
            views = storage.get_views(parameter, storage.weight_scale_inv)
            del storage
            gc.collect()
            assert storage_ref() is None
            assert not unregistrations
            del parameter
            gc.collect()
            assert not unregistrations  # UVA views still retain the CPU allocation.
            del views
        gc.collect()
        assert unregistrations == registrations
        for ref in mappings:
            mapped = ref()
            assert mapped is None or mapped.closed
        assert all(not path.exists() for path in paths)


def _check_table(
    vllm_config, cpu_offload, dp_shared_memory, n_heads, sequence_parallel
):
    parallel = vllm_config.parallel_config
    dp_size, tp_size = parallel.data_parallel_size, parallel.tensor_parallel_size
    dp_rank = parallel.data_parallel_rank
    tp_rank = engram_ops.get_tensor_model_parallel_rank()
    multithread = (
        dp_shared_memory and n_heads == 7 and sequence_parallel == (tp_size > 1)
    )
    load_config = (
        LoadConfig(
            load_format="safetensors",
            use_tqdm_on_load=False,
            model_loader_extra_config={
                "enable_multithread_load": True,
                "num_threads": 2,
            },
        )
        if multithread
        else vllm_config.load_config
    )
    with pytest.MonkeyPatch.context() as m:
        head_sizes = HEAD_SIZES[:n_heads]
        rows = sum(head_sizes)
        config = SimpleNamespace(
            hidden_size=DIM,
            hc_mult=2,
            rms_norm_eps=1e-6,
            engram_layer_ids=[0],
            engram_num_embeddings=[rows],
            engram_max_ngram_size=2,
            engram_n_heads=n_heads,
            engram_head_dim=DIM,
            engram_compressed_vocab_size=32,
            engram_pad_token_id=0,
            engram_vocab_size=HEAD_SIZES[0],
        )
        with set_current_vllm_config(vllm_config):
            weight, scale_inv = _full_table(rows)
            layout = EngramLayout(config)
            # Stub configuration inputs, not Engram's initialization or execution.
            component_config = SimpleNamespace(
                engram_config=EngramConfig(
                    cpu_offload=cpu_offload,
                    enable_engram_dp_shared_memory=dp_shared_memory,
                ),
                scheduler_config=vllm_config.scheduler_config,
                load_config=load_config,
            )
            with m.context() as init_patch, torch.device("cuda"):
                init_patch.setattr(
                    engram_ops, "get_current_vllm_config", lambda: component_config
                )
                engram = Engram(
                    config,
                    quant_config=None,
                    layout=layout,
                    layer_hash_index=0,
                    use_sequence_parallel=sequence_parallel,
                    prefix="model.layers.0.engram",
                )
            layer = engram.embed_tokens
            if multithread:
                _load_shared_multithread(layer, weight, scale_inv, load_config)
            else:
                # Only the leader has valid checkpoint payload in shared mode.
                loader_weight = weight if not dp_shared_memory or dp_rank == 0 else None
                loader_scale = (
                    scale_inv if not dp_shared_memory or dp_rank == 0 else None
                )
                layer.weight.weight_loader(layer.weight, loader_weight)
                layer.weight_scale_inv.weight_loader(
                    layer.weight_scale_inv, loader_scale
                )
            if dp_shared_memory:
                assert layer.dp_size == 1
                assert layer.head_start == tp_rank * layer.part_n_hash_cols
                assert layer.weight.is_pinned() and layer.weight_scale_inv.is_pinned()

                def unexpected_collective(*args, **kwargs):
                    raise AssertionError("shared lookup must not use DP collectives")

                m.setattr(get_engram_dp_group(), "all_gather", unexpected_collective)

            token_counts = _token_counts(dp_size)

        # Production forward runs after the construction-time config has exited.
        assert get_current_vllm_config_or_none() is None
        for batch_idx, counts in enumerate(token_counts):
            num_tokens = counts[dp_rank]
            # Ids differ per replica but must agree across its TP ranks.
            ids = _make_ids(head_sizes, num_tokens, seed=100 + dp_rank)
            if batch_idx == 2 and dp_rank == 1:
                ids.fill_(engram_ops.DEAD_ID)
            with set_forward_context(
                None,
                vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=torch.tensor(counts, dtype=torch.int32),
            ):
                gathered = gather_engram_hashes(
                    ids, dp_shared_memory=layer.dp_shared_memory
                )
                expected_tokens = (
                    num_tokens if dp_shared_memory else max(counts) * dp_size
                )
                assert gathered.shape[0] == expected_tokens
                engram.prepare_embeddings(gathered)
                looked_up = engram.embed(ids)
                standalone = layer(ids)
            expected = _reference(weight.cuda(), scale_inv.cuda(), ids)
            torch.testing.assert_close(standalone, expected, atol=0, rtol=0)
            if sequence_parallel:
                # SP hands back only this rank's token shard, zero-padded.
                chunk = -(-num_tokens // tp_size)
                expected = torch.nn.functional.pad(
                    expected, (0, 0, 0, 0, 0, chunk * tp_size - num_tokens)
                )
                expected = expected[tp_rank * chunk : (tp_rank + 1) * chunk]
            torch.testing.assert_close(looked_up, expected, atol=0, rtol=0)

        if (
            cpu_offload
            and not dp_shared_memory
            and n_heads == 7
            and not sequence_parallel
        ):
            _check_dummy_hash_model_forward(
                vllm_config, engram, head_sizes, weight, scale_inv
            )

        if dp_shared_memory and n_heads == 7 and sequence_parallel == (tp_size > 1):
            _check_shared_prefetch_replay(engram, head_sizes, weight, scale_inv)
            with m.context() as storage_patch:
                storage_patch.setattr(layer.weight, "data", layer.weight.data.clone())
                with pytest.raises(RuntimeError, match="storage must not be replaced"):
                    layer(ids)


def _check_dummy_hash_model_forward(vllm_config, engram, head_sizes, weight, scales):
    """A replica without attention metadata must join its peers' real DP lookups."""
    from vllm.models.deepseek_v4_1.common.engram import NgramHashState
    from vllm.models.deepseek_v4_1.nvidia import model as model_ops

    dp_rank = vllm_config.parallel_config.data_parallel_rank
    tokens = 4
    ids = _make_ids(head_sizes, tokens, seed=300 + dp_rank)
    state = NgramHashState.__new__(NgramHashState)
    torch.nn.Module.__init__(state)
    state.multipliers = torch.empty(1, 2, dtype=torch.int64, device="cuda")
    state.primes = torch.empty(1, 1, len(head_sizes), dtype=torch.int64, device="cuda")
    state.ensure_cache = lambda: True
    # Hash arithmetic is covered separately; keep the real model's branch and
    # NgramHashState.dummy_hashes, plus real preparation/lookup collectives.
    state.forward = lambda *args: ids.unsqueeze(1)

    class Decoder(SimpleNamespace):
        def __call__(self, hidden, positions, input_ids, *args):
            hashes, keep = args[-2:]
            assert hashes is not None and keep is not None
            assert bool(keep.all()) == (dp_rank != 1)
            if dp_rank == 1:
                assert torch.all(hashes == engram_ops.DEAD_ID)
            return engram.embed(hashes[:, 0]), None, None, None, None

    model = SimpleNamespace(
        use_mega_moe=False,
        use_sequence_parallel=False,
        engram_hash=state,
        engram_swa_prefix="swa",
        engram_dp_shared_memory=False,
        layers=[Decoder(engram=engram)],
        start_layer=0,
        end_layer=1,
        aux_hidden_state_layers=(),
    )
    metadata = (
        None
        if dp_rank == 1
        else {
            "swa": SimpleNamespace(
                query_start_loc=None, slot_mapping=None, block_table=None
            )
        }
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            model_ops,
            "get_pp_group",
            lambda: SimpleNamespace(
                is_first_rank=True,
                is_last_rank=False,
            ),
        )
        patch.setattr(model_ops, "mhc_post_tilelang", lambda hidden, *args: hidden)
        with set_forward_context(
            metadata,
            vllm_config,
            num_tokens=tokens,
            num_tokens_across_dp=torch.full(
                (vllm_config.parallel_config.data_parallel_size,),
                tokens,
                dtype=torch.int32,
            ),
        ):
            output = model_ops.DeepseekV4Model.forward(
                model,
                torch.arange(tokens, device="cuda"),
                torch.arange(tokens, device="cuda"),
                intermediate_tensors=None,
                inputs_embeds=torch.zeros(tokens, DIM, device="cuda"),
                lookback_token_ids=torch.full((1, 1), -1, device="cuda"),
            )["hidden_states"]
    expected_ids = torch.full_like(ids, engram_ops.DEAD_ID) if dp_rank == 1 else ids
    expected = _reference(weight.cuda(), scales.cuda(), expected_ids)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def _load_shared_multithread(layer, weight, scale_inv, load_config):
    """Opposite shard completion orders must leave every shared parameter ready."""
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader import weight_utils
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    tensors = {
        "weight": weight,
        "weight_scale_inv": scale_inv.view(torch.float8_e8m0fnu),
    }
    order = list(tensors)
    if get_engram_dp_group().rank_in_group % 2:
        order.reverse()
    first_loaded = Event()
    original_load = weight_utils.load_file

    def load_file(path, **kwargs):
        if Path(path).stem != order[0]:
            assert first_loaded.wait(timeout=30), "First shared weight load stalled"
        return original_load(path, **kwargs)

    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as m:
        for name, tensor in tensors.items():
            save_file({name: tensor}, str(Path(directory) / f"{name}.safetensors"))
        m.setattr(weight_utils, "load_file", load_file)
        loader = DefaultModelLoader(load_config)
        source = DefaultModelLoader.Source(directory, revision=None)
        loaded = []
        for name, tensor in loader._get_weights_iterator(source):
            param = getattr(layer, name)
            param.weight_loader(param, tensor)
            loaded.append(name)
            first_loaded.set()
        assert loaded == order

    # Check before any further collective could mask incomplete shared writes.
    for name, tensor in tensors.items():
        param = getattr(layer, name)
        expected = tensor.narrow(0, param.engram_vocab_start, param.shape[0])
        torch.testing.assert_close(
            param.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
        )


def _check_shared_prefetch_replay(engram, head_sizes, weight, scale_inv):
    """Shared host lookups must read new IDs on each breakable graph replay."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    ids = _make_ids(head_sizes, 8, seed=501)
    tp_rank = engram_ops.get_tensor_model_parallel_rank()
    tokens = 8 // engram.embed_tokens.tp_size if engram.use_sequence_parallel else 8
    output = torch.empty(
        tokens, len(head_sizes), DIM, device="cuda", dtype=torch.bfloat16
    )

    def step():
        engram.prepare_embeddings(ids.clone())
        output.copy_(engram.embed(ids))

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        step()
    graph = BreakableCUDAGraphCapture()
    with torch.cuda.stream(warmup), graph:
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    assert graph.num_eager_breaks == 2
    for iteration in range(2):
        ids.copy_(_make_ids(head_sizes, 8, seed=600 + iteration))
        graph.replay()
        expected = _reference(weight.cuda(), scale_inv.cuda(), ids)
        if engram.use_sequence_parallel:
            expected = expected[tp_rank * tokens : (tp_rank + 1) * tokens]
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.accelerator.synchronize()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("tp_size", [1, 2])
def test_engram_table_shared_across_dp_replicas(tp_size, monkeypatch):
    """Check all storage modes with one process launch per TP/DP topology."""
    if torch.accelerator.device_count() < WORLD_SIZE:
        pytest.skip(f"Need {WORLD_SIZE} GPUs to run the test.")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    mp.spawn(_worker, args=(tp_size, get_open_port()), nprocs=WORLD_SIZE)
    cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "cpu_offload,dp_size,error",
    [
        (False, 4, "requires cpu_offload=True"),
        (True, 1, "effective Engram DP size is 1"),
    ],
)
def test_engram_dp_shared_memory_runtime_requirements(
    monkeypatch, cpu_offload, dp_size, error
):
    """Report unmet storage/topology requirements before allocating CUDA memory."""
    monkeypatch.setattr(engram_ops, "get_engram_dp_size", lambda: dp_size)
    with pytest.raises(ValueError, match=error):
        ParallelEngramEmbedding(
            sum(HEAD_SIZES),
            DIM,
            HEAD_SIZES,
            cpu_offload=cpu_offload,
            dp_shared_memory=True,
        )


@pytest.mark.parametrize(
    "node_ids,dp_size,replica_size,expected",
    [
        ([0] * 4, 1, 4, 1),
        ([0] * 4, 4, 1, 4),
        ([0] * 4, 2, 2, 2),
        ([0] * 8 + [1] * 8, 8, 2, 4),
        ([0] * 6 + [1] * 6, 4, 1, 2),
        ([0] * 3 + [1] * 4, 7, 1, 1),
        ([0] * 8 + [1] * 8 + [2] * 8, 8, 3, 1),
        ([0, 1] * 4, 4, 2, 1),
        ([0] * 3 + [1] * 5, 4, 2, 1),
    ],
)
def test_engram_dp_shard_size_respects_node_boundaries(
    node_ids, dp_size, replica_size, expected, monkeypatch
):
    """Sharing must fall back when replicas or node rank ranges do not align."""
    monkeypatch.setattr(parallel_state, "get_node_count", lambda: len(set(node_ids)))
    monkeypatch.setattr(
        parallel_state, "get_world_group", lambda: SimpleNamespace(cpu_group=None)
    )
    monkeypatch.setattr(
        parallel_state,
        "in_the_same_node_as",
        lambda group, src: [node == node_ids[src] for node in node_ids],
    )
    assert (
        parallel_state._engram_dp_shard_size(len(node_ids), dp_size, replica_size)
        == expected
    )


@pytest.mark.parametrize("num_tokens", [0, 2, 4, 5])
def test_engram_hash_padding_has_no_valid_rows(num_tokens, monkeypatch):
    """Idle/padded slots must not query row zero; oversized batches are rejected."""
    monkeypatch.setattr(engram_ops, "engram_gathered_num_tokens", lambda: 4)
    monkeypatch.setattr(
        engram_ops,
        "get_engram_dp_group",
        lambda: SimpleNamespace(all_gather=lambda ids, dim: torch.cat([ids, ids], dim)),
    )
    ids = torch.full((num_tokens, 2, 3), 17, dtype=torch.int32)
    if num_tokens > 4:
        with pytest.raises(ValueError, match="exceeds the DP token slot"):
            gather_engram_hashes(ids)
        return
    gathered = gather_engram_hashes(ids).reshape(2, 4, 2, 3)
    for replica in gathered:
        torch.testing.assert_close(replica[:num_tokens], ids)
        assert torch.all(replica[num_tokens:] == engram_ops.DEAD_ID)


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("uniform", [False, True])
def test_engram_hash_padding_ignores_other_nodes(tp_size, uniform, monkeypatch):
    """Two four-GPU nodes pad only to their own maximum, including TP x DP."""
    group_size = 4 // tp_size
    counts = (
        [4] * (8 // tp_size)
        if uniform
        else [4096] + [2] * (group_size - 1) + [1, 3] + [0] * (group_size - 2)
    )
    monkeypatch.setattr(
        engram_ops,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(counts))
        ),
    )
    for dp_rank in range(8 // tp_size):
        slot = 4 if uniform else (4096 if dp_rank < group_size else 3)
        ids = torch.full((counts[dp_rank], 2, 3), 17, dtype=torch.int32)
        monkeypatch.setattr(
            engram_ops,
            "get_dp_group",
            lambda rank=dp_rank: SimpleNamespace(rank_in_group=rank),
        )
        monkeypatch.setattr(
            engram_ops,
            "get_engram_dp_group",
            lambda rank=dp_rank: SimpleNamespace(
                rank_in_group=rank % group_size,
                world_size=group_size,
                all_gather=lambda tensor, dim: torch.cat([tensor] * group_size, dim),
            ),
        )
        gathered = gather_engram_hashes(ids)
        expected = ids.new_full((slot, 2, 3), engram_ops.DEAD_ID)
        expected[: ids.shape[0]] = ids
        torch.testing.assert_close(
            gathered, expected.repeat(group_size, 1, 1), msg=f"DP rank {dp_rank}"
        )
