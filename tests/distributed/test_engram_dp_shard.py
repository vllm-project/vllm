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

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

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
from vllm.distributed.engram_pool import EngramPool, rdma, verbs
from vllm.distributed.engram_pool.group import PeerGroup
from vllm.distributed.parallel_state import (
    get_engram_dp_group,
    get_engram_dp_size,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    convert_swizzled_to_linear,
)
from vllm.models.deepseek_v41.common import engram as engram_ops
from vllm.models.deepseek_v41.common.engram import (
    Engram,
    EngramBatch,
    EngramLayout,
    ParallelEngramEmbedding,
    engram_head_shard_rank,
    gather_engram_hashes,
)
from vllm.models.qwen4_exp.common import ngram_embedding as ple_ops
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.vmm_driver import imex_channel_available

WORLD_SIZE = 4
DIM, BLOCK = 64, 32
# Prime-sized buckets, as the model's layout builds them.
HEAD_SIZES = (97, 101, 103, 107, 109, 113, 127, 131)


def _token_counts(dp_size: int) -> tuple[tuple[int, ...], ...]:
    """Even loads, then uneven ones with an idle replica to pad around."""
    return ((8,) * dp_size, (7, 0, 5, 1)[:dp_size], (7,) * dp_size)


def _full_table(
    rows: int, packed_fp4: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """The unsharded table, identical on every rank."""
    torch.manual_seed(0)
    weight = (
        torch.randint(0, 256, (rows, DIM // 2), dtype=torch.uint8)
        if packed_fp4
        else (torch.randn(rows, DIM) * 4).to(torch.float8_e4m3fn)
    )
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
    values = torch.nn.functional.embedding(safe_ids, weight)
    if weight.dtype == torch.uint8:
        nibbles = torch.stack((values & 0xF, values >> 4), dim=-1).flatten(-2).long()
        lut = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device
        )
        values = torch.where(nibbles & 8 != 0, -1.0, 1.0) * lut[nibbles & 7]
    values = values.float()
    scales = torch.nn.functional.embedding(safe_ids, scale_inv)
    scales = (scales.to(torch.int32) << 23).view(torch.float32)
    values = values.unflatten(-1, (-1, BLOCK)) * scales.unsqueeze(-1)
    return values.flatten(-2).to(torch.bfloat16).masked_fill(~valid.unsqueeze(-1), 0)


@pytest.mark.parametrize(
    "architecture,options,tp_size,edp_ranks,etp_ranks",
    [
        ("DeepseekV41ForCausalLM", {}, 2, [0, 2], [0, 1]),
        # GPU tables span every DP rank only with embedding_across_dp.
        ("DeepseekV41ForCausalLM", {"cpu_offload": False}, 1, [0, 1], [0]),
        (
            "DeepseekV41ForCausalLM",
            {"cpu_offload": False, "embedding_across_dp": True},
            1,
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ),
        ("Qwen4ExpForCausalLM", {}, 2, None, [0, 1]),
        # The offloaded Qwen4Exp PLE table is only shared over the group, so it
        # exists with dp_shared_memory unless the table is spread over DP.
        ("Qwen4ExpForCausalLM", {"dp_shared_memory": True}, 2, [0, 2], [0, 1]),
        (
            "Qwen4ExpForCausalLM",
            {"dp_shared_memory": True, "embedding_across_dp": True},
            2,
            None,
            [0, 1, 2, 3],
        ),
        (
            "Qwen4ExpForCausalLM",
            {"embedding_across_dp": True},
            2,
            None,
            [0, 1, 2, 3],
        ),
        ("Qwen4ExpForConditionalGeneration", {}, 2, None, [0, 1]),
        # HY4-Flash shares its Engram tables over the group only when asked.
        ("HYV4FlashForCausalLM", {}, 2, None, [0, 1]),
        ("HYV4FlashForCausalLM", {"dp_shared_memory": True}, 2, [0, 2], [0, 1]),
    ],
)
def test_engram_group_creation_honors_model_and_options(
    monkeypatch, architecture, options, tp_size, edp_ranks, etp_ranks
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(architecture=architecture, is_moe=False),
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp_size,
            data_parallel_size=4 // tp_size,
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
    # Two ranks per node.
    monkeypatch.setattr(parallel_state, "_engram_dp_shard_size", lambda *args: 2)

    def make_group(group_ranks, *args, **kwargs):
        ranks = next(ranks for ranks in group_ranks if 0 in ranks)
        return SimpleNamespace(ranks=ranks, rank_in_group=0, world_size=len(ranks))

    monkeypatch.setattr(parallel_state, "init_model_parallel_group", make_group)
    with set_current_vllm_config(config):
        initialize_model_parallel(tensor_model_parallel_size=tp_size, backend="nccl")
    group = get_engram_dp_group()
    assert (group.ranks if group is not None else None) == edp_ranks
    assert parallel_state.get_etp_group().ranks == etp_ranks
    # A group with the DP group's ranks reuses its communicators.
    full_dp = edp_ranks is not None and len(edp_ranks) == 4 // tp_size
    assert (group is parallel_state._DP) == full_dp


def _worker(rank: int, tp_size: int, port: int, store_dir: str) -> None:
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
        architecture="DeepseekV41ForCausalLM",
        is_moe=True,
        sleep_mode_offload_cudagraph=False,
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
        for failure, packed_fp4 in product((None, "create", "import"), (False, True)):
            _check_cumem_host_table(failure, packed_fp4)
        for shm_fits in (True, False):
            _check_ple_shared_table(shm_fits)
        _check_flash_device_local_table()
        if current_platform.is_cuda():
            # The trailing True repeat re-initializes shared tables.
            for shm_fits in (True, False, True):
                _check_flash_shared_table(shm_fits)
            _check_flash_shared_load_order()
        if tp_size == 1:
            _check_pool_tables(store_dir)

        # GPU peer reads need NVLink P2P, which CI's L4s lack.
        p2p = current_platform.is_fully_connected(
            [
                current_platform.visible_device_id_to_physical_device_id(i)
                for i in range(WORLD_SIZE)
            ]
        )
        # Reuse the processes and communication groups across storage modes.
        for (
            cpu_offload,
            dp_shared_memory,
        ), n_heads, sequence_parallel, packed_fp4 in product(
            [(False, False), (True, False), (True, True)],
            [8, 7],
            [False, True] if tp_size > 1 else [False],
            [False, True],
        ):
            _check_table(
                vllm_config,
                cpu_offload,
                dp_shared_memory,
                n_heads,
                sequence_parallel,
                p2p,
                packed_fp4,
            )
        if tp_size == 1 and p2p:
            # Without NVLink P2P, GPU tables keep the DP collectives.
            for packed_fp4 in (False, True):
                _check_table(
                    vllm_config,
                    False,
                    False,
                    8,
                    False,
                    p2p=False,
                    packed_fp4=packed_fp4,
                )
        if tp_size == 1:
            _check_fp8_batch(vllm_config)
    finally:
        cleanup_dist_env_and_memory()


def _check_pool_tables(store_dir: str):
    """GPUs of independent engines split one table in an Engram pool and read
    each other's rows, over every transport this machine has."""
    modes = []
    if current_platform.is_cuda() and imex_channel_available():
        from cuda.bindings import driver as cu

        modes.append(("nvlink", False, 4))
        # Every rank must take the same collective path in a mixed GPU group.
        attrs = cu.CUdevice_attribute
        host_vmm = (
            attrs.CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
        )
        if all(
            cu.cuDeviceGetAttribute(host_vmm, device) == (cu.CUresult.CUDA_SUCCESS, 1)
            for device in range(WORLD_SIZE)
        ):
            modes.append(("nvlink", True, 4))
    if current_platform.is_cuda() and verbs.active_devices():
        # GPUs of a node map each other's host shards; nodes of one use RDMA.
        modes += [("rdma", False, 4), ("rdma", True, 4), ("rdma", True, 1)]
    weight, scale = _full_table(sum(HEAD_SIZES))
    from vllm.models.deepseek_v41.nvidia.ops.engram_pool import PoolEngramEmbedding

    for transport, host, node in modes:
        group = PeerGroup(WORLD_SIZE, f"{store_dir}/{transport}{host}{node}/store")
        # Engines batch differently (P/D): owners size each requester's mailbox.
        max_items = 9 * len(HEAD_SIZES) * (1 + group.rank_in_group % 2)
        # node == 1: each GPU as its own machine, so rows move over RDMA.
        machine = str(group.rank_in_group) if node == 1 else rdma.node_id()
        with patch.object(rdma, "node_id", lambda machine=machine: machine):
            pool = EngramPool(group, transport, host, max_items, 72)
        # Independent engines can own one GPU at a nonzero CUDA ordinal.
        device = torch.accelerator.current_device_index()
        assigned = [current_platform.visible_device_id_to_physical_device_id(device)]
        with (
            torch.device("cuda"),
            patch(
                "vllm.platforms.interface._assigned_physical_gpu_ids",
                assigned if transport == "nvlink" else None,
            ),
        ):
            table = PoolEngramEmbedding(
                sum(HEAD_SIZES), DIM, HEAD_SIZES, False, pool, "table"
            )
        assert table.dp_size == 1
        if transport == "nvlink" and host:
            error, handle = cu.cuMemRetainAllocationHandle(table.weight.data_ptr())
            assert error == cu.CUresult.CUDA_SUCCESS
            try:
                error, prop = cu.cuMemGetAllocationPropertiesFromHandle(handle)
                assert error == cu.CUresult.CUDA_SUCCESS
                assert (
                    prop.location.type
                    == cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
                )
                assert prop.location.id == current_platform.get_device_numa_node(
                    torch.accelerator.current_device_index()
                )
            finally:
                cu.cuMemRelease(handle)
            _check_nvlink_startup_rollback(pool)
        table.weight.weight_loader(table.weight, weight)
        table.weight_scale_inv.weight_loader(table.weight_scale_inv, scale)
        table.process_weights_after_loading()
        ids = _make_ids(HEAD_SIZES, 9, seed=500 + group.rank_in_group)
        # A first fetch with no remote rows (profile runs) must not fail.
        dead = torch.full_like(ids, -1)
        torch.testing.assert_close(
            table(dead), _reference(weight.cuda(), scale.cuda(), dead), atol=0, rtol=0
        )
        ids[0, 0] = -1
        expected = _reference(weight.cuda(), scale.cuda(), ids)
        torch.testing.assert_close(table(ids), expected, atol=0, rtol=0)
        # Replays carry no host state: each one fetches the ids now in `static`.
        static, stream = ids.clone(), torch.cuda.Stream()
        with torch.cuda.stream(stream):
            table(static)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = table(static)
        for seed in range(3):
            static.copy_(
                _make_ids(HEAD_SIZES, 9, seed=600 + 8 * seed + group.rank_in_group)
            )
            graph.replay()
            expected = _reference(weight.cuda(), scale.cuda(), static)
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
        del graph
        group.barrier()
        pool.close()


def _check_nvlink_startup_rollback(pool: EngramPool):
    """A failed rank must leave no allocation handles or address reservations."""
    from cuda.bindings import driver as cu

    for failure in (
        "cuDeviceGetAttribute",
        "cuMemCreate",
        "cuMemExportToShareableHandle",
        "cuMemImportFromShareableHandle",
        "cuMemMap",
        "cuMemSetAccess",
    ):
        with ExitStack() as stack:
            calls = {
                name: stack.enter_context(
                    patch.object(cu, name, wraps=getattr(cu, name))
                )
                for name in (
                    "cuMemCreate",
                    "cuMemImportFromShareableHandle",
                    "cuMemRelease",
                    "cuMemAddressReserve",
                    "cuMemAddressFree",
                )
            }
            if pool.group.rank_in_group == 0:
                stack.enter_context(
                    patch.object(cu, failure, side_effect=MemoryError(failure))
                )
            with pytest.raises(RuntimeError, match=f"rank 0: MemoryError: {failure}"):
                pool.table(32, 8, "rollback")

        assert (
            calls["cuMemAddressFree"].call_count
            == calls["cuMemAddressReserve"].call_count
        )
        assert calls["cuMemRelease"].call_count == (
            calls["cuMemCreate"].call_count
            + calls["cuMemImportFromShareableHandle"].call_count
        )


def test_engram_peer_group_propagates_startup_failure(tmp_path):
    path = str(tmp_path / "peers")
    groups = [PeerGroup(2, path) for _ in range(2)]

    def run(group):
        with (
            pytest.raises(RuntimeError, match="rank 1: IndexError: device"),
            group.startup_stage("Allocating table"),
        ):
            if group.rank_in_group == 1:
                raise IndexError("device")

    with ThreadPoolExecutor(2) as executor:
        list(executor.map(run, groups))


def test_engram_peer_group_assigns_ranks_and_gathers(tmp_path):
    """Engines share one config, so ranks must not collide however they race,
    a per-job store directory need not exist, and a reused store is rejected
    rather than joined."""
    path = str(tmp_path / "job" / "peers")
    with ThreadPoolExecutor(4) as pool:
        groups = list(pool.map(lambda _: PeerGroup(4, path), range(4)))
        gathered = list(
            pool.map(
                lambda group: group.all_gather(b"%d" % group.rank_in_group),
                groups,
            )
        )
    assert sorted(group.rank_in_group for group in groups) == [0, 1, 2, 3]
    assert gathered == [[b"0", b"1", b"2", b"3"]] * 4
    with pytest.raises(RuntimeError, match="fresh store path"):
        PeerGroup(4, path)


def test_engram_peer_group_names_missing_gpus(tmp_path, monkeypatch):
    """A peer that never arrives fails startup with its rank, not a bare timeout."""
    monkeypatch.setenv("VLLM_ENGINE_READY_TIMEOUT_S", "1")
    group = PeerGroup(2, str(tmp_path / "peers"))
    with pytest.raises(TimeoutError, match=r"GPUs \[1\] did not reach startup step 1"):
        group.barrier()


def test_engram_pool_table_rejects_reload(monkeypatch):
    """Other engines read a table's rows at any time, so it cannot reload."""
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    pool = EngramPool.__new__(EngramPool)
    pool.group = SimpleNamespace(barrier=lambda: None)
    table = SimpleNamespace(_loaded=False)
    pool.ready(table)
    with pytest.raises(RuntimeError, match="cannot reload weights"):
        pool.ready(table)


def test_engram_pool_proxy_stops_on_failed_completion():
    """A failed RDMA completion stops the proxy instead of being skipped, so
    requesters see the failure rather than wait for the GPU timeout."""
    from vllm.distributed.engram_pool import proxy

    wc = (verbs.ibv_wc * 2)()
    proxy._check(wc, 2, "gather")
    wc[1].status, wc[1].qp_num = 5, 77
    with pytest.raises(RuntimeError, match="gather failed on QP 77"):
        proxy._check(wc, 2, "gather")


def _check_fp8_batch(vllm_config):
    """MXFP8 staging matches the tables across peer/all-to-all slots, idle
    replicas and graph replays."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    dp_rank = vllm_config.parallel_config.data_parallel_rank
    layers = []
    for index in range(2):
        weight, scale = _full_table(sum(HEAD_SIZES))
        weight = (weight.float() * (1 - 2 * index)).to(torch.float8_e4m3fn)
        table = ParallelEngramEmbedding(
            sum(HEAD_SIZES), DIM, HEAD_SIZES, cpu_offload=True
        )
        table.weight.weight_loader(table.weight, weight)
        table.weight_scale_inv.weight_loader(table.weight_scale_inv, scale)
        staged = torch.empty(0, device="cuda")
        reference = (weight.cuda(), scale.cuda())
        layers.append(SimpleNamespace(embed_tokens=table, staged_rows=staged))
        layers[-1].layer_hash_index, layers[-1].reference = index, reference
    batch = EngramBatch(layers, 257)

    def check(ids):
        tokens = ids.shape[0]
        for layer in layers:
            x = batch.wkv_input(layer, tokens)
            sf = convert_swizzled_to_linear(x.scale, tokens, x.data.shape[1], BLOCK)
            actual = dequant_mxfp8_to_bf16(x.data, sf).unflatten(-1, (-1, DIM))
            weight, scale = layer.reference
            expected = _reference(weight, scale, ids[:, layer.layer_hash_index])
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    for counts in ((7, 0, 5, 1), (127,) * 4, (128,) * 4, (257, 0, 129, 1)):
        tokens = counts[dp_rank]
        seed = 700 + 10 * dp_rank
        ids = torch.stack([_make_ids(HEAD_SIZES, tokens, seed + i) for i in (0, 1)], 1)
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=tokens,
            num_tokens_across_dp=torch.tensor(counts, dtype=torch.int32),
        ):
            batch.prepare_embeddings(ids)
            check(ids)
            for capture in ("full", "breakable") if len(set(counts)) == 1 else ():
                warmup = torch.cuda.Stream()
                warmup.wait_stream(torch.cuda.current_stream())
                if capture == "full":
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=warmup):
                        batch.prepare_embeddings(ids)
                else:
                    graph = BreakableCUDAGraphCapture()
                    with torch.cuda.stream(warmup), graph:
                        batch.prepare_embeddings(ids)
                torch.cuda.current_stream().wait_stream(warmup)
                for step in (2, 3):
                    replay_ids = _make_ids(HEAD_SIZES, tokens, seed + step)
                    ids.copy_(torch.stack([replay_ids] * 2, 1))
                    graph.replay()
                    check(ids)
                del graph


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

    engram_host_table = _engram_host_table()

    with pytest.MonkeyPatch.context() as patch:
        # Exercise the /dev/shm path even where host-memory tables are used.
        patch.setattr(engram_host_table, "share_host_table", lambda *args: None)
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


def _engram_host_table():
    if current_platform.is_rocm():
        from vllm.models.deepseek_v41.amd.ops import engram_host_table
    else:
        from vllm.models.deepseek_v41.nvidia.ops import (
            engram_fabric_table as engram_host_table,
        )
    return engram_host_table


def _check_ple_shared_table(shm_fits):
    """Qwen4Exp's PLE table is one /dev/shm mapping per EDP group, loaded by its
    leader; without room in /dev/shm, every rank loads its own table."""
    group = get_engram_dp_group()
    with pytest.MonkeyPatch.context() as patch:
        if not shm_fits:
            patch.setattr(engram_ops, "_shm_shortfall", lambda num_bytes: "too small")
        embedding = ple_ops.Qwen4ExpPLESharedHostEmbedding(
            8,
            DIM,
            params_dtype=torch.bfloat16,
            padding_size=1,
            prefix="test.ple_embedding",
            embedding_method=ple_ops.Qwen4ExpPLEUnquantizedEmbeddingMethod(),
        )
    # Every rank offers different rows; a shared table holds the leader's.
    offered = torch.full((8, DIM), group.rank_in_group + 1.0, dtype=torch.bfloat16)
    embedding.weight_loader(embedding.weight, offered)
    expected = 1.0 if shm_fits else group.rank_in_group + 1.0
    assert embedding.weight.is_pinned()
    assert torch.all(embedding.weight == expected)
    group.barrier()


def _check_flash_shared_table(shm_fits):
    """HY4-Flash Engram tables - lazily allocated from the checkpoint dtype -
    become one shared mapping per EDP group, written by the group leader;
    without room in /dev/shm, every rank keeps a private table."""
    from vllm.models.hy_v4_flash.common.engram import FlashEmbedding

    group = get_engram_dp_group()
    rank = torch.distributed.get_rank()
    sizes = [4] * 8
    with pytest.MonkeyPatch.context() as patch:
        if not shm_fits:
            patch.setattr(engram_ops, "_shm_shortfall", lambda num_bytes: "too small")

        # Lazy allocation: the dtype arrives with the checkpoint tensor. Every
        # rank offers its own rows, so a private slip shows up in the values.
        embedding = FlashEmbedding(
            sizes, 8, None, True, share_group=group, prefix="test.lazy"
        )
        offered = torch.full((sum(sizes), 8), rank + 1.0, dtype=torch.bfloat16)
        embedding.load_weights([("embedding.weight", offered)])
        expected = group.ranks[0] + 1.0 if shm_fits else rank + 1.0
        assert embedding.embedding.weight.is_pinned()
        assert torch.all(embedding.embedding.weight == expected)
        # The CUDA lookup view reads the same shared rows.
        table, scales = embedding.tables()
        assert torch.all(table == expected)
        assert scales is None

        # Eager (dummy-style) allocation shares the same way.
        eager = FlashEmbedding(
            sizes, 8, torch.bfloat16, True, share_group=group, prefix="test.eager"
        )
        eager.load_weights([("embedding.weight", offered)])
        assert torch.all(eager.embedding.weight == expected)
        assert torch.all(eager.tables()[0] == expected)

        # FP8 row scales may arrive before the table and share storage too.
        fp8 = FlashEmbedding(sizes, 8, None, True, share_group=group, prefix="test.fp8")
        scale = torch.full((sum(sizes),), rank + 3.0, dtype=torch.bfloat16)
        table8 = torch.full((sum(sizes), 8), float(rank + 1), dtype=torch.float8_e4m3fn)
        fp8.load_weights(
            [
                ("embedding.weight_scale_inv", scale),
                ("embedding.weight", table8),
            ]
        )
        assert fp8.fp8
        assert torch.all(
            fp8.embedding.weight_scale_inv
            == (group.ranks[0] + 3.0 if shm_fits else rank + 3.0)
        )
        weight8, scale_view = fp8.tables()
        assert torch.all(weight8.to(torch.float32) == expected)
        assert scale_view is not None
        assert torch.all(
            scale_view == (group.ranks[0] + 3.0 if shm_fits else rank + 3.0)
        )
    group.barrier()


def _check_flash_device_local_table():
    """Disabling CPU offload preserves GPU storage for eager and lazy loads."""
    from vllm.models.hy_v4_flash.common.engram import FlashEmbedding

    sizes = [4] * 8
    device = torch.device("cuda", torch.accelerator.current_device_index())
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        for eager in (False, True):
            with torch.device(device):
                embedding = FlashEmbedding(
                    sizes, 8, dtype if eager else None, cpu_offload=False
                )
            assert embedding.embedding.weight.device == device
            offered = torch.full((sum(sizes), 8), 2.0, dtype=dtype, device="cpu")
            weights = [("embedding.weight", offered)]
            if dtype == torch.float8_e4m3fn:
                scale = torch.full((sum(sizes),), 3.0, dtype=torch.bfloat16)
                weights.insert(0, ("embedding.weight_scale_inv", scale))
            embedding.load_weights(weights)
            table, scales = embedding.tables()
            assert table.device == device
            assert torch.all(table.to(torch.float32) == 2.0)
            if dtype == torch.float8_e4m3fn:
                assert scales is not None and scales.device == device
                assert torch.all(scales == 3.0)
            else:
                assert scales is None


def _check_flash_shared_load_order():
    """Equal-sized but different logical tables must not share a mapping."""
    from vllm.models.hy_v4_flash.common.engram import FlashEmbedding

    group = get_engram_dp_group()
    sizes = [4] * 8
    a = FlashEmbedding(sizes, 8, None, True, share_group=group, prefix="layer.a")
    b = FlashEmbedding(sizes, 8, None, True, share_group=group, prefix="layer.b")
    first = a if group.rank_in_group == 0 else b
    offered = torch.ones((sum(sizes), 8), dtype=torch.bfloat16)
    # The mismatch must be detected before entering the allocation collective.
    with pytest.MonkeyPatch.context() as patch:

        def unexpected_allocation(*args):
            pytest.fail("Entered shared allocation with mismatched tensor order")

        patch.setattr(
            engram_ops.DPSharedEngramStorage, "share_shm_table", unexpected_allocation
        )
        with pytest.raises(RuntimeError, match="identical tensor load order"):
            first.load_weights([("embedding.weight", offered)])
    assert a.embedding.weight.numel() == b.embedding.weight.numel() == 0
    group.barrier()


def _check_cumem_host_table(failure, packed_fp4):
    """Chunked cuMem tables read back across ranks; a failed allocation or
    import on any rank falls back to /dev/shm on all of them."""
    engram_host_table = _engram_host_table()
    is_rocm = current_platform.is_rocm()
    if is_rocm:
        if not engram_host_table.host_memory_supported():
            return
    elif not engram_host_table.fabric_host_memory_supported():
        return

    group = get_engram_dp_group()
    # Three 2 MiB chunks, the last one partial.
    row_bytes = DIM // 2 if packed_fp4 else DIM
    rows = 5 * (1 << 20) // row_bytes
    with pytest.MonkeyPatch.context() as patch:
        if is_rocm:
            patch.setenv("VLLM_ROCM_ENGRAM_HIP_VMM", "1")
        patch.setattr(engram_host_table, "CHUNK_BYTES", 2 << 20)
        if failure == "create" and group.rank_in_group == 0:
            patch.setattr(engram_host_table, "NUMA_RESERVE_FRACTION", 1.0)
        if failure == "import" and group.rank_in_group == 1:
            if is_rocm:
                call = engram_host_table._call

                def fail_import(name, *args):
                    if name == "hipMemImportFromShareableHandle":
                        raise RuntimeError(f"{name} failed: hipErrorInvalidValue")
                    call(name, *args)

                patch.setattr(engram_host_table, "_call", fail_import)
            else:
                patch.setattr(
                    engram_host_table.cu,
                    "cuMemImportFromShareableHandle",
                    lambda *args: (
                        engram_host_table.cu.CUresult.CUDA_ERROR_INVALID_VALUE,
                        None,
                    ),
                )
        storage = engram_ops.DPSharedEngramStorage(
            rows, DIM, BLOCK, group, packed_fp4=packed_fp4
        )
    assert storage.huge_pages == (failure is None)
    table = (
        torch.arange(rows * row_bytes)
        .remainder(251)
        .to(torch.uint8)
        .view(rows, row_bytes)
    )
    if group.rank_in_group == 0:
        storage.weight.view(torch.uint8).copy_(table)
    group.barrier()
    weight, _ = storage.get_views(storage.weight, storage.weight_scale_inv)
    assert torch.equal(weight.view(torch.uint8).cpu(), table)
    group.barrier()


def _check_table(
    vllm_config,
    cpu_offload,
    dp_shared_memory,
    n_heads,
    sequence_parallel,
    p2p,
    packed_fp4,
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
            quantization_config={
                "engram_embed_quant": "mxfp4" if packed_fp4 else "fp8"
            },
        )
        with set_current_vllm_config(vllm_config):
            weight, scale_inv = _full_table(rows, packed_fp4)
            layout = EngramLayout(config)
            # A host budget between MXFP4 and FP8 sizes must admit only MXFP4.
            with m.context() as memory_patch:
                memory_patch.setattr(
                    engram_ops, "get_available_memory", lambda: rows * DIM
                )
                assert engram_ops.can_share_engram_tables(layout) == packed_fp4
            # Stub configuration inputs, not Engram's initialization or execution.
            component_config = SimpleNamespace(
                engram_config=EngramConfig(
                    cpu_offload=cpu_offload,
                    dp_shared_memory=dp_shared_memory,
                ),
                scheduler_config=vllm_config.scheduler_config,
                load_config=load_config,
            )
            with m.context() as init_patch, torch.device("cuda"):
                init_patch.setattr(
                    engram_ops, "get_current_vllm_config", lambda: component_config
                )
                init_patch.setattr(
                    engram_ops.current_platform, "is_fully_connected", lambda _: p2p
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
            # As the model loader does: peers must not read rows still loading.
            layer.process_weights_after_loading()
            # Shared host tables and GPU peer reads look up only local tokens.
            local = layer.dp_size == 1
            assert (layer.peer_rows > 0) == (not cpu_offload and tp_size == 1 and p2p)
            if dp_shared_memory:
                assert layer.dp_size == 1
                assert layer.head_start == tp_rank * layer.part_n_hash_cols
                # A dmabuf CPU mapping cannot be pinned; lookups use the GPU views.
                if not (current_platform.is_rocm() and layer._shared_memory.huge_pages):
                    assert layer.weight.is_pinned()
                    assert layer.weight_scale_inv.is_pinned()
            if local:

                def unexpected_collective(*args, **kwargs):
                    raise AssertionError("local lookups must not use DP collectives")

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
                gathered = gather_engram_hashes(ids, dp_shared_memory=local)
                expected_tokens = num_tokens if local else max(counts) * dp_size
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
            and n_heads == 7
            and not sequence_parallel
            and (not dp_shared_memory or tp_size == 1)
        ):
            _check_dummy_hash_model_forward(
                vllm_config,
                engram,
                head_sizes,
                weight,
                scale_inv,
                dp_shared_memory,
            )

        replay = n_heads == 7 and (
            (dp_shared_memory and sequence_parallel == (tp_size > 1))
            or (
                cpu_offload
                and not dp_shared_memory
                and sequence_parallel == (tp_size > 1)
            )
            or layer.peer_rows > 0
        )
        if replay:
            # Peer reads have no eager breaks to count.
            captures = ("full",) if layer.peer_rows else ("full", "breakable")
            for capture in captures:
                _check_prefetch_replay(
                    engram,
                    head_sizes,
                    weight,
                    scale_inv,
                    local,
                    capture,
                )
        if dp_shared_memory and n_heads == 7 and sequence_parallel == (tp_size > 1):
            with m.context() as storage_patch:
                storage_patch.setattr(layer.weight, "data", layer.weight.data.clone())
                with pytest.raises(RuntimeError, match="storage must not be replaced"):
                    layer(ids)


def _check_dummy_hash_model_forward(
    vllm_config, engram, head_sizes, weight, scales, dp_shared_memory
):
    """A metadata-less replica looks up dead rows, so it joins DP sharding and
    the memory profile sees the Engram activations."""
    from vllm.models.deepseek_v41.common.engram import NgramHashState

    if current_platform.is_rocm():
        from vllm.models.deepseek_v41.amd import model as model_ops
    else:
        from vllm.models.deepseek_v41.nvidia import model as model_ops

    dp_rank = vllm_config.parallel_config.data_parallel_rank
    if dp_shared_memory and dp_rank != 1:
        return
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
        # Engram hashes and the mask stay the last positional arguments;
        # **kwargs absorbs the decoder's keyword-only flags.
        def __call__(self, hidden, positions, input_ids, *args, **kwargs):
            hashes, keep = args[-2:]
            assert hashes is not None and keep is not None
            assert bool(keep.all()) == (dp_rank != 1)
            if dp_rank == 1:
                assert torch.all(hashes == engram_ops.DEAD_ID)
            output = engram.embed(hashes[:, 0])
            # Trailing None is the NVIDIA layer's previous_aux; this stub
            # captures no aux states.
            outputs = (output, hidden, hidden, hidden, hidden, None)
            return outputs[:5] if current_platform.is_rocm() else outputs

    model = model_ops.DeepseekV4Model.__new__(model_ops.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.parallel_config = vllm_config.parallel_config
    model.use_native_mega_moe = False
    model.use_mega_gate = False
    model.use_mega_moe = False
    model.use_sequence_parallel = False
    model.fuse_mhc_all_reduce = False
    model.engram_batch = None
    model.engram_hash = state
    model.engram_swa_prefix = "swa"
    model.engram_dp_shared_memory = dp_shared_memory
    model.layers = [Decoder(engram=engram)]
    model.start_layer = 0
    model.end_layer = 1
    model.decoder_replay_start = model.end_layer
    model.decoder_replay_layers = None
    model.pipeline_relay = None
    model.aux_hidden_state_layers = ()
    model.mhc_post = lambda hidden, *args: hidden
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
        patch.setattr(
            model_ops, "mhc_post_tilelang", lambda hidden, *args: hidden, raising=False
        )
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
            output = model(
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


def _check_prefetch_replay(
    engram, head_sizes, weight, scale_inv, dp_shared_memory, capture
):
    """DP-sharded and shared prefetches must replay with new IDs."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    ids = _make_ids(head_sizes, 8, seed=501)
    tp_rank = engram_ops.get_tensor_model_parallel_rank()
    tokens = 8 // engram.embed_tokens.tp_size if engram.use_sequence_parallel else 8
    output = torch.empty(
        tokens, len(head_sizes), DIM, device="cuda", dtype=torch.bfloat16
    )

    def step():
        gathered = gather_engram_hashes(ids.clone(), dp_shared_memory=dp_shared_memory)
        engram.prepare_embeddings(gathered)
        output.copy_(engram.embed(ids))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engram_ops, "engram_gathered_num_tokens", lambda: 8)
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            step()
        if capture == "breakable":
            graph = BreakableCUDAGraphCapture()
            with torch.cuda.stream(warmup), graph:
                step()
            assert graph.num_eager_breaks == 2
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=warmup):
                step()
        torch.cuda.current_stream().wait_stream(warmup)
        for iteration in range(2):
            ids.copy_(_make_ids(head_sizes, 8, seed=600 + iteration))
            graph.replay()
            expected = _reference(weight.cuda(), scale_inv.cuda(), ids)
            if engram.use_sequence_parallel:
                expected = expected[tp_rank * tokens : (tp_rank + 1) * tokens]
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
        torch.accelerator.synchronize()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="CUDA or ROCm required"
)
@pytest.mark.parametrize("tp_size", [1, 2])
def test_engram_table_shared_across_dp_replicas(tp_size, monkeypatch, tmp_path):
    """Check all storage modes with one process launch per TP/DP topology."""
    if torch.accelerator.device_count() < WORLD_SIZE:
        pytest.skip(f"Need {WORLD_SIZE} GPUs to run the test.")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    mp.spawn(_worker, args=(tp_size, get_open_port(), str(tmp_path)), nprocs=WORLD_SIZE)
    cleanup_dist_env_and_memory()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_engram_hip_host_table_failure_clears_hip_error():
    """A failed HIP VMM call must not fail the next torch launch, which the
    /dev/shm fallback runs right after it."""
    from vllm.models.deepseek_v41.amd.ops import engram_host_table

    with pytest.raises(RuntimeError, match="hipMemCreate failed"):
        engram_host_table._create_chunk(2 << 20, numa_node=1 << 20)
    assert torch.ones(4, device="cuda").sum().item() == 4


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


@pytest.mark.parametrize("query_fails", [False, True])
def test_engram_tables_too_large_for_shm_are_not_shared(monkeypatch, query_fails):
    """A /dev/shm that is too small or cannot be queried must fall back, and the
    leader must still broadcast that decision so its peers do not hang."""
    broadcasts = []

    def broadcast(obj):
        broadcasts.append(obj)
        return obj

    def query(*args, **kwargs):
        raise OSError("statvfs failed")

    monkeypatch.setattr(engram_ops, "get_engram_dp_size", lambda: 2)
    monkeypatch.setattr(
        engram_ops,
        "get_engram_dp_group",
        lambda: SimpleNamespace(rank_in_group=0, broadcast_object=broadcast),
    )
    if query_fails:
        monkeypatch.setattr(engram_ops, "check_shm_free_space", query)
    layout = SimpleNamespace(num_embeddings=(1 << 50,), head_dim=DIM, row_bytes=DIM)
    assert not engram_ops.can_share_engram_tables(layout)
    assert len(broadcasts) == 1 and broadcasts[0] is not None


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
