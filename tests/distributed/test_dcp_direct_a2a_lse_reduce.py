# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for direct symmetric-memory DCP collectives."""

import functools
import time
from unittest.mock import MagicMock

import multiprocess as mp
import pytest
import torch
import torch.distributed as dist

import vllm.v1.attention.ops.dcp_utils as dcp_utils
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)
pytestmark = pytest.mark.skip_global_cleanup


def _has_multicast_support() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except Exception:
        return False


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float32": torch.float32,
    }[dtype_name]


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=3e-2)


def _q_gather_reference(
    local_query: torch.Tensor,
    world_size: int,
    padded_num_heads: int | None,
) -> torch.Tensor:
    num_tokens, heads_per_rank, head_dim = local_query.shape
    gathered = torch.empty(
        (world_size * num_tokens, heads_per_rank, head_dim),
        dtype=local_query.dtype,
        device=local_query.device,
    )
    dist.all_gather_into_tensor(gathered, local_query.contiguous())
    expected = (
        gathered.view(world_size, num_tokens, heads_per_rank, head_dim)
        .movedim(0, 1)
        .reshape(num_tokens, world_size * heads_per_rank, head_dim)
    )
    if padded_num_heads is not None:
        reserved = expected.new_empty((num_tokens, padded_num_heads, head_dim))
        reserved.resize_(expected.shape)
        reserved.copy_(expected)
        expected = reserved
    return expected


def _assert_q_gather_matches_reference(
    actual: torch.Tensor,
    local_query: torch.Tensor,
    world_size: int,
    padded_num_heads: int | None,
) -> None:
    expected = _q_gather_reference(local_query, world_size, padded_num_heads)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.is_contiguous()
    assert actual.stride() == expected.stride()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))

    storage_num_heads = padded_num_heads or world_size * local_query.shape[1]
    remaining_storage_bytes = (
        actual.untyped_storage().nbytes()
        - actual.storage_offset() * actual.element_size()
    )
    required_storage_bytes = (
        local_query.shape[0]
        * storage_num_heads
        * local_query.shape[2]
        * actual.element_size()
    )
    assert remaining_storage_bytes >= required_storage_bytes


def _distributed_run(fn, world_size: int, extra_env: dict[str, str]) -> None:
    port = str(get_open_port())
    processes: list[mp.Process] = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
            **extra_env,
        }
        process = mp.Process(target=fn, args=(env,))
        processes.append(process)
        process.start()

    deadline = time.monotonic() + 120
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))

    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
        assert process.exitcode == 0


class _FakeGroupCoordinator:
    device_group = None
    cpu_group = None
    world_size = 4


class _FakeProcessGroup:
    def size(self) -> int:
        return 4

    def rank(self) -> int:
        return 0


class TestDirectDCPGating:
    def test_env_disabled_returns_none(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_A2A", "0")
        dcp_utils.get_direct_dcp_a2a_workspace.cache_clear()
        workspace = dcp_utils.get_direct_dcp_a2a_workspace(
            _FakeGroupCoordinator(), torch.device("cpu"), 16, 2, 32, torch.bfloat16, 1
        )
        assert workspace is None

    def test_forced_with_unsupported_dtype_raises(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_A2A", "1")
        dcp_utils.get_direct_dcp_a2a_workspace.cache_clear()
        with pytest.raises(ValueError, match="does not support"):
            dcp_utils.get_direct_dcp_a2a_workspace(
                _FakeGroupCoordinator(),
                torch.device("cpu"),
                16,
                2,
                32,
                torch.float32,
                1,
            )

    def test_zero_ubatches_raises(self):
        with pytest.raises(ValueError, match="ubatch"):
            dcp_utils.DirectDCPA2AWorkspace(
                None, torch.device("cpu"), 16, 2, 32, torch.bfloat16, num_ubatches=0
            )

    def test_auto_with_unsupported_dtype_returns_none(self, monkeypatch):
        monkeypatch.delenv("VLLM_USE_DIRECT_DCP_A2A", raising=False)
        dcp_utils.get_direct_dcp_a2a_workspace.cache_clear()
        workspace = dcp_utils.get_direct_dcp_a2a_workspace(
            _FakeGroupCoordinator(), torch.device("cpu"), 16, 2, 32, torch.float32, 1
        )
        assert workspace is None

    def test_q_gather_env_disabled_returns_none(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_Q_GATHER", "0")
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_A2A", "1")
        dcp_utils.get_direct_dcp_q_gather_workspace.cache_clear()
        workspace = dcp_utils.get_direct_dcp_q_gather_workspace(
            _FakeGroupCoordinator(),
            torch.device("cpu"),
            16,
            2,
            32,
            torch.bfloat16,
            1,
        )
        assert workspace is None

    def test_q_gather_flag_is_independent(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_Q_GATHER", "1")
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_A2A", "0")
        monkeypatch.setattr(dcp_utils, "_symm_mem_spans_group", lambda group: True)
        dcp_utils.get_direct_dcp_q_gather_workspace.cache_clear()
        workspace = object()
        init_workspace = MagicMock(return_value=workspace)
        monkeypatch.setattr(
            dcp_utils,
            "DirectDCPQGatherWorkspace",
            init_workspace,
        )

        result = dcp_utils.get_direct_dcp_q_gather_workspace(
            _FakeGroupCoordinator(),
            torch.device("cpu"),
            16,
            2,
            32,
            torch.float32,
            1,
        )

        assert result is workspace
        assert init_workspace.call_args.args[5] == torch.float32

    def test_kv_gather_env_disabled_returns_none(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_KV_GATHER", "0")
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_A2A", "1")
        dcp_utils.get_direct_dcp_kv_gather_workspace.cache_clear()
        workspace = dcp_utils.get_direct_dcp_kv_gather_workspace(
            _FakeGroupCoordinator(),
            torch.device("cpu"),
            64,
            576,
            512,
            torch.bfloat16,
            1,
        )
        assert workspace is None

    def test_kv_gather_flag_is_independent(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_KV_GATHER", "1")
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_A2A", "0")
        monkeypatch.setattr(dcp_utils, "_symm_mem_spans_group", lambda group: True)
        dcp_utils.get_direct_dcp_kv_gather_workspace.cache_clear()
        workspace = object()
        init_workspace = MagicMock(return_value=workspace)
        monkeypatch.setattr(
            dcp_utils,
            "DirectDCPKVGatherWorkspace",
            init_workspace,
        )

        result = dcp_utils.get_direct_dcp_kv_gather_workspace(
            _FakeGroupCoordinator(),
            torch.device("cpu"),
            64,
            576,
            512,
            torch.bfloat16,
            1,
        )

        assert result is workspace

    @pytest.mark.parametrize(
        ("flag_name", "factory_name", "factory_args"),
        [
            (
                "VLLM_USE_DIRECT_DCP_Q_GATHER",
                "get_direct_dcp_q_gather_workspace",
                (16, 2, 32, torch.bfloat16, 1),
            ),
            (
                "VLLM_USE_DIRECT_DCP_KV_GATHER",
                "get_direct_dcp_kv_gather_workspace",
                (64, 576, 512, torch.bfloat16, 1),
            ),
        ],
    )
    def test_gather_requires_multicast(
        self,
        monkeypatch,
        flag_name,
        factory_name,
        factory_args,
    ):
        factory = getattr(dcp_utils, factory_name)
        monkeypatch.setenv(flag_name, "1")
        monkeypatch.setattr(dcp_utils, "_symm_mem_spans_group", lambda group: False)
        factory.cache_clear()

        assert (
            factory(
                _FakeGroupCoordinator(),
                torch.device("cpu"),
                *factory_args,
            )
            is None
        )

    def test_kv_gather_rejects_invalid_workspace_geometry(self):
        with pytest.raises(ValueError, match="ubatch"):
            dcp_utils.DirectDCPKVGatherWorkspace(
                None, torch.device("cpu"), 64, 576, 512, num_ubatches=0
            )
        with pytest.raises(ValueError, match="divide evenly"):
            dcp_utils.DirectDCPKVGatherWorkspace(
                _FakeProcessGroup(), torch.device("cpu"), 63, 576, 512
            )
        with pytest.raises(ValueError, match="16-byte"):
            dcp_utils.DirectDCPKVGatherWorkspace(
                _FakeProcessGroup(), torch.device("cpu"), 64, 16, 4
            )

    @pytest.mark.parametrize(
        ("token_dim", "plane_split_dim", "dtype", "supported"),
        [
            (576, 512, torch.bfloat16, True),
            (576, 512, torch.float16, True),
            (576, 512, torch.float8_e4m3fn, True),
            (16, 8, torch.bfloat16, True),
            (16, 4, torch.bfloat16, False),
            (24, 16, torch.float8_e4m3fn, False),
            (16, 0, torch.bfloat16, False),
            (16, 16, torch.bfloat16, False),
        ],
    )
    def test_kv_gather_requires_each_plane_aligned(
        self, token_dim, plane_split_dim, dtype, supported
    ):
        assert (
            dcp_utils._kv_gather_layout_supported(token_dim, plane_split_dim, dtype)
            is supported
        )

    def test_q_gather_rejects_invalid_workspace_geometry(self):
        with pytest.raises(ValueError, match="ubatch"):
            dcp_utils.DirectDCPQGatherWorkspace(
                None, torch.device("cpu"), 16, 2, 32, num_ubatches=0
            )
        with pytest.raises(ValueError, match="padded heads"):
            dcp_utils.DirectDCPQGatherWorkspace(
                _FakeProcessGroup(),
                torch.device("cpu"),
                16,
                2,
                32,
                padded_num_heads=7,
            )
        with pytest.raises(ValueError, match="16-byte"):
            dcp_utils.DirectDCPQGatherWorkspace(
                _FakeProcessGroup(),
                torch.device("cpu"),
                16,
                1,
                3,
            )


def _manager_config(dcp_comm_backend: str = "a2a"):
    config = MagicMock()
    config.parallel_config.num_ubatches = 1
    config.parallel_config.dcp_comm_backend = dcp_comm_backend
    config.scheduler_config.max_num_batched_tokens = 16
    config.scheduler_config.max_num_seqs = 4
    config.num_speculative_tokens = 0
    config.speculative_config = None
    config.compilation_config.max_cudagraph_capture_size = 0
    return config


def test_mla_dcp_manager_selects_direct_backends(monkeypatch):
    import vllm.v1.attention.ops.dcp_utils as dcp_manager

    group = MagicMock(world_size=2)
    monkeypatch.setattr(dcp_manager, "get_dcp_group", lambda: group)
    direct_a2a = MagicMock()
    direct_query = MagicMock()
    direct_kv = MagicMock()
    monkeypatch.setattr(
        dcp_manager, "get_direct_dcp_a2a_workspace", MagicMock(return_value=direct_a2a)
    )
    monkeypatch.setattr(
        dcp_manager,
        "get_direct_dcp_q_gather_workspace",
        MagicMock(return_value=direct_query),
    )
    monkeypatch.setattr(
        dcp_manager,
        "get_direct_dcp_kv_gather_workspace",
        MagicMock(return_value=direct_kv),
    )

    manager = dcp_manager.MLADCPManager(
        vllm_config=_manager_config(),
        device=torch.device("cpu"),
        num_heads=2,
        query_head_dim=8,
        output_head_dim=4,
        query_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        padded_num_heads=None,
        is_lse_base_on_e=False,
        use_pcp=False,
    )
    assert manager.query_gather == direct_query.gather
    assert manager.init_kv_gather(64, 16, 8, torch.bfloat16)
    dcp_manager.get_direct_dcp_kv_gather_workspace.assert_called_once_with(
        group,
        torch.device("cpu"),
        64,
        16,
        8,
        torch.bfloat16,
        1,
    )
    local_kv = torch.empty(2, 8)
    dst_rows = torch.tensor([0, 1], dtype=torch.int32)
    compact_kv = (torch.empty(2, 1, 6), torch.empty(2, 1, 2))
    direct_kv.gather.return_value = compact_kv
    assert manager.use_direct_kv_gather
    assert manager.direct_kv_gather(local_kv, dst_rows, 2, 1) is compact_kv
    direct_kv.gather.assert_called_once_with(
        local_kv,
        dst_rows,
        2,
        1,
    )
    output, lse = torch.empty(1), torch.empty(1)
    seq_lens = torch.ones(1, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    manager.combine(
        output,
        lse,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )
    direct_a2a.lse_reduce.assert_called_once_with(
        output,
        lse,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        is_lse_base_on_e=False,
    )


def test_mla_dcp_manager_selects_fallback_backends(monkeypatch):
    import vllm.v1.attention.ops.dcp_utils as dcp_manager

    group = MagicMock(world_size=2)
    gathered_query = torch.empty(1, 4, 8)
    group.all_gather.return_value = gathered_query
    monkeypatch.setattr(dcp_manager, "get_dcp_group", lambda: group)
    monkeypatch.setattr(
        dcp_manager, "get_direct_dcp_a2a_workspace", MagicMock(return_value=None)
    )
    monkeypatch.setattr(
        dcp_manager, "get_direct_dcp_q_gather_workspace", MagicMock(return_value=None)
    )
    monkeypatch.setattr(
        dcp_manager, "get_direct_dcp_kv_gather_workspace", MagicMock(return_value=None)
    )
    fallback_combine = MagicMock(return_value=torch.empty(1))
    monkeypatch.setattr(dcp_manager, "dcp_a2a_lse_reduce", fallback_combine)

    manager = dcp_manager.MLADCPManager(
        vllm_config=_manager_config(),
        device=torch.device("cpu"),
        num_heads=2,
        query_head_dim=8,
        output_head_dim=4,
        query_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        padded_num_heads=None,
        is_lse_base_on_e=True,
        use_pcp=False,
    )

    all_gather = MagicMock()
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", all_gather)
    assert not manager.init_kv_gather(64, 16, 8, torch.bfloat16)
    output, local = torch.empty(4, 8), torch.empty(2, 8)
    manager.kv_gather(output, local)
    all_gather.assert_called_once_with(output, local, group=group.device_group)

    query = torch.empty(1, 2, 8)
    assert manager.query_gather is not None
    assert manager.query_gather(query) is gathered_query
    group.all_gather.assert_called_once_with(query, dim=1)

    partial_output, partial_lse = torch.empty(1), torch.empty(1)
    seq_lens = torch.ones(1, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    manager.combine(
        partial_output,
        partial_lse,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )
    fallback_combine.assert_called_once_with(
        partial_output,
        partial_lse,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        cp_group=group,
        is_lse_base_on_e=True,
    )


def test_dcp_workspace_covers_parallel_drafting():
    config = _manager_config()
    config.scheduler_config.max_num_batched_tokens = 128
    config.num_speculative_tokens = 3
    config.speculative_config = MagicMock(parallel_drafting=True)

    assert dcp_utils.get_dcp_workspace_max_num_tokens(config) == 28


def test_mla_dcp_manager_selects_pcp_combine(monkeypatch):
    import vllm.v1.attention.ops.dcp_utils as dcp_manager

    monkeypatch.setattr(dcp_manager, "get_dcp_group", lambda: MagicMock(world_size=2))
    manager = dcp_manager.MLADCPManager(
        vllm_config=_manager_config(dcp_comm_backend="ag_rs"),
        device=torch.device("cpu"),
        num_heads=2,
        query_head_dim=8,
        output_head_dim=4,
        query_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        padded_num_heads=None,
        is_lse_base_on_e=True,
        use_pcp=True,
    )

    assert isinstance(manager.combine, functools.partial)
    assert manager.combine.func is dcp_manager.cp_lse_ag_out_ar
    assert manager.query_gather is None


def test_dcp_chunk_workspace_alignment_covers_interleave():
    from vllm.model_executor.layers.attention.mla_attention import (
        align_mla_chunked_context_workspace_size,
    )

    config = MagicMock()
    config.cache_config.block_size = 32
    config.parallel_config.decode_context_parallel_size = 8
    config.parallel_config.cp_kv_cache_interleave_size = 8

    # Alignment is lcm(block_size, dcp_size * interleave_size) = 64, and the
    # workspace only has to hold a single aligned chunk step, independent of
    # max_num_seqs.
    assert align_mla_chunked_context_workspace_size(config, 100) == 128
    assert align_mla_chunked_context_workspace_size(config, 8) == 64


@pytest.mark.parametrize("use_direct", [False, True])
def test_sparse_mla_builder_initializes_dcp_manager(monkeypatch, use_direct):
    import vllm.model_executor.layers.attention.sparse_mla_attention as sparse_mla

    monkeypatch.setattr(
        sparse_mla.AttentionMetadataBuilder,
        "__init__",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        sparse_mla,
        "get_dcp_group",
        lambda: MagicMock(world_size=2),
    )
    monkeypatch.setattr(
        sparse_mla,
        "get_mla_dims",
        lambda _: MagicMock(kv_lora_rank=8, qk_rope_head_dim=4),
    )

    manager = object.__new__(dcp_utils.MLADCPManager)
    manager.init_kv_gather = MagicMock(return_value=use_direct)
    layer = MagicMock(dcp_manager=manager)
    config = MagicMock()
    config.model_config.dtype = torch.bfloat16
    config.model_config.max_model_len = 64
    config.model_config.hf_config.index_topk = 8
    config.scheduler_config.max_num_batched_tokens = 64
    config.scheduler_config.max_num_seqs = 2
    config.cache_config.block_size = 4
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 2
    config.parallel_config.cp_kv_cache_interleave_size = 1
    config.compilation_config.static_forward_context = {"layer": layer}

    builder = sparse_mla.SparseMLACommonMetadataBuilder(
        MagicMock(),
        ["layer"],
        config,
        torch.device("cpu"),
    )

    assert builder.dcp_manager is manager
    manager.init_kv_gather.assert_called_once_with(
        builder.chunked_prefill_workspace_size,
        12,
        8,
        torch.bfloat16,
    )
    local_rows = builder.chunked_prefill_workspace_size // 2
    expected_rows = (
        local_rows
        if use_direct
        else builder.chunked_prefill_workspace_size + local_rows
    )
    assert builder.chunked_prefill_workspace.shape == (expected_rows, 12)


def test_sparse_mla_workspace_preserves_non_dcp_size():
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonMetadataBuilder,
    )

    config = MagicMock()
    config.model_config.max_model_len = 1
    config.model_config.hf_config.index_topk = 7
    config.scheduler_config.max_num_seqs = 3
    config.cache_config.block_size = 4
    config.parallel_config.decode_context_parallel_size = 1

    assert (
        SparseMLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(config)
        == 21
    )


def _distributed_direct_q_gather_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        heads_per_rank, head_dim, max_num_tokens = 6, 576, 128
        padded_num_heads = 128 if world_size == 4 else None
        active_ubatch = [0]
        dcp_utils.dbo_current_ubatch_id = lambda: active_ubatch[0]
        for dtype_idx, dtype_name in enumerate(
            ("bfloat16", "float8_e4m3fn", "float32")
        ):
            dtype = _dtype_from_name(dtype_name)
            workspace = dcp_utils.DirectDCPQGatherWorkspace(
                dist.group.WORLD,
                device,
                max_num_tokens,
                heads_per_rank,
                head_dim,
                dtype,
                num_ubatches=2,
                padded_num_heads=padded_num_heads,
            )

            cases = (
                ((1, False), (128, True), (17, True), (5, False))
                if dtype == torch.bfloat16
                else ((17, True),)
            )
            for iteration, (num_tokens, noncontiguous) in enumerate(cases):
                generator = torch.Generator(device=device)
                generator.manual_seed(9000 + rank * 101 + dtype_idx * 1009 + iteration)
                source_num_heads = (
                    heads_per_rank + 2 if noncontiguous else heads_per_rank
                )
                query_storage = torch.randn(
                    num_tokens,
                    source_num_heads,
                    head_dim,
                    device=device,
                    dtype=torch.bfloat16,
                    generator=generator,
                ).to(dtype)
                local_query = query_storage[:, :heads_per_rank]
                assert local_query.is_contiguous() is not noncontiguous

                active_ubatch[0] = iteration % 2
                actual = workspace.gather(local_query)
                torch.accelerator.synchronize()

                assert (
                    actual.data_ptr()
                    == workspace.final_query[active_ubatch[0]].data_ptr()
                )
                assert workspace.completion[active_ubatch[0]].numel() == 1
                _assert_q_gather_matches_reference(
                    actual, local_query, world_size, padded_num_heads
                )

            if env.get("TEST_CUDA_GRAPH") != "1" or dtype != torch.bfloat16:
                continue

            capture_num_tokens = 17
            capture_storage = torch.empty(
                capture_num_tokens,
                heads_per_rank + 2,
                head_dim,
                dtype=dtype,
                device=device,
            )
            captured_input = capture_storage[:, :heads_per_rank]
            assert not captured_input.is_contiguous()
            input_pattern = (
                torch.arange(captured_input.numel(), device=device, dtype=torch.int32)
                .remainder(31)
                .view(captured_input.shape)
                .to(dtype)
            )

            active_ubatch[0] = 1
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_output = workspace.gather(captured_input)
            torch.accelerator.synchronize()
            dist.barrier()

            eager_input = torch.full(
                captured_input.shape,
                96 + rank,
                dtype=dtype,
                device=device,
            )
            eager_output = workspace.gather(eager_input)
            torch.accelerator.synchronize()
            _assert_q_gather_matches_reference(
                eager_output, eager_input, world_size, padded_num_heads
            )
            epoch_before_replays = int(workspace.epoch[1].item())

            for replay in range(2):
                captured_input.copy_(input_pattern + rank * 32 + replay)
                torch.accelerator.synchronize()
                graph.replay()
                torch.accelerator.synchronize()
                assert int(workspace.epoch[1].item()) == (
                    epoch_before_replays + replay + 1
                )
                _assert_q_gather_matches_reference(
                    captured_output,
                    captured_input,
                    world_size,
                    padded_num_heads,
                )

            interleaved_eager_input = torch.full(
                captured_input.shape,
                160 + rank,
                dtype=dtype,
                device=device,
            )
            interleaved_eager_output = workspace.gather(interleaved_eager_input)
            torch.accelerator.synchronize()
            _assert_q_gather_matches_reference(
                interleaved_eager_output,
                interleaved_eager_input,
                world_size,
                padded_num_heads,
            )

            captured_input.copy_(input_pattern + rank * 32 + 2)
            torch.accelerator.synchronize()
            epoch_before_mixed_replay = int(workspace.epoch[1].item())
            graph.replay()
            torch.accelerator.synchronize()
            assert int(workspace.epoch[1].item()) == epoch_before_mixed_replay + 1
            _assert_q_gather_matches_reference(
                captured_output, captured_input, world_size, padded_num_heads
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size",
    [
        pytest.param(
            4,
            marks=pytest.mark.skipif(
                torch.accelerator.device_count() < 4 or not _has_multicast_support(),
                reason="Need 4 GPUs with symmetric-memory multicast.",
            ),
        ),
    ],
)
def test_distributed_direct_q_gather_cuda_graph_replay(world_size: int):
    _distributed_run(
        _distributed_direct_q_gather_worker,
        world_size=world_size,
        extra_env={"TEST_CUDA_GRAPH": "1"},
    )


def _distributed_direct_kv_gather_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        token_dim, plane_split_dim = 576, 512
        max_gathered_tokens = 64
        active_ubatch = [0]
        dcp_utils.dbo_current_ubatch_id = lambda: active_ubatch[0]

        full_layout = (
            [4, 4, 6],
            [4, 0, 0],
            [[8, 8, 7, 6], [4, 2, 2, 2], [6, 6, 5, 4]],
        )
        alternate_layout = (
            [4, 4, 6],
            [4, 0, 0],
            [[8, 8, 8, 6], [3, 2, 2, 2], [6, 6, 5, 4]],
        )
        small_layout = ([4], [4], [[8, 8, 7, 6]])

        def layout_maps(layout) -> tuple[list[list[int]], int]:
            padded_lens, local_starts, context_lens = layout
            maps: list[list[int]] = [[] for _ in range(world_size)]
            output_start = 0
            for padded_len, local_start, request_lens in zip(
                padded_lens, local_starts, context_lens, strict=True
            ):
                valid_lens = [
                    min(max(0, length - local_start), padded_len)
                    for length in request_lens
                ]
                for source_rank, valid_len in enumerate(valid_lens):
                    rank_start = output_start + sum(valid_lens[:source_rank])
                    maps[source_rank].extend(range(rank_start, rank_start + valid_len))
                    maps[source_rank].extend([-1] * (padded_len - valid_len))
                output_start += sum(valid_lens)
            valid_rows = sorted(row for rows in maps for row in rows if row >= 0)
            assert valid_rows == list(range(output_start))
            return maps, output_start

        def make_local(
            source_rank: int,
            iteration: int,
            num_rows: int,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            values = torch.arange(
                num_rows * token_dim,
                dtype=torch.int32,
                device=device,
            )
            values = (values + source_rank * 7 + iteration * 13).remainder(29) - 14
            return values.view(num_rows, token_dim).to(dtype)

        def expected_planes(layout, iteration: int, dtype: torch.dtype):
            maps, output_tokens = layout_maps(layout)
            output = torch.empty(output_tokens, token_dim, dtype=dtype, device=device)
            covered = torch.zeros(output_tokens, dtype=torch.bool, device=device)
            for source_rank, rows in enumerate(maps):
                source = make_local(source_rank, iteration, len(rows), dtype)
                dst_rows = torch.tensor(rows, dtype=torch.int32, device=device)
                valid = dst_rows >= 0
                assert not torch.any(covered[dst_rows[valid]])
                output[dst_rows[valid]] = source[valid]
                covered[dst_rows[valid]] = True
            assert torch.all(covered)
            return (
                output[:, :plane_split_dim].unsqueeze(1),
                output[:, plane_split_dim:].unsqueeze(1),
            )

        def publish(workspace, dtype, iteration: int, buffer_slot: int, layout):
            maps, output_tokens = layout_maps(layout)
            actual = workspace.gather(
                make_local(rank, iteration, len(maps[rank]), dtype),
                torch.tensor(maps[rank], dtype=torch.int32, device=device),
                output_tokens,
                buffer_slot,
            )
            return actual, expected_planes(layout, iteration, dtype)

        def assert_planes(actual, expected) -> None:
            for actual_plane, expected_plane in zip(actual, expected, strict=True):
                assert torch.equal(
                    actual_plane.view(torch.uint8), expected_plane.view(torch.uint8)
                )

        def check_case(
            workspace: dcp_utils.DirectDCPKVGatherWorkspace,
            dtype: torch.dtype,
            iteration: int,
            ubatch: int,
            buffer_slot: int,
            layout,
        ) -> None:
            maps, output_tokens = layout_maps(layout)
            local_kv = make_local(rank, iteration, len(maps[rank]), dtype)
            dst_rows = torch.tensor(maps[rank], dtype=torch.int32, device=device)
            active_ubatch[0] = ubatch

            other_ubatch = 1 - ubatch
            torch.accelerator.synchronize()
            other_ubatch_before = workspace.received_kv[other_ubatch].clone()
            inactive_slot_before = workspace.received_kv[
                ubatch, 1 - buffer_slot
            ].clone()
            slot = workspace.received_kv[ubatch, buffer_slot].view(-1)
            kv_c_capacity = max_gathered_tokens * plane_split_dim
            kv_c_storage = slot[:kv_c_capacity].view(
                max_gathered_tokens, plane_split_dim
            )
            k_pe_storage = slot[kv_c_capacity:].view(
                max_gathered_tokens, token_dim - plane_split_dim
            )
            kv_c_tail_before = kv_c_storage[output_tokens:].clone()
            k_pe_tail_before = k_pe_storage[output_tokens:].clone()
            epochs_before = workspace.epoch.clone()
            torch.accelerator.synchronize()

            kv_c, k_pe = workspace.gather(
                local_kv,
                dst_rows,
                output_tokens,
                buffer_slot,
            )
            torch.accelerator.synchronize()

            expected_kv_c, expected_k_pe = expected_planes(layout, iteration, dtype)
            assert kv_c.shape == expected_kv_c.shape
            assert k_pe.shape == expected_k_pe.shape
            assert kv_c.dtype == k_pe.dtype == dtype
            assert kv_c.is_contiguous() and k_pe.is_contiguous()
            assert torch.equal(kv_c.view(torch.uint8), expected_kv_c.view(torch.uint8))
            assert torch.equal(k_pe.view(torch.uint8), expected_k_pe.view(torch.uint8))
            assert (
                kv_c.data_ptr() == workspace.received_kv[ubatch, buffer_slot].data_ptr()
            )
            assert k_pe.data_ptr() == (
                kv_c.data_ptr() + kv_c_capacity * workspace.received_kv.element_size()
            )
            assert int(workspace.epoch[ubatch].item()) == (
                int(epochs_before[ubatch].item()) + 1
            )
            assert int(workspace.epoch[other_ubatch].item()) == int(
                epochs_before[other_ubatch].item()
            )
            assert not torch.count_nonzero(workspace.completion)
            assert torch.equal(
                workspace.received_kv[other_ubatch].view(torch.uint8),
                other_ubatch_before.view(torch.uint8),
            )
            assert torch.equal(
                workspace.received_kv[ubatch, 1 - buffer_slot].view(torch.uint8),
                inactive_slot_before.view(torch.uint8),
            )
            assert torch.equal(
                kv_c_storage[output_tokens:].view(torch.uint8),
                kv_c_tail_before.view(torch.uint8),
            )
            assert torch.equal(
                k_pe_storage[output_tokens:].view(torch.uint8),
                k_pe_tail_before.view(torch.uint8),
            )
            # The gather itself synchronizes publication, but a faster rank
            # may otherwise start the next test case and multicast into what
            # a slower rank is still validating as this case's inactive slot.
            # Production keeps this ordering through same-stream attention
            # consumption followed by K3's TP-wide rendezvous; the test needs
            # an explicit rank barrier around its host-side assertions.
            dist.barrier()

        # Exercise the ownership contract once in depth for BF16, then retain
        # one byte-exact kernel smoke for each additional supported dtype.
        eager_cases = {
            "bfloat16": (
                (0, 0, 0, full_layout),
                (1, 0, 1, alternate_layout),
                (2, 0, 0, small_layout),
            ),
            "float16": ((3, 0, 0, full_layout),),
            "float8_e4m3fn": ((4, 0, 0, full_layout),),
        }
        for dtype_name, cases in eager_cases.items():
            dtype = _dtype_from_name(dtype_name)
            workspace = dcp_utils.DirectDCPKVGatherWorkspace(
                dist.group.WORLD,
                device,
                max_gathered_tokens,
                token_dim,
                plane_split_dim,
                dtype,
                num_ubatches=2,
            )
            for args in cases:
                check_case(workspace, dtype, *args)

            if dtype == torch.bfloat16:
                active_ubatch[0] = 0

                # A gather on the other slot is the all-rank rendezvous that
                # makes the first slot reusable. Rank 0 deliberately consumes
                # slot 0 late; no explicit barrier protects the critical
                # slot-0 -> slot-1 -> slot-0 sequence.
                first, expected = publish(workspace, dtype, 30, 0, full_layout)
                torch.accelerator.synchronize()
                if rank == 0:
                    time.sleep(0.25)
                assert_planes(first, expected)
                publish(workspace, dtype, 31, 1, alternate_layout)
                torch.accelerator.synchronize()
                reused, expected = publish(workspace, dtype, 32, 0, small_layout)
                torch.accelerator.synchronize()
                assert_planes(reused, expected)
                dist.barrier()  # Isolate the next ownership scenario.

                # Model execution may reset to slot 0 at a layer boundary.
                # Simulate attention consumption followed by its downstream
                # TP collective, then reuse the same slot immediately.
                first, expected = publish(workspace, dtype, 33, 0, full_layout)
                consumed = sum(plane.float().sum() for plane in first)
                expected_consumed = sum(plane.float().sum() for plane in expected)
                dist.all_reduce(consumed)
                reused, expected = publish(workspace, dtype, 34, 0, small_layout)
                torch.accelerator.synchronize()
                torch.testing.assert_close(consumed, expected_consumed * world_size)
                assert_planes(reused, expected)
                dist.barrier()

            if env.get("TEST_CUDA_GRAPH") != "1" or dtype != torch.bfloat16:
                continue

            graph_workspace = dcp_utils.DirectDCPKVGatherWorkspace(
                dist.group.WORLD,
                device,
                max_gathered_tokens,
                token_dim,
                plane_split_dim,
                dtype,
                num_ubatches=2,
            )
            capture_maps, capture_output_tokens = layout_maps(full_layout)
            captured_input = make_local(rank, 20, len(capture_maps[rank]), dtype)
            captured_dst_rows = torch.tensor(
                capture_maps[rank], dtype=torch.int32, device=device
            )
            active_ubatch[0] = 1
            torch.accelerator.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_kv_c, captured_k_pe = graph_workspace.gather(
                    captured_input,
                    captured_dst_rows,
                    capture_output_tokens,
                    buffer_slot=1,
                )
            dist.barrier()
            graph.replay()
            torch.accelerator.synchronize()
            dist.barrier()
            expected_kv_c, expected_k_pe = expected_planes(full_layout, 20, dtype)
            assert torch.equal(
                captured_kv_c.view(torch.uint8), expected_kv_c.view(torch.uint8)
            )
            assert torch.equal(
                captured_k_pe.view(torch.uint8), expected_k_pe.view(torch.uint8)
            )
            dist.barrier()

            # DBO owns a separate pair of buffers per ubatch. An eager call in
            # ubatch 0 must not disturb the captured ubatch-1 slot.
            check_case(
                graph_workspace,
                dtype,
                iteration=21,
                ubatch=0,
                buffer_slot=0,
                layout=small_layout,
            )
            maps, output_tokens = layout_maps(alternate_layout)
            assert output_tokens == capture_output_tokens
            captured_input.copy_(make_local(rank, 22, len(maps[rank]), dtype))
            captured_dst_rows.copy_(
                torch.tensor(maps[rank], dtype=torch.int32, device=device)
            )
            epochs_before = graph_workspace.epoch.clone()
            torch.accelerator.synchronize()
            active_ubatch[0] = 0  # Replay retains the ubatch captured above.
            graph.replay()
            torch.accelerator.synchronize()
            expected_kv_c, expected_k_pe = expected_planes(alternate_layout, 22, dtype)
            assert torch.equal(
                captured_kv_c.view(torch.uint8), expected_kv_c.view(torch.uint8)
            )
            assert torch.equal(
                captured_k_pe.view(torch.uint8), expected_k_pe.view(torch.uint8)
            )
            assert int(graph_workspace.epoch[1].item()) == (
                int(epochs_before[1].item()) + 1
            )
            assert int(graph_workspace.epoch[0].item()) == int(epochs_before[0].item())
            assert not torch.count_nonzero(graph_workspace.completion)
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size",
    [
        pytest.param(
            4,
            marks=pytest.mark.skipif(
                torch.accelerator.device_count() < 4 or not _has_multicast_support(),
                reason="Need 4 GPUs with symmetric-memory multicast.",
            ),
        ),
    ],
)
def test_distributed_direct_kv_gather_matches_reference(world_size: int):
    _distributed_run(
        _distributed_direct_kv_gather_worker,
        world_size=world_size,
        extra_env={"TEST_CUDA_GRAPH": "1"},
    )


def _distributed_direct_a2a_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dtype = _dtype_from_name(env["TEST_DTYPE"])
        lse_dtype = _dtype_from_name(env["TEST_LSE_DTYPE"])
        is_lse_base_on_e = env["LSE_BASE_E"] == "1"
        # Match Kimi-K3's six heads per DCP rank.
        heads_per_rank, head_dim, max_num_tokens = 6, 512, 128
        total_heads = world_size * heads_per_rank
        active_ubatch = [0]
        dcp_utils.dbo_current_ubatch_id = lambda: active_ubatch[0]
        workspace = dcp_utils.DirectDCPA2AWorkspace(
            dist.group.WORLD,
            device,
            max_num_tokens,
            heads_per_rank,
            head_dim,
            dtype,
            num_ubatches=2,
        )

        def check(num_tokens: int, iteration: int, padded: bool) -> None:
            generator = torch.Generator(device=device)
            generator.manual_seed(1234 + rank + iteration * world_size)
            storage_heads = 128 if padded else total_heads
            partial_output_storage = torch.randn(
                num_tokens,
                storage_heads,
                head_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            partial_lse_storage = torch.randn(
                num_tokens,
                storage_heads,
                device=device,
                dtype=lse_dtype,
                generator=generator,
            )
            partial_output = partial_output_storage[:, :total_heads, :]
            partial_lse = partial_lse_storage[:, :total_heads]
            if padded:
                assert not partial_output.is_contiguous()
                assert not partial_lse.is_contiguous()
            active_ubatch[0] = iteration % 2
            actual = workspace.lse_reduce(partial_output, partial_lse, is_lse_base_on_e)
            torch.accelerator.synchronize()

            reference_output = partial_output.contiguous()
            reference_lse = partial_lse.contiguous()
            gathered_output = [
                torch.empty_like(reference_output) for _ in range(world_size)
            ]
            gathered_lse = [torch.empty_like(reference_lse) for _ in range(world_size)]
            dist.all_gather(gathered_output, reference_output)
            dist.all_gather(gathered_lse, reference_lse)
            outputs = torch.stack(
                [
                    value[
                        :,
                        rank * heads_per_rank : (rank + 1) * heads_per_rank,
                        :,
                    ]
                    for value in gathered_output
                ]
            ).float()
            lses = torch.stack(
                [
                    value[:, rank * heads_per_rank : (rank + 1) * heads_per_rank]
                    for value in gathered_lse
                ]
            )
            expected = _lse_weighted_combine(
                outputs, lses, is_lse_base_on_e=is_lse_base_on_e
            )
            _assert_close(actual, expected, dtype)

        def check_empty_shards(query_lens: list[int], iteration: int) -> None:
            """Verify empty local shards contribute zero weight."""
            num_seqs = len(query_lens)
            num_tokens = sum(query_lens)
            generator = torch.Generator(device=device)
            generator.manual_seed(777 + rank + iteration * world_size)
            partial_output = torch.randn(
                num_tokens,
                total_heads,
                head_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            partial_lse = torch.randn(
                num_tokens,
                total_heads,
                device=device,
                dtype=lse_dtype,
                generator=generator,
            )

            # Cover globally empty, rank-local empty, and non-empty sequences.
            def is_empty(seq_idx: int, source_rank: int) -> bool:
                return seq_idx == 0 or seq_idx == 1 + source_rank

            seq_lens = torch.tensor(
                [
                    0 if is_empty(seq_idx, rank) else seq_idx + 3
                    for seq_idx in range(num_seqs)
                ],
                dtype=torch.int32,
                device=device,
            )
            query_lens_tensor = torch.tensor(
                query_lens,
                dtype=torch.int32,
                device=device,
            )
            query_start_loc = torch.cat(
                (
                    query_lens_tensor.new_zeros(1),
                    query_lens_tensor.cumsum(0),
                )
            )
            empty_rows = torch.repeat_interleave(seq_lens == 0, query_lens_tensor)
            # Model undefined attention rows with NaNs.
            partial_output[empty_rows] = float("nan")
            partial_lse[empty_rows] = float("nan")

            active_ubatch[0] = iteration % 2
            actual = workspace.lse_reduce(
                partial_output,
                partial_lse,
                is_lse_base_on_e,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
            )
            torch.accelerator.synchronize()

            gathered_output = [
                torch.empty_like(partial_output) for _ in range(world_size)
            ]
            gathered_lse = [torch.empty_like(partial_lse) for _ in range(world_size)]
            dist.all_gather(gathered_output, partial_output.contiguous())
            dist.all_gather(gathered_lse, partial_lse.contiguous())
            head_slice = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)
            outputs = torch.stack(
                [value[:, head_slice, :] for value in gathered_output]
            ).float()
            lses = torch.stack([value[:, head_slice] for value in gathered_lse])
            for source_rank in range(world_size):
                source_empty = torch.tensor(
                    [is_empty(seq_idx, source_rank) for seq_idx in range(num_seqs)],
                    device=device,
                ).repeat_interleave(query_lens_tensor)
                outputs[source_rank][source_empty] = 0.0
                lses[source_rank][source_empty] = float("-inf")
            expected = _lse_weighted_combine(
                outputs, lses, is_lse_base_on_e=is_lse_base_on_e
            )
            all_empty = torch.tensor(
                [
                    all(is_empty(seq_idx, r) for r in range(world_size))
                    for seq_idx in range(num_seqs)
                ],
                device=device,
            ).repeat_interleave(query_lens_tensor)
            assert torch.equal(actual[all_empty], torch.zeros_like(actual[all_empty]))
            assert not torch.isnan(actual.float()).any()
            _assert_close(actual, expected, dtype)

        cases = ((1, False), (17, True), (128, True))
        for iteration, (num_tokens, padded) in enumerate(cases):
            check(num_tokens, iteration, padded)
        check_empty_shards(
            query_lens=[1, 3, 2, *([1] * (world_size - 1))],
            iteration=len(cases),
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(4321 + rank)
        partial_output_storage = torch.randn(
            128,
            128,
            head_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        partial_lse_storage = torch.randn(
            128,
            128,
            device=device,
            dtype=lse_dtype,
            generator=generator,
        )
        partial_output = partial_output_storage[:, :total_heads, :]
        partial_lse = partial_lse_storage[:, :total_heads]
        assert not partial_output.is_contiguous()
        assert not partial_lse.is_contiguous()
        torch.accelerator.synchronize()
        active_ubatch[0] = 1
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = workspace.lse_reduce(partial_output, partial_lse, is_lse_base_on_e)
        for _ in range(3):
            graph.replay()
        torch.accelerator.synchronize()

        reference_output = partial_output.contiguous()
        reference_lse = partial_lse.contiguous()
        gathered_output = [
            torch.empty_like(reference_output) for _ in range(world_size)
        ]
        gathered_lse = [torch.empty_like(reference_lse) for _ in range(world_size)]
        dist.all_gather(gathered_output, reference_output)
        dist.all_gather(gathered_lse, reference_lse)
        head_slice = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)
        outputs = torch.stack(
            [value[:, head_slice, :] for value in gathered_output]
        ).float()
        lses = torch.stack([value[:, head_slice] for value in gathered_lse])
        expected = _lse_weighted_combine(
            outputs, lses, is_lse_base_on_e=is_lse_base_on_e
        )
        _assert_close(actual, expected, dtype)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size",
    [
        pytest.param(
            2,
            marks=pytest.mark.skipif(
                torch.accelerator.device_count() < 2, reason="Need at least 2 GPUs."
            ),
        ),
        pytest.param(
            4,
            marks=pytest.mark.skipif(
                torch.accelerator.device_count() < 4, reason="Need at least 4 GPUs."
            ),
        ),
    ],
)
def test_distributed_direct_a2a_matches_reference(world_size: int):
    _distributed_run(
        _distributed_direct_a2a_worker,
        world_size=world_size,
        extra_env={
            "TEST_DTYPE": "bfloat16",
            "TEST_LSE_DTYPE": "bfloat16",
            "LSE_BASE_E": "0",
        },
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
