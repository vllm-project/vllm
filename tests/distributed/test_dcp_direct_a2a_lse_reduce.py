# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for direct symmetric-memory DCP collectives."""

import functools
from unittest.mock import MagicMock

import multiprocess as mp
import pytest
import torch
import torch.distributed as dist

import vllm.v1.attention.ops.cp_common as cp_common
import vllm.v1.attention.ops.dcp as dcp
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)


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

    for process in processes:
        process.join(timeout=120)

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
    @pytest.fixture(autouse=True)
    def supported_topology(self, monkeypatch):
        monkeypatch.setattr(
            cp_common,
            "get_symmetric_memory_topology",
            lambda _group: cp_common.SymmetricMemoryTopology.INTRA_NODE,
        )

    def test_env_disabled_returns_none(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_A2A", "0")
        dcp.get_direct_dcp_a2a_workspace.cache_clear()
        workspace = dcp.get_direct_dcp_a2a_workspace(
            _FakeGroupCoordinator(), torch.device("cpu"), 16, 2, 32, torch.bfloat16, 1
        )
        assert workspace is None

    def test_forced_with_unsupported_dtype_raises(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_A2A", "1")
        dcp.get_direct_dcp_a2a_workspace.cache_clear()
        with pytest.raises(ValueError, match="does not support"):
            dcp.get_direct_dcp_a2a_workspace(
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
            dcp.DirectDCPA2AWorkspace(
                None, torch.device("cpu"), 16, 2, 32, torch.bfloat16, num_ubatches=0
            )

    def test_auto_with_unsupported_dtype_returns_none(self, monkeypatch):
        monkeypatch.delenv("VLLM_USE_DIRECT_DCP_A2A", raising=False)
        dcp.get_direct_dcp_a2a_workspace.cache_clear()
        workspace = dcp.get_direct_dcp_a2a_workspace(
            _FakeGroupCoordinator(), torch.device("cpu"), 16, 2, 32, torch.float32, 1
        )
        assert workspace is None

    def test_q_gather_env_disabled_returns_none(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_Q_GATHER", "0")
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_A2A", "1")
        dcp.get_direct_dcp_q_gather_workspace.cache_clear()
        workspace = dcp.get_direct_dcp_q_gather_workspace(
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
        monkeypatch.setattr(cp_common, "_symm_mem_spans_group", lambda group: True)
        dcp.get_direct_dcp_q_gather_workspace.cache_clear()
        workspace = object()
        init_workspace = MagicMock(return_value=workspace)
        monkeypatch.setattr(
            dcp,
            "DirectDCPQGatherWorkspace",
            init_workspace,
        )

        result = dcp.get_direct_dcp_q_gather_workspace(
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
        dcp.get_direct_dcp_kv_gather_workspace.cache_clear()
        workspace = dcp.get_direct_dcp_kv_gather_workspace(
            _FakeGroupCoordinator(), torch.device("cpu"), 64, 576, torch.bfloat16, 1
        )
        assert workspace is None

    def test_kv_gather_flag_is_independent(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_KV_GATHER", "1")
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_A2A", "0")
        monkeypatch.setattr(cp_common, "_symm_mem_spans_group", lambda group: True)
        dcp.get_direct_dcp_kv_gather_workspace.cache_clear()
        workspace = object()
        init_workspace = MagicMock(return_value=workspace)
        monkeypatch.setattr(
            dcp,
            "DirectDCPKVGatherWorkspace",
            init_workspace,
        )

        result = dcp.get_direct_dcp_kv_gather_workspace(
            _FakeGroupCoordinator(), torch.device("cpu"), 64, 576, torch.bfloat16, 1
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
                (64, 576, torch.bfloat16, 1),
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
        factory = getattr(dcp, factory_name)
        monkeypatch.setenv(flag_name, "1")
        monkeypatch.setattr(cp_common, "_symm_mem_spans_group", lambda group: False)
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
            dcp.DirectDCPKVGatherWorkspace(
                None, torch.device("cpu"), 64, 576, num_ubatches=0
            )
        with pytest.raises(ValueError, match="divide evenly"):
            dcp.DirectDCPKVGatherWorkspace(
                _FakeProcessGroup(), torch.device("cpu"), 63, 576
            )
        with pytest.raises(ValueError, match="16-byte"):
            dcp.DirectDCPKVGatherWorkspace(
                _FakeProcessGroup(), torch.device("cpu"), 64, 3
            )

    def test_q_gather_rejects_invalid_workspace_geometry(self):
        with pytest.raises(ValueError, match="ubatch"):
            dcp.DirectDCPQGatherWorkspace(
                None, torch.device("cpu"), 16, 2, 32, num_ubatches=0
            )
        with pytest.raises(ValueError, match="padded heads"):
            dcp.DirectDCPQGatherWorkspace(
                _FakeProcessGroup(),
                torch.device("cpu"),
                16,
                2,
                32,
                padded_num_heads=7,
            )
        with pytest.raises(ValueError, match="16-byte"):
            dcp.DirectDCPQGatherWorkspace(
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
    import vllm.v1.attention.ops.dcp as dcp_manager

    group = MagicMock(world_size=2)
    monkeypatch.setattr(dcp_manager, "get_dcp_group", lambda: group)
    direct_a2a = MagicMock(max_num_tokens=16)
    direct_query = MagicMock(max_num_tokens=16)
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
    workspace = torch.empty(96, 8)

    assert manager.v_up_proj_combine is None
    # Queries that fit the direct workspace go through it.
    query = torch.empty(1, 2, 8)
    assert manager.query_gather is not None
    assert manager.query_gather(query) is direct_query.gather.return_value
    direct_query.gather.assert_called_once_with(query)
    manager.init_kv_gather(workspace, 64)
    gathered_kv, local_kv = torch.empty(4, 8), torch.empty(2, 8)
    manager.kv_gather(gathered_kv, local_kv)
    direct_kv.gather.assert_called_once_with(gathered_kv, local_kv)
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
        output, lse, False, seq_lens, query_start_loc
    )


def test_mla_dcp_manager_selects_fallback_backends(monkeypatch):
    import vllm.v1.attention.ops.dcp as dcp_manager

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
    workspace = torch.empty(96, 8)
    manager.init_kv_gather(workspace, 64)
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
    assert isinstance(manager.v_up_proj_combine, functools.partial)
    assert manager.v_up_proj_combine.func is dcp_manager.dcp_a2a_v_up_proj_lse_reduce
    assert manager.v_up_proj_combine.keywords == {
        "cp_group": group,
        "is_lse_base_on_e": True,
    }


# False: a diffusion canvas of 3 without a speculative config.
@pytest.mark.parametrize("with_spec_config", [True, False])
def test_dcp_workspace_covers_decode_width(with_spec_config):
    config = _manager_config()
    config.scheduler_config.max_num_batched_tokens = 128
    config.num_speculative_tokens = 3
    if with_spec_config:
        config.speculative_config = MagicMock(
            num_speculative_tokens=3, max_num_new_slots_for_drafting=0
        )

    assert dcp.get_dcp_workspace_max_num_tokens(config) == 16


def test_mla_dcp_manager_selects_pcp_combine(monkeypatch):
    import vllm.v1.attention.ops.dcp as dcp_manager

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
    assert manager.v_up_proj_combine is None


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


def test_sparse_mla_builder_initializes_dcp_manager(monkeypatch):
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

    manager = object.__new__(dcp.MLADCPManager)
    manager.init_kv_gather = MagicMock()
    layer = MagicMock(dcp_manager=manager, use_shardq=False)
    config = MagicMock()
    config.model_config.dtype = torch.bfloat16
    config.model_config.max_model_len = 64
    config.model_config.hf_text_config.index_topk = 8
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
        builder.chunked_prefill_workspace,
        builder.chunked_prefill_workspace_size,
    )


def test_sparse_mla_workspace_preserves_non_dcp_size():
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonMetadataBuilder,
    )

    config = MagicMock()
    config.model_config.max_model_len = 1
    config.model_config.hf_text_config.index_topk = 7
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
        dcp.dbo_current_ubatch_id = lambda: active_ubatch[0]
        for dtype_idx, dtype_name in enumerate(
            ("bfloat16", "float8_e4m3fn", "float32")
        ):
            dtype = _dtype_from_name(dtype_name)
            workspace = dcp.DirectDCPQGatherWorkspace(
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
        token_dim = 576
        max_gathered_tokens = 128 * world_size
        active_ubatch = [0]
        dcp.dbo_current_ubatch_id = lambda: active_ubatch[0]

        for dtype_idx, dtype_name in enumerate(("bfloat16", "float16")):
            dtype = _dtype_from_name(dtype_name)
            workspace = dcp.DirectDCPKVGatherWorkspace(
                dist.group.WORLD,
                device,
                max_gathered_tokens,
                token_dim,
                dtype,
                num_ubatches=2,
            )

            # Use disjoint slices of one persistent chunked-context workspace.
            storage = torch.zeros(
                (world_size + 1) * 128, token_dim, device=device, dtype=dtype
            )
            for iteration, num_tokens in enumerate((1, 128, 17)):
                generator = torch.Generator(device=device)
                generator.manual_seed(7000 + rank * 101 + dtype_idx * 977 + iteration)
                local_kv = storage[:num_tokens]
                local_kv.copy_(
                    torch.randn(
                        num_tokens,
                        token_dim,
                        device=device,
                        dtype=torch.float32,
                        generator=generator,
                    ).to(dtype)
                )
                gathered = storage[128 : 128 + num_tokens * world_size]
                active_ubatch[0] = iteration % 2
                workspace.gather(gathered, local_kv)
                torch.accelerator.synchronize()

                expected = torch.empty_like(gathered)
                dist.all_gather_into_tensor(expected, local_kv.contiguous())
                assert torch.equal(
                    gathered.view(torch.uint8), expected.view(torch.uint8)
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
def test_distributed_direct_kv_gather_matches_reference(world_size: int):
    _distributed_run(
        _distributed_direct_kv_gather_worker,
        world_size=world_size,
        extra_env={},
    )


# CuTe DSL direct A2A on one GPU: every simulated DCP rank owns ordinary device
# buffers laid out like DirectDCPA2AWorkspace's symmetric memory, and the peer
# pointer tables point at each other's buffers. The CuTe kernels run rank by
# rank (all pushes, then all combines); the CUDA op runs one stream per rank,
# at sizes where every spinning combine CTA stays resident.

_requires_cutedsl_a2a = pytest.mark.skipif(
    not dcp._use_cutedsl_direct_a2a(2, 1, 512, torch.bfloat16),
    reason="CuTe DSL direct DCP A2A requires SM100 and CuTe DSL",
)
_A2A_HEAD_DIM = 512
_A2A_MAX_NUM_TOKENS = 256
# GPU cycles (tens of ms) that hold the simulated ranks' CUDA op streams
# while the host queues their launches.
_LAUNCH_GATE_CYCLES = 50_000_000


class _SimulatedA2ARanks:
    def __init__(self, world_size: int, heads_per_rank: int) -> None:
        self.world_size = world_size
        self.heads_per_rank = heads_per_rank
        shape = (2, world_size, _A2A_MAX_NUM_TOKENS, heads_per_rank)

        def zeros(*size, dtype):
            return [
                torch.zeros(*size, dtype=dtype, device="cuda")
                for _ in range(world_size)
            ]

        self.received_output = zeros(*shape, _A2A_HEAD_DIM, dtype=torch.bfloat16)
        self.received_lse = zeros(*shape, dtype=torch.float32)
        self.received_signal = zeros(2, world_size, dtype=torch.int32)
        self.epoch = zeros(1, dtype=torch.int64)
        self.counters = zeros(world_size + 1, dtype=torch.int32)

        def table(buffers: list[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [buffer.data_ptr() for buffer in buffers],
                dtype=torch.int64,
                device="cuda",
            )

        self.peer_output_ptrs = table(self.received_output)
        self.peer_lse_ptrs = table(self.received_lse)
        self.peer_signal_ptrs = table(self.received_signal)

    def outputs(self, num_tokens: int) -> list[torch.Tensor]:
        return [
            torch.full(
                (num_tokens, self.heads_per_rank, _A2A_HEAD_DIM),
                float("nan"),
                dtype=torch.bfloat16,
                device="cuda",
            )
            for _ in range(self.world_size)
        ]

    def cutedsl(
        self, outputs, partial_outputs, partial_lses, base_e, seq_lens=None, qsl=None
    ):
        from vllm.v1.attention.ops import dcp_direct_a2a_cutedsl as kernels

        # The push of every rank completes before any combine waits on it.
        for rank in range(self.world_size):
            counters = self.counters[rank]
            rank_seq_lens = None if seq_lens is None else seq_lens[rank]
            kernels.DIRECT_DCP_A2A_PUSH_KERNEL(
                partial_outputs[rank],
                partial_lses[rank],
                counters[:1] if rank_seq_lens is None else rank_seq_lens,
                counters[:1] if qsl is None else qsl,
                0 if rank_seq_lens is None else rank_seq_lens.shape[0],
                self.peer_output_ptrs,
                self.peer_lse_ptrs,
                self.peer_signal_ptrs,
                counters,
                self.epoch[rank],
                world_size=self.world_size,
                rank=rank,
                max_num_tokens=_A2A_MAX_NUM_TOKENS,
            )
        for rank in range(self.world_size):
            kernels.DIRECT_DCP_A2A_COMBINE_KERNEL(
                self.received_output[rank],
                self.received_lse[rank],
                self.received_signal[rank],
                self.epoch[rank],
                self.counters[rank][self.world_size :],
                outputs[rank],
                world_size=self.world_size,
                max_num_tokens=_A2A_MAX_NUM_TOKENS,
                lse_scale=kernels._LOG2_E if base_e else 1.0,
            )

    def cuda(
        self, outputs, partial_outputs, partial_lses, base_e, seq_lens=None, qsl=None
    ):
        # One stream per rank. All ranks share one GPU, so a rank queued after
        # the others' spinning combines could find no free SM: hold every
        # stream until all launches are queued.
        current = torch.cuda.current_stream()
        torch.cuda._sleep(_LAUNCH_GATE_CYCLES)
        streams = [torch.cuda.Stream() for _ in range(self.world_size)]
        for rank, stream in enumerate(streams):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                torch.ops._C.direct_dcp_a2a_lse_reduce(
                    partial_outputs[rank],
                    partial_lses[rank],
                    None if seq_lens is None else seq_lens[rank],
                    qsl,
                    self.peer_output_ptrs,
                    self.peer_lse_ptrs,
                    self.peer_signal_ptrs,
                    self.received_output[rank],
                    self.received_lse[rank],
                    self.received_signal[rank],
                    self.epoch[rank],
                    outputs[rank],
                    self.world_size,
                    rank,
                    _A2A_MAX_NUM_TOKENS,
                    base_e,
                )
        for stream in streams:
            current.wait_stream(stream)


def _a2a_inputs(
    world_size, heads_per_rank, num_tokens, lse_dtype, padded, transposed, seed
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total_heads = world_size * heads_per_rank
    storage_heads = max(128, total_heads) if padded else total_heads
    partial_outputs, partial_lses = [], []
    for _ in range(world_size):
        output = torch.randn(
            num_tokens,
            storage_heads,
            _A2A_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )[:, :total_heads]
        lse = 4 * torch.randn(
            num_tokens, storage_heads, device="cuda", generator=generator
        )
        lse = lse.to(lse_dtype)[:, :total_heads]
        if transposed:
            # FlashAttention MLA returns [H, T], viewed as [T, H].
            storage = torch.empty(
                total_heads, num_tokens, dtype=lse_dtype, device="cuda"
            )
            storage.copy_(lse.t())
            lse = storage.t()
        # Undefined rows (NaN or +inf LSE) get zero weight, so their partials
        # (NaN here) must never be read.
        lse[0, 0] = float("nan")
        lse[-1, -1] = float("inf")
        output[0, 0] = float("nan")
        output[-1, -1] = float("nan")
        partial_outputs.append(output)
        partial_lses.append(lse)
    return partial_outputs, partial_lses


@_requires_cutedsl_a2a
@pytest.mark.parametrize("world_size,heads_per_rank", [(8, 12), (2, 64)])
@pytest.mark.parametrize(
    "num_tokens,lse_dtype,padded,transposed,base_e",
    [
        (1, torch.float32, False, False, False),
        (13, torch.bfloat16, True, True, True),
        (16, torch.float16, False, True, False),
    ],
)
def test_cutedsl_a2a_bit_exact_vs_cuda_op(
    world_size, heads_per_rank, num_tokens, lse_dtype, padded, transposed, base_e
):
    ranks_cute = _SimulatedA2ARanks(world_size, heads_per_rank)
    ranks_cuda = _SimulatedA2ARanks(world_size, heads_per_rank)
    for call in range(3):
        partial_outputs, partial_lses = _a2a_inputs(
            world_size,
            heads_per_rank,
            num_tokens,
            lse_dtype,
            padded,
            transposed,
            seed=call,
        )
        expected = ranks_cuda.outputs(num_tokens)
        actual = ranks_cute.outputs(num_tokens)
        ranks_cuda.cuda(expected, partial_outputs, partial_lses, base_e)
        ranks_cute.cutedsl(actual, partial_outputs, partial_lses, base_e)
        torch.accelerator.synchronize()
        for rank in range(world_size):
            torch.testing.assert_close(actual[rank], expected[rank], rtol=0, atol=0)
    # Two launches per call leave the protocol state exactly as the CUDA op.
    for rank in range(world_size):
        assert ranks_cute.epoch[rank].item() == 3
        assert torch.equal(
            ranks_cute.received_signal[rank], ranks_cuda.received_signal[rank]
        )
        assert not ranks_cute.counters[rank].any()


@_requires_cutedsl_a2a
def test_cutedsl_a2a_empty_shards_get_zero_weight():
    world_size, heads_per_rank = 8, 12
    query_lens = [1, 3, 2, *([1] * (world_size - 1))]
    num_seqs, num_tokens = len(query_lens), sum(query_lens)
    query_lens_t = torch.tensor(query_lens, dtype=torch.int32, device="cuda")
    query_start_loc = torch.cat(
        (query_lens_t.new_zeros(1), query_lens_t.cumsum(0, dtype=torch.int32))
    )

    # Sequence 0 is empty everywhere; sequence 1 + r is empty on rank r only.
    def is_empty(seq_idx: int, rank: int) -> bool:
        return seq_idx == 0 or seq_idx == 1 + rank

    seq_lens = [
        torch.tensor(
            [0 if is_empty(s, rank) else s + 3 for s in range(num_seqs)],
            dtype=torch.int32,
            device="cuda",
        )
        for rank in range(world_size)
    ]
    empty_rows = [
        torch.repeat_interleave(lengths == 0, query_lens_t) for lengths in seq_lens
    ]
    partial_outputs, partial_lses = _a2a_inputs(
        world_size, heads_per_rank, num_tokens, torch.float32, False, False, seed=3
    )
    for rows, output, lse in zip(empty_rows, partial_outputs, partial_lses):
        # Garbage the kernels must mask from seq_lens alone.
        output[rows] = float("nan")
        lse[rows] = 30.0

    ranks_cute = _SimulatedA2ARanks(world_size, heads_per_rank)
    ranks_cuda = _SimulatedA2ARanks(world_size, heads_per_rank)
    actual = ranks_cute.outputs(num_tokens)
    expected = ranks_cuda.outputs(num_tokens)
    ranks_cute.cutedsl(
        actual, partial_outputs, partial_lses, True, seq_lens, query_start_loc
    )
    ranks_cuda.cuda(
        expected, partial_outputs, partial_lses, True, seq_lens, query_start_loc
    )
    torch.accelerator.synchronize()
    for rank in range(world_size):
        torch.testing.assert_close(actual[rank], expected[rank], rtol=0, atol=0)
        # Sequence 0 is empty on every rank: all weights are zero.
        assert not actual[rank][: query_lens[0]].any()


def _distributed_direct_a2a_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        from vllm.v1.attention.ops.dcp import _lse_weighted_combine

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dtype = _dtype_from_name(env["TEST_DTYPE"])
        lse_dtype = _dtype_from_name(env["TEST_LSE_DTYPE"])
        is_lse_base_on_e = env["LSE_BASE_E"] == "1"
        # Match Kimi-K3's six heads per DCP rank.
        heads_per_rank, head_dim, max_num_tokens = 6, 512, 128
        total_heads = world_size * heads_per_rank
        active_ubatch = [0]
        dcp.dbo_current_ubatch_id = lambda: active_ubatch[0]
        workspace = dcp.DirectDCPA2AWorkspace(
            dist.group.WORLD,
            device,
            max_num_tokens,
            heads_per_rank,
            head_dim,
            dtype,
            num_ubatches=2,
        )
        # "mixed" alternates the CUDA op and the CuTe DSL kernels on the same
        # symmetric buffers: both follow one epoch and signal protocol.
        impl = env["TEST_IMPL"]
        cutedsl_counters = workspace.cutedsl_counters
        assert impl == "cuda" or cutedsl_counters is not None
        num_calls = [0]

        def lse_reduce(*args, **kwargs):
            use_cutedsl = impl == "cutedsl" or (impl == "mixed" and num_calls[0] % 2)
            workspace.cutedsl_counters = cutedsl_counters if use_cutedsl else None
            num_calls[0] += 1
            return workspace.lse_reduce(*args, **kwargs)

        def lse_layout(value: torch.Tensor) -> torch.Tensor:
            if env["TEST_LSE_LAYOUT"] == "transposed":
                # FlashAttention MLA returns [H,T], viewed as [T,H]. Allocate
                # explicitly to preserve its strides even when T == 1.
                storage = torch.empty(
                    value.shape[1], value.shape[0], device=device, dtype=lse_dtype
                )
                storage.copy_(value.transpose(0, 1))
                value = storage.transpose(0, 1)
                assert value.stride() == (1, value.shape[0])
            return value

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
            partial_lse = lse_layout(partial_lse)
            active_ubatch[0] = iteration % 2
            actual = lse_reduce(partial_output, partial_lse, is_lse_base_on_e)
            torch.accelerator.synchronize()

            # Layout alone must not alter even one output bit.
            packed = lse_reduce(
                partial_output, partial_lse.contiguous(), is_lse_base_on_e
            )
            torch.testing.assert_close(actual, packed, rtol=0, atol=0)

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
            partial_lse = lse_layout(partial_lse)

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
                    query_lens_tensor.cumsum(0, dtype=torch.int32),
                )
            )
            empty_rows = torch.repeat_interleave(seq_lens == 0, query_lens_tensor)
            # Model undefined attention rows with NaNs.
            partial_output[empty_rows] = float("nan")
            partial_lse[empty_rows] = float("nan")

            active_ubatch[0] = iteration % 2
            actual = lse_reduce(
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
            gathered_lse = [
                torch.empty_like(partial_lse, memory_format=torch.contiguous_format)
                for _ in range(world_size)
            ]
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
        partial_lse = lse_layout(partial_lse)
        torch.accelerator.synchronize()
        active_ubatch[0] = 1
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = lse_reduce(partial_output, partial_lse, is_lse_base_on_e)
        for _ in range(3):
            graph.replay()
        torch.accelerator.synchronize()

        packed = lse_reduce(partial_output, partial_lse.contiguous(), is_lse_base_on_e)
        torch.testing.assert_close(actual, packed, rtol=0, atol=0)

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
@pytest.mark.parametrize(
    "lse_layout,lse_dtype,lse_base_e",
    [("packed", "bfloat16", "0"), ("transposed", "float32", "1")],
)
@pytest.mark.parametrize(
    "impl",
    [
        "cuda",
        pytest.param("cutedsl", marks=_requires_cutedsl_a2a),
        pytest.param("mixed", marks=_requires_cutedsl_a2a),
    ],
)
def test_distributed_direct_a2a_matches_reference(
    world_size: int, lse_layout: str, lse_dtype: str, lse_base_e: str, impl: str
):
    """Combine packed and FlashAttention MLA LSE, including graph replay."""
    _distributed_run(
        _distributed_direct_a2a_worker,
        world_size=world_size,
        extra_env={
            "TEST_DTYPE": "bfloat16",
            "TEST_LSE_DTYPE": lse_dtype,
            "TEST_LSE_LAYOUT": lse_layout,
            "LSE_BASE_E": lse_base_e,
            "TEST_IMPL": impl,
        },
    )


def _distributed_direct_dcp_topk_worker(env: dict[str, str]) -> None:
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )
    from vllm.model_executor.kernels.attention.dsa.dcp_topk_direct import (
        DirectDCPTopKWorkspace,
    )

    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        topk, max_rows, local_len = 2048, 64, 3000
        workspace = DirectDCPTopKWorkspace(dist.group.WORLD, device, max_rows, topk)

        def make_inputs(rows: int, seed: int, valid: int):
            generator = torch.Generator(device=device)
            generator.manual_seed(seed * 101 + rank)
            logits = torch.randn(
                rows, local_len, device=device, generator=generator
            ).round_(decimals=1)
            local = logits.topk(topk, dim=1).indices.to(torch.int32)
            local[:, valid:] = -1
            return logits, local

        def reference(logits: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
            rows = local.shape[0]
            packed = torch.empty((rows, topk, 2), device=device)
            pack_dcp_topk_candidates_cutedsl(
                logits, local, packed, rank, world_size, 1, None
            )
            gathered = torch.empty((world_size, rows, topk, 2), device=device)
            dist.all_gather_into_tensor(gathered, packed)
            out = torch.empty((rows, topk), device=device, dtype=torch.int32)
            # The selector reads the rank-major all-gather layout as is.
            stable_topk_from_gathered_candidates_cutedsl(gathered, topk, out=out)
            return out

        def check(actual: torch.Tensor, expected: torch.Tensor) -> None:
            for row in range(actual.shape[0]):
                assert sorted(actual[row].tolist()) == sorted(expected[row].tolist())
            peers = [torch.empty_like(actual) for _ in range(world_size)]
            dist.all_gather(peers, actual)
            for peer in peers:
                assert torch.equal(peer.sort(dim=1).values, actual.sort(dim=1).values)

        # An odd number of exchanges covers both ping-pong inboxes; the
        # rounded scores create top-k boundary ties; valid=200 fits in topk.
        for it, (rows, valid) in enumerate(
            ((1, topk), (64, topk), (17, topk), (3, 200), (64, 1000))
        ):
            logits, local = make_inputs(rows, it, valid)
            expected = reference(logits, local)
            actual = local.clone()
            workspace.merge(logits, actual, 1, None)
            check(actual, expected)

        rows = 17
        static_logits, static_local = make_inputs(rows, 100, topk)
        static_topk = static_local.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            workspace.merge(static_logits, static_topk, 1, None)
        for it in range(3):
            logits, local = make_inputs(rows, 200 + it, topk)
            expected = reference(logits, local)
            static_logits.copy_(logits)
            static_topk.copy_(local)
            graph.replay()
            check(static_topk, expected)
        torch.accelerator.synchronize()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4 or not torch.cuda.is_available(),
    reason="Need 4 GPUs.",
)
def test_distributed_direct_dcp_topk_matches_all_gather():
    _distributed_run(_distributed_direct_dcp_topk_worker, world_size=4, extra_env={})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
