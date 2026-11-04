# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for direct symmetric-memory DCP collectives."""

import functools
from types import SimpleNamespace
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


def test_nixl_setup_propagates_empty_exception(monkeypatch):
    from vllm.v1.attention.ops.nixl_dcp import NixlDCPA2AWorkspace

    def empty_cuda_failure():
        raise RuntimeError

    def gather_peer_error(errors, error, *, group):
        errors[:] = [error, None]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", empty_cuda_failure)
    monkeypatch.setattr(dist, "all_gather_object", gather_peer_error)
    group = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=object())
    with pytest.raises(RuntimeError, match="setup failed across ranks"):
        NixlDCPA2AWorkspace(group, torch.device("cuda:0"), 1, 1, 512, torch.bfloat16)


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


@pytest.mark.parametrize("direct", [None, False])
def test_mla_nixl_selection_overrides_auto_and_reserves_prefill(monkeypatch, direct):
    from vllm.v1.attention.ops import nixl_dcp

    monkeypatch.setattr(dcp.envs, "VLLM_USE_NIXL_DCP_A2A", True)
    monkeypatch.setattr(dcp.envs, "VLLM_USE_DIRECT_DCP_A2A", direct)
    group = MagicMock(world_size=2)
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: group)
    monkeypatch.setattr(dcp, "get_dcp_workspace_max_num_tokens", lambda _: 4)
    monkeypatch.setattr(dcp, "get_direct_dcp_q_gather_workspace", lambda *args: None)
    direct_factory = MagicMock(side_effect=AssertionError("direct path selected"))
    monkeypatch.setattr(dcp, "get_direct_dcp_a2a_workspace", direct_factory)
    factory = MagicMock()
    monkeypatch.setattr(nixl_dcp, "get_nixl_dcp_a2a_workspace", factory)
    manager = dcp.MLADCPManager(
        _manager_config(),
        torch.device("cpu"),
        2,
        8,
        4,
        torch.bfloat16,
        torch.bfloat16,
        None,
        True,
        False,
    )
    factory.assert_called_once_with(
        group,
        torch.device("cpu"),
        16,
        2,
        4,
        torch.bfloat16,
        1,
    )
    output, lse = torch.empty(8, 4, 4), torch.empty(8, 4)
    manager.combine(output, lse, seq_lens=None, query_start_loc=None)
    factory.return_value.lse_reduce.assert_called_once_with(
        output,
        lse,
        is_lse_base_on_e=True,
        seq_lens=None,
        query_start_loc=None,
    )
    # SDK/setup errors must propagate instead of silently selecting NCCL.
    factory.side_effect = RuntimeError("SDK unavailable")
    with pytest.raises(RuntimeError, match="SDK unavailable"):
        manager._init_combine(2, 4, torch.bfloat16, True, False)


@pytest.mark.parametrize(
    "direct,a2a,pcp,message",
    [
        (True, True, False, "cannot both"),
        (False, False, False, "a2a backend"),
        (False, True, True, "without PCP"),
    ],
)
def test_mla_nixl_rejects_conflicting_configuration(
    monkeypatch,
    direct,
    a2a,
    pcp,
    message,
):
    monkeypatch.setattr(dcp.envs, "VLLM_USE_NIXL_DCP_A2A", True)
    monkeypatch.setattr(dcp.envs, "VLLM_USE_DIRECT_DCP_A2A", direct)
    manager = object.__new__(dcp.MLADCPManager)
    manager.use_a2a = a2a
    with pytest.raises(ValueError, match=message):
        manager._init_combine(2, 128, torch.bfloat16, True, pcp)


@pytest.mark.parametrize(
    "backend,dcp_size,pcp", [("ag_rs", 2, 1), ("a2a", 1, 1), ("a2a", 2, 2)]
)
def test_nixl_config_rejects_unused_flag(monkeypatch, backend, dcp_size, pcp):
    from vllm.config import DeviceConfig, ParallelConfig, VllmConfig

    monkeypatch.setattr(dcp.envs, "VLLM_USE_NIXL_DCP_A2A", True)
    monkeypatch.setattr(dcp.envs, "VLLM_USE_DIRECT_DCP_A2A", False)
    with pytest.raises(ValueError, match="VLLM_USE_NIXL_DCP_A2A requires"):
        VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            parallel_config=ParallelConfig(
                tensor_parallel_size=2,
                decode_context_parallel_size=dcp_size,
                prefill_context_parallel_size=pcp,
                dcp_comm_backend=backend,
                distributed_executor_backend="mp",
            ),
        )


def test_nixl_rejects_models_that_do_not_use_mla(monkeypatch, tmp_path):
    from transformers import OPTConfig

    from vllm.config import DeviceConfig, ModelConfig, ParallelConfig, VllmConfig

    monkeypatch.setattr(dcp.envs, "VLLM_USE_NIXL_DCP_A2A", True)
    monkeypatch.setattr(dcp.envs, "VLLM_USE_DIRECT_DCP_A2A", False)
    OPTConfig(architectures=["OPTForCausalLM"]).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="MLA models only"):
        VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            model_config=ModelConfig(
                model=str(tmp_path), skip_tokenizer_init=True, max_model_len=128
            ),
            parallel_config=ParallelConfig(
                tensor_parallel_size=2,
                decode_context_parallel_size=2,
                dcp_comm_backend="a2a",
                distributed_executor_backend="mp",
            ),
        )


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

    local_query = torch.empty(2, 2, 8)
    assert manager.query_gather(local_query) == direct_query.gather.return_value
    direct_query.gather.assert_called_once_with(local_query)
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
        output,
        lse,
        False,
        seq_lens,
        query_start_loc,
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


@pytest.mark.parametrize(
    ("parallel_drafting", "use_v2_model_runner", "expected"),
    [
        # Model runner V1's parallel drafter widens requests to 1 + 2 * 3.
        (True, False, 28),
        (True, True, 16),
        # A diffusion canvas of 3 without a speculative config.
        (None, True, 16),
    ],
)
def test_dcp_workspace_covers_decode_width(
    parallel_drafting, use_v2_model_runner, expected
):
    config = _manager_config()
    config.scheduler_config.max_num_batched_tokens = 128
    config.num_speculative_tokens = 3
    config.use_v2_model_runner = use_v2_model_runner
    if parallel_drafting is not None:
        config.speculative_config = MagicMock(
            num_speculative_tokens=3, parallel_drafting=parallel_drafting
        )

    assert dcp.get_dcp_workspace_max_num_tokens(config) == expected


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

    monkeypatch.setattr(sparse_mla, "PIN_MEMORY", False)

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
    layer = MagicMock(dcp_manager=manager)
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


def _distributed_direct_a2a_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    use_nixl = env.get("TEST_DCP_TRANSPORT") == "nixl"
    workspace = None
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
        workspace_type = dcp.DirectDCPA2AWorkspace
        workspace_group = dist.group.WORLD
        workspace_options = {}
        if use_nixl:
            from vllm.distributed.parallel_state import GroupCoordinator
            from vllm.v1.attention.ops import nixl_dcp

            nixl_dcp.dbo_current_ubatch_id = lambda: active_ubatch[0]
            workspace_type = nixl_dcp.NixlDCPA2AWorkspace
            workspace_group = GroupCoordinator(
                [list(range(world_size))],
                local_rank,
                "nccl",
                use_device_communicator=False,
            )
            workspace_options = {
                "num_channels": int(env.get("TEST_NIXL_CHANNELS", "4")),
                "tile_bytes": int(env.get("TEST_NIXL_TILE_BYTES", "32768")),
            }
        workspace = workspace_type(
            workspace_group,
            device,
            max_num_tokens,
            heads_per_rank,
            head_dim,
            dtype,
            num_ubatches=2,
            **workspace_options,
        )
        if use_nixl and env.get("TEST_REQUIRE_RDMA") == "1":
            assert any(
                "rc_mlx5_gda" in transports for transports in workspace.peer_transports
            ), workspace.peer_transports
            assert "nixl_put" in workspace.peer_copy_paths
        if use_nixl:
            remote_paths = [
                path
                for peer, path in enumerate(workspace.peer_copy_paths)
                if peer != rank
            ]
            if env.get("TEST_REQUIRE_MAPPED_P2P") == "1":
                assert all(path == "mapped_p2p" for path in remote_paths), remote_paths
            if env.get("TEST_REQUIRE_MIXED") == "1":
                assert set(remote_paths) == {"mapped_p2p", "nixl_put"}, remote_paths
            # Mapped destinations must bypass send staging throughout eager
            # calls and graph replay, as well as deliver the correct payload.
            workspace.send.fill_(0xA5)

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
            if use_nixl:
                actual, actual_lse = workspace.lse_reduce(
                    partial_output, partial_lse, is_lse_base_on_e, return_lse=True
                )
            else:
                actual = workspace.lse_reduce(
                    partial_output, partial_lse, is_lse_base_on_e
                )
            torch.accelerator.synchronize()

            if use_nixl:
                # Independently pack and exchange through NCCL, then compare
                # every received output/LSE bit before either slot is reused.
                packed_send = torch.empty(
                    (world_size, num_tokens, heads_per_rank, head_dim + 2),
                    device=device,
                    dtype=dtype,
                )
                dcp._dcp_a2a_pack_send(
                    partial_output,
                    partial_lse,
                    packed_send,
                    world_size,
                    heads_per_rank,
                    head_dim,
                    2,
                )
                expected_recv = torch.empty_like(packed_send)
                dist.all_to_all_single(expected_recv, packed_send)
                parity = int(workspace.epoch[active_ubatch[0], 0].item()) & 1
                received = workspace.receive[active_ubatch[0]]
                for source in range(world_size):
                    for first in range(0, num_tokens, workspace.tile_tokens):
                        rows = min(workspace.tile_tokens, num_tokens - first)
                        offset = (
                            (parity * world_size + source) * workspace.peer_bytes
                            + first // workspace.tile_tokens * workspace.tile_stride
                        )
                        output_bytes = rows * heads_per_rank * head_dim * 2
                        lse_offset = (output_bytes + 15) // 16 * 16
                        received_out = received[offset : offset + output_bytes]
                        received_lse = received[
                            offset + lse_offset : offset
                            + lse_offset
                            + rows * heads_per_rank * 4
                        ]
                        expected_tile = expected_recv[source, first : first + rows]
                        torch.testing.assert_close(
                            received_out.view(torch.int16).reshape(
                                rows, heads_per_rank, head_dim
                            ),
                            expected_tile[..., :head_dim]
                            .contiguous()
                            .view(torch.int16),
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            received_lse.view(torch.int32).reshape(
                                rows, heads_per_rank
                            ),
                            expected_tile[..., head_dim:]
                            .contiguous()
                            .view(torch.int32)
                            .squeeze(-1),
                            rtol=0,
                            atol=0,
                        )
                nccl_output, nccl_lse = dcp._dcp_a2a_unpack_combine(
                    expected_recv,
                    head_dim,
                    2,
                    True,
                    is_lse_base_on_e,
                )
                torch.testing.assert_close(actual, nccl_output, rtol=0, atol=0)
                torch.testing.assert_close(actual_lse, nccl_lse, rtol=0, atol=0)

            # Layout alone must not alter even one output bit.
            packed = workspace.lse_reduce(
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
            actual = workspace.lse_reduce(partial_output, partial_lse, is_lse_base_on_e)
        replays = int(env.get("TEST_NIXL_REPLAYS", "1024")) if use_nixl else 6
        if use_nixl:
            history = torch.empty((replays, 8), device=device, dtype=dtype)
            output_history = torch.empty(
                (replays, world_size, 8), device=device, dtype=dtype
            )
            lse_history = torch.empty(
                (replays, world_size), device=device, dtype=lse_dtype
            )
        for iteration in range(replays):
            if use_nixl:
                partial_output.fill_(iteration % 23 + rank + 1)
                partial_lse.fill_(((iteration + rank) % 7) * 0.25)
            else:
                partial_output.mul_(0.5).add_((rank + 1) * 0.01)
                partial_lse.add_((rank + 1) * 0.1)
            graph.replay()
            if use_nixl:
                history[iteration].copy_(actual[0, 0, :8])
                output_history[iteration].copy_(partial_output[0, ::heads_per_rank, :8])
                lse_history[iteration].copy_(partial_lse[0, ::heads_per_rank])
            if iteration % 2:
                # Replay epochs must also work after eager calls and use of
                # another ubatch, with no host-side epoch bookkeeping.
                for ubatch in (1, 0):
                    active_ubatch[0] = ubatch
                    # Smaller eager calls change the active channel count while
                    # the captured graph retains the larger shape's schedule.
                    eager_tokens = 1 if use_nixl else partial_output.shape[0]
                    workspace.lse_reduce(
                        partial_output[:eager_tokens],
                        partial_lse[:eager_tokens],
                        is_lse_base_on_e,
                    )
                active_ubatch[0] = 1
        torch.accelerator.synchronize()
        if use_nixl:
            # Check every replay's sample after the stress loop; NCCL reference
            # collectives must not add hidden rendezvous between replays.
            reference_send = torch.empty(
                (world_size, replays, 1, 10), device=device, dtype=dtype
            )
            dcp._dcp_a2a_pack_send(
                output_history, lse_history, reference_send, world_size, 1, 8, 2
            )
            reference_recv = torch.empty_like(reference_send)
            dist.all_to_all_single(reference_recv, reference_send)
            reference_history = dcp._dcp_a2a_unpack_combine(
                reference_recv, 8, 2, False, is_lse_base_on_e
            )
            torch.testing.assert_close(history, reference_history[:, 0], rtol=0, atol=0)

        packed = workspace.lse_reduce(
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
        head_slice = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)
        outputs = torch.stack(
            [value[:, head_slice, :] for value in gathered_output]
        ).float()
        lses = torch.stack([value[:, head_slice] for value in gathered_lse])
        expected = _lse_weighted_combine(
            outputs, lses, is_lse_base_on_e=is_lse_base_on_e
        )
        _assert_close(actual, expected, dtype)
        if use_nixl:
            for peer, path in enumerate(workspace.peer_copy_paths):
                if path == "mapped_p2p":
                    staged = workspace.send[:, peer]
                    torch.testing.assert_close(
                        staged, torch.full_like(staged, 0xA5), rtol=0, atol=0
                    )
    finally:
        if use_nixl and workspace is not None:
            workspace.close()
            workspace_group.destroy()
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
def test_distributed_direct_a2a_matches_reference(
    world_size: int, lse_layout: str, lse_dtype: str, lse_base_e: str
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
        },
    )


@pytest.mark.parametrize("world_size", [2, 8])
@pytest.mark.parametrize("num_channels", [1, 4])
@pytest.mark.parametrize(
    "dtype,lse_layout,lse_base_e",
    [("bfloat16", "transposed", "1"), ("float16", "packed", "0")],
)
def test_distributed_nixl_a2a_matches_reference(
    world_size, dtype, lse_layout, lse_base_e, num_channels
):
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need {world_size} CUDA GPUs")
    if not hasattr(torch.ops._C, "nixl_dcp_a2a_dispatch"):
        pytest.skip("Build vLLM with VLLM_BUILD_NIXL_DCP=ON")
    _distributed_run(
        _distributed_direct_a2a_worker,
        world_size=world_size,
        extra_env={
            "TEST_DCP_TRANSPORT": "nixl",
            "TEST_DTYPE": dtype,
            "TEST_LSE_DTYPE": "float32",
            "TEST_LSE_LAYOUT": lse_layout,
            "LSE_BASE_E": lse_base_e,
            "TEST_NIXL_CHANNELS": str(num_channels),
            "TEST_REQUIRE_MAPPED_P2P": "1",
        },
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
