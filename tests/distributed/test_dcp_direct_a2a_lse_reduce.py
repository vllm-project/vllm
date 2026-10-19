# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for direct symmetric-memory DCP collectives."""

import functools
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import multiprocess as mp
import pytest
import torch
import torch.distributed as dist

import vllm.v1.attention.ops.cp_common as cp_common
import vllm.v1.attention.ops.dcp as dcp
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor import custom_op
from vllm.model_executor.custom_op import CustomOp
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)


class _OOTQGather(dcp.DirectDCPQGather):
    calls: int = 0

    def forward_cuda(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.calls += 1
        return super().forward_cuda(*args, **kwargs)


class _OOTA2AReduce(dcp.DirectDCPA2AReduce):
    calls: int = 0

    def forward_cuda(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.calls += 1
        return super().forward_cuda(*args, **kwargs)


class _OOTKVGather(dcp.DirectDCPKVGather):
    calls: int = 0

    def forward_cuda(self, *args: Any, **kwargs: Any) -> None:
        self.calls += 1
        super().forward_cuda(*args, **kwargs)


@pytest.mark.parametrize(
    "op_cls, oot_cls",
    [
        (dcp.DirectDCPQGather, _OOTQGather),
        (dcp.DirectDCPA2AReduce, _OOTA2AReduce),
        (dcp.DirectDCPKVGather, _OOTKVGather),
    ],
)
@pytest.mark.parametrize("workspace_state", ["available", "absent", "oversized"])
@pytest.mark.parametrize("use_oot", [False, True])
@pytest.mark.parametrize(
    "mode",
    ["eager_default", "inductor_default", "all", "none", "explicit_on", "explicit_off"],
)
def test_oot_collective_dispatch_preserves_cuda_fallback(
    monkeypatch: pytest.MonkeyPatch,
    op_cls: type[CustomOp],
    oot_cls: type[CustomOp],
    mode: str,
    use_oot: bool,
    workspace_state: str,
) -> None:
    monkeypatch.setattr(custom_op, "op_registry_oot", custom_op.op_registry_oot.copy())
    for platform in ("rocm", "cpu", "tpu", "xpu", "out_of_tree"):
        monkeypatch.setattr(custom_op.current_platform, f"is_{platform}", lambda: False)
    if use_oot:
        op_cls.register_oot(oot_cls)
    op_name = op_cls.__name__ if use_oot else op_cls.name
    directives = {
        "eager_default": [],
        "inductor_default": [],
        "all": ["all"],
        "none": ["none"],
        "explicit_on": ["none", f"+{op_name}"],
        "explicit_off": [f"-{op_name}"],
    }
    workspace = SimpleNamespace(
        group=object(),
        world_size=2,
        rank=0,
        num_ubatches=1,
        max_num_tokens=2,
        heads_per_rank=1,
        head_dim=2,
        epoch=torch.zeros(1, dtype=torch.int64),
        received_signal=torch.zeros(1, 2, 2, dtype=torch.int32),
        completion=torch.zeros(1, 2, dtype=torch.int32),
    )
    args: list[Any]
    if op_cls is dcp.DirectDCPQGather:
        workspace.final_query = torch.zeros(1, 2, 2, 2)
        workspace.gathered_num_heads = workspace.padded_num_heads = 2
        workspace.multicast_ptrs = [(1, 2)]
        workspace.query_output = functools.partial(
            dcp.DirectDCPQGatherWorkspace.query_output, workspace
        )
        output = workspace.final_query[0]
        args = [torch.full((2, 1, 2), 42.0)]
        kernel_name, output_index = "direct_dcp_q_gather", 1
    elif op_cls is dcp.DirectDCPKVGather:
        workspace.received_kv = torch.zeros(1, 2, 4, 2)
        workspace.max_gathered_tokens = 4
        workspace.multicast_ptrs = [(1, 2)]
        output = torch.zeros(4, 2)
        args = [output, torch.full((2, 2), 42.0)]
        kernel_name, output_index = "direct_dcp_kv_gather", 5
    else:
        workspace.received_output = torch.zeros(1, 2, 2, 2, 1, 2)
        workspace.received_lse = torch.zeros(1, 2, 2, 2, 1)
        workspace.peer_output_ptrs = torch.zeros(1, 2, dtype=torch.int64)
        workspace.peer_lse_ptrs = torch.zeros(1, 2, dtype=torch.int64)
        workspace.peer_signal_ptrs = torch.zeros(1, 2, dtype=torch.int64)
        output = torch.zeros(2, 1, 2)
        args = [torch.full((2, 2, 2), 42.0), torch.zeros(2, 2), True]
        kernel_name, output_index = "direct_dcp_a2a_lse_reduce", 11
    group = SimpleNamespace(
        world_size=workspace.world_size,
        device_group=workspace.group,
    )
    if workspace_state == "oversized":
        workspace.max_num_tokens = 1
        workspace.max_gathered_tokens = 3
    collective_calls: list[None] = []

    def cuda_collective(*inputs: Any) -> None:
        inputs[output_index].fill_(42)

    def torch_collective(
        destination: torch.Tensor, source: torch.Tensor, *, group: Any
    ) -> None:
        assert group is workspace.group, "Native fallback must use the workspace group"
        collective_calls.append(None)
        destination.fill_(42)

    group.all_gather = MagicMock(
        side_effect=lambda query, dim: torch.cat((query, query), dim=dim)
    )
    a2a_fallback = MagicMock(return_value=output.fill_(42))
    monkeypatch.setattr(dcp, "dcp_a2a_lse_reduce", a2a_fallback)
    cuda_method = MagicMock(side_effect=cuda_collective)
    monkeypatch.setattr(torch.ops._C, kernel_name, cuda_method, raising=False)
    monkeypatch.setattr(dist, "all_gather_into_tensor", torch_collective)
    monkeypatch.setattr(dist, "all_to_all_single", torch_collective)
    config = VllmConfig(
        compilation_config=CompilationConfig(
            backend="inductor",
            mode=0 if mode == "eager_default" else 3,
            custom_ops=directives[mode],
        )
    )
    with set_current_vllm_config(config):
        op = op_cls(group, None if workspace_state == "absent" else workspace)
    assert type(op) is (oot_cls if use_oot else op_cls), (
        "Collective construction must resolve the registered implementation"
    )

    result = op(*args)
    actual = result if isinstance(result, torch.Tensor) else output
    assert torch.equal(actual, torch.full_like(output, 42)), (
        "Communication must run even with custom ops off"
    )
    cuda_enabled = mode != "explicit_off"
    uses_direct = cuda_enabled and (
        workspace_state == "available"
        or (op_cls is dcp.DirectDCPKVGather and workspace_state == "oversized")
    )
    assert cuda_method.call_count == int(uses_direct), (
        "Direct kernels must stay enabled under Inductor unless explicitly disabled"
    )
    if use_oot:
        assert op.calls == int(cuda_enabled), "Enabled dispatch must use the OOT op"
    cuda_fallback = cuda_enabled and not uses_direct
    assert group.all_gather.call_count == int(
        cuda_fallback and op_cls is dcp.DirectDCPQGather
    ), "CUDA query fallback must preserve GroupCoordinator's gather dispatch"
    assert a2a_fallback.call_count == int(
        cuda_fallback and op_cls is dcp.DirectDCPA2AReduce
    ), "CUDA A2A fallback must preserve the NCCL/Triton implementation"
    assert len(collective_calls) == int(
        not cuda_enabled or (cuda_fallback and op_cls is dcp.DirectDCPKVGather)
    ), "Native dispatch uses Torch; CUDA KV fallback preserves Torch's gather"


@pytest.mark.parametrize(
    "platform", ["cuda", "rocm", "cpu", "xpu", "tpu", "out_of_tree"]
)
@pytest.mark.parametrize(
    "op_cls, oot_cls",
    [
        (dcp.DirectDCPQGather, _OOTQGather),
        (dcp.DirectDCPA2AReduce, _OOTA2AReduce),
        (dcp.DirectDCPKVGather, _OOTKVGather),
    ],
)
@pytest.mark.parametrize("use_oot", [False, True])
def test_direct_dcp_platform_fallback(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    op_cls: type[CustomOp],
    oot_cls: type[CustomOp],
    use_oot: bool,
) -> None:
    monkeypatch.setattr(custom_op, "op_registry_oot", custom_op.op_registry_oot.copy())
    if use_oot:
        op_cls.register_oot(oot_cls)
    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["none"]))
    group = SimpleNamespace(
        world_size=2,
        device_group=object(),
        all_gather=MagicMock(
            side_effect=lambda query, dim: torch.cat((query, query + 2), dim=dim)
        ),
    )
    collective_calls: list[str] = []

    def all_gather(
        destination: torch.Tensor, source: torch.Tensor, *, group: Any
    ) -> None:
        collective_calls.append("gather")
        destination.copy_(torch.cat((source, source + 2)))

    def all_to_all(
        destination: torch.Tensor, source: torch.Tensor, *, group: Any
    ) -> None:
        collective_calls.append("a2a")
        destination.copy_(source[0].expand_as(source))
        destination[1, ..., :-1].add_(2)

    monkeypatch.setattr(dist, "all_gather_into_tensor", all_gather)
    monkeypatch.setattr(dist, "all_to_all_single", all_to_all)
    a2a_fallback = MagicMock(return_value=torch.full((2, 1, 2), 2.0))
    monkeypatch.setattr(dcp, "dcp_a2a_lse_reduce", a2a_fallback)
    with set_current_vllm_config(config):
        for name in ("cuda", "rocm", "cpu", "xpu", "tpu", "out_of_tree"):
            monkeypatch.setattr(
                custom_op.current_platform, f"is_{name}", lambda n=name: n == platform
            )
        # XPU must not inspect the CUDA symmetric-memory workspace.
        workspace = SimpleNamespace() if platform == "xpu" else None
        op = op_cls(group, workspace)
    if op_cls is dcp.DirectDCPQGather:
        result = op(torch.ones(2, 1, 2))
        expected = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]] * 2)
    elif op_cls is dcp.DirectDCPA2AReduce:
        result = op(torch.ones(2, 2, 2), torch.zeros(2, 2), True)
        expected = torch.full((2, 1, 2), 2.0)
    else:
        result = torch.empty(4, 2)
        op(result, torch.ones(2, 2))
        expected = torch.tensor([[1.0, 1.0]] * 2 + [[3.0, 3.0]] * 2)
    assert torch.equal(result, expected), (
        "Platform fallback must compute the collective"
    )
    legacy_gpu = platform in ("cuda", "rocm", "xpu")
    assert group.all_gather.call_count == int(
        legacy_gpu and op_cls is dcp.DirectDCPQGather
    ), "GPU query gather must retain the platform communicator"
    assert a2a_fallback.call_count == int(
        legacy_gpu and op_cls is dcp.DirectDCPA2AReduce
    ), "Only supported GPU platforms may launch the Triton A2A fallback"
    assert len(collective_calls) == int(
        not legacy_gpu or op_cls is dcp.DirectDCPKVGather
    ), "CPU/TPU/OOT must execute the Torch-native collective"
    if use_oot:
        assert op.calls == int(platform in ("cuda", "rocm")), (
            "CUDA-only plugin overrides must not run on CPU/XPU/TPU/OOT"
        )


@pytest.mark.parametrize("backend", ["native", "cuda"])
@pytest.mark.parametrize("world_size, num_heads", [(1, 3), (2, 3)])
def test_direct_dcp_a2a_fallback_checks(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    world_size: int,
    num_heads: int,
) -> None:
    op_cls = dcp.DirectDCPA2AReduce
    monkeypatch.setattr(op_cls, "forward_cpu", op_cls.forward_cuda)
    config = VllmConfig(
        compilation_config=CompilationConfig(
            custom_ops=[f"-{op_cls.name}"] if backend == "native" else ["all"]
        )
    )
    with set_current_vllm_config(config):
        op = op_cls(SimpleNamespace(world_size=world_size))
    partial_output = torch.ones(2, num_heads, 4)
    partial_lse = torch.zeros(2, num_heads)
    if world_size == 1:
        assert op(partial_output, partial_lse, True) is partial_output, (
            "Single-rank fallback must return its input without communication"
        )
    else:
        with pytest.raises(ValueError, match="H=3 must be divisible.*2"):
            op(partial_output, partial_lse, True)


def _torch_group_all_gather(group: Any, query: torch.Tensor, dim: int) -> torch.Tensor:
    shards = [torch.empty_like(query) for _ in range(dist.get_world_size(group))]
    dist.all_gather(shards, query.contiguous(), group=group)
    return torch.cat(shards, dim=dim)


def _collective_test_op(
    op_cls: type[CustomOp], mode: str, workspace: cp_common.DirectCPWorkspace
) -> _OOTQGather | _OOTA2AReduce | _OOTKVGather:
    oot_cls = {
        dcp.DirectDCPQGather: _OOTQGather,
        dcp.DirectDCPA2AReduce: _OOTA2AReduce,
        dcp.DirectDCPKVGather: _OOTKVGather,
    }[op_cls]
    if op_cls.__name__ not in custom_op.op_registry_oot:
        op_cls.register_oot(oot_cls)
    directives = [] if mode == "default" else [f"-{op_cls.__name__}"]
    config = VllmConfig(
        compilation_config=CompilationConfig(mode=3, custom_ops=directives)
    )
    with set_current_vllm_config(config):
        group = SimpleNamespace(
            world_size=workspace.world_size,
            device_group=workspace.group,
            all_gather=functools.partial(_torch_group_all_gather, workspace.group),
        )
        kwargs = (
            {"padded_num_heads": workspace.padded_num_heads}
            if isinstance(workspace, dcp.DirectDCPQGatherWorkspace)
            else {}
        )
        op = op_cls(group, workspace if mode == "default" else None, **kwargs)
    assert isinstance(op, (_OOTQGather, _OOTA2AReduce, _OOTKVGather)), (
        "Every worker must resolve the registered OOT collective"
    )
    return op


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


@pytest.mark.parametrize(
    "factory, workspace_cls, dimensions, expected_index",
    [
        (dcp.get_direct_dcp_a2a_workspace, "DirectDCPA2AWorkspace", (16, 2, 32), 0),
        (
            dcp.get_direct_dcp_q_gather_workspace,
            "DirectDCPQGatherWorkspace",
            (16, 2, 32),
            1,
        ),
        (
            dcp.get_direct_dcp_kv_gather_workspace,
            "DirectDCPKVGatherWorkspace",
            (64, 576),
            2,
        ),
    ],
)
@pytest.mark.parametrize("use_direct", [None, False, True])
@pytest.mark.parametrize(
    "cuda, symmetric_memory, same_node, multicast, dtype, expected",
    [
        (False, True, True, True, torch.bfloat16, (False, False, False)),
        (True, False, True, True, torch.bfloat16, (False, False, False)),
        (True, True, True, False, torch.bfloat16, (True, False, False)),
        (True, True, False, False, torch.bfloat16, (False, False, False)),
        (True, True, False, True, torch.bfloat16, (True, True, True)),
        (True, True, True, True, torch.float32, (False, True, False)),
    ],
)
def test_direct_dcp_workspace_eligibility(
    monkeypatch: pytest.MonkeyPatch,
    factory: Any,
    workspace_cls: str,
    dimensions: tuple[int, ...],
    expected_index: int,
    use_direct: bool | None,
    cuda: bool,
    symmetric_memory: bool,
    same_node: bool,
    multicast: bool,
    dtype: torch.dtype,
    expected: tuple[bool, bool, bool],
) -> None:
    env_name = (
        "VLLM_USE_DIRECT_DCP_A2A",
        "VLLM_USE_DIRECT_DCP_Q_GATHER",
        "VLLM_USE_DIRECT_DCP_KV_GATHER",
    )[expected_index]
    monkeypatch.setattr(dcp.envs, env_name, use_direct)
    if use_direct is not None:
        expected = (use_direct, use_direct and multicast, use_direct and multicast)
    monkeypatch.setattr(cp_common.current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(cp_common, "symm_mem_available", symmetric_memory)
    monkeypatch.setattr(
        cp_common, "in_the_same_node_as", lambda *args, **kw: [same_node]
    )
    monkeypatch.setattr(cp_common, "_symm_mem_spans_group", lambda group: multicast)
    workspace = object()
    allocate = MagicMock(return_value=workspace)
    monkeypatch.setattr(dcp, workspace_cls, allocate)
    factory.cache_clear()
    result = factory(
        _FakeGroupCoordinator(), torch.device("cpu"), *dimensions, dtype, 1
    )
    assert result is (workspace if expected[expected_index] else None), (
        "Workspace allocation must preserve automatic and forced env-var choices"
    )
    assert allocate.call_count == int(expected[expected_index])
    factory.cache_clear()


class TestDirectDCPGating:
    def test_zero_ubatches_raises(self):
        with pytest.raises(ValueError, match="ubatch"):
            dcp.DirectDCPA2AWorkspace(
                None, torch.device("cpu"), 16, 2, 32, torch.bfloat16, num_ubatches=0
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


def test_mla_dcp_manager_selects_direct_backends(monkeypatch, default_vllm_config):
    default_vllm_config.compilation_config.custom_ops = ["all"]
    query_gather = MagicMock(return_value=torch.empty(1, 4, 8))
    a2a_reduce = MagicMock(return_value=torch.empty(1, 2, 4))
    kv_gather = MagicMock()
    for op_cls, implementation in (
        (dcp.DirectDCPQGather, query_gather),
        (dcp.DirectDCPA2AReduce, a2a_reduce),
        (dcp.DirectDCPKVGather, kv_gather),
    ):
        monkeypatch.setattr(op_cls, "forward_cpu", implementation)
        monkeypatch.setattr(op_cls, "forward_cuda", implementation)

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

    query = torch.empty(1, 2, 8)
    assert manager.query_gather is not None, "Direct manager must bind query gather"
    assert manager.query_gather(query) is query_gather.return_value
    query_gather.assert_called_once_with(query)
    manager.init_kv_gather(workspace, 64)
    gathered_kv, local_kv = torch.empty(4, 8), torch.empty(2, 8)
    manager.kv_gather(gathered_kv, local_kv)
    kv_gather.assert_called_once_with(gathered_kv, local_kv)
    output, lse = torch.empty(1, 4, 4), torch.empty(1, 4)
    seq_lens = torch.ones(1, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    manager.combine(
        output,
        lse,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )
    a2a_reduce.assert_called_once_with(
        output,
        lse,
        is_lse_base_on_e=False,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )


def test_mla_dcp_manager_selects_fallback_backends(monkeypatch, default_vllm_config):
    for op_cls in (dcp.DirectDCPQGather, dcp.DirectDCPA2AReduce, dcp.DirectDCPKVGather):
        monkeypatch.setattr(op_cls, "forward_cpu", op_cls.forward_cuda)
    import vllm.v1.attention.ops.dcp as dcp_manager

    group = SimpleNamespace(
        world_size=2,
        device_group=object(),
        all_gather=MagicMock(
            side_effect=lambda query, dim: torch.cat((query, query), dim)
        ),
    )
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

    def gather(destination: torch.Tensor, source: torch.Tensor, *, group: Any) -> None:
        destination.copy_(torch.cat((source, source), dim=0))

    all_gather = MagicMock(side_effect=gather)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", all_gather)
    workspace = torch.empty(96, 8)
    manager.init_kv_gather(workspace, 64)
    output, local = torch.empty(4, 8), torch.ones(2, 8)
    manager.kv_gather(output, local)
    all_gather.assert_called_once_with(output, local, group=group.device_group)
    assert torch.equal(output, torch.cat((local, local), dim=0))

    query = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    assert manager.query_gather is not None
    assert torch.equal(manager.query_gather(query), torch.cat((query, query), dim=1))
    group.all_gather.assert_called_once_with(query, dim=1)

    fallback_output = torch.full((1, 2, 4), 42.0)
    a2a_fallback = MagicMock(return_value=fallback_output)
    monkeypatch.setattr(dcp_manager, "dcp_a2a_lse_reduce", a2a_fallback)
    partial_output, partial_lse = torch.full((1, 4, 4), 42.0), torch.zeros(1, 4)
    seq_lens = torch.ones(1, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    combined = manager.combine(
        partial_output,
        partial_lse,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )
    assert torch.equal(combined, torch.full((1, 2, 4), 42.0))
    a2a_fallback.assert_called_once_with(
        partial_output,
        partial_lse,
        cp_group=group,
        is_lse_base_on_e=True,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
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
    graph: torch.cuda.CUDAGraph | None = None
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        heads_per_rank, head_dim, max_num_tokens = 6, 576, 128
        padded_num_heads = 128 if world_size == 4 else None
        active_ubatch = [0]
        dcp.dbo_current_ubatch_id = lambda: active_ubatch[0]
        epoch_increment = int(env["CUSTOM_OPS"] == "default")
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
            op = _collective_test_op(dcp.DirectDCPQGather, env["CUSTOM_OPS"], workspace)

            cases = (
                ((1, False), (128, True), (129, True), (17, True), (5, False))
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
                actual = op(local_query)
                torch.accelerator.synchronize()

                if env["CUSTOM_OPS"] == "default" and num_tokens <= max_num_tokens:
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
                captured_output = op(captured_input)
            torch.accelerator.synchronize()
            dist.barrier()

            eager_input = torch.full(
                captured_input.shape,
                96 + rank,
                dtype=dtype,
                device=device,
            )
            eager_output = op(eager_input)
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
                    epoch_before_replays + (replay + 1) * epoch_increment
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
            interleaved_eager_output = op(interleaved_eager_input)
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
            assert (
                int(workspace.epoch[1].item())
                == epoch_before_mixed_replay + epoch_increment
            )
            _assert_q_gather_matches_reference(
                captured_output, captured_input, world_size, padded_num_heads
            )
        assert (op.calls > 0) == (env["CUSTOM_OPS"] == "default"), (
            "Distributed query gather must exercise the configured implementation"
        )
    finally:
        # NCCL shutdown waits for captured graphs to release their communicators.
        if graph is not None:
            graph.reset()
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
@pytest.mark.parametrize("custom_ops", ["default", "native"])
def test_distributed_direct_q_gather_cuda_graph_replay(
    world_size: int, custom_ops: str
):
    _distributed_run(
        _distributed_direct_q_gather_worker,
        world_size=world_size,
        extra_env={"TEST_CUDA_GRAPH": "1", "CUSTOM_OPS": custom_ops},
    )


def _distributed_direct_kv_gather_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    graph: torch.cuda.CUDAGraph | None = None
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
            op = _collective_test_op(
                dcp.DirectDCPKVGather, env["CUSTOM_OPS"], workspace
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
                op(gathered, local_kv)
                torch.accelerator.synchronize()

                expected = torch.empty_like(gathered)
                dist.all_gather_into_tensor(expected, local_kv.contiguous())
                assert torch.equal(
                    gathered.view(torch.uint8), expected.view(torch.uint8)
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op(gathered, local_kv)
            for replay in range(2):
                local_kv.fill_(rank + replay)
                graph.replay()
                torch.accelerator.synchronize()
                dist.all_gather_into_tensor(expected, local_kv.contiguous())
                assert torch.equal(gathered, expected), (
                    "KV gather replay must observe current local rows"
                )
        assert (op.calls > 0) == (env["CUSTOM_OPS"] == "default"), (
            "Distributed KV gather must exercise the configured implementation"
        )
    finally:
        if graph is not None:
            graph.reset()
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
@pytest.mark.parametrize("custom_ops", ["default", "native"])
def test_distributed_direct_kv_gather_matches_reference(
    world_size: int, custom_ops: str
):
    _distributed_run(
        _distributed_direct_kv_gather_worker,
        world_size=world_size,
        extra_env={"CUSTOM_OPS": custom_ops},
    )


def _distributed_direct_a2a_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    graph: torch.cuda.CUDAGraph | None = None
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
        op = _collective_test_op(dcp.DirectDCPA2AReduce, env["CUSTOM_OPS"], workspace)

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
            actual = op(partial_output, partial_lse, is_lse_base_on_e)
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
                    query_lens_tensor.cumsum(0, dtype=torch.int32),
                )
            )
            empty_rows = torch.repeat_interleave(seq_lens == 0, query_lens_tensor)
            # Model undefined attention rows with NaNs.
            partial_output[empty_rows] = float("nan")
            partial_lse[empty_rows] = float("nan")

            active_ubatch[0] = iteration % 2
            actual = op(
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

        cases = ((1, False), (17, True), (128, True), (129, True))
        for iteration, (num_tokens, padded) in enumerate(cases):
            check(num_tokens, iteration, padded)
        check_empty_shards(
            query_lens=[1, 3, 2, *([1] * (world_size - 1))],
            iteration=len(cases),
        )
        check_empty_shards(
            query_lens=[max_num_tokens + 1, 3, 2, *([1] * (world_size - 1))],
            iteration=len(cases) + 1,
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
            actual = op(partial_output, partial_lse, is_lse_base_on_e)
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
        assert (op.calls > 0) == (env["CUSTOM_OPS"] == "default"), (
            "Distributed A2A must exercise the configured implementation"
        )
    finally:
        if graph is not None:
            graph.reset()
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
@pytest.mark.parametrize("custom_ops", ["default", "native"])
@pytest.mark.parametrize("is_lse_base_on_e", [False, True])
def test_distributed_direct_a2a_matches_reference(
    world_size: int, custom_ops: str, is_lse_base_on_e: bool
):
    _distributed_run(
        _distributed_direct_a2a_worker,
        world_size=world_size,
        extra_env={
            "TEST_DTYPE": "bfloat16",
            "TEST_LSE_DTYPE": "bfloat16",
            "LSE_BASE_E": str(int(is_lse_base_on_e)),
            "CUSTOM_OPS": custom_ops,
        },
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
