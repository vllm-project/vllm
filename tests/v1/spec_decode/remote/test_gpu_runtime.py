# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the standalone host's wire, batch and partial-weight contracts."""

import json
import shutil
from contextlib import nullcontext, suppress
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from huggingface_hub import ResolvedRevision
from safetensors.torch import save_file
from transformers import PreTrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.adapter import TargetConditionedDraftAdapter
from vllm.v1.spec_decode.remote.runtime.gpu import (
    GPUDraftStepExecutor,
    _make_standalone_kv_config,
    make_input_batch,
)
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
from vllm.v1.spec_decode.remote.runtime.schema import create_runtime_contract
from vllm.v1.spec_decode.remote.runtime.step import DraftStepPlanner, DraftStepRow
from vllm.v1.spec_decode.remote.runtime.weights import selected_checkpoint_weights
from vllm.v1.spec_decode.remote.server import RemoteDraftServer
from vllm.v1.worker.gpu.input_batch import InputBuffers


class _Config(SimpleNamespace):
    def to_dict(self):
        return vars(self).copy()


@pytest.mark.parametrize("layout_name", ["LBHNC", "LBNHC", "LHBNC"])
@pytest.mark.parametrize("num_layers", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_standalone_real_kv_allocator_keeps_layers_and_blocks_disjoint(
    layout_name, num_layers, dtype
):
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.worker.utils import allocate_kv_cache

    spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=3, dtype=dtype)
    names = [f"draft.layers.{i}.attention" for i in range(num_layers)]
    layout = KVCacheLayout[layout_name]
    bytes_per_block = num_layers * spec.page_size_bytes
    budget = 6 * bytes_per_block - 1
    config = _make_standalone_kv_config(spec, names, budget, layout)
    assert config.num_blocks == 5
    caches = allocate_kv_cache(config, torch.device("cpu"), layout)
    assert set(caches) == set(names)
    for layer_index, cache in enumerate(caches.values()):
        assert cache.dtype == dtype
        assert cache.shape[0] == config.num_blocks
        assert not torch.count_nonzero(cache).item()
        for block in range(1, config.num_blocks):
            cache[block].fill_(10 * layer_index + block)
    # Check after *every* layer has written, so aliased layer views cannot
    # pass just because each write is immediately read from the same view.
    for layer_index, cache in enumerate(caches.values()):
        assert not torch.count_nonzero(cache[0]).item()
        for block in range(1, config.num_blocks):
            assert torch.all(cache[block] == 10 * layer_index + block).item()
        assert cache.untyped_storage().nbytes() == 5 * bytes_per_block
    assert len({cache.untyped_storage().data_ptr() for cache in caches.values()}) == 1
    assert 5 * bytes_per_block <= budget < 6 * bytes_per_block
    pool = DraftBlockPool(config.num_blocks, spec.block_size)
    allocated = [pool.ensure(i, spec.block_size)[0] for i in range(4)]
    assert allocated == [1, 2, 3, 4]
    assert pool.num_free_blocks == 0


@pytest.mark.parametrize("layout_name", ["BLHNC", "BLNHC", "BHLNC"])
def test_standalone_kv_config_rejects_unsupported_layouts(layout_name):
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.kv_cache_layout import KVCacheLayout

    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=3, dtype=torch.float32
    )
    with pytest.raises(ValueError, match="layer-compact"):
        _make_standalone_kv_config(
            spec, ["a", "b"], 10 * spec.page_size_bytes, KVCacheLayout[layout_name]
        )


def test_standalone_kv_budget_includes_null_block_for_every_layer():
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.kv_cache_layout import KVCacheLayout

    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=3, dtype=torch.float32
    )
    names = ["a", "b", "c"]
    group_bytes = len(names) * spec.page_size_bytes
    with pytest.raises(ValueError, match="null block plus one block"):
        _make_standalone_kv_config(
            spec, names, 2 * group_bytes - 1, KVCacheLayout.LBNHC
        )
    assert (
        _make_standalone_kv_config(
            spec, names, 2 * group_bytes, KVCacheLayout.LBNHC
        ).num_blocks
        == 2
    )


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_gpu_cli_checkpoint_code_requires_explicit_opt_in(
    monkeypatch, trust_remote_code
):
    import vllm.v1.spec_decode.remote.runtime.cli as cli
    import vllm.v1.spec_decode.remote.server as server

    captured = {}
    config, adapter = object(), object()

    def engine_args(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(create_engine_config=lambda: config)

    monkeypatch.setattr(cli, "EngineArgs", engine_args)
    monkeypatch.setattr(
        cli,
        "create_runtime_contract",
        lambda actual: SimpleNamespace(
            target_fingerprint="target", tokenizer_fingerprint="tokens"
        ),
    )
    monkeypatch.setattr(cli, "make_gpu_adapter", lambda *a, **kw: adapter)
    monkeypatch.setattr(
        server,
        "RemoteDraftServer",
        lambda *a, **kw: SimpleNamespace(
            start=lambda endpoint: endpoint,
            wait=lambda: None,
            stop=lambda: None,
            healthy=True,
        ),
    )
    argv = [
        "--endpoint",
        "unix:///tmp/unused-draft-cli.sock",
        "--adapter",
        "gpu",
        "--target-model",
        "test/target",
        "--draft-model",
        "test/draft",
    ]
    if trust_remote_code:
        argv.append("--trust-remote-code")
    server.main(argv)
    assert captured["trust_remote_code"] is trust_remote_code


@pytest.fixture
def draft_gpu_startup(monkeypatch):
    import vllm.distributed as distributed
    import vllm.ir
    import vllm.platforms as platforms
    import vllm.platforms.interface as interface
    import vllm.utils.torch_utils as torch_utils
    import vllm.v1.spec_decode.remote.runtime.gpu as gpu
    import vllm.v1.worker.workspace as workspace
    from vllm.distributed.parallel_state import _init_process_group_for_split_group

    class ProcessGroupCaptured(Exception):
        pass

    config = _config()
    config.parallel_config = SimpleNamespace(
        world_size=1, data_parallel_size=1, assigned_physical_gpu_ids=None
    )
    config.model_config.enforce_eager = True
    config.model_config.seed = 0
    config.kernel_config = SimpleNamespace(
        ir_op_priority=SimpleNamespace(set_default=lambda: None)
    )
    config.compilation_config = SimpleNamespace(ir_enable_torch_wrap=False)
    calls = []

    def init_distributed(**kwargs):
        _init_process_group_for_split_group(**kwargs, store=None, timeout=None)
        raise ProcessGroupCaptured

    monkeypatch.setattr(interface, "_assigned_physical_gpu_ids", None)
    platform = platforms.current_platform
    monkeypatch.setattr(
        type(platform), "device_control_env_var", "CUDA_VISIBLE_DEVICES"
    )
    monkeypatch.setattr(platform, "check_if_supports_dtype", lambda dtype: None)
    monkeypatch.setattr(platform, "import_kernels", lambda: None)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 3)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda device: None)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda precision: None)
    monkeypatch.setattr(
        torch.distributed, "init_process_group", lambda **kw: calls.append(kw)
    )
    monkeypatch.setattr(distributed, "init_distributed_environment", init_distributed)
    monkeypatch.setattr(torch_utils, "set_random_seed", lambda seed: None)
    monkeypatch.setattr(torch_utils, "set_torch_threads_for_runtime", lambda: None)
    monkeypatch.setattr(workspace, "init_workspace_manager", lambda *a, **kw: None)
    monkeypatch.setattr(gpu, "set_current_vllm_config", lambda config: nullcontext())
    monkeypatch.setattr(vllm.ir, "set_default_torch_wrap", lambda enable: None)

    def start(device):
        with pytest.raises(ProcessGroupCaptured):
            GPUDraftStepExecutor(config, device=device, kv_cache_bytes=1024)
        assert len(calls) == 1
        assert calls[0]["world_size"] == 1
        assert calls[0]["rank"] == 0
        assert calls[0]["device_id"] == device

    return config, start, calls


@pytest.mark.parametrize("visible", [None, "2,0,1"])
@pytest.mark.parametrize("assigned", [None, [2, 0, 1]])
@pytest.mark.parametrize("device_index", [0, 1, 2])
def test_draft_process_group_uses_selected_visible_gpu(
    draft_gpu_startup, monkeypatch, visible, assigned, device_index
):
    import vllm.platforms.interface as interface

    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(interface, "_assigned_physical_gpu_ids", assigned)
    _, start, _ = draft_gpu_startup
    start(torch.device("cuda", device_index))


def test_draft_process_group_honors_configured_gpu_assignment(
    draft_gpu_startup, monkeypatch
):
    config, start, _ = draft_gpu_startup
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0,1")
    config.parallel_config.assigned_physical_gpu_ids = [1]
    start(torch.device("cuda:2"))


@pytest.mark.parametrize("device", ["cuda", "cuda:3", "cpu"])
def test_draft_rejects_invalid_gpu_before_distributed_init(draft_gpu_startup, device):
    config, _, calls = draft_gpu_startup
    with pytest.raises(ValueError, match="CUDA device|visible CUDA"):
        GPUDraftStepExecutor(config, device=torch.device(device), kv_cache_bytes=1024)
    assert not calls


def test_draft_rejects_unassigned_gpu_before_distributed_init(
    draft_gpu_startup, monkeypatch
):
    config, _, calls = draft_gpu_startup
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    config.parallel_config.assigned_physical_gpu_ids = [1]
    with pytest.raises(ValueError, match="not assigned"):
        GPUDraftStepExecutor(config, device=torch.device("cuda:2"), kv_cache_bytes=1024)
    assert not calls


def test_draft_rejects_conflicting_gpu_assignment(draft_gpu_startup, monkeypatch):
    import vllm.platforms.interface as interface

    config, _, calls = draft_gpu_startup
    monkeypatch.setattr(interface, "_assigned_physical_gpu_ids", [0])
    config.parallel_config.assigned_physical_gpu_ids = [1]
    with pytest.raises(RuntimeError, match="conflicting values"):
        GPUDraftStepExecutor(config, device=torch.device("cuda:1"), kv_cache_bytes=1024)
    assert interface.get_assigned_physical_gpu_ids() == [0]
    assert not calls


@pytest.mark.parametrize("cache_kind", ["full", "sliding", "mla"])
@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
@pytest.mark.parametrize("max_model_len", [1024, 1025])
def test_draft_block_table_covers_attention_tile_past_context_end(
    draft_gpu_startup, monkeypatch, cache_kind, block_size, max_model_len
):
    import vllm.distributed as distributed
    import vllm.v1.attention.backends.utils as attn_utils
    import vllm.v1.spec_decode.remote.runtime.gpu as gpu
    import vllm.v1.worker.gpu.attn_utils as gpu_attn
    import vllm.v1.worker.gpu.block_table as block_table
    import vllm.v1.worker.gpu.input_batch as input_batch
    import vllm.v1.worker.gpu.spec_decode.eagle.speculator as eagle
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MLAAttentionSpec,
        SlidingWindowSpec,
    )
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.worker.utils import AttentionGroup

    config, _, _ = draft_gpu_startup
    config.model_config.max_model_len = max_model_len
    config.speculative_config.num_speculative_tokens = 1
    config.scheduler_config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=32)
    config.parallel_config.decode_context_parallel_size = 1
    kwargs = dict(
        block_size=block_size, num_kv_heads=1, head_size=256, dtype=torch.bfloat16
    )
    cache_spec = (
        SlidingWindowSpec(**kwargs, sliding_window=1024)
        if cache_kind == "sliding"
        else MLAAttentionSpec(**kwargs)
        if cache_kind == "mla"
        else FullAttentionSpec(**kwargs)
    )
    layout = KVCacheLayout.LBHNC
    monkeypatch.setattr(distributed, "init_distributed_environment", lambda **kw: None)
    monkeypatch.setattr(distributed, "initialize_model_parallel", lambda: None)
    monkeypatch.setattr(gpu, "materialize_target_shell", lambda *a, **kw: object())
    monkeypatch.setattr(
        eagle,
        "EagleSpeculator",
        lambda *a: SimpleNamespace(
            load_model=lambda model: None, model=SimpleNamespace(eval=lambda: None)
        ),
    )
    monkeypatch.setattr(
        gpu_attn, "get_kv_cache_spec", lambda config: {"draft": cache_spec}
    )
    monkeypatch.setattr(gpu, "get_layers_from_vllm_config", lambda *a: {})
    monkeypatch.setattr(
        attn_utils, "get_supported_kv_cache_layouts", lambda layers: [layout]
    )
    monkeypatch.setattr(attn_utils, "resolve_kv_cache_layout", lambda *a: layout)
    monkeypatch.setattr(input_batch, "InputBuffers", lambda *a: None)

    class TableChecked(Exception):
        pass

    class MetadataBuilder:
        requires_block_table_width = False

        def __init__(self, *args):
            pass

    def check_table(**kwargs):
        width = kwargs["max_num_blocks_per_group"][0]
        # Native kernels load a whole final tile before masking token positions.
        # A 1025-token sequence reaches logical positions 1024..1055 in Triton
        # SWA's 32-token tile, even though only the first position is live.
        table = torch.arange(width, dtype=torch.int32).unsqueeze(0)
        tile_start = (max_model_len - 1) // 32 * 32
        table[:, torch.arange(tile_start, tile_start + 32) // block_size]
        alignment = cache_spec.block_table_token_alignment
        assert alignment is not None and width * block_size % alignment == 0
        if block_size == 128:
            # FlashMLA consumes 64-token kernel pages. Exercise the native
            # remapping workspace, whose capacity comes from max_model_len.
            backend = SimpleNamespace(
                get_supported_kernel_block_sizes=lambda spec: [64],
                get_builder_cls=lambda: MetadataBuilder,
            )
            group = AttentionGroup(backend, ["draft"], cache_spec, 0)
            group.create_metadata_builders(config, torch.device("cpu"), 64)
            mapped = group.map_to_kernel_block_table(table)
            assert torch.equal(mapped, torch.arange(2 * width).unsqueeze(0))
        raise TableChecked

    monkeypatch.setattr(block_table, "BlockTables", check_table)
    with pytest.raises(TableChecked):
        GPUDraftStepExecutor(
            config,
            device=torch.device("cuda:0"),
            kv_cache_bytes=4 * cache_spec.page_size_bytes,
        )


def test_sampling_buffers_match_mrv2_prefill_kernel_token_dtypes():
    executor = object.__new__(GPUDraftStepExecutor)
    executor.device = torch.device("cpu")
    executor._init_sampling_buffers(3)
    assert executor.last_sampled.dtype == torch.int64
    assert executor.lookahead.dtype == torch.int32
    assert executor.lookahead.shape == (1, 3)
    assert executor.temperature.dtype == torch.float32
    assert executor.seeds.dtype == torch.int64


@pytest.mark.parametrize("prefilling", [False, True])
@pytest.mark.parametrize("next_token", [-1, 32, 2**32 + 1, None])
def test_selected_next_token_is_validated_before_device_work(
    monkeypatch, prefilling, next_token
):
    planner = DraftStepPlanner(
        DraftBlockPool(8, 4),
        max_num_reqs=2,
        max_model_len=16,
        num_speculative_tokens=1,
    )
    a, b = SequenceKey("v", 1, 0), SequenceKey("v", 2, 0)
    planner.open(a)
    planner.open(b)
    plan, failures = planner.plan(
        [
            DraftStepRow(b, 1, num_sampled=1, last_sampled=31),
            DraftStepRow(
                a, 1, num_sampled=0, last_sampled=-1, next_prefill_tokens=(0,)
            ),
        ],
        num_speculative_tokens=0 if prefilling else 1,
    )
    assert not failures
    # The unused source may contain a sentinel, even in a reordered mixed batch.
    plan.next_prefill_tokens[0, 0] = -1
    if next_token is not None:
        if prefilling:
            plan.next_prefill_tokens[1, 0] = next_token
        else:
            plan.last_sampled[0] = next_token
    executor = object.__new__(GPUDraftStepExecutor)
    executor.vllm_config = _config()
    executor.contract = create_runtime_contract(executor.vllm_config)
    executor.device = torch.device("cpu")
    executor.input_buffers = SimpleNamespace(max_num_tokens=2)
    features = [
        torch.tensor([1, 2], dtype=torch.int32),
        torch.from_numpy(plan.positions),
    ]

    class DeviceWorkStarted(Exception):
        pass

    def device_work(*args):
        raise DeviceWorkStarted

    monkeypatch.setattr(torch.accelerator, "set_device_index", device_work)
    if next_token is None:
        with pytest.raises(DeviceWorkStarted):
            executor.execute(
                plan, features, num_speculative_tokens=0 if prefilling else 1
            )
    else:
        with pytest.raises(ValueError, match="next input token.*vocabulary"):
            executor.execute(
                plan, features, num_speculative_tokens=0 if prefilling else 1
            )


@pytest.fixture
def cpu_block_table_executor(monkeypatch):
    import vllm.v1.spec_decode.remote.runtime.gpu as gpu
    import vllm.v1.worker.gpu.buffer_utils as buffers
    from vllm.v1.worker.gpu.block_table import BlockTables

    class CpuPool:
        def copy_to_uva(self, values):
            return torch.tensor(values, dtype=torch.int32)

    class Kernel:
        def __getitem__(self, grid):
            def apply(output, stride, indices, starts, contents, ends, *a, **kw):
                destinations = set()
                offset = 0
                for row, start, end in zip(indices.tolist(), starts, ends):
                    for col, value in enumerate(contents[offset:end], int(start)):
                        assert (row, col) not in destinations
                        destinations.add((row, col))
                        output[row, col] = value
                    offset = int(end)

            return apply

    class Prepared(Exception):
        pass

    def stop_before_model(*args):
        raise Prepared

    monkeypatch.setattr(buffers, "UvaBufferPool", lambda *a, **kw: CpuPool())
    monkeypatch.setattr(
        buffers, "async_tensor_h2d", lambda values, **kw: torch.tensor(values, **kw)
    )
    monkeypatch.setattr(buffers, "_apply_write_kernel", Kernel())
    monkeypatch.setattr(gpu, "make_input_batch", stop_before_model)
    monkeypatch.setattr(gpu, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda _: None)
    synchronized: list[torch.device] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronized.append)
    executor = object.__new__(GPUDraftStepExecutor)
    executor.device = torch.device("cpu")
    executor.vllm_config = _config()
    executor.contract = create_runtime_contract(executor.vllm_config)
    executor.input_buffers = SimpleNamespace(max_num_tokens=8)
    tables = executor.block_tables = object.__new__(BlockTables)
    staged = buffers.StagedWriteTensor((2, 4), torch.int32, executor.device)
    tables.block_tables, tables.num_kv_cache_groups = [staged], 1
    tables.num_blocks = SimpleNamespace(
        np=np.zeros((1, 2), dtype=np.int32), copy_to_uva=lambda: None
    )
    planner = DraftStepPlanner(
        DraftBlockPool(4, 1), max_num_reqs=2, max_model_len=4, num_speculative_tokens=1
    )
    adapter = TargetConditionedDraftAdapter(
        executor,
        planner,
        capabilities=executor.contract.capabilities,
        feature_schema=executor.contract.schema,
        methods={"eagle3"},
    )

    def run(plan):
        with suppress(Prepared):
            executor.execute(
                plan,
                [
                    torch.ones(plan.num_tokens, dtype=torch.int32),
                    torch.from_numpy(plan.positions),
                ],
            )

    return adapter, run, staged, synchronized


@pytest.mark.parametrize("failure", [None, "stage", "upload", "publish"])
def test_failed_block_table_preparation_does_not_replay_writes_after_slot_reuse(
    cpu_block_table_executor, monkeypatch, failure
):
    import vllm.v1.worker.gpu.buffer_utils as buffers

    adapter, run, staged, synchronized = cpu_block_table_executor
    planner = adapter.planner
    keys = [SequenceKey("tables", i, 0) for i in range(4)]
    for key in keys[:2]:
        assert planner.open(key)
    first, failures = planner.plan([DraftStepRow(key, 1, 1) for key in keys[:2]])
    assert not failures

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("injected preparation OOM")

    with monkeypatch.context() as patch:
        if failure == "stage":

            class PartialContents(list):
                def extend(self, values):
                    super().extend(values)
                    raise MemoryError("injected preparation OOM")

            patch.setattr(staged, "_staged_write_contents", PartialContents())
        elif failure == "upload":
            patch.setattr(buffers, "async_tensor_h2d", fail)
        elif failure == "publish":
            patch.setattr(adapter.executor.block_tables.num_blocks, "copy_to_uva", fail)
        if failure is None:
            run(first)
        else:
            with pytest.raises((MemoryError, torch.OutOfMemoryError)):
                run(first)
            assert not staged._staged_write_indices
            assert not staged._staged_write_starts
            assert not staged._staged_write_contents
            assert not staged._staged_write_cu_lens
    assert synchronized == ([] if failure is None else [torch.device("cpu")])
    adapter.close_sequences(tuple(keys[:2]))
    assert planner.num_open == 0
    assert planner.block_pool.num_free_blocks == 3
    for key in keys[2:]:
        assert planner.open(key)
    second, failures = planner.plan(
        [DraftStepRow(keys[2], 2, 1), DraftStepRow(keys[3], 1, 1)]
    )
    assert not failures
    run(second)
    for index in second.req_indices:
        owned = planner.block_pool.blocks(int(index))
        assert staged.gpu[index, : len(owned)].tolist() == list(owned)
        assert adapter.executor.block_tables.num_blocks.np[0, index] == len(owned)


@pytest.mark.parametrize("cleanup_failure", ["synchronize", "clear"])
def test_block_table_cleanup_failure_fences_runtime(
    cpu_block_table_executor, monkeypatch, cleanup_failure
):
    import vllm.v1.worker.gpu.buffer_utils as buffers
    from vllm.v1.spec_decode.remote.errors import (
        CompletionUnknownError,
        runtime_unusable,
    )

    adapter, run, staged, _ = cpu_block_table_executor
    key = SequenceKey("tables", 0, 0)
    adapter.planner.open(key)
    plan, failures = adapter.planner.plan([DraftStepRow(key, 1, 1)])
    assert not failures

    def fail_upload(*args, **kwargs):
        raise torch.OutOfMemoryError("upload failed")

    def fail_cleanup(*args):
        raise RuntimeError("retirement failed")

    monkeypatch.setattr(buffers, "async_tensor_h2d", fail_upload)
    if cleanup_failure == "synchronize":
        monkeypatch.setattr(torch.accelerator, "synchronize", fail_cleanup)
    else:
        monkeypatch.setattr(staged, "clear_staged_writes", fail_cleanup)
    with pytest.raises(CompletionUnknownError) as caught:
        run(plan)
    assert runtime_unusable(caught.value)
    assert str(caught.value.__cause__) == "retirement failed"


@pytest.mark.parametrize("fatal", [False, True])
def test_unusable_block_table_failure_preserves_error_without_device_cleanup(
    cpu_block_table_executor, monkeypatch, fatal
):
    import vllm.v1.worker.gpu.buffer_utils as buffers
    from vllm.v1.spec_decode.remote.errors import (
        CompletionUnknownError,
        CudaRuntimeError,
    )

    adapter, run, staged, synchronized = cpu_block_table_executor
    key = SequenceKey("tables", 0, 0)
    adapter.planner.open(key)
    plan, failures = adapter.planner.plan([DraftStepRow(key, 1, 1)])
    assert not failures
    error = (
        CudaRuntimeError("upload failed", cuda_error_code=700)
        if fatal
        else CompletionUnknownError("upload completion unknown")
    )

    def fail_upload(*args, **kwargs):
        raise error

    monkeypatch.setattr(buffers, "async_tensor_h2d", fail_upload)
    with pytest.raises(type(error)) as caught:
        run(plan)
    assert caught.value is error
    assert not synchronized
    assert not staged._staged_write_indices


class _Model(SimpleNamespace):
    def get_hidden_size(self):
        return self.hf_config.hidden_size

    def get_vocab_size(self):
        return self.hf_config.vocab_size


def _config(*, use_aux=False, method="eagle3"):
    target = _Model(
        model="target",
        revision="a" * 40,
        dtype=torch.bfloat16,
        quantization="mxfp4",
        tokenizer="target",
        tokenizer_revision="a" * 40,
        tokenizer_mode="auto",
        hf_config=_Config(
            model_type="gpt_oss",
            hidden_size=16,
            vocab_size=32,
            eos_token_id=2,
            architectures=["GptOssForCausalLM"],
        ),
    )
    draft = _Model(
        model="draft",
        revision="b" * 40,
        dtype=torch.bfloat16,
        quantization=None,
        hf_config=_Config(
            model_type="llama",
            hidden_size=16,
            vocab_size=32,
            architectures=["Eagle3LlamaForCausalLM"],
            eagle_config={"use_aux_hidden_state": use_aux},
        ),
    )
    if method == "mtp":
        target.hf_config = PreTrainedConfig(
            model_type="mimo",
            architectures=["MiMoForCausalLM"],
            hidden_size=16,
            vocab_size=32,
            num_nextn_predict_layers=1,
        )
        draft.hf_config = SpeculativeConfig.hf_config_override(
            deepcopy(target.hf_config)
        )
    return SimpleNamespace(
        model_config=target,
        speculative_config=SimpleNamespace(
            draft_model_config=draft,
            method=method,
            num_speculative_tokens=1 if method == "mtp" else 3,
            parallel_drafting=False,
            draft_sample_method="greedy",
        ),
    )


@pytest.mark.parametrize(
    (
        "method",
        "aux",
        "max_model_len",
        "num_blocks",
        "block_size",
        "num_drafts",
        "decode_has_prefix",
    ),
    [
        ("eagle3", False, 32, 16, 4, 3, True),
        ("eagle3", True, 32, 16, 4, 3, True),
        ("mtp", False, 32, 16, 4, 1, True),
        ("mtp", False, 1, 2, 2, 1, False),
        ("eagle3", False, 32, 2, 16, 16, False),
        ("eagle3", True, 64, 2, 32, 32, False),
        ("eagle3", False, 32, 16, 4, 0, True),
        ("eagle3", False, 32, 2, 16, 1, False),
        ("eagle3", False, 32, 2, 16, 8, True),
        ("eagle3", False, 1, 2, 16, 16, False),
        ("eagle3", False, 8, 2, 16, 16, False),
    ],
)
@pytest.mark.parametrize("failure", [None, "decode", "synchronize"])
def test_gpu_warmup_completes_before_listening_without_live_request_state(
    method,
    aux,
    max_model_len,
    num_blocks,
    block_size,
    num_drafts,
    decode_has_prefix,
    failure,
    monkeypatch,
):
    import vllm.v1.spec_decode.remote.runtime.gpu as gpu

    config = _config(use_aux=aux, method=method)
    config.model_config.max_model_len = max_model_len
    config.speculative_config.num_speculative_tokens = num_drafts
    executor = object.__new__(GPUDraftStepExecutor)
    executor.vllm_config = config
    executor.contract = create_runtime_contract(config)
    executor.device = torch.device("cpu")
    executor.num_blocks = num_blocks
    executor.block_size = block_size
    executor.input_buffers = InputBuffers(2, 16, executor.device)
    executor._init_sampling_buffers(2)
    planner = DraftStepPlanner(
        DraftBlockPool(executor.num_blocks, executor.block_size),
        max_num_reqs=2,
        max_model_len=max_model_len,
        num_speculative_tokens=num_drafts,
    )
    adapter = TargetConditionedDraftAdapter(
        executor,
        planner,
        capabilities=executor.contract.capabilities,
        feature_schema=executor.contract.schema,
        methods={method},
    )
    server = RemoteDraftServer(adapter)
    calls = []
    warmup_planners = []

    def warmup_planner(*args, **kwargs):
        result = DraftStepPlanner(*args, **kwargs)
        warmup_planners.append(result)
        return result

    def execute(plan, features):
        assert server.endpoint is None
        assert planner.num_open == 0
        batch = make_input_batch(plan, features[0], executor.input_buffers)
        phase = "prefill" if batch.has_prefill else "decode"
        calls.append(phase)
        assert bool(plan.num_computed_tokens[0]) is (
            phase == "decode" and decode_has_prefix
        )
        for slot, tensor in zip(executor.contract.schema.slots, features):
            count = plan.num_reqs if slot.per_sequence else plan.num_tokens
            assert tensor.shape == (count, *slot.trailing_shape)
            assert tensor.dtype == getattr(torch, slot.dtype)
        assert torch.equal(features[1], torch.from_numpy(plan.positions))
        executor.last_sampled.fill_(17)
        executor.lookahead.fill_(19)
        if phase == failure:
            raise RuntimeError("warmup failed")
        return torch.zeros(plan.num_reqs, num_drafts, dtype=torch.int64)

    def synchronize(device):
        assert server.endpoint is None
        assert device == executor.device
        calls.append("synchronize")
        if failure == "synchronize":
            raise RuntimeError("warmup failed")

    monkeypatch.setattr(gpu, "DraftStepPlanner", warmup_planner)
    monkeypatch.setattr(executor, "execute", execute)
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)
    try:
        if failure:
            with pytest.raises(RuntimeError, match="warmup failed"):
                server.start("tcp://127.0.0.1:0")
            assert server.endpoint is None
        else:
            server.start("tcp://127.0.0.1:0")
            assert server.endpoint is not None
        assert calls == ["prefill", "decode", "synchronize"]
        assert len(warmup_planners) == 1
        assert warmup_planners[0].num_open == 0
        assert warmup_planners[0].block_pool.num_free_blocks == num_blocks - 1
        assert planner.num_open == 0
        assert planner.block_pool.num_free_blocks == num_blocks - 1
        assert not executor.last_sampled.any()
        assert not executor.lookahead.any()
    finally:
        server.stop()


@pytest.mark.parametrize("failed_forward", [1, 2])
@pytest.mark.parametrize("code", [226, 700, 710, 810])
def test_fatal_warmup_preserves_error_without_touching_gpu_cleanup(
    monkeypatch, failed_forward, code
):
    from vllm.v1.spec_decode.remote.errors import CudaRuntimeError

    config = _config()
    config.model_config.max_model_len = 32
    executor = object.__new__(GPUDraftStepExecutor)
    executor.vllm_config = config
    executor.contract = create_runtime_contract(config)
    executor.device = torch.device("cpu")
    executor.num_blocks, executor.block_size = 16, 4
    executor.input_buffers = SimpleNamespace(max_num_tokens=16)
    calls = []
    executor.last_sampled = SimpleNamespace(zero_=lambda: calls.append("zero"))
    executor.lookahead = SimpleNamespace(zero_=lambda: calls.append("zero"))
    error = CudaRuntimeError("warmup forward failed", cuda_error_code=code)

    def execute(*args):
        calls.append("forward")
        if len(calls) == failed_forward:
            raise error

    monkeypatch.setattr(executor, "execute", execute)
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda _: calls.append("sync")
    )
    with pytest.raises(CudaRuntimeError) as caught:
        executor.warmup()
    assert caught.value is error
    assert calls == ["forward"] * failed_forward


def test_completion_unknown_remains_unusable_through_exception_chains():
    from vllm.v1.spec_decode.remote.errors import (
        CompletionUnknownError,
        cuda_context_fatal,
        runtime_unusable,
    )

    uncertain = CompletionUnknownError("copy did not complete")
    wrapper = RuntimeError("feature import failed")
    wrapper.__context__ = uncertain
    uncertain.__cause__ = wrapper  # Classification must tolerate cycles.
    assert runtime_unusable(wrapper)
    assert not cuda_context_fatal(wrapper)
    assert not runtime_unusable(RuntimeError("ordinary failure"))


def test_gpu_warmup_rejects_a_budget_that_cannot_admit_one_token(monkeypatch):
    import vllm.v1.spec_decode.remote.runtime.gpu as gpu

    config = _config()
    config.model_config.max_model_len = 32
    config.speculative_config.num_speculative_tokens = 17
    executor = object.__new__(GPUDraftStepExecutor)
    executor.vllm_config = config
    executor.device = torch.device("cpu")
    executor.num_blocks = 2
    executor.block_size = 16
    executor.input_buffers = SimpleNamespace(max_num_tokens=16)
    executor._init_sampling_buffers(1)
    planner = DraftStepPlanner(
        DraftBlockPool(2, 16),
        max_num_reqs=1,
        max_model_len=32,
        num_speculative_tokens=17,
    )
    calls = []
    monkeypatch.setattr(gpu, "DraftStepPlanner", lambda *a, **kw: planner)
    monkeypatch.setattr(executor, "execute", lambda *a: calls.append("execute"))
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda device: calls.append("synchronize")
    )
    with pytest.raises(ValueError, match="cannot hold a warmup step"):
        executor.warmup()
    assert calls == ["synchronize"]
    assert planner.num_open == 0
    assert planner.block_pool.num_free_blocks == 1


@pytest.mark.parametrize("aux", [False, True])
def test_contract_transports_only_the_checkpoint_hidden_state_kind(aux):
    contract = create_runtime_contract(_config(use_aux=aux))
    hidden = contract.schema.slots[2]
    assert hidden.kind == ("aux_hidden_states" if aux else "last_hidden_states")
    assert hidden.trailing_shape == (48 if aux else 16,)
    assert hidden.dtype == "bfloat16"
    assert len(contract.schema.slots) == 4
    lookahead = contract.schema.slots[-1]
    assert lookahead.optional and lookahead.per_sequence
    assert lookahead.trailing_shape == (1,)


@pytest.mark.parametrize(
    "layers",
    [
        {"target_layer_ids": [0, 1]},
        {"eagle_aux_hidden_state_layer_ids": [1, 2]},
        {"eagle_config": {"use_aux_hidden_state": True, "layer_ids": [1, 2]}},
    ],
)
def test_eagle_contract_matches_native_auxiliary_selection(layers):
    from transformers import Qwen3Config

    from vllm.transformers_utils.configs.eagle import EAGLEConfig
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    config = _config(use_aux=True)
    draft = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=16,
        vocab_size=32,
        num_aux_layers=2,
        **layers,
    )
    config.speculative_config.draft_model_config.hf_config = EAGLEConfig(
        model=draft, method="eagle3"
    )
    selected = get_eagle3_aux_layers_from_config(config.speculative_config)
    assert selected == (1, 2)
    feature = create_runtime_contract(config).schema.slots[2]
    assert feature.trailing_shape == (16 * len(selected),)


@pytest.mark.parametrize("count_field", ["num_aux_layers", "num_aux_hidden_states"])
def test_eagle_contract_rejects_projection_and_selected_layer_count_mismatch(
    count_field,
):
    config = _config(use_aux=True)
    draft = config.speculative_config.draft_model_config.hf_config
    draft.target_layer_ids = [0, 1]
    setattr(draft, count_field, 3)
    with pytest.raises(ValueError, match="auxiliary.*count"):
        create_runtime_contract(config)


def test_contract_accepts_resolved_mimo_architecture():
    config = _config(method="mtp")
    draft = config.speculative_config.draft_model_config
    assert draft.hf_config.architectures == ["MiMoMTPModel"]
    contract = create_runtime_contract(config)
    assert contract.schema.slots[2].kind == "last_hidden_states"
    assert contract.schema.slots[2].trailing_shape == (16,)


@pytest.mark.parametrize("architecture", ["MiMoMTP", "MiMoV2MTPModel"])
def test_contract_rejects_unsupported_mtp_architectures(architecture):
    config = _config(method="mtp")
    config.speculative_config.draft_model_config.hf_config.architectures = [
        architecture
    ]
    with pytest.raises(ValueError, match="unsupported standalone MTP architecture"):
        create_runtime_contract(config)


def test_contract_fingerprint_changes_with_weights_precision_and_feature_policy():
    config = _config()
    baseline = create_runtime_contract(config)
    config.speculative_config.draft_model_config.revision = "c" * 40
    changed_draft = create_runtime_contract(config)
    assert changed_draft.draft_fingerprint != baseline.draft_fingerprint
    assert changed_draft.target_fingerprint == baseline.target_fingerprint
    config.speculative_config.draft_model_config.revision = "b" * 40
    config.speculative_config.num_speculative_tokens = 5
    assert (
        create_runtime_contract(config).draft_fingerprint != baseline.draft_fingerprint
    )
    config.model_config.dtype = torch.float16
    assert (
        create_runtime_contract(config).target_fingerprint
        != baseline.target_fingerprint
    )
    assert create_runtime_contract(_config(use_aux=True)).schema != baseline.schema


@pytest.mark.parametrize("same_commit", [True, False])
def test_tokenizer_handshake_uses_pinned_commit_not_requested_branch(same_commit):
    from vllm.v1.spec_decode.remote.server import FakeDraftAdapter
    from vllm.v1.spec_decode.remote.session import (
        RemoteDraftError,
        RemoteDraftSession,
        VerifierIdentity,
    )

    config = _config()
    config.model_config.tokenizer = "test/tokenizer"
    config.model_config.tokenizer_revision = ResolvedRevision(
        "a" * 40, "main", repo_id="test/tokenizer"
    )
    expected = create_runtime_contract(config)
    config.model_config.tokenizer_revision = (
        "a" * 40
        if same_commit
        else ResolvedRevision("b" * 40, "main", repo_id="test/tokenizer")
    )
    actual = create_runtime_contract(config)
    assert (
        actual.tokenizer_fingerprint == expected.tokenizer_fingerprint
    ) == same_commit
    assert actual.target_fingerprint == expected.target_fingerprint
    adapter = FakeDraftAdapter(
        vocab_size=32,
        feature_schema=expected.schema,
        capabilities=expected.capabilities,
        draft_checkpoint_fingerprint=expected.draft_fingerprint,
    )
    server = RemoteDraftServer(
        adapter,
        target_fingerprint=expected.target_fingerprint,
        tokenizer_fingerprint=expected.tokenizer_fingerprint,
    )
    server.start("tcp://127.0.0.1:0")
    session = RemoteDraftSession(
        server.endpoint,
        VerifierIdentity(
            verifier_instance_id="tokenizer-revision",
            target_fingerprint=actual.target_fingerprint,
            tokenizer_fingerprint=actual.tokenizer_fingerprint,
            draft_checkpoint_fingerprint=actual.draft_fingerprint,
            method="eagle3",
            num_speculative_tokens=3,
            parallel_drafting=False,
            provided_features=tuple(slot.kind for slot in actual.schema.slots),
            max_num_seqs=1,
            max_num_batched_tokens=4,
            max_model_len=16,
            vocab_size=32,
        ),
        request_timeout_ms=1000,
        startup_timeout_ms=1000,
    )
    try:
        if same_commit:
            session.connect()
            assert session.connected
        else:
            with pytest.raises(RemoteDraftError, match="tokenizer fingerprint"):
                session.connect()
            assert not session.connected
    finally:
        session.close()
        server.stop()


@pytest.mark.parametrize(
    "revision",
    [
        None,
        "main",
        ResolvedRevision("a" * 40, "main", repo_id="other/tokenizer"),
        ResolvedRevision("a" * 40, "main", repo_id="target", repo_type="dataset"),
    ],
)
def test_contract_rejects_unresolved_or_foreign_tokenizer_revision(revision):
    config = _config()
    config.model_config.tokenizer_revision = revision
    with pytest.raises(ValueError, match="checkpoint revision"):
        create_runtime_contract(config)


def test_local_checkpoint_identity_tracks_bytes_instead_of_deployment_path(tmp_path):
    config = _config()
    source, copied = tmp_path / "source", tmp_path / "copied"
    source.mkdir()
    weight_file = source / "model.safetensors"
    save_file({"model.embed_tokens.weight": torch.ones(2, 2)}, weight_file)
    (source / "tokenizer.json").write_text('{"model":{"vocab":{"a":0}}}')
    config.model_config.model = config.model_config.tokenizer = str(source)
    config.model_config.revision = "main"
    original = create_runtime_contract(config)
    shutil.copytree(source, copied)
    config.model_config.model = config.model_config.tokenizer = str(copied)
    duplicate = create_runtime_contract(config)
    assert duplicate == original
    save_file(
        {"model.embed_tokens.weight": torch.zeros(2, 2)}, copied / weight_file.name
    )
    changed = create_runtime_contract(config)
    assert changed.target_fingerprint != original.target_fingerprint
    assert changed.draft_fingerprint == original.draft_fingerprint
    assert changed.tokenizer_fingerprint == original.tokenizer_fingerprint
    (copied / "tokenizer.json").write_text('{"model":{"vocab":{"b":0}}}')
    assert (
        create_runtime_contract(config).tokenizer_fingerprint
        != original.tokenizer_fingerprint
    )


@pytest.mark.parametrize("which", ["target", "draft"])
def test_contract_rejects_a_separate_unverified_weight_source(which):
    config = _config()
    original = create_runtime_contract(config)
    model = (
        config.model_config
        if which == "target"
        else config.speculative_config.draft_model_config
    )
    model.model_weights = model.model
    assert create_runtime_contract(config) == original
    model.model_weights = "other/checkpoint"
    with pytest.raises(ValueError, match="separate model_weights source"):
        create_runtime_contract(config)


def test_contract_rejects_unresolved_revision_or_missing_local_weights(tmp_path):
    config = _config()
    config.model_config.revision = "main"
    with pytest.raises(ValueError, match="immutable checkpoint revision"):
        create_runtime_contract(config)
    config.model_config.hf_config._commit_hash = "d" * 40
    with pytest.raises(ValueError, match="immutable checkpoint revision"):
        create_runtime_contract(config)
    config.model_config.model = str(tmp_path)
    with pytest.raises(ValueError, match="no verifiable"):
        create_runtime_contract(config)


def test_contract_revision_tracks_weight_commit_instead_of_external_config():
    config = _config()
    target = config.model_config
    target.hf_config._commit_hash = "c" * 40
    target.revision = ResolvedRevision("a" * 40, "main", repo_id=target.model)
    first = create_runtime_contract(config)
    target.revision = ResolvedRevision("b" * 40, "main", repo_id=target.model)
    second = create_runtime_contract(config)
    assert second.target_fingerprint != first.target_fingerprint
    target.revision = "b" * 40
    assert create_runtime_contract(config) == second
    target.hf_config._commit_hash = "d" * 40
    assert create_runtime_contract(config) == second


@pytest.mark.parametrize(
    "revision",
    [
        None,
        "main",
        "a" * 39,
        "A" * 40,
        ResolvedRevision("a" * 40, "main", repo_id="other/model"),
        ResolvedRevision("a" * 40, "main", repo_id="target", repo_type="dataset"),
        ResolvedRevision("main", "main", repo_id="target"),
    ],
)
def test_contract_rejects_unresolved_or_foreign_weight_revision(revision):
    config = _config()
    config.model_config.hf_config._commit_hash = "c" * 40
    config.model_config.revision = revision
    with pytest.raises(ValueError, match="checkpoint revision"):
        create_runtime_contract(config)


@pytest.mark.parametrize("checkpoint", ["immutable", "resolved", "local"])
def test_shared_weights_use_weight_revision_instead_of_external_config(
    monkeypatch, tmp_path, checkpoint
):
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    target = _config().model_config
    target.hf_config._commit_hash = "c" * 40
    expected_revision = "a" * 40
    if checkpoint == "resolved":
        target.revision = ResolvedRevision("a" * 40, "main", repo_id=target.model)
    elif checkpoint == "local":
        target.model = str(tmp_path)
        target.revision = "main"
        expected_revision = None
    calls = []

    class ReaderReached(Exception):
        pass

    def read(source, revision, names):
        calls.append((source, revision))
        raise ReaderReached

    monkeypatch.setattr(weights, "selected_checkpoint_weights", read)
    with pytest.raises(ReaderReached):
        weights.materialize_target_shell(target, torch.device("cpu"))
    assert calls == [(target.model, expected_revision)]


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "draft_model"),
        ("parallel_drafting", True),
        ("draft_sample_method", "probabilistic"),
    ],
)
def test_contract_rejects_unimplemented_drafting_modes(field, value):
    config = _config()
    setattr(config.speculative_config, field, value)
    with pytest.raises(ValueError):
        create_runtime_contract(config)


def test_partial_weights_download_only_shards_containing_shared_tensors(
    tmp_path, monkeypatch
):
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    embedding = torch.randn(8, 4)
    head = torch.randn(8, 4)
    save_file(
        {"model.embed_tokens.weight": embedding}, tmp_path / "embedding.safetensors"
    )
    save_file({"lm_head.weight": head}, tmp_path / "head.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "embedding.safetensors",
                    "lm_head.weight": "head.safetensors",
                    "model.layers.0.weight": "never-download.safetensors",
                }
            }
        )
    )
    fetched = []

    def download(repo, filename, *, revision):
        assert repo == "test/target" and revision == "immutable-sha"
        fetched.append(filename)
        return str(tmp_path / filename)

    monkeypatch.setattr(
        weights, "hf_api", lambda: SimpleNamespace(hf_hub_download=download)
    )
    actual = selected_checkpoint_weights(
        "test/target", "immutable-sha", ("model.embed_tokens.weight", "lm_head.weight")
    )
    assert set(fetched) == {
        "model.safetensors.index.json",
        "embedding.safetensors",
        "head.safetensors",
    }
    assert torch.equal(actual["model.embed_tokens.weight"], embedding)
    assert torch.equal(actual["lm_head.weight"], head)


@pytest.mark.parametrize("filename", ["model.safetensors", "weights.safetensors"])
def test_indexless_local_weights_accept_native_safetensor_filenames(tmp_path, filename):
    embedding = torch.randn(8, 4)
    save_file(
        {
            "model.embed_tokens.weight": embedding,
            "model.layers.0.weight": torch.ones(3),
        },
        tmp_path / filename,
    )
    actual = selected_checkpoint_weights(
        str(tmp_path), None, ("model.embed_tokens.weight", "lm_head.weight")
    )
    assert set(actual) == {"model.embed_tokens.weight"}
    torch.testing.assert_close(actual["model.embed_tokens.weight"], embedding)


def test_indexless_local_weights_select_across_shards_and_reject_duplicates(
    tmp_path,
):
    embedding, head = torch.randn(8, 4), torch.randn(8, 4)
    save_file(
        {"model.embed_tokens.weight": embedding}, tmp_path / "embedding.safetensors"
    )
    save_file({"lm_head.weight": head}, tmp_path / "head.safetensors")
    names = ("model.embed_tokens.weight", "lm_head.weight")
    actual = selected_checkpoint_weights(str(tmp_path), None, names)
    torch.testing.assert_close(actual["model.embed_tokens.weight"], embedding)
    torch.testing.assert_close(actual["lm_head.weight"], head)

    save_file({"lm_head.weight": head + 1}, tmp_path / "other.safetensors")
    with pytest.raises(ValueError, match="duplicate.*lm_head.weight"):
        selected_checkpoint_weights(str(tmp_path), None, names)


def test_partial_weight_auth_errors_do_not_fall_back_to_full_checkpoint(monkeypatch):
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    def denied(*args, **kwargs):
        raise PermissionError("access denied")

    monkeypatch.setattr(
        weights, "hf_api", lambda: SimpleNamespace(hf_hub_download=denied)
    )
    with pytest.raises(PermissionError, match="access denied"):
        selected_checkpoint_weights("test/target", "sha", ("lm_head.weight",))


def test_input_batch_preserves_reordered_request_slots_and_rejected_tail():
    planner = DraftStepPlanner(
        DraftBlockPool(32, 4),
        max_num_reqs=3,
        max_model_len=32,
        num_speculative_tokens=3,
    )
    a, b = SequenceKey("v", 1, 0), SequenceKey("v", 2, 0)
    planner.open(a)
    planner.open(b)
    prompt, failures = planner.plan([DraftStepRow(a, 5, 1), DraftStepRow(b, 3, 1)])
    assert not failures
    planner.commit(prompt)
    plan, failures = planner.plan(
        [
            DraftStepRow(b, 4, 2, num_rejected=2, last_sampled=15),
            DraftStepRow(a, 2, 2, last_sampled=19),
        ]
    )
    assert not failures
    batch = make_input_batch(
        plan, torch.tensor([1, 2, 3, 4, 5, 6]), InputBuffers(3, 16, torch.device("cpu"))
    )
    assert batch.idx_mapping.tolist() == [1, 0]
    assert batch.query_start_loc.tolist() == [0, 4, 6]
    assert batch.seq_lens.tolist() == [7, 7]
    assert batch.positions.tolist() == [3, 4, 5, 6, 5, 6]
    assert batch.num_computed_tokens_np.tolist() == [3, 5]
    assert batch.input_ids.tolist() == [1, 2, 3, 4, 5, 6]
    assert not batch.is_padding.any()
    planner.commit(plan)
    assert planner.num_computed(b) == 5
    assert planner.num_computed(a) == 7


def test_final_prompt_chunk_keeps_prefill_metadata_after_an_earlier_chunk():
    planner = DraftStepPlanner(
        DraftBlockPool(32, 4),
        max_num_reqs=1,
        max_model_len=32,
        num_speculative_tokens=3,
    )
    key = SequenceKey("v", 1, 0)
    planner.open(key)
    first, _ = planner.plan(
        [
            DraftStepRow(key, 4, 0, next_prefill_tokens=(5,)),
        ]
    )
    planner.commit(first)
    final, _ = planner.plan([DraftStepRow(key, 3, 1, last_sampled=8)])
    batch = make_input_batch(
        final,
        torch.tensor([5, 6, 7]),
        InputBuffers(1, 16, torch.device("cpu")),
    )
    assert batch.has_prefill
    assert batch.is_prefilling_np.tolist() == [True]
    assert batch.prefill_len_np.tolist() == [7]
    assert batch.positions.tolist() == [4, 5, 6]
