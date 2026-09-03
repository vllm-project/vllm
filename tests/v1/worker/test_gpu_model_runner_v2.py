# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.config import CompilationConfig, ParallelConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_qsa_circular_group_uses_custom_slot_mapping(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    runner.parallel_config = parallel_config
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    runner.model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (),
        num_new_sampled_tokens_per_step=1,
    )
    runner.speculator = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 1
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cuda")

    raw_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    compressed_spec = FullAttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["raw"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8,
                    kv_cache_specs={"raw": raw_spec},
                ),
            ),
            KVCacheGroupSpec(layer_names=["compressed"], kv_cache_spec=compressed_spec),
        ],
    )

    class FakeAttnCGSupport:
        def narrow(self, *args):
            return self

    attn_cg_support = FakeAttnCGSupport()
    monkeypatch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *args: ([], attn_cg_support, [8, 262144]),
    )
    monkeypatch.setattr(
        model_runner_module,
        "maybe_create_adaptive_verification_manager",
        lambda **kwargs: None,
    )

    captured = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(model_runner_module, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 1]
    assert captured["slot_mapping_enabled"] == [False, True]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""

    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
        parallel_config=parallel_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.blocks_per_kv_block = [1]
    block_tables.block_tables = [_BlockTable()]
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3


def test_execute_model_dummy_run_uses_prepare_runtime_dummy_inputs_not_prepare_inputs(
    monkeypatch,
):
    """A runtime (profile/DP-empty) dummy run must build its model_inputs via
    ``ModelState.prepare_runtime_dummy_inputs``, never ``prepare_inputs`` --
    the latter is the real path that would reach a table's hash/gather/pin
    methods (e.g. Qwen4Exp's mmap-staged PLE rows), which a dummy/profile
    run must never touch."""

    class _RecordingModelState:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def prepare_runtime_dummy_inputs(self, input_batch, req_states):
            self.calls.append("prepare_runtime_dummy_inputs")
            return {}

        def prepare_inputs(self, input_batch, req_states):
            pytest.fail(
                "execute_model's dummy-run branch must not call prepare_inputs "
                "-- it is the real path that reaches a table's hash/gather/pin"
            )

    class _StopAtModelCall(Exception):
        pass

    captured: dict[str, Any] = {}

    def _fake_model(**kwargs):
        captured.update(kwargs)
        raise _StopAtModelCall

    num_tokens = 4
    fake_input_batch = SimpleNamespace(
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        positions=torch.arange(num_tokens),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        is_padding=None,
    )
    fake_batch_desc = SimpleNamespace(
        num_tokens=num_tokens,
        num_reqs=1,
        max_query_len=num_tokens,
        cg_mode=CUDAGraphMode.NONE,
        num_active_loras=0,
    )

    runner: Any = GPUModelRunner.__new__(GPUModelRunner)
    runner.lora_config = None
    runner.is_encoder_decoder = False
    runner.cudagraph_manager = None
    runner.dp_size = 1
    runner.dp_rank = 0
    runner.input_buffers = None
    runner.uses_inputs_embeds = False
    runner.is_first_pp_rank = True
    runner.model_state = _RecordingModelState()
    runner.req_states = None
    runner.eplb = SimpleNamespace(prepare_forward=lambda *a, **kw: None)
    runner.step_timing = SimpleNamespace(
        record_batch=lambda *a, **kw: None, forward_start=lambda: None
    )
    runner.model_config = None
    runner.vllm_config = SimpleNamespace(
        parallel_config=ParallelConfig(), compilation_config=CompilationConfig()
    )
    runner.kv_connector = SimpleNamespace(pre_forward=lambda scheduler_output: None)
    runner.model = _fake_model

    monkeypatch.setattr(
        model_runner_module,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (fake_batch_desc, None),
    )
    monkeypatch.setattr(
        InputBatch, "make_dummy", classmethod(lambda cls, *a, **kw: fake_input_batch)
    )

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"dummy": num_tokens},
        total_num_scheduled_tokens=num_tokens,
    )

    with pytest.raises(_StopAtModelCall):
        runner.execute_model(
            scheduler_output, dummy_run=True, skip_attn_for_dummy_run=True
        )

    assert runner.model_state.calls == ["prepare_runtime_dummy_inputs"]
    assert captured["input_ids"] is fake_input_batch.input_ids
