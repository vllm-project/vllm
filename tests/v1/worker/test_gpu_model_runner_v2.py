# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import OPTConfig

import vllm.v1.worker.gpu.model_runner as model_runner_module
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import (
    combine_sampled_and_draft_tokens,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu.states import RequestState


def test_non_last_pp_rank_uses_global_batch_for_sample_feedback():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.is_last_pp_rank = False
    local_batch = object()
    global_batch = SimpleNamespace(idx_mapping=object())
    runner.pcp_manager = SimpleNamespace(
        global_batch=global_batch,
        restore_for_sampling=Mock(),
    )
    runner.pp_handler = SimpleNamespace(receive=Mock(return_value=False))
    runner.postprocess_num_computed_tokens = Mock()
    runner.model_state = SimpleNamespace(postprocess_state=Mock())
    runner.kv_connector = SimpleNamespace(post_forward=Mock(return_value=None))
    runner.eplb = SimpleNamespace(step=Mock())
    runner.execute_model_state = ExecuteModelState(
        input_batch=local_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        dp_sync=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
    )

    runner.sample_tokens(None)

    runner.pp_handler.receive.assert_called_once_with(global_batch)
    runner.postprocess_num_computed_tokens.assert_called_once_with(global_batch)
    runner.model_state.postprocess_state.assert_called_once_with(
        global_batch.idx_mapping, 0
    )
    runner.pcp_manager.restore_for_sampling.assert_not_called()


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
    runner.jit_warmup_registry = JitWarmupRegistry(runner.vllm_config)
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
        lambda *args, **kwargs: ([], attn_cg_support, [8, 262144]),
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


def _make_capture_runner(captured: bool) -> GPUModelRunner:
    """Minimal V2 runner for capture_model: fakes everything except the
    cudagraph_manager's needs_capture decision."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.cudagraph_manager = SimpleNamespace(
        needs_capture=lambda: captured,
        capture=lambda *args, **kwargs: None,
    )
    runner.lora_config = None
    runner.maybe_setup_dummy_loras = lambda _cfg: contextlib.nullcontext()
    runner.speculator = None
    runner.adaptive_verification = None
    runner.model = None
    runner.input_buffers = None
    runner.pcp_manager = None
    runner.intermediate_tensors = None
    runner.block_tables = None
    runner.attn_groups = None
    runner.kv_cache_config = None
    runner.use_aux_hidden_state_outputs = False
    runner.kv_connector = model_runner_module.NO_OP_KV_CONNECTOR
    return runner


def test_capture_model_locks_workspace_after_capture(monkeypatch):
    """A workspace resize after capture frees the buffer the captured graphs
    baked in, so capture_model must lock the workspace before returning
    (https://github.com/vllm-project/vllm/issues/55336)."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model()

    assert lock_calls == ["lock"]


def test_capture_model_skips_lock_when_nothing_captured(monkeypatch):
    """With no graphs to capture (e.g. enforce_eager) there is nothing baked
    into the workspace, so the early return must not lock it."""
    runner = _make_capture_runner(captured=False)
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    assert runner.capture_model() == 0
    assert lock_calls == []


def test_capture_model_profile_only_skips_lock(monkeypatch):
    """The memory-profiling capture pass runs before kernel warmup and the
    real capture; locking there would stop the warmup from growing the
    workspace to its scheduler-realistic size."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model(profile_only=True)

    assert lock_calls == []


def _prefix_cache_requests(tmp_path, steps, temperature):
    # Only configuration is needed; no model weights or tokenizer are loaded.
    # Two 16-token cache blocks cover 32 tokens of each identical 33-token prompt.
    OPTConfig(
        architectures=["OPTForCausalLM"],
        vocab_size=128,
        hidden_size=16,
        ffn_dim=32,
        num_attention_heads=1,
        num_hidden_layers=1,
        max_position_embeddings=128,
    ).save_pretrained(tmp_path)
    scheduler = create_scheduler(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=128,
        max_num_batched_tokens=64,
        max_num_seqs=1,
        num_speculative_tokens=steps,
        enable_prefix_caching=True,
        block_size=16,
        use_v2_model_runner=True,
    )
    requests = create_requests(
        num_requests=2,
        num_tokens=33,
        same_prompt=True,
        max_tokens=1,
    )
    for request in requests:
        request.sampling_params.temperature = temperature
    return scheduler, requests


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "temperature,target_token,block_verification",
    [
        pytest.param(1.0, 17, True, id="stochastic-block"),
        pytest.param(1.0, 17, False, id="stochastic-standard"),
        pytest.param(0.0, 17, True, id="greedy-control"),
        pytest.param(1.0, 0, True, id="valid-zero-control"),
    ],
)
@torch.inference_mode()
def test_cached_request_padding_reuses_slot_safely(
    tmp_path, temperature, target_token, block_verification
):
    """Follow a cached DSpark request from scheduling through GPU verification.

    Scheduler and GPU state transitions are real; model outputs are controlled.
    Greedy and target-zero cases guard against mistaking every zero for a bug.
    """
    steps = 3  # The serving K3 recipe; the failure does not depend on larger K.
    scheduler, (first, second) = _prefix_cache_requests(tmp_path, steps, temperature)
    vocab = 128
    device = torch.device("cuda")
    # Two slots let us check that resetting one request leaves the other alone.
    states = RequestState(2, 128, 64, steps, vocab, device)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.req_states = states
    runner.speculator = object.__new__(DSparkSpeculator)
    # Draft scores live separately from RequestState.draft_tokens.
    q = torch.zeros(2, steps, vocab, device=device)
    runner.speculator.draft_logits = q
    runner.adaptive_verification = None
    runner.pooling_runner = None
    runner.pp_handler = None
    runner.encoder_cache = None
    runner.prompt_logprobs_worker = None
    runner.is_last_pp_rank = False
    runner.sampler = None
    # Stub unrelated model/LoRA/block-table services, not request-slot handling.
    runner.model_state = Mock()
    runner.block_tables = Mock()
    runner.lora_state = Mock()

    # A starts cold: all 33 prompt tokens must be scheduled.
    scheduler.add_request(first)
    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens[first.request_id] == 33
    runner.add_requests(prefill)
    old_slot = states.req_id_to_index[first.request_id]
    # Stand in for A's last draft: strongly prefer token 31, making zero very
    # unlikely. Reusing these scores with B's zero IDs is the corruption trigger.
    q[old_slot].zero_()
    q[old_slot, :, 31] = 32.0
    states.draft_tokens[old_slot].fill_(31)
    untouched = q[1 - old_slot].clone()
    # A's controlled one-token output reaches max_tokens=1. Its worker slot can
    # now be reused, while its first 32 prompt tokens remain in the prefix cache.
    scheduler.update_from_output(
        prefill,
        ModelRunnerOutput(
            req_ids=[first.request_id],
            req_id_to_index={first.request_id: 0},
            sampled_token_ids=[[17]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    # B repeats A's prompt. Assert the trigger before testing the repair:
    # A has finished, B hits 32 cached tokens, and the remaining token is padded.
    # The scheduler's -1 markers become zero draft IDs in the worker below.
    scheduler.add_request(second)
    output = scheduler.schedule()
    assert first.request_id in output.finished_req_ids
    assert output.scheduled_new_reqs[0].num_computed_tokens == 32
    assert output.scheduled_spec_decode_tokens[second.request_id] == [-1] * steps
    num_tokens = output.num_scheduled_tokens[second.request_id]
    assert num_tokens == 1 + steps
    runner.finish_requests(output)
    runner.add_requests(output)
    slot = states.req_id_to_index[second.request_id]
    # A different slot would miss the stale-state bug; other slots must be intact.
    assert slot == old_slot
    assert torch.equal(q[1 - slot], untouched)

    # Use production kernels to assemble B's inputs instead of hand-building the
    # zero draft IDs. This connects scheduler padding to what verification sees.
    idx_mapping = torch.tensor([slot], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    input_ids = torch.full((num_tokens,), -1, dtype=torch.int32, device=device)
    positions = torch.empty(num_tokens, dtype=torch.int64, device=device)
    seq_lens = torch.empty(1, dtype=torch.int32, device=device)
    prepare_prefill_inputs(
        input_ids,
        states.next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        states.all_token_ids.gpu,
        states.prefill_len.gpu,
        states.num_computed_tokens.gpu,
    )
    prepare_pos_seq_lens(
        idx_mapping,
        query_start_loc,
        states.num_computed_tokens.gpu,
        positions,
        seq_lens,
    )
    logits_indices = combine_sampled_and_draft_tokens(
        input_ids,
        idx_mapping,
        states.last_sampled_tokens,
        query_start_loc,
        seq_lens,
        states.prefill_len.gpu,
        states.draft_tokens,
        query_start_loc,
        num_tokens,
    )
    # A failure here means the cache-hit/input-assembly setup changed, rather
    # than demonstrating stale probabilities in the verifier.
    assert input_ids.tolist() == [second.prompt_token_ids[-1]] + [0] * steps
    assert positions.tolist() == list(range(32, 32 + num_tokens))

    # Replace a model forward with scores that overwhelmingly favor target_token.
    # For target 17, zero is unlikely, but still more likely than A's draft said.
    # Target 0 is a control: valid zero output must remain possible after the fix.
    target = torch.zeros(num_tokens, vocab, device=device)
    target[:, target_token] = 24.0
    sampled, count = rejection_sample(
        target,
        q,
        input_ids[logits_indices],
        query_start_loc,
        positions[logits_indices],
        idx_mapping,
        torch.full((num_tokens,), slot, dtype=torch.int32, device=device),
        torch.arange(num_tokens, dtype=torch.int32, device=device),
        torch.full((2,), temperature, device=device),
        torch.tensor([123, 456], dtype=torch.int64, device=device),
        steps,
        use_block_verification=block_verification,
    )
    torch.accelerator.synchronize()
    emitted = sampled[0, : int(count[0])].tolist()
    # Without the reset, stochastic cases emit [0, ..., 0, 17]: stale draft
    # probabilities make the verifier accept B's placeholders. Greedy stays clean.
    assert emitted and all(token == target_token for token in emitted), emitted
    if target_token != 0:
        # Rejecting the first zero placeholder returns only its replacement token.
        assert len(emitted) == 1
