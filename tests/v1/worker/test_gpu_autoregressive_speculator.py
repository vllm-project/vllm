# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import Qwen3Config

from vllm.config.compilation import CUDAGraphMode
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.model_executor.models.exaone4_5_mtp import Exaone4_5_MTP
from vllm.model_executor.models.llama4_eagle import EagleLlama4ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.mistral_eagle import EagleMistralForCausalLM
from vllm.model_executor.models.mistral_large_3_eagle import (
    EagleMistralLarge3ForCausalLM,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends import flash_attn as flash_attn_module
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
    prepare_decode_inputs,
)
from vllm.v1.worker.gpu.spec_decode.draft_model.speculator import (
    StandaloneDraftModelSpeculator,
    prepare_draft_model_prefill_inputs,
)
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    MultiModuleMTPSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        return self.test_draft_model


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


class _MultimodalDraftModel(torch.nn.Module):
    supports_multimodal_embeddings = True

    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


class _TextOnlyDraftModel(torch.nn.Module):
    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


def _mock_base_model_load(monkeypatch):
    monkeypatch.setattr(
        base_spec_module,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        DraftModelSpeculator,
        "_validate_local_argmax_reduction",
        lambda self: None,
    )


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    speculator_cls=_TestSpeculator,
) -> AutoRegressiveSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(speculator_cls)
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


@pytest.mark.parametrize(("hc_mult", "expected"), [(None, 64), (4, 256)])
def test_speculator_uses_draft_model_hidden_size(monkeypatch, hc_mult, expected):
    # Qwen4Exp targets expose multi-stream HC residuals to the drafter.
    monkeypatch.setattr(base_spec_module, "_target_feeds_hc_residual", lambda _: True)
    hf_config = SimpleNamespace()
    if hc_mult is not None:
        hf_config.hc_mult = hc_mult
    draft_model_config = SimpleNamespace(
        hf_config=hf_config,
        get_hidden_size=lambda: 64,
        get_vocab_size=lambda: 32,
    )
    speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=3,
        draft_model_config=draft_model_config,
        use_local_argmax_reduction=False,
        draft_sample_method="greedy",
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=8,
        ),
        model_config=SimpleNamespace(
            max_model_len=32,
            dtype=torch.float32,
            use_fp64_gumbel=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
        ),
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert speculator.hidden_size == expected


@pytest.mark.cpu_test
def test_standalone_constructor_expands_only_the_draft_scheduler(tmp_path):
    """Draft capacity must not expand the target's warmup beyond its buffers."""
    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    ).save_pretrained(tmp_path)
    config = EngineArgs(
        model=str(tmp_path),
        speculative_config={
            "model": str(tmp_path),
            "method": "draft_model",
            "num_speculative_tokens": 2,
        },
        max_model_len=32,
        max_num_batched_tokens=16,
        max_num_seqs=2,
        enable_chunked_prefill=True,
        enforce_eager=True,
        skip_tokenizer_init=True,
        dtype="float32",
    ).create_engine_config()
    target_scheduler = config.scheduler_config
    target_compilation = config.compilation_config
    target_compile_ranges = list(target_compilation.compile_ranges_endpoints)

    speculator = StandaloneDraftModelSpeculator(config, torch.device("cpu"))

    assert target_scheduler.max_num_batched_tokens == 16
    assert speculator.draft_vllm_config.scheduler_config is not target_scheduler
    assert speculator.draft_vllm_config.scheduler_config.max_num_batched_tokens == 18
    assert speculator.input_buffers.input_ids.numel() == 18
    draft_compilation = speculator.draft_vllm_config.compilation_config
    assert target_compilation.compile_ranges_endpoints == target_compile_ranges
    assert draft_compilation is not target_compilation
    assert max(draft_compilation.compile_ranges_endpoints) == 18
    assert (
        draft_compilation.static_forward_context
        is target_compilation.static_forward_context
    )


def test_mm_support_configured_after_model_load(monkeypatch):
    target_model_config = object()
    draft_model_config = object()
    vllm_config = SimpleNamespace(model_config=target_model_config)
    draft_model = _MultimodalDraftModel()

    def init_base(speculator, vllm_config, device):
        speculator.vllm_config = vllm_config
        speculator.device = device
        speculator.max_num_tokens = 4
        speculator.max_num_reqs = 2
        speculator.hidden_size = 3
        speculator.dtype = torch.float32
        speculator.draft_model_config = draft_model_config
        speculator.supports_mm_inputs = False

    checked_configs = []

    def supports_multimodal_inputs(model_config):
        checked_configs.append(model_config)
        return True

    monkeypatch.setattr(DraftModelSpeculator, "__init__", init_base)
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        supports_multimodal_inputs,
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert checked_configs == []
    assert not speculator.supports_mm_inputs
    assert speculator.inputs_embeds is None

    speculator.test_draft_model = draft_model
    speculator.load_model(torch.nn.Module())

    assert checked_configs == [target_model_config]
    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)


def test_load_model_keeps_mm_support_for_capable_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    speculator.test_draft_model = draft_model
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None


def test_load_model_disables_mm_support_for_text_only_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    draft_model = _TextOnlyDraftModel()
    speculator.test_draft_model = draft_model
    warning_messages = []
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )
    monkeypatch.setattr(
        base_spec_module.logger,
        "warning_once",
        lambda message, *args: warning_messages.append(message % args),
    )

    speculator.load_model(torch.nn.Module())

    assert not speculator.supports_mm_inputs
    assert warning_messages == [
        (
            "Draft model _TextOnlyDraftModel does not support external multimodal "
            "embeddings. Embeddings from the target model will not be passed to the "
            "drafter; using text-only draft inputs instead."
        )
    ]


def test_multi_module_mm_support_configured_after_model_load(monkeypatch):
    speculator = object.__new__(MultiModuleMTPSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.cached_draft_input_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.max_num_reqs = 2
    speculator.num_speculative_steps = 3
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        MultiModuleMTPSpeculator,
        "load_draft_model",
        lambda self, target_model, target_attn_layer_names: draft_model,
    )
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)
    assert speculator.cached_draft_input_embeds is not None
    assert speculator.cached_draft_input_embeds.shape == (2, 2, 3)


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [
        (EagleLlama4ForCausalLM, True),
        (EagleMistralForCausalLM, True),
        (EagleMistralLarge3ForCausalLM, True),
        (Exaone4_5_MTP, True),
        (Eagle3LlamaForCausalLM, False),
    ],
)
def test_draft_model_multimodal_embedding_capability(model_cls, expected):
    assert supports_multimodal_embeddings(model_cls) is expected


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


@pytest.mark.parametrize(
    "speculator_cls", [_TestSpeculator, StandaloneDraftModelSpeculator]
)
def test_run_model_only_passes_hidden_states_to_conditioned_drafters(
    monkeypatch, speculator_cls
):
    speculator = _make_speculator(monkeypatch, torch.zeros(4, 3), speculator_cls)
    speculator.model = Mock(return_value=torch.zeros(4, 3))

    speculator._run_model(4, None, None, None)

    kwargs = speculator.model.call_args.kwargs
    if speculator_cls is StandaloneDraftModelSpeculator:
        assert "hidden_states" not in kwargs
    else:
        assert torch.equal(kwargs["hidden_states"], speculator.hidden_states)


@pytest.mark.parametrize(
    ("speculator_cls", "expected_positions"),
    [(_TestSpeculator, [2, 4]), (StandaloneDraftModelSpeculator, [1, 3])],
)
def test_prefill_samples_from_the_position_consumed_by_each_drafter(
    monkeypatch, speculator_cls, expected_positions
):
    speculator = _make_speculator(monkeypatch, torch.zeros(4, 3), speculator_cls)
    speculator.last_token_indices = torch.tensor([1, 3])
    speculator.idx_mapping = torch.tensor([0, 1])
    speculator.draft_tokens = torch.zeros((2, 1), dtype=torch.int64)
    speculator.sample_src_positions = torch.zeros(2, dtype=torch.int64)
    speculator.current_draft_step = torch.tensor(0)
    speculator.temperature = speculator.seeds = speculator.draft_logits = None
    speculator.sample_draft = Mock(return_value=torch.tensor([5, 6]))

    speculator._prefill(2, 4, None, None, None)

    assert speculator.sample_draft.call_args.args[1].tolist() == expected_positions
    assert speculator.sample_src_positions.tolist() == expected_positions
    assert speculator.input_buffers.positions[:2].tolist() == [1, 3]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_standalone_prefill_preserves_tokens_and_right_aligns_rejected_queries():
    """A stale target MoE mask must not hide accepted tokens from draft KV."""
    device = torch.device("cuda")

    def tensor(values, dtype=None):
        return torch.tensor(values, dtype=dtype, device=device)

    buffers = InputBuffers(5, 16, device)
    buffers.input_ids.fill_(-99)
    buffers.positions.fill_(-99)
    batch = InputBatch.make_dummy(3, 9, InputBuffers(5, 16, device))
    batch.input_ids = tensor([10, 11, 12, 20, 21, 30, 31, 98, 99], torch.int32)
    batch.positions = tensor([4, 5, 6, 6, 7, 10, 11, 12, 13])
    batch.query_start_loc = tensor([0, 3, 5, 9], torch.int32)
    batch.query_start_loc_np[:] = [0, 3, 5, 9]
    batch.num_scheduled_tokens[:] = [3, 2, 4]
    batch.seq_lens = tensor([7, 8, 14], torch.int32)
    batch.seq_lens_cpu_upper_bound[:] = torch.tensor([7, 8, 14])
    batch.idx_mapping = tensor([2, 0, 1], torch.int32)
    # Real batches can retain the dummy mask when VLLM_MOE_SKIP_PADDING=0.
    batch.is_padding.fill_(True)
    last_indices = tensor([-99] * 5)
    current_step = tensor(3)
    speculator = object.__new__(StandaloneDraftModelSpeculator)
    speculator.input_buffers = buffers
    speculator.last_token_indices = last_indices
    speculator.current_draft_step = current_step
    speculator.max_num_reqs = 5
    speculator.max_model_len = 32
    speculator.is_dummy_run = tensor(True)

    expanded = speculator.prepare_inputs(
        batch,
        torch.empty(0),
        None,
        tensor([0, 1, 2], torch.int32),
        tensor([0, 0, 2], torch.int32),
        tensor([22, 32, 999, 999, 999], torch.int64),
        tensor([[777, 777, 13, 777, 777], [888] * 5], torch.int32),
    )

    assert expanded.num_tokens == expanded.num_tokens_after_padding == 12
    assert expanded.num_scheduled_tokens.tolist() == [4, 3, 5]
    assert expanded.seq_lens_cpu_upper_bound.tolist() == [8, 9, 15]
    assert expanded.query_start_loc_np.tolist() == [0, 4, 7, 12]
    assert batch.num_tokens == 9
    assert buffers.query_start_loc.tolist() == [0, 4, 7, 12, 12, 12]
    assert buffers.seq_lens.tolist() == [8, 9, 13, 0, 0]
    assert last_indices.tolist() == [3, 6, 11, 0, 0]
    assert current_step.item() == 0
    assert not speculator.is_dummy_run.item()
    assert buffers.is_padding.tolist() == (
        [False] * 7 + [True] * 2 + [False] * 3 + [True] * 4
    )
    valid = ~buffers.is_padding
    assert buffers.input_ids[valid].tolist() == [10, 11, 12, 13, 20, 21, 22, 30, 31, 32]
    assert buffers.positions[valid].tolist() == [4, 5, 6, 7, 6, 7, 8, 10, 11, 12]

    speculator.idx_mapping = batch.idx_mapping
    speculator.block_tables = BlockTables([4], 5, 16, [8], device, [4])
    for state in range(3):
        speculator.block_tables.append_block_ids(
            state, (list(range(state * 8 + 1, state * 8 + 9)),), overwrite=True
        )
    speculator.block_tables.apply_staged_writes()
    speculator.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["draft"])]
    )
    speculator._build_draft_attn_metadata = Mock(return_value={})

    _, slots = speculator.prepare_attn(
        expanded, BatchExecutionDescriptor(CUDAGraphMode.NONE, 16, 5)
    )

    assert slots["draft"].tolist() == (
        [72, 73, 74, 75, 10, 11, 12, -1, -1, 46, 47, 48] + [-1] * 4
    )
    assert slots["draft"].data_ptr() == speculator.block_tables.slot_mappings.data_ptr()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("seq_len", [15, 16])
@pytest.mark.parametrize("num_rejected", [0, 1, 3])
def test_standalone_prefill_keeps_causal_origin_at_context_boundary(
    seq_len, num_rejected
):
    device = torch.device("cuda")
    buffers = InputBuffers(1, 5, device)
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=4,
        input_ids=torch.arange(10, 14, device=device),
        positions=torch.arange(seq_len - 4, seq_len, device=device),
        is_padding=torch.zeros(4, dtype=torch.bool, device=device),
        query_start_loc=torch.tensor([0, 4], device=device),
        seq_lens=torch.tensor([seq_len], device=device),
        idx_mapping=torch.tensor([0], device=device),
    )
    last_indices = torch.zeros(1, dtype=torch.int64, device=device)
    prepare_draft_model_prefill_inputs(
        last_indices,
        torch.zeros((), dtype=torch.int64, device=device),
        buffers,
        batch,
        torch.tensor([4 - num_rejected], device=device),
        torch.tensor([num_rejected], device=device),
        torch.tensor([42], device=device),
        torch.tensor([99], device=device),
        1,
        16,
    )

    has_bonus = seq_len - num_rejected < 16
    num_padding = num_rejected + (not has_bonus)
    valid_ids = list(range(10, 14 - num_rejected)) + ([42] if has_bonus else [])
    assert buffers.is_padding.tolist() == [True] * num_padding + [False] * len(
        valid_ids
    )
    assert buffers.input_ids[num_padding:].tolist() == valid_ids
    assert buffers.positions[num_padding:].tolist() == list(
        range(seq_len - 4, seq_len - num_rejected + has_bonus)
    )
    assert buffers.seq_lens.item() <= 16
    assert buffers.seq_lens.item() - 5 + num_padding == seq_len - 4
    assert last_indices.item() == 4


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("seq_len_offset", [0, 1])
def test_decode_inputs_include_standalone_bonus_without_changing_existing_drafters(
    seq_len_offset,
):
    device = torch.device("cuda")
    buffers = InputBuffers(4, 4, device)
    buffers.positions[:3] = torch.tensor([7, 12, 15], device=device)
    sample_positions = torch.tensor([7, 12, 15, 0], device=device)

    prepare_decode_inputs(
        torch.tensor([11, 22, 33], device=device),
        torch.tensor([8, 14, 16], device=device),
        torch.tensor([0, 2, 0], device=device),
        buffers,
        sample_positions,
        max_model_len=16,
        max_num_reqs=4,
        seq_len_offset=seq_len_offset,
    )

    assert buffers.input_ids[:3].tolist() == [11, 22, 33]
    assert buffers.positions[:3].tolist() == [8, 13, 15]
    assert sample_positions.tolist() == [8, 13, 16, 0]
    assert buffers.seq_lens.tolist() == [9 + seq_len_offset, 13 + seq_len_offset, 16, 0]
    assert buffers.query_start_loc.tolist() == [0, 1, 2, 3, 3]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_standalone_decode_slots_mask_context_limit_and_dummy_runs():
    """Clamped positions must not overwrite the final live KV token."""
    device = torch.device("cuda")
    speculator = object.__new__(StandaloneDraftModelSpeculator)
    speculator.max_model_len = 16
    speculator.input_buffers = InputBuffers(3, 4, device)
    speculator.input_buffers.positions[:2] = 15
    speculator.input_buffers.query_start_loc[:] = torch.tensor(
        [0, 1, 2, 2], device=device
    )
    speculator.idx_mapping = torch.tensor([0, 1], device=device)
    speculator.sample_src_positions = torch.tensor([15, 16, 0], device=device)
    speculator.is_dummy_run = torch.zeros((), dtype=torch.bool, device=device)
    speculator.block_tables = BlockTables([4], 3, 4, [4], device, [4])
    for state in range(2):
        speculator.block_tables.append_block_ids(
            state, (list(range(state * 4 + 1, state * 4 + 5)),), overwrite=True
        )
    speculator.block_tables.apply_staged_writes()

    for dummy_run in (False, True, False):
        speculator.is_dummy_run.fill_(dummy_run)
        slots = speculator.compute_decode_slot_mappings(2, 3)

        assert slots.tolist() == [[-1 if dummy_run else 19, -1, -1]]
        assert slots.data_ptr() == speculator.block_tables.slot_mappings.data_ptr()


@pytest.mark.parametrize(
    (
        "method_name",
        "cg_mode",
        "expected_eager_calls",
        "expected_graph_replays",
    ),
    [
        ("_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_multi_step_decode", CUDAGraphMode.FULL, 0, 3),
        ("_fused_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_fused_multi_step_decode", CUDAGraphMode.FULL, 0, 1),
    ],
)
def test_multi_step_decode_replays_captured_graph_as_expected(
    method_name,
    cg_mode,
    expected_eager_calls,
    expected_graph_replays,
):
    speculator = object.__new__(_TestSpeculator)
    speculator.num_speculative_steps = 4
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2),
        query_start_loc=torch.arange(3),
    )
    speculator.idx_mapping = torch.arange(2)
    generate_draft = Mock()
    speculator._generate_draft = generate_draft
    run_fullgraph = Mock()
    speculator.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=run_fullgraph)
    batch_desc = BatchExecutionDescriptor(
        cg_mode=cg_mode,
        num_tokens=2,
        num_reqs=2,
    )

    getattr(speculator, method_name)(
        num_reqs=2,
        skip_attn=True,
        batch_desc=batch_desc,
        seq_lens_cpu_upper_bound=None,
        num_tokens_across_dp=None,
    )

    assert generate_draft.call_count == expected_eager_calls
    assert run_fullgraph.call_count == expected_graph_replays


def test_update_draft_decode_metadata_updates_fa3_scheduler_metadata(
    monkeypatch,
):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(8, dtype=torch.int32)
    builder.cache_config = SimpleNamespace(cache_dtype="bfloat16")
    builder.kv_cache_dtype = torch.bfloat16
    builder.num_heads_q = 2
    builder.num_heads_kv = 1
    builder.headdim = 128
    builder.block_size = 16
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.aot_sliding_window = None

    expected = torch.tensor([7, 8, 9], dtype=torch.int32)

    def fake_get_scheduler_metadata(**kwargs):
        return expected

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=3,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.tensor([5, 6], dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(3, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=2,
        num_prefill_reqs=0,
        num_decode_tokens=3,
        num_prefill_tokens=0,
        scheduler_metadata=torch.tensor([-1, -1, -1], dtype=torch.int32),
        prefix_scheduler_metadata=None,
        max_num_splits=4,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert torch.equal(metadata.scheduler_metadata, expected)
    assert torch.equal(builder.scheduler_metadata[:3], expected)


def test_update_draft_decode_metadata_skips_without_scheduler_metadata(monkeypatch):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(4, dtype=torch.int32)

    called = False

    def fake_get_scheduler_metadata(**kwargs):
        nonlocal called
        called = True
        return torch.tensor([1], dtype=torch.int32)

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=1,
        num_prefill_reqs=0,
        num_decode_tokens=1,
        num_prefill_tokens=0,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=1,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert not called
    assert metadata.scheduler_metadata is None
