# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.model_executor.models.exaone4_5_mtp import Exaone4_5_MTP
from vllm.model_executor.models.llama4_eagle import EagleLlama4ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.mistral_eagle import EagleMistralForCausalLM
from vllm.model_executor.models.mistral_large_3_eagle import (
    EagleMistralLarge3ForCausalLM,
)
from vllm.v1.attention.backends import flash_attn as flash_attn_module
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.mla import indexer as indexer_module
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
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
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
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


def _speculator_cls(advance_draft_positions: bool) -> type[_TestSpeculator]:
    class _Speculator(_TestSpeculator):
        @property
        def advance_draft_positions(self) -> bool:
            return advance_draft_positions

    return _Speculator


def _make_configured_speculator(
    *,
    advance_draft_positions: bool = True,
    backend_supports_updates: bool = True,
    share_mtp_topk_indices: bool | None = None,
    num_nextn_predict_layers: int = 1,
    num_speculative_steps: int = 3,
):
    speculator = object.__new__(_speculator_cls(advance_draft_positions))
    speculator.num_speculative_steps = num_speculative_steps
    speculator.attn_groups = [
        [
            SimpleNamespace(
                supports_draft_decode_metadata_update=backend_supports_updates,
                backend=SimpleNamespace(get_name=lambda: "MOCK_BACKEND"),
            )
        ]
    ]
    speculator.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_nextn_predict_layers=num_nextn_predict_layers)
    )
    if share_mtp_topk_indices is not None:
        speculator.share_mtp_topk_indices = share_mtp_topk_indices
    return speculator


@pytest.mark.parametrize(
    ("advance", "supports", "expected_fused"),
    [
        (True, True, True),
        (True, False, False),
        # Non-advancing configs must not bypass the backend support check:
        # pre-change this enabled fused mode and crashed with
        # NotImplementedError at the first propose.
        (False, True, True),
        (False, False, False),
    ],
)
def test_configure_fused_multi_step_decode_requires_declared_support(
    advance, supports, expected_fused
):
    speculator = _make_configured_speculator(
        advance_draft_positions=advance,
        backend_supports_updates=supports,
    )

    speculator._configure_fused_multi_step_decode()

    assert speculator.use_fused_multi_step_decode is expected_fused


def test_configure_fused_multi_step_decode_disabled_for_single_step():
    speculator = _make_configured_speculator(num_speculative_steps=1)

    speculator._configure_fused_multi_step_decode()

    assert speculator.use_fused_multi_step_decode is False


def test_configure_fused_multi_step_decode_rejects_multi_layer_index_sharing():
    # With index sharing, draft steps reuse step-0 topk rows from per-layer
    # buffers while step -> layer selection cycles through the predictor
    # layers; more than one layer would read rows another layer never wrote.
    speculator = _make_configured_speculator(
        share_mtp_topk_indices=True,
        num_nextn_predict_layers=2,
    )

    with pytest.raises(ValueError, match="single MTP layer"):
        speculator._configure_fused_multi_step_decode()


def test_configure_fused_multi_step_decode_allows_multi_layer_without_sharing():
    speculator = _make_configured_speculator(num_nextn_predict_layers=2)

    speculator._configure_fused_multi_step_decode()

    assert speculator.use_fused_multi_step_decode is True


def _make_fused_loop_speculator(
    advance_draft_positions: bool, events: list[str], metadata: dict
):
    speculator = object.__new__(_speculator_cls(advance_draft_positions))
    speculator.num_speculative_steps = 4
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2),
        query_start_loc=torch.arange(3),
    )
    speculator.idx_mapping = torch.arange(2)
    speculator.block_tables = SimpleNamespace(
        compute_slot_mappings=lambda *a, **k: events.append("slots")
    )

    def _generate_draft(*args, **kwargs):
        # Represents one captured draft forward: the seq_lens increment
        # (_update_draft_inputs_kernel) launches at its tail.
        events.append("generate")

    speculator._generate_draft = _generate_draft

    def _update(received_metadata):
        assert received_metadata is metadata
        events.append("update")

    return speculator, _update


def test_generate_fused_drafts_reuses_metadata_and_skips_terminal_update():
    metadata = {"layer": object()}
    events: list[str] = []
    speculator, update = _make_fused_loop_speculator(True, events, metadata)
    speculator.attn_groups = [[SimpleNamespace(update_draft_decode_metadata=update)]]

    speculator._generate_fused_drafts(2, 2, metadata, None, None, CUDAGraphMode.NONE)

    # The seq_lens increment (tail of _generate_draft) must precede each
    # metadata update, the same metadata object is reused across steps,
    # and no update runs after the terminal forward.
    assert events == [
        "generate",
        "slots",
        "update",
        "generate",
        "slots",
        "update",
        "generate",
    ]


def test_generate_fused_drafts_non_advancing_never_updates_metadata():
    metadata = {"layer": object()}
    events: list[str] = []
    speculator, update = _make_fused_loop_speculator(False, events, metadata)
    speculator.attn_groups = [[SimpleNamespace(update_draft_decode_metadata=update)]]

    speculator._generate_fused_drafts(2, 2, metadata, None, None, CUDAGraphMode.NONE)

    assert events == ["generate", "generate", "generate"]


def _make_indexer_builder(**overrides) -> DeepseekV32IndexerMetadataBuilder:
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.compress_ratio = 1
    builder.kv_cache_spec = SimpleNamespace(num_states=16)
    builder.num_sms = 8
    for key, value in overrides.items():
        setattr(builder, key, value)
    return builder


def _make_indexer_metadata(
    common_seq_lens: torch.Tensor,
    *,
    dcp: bool,
) -> DeepseekV32IndexerMetadata:
    num_reqs = common_seq_lens.shape[0]
    decode_seq_lens = torch.zeros(num_reqs, 1, dtype=torch.int32)
    schedule_metadata = torch.full((9, 2), -1, dtype=torch.int32)
    decode = DeepSeekV32IndexerDecodeMetadata(
        block_table=torch.arange(num_reqs * 4, dtype=torch.int32).view(num_reqs, 4),
        seq_lens=decode_seq_lens,
        decode_lens=torch.ones(num_reqs, dtype=torch.int32),
        requires_padding=False,
        schedule_metadata=schedule_metadata[:5],
        global_seq_lens=common_seq_lens if dcp else None,
        indices=torch.arange(num_reqs, dtype=torch.int32) if dcp else None,
    )
    return DeepseekV32IndexerMetadata(
        seq_lens=common_seq_lens,
        max_seq_len=64,
        slot_mapping=torch.zeros(num_reqs, dtype=torch.int64),
        num_decodes=num_reqs,
        num_decode_tokens=num_reqs,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )


def _patch_indexer_deep_gemm(monkeypatch, refreshed: list) -> torch.Tensor:
    monkeypatch.setattr(indexer_module, "has_deep_gemm", lambda: True)
    monkeypatch.setattr(indexer_module.current_platform, "is_cuda", lambda: True)
    marker = torch.full((5, 2), 7, dtype=torch.int32)

    def fake_get_paged_mqa_logits_metadata(seq_lens, *args, **kwargs):
        refreshed.append(seq_lens)
        return marker

    monkeypatch.setattr(
        indexer_module,
        "get_paged_mqa_logits_metadata",
        fake_get_paged_mqa_logits_metadata,
    )
    return marker


def test_indexer_update_draft_decode_metadata_non_dcp(monkeypatch):
    builder = _make_indexer_builder()
    common = torch.tensor([10, 20, 30], dtype=torch.int32)
    metadata = _make_indexer_metadata(common, dcp=False)
    refreshed: list = []
    marker = _patch_indexer_deep_gemm(monkeypatch, refreshed)

    builder.update_draft_decode_metadata(metadata)

    # Field-by-field equivalence with a fresh build: the per-token bound
    # for the decode_lens == 1 draft batch is seq_lens[b].
    assert torch.equal(metadata.decode.seq_lens, common.view(3, 1).to(torch.int32))
    # The raw alias path reads metadata.seq_lens; global_seq_lens stays
    # absent and the schedule slice is refreshed in place.
    assert metadata.decode.global_seq_lens is None
    assert torch.equal(metadata.decode.schedule_metadata, marker)
    assert len(refreshed) == 1
    assert refreshed[0] is metadata.decode.seq_lens


def test_indexer_update_draft_decode_metadata_dcp_localizes(monkeypatch):
    builder = _make_indexer_builder(dcp_world_size=2, dcp_rank=1)
    common = torch.tensor([10, 20, 30], dtype=torch.int32)
    metadata = _make_indexer_metadata(common, dcp=True)
    refreshed: list = []
    marker = _patch_indexer_deep_gemm(monkeypatch, refreshed)
    block_table_before = metadata.decode.block_table.clone()
    indices_before = metadata.decode.indices.clone()

    builder.update_draft_decode_metadata(metadata)

    # Rank-local bounds recomputed from the advanced global lengths
    # (never an increment): [10, 20, 30] // 2 on rank 1 -> [5, 10, 15].
    assert torch.equal(
        metadata.decode.seq_lens, torch.tensor([[5], [10], [15]], dtype=torch.int32)
    )
    # The alias must be read, never written: it stays the same tensor
    # object with its advanced values intact.
    assert metadata.decode.global_seq_lens is common
    assert torch.equal(common, torch.tensor([10, 20, 30], dtype=torch.int32))
    assert torch.equal(metadata.decode.schedule_metadata, marker)
    assert torch.equal(metadata.decode.block_table, block_table_before)
    assert torch.equal(metadata.decode.indices, indices_before)
    assert len(refreshed) == 1


def test_indexer_update_draft_decode_metadata_propagates_clamped_lens(
    monkeypatch,
):
    # The speculator's increment kernel clamps seq_lens at max_model_len;
    # the hook must propagate the clamped value as-is (clamp boundary of
    # the field-by-field equivalence).
    builder = _make_indexer_builder()
    common = torch.tensor([4096, 4096], dtype=torch.int32)
    metadata = _make_indexer_metadata(common, dcp=False)
    refreshed: list = []
    _patch_indexer_deep_gemm(monkeypatch, refreshed)

    builder.update_draft_decode_metadata(metadata)

    assert torch.equal(
        metadata.decode.seq_lens, torch.tensor([[4096], [4096]], dtype=torch.int32)
    )


def test_indexer_update_draft_decode_metadata_no_decode_is_noop(monkeypatch):
    builder = _make_indexer_builder()
    common = torch.tensor([10], dtype=torch.int32)
    metadata = _make_indexer_metadata(common, dcp=False)
    metadata.decode = None
    refreshed: list = []
    _patch_indexer_deep_gemm(monkeypatch, refreshed)

    builder.update_draft_decode_metadata(metadata)

    assert refreshed == []


def _make_indexer_builder_via_init(monkeypatch) -> DeepseekV32IndexerMetadataBuilder:
    # The support flag is CUDA-scoped; patch the platform so the env
    # gating logic is exercised on any host.
    monkeypatch.setattr(indexer_module.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(indexer_module, "num_compute_units", lambda _: 8)
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=4),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3, enable_adaptive_verification=False
        ),
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda default: "fp8"
        ),
        model_config=SimpleNamespace(max_model_len=64),
        num_speculative_tokens=3,
    )
    return DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=SimpleNamespace(num_states=16),
        layer_names=["model.layers.61.self_attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        block_table_width=4,
    )


@pytest.mark.parametrize(
    ("enable", "disable", "expected"),
    [
        ("0", "0", False),
        ("1", "0", True),
        ("1", "1", False),
        ("0", "1", False),
    ],
)
def test_indexer_builder_support_flag_env_gating(
    monkeypatch, enable, disable, expected
):
    monkeypatch.setenv("VLLM_ENABLE_FUSED_DRAFT_SPARSE_MLA", enable)
    monkeypatch.setenv("VLLM_DISABLE_FUSED_DRAFT_SPARSE_MLA", disable)

    builder = _make_indexer_builder_via_init(monkeypatch)

    assert builder.supports_draft_decode_metadata_update is expected


def test_mtp_speculator_index_sharing_toggles(monkeypatch):
    speculator = object.__new__(MTPSpeculator)
    calls: list = []
    speculator.model = SimpleNamespace(
        model=SimpleNamespace(
            set_skip_topk=lambda skip: calls.append(("skip", skip)),
            compact_topk_indices=lambda ids: calls.append(("compact",)),
        )
    )
    speculator.num_speculative_steps = 3
    speculator.last_token_indices = torch.arange(4)
    speculator.share_mtp_topk_indices = True

    speculator.on_prefill_begin(2)
    speculator.on_prefill_end(2)
    speculator.on_multi_step_decode_begin(2)
    speculator.on_multi_step_decode_end(2)

    assert calls == [
        ("skip", False),
        ("compact",),
        ("skip", True),
        ("skip", False),
    ]


def test_mtp_speculator_without_index_sharing_never_hits_cache():
    speculator = object.__new__(MTPSpeculator)
    calls: list = []

    def _fail(*args, **kwargs):
        calls.append(args)

    speculator.model = SimpleNamespace(
        model=SimpleNamespace(
            set_skip_topk=_fail,
            compact_topk_indices=_fail,
        )
    )
    speculator.num_speculative_steps = 3
    speculator.last_token_indices = torch.arange(4)
    speculator.share_mtp_topk_indices = False

    speculator.on_prefill_begin(2)
    speculator.on_prefill_end(2)
    speculator.on_multi_step_decode_begin(2)
    speculator.on_multi_step_decode_end(2)

    assert calls == []
