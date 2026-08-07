# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import MappingProxyType, SimpleNamespace

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
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
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
        "Draft model _TextOnlyDraftModel does not support external multimodal "
        "embeddings. Embeddings from the target model will not be passed to the "
        "drafter; using text-only draft inputs instead."
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


def test_mrv2_sparse_mla_reuses_topk_with_sequential_draft_positions(monkeypatch):
    import vllm.v1.attention.ops.flashmla as flashmla
    import vllm.v1.worker.gpu.sparse_mla_offload as sparse_mla_offload
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.v1.worker.gpu.sparse_mla_offload import (
        SparseMLALayerView,
        SparseMLAOffloadManager,
    )

    speculator = object.__new__(_TestSpeculator)
    manager_buffers = {
        "topk_logical_ids": torch.full((1, 3, 2), -1, dtype=torch.int32),
        "tp_fence_token": torch.full((1,), -1, dtype=torch.int32),
        "request_block_ids": torch.zeros((1, 1), dtype=torch.int32),
        "request_num_blocks": torch.ones(1, dtype=torch.int32),
        "request_num_tokens": torch.ones(1, dtype=torch.int32),
        "request_active": torch.ones(1, dtype=torch.bool),
        "resident_main_kv": torch.zeros(1, 1, 1),
        "resident_logical_ids": torch.full((1, 1), -1, dtype=torch.int64),
        "resident_last_access": torch.zeros((1, 1), dtype=torch.int64),
        "newest_main_kv": torch.zeros(1, 1, 1),
        "newest_logical_ids": torch.full((1, 1), -1, dtype=torch.int64),
        "topk_physical_ids": torch.full((1, 3, 2), -1, dtype=torch.int32),
        "topk_hit_mask": torch.zeros((1, 3, 2), dtype=torch.bool),
        "miss_logical_ids": torch.full((1, 3, 2), -1, dtype=torch.int32),
        "miss_victim_slots": torch.full((1, 3, 2), -1, dtype=torch.int32),
        "miss_counts": torch.zeros((1, 3), dtype=torch.int32),
        "accepted_counts": torch.zeros((1, 3), dtype=torch.int32),
    }
    fenced_steps = []

    def broadcast(fence_token, src=0):
        assert src == 0
        fenced_steps.append(int(fence_token.item()))

    tp_group = SimpleNamespace(broadcast=broadcast)
    monkeypatch.setattr(
        sparse_mla_offload, "get_tp_group", lambda: tp_group, raising=False
    )
    manager = SparseMLAOffloadManager.__new__(SparseMLAOffloadManager)
    manager._local_buffers = manager_buffers
    manager._tp_group = tp_group
    manager._closing = manager._closed = False
    layer_view = SparseMLALayerView(
        layer_name="main.0",
        layer_index=0,
        is_host_writer=True,
        main_host_kv=torch.zeros(4, 1),
        main_host_kv_uva=torch.zeros(4, 1),
        local_buffers=MappingProxyType(manager_buffers),
        side_stream=None,
        fork_ready_events=(),
        miss_ready_events=(),
    )
    speculator._sparse_mla_offload_manager = manager
    speculator.sparse_mla_offload_manager = manager
    speculator.current_draft_step = torch.zeros(1, dtype=torch.int64)
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(3, dtype=torch.int32),
        positions=torch.zeros(3, dtype=torch.int64),
        query_start_loc=torch.zeros(2, dtype=torch.int32),
        seq_lens=torch.zeros(1, dtype=torch.int32),
    )
    speculator.hidden_states = torch.zeros(3, 1)
    speculator.draft_tokens = torch.zeros((1, 3), dtype=torch.int64)
    speculator.last_token_indices = torch.zeros(1, dtype=torch.int64)
    speculator.temperature = torch.zeros(1)
    speculator.seeds = torch.zeros(1, dtype=torch.int64)
    speculator.idx_mapping = torch.zeros(1, dtype=torch.int32)
    speculator.draft_logits = None
    speculator.eplb_state = None
    speculator.max_model_len = 64
    speculator.max_num_reqs = 1
    speculator.num_speculative_steps = 3
    speculator.dp_size = speculator.dp_rank = 1
    speculator.prefill_cudagraph_manager = None
    speculator.decode_cudagraph_manager = None
    speculator.kv_cache_config = SimpleNamespace()
    speculator.block_tables = SimpleNamespace(
        compute_slot_mappings=lambda *args: torch.empty(0, dtype=torch.int32)
    )
    speculator._build_draft_attn_metadata = lambda **kwargs: None

    observed_steps = []
    observed_positions = []
    consumed_topk = []
    saved_topk = []
    transient_topk = (
        torch.tensor([10, 11], dtype=torch.int32),
        torch.tensor([-99, -98], dtype=torch.int32),
        torch.tensor([-97, -96], dtype=torch.int32),
    )

    def run_model(
        num_tokens,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        mm_inputs=None,
    ):
        del num_tokens, attn_metadata, slot_mappings
        del num_tokens_across_dp, cudagraph_runtime_mode, mm_inputs
        step = int(speculator.current_draft_step.item())
        observed_steps.append(step)
        observed_positions.append(int(speculator.input_buffers.positions[0].item()))
        if manager_buffers["tp_fence_token"].item() == step:
            if step == 0:
                manager_buffers["topk_logical_ids"][0, 0].copy_(transient_topk[0])
            consumed_topk.append(manager_buffers["topk_logical_ids"][0, 0].clone())
        else:
            consumed_topk.append(transient_topk[step])
        saved_topk.append(manager_buffers["topk_logical_ids"][0, 0].clone())
        return torch.zeros(1, 1), torch.zeros(1, 1)

    def prepare_prefill_inputs(
        last_token_indices,
        current_draft_step,
        input_buffers,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        max_num_reqs,
    ):
        del input_batch, num_sampled, num_rejected, last_sampled
        del next_prefill_tokens, max_num_reqs
        last_token_indices.zero_()
        current_draft_step.zero_()
        input_buffers.positions[0] = 40
        input_buffers.seq_lens[0] = 41
        return last_token_indices

    def prepare_decode_inputs(
        draft_tokens,
        target_seq_lens,
        num_rejected,
        input_buffers,
        max_model_len,
        max_num_reqs,
        advance_draft_positions=True,
    ):
        del draft_tokens, target_seq_lens, num_rejected, max_model_len, max_num_reqs
        if advance_draft_positions:
            input_buffers.positions[0].add_(1)

    def update_draft_inputs(
        draft_tokens,
        current_draft_step,
        hidden_states,
        output_draft_tokens,
        next_input_hidden_states,
        input_buffers,
        num_reqs,
        max_model_len,
        num_speculative_steps,
        advance_draft_positions=True,
    ):
        del draft_tokens, hidden_states, output_draft_tokens
        del next_input_hidden_states, num_reqs, max_model_len
        if (
            advance_draft_positions
            and current_draft_step.item() < num_speculative_steps - 1
        ):
            input_buffers.positions[0].add_(1)

    monkeypatch.setattr(speculator, "_run_model", run_model)
    monkeypatch.setattr(
        speculator,
        "sample_draft",
        lambda hidden_states, *args: torch.zeros(
            hidden_states.shape[0], dtype=torch.int64
        ),
    )
    monkeypatch.setattr(spec_module, "prepare_prefill_inputs", prepare_prefill_inputs)
    monkeypatch.setattr(spec_module, "prepare_decode_inputs", prepare_decode_inputs)
    monkeypatch.setattr(spec_module, "update_draft_inputs", update_draft_inputs)
    monkeypatch.setattr(
        spec_module,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (
            spec_module.BatchExecutionDescriptor(CUDAGraphMode.NONE, 1, 1),
            None,
        ),
    )
    monkeypatch.setattr(spec_module, "build_slot_mappings_by_layer", lambda *args: {})

    input_batch = SimpleNamespace(
        num_tokens_after_padding=1,
        num_reqs=1,
        num_scheduled_tokens=torch.tensor([1]),
        seq_lens_cpu_upper_bound=torch.tensor([41], dtype=torch.int32),
        num_tokens=1,
        idx_mapping=torch.zeros(1, dtype=torch.int32),
        seq_lens=torch.tensor([41], dtype=torch.int32),
    )
    topk_ptr = manager_buffers["topk_logical_ids"].data_ptr()
    fence_ptr = manager_buffers["tp_fence_token"].data_ptr()

    speculator.propose(
        input_batch,
        {},
        {},
        torch.zeros(1, 1),
        None,
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.int64),
    )

    assert observed_steps == [0, 1, 2]
    assert observed_positions == [40, 41, 42]
    assert fenced_steps == [0, 1, 2]
    assert manager_buffers["topk_logical_ids"].data_ptr() == topk_ptr
    assert manager_buffers["tp_fence_token"].data_ptr() == fence_ptr
    assert torch.equal(
        manager_buffers["topk_logical_ids"][0, 0],
        torch.tensor([10, 11], dtype=torch.int32),
    )
    assert all(
        torch.equal(topk, torch.tensor([10, 11], dtype=torch.int32))
        for topk in consumed_topk
    )
    assert all(
        torch.equal(topk, torch.tensor([10, 11], dtype=torch.int32))
        for topk in saved_topk
    )

    live_target_topk = torch.tensor([[[22, 23]]], dtype=torch.int32)

    def native_cache_plan(
        current_main_kv,
        request_block_ids,
        request_num_blocks,
        request_num_tokens,
        request_active,
        req_id_per_token,
        live_topk_logical_ids,
        positions,
        saved_topk_logical_ids,
        tp_fence_token,
        resident_main_kv,
        resident_logical_ids,
        resident_last_access,
        newest_main_kv,
        newest_logical_ids,
        topk_physical_ids,
        topk_hit_mask,
        miss_logical_ids,
        miss_victim_slots,
        miss_counts,
        accepted_counts,
        max_num_blocks,
    ):
        del current_main_kv, request_block_ids, request_num_blocks
        del request_num_tokens, request_active, req_id_per_token, positions
        del resident_main_kv, resident_logical_ids, resident_last_access
        del newest_main_kv, newest_logical_ids, topk_physical_ids
        del topk_hit_mask, miss_logical_ids, miss_victim_slots, miss_counts
        del accepted_counts, max_num_blocks
        assert tp_fence_token.item() == -1
        assert torch.equal(live_topk_logical_ids, live_target_topk)
        assert torch.equal(
            saved_topk_logical_ids[0, 0],
            torch.tensor([10, 11], dtype=torch.int32),
        )
        saved_topk_logical_ids[0, 0].copy_(live_topk_logical_ids[0, 0])

    monkeypatch.setattr(flashmla.ops, "sparse_mla_cache_plan", native_cache_plan)
    manager._fence_mtp_target()
    assert fenced_steps == [0, 1, 2, -1]
    assert torch.equal(
        manager_buffers["topk_logical_ids"][0, 0],
        torch.tensor([10, 11], dtype=torch.int32),
    )
    forward_context = ForwardContext(
        no_compile_layers={"main.0": layer_view},
        attn_metadata={
            "main.0": SimpleNamespace(
                req_id_per_token=torch.zeros(1, dtype=torch.int32)
            )
        },
        slot_mapping={},
    )
    with override_forward_context(forward_context):
        flashmla.sparse_mla_cache_plan(
            torch.zeros(1, 1),
            live_target_topk,
            torch.tensor([43], dtype=torch.int64),
            "main.0",
        )
    assert torch.equal(
        manager_buffers["topk_logical_ids"][0, 0],
        torch.tensor([22, 23], dtype=torch.int32),
    )
    assert manager_buffers["topk_logical_ids"].data_ptr() == topk_ptr
    assert manager_buffers["tp_fence_token"].data_ptr() == fence_ptr
