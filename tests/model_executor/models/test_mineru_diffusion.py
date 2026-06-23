# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig

from vllm.config import (
    AttentionConfig,
    CUDAGraphMode,
    CacheConfig,
    CompilationConfig,
    ModelConfig,
)
from vllm.config import DiffusionConfig
from vllm.config import KVTransferConfig
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.mineru_diffusion import (
    MinerUDiffusionForConditionalGeneration,
    MinerUDiffusionModelState,
    MinerUDiffusionRequestStates,
    MinerUDiffusionSampler,
    SDARForCausalLM,
    get_num_transfer_tokens,
    sample_with_temperature_topk_topp,
    select_transfer_indices,
)
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.transformers_utils.configs.mineru_diffusion import MinerUDiffusionConfig


def test_mineru_diffusion_config_defaults():
    config = MinerUDiffusionConfig()

    assert config.model_type == "mineru_diffusion"
    assert MinerUDiffusionConfig.architectures == [
        "MinerUDiffusionForConditionalGeneration"
    ]
    assert config.text_config.model_type == "sdar"
    assert config.vision_config.model_type == "qwen2_vl_vision"
    assert config.mask_token_id == 151669
    assert config.canvas_length == 32


def test_mineru_model_type_detects_diffusion_without_canvas_length():
    model_config = object.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(model_type="mineru_diffusion")

    assert model_config.is_diffusion


def test_num_transfer_tokens_matches_uniform_schedule():
    assert get_num_transfer_tokens(32, 8) == [4, 4, 4, 4, 4, 4, 4, 4]
    assert get_num_transfer_tokens(10, 4) == [3, 3, 2, 2]


def test_select_transfer_indices_prefers_threshold_then_topk():
    confidence = torch.tensor([[0.9, 0.1, 0.8, 0.7], [0.4, 0.3, 0.2, 0.1]])

    selected = select_transfer_indices(
        confidence,
        threshold=0.75,
        transfer_count=2,
    )

    assert selected.tolist() == [
        [True, False, True, False],
        [True, True, False, False],
    ]


def test_select_transfer_indices_ignores_invalid_positions():
    confidence = torch.tensor([[0.7, -torch.inf, -torch.inf]])

    selected = select_transfer_indices(
        confidence,
        threshold=0.95,
        transfer_count=2,
    )

    assert selected.tolist() == [[True, False, False]]


def test_greedy_sampling_returns_argmax_confidence():
    logits = torch.tensor([[[0.0, 2.0, 1.0], [4.0, 0.0, 1.0]]])

    token_ids, confidence = sample_with_temperature_topk_topp(
        logits,
        temperature=0.0,
    )

    assert token_ids.tolist() == [[1, 0]]
    assert torch.allclose(
        confidence,
        torch.softmax(logits.float(), dim=-1).amax(dim=-1),
    )


def _tiny_vllm_config():
    text_config = MinerUDiffusionConfig().text_config
    text_config.vocab_size = 32
    text_config.hidden_size = 16
    text_config.intermediate_size = 32
    text_config.num_hidden_layers = 1
    text_config.num_attention_heads = 2
    text_config.num_key_value_heads = 1
    text_config.head_dim = 8
    text_config.max_position_embeddings = 64

    vision_config = Qwen2VLVisionConfig(
        depth=1,
        embed_dim=16,
        hidden_size=16,
        hidden_act="quick_gelu",
        mlp_ratio=1,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=1,
    )
    config = MinerUDiffusionConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            hf_text_config=config.text_config,
            dtype=torch.float32,
            is_mm_prefix_lm=False,
            max_model_len=64,
            uses_mrope=False,
            uses_xdrope_dim=0,
            get_inputs_embeds_size=lambda: config.text_config.hidden_size,
            get_vocab_size=lambda: config.text_config.vocab_size,
        )
    )


def _tiny_runtime_vllm_config():
    vllm_config = _tiny_vllm_config()
    vllm_config.cache_config = CacheConfig()
    vllm_config.quant_config = None
    vllm_config.compilation_config = CompilationConfig()
    vllm_config.attention_config = AttentionConfig()
    vllm_config.kv_transfer_config = KVTransferConfig()
    vllm_config.diffusion_config = DiffusionConfig(
        canvas_length=4,
        max_denoising_steps=4,
    )
    vllm_config.scheduler_config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )
    return vllm_config


def test_mineru_native_model_has_real_modules_and_logits_forward():
    model = MinerUDiffusionForConditionalGeneration(vllm_config=_tiny_vllm_config())

    assert hasattr(model, "language_model")
    assert hasattr(model, "vision_model")
    assert model.get_input_embeddings().num_embeddings == 32

    input_ids = torch.tensor([1, 2, 3])
    hidden_states = model(input_ids=input_ids, positions=torch.arange(3))
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (3, 16)
    assert logits is not None
    assert logits.shape == (3, 32)


def test_mineru_model_provides_v2_diffusion_model_state():
    model_state_cls = MinerUDiffusionForConditionalGeneration.get_model_state_cls()

    assert issubclass(model_state_cls, ModelState)
    assert model_state_cls.num_new_sampled_tokens_per_step == 0


def test_mineru_model_state_mm_embeddings_profiles_substeps():
    source = inspect.getsource(MinerUDiffusionModelState.get_mm_embeddings)

    assert "mineru_mm_embeddings: prepare" in source
    assert "mineru_mm_embeddings: encoder" in source
    assert "mineru_mm_embeddings: gather" in source
    assert "mineru_mm_embeddings: input_embeds" in source


def test_mineru_sdar_attention_registers_vllm_kv_cache_layer():
    vllm_config = _tiny_runtime_vllm_config()

    with set_current_vllm_config(vllm_config):
        model = MinerUDiffusionForConditionalGeneration(vllm_config=vllm_config)

    self_attn = model.language_model.model.layers[0].self_attn
    assert isinstance(self_attn.attn, Attention)
    assert (
        "language_model.model.layers.0.self_attn.attn"
        in vllm_config.compilation_config.static_forward_context
    )

    kv_spec = self_attn.attn.get_kv_cache_spec(vllm_config)
    assert kv_spec is not None
    assert kv_spec.num_kv_heads == 1
    assert kv_spec.head_size == 8


def test_mineru_sdar_vllm_path_preserves_flat_runtime_tokens(monkeypatch):
    vllm_config = _tiny_runtime_vllm_config()

    with set_current_vllm_config(vllm_config):
        model = MinerUDiffusionForConditionalGeneration(vllm_config=vllm_config)

    self_attn = model.language_model.model.layers[0].self_attn
    captured = {}

    def fake_attention(query, key, value):
        captured["query_shape"] = tuple(query.shape)
        captured["key_shape"] = tuple(key.shape)
        captured["value_shape"] = tuple(value.shape)
        return torch.zeros_like(query)

    monkeypatch.setattr(self_attn.attn, "forward", fake_attention)

    inputs_embeds = torch.randn(3, 16)
    hidden_states = model.language_model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        position_ids=torch.arange(3),
    )

    assert hidden_states.shape == (3, 16)
    assert captured["query_shape"] == (3, 16)
    assert captured["key_shape"] == (3, 8)
    assert captured["value_shape"] == (3, 8)


def test_mineru_model_state_prepare_attn_uses_scheduled_block_phase(monkeypatch):
    vllm_config = _tiny_runtime_vllm_config()
    state = MinerUDiffusionModelState(
        vllm_config=vllm_config,
        model=torch.nn.Module(),
        encoder_cache=None,
        device=torch.device("cpu"),
    )
    state.diffusion_states.is_encoder_phase[:3] = torch.tensor([True, True, False])
    captured = {}

    def fake_build_attn_metadata(**kwargs):
        captured.update(kwargs)
        return {"metadata": "ok"}

    monkeypatch.setattr(
        "vllm.model_executor.models.mineru_diffusion.build_attn_metadata",
        fake_build_attn_metadata,
    )
    input_batch = SimpleNamespace(
        num_reqs=3,
        num_reqs_after_padding=3,
        num_tokens=8,
        num_tokens_after_padding=8,
        query_start_loc_np=np.array([0, 3, 5, 8], dtype=np.int32),
        query_start_loc=torch.tensor([0, 3, 5, 8], dtype=torch.int32),
        num_scheduled_tokens=np.array([3, 2, 3], dtype=np.int32),
        num_draft_tokens_per_req=np.array([0, 2, 3], dtype=np.int32),
        idx_mapping=torch.tensor([0, 1, 2], dtype=torch.int64),
        seq_lens=torch.tensor([3, 5, 8], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([3, 5, 8], dtype=torch.int32),
        dcp_local_seq_lens=None,
        positions=torch.arange(8),
    )

    out = state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(torch.empty(0),),
        slot_mappings=torch.empty(1, 8, dtype=torch.int32),
        attn_groups=[],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )

    assert out == {"metadata": "ok"}
    assert captured["causal"].tolist() == [True, False, False]
    assert captured["num_tokens"] == 8
    assert captured["max_seq_len"] == 8


def test_mineru_sampler_prefill_denoise_then_commit_block():
    states = MinerUDiffusionRequestStates(
        max_num_reqs=1,
        canvas_length=2,
        mask_token_id=0,
        device=torch.device("cpu"),
    )

    class FakeBaseSampler:
        def __init__(self):
            self.sampling_states = SimpleNamespace()
            self.req_states = SimpleNamespace(
                draft_tokens=torch.full((1, 2), -1, dtype=torch.int64)
            )

        def sample(self, logits, *args, **kwargs):
            return torch.tensor([1, 2]), logits

    sampler = MinerUDiffusionSampler(
        FakeBaseSampler(),
        diffusion_states=states,
        canvas_length=2,
        mask_token_id=0,
        denoising_steps=2,
        dynamic_threshold=0.0,
    )
    prefill_batch = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=0,
        idx_mapping_np=np.array([0], dtype=np.int64),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        num_scheduled_tokens=np.array([3], dtype=np.int32),
        prefill_len_np=np.array([3], dtype=np.int32),
    )

    prefill_out = sampler(logits=torch.empty(0, 3), input_batch=prefill_batch)

    assert prefill_out.num_sampled.tolist() == [0]
    assert states.is_encoder_phase.tolist() == [False]
    assert sampler.req_states.draft_tokens.tolist() == [[0, 0]]

    decode_batch = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=2,
        idx_mapping_np=np.array([0], dtype=np.int64),
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        cu_num_logits_np=np.array([0, 2], dtype=np.int32),
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([0, 0], dtype=torch.int64),
        positions=torch.arange(2),
        logits_indices=torch.arange(2),
        input_ids=torch.tensor([0, 0], dtype=torch.int64),
        expanded_local_pos=torch.arange(2),
    )
    logits = torch.tensor([[0.0, 9.0, 1.0], [0.0, 1.0, 9.0]])

    denoise_out = sampler(logits=logits, input_batch=decode_batch)

    assert denoise_out.num_sampled.tolist() == [0]
    assert states.canvas.tolist() == [[1, 2]]
    assert states.is_encoder_phase.tolist() == [True]
    assert sampler.req_states.draft_tokens.tolist() == [[1, 2]]

    commit_out = sampler(logits=logits, input_batch=decode_batch)

    assert commit_out.num_sampled.tolist() == [2]
    assert commit_out.sampled_token_ids.tolist() == [[1, 2]]
    assert states.is_encoder_phase.tolist() == [False]
    assert sampler.req_states.draft_tokens.tolist() == [[0, 0]]


def test_mineru_sampler_debug_trace_records_prefill_canvas_and_commit(
    tmp_path,
    monkeypatch,
):
    trace_path = tmp_path / "mineru_trace.jsonl"
    monkeypatch.setenv("VLLM_MINERU_DIFFUSION_DEBUG_TRACE", str(trace_path))
    monkeypatch.setenv("VLLM_MINERU_DIFFUSION_DEBUG_TRACE_LIMIT", "10")

    states = MinerUDiffusionRequestStates(
        max_num_reqs=1,
        canvas_length=2,
        mask_token_id=0,
        device=torch.device("cpu"),
    )

    class FakeBaseSampler:
        def __init__(self):
            self.sampling_states = SimpleNamespace()
            self.req_states = SimpleNamespace(
                draft_tokens=torch.full((1, 2), -1, dtype=torch.int64)
            )

        def sample(self, logits, *args, **kwargs):
            return torch.tensor([1, 2]), logits

    sampler = MinerUDiffusionSampler(
        FakeBaseSampler(),
        diffusion_states=states,
        canvas_length=2,
        mask_token_id=0,
        denoising_steps=2,
        dynamic_threshold=0.0,
    )
    prefill_batch = SimpleNamespace(
        req_ids=["req-0"],
        num_reqs=1,
        num_draft_tokens=0,
        idx_mapping_np=np.array([0], dtype=np.int64),
        num_computed_tokens_np=np.array([0], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        num_scheduled_tokens=np.array([3], dtype=np.int32),
        prefill_len_np=np.array([3], dtype=np.int32),
        query_start_loc_np=np.array([0, 3], dtype=np.int32),
    )

    sampler(logits=torch.empty(0, 3), input_batch=prefill_batch)

    decode_batch = SimpleNamespace(
        req_ids=["req-0"],
        num_reqs=1,
        num_draft_tokens=2,
        num_draft_tokens_per_req=np.array([2], dtype=np.int32),
        idx_mapping_np=np.array([0], dtype=np.int64),
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        cu_num_logits_np=np.array([0, 2], dtype=np.int32),
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([0, 0], dtype=torch.int64),
        positions=torch.arange(2),
        logits_indices=torch.arange(2),
        input_ids=torch.tensor([0, 0], dtype=torch.int64),
        expanded_local_pos=torch.arange(2),
        num_computed_tokens_np=np.array([3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([3], dtype=np.int32),
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        prefill_len_np=np.array([3], dtype=np.int32),
    )
    logits = torch.tensor([[0.0, 9.0, 1.0], [0.0, 1.0, 9.0]])

    sampler(logits=logits, input_batch=decode_batch)
    sampler(logits=logits, input_batch=decode_batch)

    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["event"] for record in records] == [
        "mineru_diffusion_sampler_prefill",
        "mineru_diffusion_sampler_denoise",
        "mineru_diffusion_sampler_denoise",
    ]
    assert records[1]["batch"]["num_computed_tokens_np"] == [3]
    assert records[1]["batch"]["prefill_len_np"] == [3]
    assert records[1]["batch"]["num_scheduled_tokens"] == [2]
    assert records[1]["batch"]["num_draft_tokens_per_req"] == [2]
    assert records[1]["batch"]["position_ranges"] == [
        {"len": 2, "max": 1, "min": 0}
    ]

    denoise_req = records[1]["requests"][0]
    assert denoise_req["is_commit"] is False
    assert denoise_req["mask_count_before"] == 2
    assert denoise_req["transferred"] == 2
    assert denoise_req["num_sampled"] == 0
    assert denoise_req["num_rejected"] == 2

    commit_req = records[2]["requests"][0]
    assert commit_req["is_commit"] is True
    assert commit_req["num_sampled"] == 2
    assert commit_req["num_rejected"] == 0


def test_mineru_sampler_debug_trace_uses_original_req_idx_for_mixed_batches(
    tmp_path,
    monkeypatch,
):
    trace_path = tmp_path / "mineru_trace.jsonl"
    monkeypatch.setenv("VLLM_MINERU_DIFFUSION_DEBUG_TRACE", str(trace_path))

    states = MinerUDiffusionRequestStates(
        max_num_reqs=2,
        canvas_length=2,
        mask_token_id=0,
        device=torch.device("cpu"),
    )
    states.is_encoder_phase[1] = True
    states.canvas[1] = torch.tensor([1, 2], dtype=torch.int64)

    class FakeBaseSampler:
        def __init__(self):
            self.sampling_states = SimpleNamespace()
            self.req_states = SimpleNamespace(
                draft_tokens=torch.full((2, 2), -1, dtype=torch.int64)
            )

        def sample(self, logits, *args, **kwargs):
            return torch.tensor([1, 2]), logits

    sampler = MinerUDiffusionSampler(
        FakeBaseSampler(),
        diffusion_states=states,
        canvas_length=2,
        mask_token_id=0,
        denoising_steps=2,
    )
    mixed_batch = SimpleNamespace(
        req_ids=["prefill", "commit"],
        num_reqs=2,
        num_draft_tokens=2,
        num_draft_tokens_per_req=np.array([0, 2], dtype=np.int32),
        idx_mapping_np=np.array([0, 1], dtype=np.int64),
        idx_mapping=torch.tensor([0, 1], dtype=torch.int64),
        cu_num_logits_np=np.array([0, 0, 2], dtype=np.int32),
        query_start_loc_np=np.array([0, 3, 5], dtype=np.int32),
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([1, 1], dtype=torch.int64),
        positions=torch.arange(5),
        logits_indices=torch.arange(2),
        input_ids=torch.tensor([3, 4, 5, 0, 0], dtype=torch.int64),
        expanded_local_pos=torch.arange(2),
        num_computed_tokens_np=np.array([0, 3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 3], dtype=np.int32),
        num_scheduled_tokens=np.array([3, 2], dtype=np.int32),
        prefill_len_np=np.array([3, 3], dtype=np.int32),
    )
    logits = torch.tensor([[0.0, 9.0, 1.0], [0.0, 1.0, 9.0]])

    sampler(logits=logits, input_batch=mixed_batch)

    record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    traced_req = record["requests"][0]
    assert traced_req["req_idx"] == 1
    assert traced_req["num_sampled"] == 2
    assert traced_req["num_rejected"] == 0


def test_mineru_sampler_finishes_prefills_in_mixed_decode_batch():
    states = MinerUDiffusionRequestStates(
        max_num_reqs=2,
        canvas_length=2,
        mask_token_id=0,
        device=torch.device("cpu"),
    )
    states.is_encoder_phase[:] = torch.tensor([False, True])
    states.canvas[0] = torch.tensor([0, 0], dtype=torch.int64)

    class FakeBaseSampler:
        def __init__(self):
            self.sampling_states = SimpleNamespace()
            self.req_states = SimpleNamespace(
                draft_tokens=torch.full((2, 2), -1, dtype=torch.int64)
            )

        def sample(self, logits, *args, **kwargs):
            return torch.tensor([1, 2]), logits

    sampler = MinerUDiffusionSampler(
        FakeBaseSampler(),
        diffusion_states=states,
        canvas_length=2,
        mask_token_id=0,
        denoising_steps=2,
        dynamic_threshold=0.0,
    )
    mixed_batch = SimpleNamespace(
        num_reqs=2,
        num_draft_tokens=2,
        idx_mapping_np=np.array([0, 1], dtype=np.int64),
        idx_mapping=torch.tensor([0, 1], dtype=torch.int64),
        cu_num_logits_np=np.array([0, 2, 2], dtype=np.int32),
        query_start_loc_np=np.array([0, 2, 5], dtype=np.int32),
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([0, 0], dtype=torch.int64),
        positions=torch.arange(5),
        logits_indices=torch.arange(2),
        input_ids=torch.tensor([0, 0, 3, 4, 5], dtype=torch.int64),
        expanded_local_pos=torch.arange(2),
        num_computed_prefill_tokens_np=np.array([3, 0], dtype=np.int32),
        num_scheduled_tokens=np.array([2, 3], dtype=np.int32),
        prefill_len_np=np.array([3, 3], dtype=np.int32),
    )
    logits = torch.tensor([[0.0, 9.0, 1.0], [0.0, 1.0, 9.0]])

    out = sampler(logits=logits, input_batch=mixed_batch)

    assert out.num_sampled.tolist() == [0, 0]
    assert states.is_encoder_phase.tolist() == [True, False]
    assert sampler.req_states.draft_tokens.tolist() == [[1, 2], [0, 0]]


def test_mineru_sampler_commits_after_max_denoising_steps_even_if_mask_sampled():
    states = MinerUDiffusionRequestStates(
        max_num_reqs=1,
        canvas_length=2,
        mask_token_id=0,
        device=torch.device("cpu"),
    )
    states.is_encoder_phase[0] = False
    states.canvas[0] = torch.tensor([0, 0], dtype=torch.int64)

    class FakeBaseSampler:
        def __init__(self):
            self.sampling_states = SimpleNamespace()
            self.req_states = SimpleNamespace(
                draft_tokens=torch.full((1, 2), -1, dtype=torch.int64)
            )

        def sample(self, logits, *args, **kwargs):
            return torch.tensor([0, 0]), logits

    sampler = MinerUDiffusionSampler(
        FakeBaseSampler(),
        diffusion_states=states,
        canvas_length=2,
        mask_token_id=0,
        denoising_steps=1,
        dynamic_threshold=0.0,
    )
    decode_batch = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=2,
        idx_mapping_np=np.array([0], dtype=np.int64),
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        cu_num_logits_np=np.array([0, 2], dtype=np.int32),
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([0, 0], dtype=torch.int64),
        positions=torch.arange(2),
        logits_indices=torch.arange(2),
        input_ids=torch.tensor([0, 0], dtype=torch.int64),
        expanded_local_pos=torch.arange(2),
    )
    logits = torch.tensor([[9.0, 1.0, 0.0], [9.0, 1.0, 0.0]])

    out = sampler(logits=logits, input_batch=decode_batch)

    assert out.num_sampled.tolist() == [0]
    assert states.step.tolist() == [1]
    assert states.canvas.tolist() == [[0, 0]]
    assert states.is_encoder_phase.tolist() == [True]
    assert sampler.req_states.draft_tokens.tolist() == [[0, 0]]


def test_mineru_sampler_mask_only_sampling_skips_accepted_rows(monkeypatch):
    monkeypatch.setenv("VLLM_MINERU_DIFFUSION_MASK_ONLY_SAMPLING", "1")
    states = MinerUDiffusionRequestStates(
        max_num_reqs=1,
        canvas_length=3,
        mask_token_id=0,
        device=torch.device("cpu"),
    )
    states.is_encoder_phase[0] = False
    states.canvas[0] = torch.tensor([0, 9, 0], dtype=torch.int64)

    class FakeBaseSampler:
        def __init__(self):
            self.sampling_states = SimpleNamespace()
            self.req_states = SimpleNamespace(
                draft_tokens=torch.full((1, 3), -1, dtype=torch.int64)
            )
            self.calls = []

        def sample(
            self,
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            return_logprobs=False,
        ):
            self.calls.append(
                {
                    "num_logits": logits.shape[0],
                    "pos": pos.tolist(),
                    "input_ids": input_ids.tolist(),
                    "expanded_local_pos": expanded_local_pos.tolist(),
                    "return_logprobs": return_logprobs,
                }
            )
            if logits.shape[0] == 2:
                return torch.tensor([1, 2]), logits
            return torch.tensor([1, 3, 2]), logits

    base_sampler = FakeBaseSampler()
    sampler = MinerUDiffusionSampler(
        base_sampler,
        diffusion_states=states,
        canvas_length=3,
        mask_token_id=0,
        denoising_steps=1,
        dynamic_threshold=0.99,
    )
    decode_batch = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=3,
        idx_mapping_np=np.array([0], dtype=np.int64),
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        cu_num_logits_np=np.array([0, 3], dtype=np.int32),
        query_start_loc_np=np.array([0, 3], dtype=np.int32),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([0, 0, 0], dtype=torch.int64),
        positions=torch.tensor([10, 11, 12], dtype=torch.int64),
        logits_indices=torch.arange(3),
        input_ids=torch.tensor([0, 9, 0], dtype=torch.int64),
        expanded_local_pos=torch.tensor([0, 1, 2], dtype=torch.int64),
    )
    logits = torch.tensor(
        [
            [0.0, 9.0, 1.0, 2.0],
            [0.0, 1.0, 2.0, 9.0],
            [0.0, 1.0, 9.0, 2.0],
        ]
    )

    out = sampler(logits=logits, input_batch=decode_batch)

    assert out.num_sampled.tolist() == [0]
    assert states.canvas.tolist() == [[1, 9, 2]]
    assert states.is_encoder_phase.tolist() == [True]
    assert base_sampler.calls == [
        {
            "num_logits": 2,
            "pos": [10, 12],
            "input_ids": [0, 0],
            "expanded_local_pos": [0, 2],
            "return_logprobs": False,
        }
    ]


def test_mineru_sampler_accepts_runner_compacted_logits(monkeypatch):
    monkeypatch.setenv("VLLM_MINERU_DIFFUSION_MASK_ONLY_SAMPLING", "1")
    states = MinerUDiffusionRequestStates(
        max_num_reqs=1,
        canvas_length=3,
        mask_token_id=0,
        device=torch.device("cpu"),
    )
    states.is_encoder_phase[0] = False
    states.canvas[0] = torch.tensor([0, 9, 0], dtype=torch.int64)

    class FakeBaseSampler:
        def __init__(self):
            self.sampling_states = SimpleNamespace()
            self.req_states = SimpleNamespace(
                draft_tokens=torch.full((1, 3), -1, dtype=torch.int64)
            )
            self.calls = []

        def sample(
            self,
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            return_logprobs=False,
        ):
            self.calls.append(
                {
                    "num_logits": logits.shape[0],
                    "pos": pos.tolist(),
                    "input_ids": input_ids.tolist(),
                    "expanded_local_pos": expanded_local_pos.tolist(),
                }
            )
            return torch.tensor([1, 2]), logits

    base_sampler = FakeBaseSampler()
    sampler = MinerUDiffusionSampler(
        base_sampler,
        diffusion_states=states,
        canvas_length=3,
        mask_token_id=0,
        denoising_steps=1,
        dynamic_threshold=0.99,
    )
    decode_batch = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=3,
        idx_mapping_np=np.array([0], dtype=np.int64),
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        cu_num_logits_np=np.array([0, 3], dtype=np.int32),
        query_start_loc_np=np.array([0, 3], dtype=np.int32),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([0, 0, 0], dtype=torch.int64),
        positions=torch.tensor([10, 11, 12], dtype=torch.int64),
        logits_indices=torch.arange(3),
        input_ids=torch.tensor([0, 9, 0], dtype=torch.int64),
        expanded_local_pos=torch.tensor([0, 1, 2], dtype=torch.int64),
        logits_row_indices=torch.tensor([0, 2], dtype=torch.int64),
    )
    compact_logits = torch.tensor(
        [
            [0.0, 9.0, 1.0, 2.0],
            [0.0, 1.0, 9.0, 2.0],
        ]
    )

    out = sampler(logits=compact_logits, input_batch=decode_batch)

    assert out.num_sampled.tolist() == [0]
    assert states.canvas.tolist() == [[1, 9, 2]]
    assert states.is_encoder_phase.tolist() == [True]
    assert base_sampler.calls == [
        {
            "num_logits": 2,
            "pos": [10, 12],
            "input_ids": [0, 0],
            "expanded_local_pos": [0, 2],
        }
    ]


def test_mineru_sampler_uses_request_max_denoising_steps_from_extra_args():
    states = MinerUDiffusionRequestStates(
        max_num_reqs=1,
        canvas_length=2,
        mask_token_id=0,
        device=torch.device("cpu"),
    )

    class FakeBaseSampler:
        def __init__(self):
            self.sampling_states = SimpleNamespace()
            self.req_states = SimpleNamespace(
                draft_tokens=torch.full((1, 2), -1, dtype=torch.int64)
            )

        def add_request(self, req_idx, prompt_len, sampling_params):
            pass

        def sample(self, logits, *args, **kwargs):
            return torch.tensor([1, 2]), logits

    sampler = MinerUDiffusionSampler(
        FakeBaseSampler(),
        diffusion_states=states,
        canvas_length=2,
        mask_token_id=0,
        denoising_steps=2,
        dynamic_threshold=0.99,
    )
    sampler.add_request(
        0,
        3,
        SimpleNamespace(extra_args={"max_denoising_steps": 1}),
    )
    states.is_encoder_phase[0] = False
    states.canvas[0] = torch.tensor([0, 0], dtype=torch.int64)
    decode_batch = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=2,
        idx_mapping_np=np.array([0], dtype=np.int64),
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        cu_num_logits_np=np.array([0, 2], dtype=np.int32),
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([0, 0], dtype=torch.int64),
        positions=torch.arange(2),
        logits_indices=torch.arange(2),
        input_ids=torch.tensor([0, 0], dtype=torch.int64),
        expanded_local_pos=torch.arange(2),
    )
    logits = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    out = sampler(logits=logits, input_batch=decode_batch)

    assert out.num_sampled.tolist() == [0]
    assert states.step.tolist() == [1]
    assert states.canvas.tolist() == [[1, 2]]
    assert states.is_encoder_phase.tolist() == [True]
    assert sampler.req_states.draft_tokens.tolist() == [[1, 2]]


def test_mineru_sampler_uses_request_dynamic_threshold_from_extra_args():
    states = MinerUDiffusionRequestStates(
        max_num_reqs=1,
        canvas_length=2,
        mask_token_id=0,
        device=torch.device("cpu"),
    )

    class FakeBaseSampler:
        def __init__(self):
            self.sampling_states = SimpleNamespace()
            self.req_states = SimpleNamespace(
                draft_tokens=torch.full((1, 2), -1, dtype=torch.int64)
            )
            self.added = []

        def add_request(self, req_idx, prompt_len, sampling_params):
            self.added.append((req_idx, prompt_len, sampling_params))

        def sample(self, logits, *args, **kwargs):
            return torch.tensor([1, 2]), logits

    sampler = MinerUDiffusionSampler(
        FakeBaseSampler(),
        diffusion_states=states,
        canvas_length=2,
        mask_token_id=0,
        denoising_steps=2,
        dynamic_threshold=0.0,
    )
    sampler.add_request(
        0,
        3,
        SimpleNamespace(extra_args={"dynamic_threshold": 0.95}),
    )
    states.is_encoder_phase[0] = False
    states.canvas[0] = torch.tensor([0, 0], dtype=torch.int64)
    decode_batch = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=2,
        idx_mapping_np=np.array([0], dtype=np.int64),
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        cu_num_logits_np=np.array([0, 2], dtype=np.int32),
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([0, 0], dtype=torch.int64),
        positions=torch.arange(2),
        logits_indices=torch.arange(2),
        input_ids=torch.tensor([0, 0], dtype=torch.int64),
        expanded_local_pos=torch.arange(2),
    )
    logits = torch.tensor([[0.0, 9.0, 1.0], [0.0, 1.0, 2.0]])

    out = sampler(logits=logits, input_batch=decode_batch)

    assert out.num_sampled.tolist() == [0]
    assert states.canvas.tolist() == [[1, 0]]
    assert states.is_encoder_phase.tolist() == [False]
    assert sampler.req_states.draft_tokens.tolist() == [[1, 0]]


def test_mineru_sampler_rejects_request_block_size_mismatch():
    states = MinerUDiffusionRequestStates(
        max_num_reqs=1,
        canvas_length=2,
        mask_token_id=0,
        device=torch.device("cpu"),
    )

    class FakeBaseSampler:
        def __init__(self):
            self.sampling_states = SimpleNamespace()
            self.req_states = SimpleNamespace(
                draft_tokens=torch.full((1, 2), -1, dtype=torch.int64)
            )

        def add_request(self, req_idx, prompt_len, sampling_params):
            pass

    sampler = MinerUDiffusionSampler(
        FakeBaseSampler(),
        diffusion_states=states,
        canvas_length=2,
        mask_token_id=0,
        denoising_steps=2,
    )

    with pytest.raises(ValueError, match="block_size"):
        sampler.add_request(
            0,
            3,
            SimpleNamespace(extra_args={"block_size": 1}),
        )


def test_mineru_native_load_weights_copies_matching_parameters():
    model = MinerUDiffusionForConditionalGeneration(vllm_config=_tiny_vllm_config())
    weight = torch.full_like(model.language_model.lm_head.weight, 0.25)

    loaded = model.load_weights([("language_model.lm_head.weight", weight)])

    assert "language_model.lm_head.weight" in loaded
    assert torch.equal(model.language_model.lm_head.weight, weight)


def test_mineru_embed_multimodal_splits_image_features(monkeypatch):
    model = MinerUDiffusionForConditionalGeneration(vllm_config=_tiny_vllm_config())
    features = torch.arange(3 * 16, dtype=torch.float32).view(3, 16)

    monkeypatch.setattr(
        model,
        "get_image_features",
        lambda pixel_values, image_grid_thw: features,
    )

    outputs = model.embed_multimodal(
        pixel_values=torch.zeros(12, 12),
        image_grid_thw=torch.tensor([[1, 2, 4], [1, 2, 2]]),
    )

    assert len(outputs) == 2
    assert outputs[0].shape == (2, 16)
    assert outputs[1].shape == (1, 16)


def test_mineru_embed_input_ids_accepts_vllm_multimodal_embeddings():
    model = MinerUDiffusionForConditionalGeneration(vllm_config=_tiny_vllm_config())
    input_ids = torch.tensor([1, 2, 3, 4])
    is_multimodal = torch.tensor([False, True, True, False])
    multimodal_embeddings = [torch.full((2, 16), 7.0)]

    outputs = model.embed_input_ids(
        input_ids,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )

    assert torch.equal(outputs[is_multimodal], multimodal_embeddings[0])


def test_sdar_generate_with_embeds_fills_masked_block(monkeypatch):
    config = MinerUDiffusionConfig().text_config
    config.vocab_size = 8
    config.hidden_size = 8
    config.intermediate_size = 16
    config.num_hidden_layers = 1
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.head_dim = 4
    model = SDARForCausalLM(config)

    call_count = 0

    def fake_forward(*, input_ids=None, inputs_embeds=None, position_ids=None,
                     attention_mask=None):
        nonlocal call_count
        call_count += 1
        assert input_ids is None
        assert inputs_embeds is not None
        assert attention_mask is not None
        hidden_states = torch.zeros(
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            config.hidden_size,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        return hidden_states

    def fake_lm_head(hidden_states):
        logits = torch.full(
            (*hidden_states.shape[:-1], config.vocab_size),
            -100.0,
            device=hidden_states.device,
        )
        logits[:, :, 3] = 100.0
        logits[:, -1, 4] = 200.0
        return logits

    monkeypatch.setattr(model, "forward", fake_forward)
    monkeypatch.setattr(model.lm_head, "forward", fake_lm_head)

    prompt_embeds = torch.zeros(1, 3, config.hidden_size)

    generated, step_map, _ = model.generate_with_embeds(
        inputs_embeds=prompt_embeds,
        gen_length=2,
        block_length=2,
        mask_token_id=0,
        denoising_steps=2,
        temperature=0.0,
        dynamic_threshold=0.95,
    )

    assert generated.tolist() == [[3, 4]]
    assert step_map.tolist() == [[1, 1]]
    assert call_count == 1


def test_mineru_generate_routes_through_language_diffusion(monkeypatch):
    model = MinerUDiffusionForConditionalGeneration(vllm_config=_tiny_vllm_config())
    model.config.image_token_id = 7
    input_ids = torch.tensor([[1, model.config.image_token_id, 2]])
    image_features = torch.full((1, 16), 5.0)

    monkeypatch.setattr(
        model,
        "get_image_features",
        lambda pixel_values, image_grid_thw: image_features,
    )

    captured = {}

    def fake_generate_with_embeds(**kwargs):
        captured.update(kwargs)
        return (
            torch.tensor([[3, 4]]),
            torch.tensor([[1, 1]]),
            torch.zeros(1, 2),
        )

    monkeypatch.setattr(
        model.language_model,
        "generate_with_embeds",
        fake_generate_with_embeds,
    )

    generated, step_map, _ = model.generate(
        input_ids=input_ids,
        pixel_values=torch.zeros(4, 4),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        gen_length=2,
        block_length=2,
        mask_token_id=0,
    )

    assert generated.tolist() == [[3, 4]]
    assert step_map.tolist() == [[1, 1]]
    assert torch.equal(captured["inputs_embeds"][0, 1], image_features[0])
    assert captured["gen_length"] == 2
