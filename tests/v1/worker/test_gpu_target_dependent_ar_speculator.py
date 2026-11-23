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
from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP
from vllm.v1.attention.backends import flash_attn as flash_attn_module
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    MultiModuleMTPSpeculator,
    prepare_input_buffers,
    prepare_input_hidden_states_and_embeddings,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)
from vllm.v1.worker.utils import AttentionGroup


class _TestSpeculator(TargetDependentARSpeculator):
    # Set by each test to the draft model `load_draft_model` should hand back.
    test_draft_model: torch.nn.Module

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


@pytest.mark.parametrize("cg_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_pcp_draft_metadata_keeps_graph_padding_in_decode(cg_mode):
    def build(common_prefix_len, common_attn_metadata):
        return split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=1,
            require_uniform=True,
            treat_short_extends_as_decodes=False,
        )

    speculator = object.__new__(_TestSpeculator)
    speculator.arange_np = torch.arange(5, dtype=torch.int32).numpy()
    speculator.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=32)
    )
    speculator.draft_max_seq_len = 32
    speculator.draft_is_prefilling = torch.zeros(4, dtype=torch.bool)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(4, dtype=torch.int64),
        query_start_loc=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
        seq_lens=torch.tensor([11, 21, 0, 0], dtype=torch.int32),
    )
    speculator.dcp_size = 1
    speculator.block_tables = SimpleNamespace(
        cp_size=1,
        input_block_tables=[torch.zeros(4, 1, dtype=torch.int32)],
        slot_mappings=torch.tensor([[10, 20, -1, -1]]),
    )
    speculator.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    builder = SimpleNamespace(build=build, supports_update_block_table=False)
    speculator.attn_groups = [
        [
            # Stand-in group: build_attn_metadata only reads these members.
            SimpleNamespace(  # type: ignore[list-item]  # stand-in group
                metadata_builders=[builder],
                get_metadata_builder=lambda _: builder,
                build_metadata=lambda cm, ubatch_idx=0, **_: build(0, cm),
                layer_names=["draft"],
            )
        ]
    ]
    num_reqs_padded = 4 if cg_mode == CUDAGraphMode.FULL else 2
    metadata = speculator._build_uniform_attn_metadata(
        batch_desc=BatchExecutionDescriptor(cg_mode, 4, num_reqs_padded),
        num_reqs=2,
        num_query_per_req=1,
        seq_lens_cpu_upper_bound=torch.tensor([10, 20], dtype=torch.int32),
        step=1,
    )
    assert metadata is not None
    assert metadata["draft"] == (num_reqs_padded, 0, num_reqs_padded, 0)


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
    speculator.pcp_manager = None
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
        draft_decode_context_parallel_size=1,
        use_local_argmax_reduction=False,
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
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
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert speculator.hidden_size == expected


def test_mm_support_configured_after_model_load(monkeypatch):
    checked_configs = []

    class _ModelConfig:
        @property
        def supports_multimodal_inputs(self):
            checked_configs.append(self)
            return True

    target_model_config = _ModelConfig()
    draft_model_config = _ModelConfig()
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
        speculator.use_acceptance_estimator = False
        speculator.use_draft_logits = False

    monkeypatch.setattr(DraftModelSpeculator, "__init__", init_base)
    _mock_base_model_load(monkeypatch)

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
    speculator.use_acceptance_estimator = False
    speculator.use_draft_logits = False
    speculator.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(supports_multimodal_inputs=True)
    )
    speculator.max_num_tokens = 4
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    speculator.test_draft_model = draft_model
    _mock_base_model_load(monkeypatch)

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None


def test_load_model_disables_mm_support_for_text_only_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.use_acceptance_estimator = False
    speculator.use_draft_logits = False
    speculator.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(supports_multimodal_inputs=True)
    )
    draft_model = _TextOnlyDraftModel()
    speculator.test_draft_model = draft_model
    warning_messages = []
    _mock_base_model_load(monkeypatch)
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
    speculator.use_acceptance_estimator = False
    speculator.use_draft_logits = False
    speculator.cached_draft_input_embeds = None
    speculator.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(supports_multimodal_inputs=True)
    )
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

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)
    assert speculator.cached_draft_input_embeds is not None
    assert speculator.cached_draft_input_embeds.shape == (2, 2, 3)


@pytest.mark.parametrize("use_input_embeds", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_multi_module_mtp_reprefill_uses_only_cached_request_history(
    use_input_embeds,
):
    """Rejecting all drafts must not replay the previous slot occupant's data."""
    device = torch.device("cuda")
    steps, hidden_size, max_reqs, max_tokens, slot = 4, 8, 2, 5, 1
    target_buffers = InputBuffers(max_reqs, max_tokens, device)
    batch = InputBatch.make_dummy(1, max_tokens, target_buffers, is_padding=False)
    batch.idx_mapping.fill_(slot)
    batch.idx_mapping_np.fill(slot)
    batch.input_ids.copy_(torch.tensor([5100, 1, 2, 3, 4], device=device))
    batch.positions.copy_(torch.arange(2, 7, device=device))
    batch.seq_lens.fill_(7)
    buffers = InputBuffers(max_reqs, max_tokens, device)
    buffers.query_start_loc.copy_(target_buffers.query_start_loc)
    hidden_states = torch.zeros(max_tokens, hidden_size, device=device)
    embeds = torch.full_like(hidden_states, 30.0) if use_input_embeds else None
    cached_ids = torch.tensor([[0, 0, 0], [9005, 5001, 5100]], device=device)
    cached_hidden = torch.full((max_reqs, steps - 1, hidden_size), 3.0, device=device)
    cached_hidden[slot, 0].fill_(7.0)
    num_cached_tokens = torch.tensor([0, 2], dtype=torch.int32, device=device)
    rejected = torch.tensor([4], dtype=torch.int32, device=device)
    last_indices = torch.zeros(max_reqs, dtype=torch.int64, device=device)
    prepare_input_buffers(
        1,
        batch,
        cached_ids,
        num_cached_tokens,
        torch.ones(1, dtype=torch.int32, device=device),
        rejected,
        torch.full((max_reqs,), 5200, device=device),
        torch.zeros(steps, max_reqs, dtype=torch.int32, device=device),
        torch.full((max_reqs, steps - 1), -1, device=device),
        buffers,
        last_indices,
        max_reqs,
        steps,
    )
    prepare_input_hidden_states_and_embeddings(
        1,
        max_tokens,
        hidden_states,
        torch.full_like(hidden_states, 3.0),
        cached_hidden,
        num_cached_tokens,
        embeds,
        10 * cached_hidden if use_input_embeds else None,
        batch,
        buffers,
        rejected,
        steps,
    )

    assert buffers.input_ids[:3].tolist() == [5001, 5100, 5200]
    assert buffers.positions[:3].tolist() == [0, 1, 2]
    assert last_indices[0].item() == 2
    torch.testing.assert_close(
        hidden_states[:3],
        torch.full_like(hidden_states[:3], 3.0),
        rtol=0,
        atol=0,
    )
    if embeds is not None:
        torch.testing.assert_close(
            embeds[:3],
            torch.full_like(embeds[:3], 30.0),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [
        (EagleLlama4ForCausalLM, True),
        (EagleMistralForCausalLM, True),
        (EagleMistralLarge3ForCausalLM, True),
        (Exaone4_5_MTP, True),
        (NemotronHMTP, True),
        (Eagle3LlamaForCausalLM, False),
    ],
)
def test_draft_model_multimodal_embedding_capability(model_cls, expected):
    assert supports_multimodal_embeddings(model_cls) is expected


@pytest.mark.parametrize(
    "model_cls", [EagleLlama4ForCausalLM, EagleMistralLarge3ForCausalLM]
)
def test_mm_capable_drafter_merges_multimodal_embeddings(model_cls):
    """Drafters advertising multimodal embeddings must embed text ids and
    scatter the multimodal embeddings themselves; they do not inherit
    SupportsMultiModal's helpers."""
    table = torch.arange(8, dtype=torch.float32).repeat(3, 1).T  # [8, 3]
    drafter = object.__new__(model_cls)
    drafter.model = SimpleNamespace(embed_input_ids=lambda ids: table[ids])
    input_ids = torch.tensor([1, 5, 6, 2])

    text_only = model_cls.embed_input_ids(drafter, input_ids)
    assert torch.equal(text_only, table[input_ids])

    mm_embeds = [torch.full((2, 3), -1.0)]
    is_multimodal = torch.tensor([False, True, True, False])
    merged = model_cls.embed_input_ids(
        drafter, input_ids, mm_embeds, is_multimodal=is_multimodal
    )
    assert torch.equal(merged[[0, 3]], table[[1, 2]])
    assert torch.equal(merged[[1, 2]], mm_embeds[0])


def test_nemotron_h_mtp_merges_multimodal_embeddings():
    """NemotronH MTP is trained on the target input stream, so image and audio
    embeddings must replace placeholder token embeddings."""
    table = torch.arange(8, dtype=torch.float32).repeat(3, 1).T  # [8, 3]

    class _EmbeddingStub(NemotronHMTP):
        def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
            return table[input_ids]

    drafter = object.__new__(_EmbeddingStub)
    input_ids = torch.tensor([1, 5, 6, 2])

    text_only = NemotronHMTP.embed_input_ids(drafter, input_ids)
    assert torch.equal(text_only, table[input_ids])

    mm_embeds = [torch.full((2, 3), -1.0)]
    is_multimodal = torch.tensor([False, True, True, False])
    merged = NemotronHMTP.embed_input_ids(
        drafter, input_ids, mm_embeds, is_multimodal=is_multimodal
    )
    assert torch.equal(merged[[0, 3]], table[[1, 2]])
    assert torch.equal(merged[[1, 2]], mm_embeds[0])

    with pytest.raises(ValueError, match="is_multimodal"):
        NemotronHMTP.embed_input_ids(drafter, input_ids, mm_embeds)


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


@pytest.mark.parametrize("partitioned", [False, True])
def test_run_model_uses_complete_pcp_padding(monkeypatch, partitioned):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)
    gathered_is_padding = torch.tensor(
        [False, False, False, True, False, False, True, True]
    )
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        max_num_tokens=4,
        moe_gathers_tokens=True,
    )
    if partitioned:
        manager.draft_prefill_batch = SimpleNamespace(
            input_ids=torch.arange(4),
            positions=torch.arange(4),
            is_padding=gathered_is_padding,
        )
    else:
        local_is_padding = torch.tensor([False, False, False, True])
        speculator.input_buffers.is_padding = local_is_padding
        gathered_is_padding = torch.cat([local_is_padding, local_is_padding])
    seen = {}
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: (seen.update(kwargs), nullcontext())[1],
    )
    speculator.pcp_manager = manager

    speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    torch.testing.assert_close(seen["is_padding"], gathered_is_padding)
    if partitioned:
        assert seen["is_padding"] is gathered_is_padding


@pytest.mark.parametrize(
    ("cg_mode", "use_pcp", "expected_prefill_mode"),
    [
        (
            CUDAGraphMode.FULL_AND_PIECEWISE,
            False,
            CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
        (CUDAGraphMode.FULL_AND_PIECEWISE, True, CUDAGraphMode.PIECEWISE),
        (CUDAGraphMode.FULL_DECODE_ONLY, True, CUDAGraphMode.NONE),
        (CUDAGraphMode.PIECEWISE, True, CUDAGraphMode.PIECEWISE),
    ],
)
def test_pcp_draft_prefill_skips_full_cudagraphs(
    monkeypatch, cg_mode, use_pcp, expected_prefill_mode
):
    # A FULL draft-prefill replay would skip the PCP hidden-state all-gather.
    modes = []
    monkeypatch.setattr(
        spec_module,
        "SpeculatorCudaGraphManager",
        lambda vllm_config, device, mode, *args, **kwargs: modes.append(mode),
    )
    monkeypatch.setattr(
        spec_module.AutoRegressiveSpeculator,
        "init_cudagraph_manager",
        lambda self, mode: modes.append(
            CUDAGraphMode.FULL_DECODE_ONLY
            if mode.decode_mode() == CUDAGraphMode.FULL
            else CUDAGraphMode.NONE
        ),
    )
    speculator = object.__new__(_TestSpeculator)
    speculator.vllm_config = SimpleNamespace(
        get_decode_cudagraph_query_lens=lambda qlen: [qlen],
    )
    speculator.device = torch.device("cpu")
    speculator.num_speculative_steps = 3
    speculator.enable_adaptive_verification = False
    speculator.pcp_manager = object() if use_pcp else None

    speculator.init_cudagraph_manager(cg_mode)

    expected_decode_mode = (
        CUDAGraphMode.FULL_DECODE_ONLY
        if cg_mode.decode_mode() == CUDAGraphMode.FULL
        else CUDAGraphMode.NONE
    )
    assert modes == [expected_prefill_mode, expected_decode_mode]


@pytest.mark.parametrize("pcp_rank", [0, 1])
def test_pcp_sharded_decode_propose_preserves_uniform_token_count_and_pad_alignment(
    monkeypatch, pcp_rank
):
    """When PCP shards a single uniform decode request (query_len=3) across 2
    ranks and draft prefill pads from 3 to a 4-token PIECEWISE bucket, rank 1
    gets local_num_tokens=0 and num_tokens_after_padding=3. propose() must
    classify uniform_token_count from the global batch, extend the draft prefill
    views to 4 tokens, and restore the gathered hidden states across the 4-token
    padded stride.
    """
    from dataclasses import replace

    import numpy as np

    from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
        SpeculatorCudaGraphManager,
    )

    device = torch.device("cpu")
    monkeypatch.setattr(
        pcp_manager_module,
        "async_tensor_h2d",
        lambda val, device=None, dtype=None, out=None: (
            out.copy_(torch.from_numpy(val) if isinstance(val, np.ndarray) else val)
            if out is not None
            else (torch.from_numpy(val) if isinstance(val, np.ndarray) else val)
        ),
    )
    monkeypatch.setattr(base_spec_module, "_target_feeds_hc_residual", lambda _: False)

    def _prepare_prefill_inputs_cpu(
        last_token_indices,
        draft_current_step,
        input_buffers,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        max_num_reqs,
    ):
        draft_current_step.zero_()
        last_token_indices[:1] = 2
        input_buffers.input_ids[:3] = torch.tensor([21, 22, 23], dtype=torch.int32)
        input_buffers.positions[:3] = input_batch.positions[:3]

    monkeypatch.setattr(
        spec_module, "prepare_prefill_inputs", _prepare_prefill_inputs_cpu
    )
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    class _ConcatDraftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_shapes: list[tuple[torch.Size, torch.Size, torch.Size]] = []

        def forward(
            self, input_ids, positions, hidden_states, inputs_embeds=None, **kw
        ):
            self.seen_shapes.append(
                (input_ids.shape, positions.shape, hidden_states.shape)
            )
            assert input_ids.shape[0] == positions.shape[0] == hidden_states.shape[0]
            return hidden_states + input_ids.unsqueeze(-1).to(hidden_states.dtype)

        def compute_logits(self, hidden_states):
            logits = torch.zeros((hidden_states.shape[0], 16), dtype=torch.float32)
            logits[:, 7] = hidden_states[:, 0]
            return logits

    draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(),
        get_hidden_size=lambda: 3,
        get_vocab_size=lambda: 16,
    )
    speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=2,
        draft_model_config=draft_model_config,
        draft_decode_context_parallel_size=1,
        use_local_argmax_reduction=False,
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
    )
    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        prefill_cudagraph_capture_token_sizes=[4],
        decode_cudagraph_capture_request_sizes=[1, 2],
        max_cudagraph_capture_size=4,
        cudagraph_num_of_warmups=0,
        use_breakable_cudagraph=False,
        use_inductor_graph_partition=False,
        splitting_ops=[],
        compile_ranges_endpoints=[],
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        cache_config=SimpleNamespace(use_kda_recoverssm=False),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=8,
        ),
        model_config=SimpleNamespace(
            max_model_len=64,
            dtype=torch.float32,
            head_dtype=torch.float32,
            use_fp64_gumbel=False,
            supports_multimodal_inputs=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            prefill_context_parallel_size=2,
            data_parallel_size=2,
            data_parallel_rank=0,
            decode_context_parallel_size=1,
            use_ubatching=False,
            enable_fault_tolerance=False,
        ),
        uniform_decode_query_len=3,
        get_decode_cudagraph_query_lens=lambda qlen: {qlen: 4},
    )

    _mock_base_model_load(monkeypatch)
    speculator = _TestSpeculator(vllm_config, device)
    speculator.test_draft_model = _ConcatDraftModel()
    speculator.load_model(torch.nn.Module())
    speculator.pcp_manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=pcp_rank,
        device=device,
        shard_decode_requests=True,
        max_num_reqs=4,
        max_num_tokens=8,
        moe_gathers_tokens=True,
    )
    speculator.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
        vllm_config,
        device,
        CUDAGraphMode.PIECEWISE,
        {3: 4},
        varlen_decode=False,
    )
    monkeypatch.setattr(
        speculator.prefill_cudagraph_manager,
        "run_pw_graph",
        lambda model, model_inputs: model(**model_inputs),
    )

    global_buffers = InputBuffers(4, 8, device)
    base_batch = InputBatch.make_dummy(1, 3, global_buffers)
    global_buffers.input_ids[:3] = torch.tensor([11, 12, 13], dtype=torch.int32)
    global_buffers.positions[:3] = torch.tensor([10, 11, 12], dtype=torch.int64)
    global_buffers.seq_lens[:1] = torch.tensor([13], dtype=torch.int32)
    global_buffers.query_start_loc[:2] = torch.tensor([0, 3], dtype=torch.int32)
    global_batch = replace(
        base_batch,
        req_ids=["req_0"],
        num_reqs=1,
        num_reqs_after_padding=1,
        idx_mapping=torch.tensor([0], dtype=torch.int64, device=device),
        idx_mapping_np=np.array([0], dtype=np.int64),
        num_scheduled_tokens=np.array([3], dtype=np.int32),
        num_tokens=3,
        num_tokens_after_padding=3,
        query_start_loc=global_buffers.query_start_loc[:2],
        query_start_loc_np=np.array([0, 3], dtype=np.int32),
        seq_lens=global_buffers.seq_lens[:1],
        seq_lens_cpu_upper_bound=torch.tensor([13], dtype=torch.int32),
        num_computed_tokens_np=np.array([10], dtype=np.int32),
        prefill_len_np=np.array([10], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([10], dtype=np.int32),
        is_prefilling_np=np.array([False], dtype=np.bool_),
        has_prefill=False,
        decode_graph_eligible=True,
        input_ids=global_buffers.input_ids[:3],
        positions=global_buffers.positions[:3],
        is_padding=global_buffers.is_padding[:3],
    )
    speculator.pcp_manager.partition_batch(
        global_batch,
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=3,
            num_reqs=1,
            uniform_token_count=3,
        ),
    )

    rank0_out = torch.tensor(
        [[101.0, 1.0, 1.0], [102.0, 1.0, 1.0], [103.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    )
    rank1_out = torch.zeros((4, 3))
    monkeypatch.setattr(
        pcp_manager_module,
        "get_pcp_group",
        lambda: SimpleNamespace(
            all_gather=lambda tensor, dim=0: torch.cat([rank0_out, rank1_out], dim=dim)
        ),
    )

    dp_sync_state = DPSyncState(
        num_tokens_across_dp=torch.tensor([3, 3], dtype=torch.int32),
        uniform_token_count=3,
        max_query_len=3,
        eager=False,
        num_reqs=2,
    )
    drafts = speculator.propose(
        input_batch=global_batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.full((3, 3), 80.0),
        aux_hidden_states=None,
        num_sampled=torch.ones(1, dtype=torch.int32),
        num_rejected=torch.zeros(1, dtype=torch.int32),
        last_sampled=torch.zeros(4, dtype=torch.int32),
        next_prefill_tokens=torch.zeros(4, dtype=torch.int32),
        temperature=torch.zeros(4, dtype=torch.float32),
        seeds=torch.zeros(4, dtype=torch.int64),
        dp_sync_state=dp_sync_state,
        skip_attn_for_dummy_run=True,
        num_speculative_tokens=1,
    )

    assert speculator.test_draft_model.seen_shapes == [
        (torch.Size([4]), torch.Size([4]), torch.Size([4, 3]))
    ]
    assert drafts.tolist() == [[7]]
    assert torch.equal(speculator.hidden_states[0], rank0_out[2])


@pytest.mark.parametrize(
    (
        "method_name",
        "cg_mode",
        "num_speculative_steps",
        "expected_eager_calls",
        "expected_graph_replays",
    ),
    [
        ("_multi_step_decode", CUDAGraphMode.NONE, 4, 3, 0),
        ("_multi_step_decode", CUDAGraphMode.FULL, 4, 0, 3),
        ("_run_fused_multi_step_decode", CUDAGraphMode.NONE, 4, 3, 0),
        ("_run_fused_multi_step_decode", CUDAGraphMode.FULL, 4, 0, 1),
        ("_multi_step_decode", CUDAGraphMode.FULL, 3, 0, 2),
    ],
)
def test_multi_step_decode_replays_captured_graph_as_expected(
    monkeypatch,
    method_name,
    cg_mode,
    num_speculative_steps,
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
    speculator.attn_groups = []
    speculator.block_tables = SimpleNamespace(input_block_tables=[])
    decode = Mock()
    # Stub the bound method on a hand-built speculator to count eager draft calls.
    monkeypatch.setattr(speculator, "_decode", decode)
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
        num_speculative_steps=num_speculative_steps,
    )

    assert decode.call_count == expected_eager_calls
    assert run_fullgraph.call_count == expected_graph_replays


def test_fused_full_graph_maps_split_group_block_tables():
    """The fused FULL replay skips build(), so split groups' tables are mapped first."""
    spec = MLAAttentionSpec(
        block_size=1024, num_kv_heads=1, head_size=128, dtype=torch.uint8
    )
    kernel_block_table = torch.zeros(1, 4, 32, dtype=torch.int32)
    group = AttentionGroup(
        SimpleNamespace(),
        ["layer.0"],
        spec,
        0,
        kernel_block_size=256,
        kernel_block_table=kernel_block_table,
        kernel_block_offsets=torch.arange(4, dtype=torch.int32),
    )
    speculator = object.__new__(_TestSpeculator)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2), query_start_loc=torch.arange(3)
    )
    speculator.idx_mapping = torch.arange(2)
    speculator.attn_groups = [[group]]
    speculator.block_tables = SimpleNamespace(
        input_block_tables=[torch.tensor([[1, 2], [0, 3]], dtype=torch.int32)]
    )
    replayed_tables = []
    speculator.decode_cudagraph_manager = SimpleNamespace(
        run_fullgraph=lambda _: replayed_tables.append(
            kernel_block_table[0, :2, :8].tolist()
        )
    )

    speculator._run_fused_multi_step_decode(
        num_reqs=2,
        skip_attn=False,
        batch_desc=BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL, num_tokens=2, num_reqs=2
        ),
        num_tokens_across_dp=None,
        seq_lens_cpu_upper_bound=None,
        num_speculative_steps=4,
    )

    assert replayed_tables == [
        [[4, 5, 6, 7, 8, 9, 10, 11], [0, 1, 2, 3, 12, 13, 14, 15]]
    ]


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
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=2,
        num_prefill_reqs=0,
        num_decode_tokens=3,
        num_prefill_tokens=0,
        scheduler_metadata=torch.tensor([-1, -1, -1], dtype=torch.int32),
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
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=1,
        num_prefill_reqs=0,
        num_decode_tokens=1,
        num_prefill_tokens=0,
        scheduler_metadata=None,
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


@pytest.mark.parametrize(("num_speculative_steps", "expected"), [(1, False), (3, True)])
def test_decode_metadata_update_needs_draft_decode_steps(
    num_speculative_steps, expected
):
    """One speculative step runs no draft decodes, so the in-place metadata
    update (and its fallback log) does not apply even on a capable backend."""
    speculator = object.__new__(_TestSpeculator)
    speculator.num_speculative_steps = num_speculative_steps
    speculator.attn_groups = [
        [
            SimpleNamespace(  # type: ignore[list-item]  # stand-in group
                metadata_builders=[object()],
                supports_draft_decode_metadata_update=True,
                backend=SimpleNamespace(get_name=lambda: "CAPABLE"),
            )
        ]
    ]

    speculator._configure_decode_metadata_update()

    assert speculator.supports_decode_metadata_update is expected


def test_rebuilt_prefill_metadata_uses_target_batch_slots():
    # Under DBO the target builds only per-microbatch metadata, so the drafter
    # rebuilds the full batch's on its own builders from the target's slots.
    speculator = object.__new__(_TestSpeculator)
    target = ({"draft": "target metadata"}, {"draft": torch.zeros(6)})
    assert speculator._build_drafter_attn_metadata(None, *target) == target

    speculator.can_reuse_target_attn_metadata = False
    speculator.attn_groups = [[Mock()]]
    speculator.model_state = Mock()
    speculator.model_state.prepare_attn.return_value = {"draft": "metadata"}
    speculator.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["target", "draft"])]
    )
    speculator.block_tables = SimpleNamespace(
        input_block_tables=[torch.arange(8, dtype=torch.int32).view(4, 2)],
        slot_mappings=torch.arange(8).view(1, 8),
    )
    input_batch = SimpleNamespace(num_reqs_after_padding=2, num_tokens_after_padding=6)

    attn_metadata, slot_mappings = speculator._build_drafter_attn_metadata(
        input_batch, *target, CUDAGraphMode.FULL
    )

    assert attn_metadata == {"draft": "metadata"}
    args = speculator.model_state.prepare_attn.call_args.args
    batch, cg_mode, (block_table,), target_slots, attn_groups, _ = args
    assert batch is input_batch and cg_mode == CUDAGraphMode.FULL
    assert attn_groups is speculator.attn_groups
    assert block_table.shape == (2, 2)
    assert target_slots.data_ptr() == speculator.block_tables.slot_mappings.data_ptr()
    assert target_slots.shape == (1, 6)
    assert slot_mappings["draft"].tolist() == list(range(6))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_captured_draft_update_reads_reduced_max_model_len():
    """Auto-fit can shrink max_model_len after the draft graphs are captured.
    A replay must clamp draft positions to the new length, as a server started
    with that length would, not to the length seen at capture time."""
    device = torch.device("cuda")
    num_reqs, hidden_size, num_steps = 2, 8, 3
    input_buffers = SimpleNamespace(
        input_ids=torch.zeros(num_reqs, dtype=torch.int32, device=device),
        positions=torch.zeros(num_reqs, dtype=torch.int64, device=device),
        seq_lens=torch.zeros(num_reqs, dtype=torch.int32, device=device),
    )
    max_model_len = torch.tensor(100, dtype=torch.int32, device=device)
    draft_tokens = torch.zeros(num_reqs, dtype=torch.int64, device=device)
    output_draft_tokens = torch.zeros(
        num_reqs, num_steps, dtype=torch.int64, device=device
    )
    hidden_states = torch.zeros(num_reqs, hidden_size, device=device)
    next_hidden_states = torch.zeros_like(hidden_states)
    sample_src_positions = torch.zeros(num_reqs, dtype=torch.int64, device=device)
    current_step = torch.zeros((), dtype=torch.int64, device=device)

    def update():
        spec_module.update_draft_inputs(
            draft_tokens,
            current_step,
            hidden_states,
            output_draft_tokens,
            next_hidden_states,
            input_buffers,
            sample_src_positions,
            num_reqs,
            max_model_len,
            num_steps,
        )

    update()  # compile before capture
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update()

    max_model_len.fill_(8)
    input_buffers.positions.copy_(torch.tensor([5, 7], device=device))
    input_buffers.seq_lens.copy_(torch.tensor([6, 8], device=device))
    graph.replay()
    assert input_buffers.positions.tolist() == [6, 7]
    assert input_buffers.seq_lens.tolist() == [7, 8]
