# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    try_get_attention_backend,
)
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.load import LoadConfig
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.gpu import sparse_mla_offload as sparse_mla_offload_module
from vllm.v1.worker.gpu.sparse_mla_offload import SparseMLAOffloadManager

mimo_7b_dir = "XiaomiMiMo/MiMo-7B-Base"
DEVICE_TYPE = current_platform.device_type


def _create_mtp_proposer(num_speculative_tokens: int) -> EagleProposer:
    """Create an MTP proposer with unified model configuration."""
    model_config = ModelConfig(
        model=mimo_7b_dir, runner="generate", max_model_len=100, trust_remote_code=True
    )

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=mimo_7b_dir,
        method="mtp",
        num_speculative_tokens=num_speculative_tokens,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=DEVICE_TYPE),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
    )

    return EagleProposer(vllm_config=vllm_config, device=DEVICE_TYPE)


@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_pp_group")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_layers_from_vllm_config")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_model")
def test_mtp_load_model_unified(mock_get_model, mock_get_layers, mock_get_pp_group):
    """Test MTP-specific model loading with unified model approach."""

    # Setup mocks
    mock_model = mock.MagicMock()
    mock_model.model.embed_tokens.weight.shape = (131072, 4096)
    mock_get_model.return_value = mock_model
    # MTP does not have its own embed_tokens or lm_head
    # so it should share them with the target model
    mock_model.has_own_embed_tokens = False
    mock_model.has_own_lm_head = False

    target_attn_layers = {"target_attn_1": mock.MagicMock()}
    all_attn_layers = {**target_attn_layers, "draft_attn_1": mock.MagicMock()}
    target_indexer_layers: dict = {}
    all_indexer_layers: dict = {}

    mock_get_layers.side_effect = [
        target_attn_layers,
        target_indexer_layers,
        all_attn_layers,
        all_indexer_layers,
    ]

    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = 1
    mock_get_pp_group.return_value = mock_pp_group

    # Create target model
    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock

    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.model.embed_tokens.weight.shape = (131072, 4096)
    target_model.lm_head = mock.MagicMock()

    # Create MTP proposer
    proposer = _create_mtp_proposer(num_speculative_tokens=4)
    proposer.load_model(target_model)

    # Verify MTP-specific behavior:
    # Model is loaded
    mock_get_model.assert_called_once()
    # MTP shares lm_head with target model
    assert proposer.model.lm_head == target_model.lm_head
    # MTP shares embed_tokens with target model
    assert proposer.model.model.embed_tokens == target_model.model.embed_tokens


@pytest.mark.parametrize("num_speculative_tokens", [1])
def test_mtp_propose(num_speculative_tokens, monkeypatch):
    """Test that MTP's forward method returns hidden states directly"""

    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
    vocab_size = 100

    proposer = _create_mtp_proposer(num_speculative_tokens)
    hidden_size = proposer.hidden_size

    # Mock the MTP model to verify it returns hidden states directly
    model_mock = mock.MagicMock()

    # MTP returns hidden states directly
    if num_speculative_tokens == 1:
        model_mock.return_value = torch.zeros(total_tokens, hidden_size, device=device)
    else:
        # Multiple forward passes for multi-token speculation
        forward_returns = []
        for i in range(num_speculative_tokens):
            if i == 0:
                h_states = torch.zeros(total_tokens, hidden_size, device=device)
            else:
                h_states = torch.zeros(batch_size, hidden_size, device=device)
            forward_returns.append(h_states)
        model_mock.side_effect = forward_returns

    # Mock compute_logits
    def create_deterministic_logits(batch_size, vocab_size, token_offset):
        logits = torch.full((batch_size, vocab_size), -100.0, device=device)
        logits[:, token_offset] = 100.0
        return logits

    if num_speculative_tokens == 1:
        model_mock.compute_logits.return_value = create_deterministic_logits(
            batch_size, vocab_size, 42
        )
    else:
        logits_returns = [
            create_deterministic_logits(batch_size, vocab_size, 42 + i)
            for i in range(num_speculative_tokens)
        ]
        model_mock.compute_logits.side_effect = logits_returns

    proposer.model = model_mock
    proposer._draft_attn_layer_names = {"layer.0"}

    # Prepare inputs
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )

    target_token_ids = torch.randint(0, vocab_size, (total_tokens,), device=device)
    target_positions = torch.cat(
        [
            torch.arange(seq_lens[0], device=device),
            torch.arange(seq_lens[1], device=device),
        ]
    )
    target_hidden_states = torch.randn(total_tokens, hidden_size, device=device)
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device=device
    )
    sampling_metadata = mock.MagicMock()

    # Setup attention metadata
    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )

    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=list(proposer._draft_attn_layer_names),
        vllm_config=proposer.vllm_config,
        device=device,
    )

    proposer.runner = mock.MagicMock()
    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value = attn_metadata_builder
    mock_attn_group.layer_names = list(proposer._draft_attn_layer_names)
    mock_attn_group.kv_cache_spec = attn_metadata_builder.kv_cache_spec
    proposer.draft_attn_groups = [mock_attn_group]

    # Run propose
    result = proposer.propose(
        num_speculative_tokens=num_speculative_tokens,
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
    )

    # Verify the model was called correctly
    assert model_mock.called
    # Verify output shape
    assert result.shape == (batch_size, num_speculative_tokens)


@pytest.mark.parametrize("outcome", ["accept_all", "reject_all", "partial_prefix"])
def test_mtp_sparse_mla_rollback(outcome, monkeypatch):
    """Finalize the postprocessed MTP prefix before proposing more drafts."""
    boundaries = {
        "accept_all": 11,
        "reject_all": 8,
        "partial_prefix": 10,
    }
    committed_boundary = boundaries[outcome]
    postprocessed_num_computed_tokens = torch.tensor(
        [3, committed_boundary], dtype=torch.int32
    )
    idx_mapping = torch.tensor([1], dtype=torch.int32)
    layer_names = ("main.0", "main.1")
    block_size = 8
    newest_ids = torch.tensor([[[9, 10, 8]], [[9, 10, 8]]], dtype=torch.int64)
    num_newest_slots = newest_ids.shape[-1]
    resident_ids = torch.tensor([[[7, 8, 11, 12]], [[7, 8, 11, 12]]], dtype=torch.int64)
    resident_access = torch.tensor(
        [[[70, 80, 110, 120]], [[170, 180, 210, 220]]], dtype=torch.int64
    )
    provisional_slots = torch.tensor(
        [
            [[[0, 1], [2, 3], [4, 5]]],
            [[[6, 7], [8, 9], [10, 11]]],
        ],
        dtype=torch.int32,
    )
    newest_main_kv = torch.tensor(
        [[[[109], [110], [108]]], [[[209], [210], [208]]]],
        dtype=torch.bfloat16,
    )

    manager = SparseMLAOffloadManager.__new__(SparseMLAOffloadManager)
    manager._closing = manager._closed = False
    manager._is_host_writer = True
    manager._plan = SimpleNamespace(main_layer_names=layer_names)
    manager._local_buffers = {
        "request_block_ids": torch.tensor([[0, 1]], dtype=torch.int32),
        "request_num_blocks": torch.tensor([2], dtype=torch.int32),
        "request_num_tokens": torch.tensor([99], dtype=torch.int32),
        "request_active": torch.tensor([True]),
        "resident_main_kv": torch.zeros((2, 1, 4, 1), dtype=torch.bfloat16),
        "resident_logical_ids": resident_ids.clone(),
        "resident_last_access": resident_access.clone(),
        "newest_main_kv": newest_main_kv.clone(),
        "newest_logical_ids": newest_ids.clone(),
        "provisional_slots": provisional_slots.clone(),
        "tp_fence_token": torch.tensor([-1], dtype=torch.int32),
    }
    host_views = {
        name: torch.full((2, block_size, 1), -77, dtype=torch.bfloat16)
        for name in layer_names
    }
    manager._host_views = host_views
    manager._layer_views = {
        name: SimpleNamespace(
            layer_name=name,
            layer_index=index,
            is_host_writer=True,
            main_host_kv=host_views[name],
            main_host_kv_uva=host_views[name],
            local_buffers=manager._local_buffers,
        )
        for index, name in enumerate(layer_names)
    }

    finalizer = getattr(manager, "_finalize_mtp_batch", None)
    assert callable(finalizer), "C7 Manager finalize seam is missing"

    native_calls: list[
        tuple[
            tuple[torch.Tensor | bool | int, ...],
            dict[str, torch.Tensor | bool | int],
        ]
    ] = []
    native_phase_events: list[tuple[int, int]] = []

    def finalize_transfer(*args, **kwargs):
        native_calls.append((args, kwargs))
        buffers = manager._local_buffers
        assert len(args) == 22, "C7 finalize transfer requires phase/status ABI"
        phase = args[21]
        assert isinstance(phase, int)
        validation_count = sum(
            event_phase == 0 for event_phase, _ in native_phase_events
        )
        commit_count = sum(event_phase == 1 for event_phase, _ in native_phase_events)
        if phase == 0:
            assert commit_count == 0
            layer_index = validation_count
        elif phase == 1:
            assert validation_count == len(layer_names)
            layer_index = commit_count
        else:
            pytest.fail("C7 finalize transfer phase must be validate or commit")
        assert torch.equal(
            buffers["request_num_tokens"],
            torch.tensor([committed_boundary], dtype=torch.int32),
        )
        status = args[17]
        expected_status = buffers["tp_fence_token"]
        assert isinstance(status, torch.Tensor)
        assert status.data_ptr() == expected_status.data_ptr()
        assert status.shape == expected_status.shape
        assert status.stride() == expected_status.stride()
        assert status.storage_offset() == expected_status.storage_offset()
        assert status.dtype == expected_status.dtype
        assert status.device == expected_status.device
        assert args[20] is True
        expected_argument_views = {
            7: buffers["newest_logical_ids"][layer_index],
            8: buffers["provisional_slots"][layer_index],
            9: buffers["provisional_slots"][layer_index],
            13: buffers["resident_logical_ids"][layer_index],
            14: buffers["resident_last_access"][layer_index],
            15: buffers["provisional_slots"][layer_index],
        }
        for argument_index, expected_view in expected_argument_views.items():
            actual_view = args[argument_index]
            assert isinstance(actual_view, torch.Tensor)
            assert actual_view.data_ptr() == expected_view.data_ptr()
            assert actual_view.shape == expected_view.shape
            assert actual_view.stride() == expected_view.stride()
            assert actual_view.storage_offset() == expected_view.storage_offset()
            assert actual_view.dtype == expected_view.dtype
            assert actual_view.device == expected_view.device
        native_phase_events.append((phase, layer_index))
        if phase == 0:
            assert torch.equal(status, torch.zeros_like(status))
            if any(
                logical_id >= 0 and newest_slot != logical_id % num_newest_slots
                for newest_slot, logical_id in enumerate(
                    buffers["newest_logical_ids"][layer_index, 0].tolist()
                )
            ):
                status.fill_(1)
            return
        if torch.equal(status, torch.ones_like(status)):
            return
        assert torch.equal(status, torch.zeros_like(status))
        host_view = tuple(host_views.values())[layer_index]
        for newest_slot, logical_id in enumerate(
            buffers["newest_logical_ids"][layer_index, 0].tolist()
        ):
            if logical_id < 0:
                continue
            if newest_slot != logical_id % num_newest_slots:
                raise ValueError("newest logical ID is in the wrong fixed slot")
            if logical_id < committed_boundary and manager._is_host_writer:
                block = logical_id // block_size
                host_view[block, logical_id % block_size].copy_(
                    buffers["newest_main_kv"][layer_index, 0, newest_slot]
                )
            if logical_id >= committed_boundary:
                buffers["newest_logical_ids"][layer_index, 0, newest_slot] = -1
        invalid_residents = (
            buffers["resident_logical_ids"][layer_index, 0] >= committed_boundary
        )
        buffers["resident_logical_ids"][layer_index, 0][invalid_residents] = -1
        buffers["resident_last_access"][layer_index, 0][invalid_residents] = 0
        buffers["provisional_slots"][layer_index, 0].fill_(-1)

    monkeypatch.setattr(
        sparse_mla_offload_module,
        "ops",
        SimpleNamespace(sparse_mla_offload_transfer=finalize_transfer),
        raising=False,
    )
    fixed_pointers = {
        name: tensor.data_ptr()
        for name, tensor in {**manager._local_buffers, **host_views}.items()
    }

    finalizer(idx_mapping, postprocessed_num_computed_tokens)

    assert len(native_calls) == 2 * len(layer_names)
    assert native_phase_events == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert manager._local_buffers["request_num_tokens"].tolist() == [committed_boundary]
    for layer_index, host_view in enumerate(host_views.values()):
        for newest_slot, logical_id in enumerate((9, 10, 8)):
            host_row = host_view[logical_id // block_size, logical_id % block_size]
            if logical_id < committed_boundary:
                assert torch.equal(
                    host_row,
                    newest_main_kv[layer_index, 0, newest_slot],
                )
            else:
                assert torch.equal(host_row, torch.tensor([-77], dtype=torch.bfloat16))
        expected_newest = torch.tensor(
            [
                logical_id if logical_id < committed_boundary else -1
                for logical_id in (9, 10, 8)
            ],
            dtype=torch.int64,
        )
        assert torch.equal(
            manager._local_buffers["newest_logical_ids"][layer_index, 0],
            expected_newest,
        )
        expected_resident_ids = torch.tensor(
            [
                logical_id if logical_id < committed_boundary else -1
                for logical_id in (7, 8, 11, 12)
            ],
            dtype=torch.int64,
        )
        expected_resident_access = torch.tensor(
            [
                access if logical_id < committed_boundary else 0
                for logical_id, access in zip(
                    (7, 8, 11, 12), resident_access[layer_index, 0].tolist()
                )
            ],
            dtype=torch.int64,
        )
        assert torch.equal(
            manager._local_buffers["resident_logical_ids"][layer_index, 0],
            expected_resident_ids,
        )
        assert torch.equal(
            manager._local_buffers["resident_last_access"][layer_index, 0],
            expected_resident_access,
        )
        assert torch.all(
            manager._local_buffers["provisional_slots"][layer_index, 0] == -1
        )

    host_after_valid_finalize = {
        name: value.clone() for name, value in host_views.items()
    }
    for malformed_slot in range(num_newest_slots):
        for name, value in host_after_valid_finalize.items():
            host_views[name].copy_(value)
        manager._local_buffers["resident_main_kv"].zero_()
        manager._local_buffers["resident_logical_ids"].copy_(resident_ids)
        manager._local_buffers["resident_last_access"].copy_(resident_access)
        manager._local_buffers["newest_main_kv"].copy_(newest_main_kv)
        manager._local_buffers["newest_logical_ids"].copy_(newest_ids)
        manager._local_buffers["provisional_slots"].copy_(provisional_slots)
        malformed_logical_id = newest_ids[1, 0, (malformed_slot + 1) % num_newest_slots]
        assert malformed_logical_id % num_newest_slots != malformed_slot
        manager._local_buffers["newest_logical_ids"][1, 0, malformed_slot] = (
            malformed_logical_id
        )
        manager._local_buffers["tp_fence_token"].fill_(-1)
        host_before_mismatch = {
            name: value.clone() for name, value in host_views.items()
        }
        malformed_buffers_before = {
            name: manager._local_buffers[name].clone()
            for name in (
                "resident_main_kv",
                "resident_logical_ids",
                "resident_last_access",
                "newest_main_kv",
                "newest_logical_ids",
                "provisional_slots",
            )
        }
        native_calls.clear()
        native_phase_events.clear()
        finalizer(idx_mapping, postprocessed_num_computed_tokens)
        assert native_phase_events == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert all(
            torch.equal(host_views[name], value)
            for name, value in host_before_mismatch.items()
        )
        assert all(
            torch.equal(manager._local_buffers[name], value)
            for name, value in malformed_buffers_before.items()
        )

    manager._is_host_writer = False
    for layer_view in manager._layer_views.values():
        layer_view.is_host_writer = False
    manager._local_buffers["resident_logical_ids"].copy_(resident_ids)
    manager._local_buffers["resident_last_access"].copy_(resident_access)
    manager._local_buffers["newest_logical_ids"].copy_(newest_ids)
    manager._local_buffers["provisional_slots"].copy_(provisional_slots)
    manager._local_buffers["tp_fence_token"].fill_(-1)
    host_before_follower = {name: value.clone() for name, value in host_views.items()}
    native_calls.clear()
    native_phase_events.clear()
    finalizer(idx_mapping, postprocessed_num_computed_tokens)
    assert all(
        torch.equal(host_views[name], value)
        for name, value in host_before_follower.items()
    )
    for layer_index in range(len(layer_names)):
        assert torch.equal(
            manager._local_buffers["newest_logical_ids"][layer_index, 0],
            torch.tensor(
                [
                    logical_id if logical_id < committed_boundary else -1
                    for logical_id in (9, 10, 8)
                ],
                dtype=torch.int64,
            ),
        )
        assert torch.equal(
            manager._local_buffers["resident_logical_ids"][layer_index, 0],
            torch.tensor(
                [
                    logical_id if logical_id < committed_boundary else -1
                    for logical_id in (7, 8, 11, 12)
                ],
                dtype=torch.int64,
            ),
        )
        assert torch.equal(
            manager._local_buffers["resident_last_access"][layer_index, 0],
            torch.tensor(
                [
                    access if logical_id < committed_boundary else 0
                    for logical_id, access in zip(
                        (7, 8, 11, 12), resident_access[layer_index, 0].tolist()
                    )
                ],
                dtype=torch.int64,
            ),
        )
        assert torch.all(
            manager._local_buffers["provisional_slots"][layer_index, 0] == -1
        )
    assert all(
        tensor.data_ptr() == fixed_pointers[name]
        for name, tensor in {**manager._local_buffers, **host_views}.items()
    )
