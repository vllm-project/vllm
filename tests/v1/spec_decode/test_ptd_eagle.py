# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    AttentionConfig,
    CacheConfig,
    CUDAGraphMode,
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
from vllm.v1.spec_decode.ptd_eagle import PtdEagleProposer

model_dir = "openai/gpt-oss-120b"
draft_model_dir = "nvidia/gpt-oss-120b-Eagle3-throughput"

KERNEL_CONFIG = {
    "hidden_size": 256,
    "block_size": 16,
    "max_blocks": 2048,
    "mask_token_id": 128256,
    "max_model_len": 32768,
    "HIDDEN_TILE_SIZE": 256,
}


def build_kernel_test_data(
    K: int,
    num_verified: list[int],
    start_pos: list[int] | None = None,
) -> dict:
    """Build kernel test inputs and expected outputs from minimal parameters."""
    batch_size = len(num_verified)
    draft_len = K - 1
    mask_token_id = KERNEL_CONFIG["mask_token_id"]

    if start_pos is None:
        start_pos = [0] * batch_size

    target_token_ids = []
    target_positions = []
    input_query_start_loc = [0]
    output_query_start_loc = [0]
    last_token_indices = []

    token_counter = 0
    for i, num_verified_tokens in enumerate(num_verified):
        target_token_ids.extend(
            range(token_counter, token_counter + num_verified_tokens)
        )
        pos_range = range(start_pos[i], start_pos[i] + num_verified_tokens)
        target_positions.extend(pos_range)
        input_query_start_loc.append(input_query_start_loc[-1] + num_verified_tokens)
        output_query_start_loc.append(
            output_query_start_loc[-1] + num_verified_tokens + draft_len
        )
        last_token_indices.append(input_query_start_loc[-2] + num_verified_tokens - 1)
        token_counter += num_verified_tokens

    next_token_ids = [100 + i for i in range(batch_size)]

    expected_token_ids = []
    expected_positions = []
    for i, num_verified_tokens in enumerate(num_verified):
        request_start_idx = input_query_start_loc[i]
        # Shift left: drop first, keep [1:n], append next_token
        for j in range(1, num_verified_tokens):
            expected_token_ids.append(target_token_ids[request_start_idx + j])
            expected_positions.append(target_positions[request_start_idx + j - 1])
        expected_token_ids.append(next_token_ids[i])
        expected_positions.append(
            target_positions[request_start_idx + num_verified_tokens - 1]
        )
        # Append K-1 mask tokens
        last_position = target_positions[request_start_idx + num_verified_tokens - 1]
        for d in range(1, K):
            expected_token_ids.append(mask_token_id)
            expected_positions.append(last_position + d)

    return {
        "batch_size": batch_size,
        "num_spec_tokens": K,
        "target_token_ids": target_token_ids,
        "target_positions": target_positions,
        "next_token_ids": next_token_ids,
        "last_token_indices": last_token_indices,
        "input_query_start_loc": input_query_start_loc,
        "output_query_start_loc": output_query_start_loc,
        "expected_token_ids": expected_token_ids,
        "expected_positions": expected_positions,
    }


def _create_proposer(
    method: str,
    num_speculative_tokens: int,
    attention_backend: str | None = None,
) -> PtdEagleProposer:
    model_config = ModelConfig(model=model_dir, runner="generate", max_model_len=100)

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=draft_model_dir,
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        parallel_draft=True,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=current_platform.device_type),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        attention_config=AttentionConfig(backend=attention_backend),
    )

    return PtdEagleProposer(
        vllm_config=vllm_config, device=current_platform.device_type
    )


def run_ptd_kernel(
    device: torch.device,
    config: dict,
    *,
    batch_size: int,
    target_token_ids: list[int],
    target_positions: list[int],
    target_hidden: torch.Tensor | None = None,
    mask_hidden_val: float = 99.0,
    next_token_ids: list[int],
    last_token_indices: list[int],
    original_slot_mapping: list[int] | None = None,
    in_query_start_loc: list[int],
    out_query_start_loc: list[int],
    block_table: torch.Tensor | None = None,
    max_model_len: int | None = None,
) -> dict:
    """Run parallel drafting kernel and return outputs."""
    from vllm.v1.spec_decode.ptd_eagle import ptd_prepare_inputs_kernel

    hidden_size = config["hidden_size"]
    block_size = config["block_size"]
    max_blocks = config["max_blocks"]
    mask_token_id = config["mask_token_id"]
    HIDDEN_TILE_SIZE = config["HIDDEN_TILE_SIZE"]
    if max_model_len is None:
        max_model_len = config["max_model_len"]

    num_tokens = len(target_token_ids)
    total_output_tokens = out_query_start_loc[-1]

    target_token_ids_gpu = torch.tensor(
        target_token_ids, dtype=torch.int32, device=device
    )
    target_positions_gpu = torch.tensor(
        target_positions, dtype=torch.int32, device=device
    )
    if target_hidden is None:
        target_hidden = torch.randn(num_tokens, hidden_size, device=device)
    mask_hidden = torch.full((hidden_size,), mask_hidden_val, device=device)
    next_token_ids_gpu = torch.tensor(next_token_ids, dtype=torch.int32, device=device)
    last_token_indices_gpu = torch.tensor(
        last_token_indices, dtype=torch.int32, device=device
    )

    if original_slot_mapping is None:
        original_slot_mapping = target_positions
    original_slot_mapping_gpu = torch.tensor(
        original_slot_mapping, dtype=torch.int64, device=device
    )

    if block_table is None:
        block_table = torch.arange(max_blocks, dtype=torch.int32, device=device)
        block_table = block_table.unsqueeze(0).expand(batch_size, -1).contiguous()

    in_query_start_loc_gpu = torch.tensor(
        in_query_start_loc, dtype=torch.int32, device=device
    )
    out_query_start_loc_gpu = torch.tensor(
        out_query_start_loc, dtype=torch.int32, device=device
    )

    out_input_ids = torch.zeros(total_output_tokens, dtype=torch.int32, device=device)
    out_positions = torch.zeros(total_output_tokens, dtype=torch.int32, device=device)
    out_hidden = torch.zeros(total_output_tokens, hidden_size, device=device)
    out_slot_mapping = torch.zeros(
        total_output_tokens, dtype=torch.int64, device=device
    )

    num_hidden_tiles = (hidden_size + HIDDEN_TILE_SIZE - 1) // HIDDEN_TILE_SIZE

    ptd_prepare_inputs_kernel[(total_output_tokens, num_hidden_tiles)](
        target_token_ids_gpu,
        target_positions_gpu,
        target_hidden,
        mask_hidden,
        next_token_ids_gpu,
        last_token_indices_gpu,
        original_slot_mapping_gpu,
        block_table,
        in_query_start_loc_gpu,
        out_query_start_loc_gpu,
        out_input_ids,
        out_positions,
        out_hidden,
        out_slot_mapping,
        batch_size=batch_size,
        hidden_size=hidden_size,
        block_size=block_size,
        max_blocks=block_table.shape[1],
        mask_token_id=mask_token_id,
        max_model_len=max_model_len,
        HIDDEN_TILE_SIZE=HIDDEN_TILE_SIZE,
    )

    return {
        "input_ids": out_input_ids.cpu(),
        "positions": out_positions.cpu(),
        "hidden": out_hidden,
        "slot_mapping": out_slot_mapping.cpu(),
        "target_hidden": target_hidden,
        "mask_hidden": mask_hidden,
        "mask_token_id": mask_token_id,
    }


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton kernel requires CUDA"
)
@pytest.mark.parametrize(
    "K, num_verified, start_pos",
    [
        # Single request scenarios
        pytest.param(1, [3], [5], id="K=1_single"),
        pytest.param(4, [1], [10], id="K=4_all_rejected"),
        pytest.param(4, [4], [0], id="K=4_all_accepted"),
        pytest.param(4, [3], [0], id="K=4_partial"),
        pytest.param(2, [3], [0], id="K=2_single"),
        pytest.param(6, [2], [0], id="K=6_single"),
        # Batched scenarios (batch_size=2)
        pytest.param(1, [3, 2], [0, 0], id="K=1_batch2"),
        pytest.param(4, [1, 1], [10, 20], id="K=4_batch2_all_rejected"),
        pytest.param(4, [4, 3], [0, 0], id="K=4_batch2_all_accepted"),
        pytest.param(4, [3, 1], [5, 10], id="K=4_batch2_mixed"),
        pytest.param(2, [3, 2], [0, 0], id="K=2_batch2"),
        # Batch size 3
        pytest.param(4, [4, 2, 1], [0, 0, 0], id="K=4_batch3_varying"),
        # Larger K value
        pytest.param(8, [3], [0], id="K=8_single"),
        # Long sequence positions (deep in context)
        pytest.param(4, [3], [8000], id="K=4_pos_8k"),
        pytest.param(4, [2, 3], [16000, 8000], id="K=4_batch2_long_pos"),
        # Larger batch
        pytest.param(4, [2, 3, 1, 4], [0, 0, 0, 0], id="K=4_batch4"),
    ],
)
def test_ptd_kernel_scenarios(K, num_verified, start_pos):
    """Parametrized test covering core kernel scenarios."""
    device = torch.device("cuda")
    scenario = build_kernel_test_data(K, num_verified, start_pos)

    result = run_ptd_kernel(
        device,
        KERNEL_CONFIG,
        batch_size=scenario["batch_size"],
        target_token_ids=scenario["target_token_ids"],
        target_positions=scenario["target_positions"],
        next_token_ids=scenario["next_token_ids"],
        last_token_indices=scenario["last_token_indices"],
        in_query_start_loc=scenario["input_query_start_loc"],
        out_query_start_loc=scenario["output_query_start_loc"],
    )

    expected_token_ids = torch.tensor(scenario["expected_token_ids"], dtype=torch.int32)
    assert torch.equal(result["input_ids"], expected_token_ids), (
        f"input_ids mismatch: got {result['input_ids']}, expected {expected_token_ids}"
    )

    expected_positions = torch.tensor(scenario["expected_positions"], dtype=torch.int32)
    assert torch.equal(result["positions"], expected_positions), (
        f"positions mismatch: got {result['positions']}, expected {expected_positions}"
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton kernel requires CUDA"
)
def test_ptd_kernel_slot_mapping():
    """Test that slot mapping is computed correctly."""
    device = torch.device("cuda")
    scenario = build_kernel_test_data(K=4, num_verified=[3], start_pos=[5])

    result = run_ptd_kernel(
        device,
        KERNEL_CONFIG,
        batch_size=scenario["batch_size"],
        target_token_ids=scenario["target_token_ids"],
        target_positions=scenario["target_positions"],
        next_token_ids=scenario["next_token_ids"],
        last_token_indices=scenario["last_token_indices"],
        in_query_start_loc=scenario["input_query_start_loc"],
        out_query_start_loc=scenario["output_query_start_loc"],
    )

    # Slots should match positions for simple block table (identity mapping)
    expected_slots = torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.int64)
    assert torch.equal(result["slot_mapping"], expected_slots), (
        f"slot_mapping mismatch: got {result['slot_mapping']}, "
        f"expected {expected_slots}"
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton kernel requires CUDA"
)
def test_ptd_overflow_uses_padding_slot():
    """Verify PADDING_SLOT_ID (-1) is used when positions exceed max_model_len."""
    device = torch.device("cuda")
    PADDING_SLOT_ID = -1

    result = run_ptd_kernel(
        device,
        KERNEL_CONFIG,
        batch_size=1,
        target_token_ids=[10, 20, 30],
        target_positions=[97, 98, 99],
        next_token_ids=[99],
        last_token_indices=[2],
        original_slot_mapping=[97, 98, 99],
        in_query_start_loc=[0, 3],
        out_query_start_loc=[0, 6],
        max_model_len=100,
    )

    # Draft positions (100, 101, 102) exceed max_model_len=100
    draft_slots = result["slot_mapping"][3:]
    assert torch.all(draft_slots == PADDING_SLOT_ID), (
        f"Expected PADDING_SLOT_ID for overflow, got {draft_slots}"
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton kernel requires CUDA"
)
def test_ptd_slot_crosses_block_boundary():
    """Test slot computation when draft positions span KV cache block boundaries."""
    device = torch.device("cuda")
    max_blocks = KERNEL_CONFIG["max_blocks"]

    # Create block table where block 0 -> physical 5, block 1 -> physical 7
    block_table = torch.zeros(max_blocks, dtype=torch.int32, device=device)
    block_table[0] = 5
    block_table[1] = 7
    block_table = block_table.unsqueeze(0)

    result = run_ptd_kernel(
        device,
        KERNEL_CONFIG,
        batch_size=1,
        target_token_ids=[10, 20],
        target_positions=[14, 15],  # End of block 0
        next_token_ids=[42],
        last_token_indices=[1],
        original_slot_mapping=[14, 15],
        in_query_start_loc=[0, 2],
        out_query_start_loc=[0, 5],
        block_table=block_table,
    )

    # Verified: from slot_mapping. Draft: computed via block_table (block 1 = phys 7)
    expected_slots = torch.tensor([14, 15, 112, 113, 114], dtype=torch.int64)
    assert torch.equal(result["slot_mapping"], expected_slots)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton kernel requires CUDA"
)
def test_ptd_hidden_states_copied_correctly():
    """Verify hidden states are copied from target or mask_hidden correctly."""
    device = torch.device("cuda")
    hidden_size = KERNEL_CONFIG["hidden_size"]

    target_hidden = torch.zeros(3, hidden_size, device=device)
    target_hidden[0, :] = 1.0
    target_hidden[1, :] = 2.0
    target_hidden[2, :] = 3.0

    result = run_ptd_kernel(
        device,
        KERNEL_CONFIG,
        batch_size=1,
        target_token_ids=[10, 20, 30],
        target_positions=[5, 6, 7],
        target_hidden=target_hidden,
        mask_hidden_val=99.0,
        next_token_ids=[42],
        last_token_indices=[2],
        in_query_start_loc=[0, 3],
        out_query_start_loc=[0, 6],
    )

    assert torch.allclose(result["hidden"][0], target_hidden[0])
    assert torch.allclose(result["hidden"][1], target_hidden[1])
    assert torch.allclose(result["hidden"][2], target_hidden[2])
    for i in range(3, 6):
        assert torch.allclose(result["hidden"][i], result["mask_hidden"])


@mock.patch("vllm.v1.spec_decode.eagle.get_pp_group")
@mock.patch("vllm.v1.spec_decode.eagle.get_layers_from_vllm_config")
@mock.patch("vllm.v1.spec_decode.eagle.get_model")
def test_ptd_load_model(
    mock_get_model,
    mock_get_layers,
    mock_get_pp_group,
):
    """Test load_model sets up mask_token_id and mask_hidden."""
    proposer = _create_proposer(
        "eagle3", num_speculative_tokens=8, attention_backend="FLASH_ATTN"
    )
    proposer.draft_model_config.hf_config.ptd_token_id = "128256"

    draft_model = mock.MagicMock()
    draft_model.model = mock.MagicMock()
    draft_model.has_own_embed_tokens = False
    draft_model.model.embed_tokens = mock.MagicMock()
    draft_model.has_own_lm_head = False
    draft_model.lm_head = mock.MagicMock()
    draft_model.mask_hidden = torch.ones(
        proposer.hidden_size, dtype=torch.float32, device=proposer.device
    )
    mock_get_model.return_value = draft_model

    target_attn_layers = {"target_attn_1": mock.MagicMock()}
    all_attn_layers = {**target_attn_layers, "draft_extra_attn": mock.MagicMock()}
    mock_get_layers.side_effect = [target_attn_layers, {}, all_attn_layers, {}]

    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = 1
    mock_get_pp_group.return_value = mock_pp_group

    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock

    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.lm_head = mock.MagicMock()
    target_model.model.embed_tokens = mock.MagicMock()

    proposer.load_model(target_model)

    assert proposer.mask_token_id == 128256
    assert proposer.mask_hidden is draft_model.mask_hidden


@mock.patch("vllm.v1.spec_decode.eagle.get_pp_group")
@mock.patch("vllm.v1.spec_decode.eagle.get_layers_from_vllm_config")
@mock.patch("vllm.v1.spec_decode.eagle.get_model")
def test_ptd_load_model_requires_ptd_token_id(
    mock_get_model,
    mock_get_layers,
    mock_get_pp_group,
):
    """Test that missing ptd_token_id raises ValueError."""
    from types import SimpleNamespace

    proposer = _create_proposer("eagle3", num_speculative_tokens=4)
    proposer.draft_model_config.hf_config = SimpleNamespace()

    draft_model = mock.MagicMock()
    draft_model.model = mock.MagicMock()
    draft_model.has_own_embed_tokens = False
    draft_model.model.embed_tokens = mock.MagicMock()
    draft_model.has_own_lm_head = False
    draft_model.lm_head = mock.MagicMock()
    draft_model.mask_hidden = torch.zeros(proposer.hidden_size, dtype=torch.float32)
    mock_get_model.return_value = draft_model

    target_attn_layers = {"target_attn": mock.MagicMock()}
    all_attn_layers = {**target_attn_layers, "draft_extra_attn": mock.MagicMock()}
    mock_get_layers.side_effect = [target_attn_layers, {}, all_attn_layers, {}]

    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = 1
    mock_get_pp_group.return_value = mock_pp_group

    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock

    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.lm_head = mock.MagicMock()
    target_model.model.embed_tokens = mock.MagicMock()

    with pytest.raises(ValueError, match="ptd_token_id"):
        proposer.load_model(target_model)


@pytest.mark.parametrize("num_speculative_tokens", [2, 4])
def test_ptd_propose(num_speculative_tokens):
    """Test propose returns correct draft tokens."""
    device = torch.device(current_platform.device_type)

    batch_size = 2
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
    vocab_size = 128

    proposer = _create_proposer(
        "eagle3", num_speculative_tokens, attention_backend="FLASH_ATTN"
    )
    hidden_size = proposer.hidden_size

    model_mock = mock.MagicMock()
    proposer.model = model_mock
    proposer.attn_layer_names = ["layer.0"]

    backend_enum = AttentionBackendEnum.FLASH_ATTN

    attn_metadata_builder_cls, _ = try_get_attention_backend(backend_enum)
    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=proposer.attn_layer_names,
        vllm_config=proposer.vllm_config,
        device=device,
    )

    proposer.runner = mock.MagicMock()
    proposer.runner.attn_groups.append([mock.MagicMock()])
    proposer.runner.attn_groups[0][
        0
    ].get_metadata_builder.return_value = attn_metadata_builder
    proposer._get_attention_metadata_builder = mock.MagicMock(
        return_value=attn_metadata_builder
    )

    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )

    target_token_ids = torch.randint(
        0, vocab_size, (total_tokens,), dtype=torch.int32, device=device
    )
    target_positions = torch.cat(
        [torch.arange(s, device=device, dtype=torch.int32) for s in seq_lens]
    )
    target_hidden_states = torch.randn(total_tokens, hidden_size, device=device)
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device=device
    )
    sampling_metadata = mock.MagicMock()

    draft_len = num_speculative_tokens - 1
    total_output_tokens = (
        common_attn_metadata.num_actual_tokens + batch_size * draft_len
    )
    slot_mapping = torch.arange(total_output_tokens, device=device, dtype=torch.int64)

    proposer._prepare_ptd_inputs = mock.MagicMock(return_value=slot_mapping)
    proposer._get_ptd_cudagraph_config = mock.MagicMock(
        return_value=(total_output_tokens, CUDAGraphMode.NONE)
    )

    hidden_states = torch.zeros(total_output_tokens, hidden_size, device=device)
    proposer._run_ptd_forward = mock.MagicMock(return_value=hidden_states)

    base_token_ids = [42, 60]
    logits = torch.full(
        (batch_size * num_speculative_tokens, vocab_size), -100.0, device=device
    )
    for i in range(batch_size):
        for j in range(num_speculative_tokens):
            logits[i * num_speculative_tokens + j, base_token_ids[i] + j] = 100.0
    model_mock.compute_logits.return_value = logits

    result = proposer.propose(
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        last_token_indices=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
    )

    expected_tokens = torch.tensor(
        [
            [base_token_ids[0] + i for i in range(num_speculative_tokens)],
            [base_token_ids[1] + i for i in range(num_speculative_tokens)],
        ],
        device=device,
    )
    assert torch.equal(result, expected_tokens)


def test_ptd_propose_rejects_multimodal():
    """Test that multimodal inputs raise NotImplementedError."""
    device = torch.device(current_platform.device_type)
    proposer = _create_proposer("eagle3", num_speculative_tokens=4)
    dummy_tensor = torch.zeros(1, device=device)

    with pytest.raises(NotImplementedError):
        proposer.propose(
            target_token_ids=dummy_tensor,
            target_positions=dummy_tensor,
            target_hidden_states=dummy_tensor,
            next_token_ids=dummy_tensor,
            last_token_indices=None,
            common_attn_metadata=mock.MagicMock(),
            sampling_metadata=mock.MagicMock(),
            mm_embed_inputs=([dummy_tensor], dummy_tensor),
        )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton kernel requires CUDA"
)
def test_ptd_propose_updates_metadata():
    """Verify propose() correctly updates common_attn_metadata fields."""
    device = torch.device("cuda")

    num_speculative_tokens = 4
    proposer = _create_proposer(
        "eagle3", num_speculative_tokens, attention_backend="FLASH_ATTN"
    )
    hidden_size = proposer.hidden_size

    proposer.mask_token_id = 128256
    proposer.mask_hidden = torch.randn(hidden_size, device=device)

    batch_size = 2
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
    vocab_size = 128

    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )

    original_num_tokens = common_attn_metadata.num_actual_tokens
    original_max_query_len = common_attn_metadata.max_query_len

    target_token_ids = torch.randint(
        0, vocab_size, (total_tokens,), dtype=torch.int32, device=device
    )
    target_positions = torch.cat(
        [torch.arange(s, device=device, dtype=torch.int32) for s in seq_lens]
    )
    target_hidden_states = torch.randn(total_tokens, hidden_size, device=device)
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device=device
    )
    sampling_metadata = mock.MagicMock()

    model_mock = mock.MagicMock()
    # combine_hidden_states is called for eagle3 - return input unchanged
    model_mock.combine_hidden_states.side_effect = lambda x: x
    proposer.model = model_mock
    proposer.attn_layer_names = ["layer.0"]

    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )
    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=proposer.attn_layer_names,
        vllm_config=proposer.vllm_config,
        device=device,
    )

    proposer.runner = mock.MagicMock()
    proposer.runner.attn_groups.append([mock.MagicMock()])
    proposer.runner.attn_groups[0][
        0
    ].get_metadata_builder.return_value = attn_metadata_builder
    proposer._get_attention_metadata_builder = mock.MagicMock(
        return_value=attn_metadata_builder
    )

    draft_len = num_speculative_tokens - 1
    total_output_tokens = original_num_tokens + batch_size * draft_len

    proposer._get_ptd_cudagraph_config = mock.MagicMock(
        return_value=(total_output_tokens, CUDAGraphMode.NONE)
    )
    hidden_states = torch.zeros(total_output_tokens, hidden_size, device=device)
    proposer._run_ptd_forward = mock.MagicMock(return_value=hidden_states)

    logits = torch.full(
        (batch_size * num_speculative_tokens, vocab_size), -100.0, device=device
    )
    for i in range(batch_size):
        logits[i * num_speculative_tokens, 42 + i] = 100.0
    model_mock.compute_logits.return_value = logits

    proposer.propose(
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        last_token_indices=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
    )

    assert common_attn_metadata.num_actual_tokens == total_output_tokens
    assert common_attn_metadata.max_query_len == original_max_query_len + draft_len
    assert common_attn_metadata.slot_mapping is not None
    assert common_attn_metadata.slot_mapping.shape[0] == total_output_tokens


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton kernel requires CUDA"
)
def test_ptd_prepare_inputs_method():
    """Test _prepare_ptd_inputs method with real kernel execution."""
    device = torch.device("cuda")

    num_speculative_tokens = 4
    proposer = _create_proposer("eagle3", num_speculative_tokens)

    proposer.mask_token_id = 128256
    proposer.mask_hidden = torch.randn(
        proposer.hidden_size, device=device, dtype=torch.float32
    )

    batch_size = 2
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)

    target_token_ids = torch.arange(total_tokens, dtype=torch.int32, device=device)
    target_positions = torch.cat(
        [torch.arange(s, device=device, dtype=torch.int32) for s in seq_lens]
    )
    target_hidden_states = torch.randn(
        total_tokens, proposer.hidden_size, device=device
    )
    next_token_ids = torch.tensor([100, 101], dtype=torch.int32, device=device)

    last_token_indices = torch.tensor([4, 7], dtype=torch.int32, device=device)

    slot_mapping = torch.arange(total_tokens, dtype=torch.int64, device=device)
    block_table = torch.arange(
        KERNEL_CONFIG["max_blocks"], dtype=torch.int32, device=device
    )
    block_table = block_table.unsqueeze(0).expand(batch_size, -1).contiguous()

    input_query_start_loc = torch.tensor(
        [0, seq_lens[0], total_tokens], dtype=torch.int32, device=device
    )

    draft_len = num_speculative_tokens - 1
    accepted_lengths = last_token_indices - input_query_start_loc[:batch_size] + 1
    out_lens = accepted_lengths + draft_len
    output_query_start_loc = torch.zeros(
        batch_size + 1, dtype=torch.int32, device=device
    )
    output_query_start_loc[1:] = torch.cumsum(out_lens, dim=0)

    total_output_tokens = total_tokens + batch_size * draft_len

    result_slot_mapping = proposer._prepare_ptd_inputs(
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        last_token_indices,
        slot_mapping,
        block_table,
        input_query_start_loc,
        output_query_start_loc,
        total_output_tokens,
        batch_size,
    )

    assert result_slot_mapping.shape[0] == total_output_tokens
    assert proposer.input_ids[:total_output_tokens].shape[0] == total_output_tokens
    assert proposer.positions[:total_output_tokens].shape[0] == total_output_tokens
    assert proposer.hidden_states[:total_output_tokens].shape == (
        total_output_tokens,
        proposer.hidden_size,
    )

    out_input_ids = proposer.input_ids[:total_output_tokens].cpu()
    output_query_start_locs = output_query_start_loc.cpu().tolist()
    for i in range(batch_size):
        start = output_query_start_locs[i]
        end = output_query_start_locs[i + 1]
        non_draft_tokens = out_input_ids[start : end - draft_len]
        assert torch.all(non_draft_tokens != proposer.mask_token_id)
        draft_tokens = out_input_ids[end - draft_len : end]
        assert torch.all(draft_tokens == proposer.mask_token_id)
