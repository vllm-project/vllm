# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import pytest
import torch

from tests.utils import get_attn_backend_list_based_on_platform
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    try_get_attention_backend,
)
from vllm.config import (
    AttentionConfig,
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
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

model_dir = "meta-llama/Llama-3.1-8B-Instruct"
eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
eagle3_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


def _create_proposer(
    method: str,
    num_speculative_tokens: int,
    attention_backend: str | None = None,
    speculative_token_tree: list[tuple[int, ...]] | None = None,
) -> EagleProposer:
    model_config = ModelConfig(model=model_dir, runner="generate", max_model_len=100)

    # Choose model directory based on method
    draft_model_dir = eagle_dir if method == "eagle" else eagle3_dir

    spec_token_tree_str = None
    if speculative_token_tree is not None:
        assert num_speculative_tokens == len(speculative_token_tree)
        spec_token_tree_str = str(speculative_token_tree)

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=draft_model_dir,
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        speculative_token_tree=spec_token_tree_str,
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

    return EagleProposer(vllm_config=vllm_config, device=current_platform.device_type)


def test_prepare_next_token_ids():
    """
    Test for prepare_next_token_ids_cpu and prepare_next_token_ids_padded.
    Each will produce a device tensor of next_token_ids, taking as input
    either the GPU tensor of sampled_token_ids with -1 for rejected tokens,
    or the CPU python list[list[int]] with the rejected tokens removed.
    """
    device = torch.device(current_platform.device_type)

    num_requests = 4
    num_speculative_tokens = 4
    batch_spec = BatchSpec(
        seq_lens=[num_speculative_tokens + 1] * num_requests,
        query_lens=[num_speculative_tokens + 1] * num_requests,
    )

    req_ids = [f"req_{i + 1}" for i in range(num_requests)]
    mock_input_batch = mock.MagicMock(spec=InputBatch)
    mock_input_batch.req_ids = req_ids
    mock_input_batch.num_reqs = num_requests
    mock_input_batch.vocab_size = 100

    mock_num_scheduled_tokens = {req_id: 0 for req_id in req_ids}
    mock_requests = {}
    for req_id in req_ids:
        mock_request = mock.MagicMock(spec=CachedRequestState)
        # Each request will have a backup next token id of 10, 20, 30, 40
        mock_request.get_token_id.return_value = int(req_id.split("_")[1]) * 10
        mock_request.num_computed_tokens = 0
        mock_requests[req_id] = mock_request

    # explicitly discard the last request
    discarded_req_mask = torch.tensor(
        [False, False, False, True], dtype=torch.bool, device=device
    )
    sampled_token_ids = [
        [0, 1, -1, -1, -1],  # 1 accepted, 3 rejected, "1" sampled
        [0, 1, 2, 3, 4],  # all accepted, "4" sampled
        [-1, -1, -1, -1, -1],  # sampling skipped, use backup token "30"
        [0, 1, 2, -1, -1],  # explicitly discarded, sampling should be ignored
    ]
    sampled_token_ids_tensor = torch.tensor(
        sampled_token_ids, dtype=torch.int32, device=device
    )
    sampled_token_ids_cpu = [[i for i in seq if i != -1] for seq in sampled_token_ids]
    for i in range(len(sampled_token_ids_cpu)):
        if discarded_req_mask[i]:
            sampled_token_ids_cpu[i] = []

    expected_next_token_ids_cpu = [1, 4, 30, 40]
    expected_next_token_ids_tensor = torch.tensor(
        expected_next_token_ids_cpu, dtype=torch.int32, device=device
    )

    proposer = _create_proposer("eagle", num_speculative_tokens)

    next_token_ids_from_cpu = proposer.prepare_next_token_ids_cpu(
        sampled_token_ids_cpu,
        mock_requests,
        mock_input_batch,
        mock_num_scheduled_tokens,
    )

    assert torch.equal(next_token_ids_from_cpu, expected_next_token_ids_tensor)

    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=16,
        device=device,
    )
    kv_cache_group_id = 0
    common_attn_metadata_by_gid = {kv_cache_group_id: common_attn_metadata}
    # Mock connection of layer names to kv_cache_group_ids
    proposer.attn_layer_names = ["layer0"]
    proposer.layer_names_to_kv_cache_gid = {"layer0": kv_cache_group_id}

    expected_valid_sampled_tokens_count = torch.tensor(
        [2, 5, 0, 0], dtype=torch.int32, device=device
    )

    next_token_ids_from_padded, valid_sampled_tokens_count = (
        proposer.prepare_next_token_ids_padded(
            common_attn_metadata_by_gid,
            sampled_token_ids_tensor,
            mock_requests,
            mock_input_batch,
            discarded_req_mask,
        )
    )

    assert torch.equal(next_token_ids_from_padded, expected_next_token_ids_tensor)
    assert torch.equal(valid_sampled_tokens_count, expected_valid_sampled_tokens_count)


def test_prepare_inputs():
    """
    cu_target_query_lens: [0, a, a + b, a + b + c]
    num_rejected_tokens: [n1, n2, n3]
    num_tokens_per_req: [a - n1, b - n2, c - n3]
    cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
    token_indices: [0, 1, ..., a - n1 - 1,
                    a, a + 1, ..., a + b - n2 - 1,
                    a + b, a + b + 1, ..., a + b + c - n3 - 1]
    """
    device = torch.device(current_platform.device_type)

    # q1 = 4, q2 = 7, q3 = 5
    # n1 = 1, n2 = 3, n3 = 2

    batch_spec = BatchSpec(
        seq_lens=[4, 7, 5],
        query_lens=[4, 7, 5],
    )

    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=16,
        device=device,
    )

    # If there are `k` sampled tokens, then `k-1` tokens are draft tokens
    # from the previous iteration, and the last token is the bonus token sampled
    # from the base model.
    num_draft_tokens = [3, 6, 4]  # one less than query_lens
    # num rejected tokens is [1, 3, 2]
    ACCEPT_TOKEN = 0
    BONUS_TOKEN = 1
    REJECT_TOKEN = -1
    sampled_token_ids = [
        [ACCEPT_TOKEN, ACCEPT_TOKEN, REJECT_TOKEN, BONUS_TOKEN],
        [
            ACCEPT_TOKEN,
            ACCEPT_TOKEN,
            ACCEPT_TOKEN,
            REJECT_TOKEN,
            REJECT_TOKEN,
            REJECT_TOKEN,
            BONUS_TOKEN,
        ],
        [ACCEPT_TOKEN, ACCEPT_TOKEN, REJECT_TOKEN, REJECT_TOKEN, BONUS_TOKEN],
    ]
    sampled_token_ids = [
        [i for i in seq if i != REJECT_TOKEN] for seq in sampled_token_ids
    ]

    # Expected calculations:
    # query_len_per_req = [4, 7, 5]
    # num_tokens_per_req = [3, 4, 3]  (after subtracting rejected tokens)
    # Expected cumulative counts: [0, 3, 7, 10]
    expected_cu_num_tokens = torch.tensor(
        [0, 3, 7, 10], dtype=torch.int32, device=device
    )

    # Expected token indices (mapped from original positions):
    # First request: indices 0, 1, 2      (keeping first 3 from positions 0-3)
    # Second request: indices 4, 5, 6, 7  (keeping first 4 from positions 4-10)
    # Third request: indices 11, 12, 13   (keeping first 3 from positions 11-15)
    expected_token_indices = torch.tensor(
        [
            0,
            1,
            2,  # First request: 3 tokens (4-1)
            4,
            5,
            6,
            7,  # Second request: 4 tokens (7-3)
            11,
            12,
            13,  # Third request: 3 tokens (5-2)
        ],
        dtype=torch.int32,
        device=device,
    )
    proposer = _create_proposer("eagle", 1)

    # Mock connection of layer names to kv_cache_group_ids
    kv_cache_group_id = 0
    common_attn_metadata_by_gid = {kv_cache_group_id: common_attn_metadata}
    proposer.attn_layer_names = ["layer0"]
    proposer.layer_names_to_kv_cache_gid = {"layer0": kv_cache_group_id}

    updated_cm_by_gid, token_indices = proposer.prepare_inputs(
        common_attn_metadata_by_gid, sampled_token_ids, num_draft_tokens
    )

    updated_metadata = updated_cm_by_gid[kv_cache_group_id]
    assert torch.equal(updated_metadata.query_start_loc, expected_cu_num_tokens)
    assert token_indices.shape[0] == expected_cu_num_tokens[-1].item()
    assert torch.equal(token_indices, expected_token_indices)


def test_prepare_inputs_padded():
    """
    Input scenario is 3 requests with num_speculative_tokens == 2 and:
    - Request 1: query_len = 3, rejected = 1
    - Request 2: query_len = 3, rejected = 0
    - Request 3: query_len = 3, rejected = 2

    Expected outputs:
    token_indices_to_sample: [1, 5, 6]
    Reason: After accounting for rejections, these are the valid token positions
            from the original indices to sample from.
    """

    device = torch.device(current_platform.device_type)

    expected_token_indices_to_sample = torch.tensor(
        [1, 5, 6], dtype=torch.int32, device=device
    )

    num_speculative_tokens = 2
    batch_spec = BatchSpec(
        seq_lens=[3, 3, 3],
        query_lens=[3, 3, 3],
    )

    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=16,
        device=device,
    )

    # Needed for cu_num_draft_tokens, which is expected to be [3, 6, 9]
    expected_query_start_loc = torch.tensor(
        [0, 3, 6, 9], dtype=torch.int32, device=device
    )
    spec_decode_metadata = SpecDecodeMetadata.make_dummy(
        draft_token_ids=[[0] * num_speculative_tokens] * 3,
        device=device,
    )

    # num_rejected_tokens = [1, 0, 2]
    # num_draft_tokens = [2, 2, 2]
    # valid_sampled_tokens_count = num_draft_tokens + 1 - num_rejected_tokens
    valid_sampled_tokens_count = torch.tensor(
        [2, 3, 1], dtype=torch.int32, device=device
    )

    proposer = _create_proposer("eagle", num_speculative_tokens)

    # Mock connection of layer names to kv_cache_group_ids
    kv_cache_group_id = 0
    common_attn_metadata_by_gid = {kv_cache_group_id: common_attn_metadata}
    proposer.attn_layer_names = ["layer0"]
    proposer.layer_names_to_kv_cache_gid = {"layer0": kv_cache_group_id}

    output_cm_by_gid, token_indices_to_sample, num_rejected_tokens_gpu = (
        proposer.prepare_inputs_padded(
            common_attn_metadata_by_gid,
            spec_decode_metadata,
            valid_sampled_tokens_count,
        )
    )

    # Verify num_rejected_tokens_gpu is calculated correctly
    expected_num_rejected = torch.tensor([1, 0, 2], dtype=torch.int32, device=device)
    assert torch.equal(num_rejected_tokens_gpu, expected_num_rejected)

    output_metadata = output_cm_by_gid[kv_cache_group_id]
    assert output_metadata.max_query_len == 3
    assert torch.equal(output_metadata.query_start_loc, expected_query_start_loc)
    assert torch.equal(token_indices_to_sample, expected_token_indices_to_sample)


@pytest.mark.parametrize("method", ["eagle", "eagle3"])
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("use_distinct_embed_tokens", [True, False])
@pytest.mark.parametrize("use_distinct_lm_head", [True, False])
@mock.patch("vllm.v1.spec_decode.eagle.get_pp_group")
@mock.patch("vllm.v1.spec_decode.eagle.get_layers_from_vllm_config")
@mock.patch("vllm.v1.spec_decode.eagle.get_model")
def test_load_model(
    mock_get_model,
    mock_get_layers,
    mock_get_pp_group,
    method,
    attn_backend,
    pp_size,
    use_distinct_embed_tokens,
    use_distinct_lm_head,
    monkeypatch,
):
    if attn_backend == "TRITON_ATTN" and not current_platform.is_rocm():
        pytest.skip(
            "TRITON_ATTN does not support "
            "multi-token eagle spec decode on current platform"
        )

    if attn_backend == "ROCM_AITER_FA" and current_platform.is_rocm():
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    # Setup draft model mock
    mock_model = mock.MagicMock()
    mock_model.model = mock.MagicMock()
    mock_model.has_own_embed_tokens = use_distinct_embed_tokens
    if use_distinct_embed_tokens:
        mock_model.model.embed_tokens = mock.MagicMock()
    mock_model.has_own_lm_head = use_distinct_lm_head
    if use_distinct_lm_head:
        mock_model.lm_head = mock.MagicMock()

    mock_get_model.return_value = mock_model

    # Setup mocks for attention layers
    target_attn_layers = {
        "target_attn_1": mock.MagicMock(),
        "target_attn_2": mock.MagicMock(),
    }
    target_indx_layers: dict[str, mock.MagicMock] = {}
    # Draft model has one extra attention layer compared to target model
    all_attn_layers = {**target_attn_layers, "draft_extra_attn": mock.MagicMock()}

    all_indx_layers: dict[str, mock.MagicMock] = {}

    # Make mock_get_layers return different values for each call
    mock_get_layers.side_effect = [
        target_attn_layers,
        target_indx_layers,
        all_attn_layers,
        all_indx_layers,
    ]

    # Setup mock for pp group to return the appropriate value for world size
    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = pp_size
    mock_get_pp_group.return_value = mock_pp_group

    # Set up the target model mock with a custom class so that
    # isinstance() checks match the expected type.
    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock

    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.lm_head = mock.MagicMock()
    target_model.model.embed_tokens = mock.MagicMock()

    from vllm.model_executor.models import SupportsMultiModal

    assert not isinstance(target_model, SupportsMultiModal)

    # Create proposer using the helper function
    proposer = _create_proposer(
        method, num_speculative_tokens=8, attention_backend=attn_backend
    )

    # Call the method under test
    proposer.load_model(target_model)

    # Verify common interactions
    mock_get_model.assert_called_once()

    # Verify that the lm head is set correctly
    if use_distinct_lm_head:
        assert proposer.model.lm_head is not target_model.lm_head
    else:
        assert proposer.model.lm_head is target_model.lm_head

    # Verify that the embed tokens are set correctly
    # If pp_size is > 1, the embed tokens should be distinct
    if pp_size > 1 or use_distinct_embed_tokens:
        assert proposer.model.model.embed_tokens is not target_model.model.embed_tokens
    else:
        assert proposer.model.model.embed_tokens is target_model.model.embed_tokens


@pytest.mark.parametrize("method", ["eagle", "eagle3"])
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 8])
def test_propose(method, attn_backend, num_speculative_tokens, monkeypatch):
    if attn_backend == "TRITON_ATTN" and not current_platform.is_rocm():
        pytest.skip(
            "TRITON_ATTN does not support "
            "multi-token eagle spec decode on current platform"
        )

    if attn_backend == "TREE_ATTN":
        pytest.skip(
            "TREE_ATTN is tested separately in test_propose_tree"
            "because it requires special input mocking."
        )

    if attn_backend == "ROCM_AITER_FA" and current_platform.is_rocm():
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    # Use GPU device
    device = torch.device(current_platform.device_type)

    # Setup test parameters
    batch_size = 2
    seq_len_1 = 5
    seq_len_2 = 3
    total_tokens = seq_len_1 + seq_len_2
    vocab_size = 100
    seq_lens = [seq_len_1, seq_len_2]

    # Create proposer first so we can use its actual hidden_size
    proposer = _create_proposer(
        "eagle", num_speculative_tokens, attention_backend=attn_backend
    )
    # Get the hidden_size from the proposer to ensure consistency
    hidden_size = proposer.hidden_size

    # Helper to create deterministic logits that will produce specific tokens
    def create_deterministic_logits(token_ids):
        logits = torch.full((batch_size, vocab_size), -100.0, device=device)
        for i, token_id in enumerate(token_ids):
            logits[i, token_id] = 100.0
        return logits

    # We mock a model that returns deterministic logits
    # Sequence 1: 42, 43, 44, ...
    # Sequence 2: 60, 61, 62, ...
    base_token_ids = [42, 60]

    # Skip loading the model and replace it with a mock directly
    # Create the mock model with deterministic outputs
    model_mock = mock.MagicMock()

    # Setup for model forward calls
    forward_returns = []
    for i in range(num_speculative_tokens):
        if i == 0:
            # First call uses all tokens
            h_logits = torch.zeros(total_tokens, hidden_size, device=device)
            h_states = torch.zeros(total_tokens, hidden_size, device=device)
        else:
            # Subsequent calls use batch_size tokens
            h_logits = torch.zeros(batch_size, hidden_size, device=device)
            h_states = torch.zeros(batch_size, hidden_size, device=device)
        forward_returns.append((h_logits, h_states))

    # For single token case, we only need the first item;
    # for multi-token, we need the sequence
    if num_speculative_tokens == 1:
        model_mock.return_value = forward_returns[0]
    else:
        model_mock.side_effect = forward_returns

    # Setup for compute_logits calls
    logits_returns = []
    for i in range(num_speculative_tokens):
        # For each call, increment the base token IDs
        current_tokens = [base_id + i for base_id in base_token_ids]
        logits_returns.append(create_deterministic_logits(current_tokens))

    if num_speculative_tokens == 1:
        model_mock.compute_logits.return_value = logits_returns[0]
    else:
        model_mock.compute_logits.side_effect = logits_returns

    # Assign the mock to the proposer
    proposer.model = model_mock

    # Assign draft attn_layer_names since load_model is not invoked
    proposer.attn_layer_names = ["layer.0"]

    # Create input tensors
    batch_spec = BatchSpec(
        seq_lens=seq_lens,
        query_lens=seq_lens,
    )

    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=16,
        device=device,
    )

    target_token_ids = torch.randint(0, vocab_size, (total_tokens,), device=device)
    target_positions = torch.cat(
        [torch.arange(seq_len_1, device=device), torch.arange(seq_len_2, device=device)]
    )
    target_hidden_states = torch.randn(total_tokens, hidden_size, device=device)
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device=device
    )
    sampling_metadata = mock.MagicMock()

    if attn_backend == "FLASH_ATTN":
        attn_metadata_builder_cls, _ = try_get_attention_backend(
            AttentionBackendEnum.FLASH_ATTN
        )
    elif attn_backend == "TRITON_ATTN":
        attn_metadata_builder_cls, _ = try_get_attention_backend(
            AttentionBackendEnum.TRITON_ATTN
        )
    elif attn_backend == "TREE_ATTN":
        attn_metadata_builder_cls, _ = try_get_attention_backend(
            AttentionBackendEnum.TREE_ATTN
        )
    elif attn_backend == "ROCM_AITER_FA":
        attn_metadata_builder_cls, _ = try_get_attention_backend(
            AttentionBackendEnum.ROCM_AITER_FA
        )
    else:
        raise ValueError(f"Unsupported attention backend: {attn_backend}")

    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=proposer.attn_layer_names,
        vllm_config=proposer.vllm_config,
        device=device,
    )

    # Mock runner for attention metadata building
    proposer.runner = mock.MagicMock()
    attn_group = mock.MagicMock()
    attn_group.get_metadata_builder.return_value = attn_metadata_builder
    attn_group.layer_names = proposer.attn_layer_names
    proposer.runner.attn_groups = [[attn_group]]
    proposer._get_attention_metadata_builder = mock.MagicMock(
        return_value=attn_metadata_builder
    )
    kv_cache_group_id = 0
    common_attn_metadata_by_gid = {kv_cache_group_id: common_attn_metadata}
    proposer.layer_names_to_kv_cache_gid = {"layer.0": kv_cache_group_id}

    result = proposer.propose(
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        last_token_indices=None,
        sampling_metadata=sampling_metadata,
        cm_by_gid=common_attn_metadata_by_gid,
    )

    assert result.shape == (batch_size, num_speculative_tokens)

    # Create expected tokens based on our token pattern
    if num_speculative_tokens == 1:
        # Example for num_speculative_tokens=1:
        # [[42], [60]]
        expected_tokens = torch.tensor(
            [[base_token_ids[0]], [base_token_ids[1]]], device=device
        )
    else:
        # Example for num_speculative_tokens=3:
        # [[42, 43, 44], [60, 61, 62]]
        expected_tokens = torch.zeros(
            (batch_size, num_speculative_tokens), dtype=torch.int64, device=device
        )
        for i in range(batch_size):
            for j in range(num_speculative_tokens):
                expected_tokens[i, j] = base_token_ids[i] + j

    # Verify all tokens match our expectations
    assert torch.equal(result, expected_tokens)


@pytest.mark.parametrize(
    "spec_token_tree",
    [
        [(0,)],  # A single token
        [(0,), (0, 0), (0, 0, 0)],  # Chain
        [(0,), (1,), (2,)],  # Parallel
        [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # Tree
    ],
)
def test_propose_tree(spec_token_tree):
    # Get GPU device.
    device = torch.device(current_platform.device_type)

    # Setup test parameters.
    batch_size = 2
    seq_len_1 = 5
    seq_len_2 = 3
    total_tokens = seq_len_1 + seq_len_2
    vocab_size = 100
    seq_lens = [seq_len_1, seq_len_2]
    num_speculative_tokens = len(spec_token_tree)

    # Create proposer first so we can use its actual hidden_size.
    proposer = _create_proposer(
        "eagle",
        num_speculative_tokens,
        speculative_token_tree=spec_token_tree,
    )
    # Get the hidden_size from the proposer to ensure consistency.
    hidden_size = proposer.hidden_size

    # Helper to create deterministic logits that will produce specific tokens
    def create_deterministic_logits(token_ids, k: int):
        logits = torch.full((batch_size, vocab_size), -100.0, device=device)
        for i, token_id in enumerate(token_ids):
            # Assign decreasing values to the k, consecutive, tokens.
            for j in range(k):
                logits[i, token_id + j] = 100.0 - j
        return logits

    # Mock a model that returns deterministic logits.
    base_token_ids = torch.tensor([42, 60], dtype=torch.int64, device=device)

    # Skip loading the model and replace it with a mock that returns
    # deterministic outputs.
    model_mock = mock.MagicMock()

    # Mock the model forward calls.
    forward_returns = [
        (
            torch.zeros(total_tokens, hidden_size, device=device),
            torch.zeros(total_tokens, hidden_size, device=device),
        )
    ]
    for cu_num_drafts in proposer.cu_drafts_per_level:
        h_logits = torch.zeros(batch_size * cu_num_drafts, hidden_size, device=device)
        h_states = torch.zeros(batch_size * cu_num_drafts, hidden_size, device=device)
        forward_returns.append((h_logits, h_states))
    model_mock.side_effect = forward_returns

    # Mock the compute_logits calls.
    cu_num_drafts_tensor = torch.tensor(
        [0] + proposer.cu_drafts_per_level, dtype=torch.int32, device=device
    )
    logits_returns = []
    for level, num_children in enumerate(proposer.child_drafts_per_level):
        token_ids = base_token_ids + cu_num_drafts_tensor[level]
        level_num_drafts = cu_num_drafts_tensor[level + 1] - cu_num_drafts_tensor[level]
        level_logits = []
        for i in range(level_num_drafts // num_children):
            level_logits.append(
                create_deterministic_logits(token_ids + i * num_children, num_children)
            )
        logits_returns.append(torch.stack(level_logits, dim=1))
    model_mock.compute_logits.side_effect = logits_returns

    # Assign the mock to the proposer
    proposer.model = model_mock

    # Assign draft attn_layer_names since load_model is not invoked
    proposer.attn_layer_names = ["layer.0"]

    # Get the tree attention metadata builder.
    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.TREE_ATTN
    )
    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=proposer.attn_layer_names,
        vllm_config=proposer.vllm_config,
        device=device,
    )

    # Mock runner for attention metadata building.
    proposer.runner = mock.MagicMock()
    attn_group = mock.MagicMock()
    attn_group.layer_names = proposer.attn_layer_names
    attn_group.metadata_builders = [attn_metadata_builder]
    attn_group.get_metadata_builder.return_value = attn_metadata_builder
    proposer.runner.attn_groups = [[attn_group]]
    proposer._get_attention_metadata_builder = mock.MagicMock(
        return_value=attn_metadata_builder
    )
    kv_cache_group_id = 0
    proposer.layer_names_to_kv_cache_gid = {"layer.0": kv_cache_group_id}

    # Setup inputs for the proposer.
    target_token_ids = torch.randint(0, vocab_size, (total_tokens,), device=device)
    target_positions = torch.cat(
        [torch.arange(seq_len_1, device=device), torch.arange(seq_len_2, device=device)]
    )
    target_hidden_states = torch.randn(total_tokens, hidden_size, device=device)
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device=device
    )
    batch_spec = BatchSpec(
        seq_lens=seq_lens,
        query_lens=seq_lens,
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=16,
        device=device,
    )
    common_attn_metadata_by_gid = {kv_cache_group_id: common_attn_metadata}
    sampling_metadata = mock.MagicMock()

    # Propose draft tokens.
    result = proposer.propose(
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        last_token_indices=None,
        sampling_metadata=sampling_metadata,
        cm_by_gid=common_attn_metadata_by_gid,
    )
    assert result.shape == (batch_size, num_speculative_tokens)

    # The tokens are expected to be consecutive integers starting
    # from the base token IDs.
    expected_tokens = base_token_ids[:, None] + torch.arange(
        num_speculative_tokens, dtype=torch.int64, device=device
    )

    # Verify that the draft tokens match our expectations.
    assert torch.equal(result, expected_tokens)
