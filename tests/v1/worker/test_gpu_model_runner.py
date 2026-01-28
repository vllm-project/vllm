# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.kv_cache_utils import estimate_max_model_len, get_kv_cache_configs
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import AttentionGroup

BLOCK_SIZE = 16
NUM_BLOCKS = 10
DEVICE = current_platform.device_type


def initialize_kv_cache(runner: GPUModelRunner):
    """
    Only perform necessary steps in GPUModelRunner.initialize_kv_cache()
    """
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=runner.model_config.get_num_kv_heads(runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size, shared_by=["layer.0"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)
        ],
    )
    runner.kv_cache_config = kv_cache_config
    runner.input_batch = InputBatch(
        max_num_reqs=runner.max_num_reqs,
        max_model_len=runner.max_model_len,
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        pin_memory=runner.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size],
        kernel_block_sizes=[
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        ],
    )
    runner.initialize_attn_backend(kv_cache_config)


def get_vllm_config():
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config


@pytest.fixture
def model_runner():
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        model_config = vllm_config.model_config
        num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        head_size = model_config.get_head_size()
        vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(
            num_heads, head_size, 0.1
        )
        runner = GPUModelRunner(vllm_config, DEVICE)
        initialize_kv_cache(runner)
        yield runner


model_runner_2 = model_runner


def _schedule_new_request(*req_ids: str) -> SchedulerOutput:
    new_reqs = []
    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    for req_id in req_ids:
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=[1, 2, 3],
                mm_features=[],
                sampling_params=SamplingParams(),
                pooling_params=None,
                block_ids=([0],),
                num_computed_tokens=0,
                lora_request=None,
            )
        )
        num_scheduled_tokens[req_id] = 3
        total_num_scheduled_tokens += num_scheduled_tokens[req_id]

    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _is_req_scheduled(model_runner, req_id: str) -> bool:
    return req_id in model_runner.input_batch.req_id_to_index


def _is_req_added(model_runner, req_id: str) -> bool:
    return req_id in model_runner.requests


def _is_sampling_metadata_changed(
    model_runner, sampling_metadata_before: SamplingMetadata
):
    return model_runner.input_batch.sampling_metadata is not (sampling_metadata_before)


def _is_req_state_block_table_match(model_runner, req_id: str) -> bool:
    req_index = model_runner.input_batch.req_id_to_index[req_id]
    block_table = model_runner.input_batch.block_table[0]
    req_state = model_runner.requests[req_id]
    if block_table.num_blocks_per_row[req_index] != len(req_state.block_ids[0]):
        return False
    num_blocks = block_table.num_blocks_per_row[req_index]
    return (
        block_table.block_table.np[req_index, :num_blocks] == req_state.block_ids[0]
    ).all()


def _make_mock_backend_for_kernel_block_size(
    supported_sizes: list[int | MultipleOf],
):
    class _MockBackend:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return supported_sizes

    return _MockBackend()


def _make_kv_cache_spec() -> FullAttentionSpec:
    return FullAttentionSpec(block_size=1, num_kv_heads=1, head_size=1, dtype="float16")


def test_select_common_block_size_prefers_manager_block_size():
    backend_a = _make_mock_backend_for_kernel_block_size([MultipleOf(32)])
    backend_b = _make_mock_backend_for_kernel_block_size([64, MultipleOf(16)])
    attn_groups = [
        AttentionGroup(backend_a, [], [], _make_kv_cache_spec(), 0),
        AttentionGroup(backend_b, [], [], _make_kv_cache_spec(), 0),
    ]

    selected_size = GPUModelRunner.select_common_block_size(128, attn_groups)
    assert selected_size == 128


def test_select_common_block_size_uses_largest_shared_int():
    backend_a = _make_mock_backend_for_kernel_block_size([128, 64])
    backend_b = _make_mock_backend_for_kernel_block_size([64, 32])
    attn_groups = [
        AttentionGroup(backend_a, [], [], _make_kv_cache_spec(), 0),
        AttentionGroup(backend_b, [], [], _make_kv_cache_spec(), 0),
    ]

    selected_size = GPUModelRunner.select_common_block_size(256, attn_groups)
    assert selected_size == 64


def test_select_common_block_size_no_valid_option():
    backend_a = _make_mock_backend_for_kernel_block_size([64])
    backend_b = _make_mock_backend_for_kernel_block_size([MultipleOf(16)])
    attn_groups = [
        AttentionGroup(backend_a, [], [], _make_kv_cache_spec(), 0),
        AttentionGroup(backend_b, [], [], _make_kv_cache_spec(), 0),
    ]

    with pytest.raises(ValueError):
        GPUModelRunner.select_common_block_size(48, attn_groups)


def test_update_states_new_request(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_finished(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # finish req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids={req_id},
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert not _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)


def test_update_states_request_resumed(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # unschedule req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)

    # resume req
    cached_req_data = CachedRequestData(
        req_ids=[req_id],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[([0],)],
        num_computed_tokens=[0],
        num_output_tokens=[0],
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_get_nans_in_logits(model_runner, dist_init):
    req_ids = ("req_0", "req_1")

    scheduler_output = _schedule_new_request(*req_ids)
    model_runner._update_states(scheduler_output)

    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ],
        device=DEVICE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 0, "req_1": 0}

    logits = torch.tensor(
        [
            [1.0, float("nan"), 3.0],
            [4.0, float("nan"), float("nan")],
        ],
        device=DEVICE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 1, "req_1": 2}

    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, float("nan"), float("nan")],
        ],
        device=DEVICE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 0, "req_1": 2}

    result = model_runner._get_nans_in_logits(logits=None)
    assert result == {"req_0": 0, "req_1": 0}

    logits = torch.tensor(
        [
            [1.0, float("nan"), 3.0],
        ],
        device=DEVICE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 1, "req_1": 0}

    logits = torch.tensor(
        [
            [float("nan"), float("nan"), 2.0],
            [1.0, 2.0, 3.0],
            [float("nan"), 2.0, 3.0],
        ],
        device=DEVICE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 2, "req_1": 0}


def test_update_states_no_changes(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # schedule req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert not _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_unscheduled(model_runner, dist_init):
    req_ids = ("req_0", "req_1")

    # new reqs
    scheduler_output = _schedule_new_request(*req_ids)

    model_runner._update_states(scheduler_output)

    assert _is_req_added(model_runner, req_ids[0])
    assert _is_req_scheduled(model_runner, req_ids[0])

    assert _is_req_added(model_runner, req_ids[1])
    assert _is_req_scheduled(model_runner, req_ids[1])

    # unschedule req_1
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_ids[0]: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)

    assert _is_req_added(model_runner, req_ids[0])
    assert _is_req_scheduled(model_runner, req_ids[0])

    assert _is_req_added(model_runner, req_ids[1])
    assert not _is_req_scheduled(model_runner, req_ids[1])


def test_kv_cache_stride_order(monkeypatch, model_runner):
    # This test checks if GPUModelRunner initializes correctly when an attention
    # backend enforces a non-default KV cache stride order.
    n_heads = model_runner.model_config.get_num_kv_heads(model_runner.parallel_config)
    head_size = model_runner.model_config.get_head_size()

    # Get the expected shape from the backend's get_kv_cache_shape method
    # to ensure compatibility with different backends (triton vs flexattention)
    attn_backend = None
    for attn_group in model_runner._attn_group_iterator():
        attn_backend = attn_group.backend
        break

    assert attn_backend is not None, "No attention backend found"
    expected_kv_cache_shape = list(
        attn_backend.get_kv_cache_shape(NUM_BLOCKS, BLOCK_SIZE, n_heads, head_size)
    )

    # TODO mla test
    default_stride = tuple(range(5))
    # Permutation that gets you back to expected kv shape
    for test_stride in ((1, 4, 0, 2, 3), (0, 1, 2, 3, 4)):

        def rnd_stride_order(
            include_num_layers_dimension: bool = False, test_stride=test_stride
        ):
            assert not include_num_layers_dimension
            return test_stride

        # Patch the attention backend class and re-trigger the KV cache creation
        for attn_group in model_runner._attn_group_iterator():
            attn_backend = attn_group.backend
            monkeypatch.setattr(
                attn_backend, "get_kv_cache_stride_order", rnd_stride_order
            )

        model_runner.attn_groups = []
        model_runner.kv_caches = []
        model_runner.initialize_kv_cache(model_runner.kv_cache_config)

        # Shape is unchanged, but layout may differ
        kv_cache_shape = model_runner.kv_caches[0].shape
        assert list(kv_cache_shape) == expected_kv_cache_shape
        if default_stride == test_stride:
            assert all(kv.is_contiguous() for kv in model_runner.kv_caches)
        else:
            assert all(not kv.is_contiguous() for kv in model_runner.kv_caches)


def test_update_config(model_runner):
    # Simple update
    model_runner.update_config({"load_config": {"load_format": "dummy"}})
    assert model_runner.load_config.load_format == "dummy"
    # Raise error on non-existing config
    with pytest.raises(AssertionError):
        model_runner.update_config({"do_not_exist_config": "dummy"})


def test_load_model_weights_inplace(dist_init, model_runner, model_runner_2):
    # In this test, model_runner loads model + weights in one go, while
    # model_runner_2 loads dummy weights first then load real weights inplace
    model_runner.load_model()
    original_load_format = model_runner_2.load_config.load_format
    model_runner_2.update_config({"load_config": {"load_format": "dummy"}})
    model_runner_2.load_model()  # Initial model loading with dummy weights
    assert str(model_runner.get_model().state_dict()) != str(
        model_runner_2.get_model().state_dict()
    )
    model_runner_2.update_config({"load_config": {"load_format": original_load_format}})
    model_runner_2.reload_weights()  # Load real weights inplace
    assert str(model_runner.get_model().state_dict()) == str(
        model_runner_2.get_model().state_dict()
    )


def test_reload_weights_before_load_model(model_runner):
    with pytest.raises(AssertionError):
        model_runner.reload_weights()


def test_init_kv_cache_with_kv_sharing_invalid_target_layer_order(default_vllm_config):
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} must come before the current layer"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
                kv_sharing_target_layer_name=layer_1,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_layer_not_exist(default_vllm_config):
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    invalid_layer = "model.layers.0.cross_attn.attn"
    error_msg = f"{invalid_layer} is not a valid Attention layer in the model"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                # invalid layer: cross_attn.atn doesn't exist!
                kv_sharing_target_layer_name=invalid_layer,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_same_as_current(default_vllm_config):
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} cannot be the same as the current layer"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                kv_sharing_target_layer_name=layer_1,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_without_kv_sharing(default_vllm_config):
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        fwd_context = {
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None
    # Set high context length to test max context length estimation
    vllm_config.model_config.max_model_len = 3_000_000
    vllm_ctx = vllm_config.compilation_config.static_forward_context
    runner = GPUModelRunner(vllm_config, DEVICE)
    kv_cache_spec = runner.get_kv_cache_spec()
    assert len(kv_cache_spec) == 2
    assert len(runner.shared_kv_cache_layers) == 0

    available_memory = 20 * GiB_bytes
    # page size for layer 0's kv_cache_spec is 32KB
    num_expected_blocks = 327680  # 20GB / 32KB / 2 (num layers)
    kv_cache_config = get_kv_cache_configs(
        vllm_config, [kv_cache_spec], [available_memory]
    )[0]
    assert kv_cache_config.num_blocks == num_expected_blocks
    assert len(kv_cache_config.kv_cache_tensors) == 2
    assert kv_cache_config.kv_cache_tensors[0].size == available_memory // 2
    assert kv_cache_config.kv_cache_tensors[1].size == available_memory // 2

    max_context_len = estimate_max_model_len(vllm_config, kv_cache_spec, 5 * GiB_bytes)
    # max context len with KV sharing should be 2x as large as without
    assert max_context_len == 1310720

    # important: override tensor size to prevent large mem alloc during test
    # this will only allocate 2 block worth of memory (2 * 32kb)
    kv_cache_config.num_blocks = 1
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        kv_cache_tensor.size = kv_cache_spec[
            kv_cache_tensor.shared_by[0]
        ].page_size_bytes

    runner.initialize_kv_cache(kv_cache_config)

    layer_0_kv = vllm_ctx[layer_0].kv_cache[0]
    layer_1_kv = vllm_ctx[layer_1].kv_cache[0]
    # check layer 1 kv cache does NOT share memory with layer 0
    assert id(layer_1_kv) != id(layer_0_kv)

    # check layer 1 added to kv cache group's layer names
    assert len(kv_cache_config.kv_cache_groups) == 1
    assert len(kv_cache_config.kv_cache_groups[0].layer_names) == 2
    assert kv_cache_config.kv_cache_groups[0].layer_names[0] == layer_0
    assert kv_cache_config.kv_cache_groups[0].layer_names[1] == layer_1


def test_init_kv_cache_with_kv_sharing_valid(default_vllm_config):
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        fwd_context = {
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                kv_sharing_target_layer_name="model.layers.0.self_attn.attn",
            ),
        }
        # suppress var not used error
        assert fwd_context is not None
    # Set high context length to test max context length estimation
    vllm_config.model_config.max_model_len = 3_000_000
    vllm_ctx = vllm_config.compilation_config.static_forward_context
    runner = GPUModelRunner(vllm_config, DEVICE)
    kv_cache_spec = runner.get_kv_cache_spec()
    assert len(kv_cache_spec) == 1
    assert layer_0 in kv_cache_spec
    assert runner.shared_kv_cache_layers[layer_1] == layer_0

    available_memory = 20 * GiB_bytes
    # page size for layer 0's kv_cache_spec is 32KB
    # with KV sharing, we can allocate (available_mem//page_size//1) blocks
    # which is twice as many as without KV sharing
    num_expected_blocks = 655360  # 20GB / 32KB
    kv_cache_config = get_kv_cache_configs(
        vllm_config, [kv_cache_spec], [available_memory]
    )[0]
    assert kv_cache_config.num_blocks == num_expected_blocks
    assert len(kv_cache_config.kv_cache_tensors) == 1
    # Each layer now has twice the available memory for KV cache
    # compared to no KV sharing
    assert kv_cache_config.kv_cache_tensors[0].size == available_memory

    max_context_len = estimate_max_model_len(vllm_config, kv_cache_spec, 5 * GiB_bytes)
    # max context len with KV sharing should be 2x as large as without
    assert max_context_len == 2 * 1310720

    # important: override tensor size to prevent large mem alloc during test
    # this will only allocate 1 block worth of memory (32kb)
    kv_cache_config.num_blocks = 1
    kv_cache_config.kv_cache_tensors[0].size = kv_cache_spec[layer_0].page_size_bytes

    runner.initialize_kv_cache(kv_cache_config)
    kv_cache_config_after_init = runner.kv_cache_config

    layer_0_kv = vllm_ctx[layer_0].kv_cache[0]
    layer_1_kv = vllm_ctx[layer_1].kv_cache[0]
    # check layer 1 kv cache shares memory with layer 0
    assert id(layer_1_kv) == id(layer_0_kv)

    # check layer 1 added to kv cache group's layer names
    assert len(kv_cache_config_after_init.kv_cache_groups) == 1
    assert len(kv_cache_config_after_init.kv_cache_groups[0].layer_names) == 2
    assert kv_cache_config_after_init.kv_cache_groups[0].layer_names[0] == layer_0
    assert kv_cache_config_after_init.kv_cache_groups[0].layer_names[1] == layer_1


@pytest.mark.skipif(
    current_platform.is_rocm(),
    reason="Attention backend FLASHINFER is not supported on ROCm.",
)
def test_hybrid_attention_mamba_tensor_shapes():
    """
    The GPU model runner creates different views into the
    KVCacheTensors for the attention and mamba layers
    (via _reshape_kv_cache_tensors function). This test verifies
    that the views are compatible: writing a mamba block
    will not corrupt an attention block and vice versa
    """

    set_random_seed(42)

    update_environment_variables(
        {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
        }
    )
    from tests.utils import ensure_current_vllm_config
    with ensure_current_vllm_config():
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=1)
    torch.set_default_dtype(torch.float16)

    model_config = ModelConfig(
        model="ibm-granite/granite-4.0-tiny-preview",
        dtype="float16",
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    attention_config = AttentionConfig(backend=AttentionBackendEnum.FLASHINFER)
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        attention_config=attention_config,
    )

    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    layer_2 = "model.layers.2.mixer"
    layer_3 = "model.layers.3.mixer"
    layer_4 = "model.layers.4.mixer"
    layer_5 = "model.layers.5.mixer"

    with set_current_vllm_config(vllm_config):
        hf_config = vllm_config.model_config.hf_config
        fwd_context = {}
        for key in [layer_0, layer_1]:
            fwd_context[key] = Attention(
                num_heads=model_config.get_num_attention_heads(parallel_config),
                num_kv_heads=model_config.get_num_kv_heads(parallel_config),
                head_size=model_config.get_head_size(),
                scale=1.0,
                prefix=key,
            )
        for key in [layer_2, layer_3, layer_4, layer_5]:
            fwd_context[key] = MambaMixer2(
                hidden_size=hf_config.hidden_size,
                ssm_state_size=hf_config.mamba_d_state,
                conv_kernel_size=hf_config.mamba_d_conv,
                intermediate_size=hf_config.mamba_expand * hf_config.hidden_size,
                use_conv_bias=hf_config.mamba_conv_bias,
                use_bias=hf_config.mamba_proj_bias,
                n_groups=hf_config.mamba_n_groups,
                num_heads=hf_config.mamba_n_heads,
                head_dim=hf_config.mamba_d_head,
                rms_norm_eps=hf_config.rms_norm_eps,
                activation=hf_config.hidden_act,
                cache_config=cache_config,
                model_config=model_config,
                prefix=key,
            )
        # suppress var not used error
        assert fwd_context is not None
        vllm_ctx = vllm_config.compilation_config.static_forward_context

        runner = GPUModelRunner(vllm_config, DEVICE)
        kv_cache_spec = runner.get_kv_cache_spec()

        available_memory = 5 * GiB_bytes
        kv_cache_config = get_kv_cache_configs(
            vllm_config, [kv_cache_spec], [available_memory]
        )[0]
        runner.initialize_kv_cache(kv_cache_config)

    # random partition of blocks
    # blocks0 will be assigned to attention layers
    # blocks1 will be assigned to mamba layers
    num_blocks = kv_cache_config.num_blocks
    ind = np.arange(num_blocks)
    np.random.shuffle(ind)
    blocks0, blocks1 = ind[: (num_blocks // 2)], ind[(num_blocks // 2) :]

    attn_shape = vllm_ctx[layer_0].kv_cache[0].shape
    conv_shape = vllm_ctx[layer_2].kv_cache[0][0].shape
    ssm_shape = vllm_ctx[layer_2].kv_cache[0][1].shape

    # assert we are using FlashInfer
    assert attn_shape[0] % num_blocks == 0
    block_split_ratio = attn_shape[0] // num_blocks

    # use small blocks for testing to avoid memory issues
    test_block_size = min(2, len(blocks0), len(blocks1))

    # use non-overlapping blocks to avoid data contamination
    # Split kernel blocks: first half for attention, second half for mamba
    mid_point = num_blocks // 2

    # attention uses kernel blocks from first half (mapped to logical blocks)
    kv_blocks_for_attention = np.array([0, 1])[:test_block_size]

    # mamba uses kernel blocks from second half
    kv_blocks_for_mamba = np.array([mid_point, mid_point + 1])[:test_block_size]

    # create small constant tensors for testing with corrected shapes
    # attention: [block_size, ...] starting from dimension 2
    attn_constant_shape = attn_shape[2:]
    conv_constant_shape = conv_shape[1:]
    ssm_constant_shape = ssm_shape[1:]

    attn_blocks_constant = torch.full(
        (test_block_size, *attn_constant_shape), device=DEVICE, fill_value=3.33
    )
    conv_blocks_constant = torch.full(
        (test_block_size, *conv_constant_shape), device=DEVICE, fill_value=6.66
    )
    ssm_blocks_constant = torch.full(
        (test_block_size, *ssm_constant_shape), device=DEVICE, fill_value=9.99
    )

    # Fill attention blocks with constants using kv block indices
    kernel_blocks_for_attention = kv_blocks_for_attention * block_split_ratio

    for layer in [layer_0, layer_1]:
        # attention: kv_cache[0][kernel_block_idx, kv_idx, ...]
        for i, kernel_block in enumerate(kernel_blocks_for_attention):
            vllm_ctx[layer].kv_cache[0][kernel_block, :] = attn_blocks_constant[i]

    # fill mamba blocks with constants using kernel block indices
    for layer in [layer_2, layer_3, layer_4, layer_5]:
        # mamba: kv_cache[0][component][kernel_block_idx, ...]
        for i, kv_block in enumerate(kv_blocks_for_mamba):
            vllm_ctx[layer].kv_cache[0][0][kv_block, :] = conv_blocks_constant[i]
            vllm_ctx[layer].kv_cache[0][1][kv_block, :] = ssm_blocks_constant[i]

    # verify attention and mamba contents are correct
    for layer in [layer_0, layer_1]:
        for i, kernel_block in enumerate(kernel_blocks_for_attention):
            actual_kv = vllm_ctx[layer].kv_cache[0][kernel_block, :]
            expected = attn_blocks_constant[i]

            # Check K and V separately
            assert torch.equal(actual_kv[0], expected)
            assert torch.equal(actual_kv[1], expected)

    for layer in [layer_2, layer_3, layer_4, layer_5]:
        for i, kv_block in enumerate(kv_blocks_for_mamba):
            actual_conv = vllm_ctx[layer].kv_cache[0][0][kv_block, :]
            actual_ssm = vllm_ctx[layer].kv_cache[0][1][kv_block, :]
            expected_conv = conv_blocks_constant[i]
            expected_ssm = ssm_blocks_constant[i]

            assert torch.equal(actual_conv, expected_conv)
            assert torch.equal(actual_ssm, expected_ssm)

    for layer in [layer_2, layer_3, layer_4, layer_5]:
        for i, kv_block in enumerate(kv_blocks_for_mamba):
            actual_conv = vllm_ctx[layer].kv_cache[0][0][kv_block, :]
            actual_ssm = vllm_ctx[layer].kv_cache[0][1][kv_block, :]
            expected_conv = conv_blocks_constant[i]
            expected_ssm = ssm_blocks_constant[i]
            assert torch.equal(actual_conv, expected_conv)
            assert torch.equal(actual_ssm, expected_ssm)


def test_hybrid_block_table_initialization():
    """Test hybrid block table with different kernel and kvcache_manager block
    sizes."""
    from vllm.v1.worker.block_table import BlockTable

    # Test configuration: kvcache_manager block size = 32,
    # kernel block size = 16
    block_size = 32
    kernel_block_sizes = [16]
    max_num_reqs = 10
    max_num_blocks_per_req = 20
    max_num_batched_tokens = 512
    cp_kv_cache_interleave_size = 8

    block_table = BlockTable(
        block_size=block_size,
        max_num_reqs=max_num_reqs,
        max_num_blocks_per_req=max_num_blocks_per_req,
        max_num_batched_tokens=max_num_batched_tokens,
        pin_memory=False,
        device=torch.device(DEVICE),
        kernel_block_size=kernel_block_sizes[0],
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )

    # Verify hybrid block configuration
    assert block_table.use_hybrid_blocks is True
    assert block_table.block_size == kernel_block_sizes[0]
    assert block_table.blocks_per_kv_block == (
        block_size // kernel_block_sizes[0]
    )  # Changed to use first element

    # Test block table conversion logic
    # One kvcache_manager block should map to multiple kernel blocks
    kvcache_manager_blocks = [0, 1, 2]

    # Verify that kvcache_manager blocks can be converted to kernel blocks
    # and that block table operations work correctly.
    req_index = 0
    block_table.append_row(kvcache_manager_blocks, req_index)
    # Get expected kernel blocks from the implementation for verification.
    expected_kernel_blocks = block_table.map_to_kernel_blocks(
        np.array(kvcache_manager_blocks),
        block_table.blocks_per_kv_block,
        block_table._kernel_block_arange,
    )
    # Verify block table state
    assert block_table.num_blocks_per_row[req_index] == len(expected_kernel_blocks)
    assert np.array_equal(
        block_table.block_table.np[req_index, : len(expected_kernel_blocks)],
        expected_kernel_blocks,
    )


def test_input_batch_with_kernel_block_sizes():
    """Test InputBatch initialization with kernel_block_sizes parameter."""
    max_num_reqs = 10
    max_model_len = 512
    max_num_batched_tokens = 512
    device = torch.device(DEVICE)
    pin_memory = False
    vocab_size = 50272

    # Test with different kernel block sizes
    block_sizes = [32, 64]
    kernel_block_sizes = [16, 32]

    input_batch = InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        device=device,
        pin_memory=pin_memory,
        vocab_size=vocab_size,
        block_sizes=block_sizes,
        kernel_block_sizes=kernel_block_sizes,
    )

    # Verify that block tables were created with kernel block sizes
    assert len(input_batch.block_table.block_tables) == len(block_sizes)

    for i, (kv_size, kernel_size) in enumerate(zip(block_sizes, kernel_block_sizes)):
        block_table = input_batch.block_table.block_tables[i]
        if kv_size != kernel_size:
            assert block_table.use_hybrid_blocks is True
            assert block_table.block_size == kernel_size
        else:
            assert block_table.use_hybrid_blocks is False
            assert block_table.block_size == kernel_size


def test_hybrid_cache_integration(default_vllm_config, dist_init):
    """Test hybrid cache architecture integration with GPUModelRunner."""
    # Create a new model runner with hybrid cache configuration
    vllm_config = get_vllm_config()

    # Configure hybrid cache with different kvcache_manager block size
    vllm_config.cache_config.block_size = 32

    model_config = vllm_config.model_config
    num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = model_config.get_head_size()
    vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(
        num_heads, head_size, 0.1
    )

    runner = GPUModelRunner(vllm_config, DEVICE)

    # Initialize KV cache with configuration
    attn_spec = FullAttentionSpec(
        block_size=16,  # Use kernel block size directly
        num_kv_heads=runner.model_config.get_num_kv_heads(runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size, shared_by=["layer.0"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)
        ],
    )
    runner.kv_cache_config = kv_cache_config

    # Initialize input batch with kernel block sizes
    runner.input_batch = InputBatch(
        max_num_reqs=runner.max_num_reqs,
        max_model_len=runner.max_model_len,
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        pin_memory=runner.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size],
        kernel_block_sizes=[16],
    )  # Use kernel block size

    runner.initialize_attn_backend(kv_cache_config)

    # Verify hybrid block table configuration
    block_table = runner.input_batch.block_table.block_tables[0]
    assert block_table.block_size == (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
    )

    # Test request processing with hybrid blocks
    req_id = "hybrid_req_0"
    scheduler_output = _schedule_new_request(req_id)

    # Update states should work with hybrid blocks
    runner._update_states(scheduler_output)
    assert _is_req_scheduled(runner, req_id)
    assert _is_req_state_block_table_match(runner, req_id)


def test_is_uniform_decode() -> None:
    # Normal
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=16,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=2,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=16,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=15,
    )
    # Spec decoding
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=5,
        uniform_decode_query_len=5,
        num_tokens=30,
        num_reqs=6,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=5,
        uniform_decode_query_len=4,
        num_tokens=30,
        num_reqs=6,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=5,
        uniform_decode_query_len=5,
        num_tokens=30,
        num_reqs=7,
    )
    # Force uniform decode
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=16,
        force_uniform_decode=True,
    )
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=2,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=16,
        force_uniform_decode=True,
    )
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=15,
        force_uniform_decode=True,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=16,
        force_uniform_decode=False,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=2,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=16,
        force_uniform_decode=False,
    )
    assert not GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        num_tokens=16,
        num_reqs=15,
        force_uniform_decode=False,
    )
