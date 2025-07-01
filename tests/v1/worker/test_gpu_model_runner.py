# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from vllm.attention import Attention
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig, set_current_vllm_config)
from vllm.sampling_params import SamplingParams
from vllm.utils import GiB_bytes
from vllm.v1.core.kv_cache_utils import (estimate_max_model_len,
                                         get_kv_cache_config)
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

BLOCK_SIZE = 16
NUM_BLOCKS = 10
DEVICE = "cuda"


def initialize_kv_cache(runner: GPUModelRunner):
    """
    Only perform necessary steps in GPUModelRunner.initialize_kv_cache()
    """
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=runner.model_config.get_num_kv_heads(
            runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
        use_mla=False,
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
        block_sizes=[
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        ],
    )
    runner.initialize_attn_backend(kv_cache_config)


def get_vllm_config():
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        task="generate",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
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
    model_config = vllm_config.model_config
    num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = model_config.get_head_size()
    vllm_config.compilation_config.static_forward_context[
        "layer.0"] = Attention(num_heads, head_size, 0.1)
    runner = GPUModelRunner(vllm_config, DEVICE)
    initialize_kv_cache(runner)
    return runner


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
                mm_inputs=[],
                mm_hashes=[],
                mm_positions=[],
                sampling_params=SamplingParams(),
                pooling_params=None,
                block_ids=([0], ),
                num_computed_tokens=0,
                lora_request=None,
            ))
        num_scheduled_tokens[req_id] = 3
        total_num_scheduled_tokens += num_scheduled_tokens[req_id]

    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )


def _is_req_scheduled(model_runner, req_id: str) -> bool:
    return req_id in model_runner.input_batch.req_id_to_index


def _is_req_added(model_runner, req_id: str) -> bool:
    return req_id in model_runner.requests


def _is_sampling_metadata_changed(model_runner,
                                  sampling_metadata_before: SamplingMetadata):
    return model_runner.input_batch.sampling_metadata is not (
        sampling_metadata_before)


def _is_req_state_block_table_match(model_runner, req_id: str) -> bool:
    req_index = model_runner.input_batch.req_id_to_index[req_id]
    block_table = model_runner.input_batch.block_table[0]
    req_state = model_runner.requests[req_id]
    if block_table.num_blocks_per_row[req_index] != len(
            req_state.block_ids[0]):
        return False
    num_blocks = block_table.num_blocks_per_row[req_index]
    return (block_table.block_table_np[req_index, :num_blocks] ==
            req_state.block_ids[0]).all()


def test_update_states_new_request(model_runner):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_finished(model_runner):
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
        num_common_prefix_blocks=0,
        finished_req_ids={req_id},
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert not _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)


def test_update_states_request_resumed(model_runner):
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
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)

    # resume req
    cached_req_data = CachedRequestData(
        req_ids=[req_id],
        resumed_from_preemption=[False],
        new_token_ids=[[]],
        new_block_ids=([[0]], ),
        num_computed_tokens=[0],
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_get_nans_in_logits(model_runner):
    req_ids = ("req_0", "req_1")

    scheduler_output = _schedule_new_request(*req_ids)
    model_runner._update_states(scheduler_output)

    logits = torch.tensor([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
    ], device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 0, "req_1": 0}

    logits = torch.tensor([
        [1.0, float('nan'), 3.0],
        [4.0, float('nan'), float('nan')],
    ],
                          device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 1, "req_1": 2}

    logits = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, float('nan'), float('nan')],
    ],
                          device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 0, "req_1": 2}

    result = model_runner._get_nans_in_logits(logits=None)
    assert result == {"req_0": 0, "req_1": 0}

    logits = torch.tensor([
        [1.0, float('nan'), 3.0],
    ], device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {'req_0': 1, 'req_1': 0}

    logits = torch.tensor([
        [float('nan'), float('nan'), 2.0],
        [1.0, 2.0, 3.0],
        [float('nan'), 2.0, 3.0],
    ],
                          device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {'req_0': 2, 'req_1': 0}


def test_update_states_no_changes(model_runner):
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
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert not _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_unscheduled(model_runner):
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
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
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
    n_heads = model_runner.model_config.get_num_kv_heads(
        model_runner.parallel_config)
    expected_kv_cache_shape = [
        2, NUM_BLOCKS, BLOCK_SIZE, n_heads,
        model_runner.model_config.get_head_size()
    ]
    # TODO mla test
    default_stride = list(range(5))
    # Permutation that gets you back to expected kv shape
    rnd_stride = tuple(random.sample(default_stride, len(default_stride)))

    def rnd_stride_order():
        return rnd_stride

    # Patch the attention backend class and re-trigger the KV cache creation.
    for attn_backend in model_runner.attn_backends:
        monkeypatch.setattr(attn_backend, "get_kv_cache_stride_order",
                            rnd_stride_order)

    model_runner.attn_backends = []
    model_runner.attn_metadata_builders = []
    model_runner.initialize_kv_cache(model_runner.kv_cache_config)

    # Shape is unchanged, but layout may differ
    kv_cache_shape = model_runner.kv_caches[0].shape
    assert list(kv_cache_shape) == expected_kv_cache_shape
    if default_stride == rnd_stride:
        assert all(kv.is_contiguous() for kv in model_runner.kv_caches)
    else:
        assert all(not kv.is_contiguous() for kv in model_runner.kv_caches)


def test_load_model_weights_inplace(dist_init, model_runner, model_runner_2):
    # In this test, model_runner loads model + weights in one go, while
    # model_runner_2 loads dummy weights first then load real weights inplace
    model_runner.load_model()
    original_load_format = model_runner_2.load_config.load_format
    model_runner_2.load_config.load_format = "dummy"
    model_runner_2.load_model()  # Initial model loading with dummy weights
    assert str(model_runner.get_model().state_dict()) != str(
        model_runner_2.get_model().state_dict())
    model_runner_2.load_config.load_format = original_load_format
    model_runner_2.load_model()  # Load real weights inplace
    assert str(model_runner.get_model().state_dict()) == str(
        model_runner_2.get_model().state_dict())


def test_init_kv_cache_with_kv_sharing_invalid_target_layer_order():
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} must come before the current layer"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
                kv_sharing_target_layer_name=layer_1,
            ),
            layer_1:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
            )
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_layer_not_exist():
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    invalid_layer = "model.layers.0.cross_attn.attn"
    error_msg = f"{invalid_layer} is not a valid Attention layer in the model"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            layer_0:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                # invalid layer: cross_attn.atn doesn't exist!
                kv_sharing_target_layer_name=invalid_layer,
            )
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_same_as_current():
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} cannot be the same as the current layer"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                kv_sharing_target_layer_name=layer_1,
            )
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_without_kv_sharing():
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        fwd_context = {
            layer_0:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
            )
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
    kv_cache_config = get_kv_cache_config(vllm_config, kv_cache_spec,
                                          available_memory)
    assert kv_cache_config.num_blocks == num_expected_blocks
    assert len(kv_cache_config.kv_cache_tensors) == 2
    assert kv_cache_config.kv_cache_tensors[0].size == available_memory // 2
    assert kv_cache_config.kv_cache_tensors[1].size == available_memory // 2

    max_context_len =\
        estimate_max_model_len(vllm_config, kv_cache_spec, 5 * GiB_bytes)
    # max context len with KV sharing should be 2x as large as without
    assert max_context_len == 1310720

    # important: override tensor size to prevent large mem alloc during test
    # this will only allocate 2 block worth of memory (2 * 32kb)
    kv_cache_config.num_blocks = 1
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        kv_cache_tensor.size = (
            kv_cache_spec[kv_cache_tensor.shared_by[0]].page_size_bytes)

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


def test_init_kv_cache_with_kv_sharing_valid():
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        fwd_context = {
            layer_0:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                kv_sharing_target_layer_name="model.layers.0.self_attn.attn",
            )
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
    kv_cache_config = get_kv_cache_config(vllm_config, kv_cache_spec,
                                          available_memory)
    assert kv_cache_config.num_blocks == num_expected_blocks
    assert len(kv_cache_config.kv_cache_tensors) == 1
    # Each layer now has twice the available memory for KV cache
    # compared to no KV sharing
    assert kv_cache_config.kv_cache_tensors[0].size == available_memory

    max_context_len =\
        estimate_max_model_len(vllm_config, kv_cache_spec, 5 * GiB_bytes)
    # max context len with KV sharing should be 2x as large as without
    assert max_context_len == 2 * 1310720

    # important: override tensor size to prevent large mem alloc during test
    # this will only allocate 1 block worth of memory (32kb)
    kv_cache_config.num_blocks = 1
    kv_cache_config.kv_cache_tensors[0].size =\
        kv_cache_spec[layer_0].page_size_bytes

    runner.initialize_kv_cache(kv_cache_config)

    layer_0_kv = vllm_ctx[layer_0].kv_cache[0]
    layer_1_kv = vllm_ctx[layer_1].kv_cache[0]
    # check layer 1 kv cache shares memory with layer 0
    assert id(layer_1_kv) == id(layer_0_kv)

    # check layer 1 added to kv cache group's layer names
    assert len(kv_cache_config.kv_cache_groups) == 1
    assert len(kv_cache_config.kv_cache_groups[0].layer_names) == 2
    assert kv_cache_config.kv_cache_groups[0].layer_names[0] == layer_0
    assert kv_cache_config.kv_cache_groups[0].layer_names[1] == layer_1
